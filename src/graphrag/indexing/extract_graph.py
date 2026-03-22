"""Extract entities, relationships, claims, and covariates from TextUnits; persist to Neo4j."""

import hashlib
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

from graphrag.config import ENTITY_DESCRIPTION_MAX_CHARS, OPENAI_API_KEY, LLM_MODEL
from graphrag.indexing.entity_resolution import (
    backfill_entity_keys_cypher,
    make_entity_key,
    merge_canonical_name_and_aliases,
    normalize_token,
    resolve_surface_to_entity_key,
)
from graphrag.store.neo4j_graph import get_neo4j_graph


class Entity(BaseModel):
    name: str
    type: str = Field(description="e.g. Person, Place, Organization, Concept")
    description: str = ""


class Relationship(BaseModel):
    source: str = Field(description="Entity name")
    target: str = Field(description="Entity name")
    type: str = Field(description="Relationship type")
    description: str = ""


class Claim(BaseModel):
    subject: str = Field(description="Entity name the claim is about (must match an extracted entity when possible)")
    text: str = Field(description="Single atomic factual claim supported by the text")


class Covariate(BaseModel):
    entity_name: str = Field(description="Entity name this attribute belongs to")
    attribute: str = Field(description="Attribute name, e.g. volume_mcf, year, disposition_type")
    value: str = Field(description="Value as string")
    unit: str = Field(default="", description="Unit if applicable, else empty")


class ExtractedGraph(BaseModel):
    entities: List[Entity] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)
    claims: List[Claim] = Field(default_factory=list)
    covariates: List[Covariate] = Field(default_factory=list)


EXTRACT_SYSTEM = """You are an expert at extracting structured knowledge from text.
For the given text chunk, extract:
1) Entities: people, places, organizations, concepts — name, type, short description.
2) Relationships between entities: source, target, relationship type, short description.
3) Claims: atomic factual statements; each has subject (entity name) and text (one claim).
4) Covariates: structured attributes tied to an entity — entity_name, attribute, value, optional unit (e.g. volumes, dates).
Only use information explicitly stated or clearly implied in the text."""

EXTRACT_SYSTEM_OIL_GAS = """You are an expert in US oil and gas production/disposition data extraction.
For the given text chunk, extract:
1) Entities and relationships explicitly present or clearly implied.
   Prioritize: State, County, OffshoreRegion, Commodity, DispositionType, TimePeriod, Measurement.
   Use relationship types such as LOCATED_IN, HAS_DISPOSITION, HAS_COMMODITY, IN_PERIOD, MEASURED_VOLUME.
2) Claims: one fact per item (e.g. which county had which volume for which disposition/year).
3) Covariates: numeric or categorical fields (entity_name, attribute, value, unit) e.g. Gas_Mcf=1234, year=2024.
Keep names normalized when possible."""


def _build_extract_prompt() -> ChatPromptTemplate:
    domain = os.environ.get("GRAPHRAG_EXTRACT_DOMAIN", "").strip().lower()
    system_prompt = EXTRACT_SYSTEM_OIL_GAS if domain == "oil_gas" else EXTRACT_SYSTEM
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Text:\n{text}"),
        ]
    )


def _get_extract_chain():
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0, api_key=OPENAI_API_KEY)
    structured_llm = llm.with_structured_output(ExtractedGraph)
    return _build_extract_prompt() | structured_llm


def extract_from_text(text: str) -> ExtractedGraph:
    """Extract entities and relationships from a single text chunk."""
    chain = _get_extract_chain()
    return chain.invoke({"text": text[:8000]})  # cap length


def _stable_id(*parts: str) -> str:
    h = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:28]
    return h


def _norm_desc(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def merge_entity_descriptions(
    existing: Optional[str],
    incoming: str,
    *,
    max_chars: int = ENTITY_DESCRIPTION_MAX_CHARS,
    separator: str = "\n---\n",
) -> str:
    """
    Acumula descrições da mesma entidade ao longo de vários TextUnits (estilo GraphRAG).
    Evita duplicar texto idêntico ou contido; limita o tamanho total.
    """
    inc = (incoming or "").strip()
    old = (existing or "").strip()
    if not inc:
        return old
    if not old:
        return inc[:max_chars] if max_chars > 0 else inc
    if _norm_desc(inc) == _norm_desc(old):
        return old
    if inc in old:
        return old
    if old in inc:
        return inc[:max_chars] if max_chars > 0 else inc
    for part in old.split(separator):
        if _norm_desc(part) == _norm_desc(inc):
            return old
    merged = old + separator + inc
    if max_chars <= 0 or len(merged) <= max_chars:
        return merged
    mid = "\n---\n...[truncated]...\n---\n"
    usable = max_chars - len(mid)
    if usable < 64:
        return merged[-max_chars:]
    head_budget = usable // 2
    tail_budget = usable - head_budget
    head = old[:head_budget]
    tail = inc[-tail_budget:] if len(inc) > tail_budget else inc
    out = head + mid + tail
    return out[:max_chars]


def _dedupe_entities_for_chunk(entities: List[Entity]) -> List[Entity]:
    """Uma entidade canónica por (nome normalizado, tipo normalizado) no mesmo TextUnit."""
    merged: dict[Tuple[str, str], Entity] = {}
    for e in entities:
        key = (normalize_token(e.name), normalize_token(e.type))
        if key not in merged:
            merged[key] = e
            continue
        cur = merged[key]
        desc = merge_entity_descriptions(cur.description, e.description)
        longer_name = cur.name if len(cur.name) >= len(e.name) else e.name
        longer_type = cur.type if len(cur.type) >= len(e.type) else e.type
        merged[key] = Entity(name=longer_name, type=longer_type, description=desc)
    return list(merged.values())


def _norm_name_to_entity_keys(deduped: List[Entity]) -> Dict[str, List[str]]:
    m: Dict[str, List[str]] = {}
    for e in deduped:
        k = make_entity_key(e.name, e.type)
        m.setdefault(normalize_token(e.name), []).append(k)
    return m


def persist_extraction_to_neo4j(tu_id: str, extraction: ExtractedGraph) -> None:
    """Persiste grafo com resolução por `entity_key`, aliases e ligações resolvidas por nome canónico."""
    graph = get_neo4j_graph()
    driver = graph._driver
    with driver.session() as session:
        deduped = _dedupe_entities_for_chunk(extraction.entities)
        norm_map = _norm_name_to_entity_keys(deduped)

        for e in deduped:
            ekey = make_entity_key(e.name, e.type)
            row = session.run(
                """
                MATCH (ex:Entity {entity_key: $k})
                RETURN ex.description AS d, ex.name AS n, ex.aliases AS a
                """,
                k=ekey,
            ).single()
            prev_desc = row["d"] if row else None
            description_merged = merge_entity_descriptions(prev_desc, e.description)
            prev_aliases = list(row["a"]) if row and row.get("a") else None
            canon, aliases = merge_canonical_name_and_aliases(
                row["n"] if row else None,
                prev_aliases,
                e.name,
            )
            session.run(
                """
                MERGE (x:Entity {entity_key: $entity_key})
                SET x.name = $name,
                    x.type = $type,
                    x.description = $description,
                    x.aliases = $aliases
                WITH x
                MATCH (t:TextUnit {id: $tu_id})
                MERGE (t)-[:MENTIONS]->(x)
                """,
                entity_key=ekey,
                name=canon,
                type=e.type.strip(),
                description=description_merged,
                aliases=aliases,
                tu_id=tu_id,
            )

        for r in extraction.relationships:
            sk = resolve_surface_to_entity_key(session, r.source, norm_map)
            tk = resolve_surface_to_entity_key(session, r.target, norm_map)
            if not sk or not tk:
                continue
            session.run(
                """
                MATCH (a:Entity {entity_key: $sa}), (b:Entity {entity_key: $sb})
                MERGE (a)-[rel:RELATES_TO {type: $rtype}]->(b)
                ON CREATE SET rel.description = $description
                """,
                sa=sk,
                sb=tk,
                rtype=r.type,
                description=r.description,
            )

        for cl in extraction.claims:
            cid = _stable_id("claim", tu_id, cl.subject, cl.text)
            session.run(
                """
                MERGE (c:Claim {id: $cid})
                SET c.text = $text, c.subject = $subject, c.source_tu = $tu_id
                """,
                cid=cid,
                text=cl.text,
                subject=cl.subject,
                tu_id=tu_id,
            )
            session.run(
                """
                MATCH (c:Claim {id: $cid}), (t:TextUnit {id: $tu_id})
                MERGE (t)-[:EVIDENCE_FOR]->(c)
                """,
                cid=cid,
                tu_id=tu_id,
            )
            ek = resolve_surface_to_entity_key(session, cl.subject, norm_map)
            if ek:
                session.run(
                    """
                    MATCH (e:Entity {entity_key: $ek}), (c:Claim {id: $cid})
                    MERGE (e)-[:HAS_CLAIM]->(c)
                    """,
                    ek=ek,
                    cid=cid,
                )

        for cv in extraction.covariates:
            vid = _stable_id("cov", tu_id, cv.entity_name, cv.attribute, cv.value)
            session.run(
                """
                MERGE (v:Covariate {id: $vid})
                SET v.name = $attr, v.value = $value, v.unit = $unit, v.source_tu = $tu_id,
                    v.entity_name = $entity_name
                """,
                vid=vid,
                attr=cv.attribute,
                value=cv.value,
                unit=cv.unit or "",
                entity_name=cv.entity_name,
                tu_id=tu_id,
            )
            session.run(
                """
                MATCH (v:Covariate {id: $vid}), (t:TextUnit {id: $tu_id})
                MERGE (t)-[:EVIDENCE_FOR]->(v)
                """,
                vid=vid,
                tu_id=tu_id,
            )
            ek = resolve_surface_to_entity_key(session, cv.entity_name, norm_map)
            if ek:
                session.run(
                    """
                    MATCH (e:Entity {entity_key: $ek}), (v:Covariate {id: $vid})
                    MERGE (e)-[:HAS_COVARIATE]->(v)
                    """,
                    ek=ek,
                    vid=vid,
                )


def run_extract_on_chunks(chunks: List[dict]) -> None:
    """Run extraction on each chunk. Each chunk is a dict with tu_id and text."""
    if chunks:
        with get_neo4j_graph()._driver.session() as session:
            backfill_entity_keys_cypher(session)
    for chunk in chunks:
        tu_id = chunk.get("tu_id", str(id(chunk)))
        text = chunk.get("text", "")
        extraction = extract_from_text(text)
        persist_extraction_to_neo4j(tu_id, extraction)
