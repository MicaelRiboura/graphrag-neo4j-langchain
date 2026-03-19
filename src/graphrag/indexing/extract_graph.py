"""Extract entities and relationships from TextUnits using LLM; persist to Neo4j."""

import os
from typing import List, Any

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

from graphrag.config import OPENAI_API_KEY, LLM_MODEL
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


class ExtractedGraph(BaseModel):
    entities: List[Entity] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)


EXTRACT_SYSTEM = """You are an expert at extracting structured knowledge from text.
For the given text chunk, extract entities (people, places, organizations, concepts) and relationships between them.
For each entity give: name, type, short description.
For each relationship give: source entity name, target entity name, type, short description.
Only use entities and relationships explicitly mentioned or clearly implied in the text."""

EXTRACT_SYSTEM_OIL_GAS = """You are an expert in US oil and gas production/disposition data extraction.
For the given text chunk, extract entities and relationships explicitly present or clearly implied.
Prioritize these entity types when applicable: State, County, OffshoreRegion, Commodity, DispositionType, TimePeriod, Measurement.
Use concise relationship types such as LOCATED_IN, HAS_DISPOSITION, HAS_COMMODITY, IN_PERIOD, MEASURED_VOLUME.
Include short descriptions and keep names normalized when possible."""


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


def persist_extraction_to_neo4j(tu_id: str, extraction: ExtractedGraph) -> None:
    """Create Entity and Relationship nodes and link to TextUnit."""
    graph = get_neo4j_graph()
    driver = graph._driver
    with driver.session() as session:
        for e in extraction.entities:
            session.run(
                """
                MERGE (e:Entity {name: $name, type: $type})
                ON CREATE SET e.description = $description
                WITH e
                MATCH (t:TextUnit {id: $tu_id})
                MERGE (t)-[:MENTIONS]->(e)
                """,
                name=e.name,
                type=e.type,
                description=e.description,
                tu_id=tu_id,
            )
        for r in extraction.relationships:
            session.run(
                """
                MATCH (a:Entity {name: $source}), (b:Entity {name: $target})
                MERGE (a)-[rel:RELATES_TO {type: $type}]->(b)
                ON CREATE SET rel.description = $description
                """,
                source=r.source,
                target=r.target,
                type=r.type,
                description=r.description,
            )


def run_extract_on_chunks(chunks: List[dict]) -> None:
    """Run extraction on each chunk. Each chunk is a dict with tu_id and text."""
    for chunk in chunks:
        tu_id = chunk.get("tu_id", str(id(chunk)))
        text = chunk.get("text", "")
        extraction = extract_from_text(text)
        persist_extraction_to_neo4j(tu_id, extraction)
