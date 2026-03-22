"""Local mixed retrieval alinhado ao GraphRAG: entidades, fan-out, claims, covariates, ranking unificado, orçamento em tokens."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from graphrag.config import (
    LOCAL_CLAIMS_POOL,
    LOCAL_COVARIATES_POOL,
    LOCAL_ENTITY_K_PER_QUERY,
    LOCAL_ENTITY_LINKED_REPORTS_POOL,
    LOCAL_ENTITY_TOP_K,
    LOCAL_MAX_DATA_TOKENS,
    LOCAL_NEIGHBOR_ENTITIES_POOL,
    LOCAL_RELATIONSHIP_LINES_POOL,
    LOCAL_SYNTH_CONTEXT_DOC_CAP,
    LOCAL_TEXT_UNITS_FROM_GRAPH_POOL,
    LOCAL_TEXT_UNITS_VECTOR_POOL,
    RETRIEVAL_TOP_K,
)
from graphrag.retrieval.token_budget import RankedContextChunk, pack_chunks_by_token_budget
from graphrag.store.neo4j_graph import get_neo4j_graph
from graphrag.store.vector_index import get_vector_index_entities, get_vector_index_text_units


def _retrieval_queries(state: dict) -> List[str]:
    ordered: List[str] = []
    seen: set[str] = set()
    q = (state.get("question") or "").strip()
    if q:
        ordered.append(q)
        seen.add(q)
    for sq in state.get("subqueries") or []:
        s = getattr(sq, "sub_query", None)
        if s is None and isinstance(sq, dict):
            s = sq.get("sub_query")
        if not s:
            continue
        t = str(s).strip()
        if t and t not in seen:
            seen.add(t)
            ordered.append(t)
    return ordered


def _entity_name_from_doc(doc: Any) -> Optional[str]:
    md = getattr(doc, "metadata", None) or {}
    name = md.get("name")
    if name is not None and str(name).strip():
        return str(name).strip()
    return None


def collect_seed_entities_scored(
    queries: Sequence[str],
    k_per_query: int,
    max_total: int,
) -> Tuple[List[str], Dict[str, float]]:
    """Entidades âncora com score de similaridade (maior = mais alinhado à query)."""
    store = get_vector_index_entities()
    if store is None or max_total <= 0:
        return [], {}
    best: Dict[str, float] = {}
    for q in queries:
        if not q or not q.strip():
            continue
        try:
            pairs = store.similarity_search_with_score(q.strip(), k=k_per_query)
        except Exception:
            continue
        for doc, sc in pairs:
            name = _entity_name_from_doc(doc)
            if not name:
                continue
            s = float(sc)
            if name not in best or s > best[name]:
                best[name] = s
    ordered = sorted(best.keys(), key=lambda n: best[n], reverse=True)[:max_total]
    slim = {n: best[n] for n in ordered}
    return ordered, slim


def _norm_entity_scores(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    span = hi - lo if hi > lo else 1.0
    return {k: (v - lo) / span for k, v in scores.items()}


def _fetch_text_units_via_graph(
    names: List[str], limit: int
) -> List[Tuple[str, str, int]]:
    """Retorna (id, text, overlap_count)."""
    if not names or limit <= 0:
        return []
    driver = get_neo4j_graph()._driver
    with driver.session() as session:
        result = session.run(
            """
            MATCH (t:TextUnit)-[:MENTIONS]->(e:Entity)
            WHERE e.name IN $names
            WITH t, count(DISTINCT e) AS overlap
            ORDER BY overlap DESC, t.id
            LIMIT $limit
            RETURN t.id AS id, t.text AS text, overlap
            """,
            names=names,
            limit=int(limit),
        )
        return [
            (str(r["id"]), (r["text"] or "")[:12000], int(r["overlap"]))
            for r in result
        ]


def _fetch_relationship_lines(names: List[str], limit: int) -> List[str]:
    if not names or limit <= 0:
        return []
    driver = get_neo4j_graph()._driver
    with driver.session() as session:
        result = session.run(
            """
            MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
            WHERE a.name IN $names OR b.name IN $names
            RETURN DISTINCT a.name AS src, b.name AS tgt, r.type AS rt, coalesce(r.description, '') AS rd
            LIMIT $limit
            """,
            names=names,
            limit=int(limit),
        )
        lines = []
        for r in result:
            extra = f" — {r['rd']}" if r["rd"] else ""
            lines.append(f"- ({r['src']})-[:{r['rt']}]->({r['tgt']}){extra}")
        return lines


def _fetch_neighbor_entity_lines(names: List[str], limit: int) -> List[str]:
    if not names or limit <= 0:
        return []
    driver = get_neo4j_graph()._driver
    with driver.session() as session:
        result = session.run(
            """
            MATCH (a:Entity)-[:RELATES_TO]-(b:Entity)
            WHERE a.name IN $names AND NOT b.name IN $names
            RETURN DISTINCT b.name AS name, b.type AS typ, coalesce(b.description, '') AS desc
            LIMIT $limit
            """,
            names=names,
            limit=int(limit),
        )
        lines = []
        for r in result:
            desc = (r["desc"] or "").strip()
            tail = f": {desc}" if desc else ""
            lines.append(f"- {r['name']} ({r['typ']}){tail}")
        return lines


def _fetch_entity_linked_reports(
    names: List[str], limit: int
) -> List[Tuple[str, str, int]]:
    """(community_id, content, relevance = count of seed entities in community)."""
    if not names or limit <= 0:
        return []
    driver = get_neo4j_graph()._driver
    with driver.session() as session:
        result = session.run(
            """
            MATCH (e:Entity)-[:IN_COMMUNITY]->(c:Community)-[:HAS_REPORT]->(r:CommunityReport)
            WHERE e.name IN $names
            WITH r, count(DISTINCT e) AS relevance
            ORDER BY relevance DESC
            LIMIT $limit
            RETURN r.community_id AS cid, r.content AS content, relevance
            """,
            names=names,
            limit=int(limit),
        )
        return [
            (str(r["cid"]), (r["content"] or "")[:12000], int(r["relevance"]))
            for r in result
        ]


def _fetch_claims(names: List[str], limit: int) -> List[Tuple[str, str, str]]:
    """(id, text, subject entity name)."""
    if not names or limit <= 0:
        return []
    driver = get_neo4j_graph()._driver
    with driver.session() as session:
        result = session.run(
            """
            MATCH (e:Entity)-[:HAS_CLAIM]->(c:Claim)
            WHERE e.name IN $names
            RETURN c.id AS id, c.text AS text, c.subject AS subject
            LIMIT $limit
            """,
            names=names,
            limit=int(limit),
        )
        rows = []
        for r in result:
            rows.append((str(r["id"]), (r["text"] or "").strip(), str(r["subject"] or "")))
        return rows


def _fetch_covariate_rows(names: List[str], limit: int) -> List[Tuple[str, str, str, str]]:
    """(entity_name, attr, value, unit)."""
    if not names or limit <= 0:
        return []
    driver = get_neo4j_graph()._driver
    with driver.session() as session:
        result = session.run(
            """
            MATCH (e:Entity)-[:HAS_COVARIATE]->(v:Covariate)
            WHERE e.name IN $names
            RETURN e.name AS en, v.name AS attr, v.value AS val, coalesce(v.unit, '') AS unit
            LIMIT $limit
            """,
            names=names,
            limit=int(limit),
        )
        return [
            (str(r["en"]), str(r["attr"]), str(r["val"]), str(r["unit"] or ""))
            for r in result
        ]


def _vector_text_units(queries: Sequence[str], k: int) -> List[Tuple[str, str]]:
    store = get_vector_index_text_units()
    if store is None or k <= 0:
        return []
    primary = queries[0].strip() if queries else ""
    if not primary:
        return []
    try:
        docs = store.similarity_search(primary, k=k)
    except Exception:
        return []
    out: List[Tuple[str, str]] = []
    for d in docs:
        md = getattr(d, "metadata", None) or {}
        tid = md.get("id")
        if tid is None:
            tid = md.get("t.id")
        text = getattr(d, "page_content", None) or ""
        key = str(tid) if tid is not None else f"vec:{hash(text[:100])}"
        out.append((key, text[:12000]))
    return out


def _split_graph_and_vector_tus(
    graph_rows: List[Tuple[str, str, int]],
    vec_rows: List[Tuple[str, str]],
    cap: int,
) -> Tuple[List[Tuple[str, str, int]], List[Tuple[str, str, int]]]:
    """Prioriza TextUnits do grafo; completa com vetor até `cap` ids únicos."""
    if cap <= 0:
        return [], []
    seen: set[str] = set()
    gout: List[Tuple[str, str, int]] = []
    for tid, text, ov in graph_rows:
        if tid in seen:
            continue
        seen.add(tid)
        gout.append((tid, text, ov))
        if len(gout) >= cap:
            return gout, []
    vout: List[Tuple[str, str, int]] = []
    room = cap - len(gout)
    for tid, text in vec_rows:
        if room <= 0:
            break
        if tid in seen:
            continue
        seen.add(tid)
        vout.append((tid, text, 0))
        room -= 1
    return gout, vout


def _doc_from_ranked(ch: RankedContextChunk) -> Dict[str, Any]:
    return {
        "page_content": ch.content,
        "metadata": {
            "source": "local_search",
            "role": ch.role,
            "score": ch.score,
            **ch.metadata,
        },
    }


def build_local_search_context(state: dict) -> Tuple[List[dict], List[str]]:
    """
    Candidatos multi-fonte → scores unificados → ordenação → pack por LOCAL_MAX_DATA_TOKENS.
    Inclui claims e covariates quando existirem no grafo (pós re-indexação com extract atualizado).
    """
    queries = _retrieval_queries(state)
    seeds, entity_scores = collect_seed_entities_scored(
        queries,
        k_per_query=LOCAL_ENTITY_K_PER_QUERY,
        max_total=LOCAL_ENTITY_TOP_K,
    )
    norm_scores = _norm_entity_scores(entity_scores)
    avg_seed_norm = sum(norm_scores.values()) / len(norm_scores) if norm_scores else 0.0

    candidates: List[RankedContextChunk] = []

    if seeds:
        driver = get_neo4j_graph()._driver
        with driver.session() as session:
            res = session.run(
                """
                MATCH (e:Entity)
                WHERE e.name IN $names
                RETURN e.name AS name, e.type AS typ, coalesce(e.description, '') AS desc
                ORDER BY e.name
                """,
                names=seeds,
            )
            lines = [
                f"- {r['name']} ({r['typ']}): {(r['desc'] or '').strip()}".strip()
                for r in res
            ]
        if lines:
            seed_line_score = 90.0 + 10.0 * avg_seed_norm
            candidates.append(
                RankedContextChunk(
                    role="seed_entities",
                    content="=== Entidades relevantes (âncoras) ===\n" + "\n".join(lines),
                    score=seed_line_score,
                    metadata={"kind": "seed_entities"},
                )
            )

        rel_lines = _fetch_relationship_lines(seeds, LOCAL_RELATIONSHIP_LINES_POOL)
        if rel_lines:
            candidates.append(
                RankedContextChunk(
                    role="relationships",
                    content="=== Relacionamentos ===\n" + "\n".join(rel_lines),
                    score=76.0 + 8.0 * avg_seed_norm,
                    metadata={"kind": "relationships", "lines": len(rel_lines)},
                )
            )

        nb_lines = _fetch_neighbor_entity_lines(seeds, LOCAL_NEIGHBOR_ENTITIES_POOL)
        if nb_lines:
            candidates.append(
                RankedContextChunk(
                    role="neighbor_entities",
                    content="=== Entidades vizinhas (1 salto) ===\n" + "\n".join(nb_lines),
                    score=68.0 + 6.0 * avg_seed_norm,
                    metadata={"kind": "neighbors"},
                )
            )

        for cid, content, rel in _fetch_entity_linked_reports(
            seeds, LOCAL_ENTITY_LINKED_REPORTS_POOL
        ):
            candidates.append(
                RankedContextChunk(
                    role="entity_community_report",
                    content=f"=== Relatório de comunidade ({cid}) ===\n{content}",
                    score=72.0 + min(18.0, float(rel) * 3.0),
                    metadata={"kind": "community_report", "community_id": cid, "relevance": rel},
                )
            )

        seen_claims: set[str] = set()
        for cid, text, subj in _fetch_claims(seeds, LOCAL_CLAIMS_POOL):
            if cid in seen_claims or not text:
                continue
            seen_claims.add(cid)
            sn = norm_scores.get(subj, avg_seed_norm)
            candidates.append(
                RankedContextChunk(
                    role="claim",
                    content=f"=== Afirmação (claim) [{subj}] ===\n{text}",
                    score=85.0 + 12.0 * sn,
                    metadata={"kind": "claim", "claim_id": cid, "subject": subj},
                )
            )

        cov_rows = _fetch_covariate_rows(seeds, LOCAL_COVARIATES_POOL)
        if cov_rows:
            cov_lines = []
            for en, attr, val, unit in cov_rows:
                u = f" {unit}" if unit else ""
                cov_lines.append(f"- {en} :: {attr} = {val}{u}")
            candidates.append(
                RankedContextChunk(
                    role="covariates",
                    content="=== Covariates (atributos ligados a entidades) ===\n" + "\n".join(cov_lines),
                    score=78.0 + 7.0 * avg_seed_norm,
                    metadata={"kind": "covariates", "rows": len(cov_rows)},
                )
            )

    pool_cap = LOCAL_TEXT_UNITS_FROM_GRAPH_POOL + LOCAL_TEXT_UNITS_VECTOR_POOL
    graph_tu = (
        _fetch_text_units_via_graph(seeds, LOCAL_TEXT_UNITS_FROM_GRAPH_POOL) if seeds else []
    )
    vec_tu = _vector_text_units(queries, LOCAL_TEXT_UNITS_VECTOR_POOL)
    graph_merged, vec_merged = _split_graph_and_vector_tus(graph_tu, vec_tu, pool_cap)

    for tid, text, overlap in graph_merged:
        candidates.append(
            RankedContextChunk(
                role="text_unit",
                content=f"=== Trecho (TextUnit {tid}, menções a âncoras: {overlap}) ===\n{text}",
                score=58.0 + min(22.0, float(overlap) * 5.0) + 8.0 * avg_seed_norm,
                metadata={"kind": "text_unit_graph", "text_unit_id": tid, "overlap": overlap},
            )
        )
    for i, (tid, text, _) in enumerate(vec_merged):
        candidates.append(
            RankedContextChunk(
                role="text_unit",
                content=f"=== Trecho (TextUnit {tid}, similaridade vetorial) ===\n{text}",
                score=48.0 - min(12.0, float(i) * 1.5),
                metadata={"kind": "text_unit_vector", "text_unit_id": tid, "vec_rank": i},
            )
        )

    if not candidates and queries:
        for tid, text in _vector_text_units(queries, RETRIEVAL_TOP_K):
            candidates.append(
                RankedContextChunk(
                    role="text_unit",
                    content=f"=== Trecho (TextUnit {tid}) ===\n{text}",
                    score=45.0,
                    metadata={"kind": "text_unit_fallback", "text_unit_id": tid},
                )
            )

    packed = pack_chunks_by_token_budget(candidates, LOCAL_MAX_DATA_TOKENS)
    docs = [_doc_from_ranked(ch) for ch in packed][:LOCAL_SYNTH_CONTEXT_DOC_CAP]
    return docs, seeds
