"""Global search: retrieve community reports + map-reduce (Microsoft GraphRAG-style)."""

from __future__ import annotations

import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from graphrag.config import (
    GLOBAL_MAP_BATCH_SIZE,
    GLOBAL_MAP_MAX_CONCURRENT,
    GLOBAL_MAP_MAX_POINTS_PER_BATCH,
    GLOBAL_REDUCE_TOP_POINTS,
    GLOBAL_REPORTS_POOL_K,
    OPENAI_API_KEY,
    LLM_MODEL,
)
from graphrag.prompts.global_search import GLOBAL_MAP_PROMPT, GLOBAL_REDUCE_PROMPT
from graphrag.prompts.synthesis import SYNTHESIS_PROMPT
from graphrag.store.vector_index import get_vector_index_reports


class RatedPoint(BaseModel):
    """Um ponto intermediário do estágio map."""

    description: str = Field(description="Factual claim grounded in the community reports")
    score: int = Field(ge=0, le=100, description="Importance 0-100 for answering the question")


class MapBatchOutput(BaseModel):
    points: List[RatedPoint] = Field(default_factory=list)


def fetch_global_community_reports(question: str) -> List[str]:
    """
    Recupera um conjunto amplo de relatórios (vetor) para alimentar o map-reduce.
    A lista é embaralhada para reduzir viés de ordem, como no Global Search do GraphRAG.
    """
    store = get_vector_index_reports()
    if store is None:
        return []
    k = max(1, GLOBAL_REPORTS_POOL_K)
    try:
        docs = store.similarity_search(question.strip(), k=k)
    except Exception:
        return []
    reports = [d.page_content or "" for d in docs if (d.page_content or "").strip()]
    rng = random.Random((hash(question) & 0xFFFFFFFF) ^ len(reports))
    rng.shuffle(reports)
    return reports


def _normalize_point_key(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"\s+", " ", t)
    return t[:500]


def _dedupe_points(points: List[RatedPoint]) -> List[RatedPoint]:
    best: dict[str, RatedPoint] = {}
    for p in points:
        key = _normalize_point_key(p.description)
        if not key:
            continue
        prev = best.get(key)
        if prev is None or p.score > prev.score:
            best[key] = p
    return list(best.values())


def _map_one_batch(question: str, batch_text: str, llm: ChatOpenAI) -> List[RatedPoint]:
    structured = llm.with_structured_output(MapBatchOutput)
    chain = GLOBAL_MAP_PROMPT | structured
    try:
        out = chain.invoke({"question": question, "batch": batch_text})
        if isinstance(out, MapBatchOutput):
            pts = out.points[:GLOBAL_MAP_MAX_POINTS_PER_BATCH]
            return pts
    except Exception:
        pass
    return []


def _batch_reports(reports: List[str], batch_size: int) -> List[str]:
    if not reports:
        return []
    bs = max(1, batch_size)
    return ["\n\n---\n\n".join(reports[i : i + bs]) for i in range(0, len(reports), bs)]


def global_search_map_reduce(question: str, reports: List[str]) -> str:
    """
    Map: cada lote de relatórios → pontos com score.
    Reduce: ordena, deduplica, trunca e gera a resposta final.
    """
    q = (question or "").strip()
    if not reports:
        return "Nenhum relatório de comunidade encontrado. Tente indexar documentos primeiro."

    llm_map = ChatOpenAI(model=LLM_MODEL, temperature=0, api_key=OPENAI_API_KEY)
    llm_reduce = ChatOpenAI(model=LLM_MODEL, temperature=0, api_key=OPENAI_API_KEY)

    batches = _batch_reports(reports, GLOBAL_MAP_BATCH_SIZE)
    all_points: List[RatedPoint] = []

    max_workers = min(GLOBAL_MAP_MAX_CONCURRENT, len(batches)) or 1
    if max_workers <= 1:
        for b in batches:
            all_points.extend(_map_one_batch(q, b, llm_map))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(_map_one_batch, q, b, llm_map): i for i, b in enumerate(batches)}
            for fut in as_completed(futs):
                all_points.extend(fut.result() or [])

    all_points = _dedupe_points(all_points)
    all_points.sort(key=lambda p: p.score, reverse=True)
    top = all_points[: max(1, GLOBAL_REDUCE_TOP_POINTS)]

    if not top:
        # Fallback: uma passagem direta nos relatórios (comportamento legado).
        context = "\n\n---\n\n".join(reports[: min(8, len(reports))])
        chain = SYNTHESIS_PROMPT | llm_reduce
        out = chain.invoke({"context": context, "question": q})
        return out.content if hasattr(out, "content") else str(out)

    points_block = "\n".join(
        f"- [{p.score}] {p.description.strip()}" for p in top if p.description.strip()
    )
    chain = GLOBAL_REDUCE_PROMPT | llm_reduce
    out = chain.invoke({"question": q, "points": points_block})
    return out.content if hasattr(out, "content") else str(out)
