"""LangGraph state for GraphRAG query flow."""

from typing import Literal, TypedDict, Any, List, Optional


class GraphRAGState(TypedDict, total=False):
    """State passed through the query graph."""

    question: str
    search_type: Literal["local", "global"]
    subqueries: Any  # list of subquery objects from decompose
    seed_entities: List[str]  # entity names from local vector match (GraphRAG-style anchors)
    context_docs: List[Any]
    cypher_result: Any
    cypher_error: str
    community_reports: List[str]
    final_answer: str
