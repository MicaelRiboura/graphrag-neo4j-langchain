"""LangGraph state for GraphRAG query flow."""

from typing import Literal, TypedDict, Any, List, Optional


class GraphRAGState(TypedDict, total=False):
    """State passed through the query graph."""

    question: str
    search_type: Literal["local", "global"]
    subqueries: Any  # list of subquery objects from decompose
    context_docs: List[Any]
    cypher_result: Any
    community_reports: List[str]
    final_answer: str
