"""Retrieval: vector search over Neo4j TextUnits. Returns a runnable that yields source_documents."""

from typing import Optional, Callable

from graphrag.config import RETRIEVAL_TOP_K
from graphrag.store.vector_index import get_vector_index_text_units


def get_retrieval_chain() -> Optional[Callable]:
    """Return an invokable that takes {"query": str} and returns {"source_documents": list}. None if index not yet created."""
    vector_store = get_vector_index_text_units()
    if vector_store is None:
        return None
    retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_TOP_K})

    def invoke(inputs: dict) -> dict:
        query = inputs.get("query", "")
        docs = retriever.invoke(query)
        return {"source_documents": docs}

    return invoke
