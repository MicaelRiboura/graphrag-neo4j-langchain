"""Neo4j vector indexes for TextUnits, Entities, and Community Reports."""

from typing import Optional

from langchain_community.vectorstores import Neo4jVector

from graphrag.config import (
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
    INDEX_NAME_TEXT_UNITS,
    INDEX_NAME_ENTITIES,
    INDEX_NAME_REPORTS,
    EMBEDDING_DIMENSION,
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
)
from graphrag.monitoring.token_cost import TrackedOpenAIEmbeddings


def _get_embeddings() -> TrackedOpenAIEmbeddings:
    return TrackedOpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY,
    )


def _neo4j_vector_store(
    index_name: str,
    node_label: str,
    text_property: str,
    embedding_property: str = "embedding",
) -> Neo4jVector:
    return Neo4jVector.from_existing_index(
        embedding=_get_embeddings(),
        url=NEO4J_URI,
        username=NEO4J_USER,
        password=NEO4J_PASSWORD,
        index_name=index_name,
        node_label=node_label,
        text_node_property=text_property,
        embedding_node_property=embedding_property,
    )


def get_vector_index_text_units() -> Optional[Neo4jVector]:
    """Return Neo4j vector store for TextUnit chunks. None if index does not exist yet."""
    try:
        return _neo4j_vector_store(
            index_name=INDEX_NAME_TEXT_UNITS,
            node_label="TextUnit",
            text_property="text",
        )
    except Exception:
        return None


def get_vector_index_entities() -> Optional[Neo4jVector]:
    """Return Neo4j vector store for Entity descriptions. None if index does not exist yet."""
    try:
        return _neo4j_vector_store(
            index_name=INDEX_NAME_ENTITIES,
            node_label="Entity",
            text_property="description",
        )
    except Exception:
        return None


def get_vector_index_reports() -> Optional[Neo4jVector]:
    """Return Neo4j vector store for Community Reports. None if index does not exist yet."""
    try:
        return _neo4j_vector_store(
            index_name=INDEX_NAME_REPORTS,
            node_label="CommunityReport",
            text_property="content",
        )
    except Exception:
        return None
