"""Generate embeddings for TextUnits, Entity descriptions, and CommunityReport content; write to Neo4j vector indexes."""

from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from graphrag.config import (
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    INDEX_NAME_TEXT_UNITS,
    INDEX_NAME_ENTITIES,
    INDEX_NAME_REPORTS,
    EMBEDDING_DIMENSION,
)
from graphrag.indexing.entity_resolution import backfill_entity_keys_cypher
from graphrag.store.neo4j_graph import get_neo4j_graph


def _get_embeddings():
    return OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY)


def create_vector_index_if_not_exists(index_name: str, node_label: str, text_property: str) -> None:
    """Create a Neo4j vector index (Neo4j 5.x) if it does not exist."""
    driver = get_neo4j_graph()._driver
    with driver.session() as session:
        session.run(
            f"""
            CREATE VECTOR INDEX {index_name} IF NOT EXISTS
            FOR (n:{node_label}) ON (n.embedding)
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: {EMBEDDING_DIMENSION},
                `vector.similarity_function`: 'cosine'
            }}}}
            """
        )


def embed_text_units() -> None:
    """Fetch all TextUnits, compute embeddings, set property and ensure vector index."""
    driver = get_neo4j_graph()._driver
    embeddings = _get_embeddings()
    with driver.session() as session:
        result = session.run("MATCH (t:TextUnit) RETURN t.id AS id, t.text AS text")
        rows = list(result)
    if not rows:
        return
    texts = [r["text"] or "" for r in rows]
    ids = [r["id"] for r in rows]
    vecs = embeddings.embed_documents(texts)
    driver = get_neo4j_graph()._driver
    with driver.session() as session:
        for i, (nid, vec) in enumerate(zip(ids, vecs)):
            session.run(
                "MATCH (t:TextUnit {id: $id}) SET t.embedding = $embedding",
                id=nid,
                embedding=vec,
            )
    create_vector_index_if_not_exists(INDEX_NAME_TEXT_UNITS, "TextUnit", "text")


def embed_entities() -> None:
    """Fetch all Entity descriptions, compute embeddings, set property and ensure vector index."""
    driver = get_neo4j_graph()._driver
    embeddings = _get_embeddings()
    with driver.session() as session:
        backfill_entity_keys_cypher(session)
        result = session.run(
            "MATCH (e:Entity) RETURN e.entity_key AS ek, e.description AS description, e.name AS name"
        )
        rows = [r for r in result if r.get("ek")]
    if not rows:
        return
    texts = [r["description"] or r["name"] or "" for r in rows]
    keys = [r["ek"] for r in rows]
    vecs = embeddings.embed_documents(texts)
    with driver.session() as session:
        for ek, vec in zip(keys, vecs):
            session.run(
                "MATCH (e:Entity {entity_key: $ek}) SET e.embedding = $embedding",
                ek=ek,
                embedding=vec,
            )
    create_vector_index_if_not_exists(INDEX_NAME_ENTITIES, "Entity", "description")


def embed_reports() -> None:
    """Fetch all CommunityReport content, compute embeddings, set property and ensure vector index."""
    driver = get_neo4j_graph()._driver
    embeddings = _get_embeddings()
    with driver.session() as session:
        result = session.run(
            "MATCH (r:CommunityReport) RETURN r.community_id AS id, r.content AS content"
        )
        rows = list(result)
    if not rows:
        return
    texts = [r["content"] or "" for r in rows]
    ids = [r["id"] for r in rows]
    vecs = embeddings.embed_documents(texts)
    with driver.session() as session:
        for cid, vec in zip(ids, vecs):
            session.run(
                "MATCH (r:CommunityReport {community_id: $id}) SET r.embedding = $embedding",
                id=cid,
                embedding=vec,
            )
    create_vector_index_if_not_exists(INDEX_NAME_REPORTS, "CommunityReport", "content")


def run_embed_all() -> None:
    """Run embedding for TextUnits, Entities, and CommunityReports."""
    embed_text_units()
    embed_entities()
    embed_reports()
