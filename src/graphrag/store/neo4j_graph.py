"""Neo4j graph connection for LangChain."""

from langchain_community.graphs import Neo4jGraph

from graphrag.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

_graph: Neo4jGraph | None = None


def get_neo4j_graph() -> Neo4jGraph:
    """Return a singleton Neo4jGraph instance."""
    global _graph
    if _graph is None:
        _graph = Neo4jGraph(
            url=NEO4J_URI,
            username=NEO4J_USER,
            password=NEO4J_PASSWORD,
        )
    return _graph
