"""Community detection on entity graph; persist Community and ENTITY_IN_COMMUNITY to Neo4j."""

from typing import List, Set

from graphrag.store.neo4j_graph import get_neo4j_graph


def get_entity_graph_from_neo4j() -> tuple[List[str], List[tuple[str, str]]]:
    """Fetch (nodes, edges) from Neo4j: nodes = entity names, edges = (source, target)."""
    graph = get_neo4j_graph()
    driver = graph._driver
    nodes = []
    edges = []
    with driver.session() as session:
        result = session.run("MATCH (e:Entity) RETURN e.name AS name")
        nodes = [r["name"] for r in result]
        result = session.run(
            """
            MATCH (a:Entity)-[:RELATES_TO]->(b:Entity)
            RETURN a.name AS source, b.name AS target
            """
        )
        edges = [(r["source"], r["target"]) for r in result]
    return nodes, edges


def detect_communities_leiden(nodes: List[str], edges: List[tuple[str, str]], min_community_size: int = 2) -> List[Set[str]]:
    """Run Leiden community detection. Returns list of sets of entity names. Falls back to single-node communities."""
    try:
        import igraph as ig
    except ImportError:
        try:
            import networkx as nx
            from networkx.algorithms import community
            G = nx.Graph()
            G.add_nodes_from(nodes)
            G.add_edges_from(edges)
            communities = community.louvain_communities(G)
            print('nx funcionou')
            return [set(c) for c in communities if len(c) >= min_community_size or len(c) == 1]
        except ImportError:
            return [{n} for n in nodes]
    print('ig funcionou')
    name_to_idx = {n: i for i, n in enumerate(nodes)}
    g = ig.Graph(directed=False)
    g.add_vertices(len(nodes))
    for a, b in edges:
        if a in name_to_idx and b in name_to_idx:
            g.add_edges([(name_to_idx[a], name_to_idx[b])])
    try:
        part = g.community_leiden(objective_function="modularity")
    except Exception:
        part = g.community_multilevel()
    communities = []
    for cluster in part:
        names = {nodes[i] for i in cluster}
        if len(names) >= min_community_size or len(names) == 1:
            communities.append(names)
    return communities if communities else [{n} for n in nodes]


def persist_communities_to_neo4j(communities: List[Set[str]]) -> None:
    """Create Community nodes and (Entity)-[:IN_COMMUNITY]->(Community)."""
    driver = get_neo4j_graph()._driver
    with driver.session() as session:
        for i, entity_names in enumerate(communities):
            comm_id = f"community_{i}"
            session.run(
                "MERGE (c:Community {id: $id})",
                id=comm_id,
            )
            for name in entity_names:
                session.run(
                    """
                    MATCH (e:Entity {name: $name}), (c:Community {id: $comm_id})
                    MERGE (e)-[:IN_COMMUNITY]->(c)
                    """,
                    name=name,
                    comm_id=comm_id,
                )


def run_communities(min_community_size: int = 2) -> List[Set[str]]:
    """Load graph from Neo4j, run community detection, persist; return communities."""
    nodes, edges = get_entity_graph_from_neo4j()
    if not nodes:
        return []
    communities = detect_communities_leiden(nodes, edges, min_community_size=min_community_size)
    persist_communities_to_neo4j(communities)
    return communities
