"""Deteção hierárquica de comunidades (estilo GraphRAG) e persistência no Neo4j."""

from __future__ import annotations

from typing import List, Set, Tuple

from graphrag.config import COMMUNITY_MAX_LEVELS, COMMUNITY_MIN_SIZE
from graphrag.store.neo4j_graph import get_neo4j_graph

# Cada nível: lista de (community_id, membros). L0: membros = nomes de entidades. L1+: membros = ids das comunidades filhas.
HierarchyLevel = List[Tuple[str, Set[str]]]


def get_entity_graph_from_neo4j() -> tuple[List[str], List[tuple[str, str]]]:
    """Fetch (nodes, edges) from Neo4j: nodes = entity names, edges = (source, target)."""
    graph = get_neo4j_graph()
    driver = graph._driver
    nodes: List[str] = []
    edges: List[tuple[str, str]] = []
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


def detect_communities_partition(
    nodes: List[str],
    edges: List[tuple[str, str]],
    min_community_size: int = 2,
) -> List[Set[str]]:
    """
    Partição em comunidades (Leiden ou Louvain). `nodes` pode ser nomes de entidades ou ids de comunidades.
    """
    if not nodes:
        return []
    nodes = list(dict.fromkeys(nodes))
    if not edges:
        return [{n} for n in nodes]

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
            out = [set(c) for c in communities if len(c) >= min_community_size or len(c) == 1]
            return out if out else [{n} for n in nodes]
        except ImportError:
            return [{n} for n in nodes]

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
    communities: List[Set[str]] = []
    for cluster in part:
        names = {nodes[i] for i in cluster}
        if len(names) >= min_community_size or len(names) == 1:
            communities.append(names)
    return communities if communities else [{n} for n in nodes]


def build_community_hierarchy(
    entity_nodes: List[str],
    entity_edges: List[tuple[str, str]],
    *,
    min_community_size: int = COMMUNITY_MIN_SIZE,
    max_levels: int = COMMUNITY_MAX_LEVELS,
) -> List[HierarchyLevel]:
    """
    Constrói hierarquia L0 → L1 → … (GraphRAG-style).

    - L0: comunidades de entidades.
    - Lk (k≥1): cada comunidade agrupa ids de comunidades do nível anterior;
      arestas no super-grafo existem se houver pelo menos uma aresta de entidade entre elas.
    Para quando só resta uma comunidade no nível atual, o particionamento colapsa a 1, ou atinge max_levels.
    """
    if not entity_nodes:
        return []

    entity_nodes = list(dict.fromkeys(entity_nodes))
    entity_comm: dict[str, str] = {}

    levels: List[HierarchyLevel] = []

    parts0 = detect_communities_partition(
        entity_nodes, entity_edges, min_community_size=min_community_size
    )
    l0_ids = [f"L0_{i}" for i in range(len(parts0))]
    for i, memb in enumerate(parts0):
        cid = l0_ids[i]
        for e in memb:
            entity_comm[e] = cid
    levels.append([(l0_ids[i], set(parts0[i])) for i in range(len(parts0))])

    if len(l0_ids) <= 1:
        return levels

    lvl = 1
    while lvl < max_levels:
        nodes_now = sorted(set(entity_comm.values()))
        if len(nodes_now) <= 1:
            break

        edge_pairs: set[tuple[str, str]] = set()
        for a, b in entity_edges:
            ca = entity_comm.get(a)
            cb = entity_comm.get(b)
            if ca is None or cb is None or ca == cb:
                continue
            edge_pairs.add((ca, cb) if ca < cb else (cb, ca))
        super_edges = list(edge_pairs)
        if not super_edges:
            break

        parts = detect_communities_partition(
            nodes_now, super_edges, min_community_size=1
        )
        if len(parts) <= 1:
            break
        if len(parts) >= len(nodes_now):
            break

        new_ids = [f"L{lvl}_{i}" for i in range(len(parts))]
        old_to_new: dict[str, str] = {}
        for i, cluster in enumerate(parts):
            nid = new_ids[i]
            for old_c in cluster:
                old_to_new[old_c] = nid

        for e in entity_nodes:
            if e in entity_comm:
                entity_comm[e] = old_to_new[entity_comm[e]]

        levels.append([(new_ids[i], set(parts[i])) for i in range(len(parts))])
        lvl += 1

    return levels


def clear_communities_and_reports() -> None:
    """Remove Community e CommunityReport para reindexação limpa (evita ids órfãos)."""
    driver = get_neo4j_graph()._driver
    with driver.session() as session:
        session.run("MATCH (r:CommunityReport) DETACH DELETE r")
        session.run("MATCH (c:Community) DETACH DELETE c")


def persist_hierarchical_communities_to_neo4j(
    levels: List[HierarchyLevel],
    *,
    clear_existing: bool = True,
) -> None:
    """
    Persiste comunidades com propriedade `level` e relações:
    - (Entity)-[:IN_COMMUNITY]->(Community) apenas no nível 0
    - (Community filha)-[:PART_OF]->(Community pai) para níveis > 0
    """
    driver = get_neo4j_graph()._driver
    if clear_existing:
        clear_communities_and_reports()

    with driver.session() as session:
        for level_idx, row in enumerate(levels):
            for comm_id, _ in row:
                session.run(
                    """
                    MERGE (c:Community {id: $id})
                    SET c.level = $level
                    """,
                    id=comm_id,
                    level=level_idx,
                )

        level0 = levels[0]
        for comm_id, entity_names in level0:
            for name in entity_names:
                session.run(
                    """
                    MATCH (e:Entity {name: $name}), (c:Community {id: $comm_id})
                    MERGE (e)-[:IN_COMMUNITY]->(c)
                    """,
                    name=name,
                    comm_id=comm_id,
                )

        for level_idx in range(1, len(levels)):
            for parent_id, child_ids in levels[level_idx]:
                for child_id in child_ids:
                    session.run(
                        """
                        MATCH (child:Community {id: $child_id}), (parent:Community {id: $parent_id})
                        MERGE (child)-[:PART_OF]->(parent)
                        """,
                        child_id=child_id,
                        parent_id=parent_id,
                    )


def run_communities(
    min_community_size: int | None = None,
    max_levels: int | None = None,
    clear_existing: bool = True,
) -> List[Set[str]]:
    """
    Deteta hierarquia, persiste no Neo4j.

    Retorno (compatível com código legado): apenas as comunidades de **nível 0** como conjuntos de entidades.
    """
    ms = min_community_size if min_community_size is not None else COMMUNITY_MIN_SIZE
    ml = max_levels if max_levels is not None else COMMUNITY_MAX_LEVELS

    nodes, edges = get_entity_graph_from_neo4j()
    if not nodes:
        return []

    levels = build_community_hierarchy(nodes, edges, min_community_size=ms, max_levels=ml)
    persist_hierarchical_communities_to_neo4j(levels, clear_existing=clear_existing)

    return [set(members) for _, members in levels[0]]


def run_communities_hierarchical(
    min_community_size: int | None = None,
    max_levels: int | None = None,
    clear_existing: bool = True,
) -> List[HierarchyLevel]:
    """Como `run_communities`, mas devolve todos os níveis `(id, membros)`."""
    ms = min_community_size if min_community_size is not None else COMMUNITY_MIN_SIZE
    ml = max_levels if max_levels is not None else COMMUNITY_MAX_LEVELS
    nodes, edges = get_entity_graph_from_neo4j()
    if not nodes:
        return []
    levels = build_community_hierarchy(nodes, edges, min_community_size=ms, max_levels=ml)
    persist_hierarchical_communities_to_neo4j(levels, clear_existing=clear_existing)
    return levels


def persist_communities_to_neo4j(communities: List[Set[str]]) -> None:
    """Legado: persiste só nível 0 (lista plana de comunidades de entidades)."""
    l0: HierarchyLevel = [(f"L0_{i}", set(s)) for i, s in enumerate(communities)]
    persist_hierarchical_communities_to_neo4j([l0], clear_existing=True)
