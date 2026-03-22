"""Ligações pós-comunidade: claims e covariates → comunidades de nível 0 (via entidade âncora)."""

from graphrag.store.neo4j_graph import get_neo4j_graph


def link_claims_and_covariates_to_communities() -> dict[str, int]:
    """
    Cria (:Claim)-[:ANCHORED_IN_COMMUNITY]->(:Community) e o análogo para Covariate,
    quando o sujeito está em IN_COMMUNITY (nível 0). Alinha o modelo a relatórios/comunidades.
    """
    driver = get_neo4j_graph()._driver
    out = {"claim_links": 0, "covariate_links": 0}
    with driver.session() as session:
        r1 = session.run(
            """
            MATCH (e:Entity)-[:HAS_CLAIM]->(c:Claim)
            MATCH (e)-[:IN_COMMUNITY]->(comm:Community)
            WHERE coalesce(comm.level, 0) = 0
            WITH DISTINCT c, comm
            MERGE (c)-[:ANCHORED_IN_COMMUNITY]->(comm)
            RETURN count(*) AS n
            """
        ).single()
        out["claim_links"] = int(r1["n"] or 0) if r1 else 0

        r2 = session.run(
            """
            MATCH (e:Entity)-[:HAS_COVARIATE]->(v:Covariate)
            MATCH (e)-[:IN_COMMUNITY]->(comm:Community)
            WHERE coalesce(comm.level, 0) = 0
            WITH DISTINCT v, comm
            MERGE (v)-[:ANCHORED_IN_COMMUNITY]->(comm)
            RETURN count(*) AS n
            """
        ).single()
        out["covariate_links"] = int(r2["n"] or 0) if r2 else 0
    return out
