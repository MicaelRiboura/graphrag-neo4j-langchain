"""Generate Community Reports (LLM summaries) and persist as CommunityReport nodes."""

from typing import List, Set

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from graphrag.config import OPENAI_API_KEY, LLM_MODEL
from graphrag.store.neo4j_graph import get_neo4j_graph


REPORT_SYSTEM = """You are an expert at summarizing knowledge. Given a set of entity names and their relationships from a community in a knowledge graph, write a short structured report (2-5 paragraphs) that captures the main facts and themes. Write in clear, neutral language."""

_report_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", REPORT_SYSTEM),
        ("human", "Entities and relationships:\n{context}"),
    ]
)


def _get_community_context(community_id: str) -> str:
    """Fetch entity names and relationship descriptions for a community from Neo4j."""
    driver = get_neo4j_graph()._driver
    with driver.session() as session:
        result = session.run(
            """
            MATCH (c:Community {id: $id})<-[:IN_COMMUNITY]-(e:Entity)
            OPTIONAL MATCH (e)-[r:RELATES_TO]->(e2:Entity)-[:IN_COMMUNITY]->(c)
            WITH e, collect(DISTINCT r.description) AS rels
            RETURN e.name AS name, e.description AS desc, rels
            """,
            id=community_id,
        )
        lines = []
        for r in result:
            name = r["name"]
            desc = r["desc"] or ""
            rels = [x for x in (r["rels"] or []) if x]
            lines.append(f"- {name}: {desc}" + (" Relationships: " + "; ".join(rels) if rels else ""))
        return "\n".join(lines) if lines else "No data"


def generate_report_for_community(community_id: str) -> str:
    """Generate one Community Report text for a community."""
    context = _get_community_context(community_id)
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0, api_key=OPENAI_API_KEY)
    chain = _report_prompt | llm
    out = chain.invoke({"context": context})
    return out.content if hasattr(out, "content") else str(out)


def persist_report_to_neo4j(community_id: str, content: str) -> None:
    """Create CommunityReport node and link to Community."""
    driver = get_neo4j_graph()._driver
    with driver.session() as session:
        session.run(
            """
            MATCH (c:Community {id: $comm_id})
            MERGE (r:CommunityReport {community_id: $comm_id})
            SET r.content = $content
            MERGE (c)-[:HAS_REPORT]->(r)
            """,
            comm_id=community_id,
            content=content,
        )


def run_reports(community_ids: List[str] | None = None) -> None:
    """Generate and persist reports for all communities (or given IDs)."""
    if community_ids is None:
        driver = get_neo4j_graph()._driver
        with driver.session() as session:
            result = session.run("MATCH (c:Community) RETURN c.id AS id")
            community_ids = [r["id"] for r in result]
    for cid in community_ids:
        content = generate_report_for_community(cid)
        persist_report_to_neo4j(cid, content)
