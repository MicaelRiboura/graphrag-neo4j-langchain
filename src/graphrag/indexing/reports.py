"""Generate Community Reports (LLM summaries) and persist as CommunityReport nodes."""

from typing import List, Set

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from graphrag.config import OPENAI_API_KEY, LLM_MODEL
from graphrag.store.neo4j_graph import get_neo4j_graph


REPORT_SYSTEM = """You are an expert at summarizing knowledge. 
Given a set of entity names and their relationships from a community in a knowledge graph, write a short structured report (2-5 paragraphs) 
that captures the main facts and themes. Write in clear, neutral language."""

REPORT_SYSTEM_OIL_GAS = """You are an expert Data Analyst in the Oil and Gas (O&G) sector. Given a set of entities and relationships representing a community from an O&G knowledge graph, write a highly structured, analytical report (3-5 paragraphs) that synthesizes the data to answer global search queries. 

This community contains data regarding States, Counties, Years, Commodities (Oil in bbl / Gas in Mcf), Disposition Types (e.g., Sales, Transferred, Flared, Adjustments), and Total Reported Volumes.

Your report must capture aggregations, trends, and major contributors. Do not just list facts; connect them logically. Structure your report using the following thematic guidelines:

1. **High-Level Overview:** Summarize the core focus of this community (e.g., "This community primarily details Gas production in Texas counties between 2020 and 2024").
2. **Key Contributors & Dominant Volumes:** Identify the top-producing states or counties, the most common disposition types (e.g., 'Sales-Royalty Due-MEASURED' vs. 'Transferred to Facility'), and highly prominent volume figures.
3. **Temporal Trends & Comparisons:** Highlight noticeable shifts across the years present in the data (e.g., massive increases or drops in volume, shifts in where commodities are being transferred).
4. **Anomalies & Specific Adjustments:** Explicitly mention any unique patterns, such as negative volumes (e.g., 'Buy-Back', 'Differences/Adjustments'), zero volumes, or environmental indicators like 'Flared Gas'.

Write in clear, objective, and precise analytical language. Strictly ground your summary in the provided relationships. Do not infer data outside the provided community graph."""

_report_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", REPORT_SYSTEM_OIL_GAS),
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
