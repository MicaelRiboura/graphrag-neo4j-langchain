"""Cypher generation prompts for GraphCypherQAChain."""

from langchain_core.prompts import PromptTemplate

from graphrag.config import CYPHER_PROMPT_CONTEXT_DOCS


CYPHER_GENERATION_TEMPLATE = """You are an expert Neo4j Cypher writer. Given the schema and the question, write a Cypher query to answer the question.

Schema:
{schema}

Question: {query}

Only return the Cypher query, no explanation."""


def create_cypher_prompt() -> PromptTemplate:
    """Return a prompt template for Cypher generation (schema + question)."""
    return PromptTemplate(
        input_variables=["schema", "query"],
        template=CYPHER_GENERATION_TEMPLATE,
    )


def create_cypher_prompt_with_context(state: dict) -> PromptTemplate:
    """Return a Cypher prompt with local retrieval context (docs + optional seed entities)."""
    context = state.get("context_docs") or []
    seeds = state.get("seed_entities") or []
    if not context and not seeds:
        return create_cypher_prompt()
    blocks = []
    if seeds:
        blocks.append(
            "Prioritize these entity names when filtering or matching nodes: "
            + ", ".join(str(s) for s in seeds[:40])
        )
    n = max(CYPHER_PROMPT_CONTEXT_DOCS, 1)
    for d in context[:n]:
        blocks.append(str(d.get("page_content", d) if isinstance(d, dict) else d))
    context_str = "\n\n".join(blocks)
    template = """You are an expert Neo4j Cypher writer. Use the following context about the data when relevant.

Context from local GraphRAG retrieval:
{context}

Schema:
{schema}

Question: {query}

Only return the Cypher query, no explanation."""
    return PromptTemplate(
        input_variables=["schema", "query"],
        template=template,
        partial_variables={"context": context_str},
    )
