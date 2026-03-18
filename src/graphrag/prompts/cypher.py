"""Cypher generation prompts for GraphCypherQAChain."""

from langchain_core.prompts import PromptTemplate


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
    """Return a Cypher prompt that includes optional context from retrieved docs."""
    context = state.get("context_docs") or []
    if not context:
        return create_cypher_prompt()
    context_str = "\n".join(
        str(d.get("page_content", d) if isinstance(d, dict) else d) for d in context[:5]
    )
    template = """You are an expert Neo4j Cypher writer. Use the following context about the data when relevant.

Context from retrieved documents:
{context}

Schema:
{schema}

Question: {query}

Only return the Cypher query, no explanation."""
    return PromptTemplate(
        input_variables=["schema", "query", "context"],
        template=template,
        partial_variables={"context": context_str},
    )
