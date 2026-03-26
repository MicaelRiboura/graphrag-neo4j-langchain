"""Graph Cypher QA chain over Neo4j."""

from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain

from graphrag.config import OPENAI_API_KEY, LLM_MODEL
from graphrag.monitoring.token_cost import tracked_chat_openai
from graphrag.store.neo4j_graph import get_neo4j_graph
from graphrag.prompts.cypher import create_cypher_prompt


def get_graph_qa_chain(cypher_prompt=None):
    """Return a GraphCypherQAChain. Optionally pass a custom cypher_prompt."""
    graph = get_neo4j_graph()
    llm = tracked_chat_openai(model=LLM_MODEL, temperature=0, api_key=OPENAI_API_KEY)
    prompt = cypher_prompt or create_cypher_prompt()
    return GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        cypher_prompt=prompt,
        validate_cypher=True,
        return_direct=True,
        verbose=False,
        allow_dangerous_requests=True
    )
