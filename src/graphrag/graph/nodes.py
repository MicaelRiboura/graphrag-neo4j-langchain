"""LangGraph nodes: router, decompose, local_retrieve, graph_qa, synthesize, global_stub."""

from graphrag.state import GraphRAGState
from graphrag.chains.router import route_chain
from graphrag.chains.decompose import decompose_chain
from graphrag.chains.retrieval import get_retrieval_chain
from graphrag.chains.graph_qa import get_graph_qa_chain
from graphrag.prompts.cypher import create_cypher_prompt, create_cypher_prompt_with_context
from graphrag.prompts.synthesis import SYNTHESIS_PROMPT
from graphrag.store.neo4j_graph import get_neo4j_graph
from langchain_openai import ChatOpenAI
from graphrag.config import OPENAI_API_KEY, LLM_MODEL


def router_node(state: GraphRAGState) -> dict:
    """Classify question as local or global."""
    decision = route_chain.invoke({"question": state["question"]})
    print(f"Decision: {decision.search_type}")
    return {"search_type": decision.search_type}


def decompose_node(state: GraphRAGState) -> dict:
    """Break question into subqueries."""
    result = decompose_chain.invoke({"question": state["question"]})
    print(f"Subqueries: {result.subqueries}")
    return {"subqueries": result.subqueries}


def local_retrieve_node(state: GraphRAGState) -> dict:
    """Vector search over TextUnits; fill context_docs."""
    chain_invoke = get_retrieval_chain()
    if chain_invoke is None:
        return {"context_docs": []}
    question = state["question"]
    subqueries = state.get("subqueries") or []
    query = subqueries[0].sub_query if subqueries else question
    out = chain_invoke({"query": query})
    docs = out.get("source_documents") or []
    context_docs = [{"page_content": d.page_content, "metadata": getattr(d, "metadata", {})} for d in docs]
    return {"context_docs": context_docs}


def graph_qa_node(state: GraphRAGState) -> dict:
    """Run Cypher QA; fill cypher_result."""
    graph = get_neo4j_graph()
    question = state["question"]
    subqueries = state.get("subqueries") or []
    query = subqueries[1].sub_query if len(subqueries) > 1 else question
    cypher_prompt = create_cypher_prompt_with_context(state) if state.get("context_docs") else create_cypher_prompt()
    chain = get_graph_qa_chain(cypher_prompt=cypher_prompt)
    result = chain.invoke({"query": query})
    return {"cypher_result": result}


def synthesize_node(state: GraphRAGState) -> dict:
    """Combine context_docs + cypher_result into final_answer."""
    parts = []
    for doc in (state.get("context_docs") or [])[:5]:
        content = doc.get("page_content", doc) if isinstance(doc, dict) else str(doc)
        parts.append(content)
    cypher_result = state.get("cypher_result")
    if cypher_result:
        parts.append(str(cypher_result))
    context = "\n\n".join(parts) if parts else "No relevant context found."
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0, api_key=OPENAI_API_KEY)
    chain = SYNTHESIS_PROMPT | llm
    answer = chain.invoke({"context": context, "question": state["question"]})
    if hasattr(answer, "content"):
        answer = answer.content
    return {"final_answer": answer}


def global_stub_node(state: GraphRAGState) -> dict:
    """Fallback when no community reports index: return message."""
    return {"final_answer": "Busca global ainda não implementada. Indexe documentos e execute o pipeline de indexação."}


def global_retrieve_node(state: GraphRAGState) -> dict:
    """Retrieve top-k Community Reports by similarity to the question."""
    from graphrag.store.vector_index import get_vector_index_reports
    from graphrag.config import GLOBAL_REPORTS_TOP_K

    store = get_vector_index_reports()
    if store is None:
        return {"community_reports": []}
    docs = store.similarity_search(state["question"], k=GLOBAL_REPORTS_TOP_K)
    reports = [d.page_content for d in docs]
    return {"community_reports": reports}


def global_synthesize_node(state: GraphRAGState) -> dict:
    """Synthesize final answer from community reports (global search)."""
    reports = state.get("community_reports") or []
    if not reports:
        return {"final_answer": "Nenhum relatório de comunidade encontrado. Tente indexar documentos primeiro."}
    context = "\n\n---\n\n".join(reports)
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0, api_key=OPENAI_API_KEY)
    chain = SYNTHESIS_PROMPT | llm
    answer = chain.invoke({"context": context, "question": state["question"]})
    if hasattr(answer, "content"):
        answer = answer.content
    return {"final_answer": answer}
