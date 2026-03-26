"""LangGraph nodes: router, decompose, local_retrieve, graph_qa, synthesize, global_stub."""

from graphrag.state import GraphRAGState
from graphrag.chains.router import route_chain
from graphrag.chains.decompose import decompose_chain
from graphrag.retrieval.local_search import build_local_search_context
from graphrag.retrieval.global_search import fetch_global_community_reports, global_search_map_reduce
from graphrag.chains.graph_qa import get_graph_qa_chain
from graphrag.prompts.cypher import create_cypher_prompt, create_cypher_prompt_with_context
from graphrag.prompts.synthesis import SYNTHESIS_PROMPT
from graphrag.config import OPENAI_API_KEY, LLM_MODEL, LOCAL_SYNTH_CONTEXT_DOC_CAP
from graphrag.monitoring.token_cost import tracked_chat_openai


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
    """GraphRAG-style local search: match entities, fan-out to text units, rels, neighbors, community reports, + vector text."""
    context_docs, seed_entities = build_local_search_context(state)
    print(f"Local retrieve: {len(seed_entities)} seed entities, {len(context_docs)} context chunks")
    return {"context_docs": context_docs, "seed_entities": seed_entities}


def graph_qa_node(state: GraphRAGState) -> dict:
    """Run Cypher QA; fill cypher_result."""
    subqueries = state.get("subqueries") or []
    query = subqueries[1].sub_query if len(subqueries) > 1 else state["question"]
    use_ctx = bool(state.get("context_docs") or state.get("seed_entities"))
    cypher_prompt = create_cypher_prompt_with_context(state) if use_ctx else create_cypher_prompt()
    chain = get_graph_qa_chain(cypher_prompt=cypher_prompt)
    result = chain.invoke({"query": query})
    return {"cypher_result": result}


def synthesize_node(state: GraphRAGState) -> dict:
    """Combine context_docs + cypher_result into final_answer."""
    parts = []
    for doc in (state.get("context_docs") or [])[:LOCAL_SYNTH_CONTEXT_DOC_CAP]:
        content = doc.get("page_content", doc) if isinstance(doc, dict) else str(doc)
        parts.append(content)
    cypher_result = state.get("cypher_result")
    if cypher_result:
        parts.append(str(cypher_result))
    context = "\n\n".join(parts) if parts else "No relevant context found."
    llm = tracked_chat_openai(model=LLM_MODEL, temperature=0, api_key=OPENAI_API_KEY)
    chain = SYNTHESIS_PROMPT | llm
    answer = chain.invoke({"context": context, "question": state["question"]})
    if hasattr(answer, "content"):
        answer = answer.content
    return {"final_answer": answer}


def global_stub_node(state: GraphRAGState) -> dict:
    """Fallback when no community reports index: return message."""
    return {"final_answer": "Busca global ainda não implementada. Indexe documentos e execute o pipeline de indexação."}


def global_retrieve_node(state: GraphRAGState) -> dict:
    """Recupera pool amplo de Community Reports (vetor) e embaralha para o map-reduce global."""
    reports = fetch_global_community_reports(state["question"])
    print(f"Global retrieve: {len(reports)} community reports (pooled + shuffled)")
    return {"community_reports": reports}


def global_synthesize_node(state: GraphRAGState) -> dict:
    """Global search estilo GraphRAG: map (pontos pontuados por lote) → reduce (resposta final)."""
    reports = state.get("community_reports") or []
    answer = global_search_map_reduce(state["question"], reports)
    return {"final_answer": answer}
