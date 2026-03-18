"""LangGraph StateGraph: route -> local (decompose -> retrieve -> graph_qa -> synthesize) or global (stub)."""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from graphrag.state import GraphRAGState
from graphrag.graph.nodes import (
    router_node,
    decompose_node,
    local_retrieve_node,
    graph_qa_node,
    synthesize_node,
    global_retrieve_node,
    global_synthesize_node,
)


def _route_decision(state: GraphRAGState) -> str:
    """Return next node name from router result."""
    return "local" if state.get("search_type") == "local" else "global"


def get_compiled_graph(checkpointer=None):
    """Build and compile the query StateGraph."""
    graph = StateGraph(GraphRAGState)

    graph.add_node("router", router_node)
    graph.add_node("decompose", decompose_node)
    graph.add_node("local_retrieve", local_retrieve_node)
    graph.add_node("graph_qa", graph_qa_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("global_retrieve", global_retrieve_node)
    graph.add_node("global_synthesize", global_synthesize_node)

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        _route_decision,
        {"local": "decompose", "global": "global_retrieve"},
    )
    graph.add_edge("decompose", "local_retrieve")
    graph.add_edge("local_retrieve", "graph_qa")
    graph.add_edge("graph_qa", "synthesize")
    graph.add_edge("synthesize", END)
    graph.add_edge("global_retrieve", "global_synthesize")
    graph.add_edge("global_synthesize", END)

    return graph.compile(checkpointer=checkpointer or MemorySaver())


def run_query(question: str) -> str:
    """Run the query graph and return the final answer."""
    compiled = get_compiled_graph()
    config = {"configurable": {"thread_id": "default"}}
    initial: GraphRAGState = {"question": question}
    result = compiled.invoke(initial, config=config)
    return result.get("final_answer", "")
