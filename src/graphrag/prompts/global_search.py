"""Prompts for GraphRAG-style global search (map → reduce over community reports)."""

from langchain_core.prompts import ChatPromptTemplate

# Map: extrai pontos com importância (equivalente ao map do Global Search do GraphRAG).
GLOBAL_MAP_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are helping answer a question about a document collection summarized as community reports. "
            "Read ONLY the report batch below. Extract factual points that could help answer the user question. "
            "Each point must be grounded in the reports. Assign an importance score from 0 to 100 "
            "(100 = critical for answering). Omit speculation. If the batch is irrelevant, return few or no points.",
        ),
        (
            "human",
            "User question:\n{question}\n\nCommunity report batch:\n{batch}\n\n"
            "Respond with structured output: a list of points with description and score.",
        ),
    ]
)

# Reduce: síntese final a partir dos pontos agregados (não dos relatórios brutos).
GLOBAL_REDUCE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer questions using ONLY the ranked key points provided. "
            "Synthesize a coherent answer. Do not invent facts. "
            "If the points are insufficient, say what is missing. "
            "Prefer precision over breadth.",
        ),
        (
            "human",
            "User question:\n{question}\n\nAggregated key points (from map stage):\n{points}\n\nAnswer:",
        ),
    ]
)
