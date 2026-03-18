"""Prompt for final answer synthesis (question + context -> answer)."""

from langchain_core.prompts import ChatPromptTemplate

SYNTHESIS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that answers questions based only on the provided context. "
            "If the context does not contain enough information, say so. Do not invent facts.",
        ),
        ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"),
    ]
)
