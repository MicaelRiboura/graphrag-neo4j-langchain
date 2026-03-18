"""Router chain: classify question as local vs global search."""

from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

from graphrag.config import OPENAI_API_KEY, LLM_MODEL


class RouteDecision(BaseModel):
    """Output of the router: local or global."""

    search_type: Literal["local", "global"] = Field(
        description="Use 'local' for questions about specific entities or facts. "
        "Use 'global' for questions about overall themes, trends, or high-level summaries.",
    )


_llm = ChatOpenAI(
    model=LLM_MODEL,
    temperature=0,
    api_key=OPENAI_API_KEY,
)
_structured_llm = _llm.with_structured_output(RouteDecision)

ROUTER_SYSTEM = """You are a router for a GraphRAG system. Classify the user question into one of:
- local: questions about specific entities, people, events, or concrete facts (e.g. "What did X do?", "Who is Y?")
- global: questions about overall themes, main topics, trends, or dataset-wide summaries (e.g. "What are the main themes?", "Summarize the key points")"""

_route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", ROUTER_SYSTEM),
        ("human", "{question}"),
    ]
)

route_chain = _route_prompt | _structured_llm
