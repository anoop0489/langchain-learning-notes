# ---------------------------------------------------------------------------
# router.py - Question routing chain (vectorstore vs web search)
# ---------------------------------------------------------------------------
# Uses structured output to classify the user's question and decide whether
# to route it to the local vectorstore (agents, prompt eng, adversarial
# attacks) or to Tavily web search (everything else).
#
# This is the ADAPTIVE RAG entry point - the graph doesn't always start
# at the same node because of this router.
# ---------------------------------------------------------------------------

import os
import sys

import truststore
truststore.inject_into_ssl()
sys.stdout.reconfigure(encoding="utf-8")

from typing import Literal

from dotenv import load_dotenv
load_dotenv()

os.environ["LANGSMITH_PROJECT"] = "agentic-rag"

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "websearch"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )


llm = ChatOpenAI(temperature=0)
structured_llm_router = llm.with_structured_output(RouteQuery)

system = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. For all else, use web-search."""

route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router
