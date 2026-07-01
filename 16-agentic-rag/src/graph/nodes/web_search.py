# ---------------------------------------------------------------------------
# web_search.py - Tavily web search fallback node
# ---------------------------------------------------------------------------
# Called when: (1) router decides question is off-topic for vectorstore, or
# (2) grade_documents finds irrelevant docs, or (3) answer grader says
# generation doesn't address the question. Appends web results to documents.
# ---------------------------------------------------------------------------

import os
import sys

import truststore
truststore.inject_into_ssl()
sys.stdout.reconfigure(encoding="utf-8")

from typing import Any, Dict

from dotenv import load_dotenv
load_dotenv()

os.environ["LANGSMITH_PROJECT"] = "agentic-rag"

from langchain.schema import Document
from langchain_tavily import TavilySearch

from graph.state import GraphState

web_search_tool = TavilySearch(max_results=3)


def web_search(state: GraphState) -> Dict[str, Any]:
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state.get("documents") or []

    tavily_results = web_search_tool.invoke({"query": question})["results"]
    joined_tavily_result = "\n".join(
        [tavily_result["content"] for tavily_result in tavily_results]
    )
    web_results = Document(page_content=joined_tavily_result)
    documents.append(web_results)
    return {"documents": documents, "question": question}
