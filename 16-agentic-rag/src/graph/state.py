from typing import List, TypedDict


class GraphState(TypedDict):
    """
    Typed state for the Agentic RAG graph.

    Attributes:
        question: The user's original question
        generation: The LLM-generated answer
        web_search: Flag indicating whether web search is needed
        documents: List of retrieved/searched documents
    """

    question: str
    generation: str
    web_search: bool
    documents: List[str]
