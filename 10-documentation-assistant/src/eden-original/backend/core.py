# =============================================================================
# EDEN'S ORIGINAL: Agentic RAG Backend (backend/core.py)
# =============================================================================
# This is Eden Marco's original backend — kept as-is for reference.
# Compare with our adapted version at: ../../backend/core.py
#
# WHAT THIS DOES:
#   Creates an agent that decides WHEN to search the documentation vector
#   store. Uses LangChain's create_agent() with a retrieval tool.
#
# KEY PATTERNS:
#   - create_agent(): Modern LangChain factory for tool-calling agents
#   - @tool(response_format="content_and_artifact"): Returns two values:
#       1. Serialized text → ToolMessage.content (what the LLM reads)
#       2. Raw Document list → ToolMessage.artifact (what the app uses)
#   - init_chat_model(): Provider-agnostic model init (swap provider easily)
#   - ToolMessage artifact extraction: Loop through messages to find
#     retrieved docs for source citations in the UI
#
# DIFFERENCES FROM OUR ADAPTED VERSION (../../backend/core.py):
#   - Model: Uses gpt-5.2 (Eden's course recording) vs our gpt-4o
#   - Index: "langchain-docs-2026" vs our "doc-helper-index"
#   - No truststore/SSL handling (Eden handles SSL in ingestion.py only)
#   - No sys.path manipulation
#   - Minimal comments
#
# NOTE: This file is NOT meant to be run — it's a reference copy.
# =============================================================================

from typing import Any, Dict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# Same embedding model as ingestion — MUST match or vectors won't align
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Connect to Pinecone index populated by ingestion.py
# Eden used "langchain-docs-2026" — we use "doc-helper-index"
vectorstore = PineconeVectorStore(
    index_name="langchain-docs-2026", embedding=embeddings
)
# init_chat_model() is provider-agnostic — change model_provider to
# "anthropic", "google_genai", etc. without touching other code
model = init_chat_model("gpt-5.2", model_provider="openai")


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve relevant documentation to help answer user queries about LangChain."""
    # Retrieve top 4 most similar documents
    retrieved_docs = vectorstore.as_retriever().invoke(query, k=4)

    # Serialize documents for the model
    serialized = "\n\n".join(
        (f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )

    # Return both serialized content and raw documents
    return serialized, retrieved_docs


def run_llm(query: str) -> Dict[str, Any]:
    """
    Run the RAG pipeline to answer a query using retrieved documentation.

    Args:
        query: The user's question

    Returns:
        Dictionary containing:
            - answer: The generated answer
            - context: List of retrieved documents
    """
    # Create the agent with retrieval tool
    system_prompt = (
        "You are a helpful AI assistant that answers questions about LangChain documentation. "
        "You have access to a tool that retrieves relevant documentation. "
        "Use the tool to find relevant information before answering questions. "
        "Always cite the sources you use in your answers. "
        "If you cannot find the answer in the retrieved documentation, say so."
    )

    agent = create_agent(model, tools=[retrieve_context], system_prompt=system_prompt)

    # Build messages list
    messages = [{"role": "user", "content": query}]

    # Invoke the agent
    response = agent.invoke({"messages": messages})

    # Extract the answer from the last AI message
    answer = response["messages"][-1].content

    # Extract context documents from ToolMessage artifacts
    context_docs = []
    for message in response["messages"]:
        # Check if this is a ToolMessage with artifact
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            # The artifact should contain the list of Document objects
            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)

    return {
        "answer": answer,
        "context": context_docs
    }

if __name__ == '__main__':
    result = run_llm(query="what are deep agents?")
    print(result)
