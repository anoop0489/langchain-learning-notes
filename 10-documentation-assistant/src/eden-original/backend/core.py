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
#   - Vector store: Configurable — toggle between Chroma and Pinecone
#     via VECTOR_STORE_TYPE env var or constant (default: chroma)
#   - Model: Uses gpt-4o (Eden's original used gpt-5.2)
#   - Has truststore.inject_into_ssl() for corporate proxy
#   - No sys.path manipulation
#   - Minimal comments
#
# CONFIG: Set VECTOR_STORE_TYPE in .env to switch:
#   VECTOR_STORE_TYPE=chroma    → local chroma_db/ folder (default)
#   VECTOR_STORE_TYPE=pinecone  → cloud Pinecone (needs PINECONE_API_KEY)
#
# RUN: cd 10-documentation-assistant/src/eden-original && uv run python backend/core.py
# =============================================================================

import os
import sys
from typing import Any, Dict

import truststore
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# ADAPTED: Use truststore for corporate proxy SSL
truststore.inject_into_ssl()

# =================================================================
# CONFIGURATION — Toggle vector store and model here or via .env
# =================================================================
# VECTOR_STORE_TYPE: "chroma" (local, no API key) or "pinecone" (cloud)
# Set in .env as VECTOR_STORE_TYPE=pinecone to override
VECTOR_STORE_TYPE = os.environ.get("VECTOR_STORE_TYPE", "chroma").lower()

# Pinecone index name (only used when VECTOR_STORE_TYPE=pinecone)
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "doc-helper-index")
# =================================================================

# Same embedding model as ingestion — MUST match or vectors won't align
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Vector store initialization — pick backend based on config
if VECTOR_STORE_TYPE == "pinecone":
    from langchain_pinecone import PineconeVectorStore
    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME, embedding=embeddings
    )
else:
    from langchain_chroma import Chroma
    # Resolve chroma_db/ path relative to this file's location
    # __file__ = .../eden-original/backend/core.py → up 2 = .../eden-original/
    _EDEN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _CHROMA_DB = os.path.join(_EDEN_DIR, "chroma_db")
    vectorstore = Chroma(persist_directory=_CHROMA_DB, embedding_function=embeddings)

# init_chat_model() is provider-agnostic — change model_provider to
# "anthropic", "google_genai", etc. without touching other code
# Eden's original used gpt-5.2 — we use gpt-4o (available in our env)
model = init_chat_model("gpt-4o", model_provider="openai")


# ========================= RETRIEVAL TOOL =========================
# @tool converts this function into a LangChain Tool object that the agent
# can call. The decorator reads the docstring to generate the tool description
# that the LLM sees when deciding which tools to use.
#
# response_format="content_and_artifact" means this function returns a TUPLE:
#   (content_for_llm, artifact_for_app)
# - content → becomes ToolMessage.content (the LLM reads this to generate answer)
# - artifact → becomes ToolMessage.artifact (the app uses this for source citations)
#
# WHY TWO RETURN VALUES?
# The LLM needs formatted text to reason over. The app needs raw Document objects
# for metadata (URLs, page numbers). Splitting them avoids polluting either.
@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve relevant documentation to help answer user queries about LangChain."""
    # .as_retriever() converts the vector store into a Retriever interface
    # .invoke(query, k=4) embeds the query and finds top-4 most similar chunks
    # C# equivalent: vectorStore.AsRetriever().Invoke(query, topK: 4)
    retrieved_docs = vectorstore.as_retriever().invoke(query, k=4)

    # Format documents as readable text for the LLM
    # Generator expression inside join() — like string.Join("\n\n", docs.Select(...))
    serialized = "\n\n".join(
        (f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )

    # Return TUPLE: (text for LLM to read, raw docs for app to show sources)
    # The framework splits this into ToolMessage.content and ToolMessage.artifact
    return serialized, retrieved_docs


def run_llm(query: str) -> Dict[str, Any]:
    """Run the agentic RAG pipeline to answer a query.

    FLOW:
      1. create_agent() builds an agent with our retrieval tool
      2. Agent receives user query as a message
      3. Agent decides: "Should I call retrieve_context?" (function calling)
      4. If yes → calls tool → reads results → generates answer with citations
      5. If no → answers from parametric knowledge (rare for doc questions)
      6. We extract both the answer text and source Documents from the response

    Args:
        query: The user's question about LangChain

    Returns:
        dict with:
        - "answer": str — the generated response
        - "context": list[Document] — retrieved docs for source citations
    """
    # System prompt: tells the agent its role and when to use the retrieval tool
    # This is injected as the first message in every conversation
    system_prompt = (
        "You are a helpful AI assistant that answers questions about LangChain documentation. "
        "You have access to a tool that retrieves relevant documentation. "
        "Use the tool to find relevant information before answering questions. "
        "Always cite the sources you use in your answers. "
        "If you cannot find the answer in the retrieved documentation, say so."
    )

    # create_agent() is LangChain's modern factory for tool-calling agents
    # It builds a graph: User Message → LLM → (Tool Call?) → Tool → LLM → Final Answer
    # C# equivalent: new AgentBuilder(model).AddTools(tools).SetSystemPrompt(prompt).Build()
    agent = create_agent(model, tools=[retrieve_context], system_prompt=system_prompt)

    # Messages use the OpenAI chat format: [{"role": "user", "content": "..."}]
    messages = [{"role": "user", "content": query}]

    # .invoke() runs the full agent loop until the LLM produces a final answer
    # The response contains ALL messages exchanged (user, AI, tool calls, tool results)
    response = agent.invoke({"messages": messages})

    # The last message in the chain is always the final AI answer
    # response["messages"][-1] = Python for "last element" (like .Last() in LINQ)
    answer = response["messages"][-1].content

    # Extract source documents from ToolMessage artifacts
    # The agent loop produces multiple message types:
    #   HumanMessage → AIMessage (with tool_calls) → ToolMessage → AIMessage (final)
    # We scan for ToolMessages that have .artifact (our raw Document list)
    context_docs = []
    for message in response["messages"]:
        # isinstance() = C#'s "is" keyword — checks if message is a ToolMessage
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            # artifact is the second value from our retrieve_context() return tuple
            if isinstance(message.artifact, list):
                # .extend() = .AddRange() in C# — adds all items from artifact list
                context_docs.extend(message.artifact)

    return {
        "answer": answer,
        "context": context_docs
    }


if __name__ == '__main__':
    # Quick test — run this file directly to verify the backend works
    # C# equivalent: static void Main() { var result = RunLlm("..."); Console.Write(result); }
    result = run_llm(query="what are deep agents?")
    print(result)
