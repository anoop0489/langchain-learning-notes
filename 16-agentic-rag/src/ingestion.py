# ---------------------------------------------------------------------------
# ingestion.py - Agentic RAG Document Ingestion Pipeline
# ---------------------------------------------------------------------------
# Loads web pages (Lilian Weng's blog posts on Agents, Prompt Engineering,
# and Adversarial Attacks on LLMs), splits them into chunks, and stores them
# in a ChromaDB vector store with OpenAI embeddings.
#
# Run ONCE to create the vector store:
#     cd 16-agentic-rag/src
#     uv run python ingestion.py
#
# After ingestion, the retriever is importable from this module for use by
# the graph's retrieve node.
# ---------------------------------------------------------------------------

import os
import sys

import truststore
truststore.inject_into_ssl()
sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()

os.environ["LANGSMITH_PROJECT"] = "agentic-rag"

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings

CHROMA_PERSIST_DIR = "./.chroma"
COLLECTION_NAME = "rag-chroma"

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]


def check_prerequisites():
    required = ["OPENAI_API_KEY"]
    missing = [var for var in required if not os.environ.get(var)]
    if missing:
        print(f"Missing environment variables: {', '.join(missing)}")
        print("   Add them to your .env file")
        sys.exit(1)


def ingest_documents():
    print("Loading documents from web...")
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    print(f"   Loaded {len(docs_list)} documents")

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)
    print(f"   Split into {len(doc_splits)} chunks")

    print("Embedding and storing in ChromaDB...")
    Chroma.from_documents(
        documents=doc_splits,
        collection_name=COLLECTION_NAME,
        embedding=OpenAIEmbeddings(),
        persist_directory=CHROMA_PERSIST_DIR,
    )
    print("Ingestion complete!")


retriever = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=CHROMA_PERSIST_DIR,
    embedding_function=OpenAIEmbeddings(),
).as_retriever()


if __name__ == "__main__":
    check_prerequisites()
    ingest_documents()
