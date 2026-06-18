# =============================================================================
# EDEN'S ORIGINAL: Documentation Ingestion Pipeline (ingestion.py)
# =============================================================================
# This is Eden Marco's original ingestion script — kept as-is for reference.
# Compare with our adapted version at: ../ingestion.py
#
# PIPELINE FLOW:
#   1. TavilyCrawl crawls python.langchain.com (max_depth=2)
#   2. Raw content → LangChain Document objects (one per page)
#   3. RecursiveCharacterTextSplitter chunks docs (4000 chars, 200 overlap)
#   4. Async batch storage into vector store (500 docs per batch)
#
# KEY DIFFERENCES FROM OUR ADAPTED VERSION (../ingestion.py):
#   - Vector store: Uses Chroma (local) instead of Pinecone (cloud)
#     Eden has Pinecone commented out — he demoed both in the course
#   - SSL: Uses certifi package instead of truststore.inject_into_ssl()
#   - Imports: Uses langchain_classic.text_splitter (older package)
#     vs our langchain_text_splitters (current package)
#   - All three Tavily tools initialized (TavilyCrawl, TavilyExtract,
#     TavilyMap) — only TavilyCrawl is actually used in this script
#   - Logger: Imports from local logger.py (colored console output)
#   - No check_prerequisites() — assumes env vars are set
#   - No configurable constants (MAX_PAGES, BATCH_SIZE, etc.)
#
# NOTE: This file is NOT meant to be run — it's a reference copy.
#   To run ingestion, use: uv run ../ingestion.py
# =============================================================================

import asyncio
import os
import ssl
from typing import Any, Dict, List

import certifi
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

from logger import (Colors, log_error, log_header, log_info, log_success,
                    log_warning)

load_dotenv()

# Eden uses certifi for SSL — we use truststore.inject_into_ssl() instead
# because we're behind a corporate proxy that needs Windows cert store
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


# OpenAI embeddings — text-embedding-3-small produces 1536-dim vectors
# chunk_size=50 means send 50 texts per API call to OpenAI's embedding endpoint
# retry_min_seconds=10 waits at least 10s before retrying failed API calls
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    show_progress_bar=False,
    chunk_size=50,
    retry_min_seconds=10,
)
# Chroma: local vector DB (stores in chroma_db/ folder on disk)
# Unlike Pinecone (cloud), this doesn't need an API key or internet
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
# Pinecone alternative (commented out) — what we use in our adapted version
# vectorstore = PineconeVectorStore(
#     index_name="langchain-docs-2025", embedding=embeddings
# )

# All three Tavily tools initialized, but only tavily_crawl is used below
# TavilyExtract: pull content from specific URLs (like a focused scraper)
# TavilyMap: discover all URLs on a site (like a sitemap generator)
# TavilyCrawl: full recursive crawl (what we actually use for ingestion)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()


async def index_documents_async(documents: List[Document], batch_size: int = 50):
    """Process documents in batches asynchronously."""
    log_header("VECTOR STORAGE PHASE")
    log_info(
        f"📚 VectorStore Indexing: Preparing to add {len(documents)} documents to vector store",
        Colors.DARKCYAN,
    )

    # Create batches
    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]

    log_info(
        f"📦 VectorStore Indexing: Split into {len(batches)} batches of {batch_size} documents each"
    )

    # Process all batches concurrently
    async def add_batch(batch: List[Document], batch_num: int):
        try:
            await vectorstore.aadd_documents(batch)
            log_success(
                f"VectorStore Indexing: Successfully added batch {batch_num}/{len(batches)} ({len(batch)} documents)"
            )
        except Exception as e:
            log_error(f"VectorStore Indexing: Failed to add batch {batch_num} - {e}")
            return False
        return True

    # Process batches concurrently
    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Count successful batches
    successful = sum(1 for result in results if result is True)

    if successful == len(batches):
        log_success(
            f"VectorStore Indexing: All batches processed successfully! ({successful}/{len(batches)})"
        )
    else:
        log_warning(
            f"VectorStore Indexing: Processed {successful}/{len(batches)} batches successfully"
        )


async def main():
    """Main async function to orchestrate the entire process."""
    log_header("DOCUMENTATION INGESTION PIPELINE")

    log_info(
        "🗺️  TavilyCrawl: Starting to crawl the documentation site",
        Colors.PURPLE,
    )
    # Crawl the documentation site

    res = tavily_crawl.invoke(
        {
            "url": "https://python.langchain.com/",
            "max_depth": 2,
            "extract_depth": "advanced",
        }
    )

    # Convert Tavily crawl results to LangChain Document objects
    all_docs = []
    for tavily_crawl_result_item in res["results"]:
        log_info(
            f"TavilyCrawl: Successfully crawled {tavily_crawl_result_item['url']} from documentation site"
        )
        all_docs.append(
            Document(
                page_content=tavily_crawl_result_item["raw_content"],
                metadata={"source": tavily_crawl_result_item["url"]},
            )
        )

    # Split documents into chunks
    log_header("DOCUMENT CHUNKING PHASE")
    log_info(
        f"✂️  Text Splitter: Processing {len(all_docs)} documents with 4000 chunk size and 200 overlap",
        Colors.YELLOW,
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(
        f"Text Splitter: Created {len(splitted_docs)} chunks from {len(all_docs)} documents"
    )

    # Process documents asynchronously
    await index_documents_async(splitted_docs, batch_size=500)

    log_header("PIPELINE COMPLETE")
    log_success("🎉 Documentation ingestion pipeline finished successfully!")
    log_info("📊 Summary:", Colors.BOLD)
    log_info(f"   • Documents extracted: {len(all_docs)}")
    log_info(f"   • Chunks created: {len(splitted_docs)}")


if __name__ == "__main__":
    asyncio.run(main())
