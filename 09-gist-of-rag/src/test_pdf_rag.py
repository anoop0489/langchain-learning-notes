# =============================================================================
# RAG TEST: PDF Ingestion + Retrieval Demo
# =============================================================================
# This is a self-contained test program that demonstrates the full RAG pipeline
# with a PDF document. It ingests a PDF, queries it, and shows the difference
# between asking with and without RAG.
#
# PREREQUISITES (do these ONCE before running):
#   1. Pinecone account: https://pinecone.io (free tier)
#   2. Create an index: dimensions=1536, metric=cosine, type=dense, serverless
#   3. .env file with: OPENAI_API_KEY, PINECONE_API_KEY, INDEX_NAME
#   4. Packages: uv add langchain-pinecone langchain-community langchain-text-splitters pypdf
#
# USAGE:
#   Place your PDF file in the same folder as this script (or update PDF_PATH below).
#   python test_pdf_rag.py
#
# WHAT IT DOES:
#   Phase 1 (Ingestion): PDF → Load → Chunk → Embed → Store in Pinecone
#   Phase 2 (Retrieval): Query → Similarity Search → Augment Prompt → LLM Answer
# =============================================================================

import os
import sys

from dotenv import load_dotenv

load_dotenv()

# ========================= CONFIGURATION =========================
# Change this to your PDF file path
PDF_PATH = "sample.pdf"

# The query to ask about the PDF content
QUERY = "What is the main topic of this document? Provide a summary."

# Number of relevant chunks to retrieve
TOP_K = 3
# =================================================================


def check_prerequisites():
    """Verify all required env vars and files exist before proceeding."""
    errors = []

    if not os.environ.get("OPENAI_API_KEY"):
        errors.append("OPENAI_API_KEY not found in .env")
    if not os.environ.get("PINECONE_API_KEY"):
        errors.append("PINECONE_API_KEY not found in .env")
    if not os.environ.get("INDEX_NAME"):
        errors.append("INDEX_NAME not found in .env")
    if not os.path.exists(PDF_PATH):
        errors.append(f"PDF file not found: {PDF_PATH}")

    if errors:
        print("❌ Prerequisites check FAILED:")
        for e in errors:
            print(f"   - {e}")
        print("\nSetup steps:")
        print("  1. Create a Pinecone account at https://pinecone.io (free tier)")
        print("  2. Create an index: dimensions=1536, metric=cosine")
        print("  3. Add OPENAI_API_KEY, PINECONE_API_KEY, INDEX_NAME to your .env file")
        print(f"  4. Place your PDF file at: {PDF_PATH}")
        sys.exit(1)

    print("✅ All prerequisites met!")
    print(f"   OpenAI key: {os.environ['OPENAI_API_KEY'][:8]}...")
    print(f"   Pinecone key: {os.environ['PINECONE_API_KEY'][:8]}...")
    print(f"   Index: {os.environ['INDEX_NAME']}")
    print(f"   PDF: {PDF_PATH}")
    print()


def run_ingestion():
    """Phase 1: Load PDF → Chunk → Embed → Store in Pinecone."""
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_openai import OpenAIEmbeddings
    from langchain_pinecone import PineconeVectorStore
    from langchain_text_splitters import CharacterTextSplitter

    # Step 1: Load PDF
    # PyPDFLoader creates one Document per page (unlike TextLoader which is one Document total)
    # Each Document's metadata includes: {"source": "file.pdf", "page": 0}
    print("📄 Loading PDF...")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    print(f"   Loaded {len(documents)} pages")

    # Step 2: Split into chunks
    # Even though PDF gives us pages, pages can be very long.
    # Chunking ensures each piece is small enough for effective retrieval.
    print("✂️  Splitting into chunks...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    print(f"   Created {len(chunks)} chunks")

    # Show a sample chunk so you can see what the data looks like
    if chunks:
        sample = chunks[0]
        print(f"\n   📋 Sample chunk (first 200 chars):")
        print(f"   \"{sample.page_content[:200]}...\"")
        print(f"   Metadata: {sample.metadata}")
        print()

    # Step 3: Embed and store
    # IMPORTANT: The embedding model MUST match what you used when creating the Pinecone index.
    # OpenAIEmbeddings defaults to text-embedding-ada-002 → 1536 dimensions.
    print("🧮 Embedding and storing in Pinecone...")
    embeddings = OpenAIEmbeddings()
    PineconeVectorStore.from_documents(
        chunks, embeddings, index_name=os.environ["INDEX_NAME"]
    )
    print(f"   ✅ Stored {len(chunks)} vectors in Pinecone!\n")


def run_retrieval(query: str):
    """Phase 2: Query → Retrieve → Augment → Generate."""
    from langchain_core.messages import HumanMessage
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_pinecone import PineconeVectorStore

    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI()

    vectorstore = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"], embedding=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    prompt_template = ChatPromptTemplate.from_template(
        """Answer the question based only on the following context:

{context}

Question: {question}

Provide a detailed answer:"""
    )

    # ---- Without RAG (baseline) ----
    print("=" * 70)
    print("🤖 WITHOUT RAG (raw LLM — no context)")
    print("=" * 70)
    raw_response = llm.invoke([HumanMessage(content=query)])
    print(raw_response.content)
    print()

    # ---- With RAG ----
    print("=" * 70)
    print("📚 WITH RAG (grounded in your PDF)")
    print("=" * 70)

    # Retrieve relevant chunks
    docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)

    # Show what was retrieved
    print(f"\n🔍 Retrieved {len(docs)} relevant chunks:")
    for i, doc in enumerate(docs):
        source_info = f"page {doc.metadata.get('page', '?')}" if 'page' in doc.metadata else doc.metadata.get('source', '?')
        print(f"   Chunk {i+1} ({source_info}): \"{doc.page_content[:80]}...\"")
    print()

    # Generate answer
    messages = prompt_template.format_messages(context=context, question=query)
    response = llm.invoke(messages)
    print("💡 Answer:")
    print(response.content)


if __name__ == "__main__":
    check_prerequisites()

    print("=" * 70)
    print("PHASE 1: INGESTION (PDF → Pinecone)")
    print("=" * 70)
    run_ingestion()

    print("=" * 70)
    print("PHASE 2: RETRIEVAL (Query → RAG Answer)")
    print("=" * 70)
    run_retrieval(QUERY)

    print("\n✅ Done! You can modify QUERY at the top of this file to ask different questions.")
