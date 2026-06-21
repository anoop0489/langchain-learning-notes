# Copilot Instructions — LangChain Learning Notes

## What This Repo Is

A personal study/interview-prep repository for mastering LangChain, LangGraph, and Agentic AI. The user is a **Senior .NET/C# engineer** learning Python/LangChain for AI engineering roles. All docs and code serve as revision material.

---

## Repository Structure

Each course section gets its own folder with numbered `.md` files and a `src/` subfolder for runnable Python scripts:
XX-section-name/
├── NN_Theory_And_Concepts.md      # Theory, definitions, interview Q&A
├── MM_Implementation.md           # Implementation walkthrough, code links
├── assets/                        # Diagrams, images
└── src/
	├── main.py                    # Core example from the course
	├── ingestion.py               # If applicable (RAG sections)
	└── test_*.py                  # Extended examples (multimodal, conversational, etc.)
- Root `README.md` is the course roadmap with links to every section.
- `reference-guides/` holds cross-cutting references (e.g., Python-to-C# glossary).
- `01-introduction/00_Environment_Setup.md` is the setup guide (uv, .env, Pinecone, Visual Studio).

---

## Documentation Conventions

### Theory/Concepts Docs (`*_Theory_And_Concepts.md`)

1. **Table of Contents** at the top — numbered table with anchor links and "What You'll Learn" column.
2. **Key Definitions table** — `| Term | Quick Recall | Full Definition |` format. The "Quick Recall" column is the one-liner to say first in an interview.
3. **Deep Dive sections** — each major topic gets `## Deep Dive: <Topic>`. Use tables, code blocks, and analogies.
4. **C# Analogies** — wherever possible, map Python/LangChain concepts to C#/Java equivalents (e.g., `RunnableLambda` ↔ `Func<T>`, `RecordManager` ↔ EF Core Migrations, `.astream()` ↔ `IAsyncEnumerable`).
5. **Interview Q&A Anchors** — at the bottom, `## Interview Q&A Anchors` with `**Q:**` / `> **A:**` format. Each answer is 2-4 sentences, production-grade.
6. **References** — at the very end, bullet list of relevant links.
7. **Keep as one file** — don't split into subpages. Use the TOC for navigation. Ctrl+F across one file is faster for revision.

### Implementation Docs (`*_Implementation.md`)

1. **Project structure tree** at the top showing the folder layout.
2. **Dependencies** section with `uv add` command.
3. **Environment variables** section with `.env` template.
4. **Step-by-step walkthrough** with code blocks and explanations.
5. **"Beyond Basic"** section at the bottom linking to extended examples with relative file links (clickable in VS/GitHub).

### Python Scripts (`src/*.py`)

1. **Header docblock** — large comment block explaining: what the script does, the problem it solves, prerequisites, and `uv run` command.
2. **`truststore.inject_into_ssl()`** — always include at the top of every script. The user is behind a corporate proxy and needs Windows certificate store injection.
3. **`check_prerequisites()`** — validate env vars before doing anything.
4. **`if __name__ == "__main__":`** — always use a main guard.
5. **Naming**: `test_<capability>.py` for extended examples (e.g., `test_multimodal_pdf_rag.py`, `test_conversational_rag.py`, `test_streaming_rag.py`).

---

## Code Style

- **Python 3.12** with `uv` for dependency management and execution.
- **`dotenv`** — always `from dotenv import load_dotenv; load_dotenv()`.
- **LangChain imports** — use the specific packages (`langchain-openai`, `langchain-pinecone`, etc.), not the monolithic `langchain` package.
- **No unnecessary comments** — code should be self-documenting. Comments only for "why", not "what".
- **Print output** — use emoji prefixes and `=`/`-` separators for readable CLI output in demo scripts.

---

## Key Technical Decisions (Already Made)

| Decision | Choice | Why |
|----------|--------|-----|
| Vector database | Pinecone (serverless, cosine, 1536 dims) | Course default, free tier sufficient |
| Embedding model | OpenAI `text-embedding-ada-002` (LangChain default) | 1536 dimensions, widely used |
| LLM (Cloud) | GPT-4o via `ChatOpenAI` | Best quality for multimodal and generation |
| LLM (Local) | Ollama (`qwen3:1.7b` or similar) via `ollama` package | Free, no API costs, tool-calling support |
| PDF loader | Multimodal Vision (PyMuPDF + GPT-4o) as default | Production PDFs have diagrams |
| PDF libraries | `pymupdf` (primary, Section 9), `pypdf` (fallback), `pdf2image` (image extraction) | Full coverage of PDF processing needs |
| RAG architecture | Deterministic (not agentic) | Lower cost, predictable, production-safe |
| Indexing strategy | `SQLRecordManager` + `cleanup="incremental"` | Prevents duplicates, saves embedding costs |
| Conversational RAG | Question reformulation with conservative prompt | Handles follow-ups without over-connecting new topics |
| Agent search tool | Tavily via `langchain-tavily` (Section 10) | LangChain integration for web crawling/search |
| HTTP client | `httpx` (Section 4-7, async SSL workaround) | Used for custom SSL context in agent scripts |
| SSL | `truststore.inject_into_ssl()` | Corporate proxy requires Windows cert store |
| Streaming | `.stream()` / `.astream()` on same LCEL chain | No pipeline changes needed |
| Observability | LangSmith (`langsmith` package) | Tracing, debugging, evaluation |
| UI (prototyping) | Streamlit (`streamlit`, Section 10) | Fast chatbot/demo UIs without frontend code |

---

## Workflow Preferences

1. **Don't split large docs** — keep theory in one file with a TOC.
2. **Runnable examples over doc-embedded code** — put working code in `src/*.py`, reference from docs with relative links.
3. **Commit messages** — descriptive, single-line: `Add TOC, gap coverage, streaming & indexing examples for RAG section`.
4. **Don't auto-push** — commit locally, user decides when to push.
5. **Don't commit `.env` or company PDFs** — these are in `.gitignore`.
6. **Update `README.md`** when adding new sections — add the section header and file links.
7. **Update implementation doc** project structure tree when adding new scripts.
8. **Identify and call out bugs and improvements** explicitly when reviewing code, rather than silently fixing them.

---

## What NOT to Do

- Don't add features or refactor code beyond what's asked.
- Don't create subpages when a TOC will do.
- Don't add type annotations or docstrings to code you didn't write.
- Don't rename existing files without asking.
- Don't assume open-source repo files match — always read the workspace files.
- Don't push to remote without explicit permission.

---

## Full Dependency List (from `pyproject.toml`)

| Package | Purpose |
|---------|--------|
| `langchain` | Core orchestration framework |
| `langchain-openai` | OpenAI ChatGPT/Embeddings integration |
| `langchain-ollama` | Local LLM via Ollama |
| `langchain-pinecone` | Pinecone vector store integration |
| `langchain-community` | Community integrations (loaders, tools) |
| `langchain-text-splitters` | Document chunking strategies |
| `langchain-tavily` | Tavily web search tool for agents |
| `langsmith` | Observability, tracing, evaluation |
| `ollama` | Direct Ollama Python client |
| `python-dotenv` | `.env` file loading |
| `truststore` | Windows cert store injection (corporate proxy) |
| `httpx` | Async-capable HTTP client |
| `requests` | Standard HTTP client |
| `urllib3` | Low-level HTTP library |
| `pymupdf` | PDF parsing (primary, supports images) |
| `pypdf` | PDF parsing (fallback, pure Python) |
| `pdf2image` | PDF page to image conversion |
| `streamlit` | Rapid UI prototyping for chatbots/demos |

---

## Current Progress

| Section | Folder | Status |
|---------|--------|--------|
| 1. Introduction | `01-introduction/` | ✅ Complete |
| 2. GIST of LangChain | `02-gist-of-langchain/` | ✅ Complete |
| 3. GIST of AI Agents | `03-gist-of-ai-agents/` | ✅ Complete |
| 4-7. Agents Under the Hood | `04-07-agents-under-the-hood/` | ✅ Complete |
| 8. Function Calling | `08-function-calling/` | ✅ Complete |
| 9. GIST of RAG | `09-gist-of-rag/` | ✅ Complete |
| 10. Documentation Assistant | `10-documentation-assistant/` | 🔲 In Progress |
