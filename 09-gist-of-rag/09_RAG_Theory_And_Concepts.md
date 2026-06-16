# 09. The GIST of RAG — Embeddings, Vector Databases & Retrieval

A comprehensive guide to Retrieval Augmented Generation: what it is, why it matters, and how every component fits together.

*Based on Section 9: The GIST of RAG (Chapters 42–50)*

---

## What is this section about?

This section teaches you how to build a **RAG pipeline** — the most common production pattern for making LLMs answer questions about your private data. We start with the "why" (motivation), then move through every building block (embeddings, vector stores, chunking, retrieval), and end with two complete implementations.

---

## Key Definitions (Interview-Ready)

Use these as your opening sentence when asked "What is X?" in an interview:

| Term | Quick Recall (say this first) | Full Definition |
|------|------|------------|
| **RAG** | "Retrieve context, augment the prompt, generate the answer" | A technique that retrieves relevant documents from a knowledge base, injects them into the LLM's prompt as context, and generates a grounded answer — solving the problem of LLMs not knowing your private data. |
| **Embedding** | "Text → vector of numbers" | A numerical representation (vector) of text where semantically similar texts are close together in vector space, enabling mathematical similarity comparisons. Not all vectors are embeddings, but all embeddings are vectors. |
| **Vector** | "A list of numbers representing something" | A sequence of numerical values that represent the characteristics of an object (word, image, etc.) in a high-dimensional space. The building block of embeddings. |
| **Vector Database** | "Database optimized for similarity search" | A specialized database that stores high-dimensional vectors and provides fast nearest-neighbor search — enabling retrieval of semantically similar documents at scale. |
| **Chunking** | "Split large docs into digestible pieces" | The process of breaking a large document into smaller segments (chunks) that fit within the LLM's context window while preserving semantic meaning. |
| **Similarity Search** | "Find the closest vectors" | A query operation that takes a vector and returns the k nearest vectors in the database, measured by distance metrics like cosine similarity. |
| **Retriever** | "Query → relevant documents" | A component that takes a user query, embeds it, and returns the top-k most relevant document chunks from the vector store. In LangChain, a Runnable with `invoke()`. |
| **Context Window** | "Max tokens the LLM can process" | The hard limit on how many tokens (input + output) an LLM can handle in a single request — exceeded tokens are rejected. |
| **Document (LangChain)** | "`page_content` + `metadata`" | LangChain's universal data object for any loaded data. Has two attributes: `page_content` (the actual text) and `metadata` (a dict with `source`, custom fields for filtering/provenance). |
| **Document Loader** | "Any source → LangChain Document" | A LangChain abstraction that loads data from any source (PDF, text, Notion, Google Drive, Slack, WhatsApp) into a standardized `Document` object. Same `.load()` interface regardless of source. |
| **Text Splitter** | "Chunk with strategy" | A LangChain utility that splits documents into chunks using configurable strategies (character count, token count, recursive splitting) with optional overlap. |
| **Chunk Overlap** | "Shared text between adjacent chunks" | The number of characters/tokens that consecutive chunks share, preventing context loss when an answer spans a chunk boundary. |
| **`from_documents()`** | "Embed + store in one call" | LangChain's method on vector stores that iterates through documents, embeds each one, and stores vectors — handling batching, rate limits, and threading automatically. |
| **`as_retriever()`** | "Vector store → searchable Runnable" | Converts a vector store into a LangChain Retriever (a Runnable with `invoke()`) that wraps similarity search functionality. |
| **LCEL** | "Pipe operator for LangChain" | LangChain Expression Language — a declarative syntax using the `|` operator to compose Runnables into chains with built-in streaming, async, and batch support. |
| **RunnablePassthrough** | "Pass input through, optionally add keys" | A LangChain Runnable that forwards its input unchanged while optionally computing and assigning new keys to the output dictionary via `.assign()`. |
| **RunnableLambda** | "Auto-wrapped Python function" | When you pipe a regular Python function in an LCEL chain, LangChain auto-converts it into a `RunnableLambda` — giving it `invoke()`, `stream()`, and `batch()` capabilities. |
| **StrOutputParser** | "AIMessage → just the text" | A LangChain parser that extracts the `.content` string from an `AIMessage` response, so the chain output is a clean string instead of a message object. |
| **Grounding** | "Answer backed by evidence" | The practice of ensuring an LLM's response is based on retrieved factual data (the context) rather than its parametric knowledge or hallucination. |
| **Knowledge Cutoff** | "LLM doesn't know anything after date X" | The training data deadline of an LLM. Facts, products, or events after this date are unknown to the model — a key motivation for RAG. |
| **Hallucination** | "LLM confidently invents wrong facts" | When an LLM generates plausible-sounding but factually incorrect information because it lacks the real data. RAG directly combats this by providing actual source text. |

---

## The Problem RAG Solves

### Why can't we just ask the LLM directly?

LLMs are trained on public internet data up to a knowledge cutoff date. They have **no knowledge** of:
- Your company's internal documents
- Recent events after their training cutoff
- Private databases, PDFs, or codebases

### The Naive Solution (and why it fails)

> "Just stuff the entire document into the prompt."

This breaks for **four reasons**:

| Problem | Explanation |
|---------|-------------|
| **1. Token Limit** | LLMs have a hard cap (e.g., 4K, 128K, 1M tokens). Exceeding it = request rejected. |
| **2. Needle in a Haystack** | Research proves LLMs become less accurate with longer contexts — they "lose" information in the middle of long prompts. |
| **3. Cost** | More tokens = higher API costs. Sending 100K tokens when you need 500 is wasteful. |
| **4. Latency** | More tokens = slower processing time. Users wait longer for answers. |

### The RAG Solution

Instead of sending everything, we:
1. **Pre-process**: Split the document into small chunks and store them as vectors
2. **Retrieve**: Find only the chunks relevant to the user's question
3. **Augment**: Inject those chunks into the prompt as context
4. **Generate**: Let the LLM answer grounded in the relevant context

This solves all four problems: stays within token limits, focuses the LLM on relevant data, reduces cost, and improves latency.

### The Knowledge Cutoff Problem (Real Example from Course)

Eden demonstrates this live with the query *"What is Pinecone in machine learning?"*:

| Model | Response (No RAG) | Why? |
|-------|-------------------|------|
| **GPT-3.5** | "A pinecone algorithm is a method for searching hyperparameters..." | ❌ **Hallucination.** GPT-3.5 was trained before Pinecone (the vector DB) was widely known. |
| **GPT-4o/5** | "Pinecone is a managed vector database used in ML to store and index..." | ✅ Correct — trained on more recent data. |
| **GPT-3.5 + RAG** | "Pinecone is a fully managed cloud-based vector database..." | ✅ Correct — grounded in the retrieved blog content. |

**Key insight**: RAG lets even an older/cheaper model answer correctly by providing the source text. You don't need the most expensive model if you have good retrieval.

---

## The RAG Pipeline (Two Phases)

![Basic RAG Pipeline](./assets/basic_rag_pipeline.png)

### Phase 1: Ingestion (Offline, One-Time)

```
Document → Load → Chunk → Embed → Store in Vector DB
```

| Step | What Happens | LangChain Component |
|------|-------------|-------------------|
| **Load** | Read the file into a LangChain `Document` | `TextLoader`, `PyPDFLoader`, etc. |
| **Chunk** | Split into smaller pieces (e.g., 1000 chars) | `CharacterTextSplitter` |
| **Embed** | Convert each chunk into a vector | `OpenAIEmbeddings` |
| **Store** | Save vectors + metadata in vector DB | `PineconeVectorStore.from_documents()` |

### Phase 2: Retrieval + Generation (Online, Per-Query)

```
User Query → Embed → Similarity Search → Top-K Chunks → Augment Prompt → LLM → Answer
```

| Step | What Happens | LangChain Component |
|------|-------------|-------------------|
| **Embed Query** | Convert user question into a vector | `OpenAIEmbeddings` |
| **Search** | Find k nearest chunk vectors | `retriever.invoke(query)` |
| **Format** | Join chunks into a context string | `format_docs()` |
| **Augment** | Plug context + question into prompt template | `ChatPromptTemplate` |
| **Generate** | Send augmented prompt to LLM | `ChatOpenAI` |

---

## Deep Dive: LangChain Document & Document Loaders

### The `Document` Object

Everything in LangChain's RAG pipeline flows through `Document` objects. A Document has exactly two attributes:

```python
document.page_content   # str — the actual text content
document.metadata       # dict — metadata about the source
```

**Example after loading `mediumblog.txt`:**
```python
Document(
    page_content="Vector Database: What is it and why you should know it?...",
    metadata={"source": "mediumblog.txt"}
)
```

The `metadata.source` field is critical for **provenance** — it tells users (and the system) where the answer came from. In production, metadata can include page numbers, timestamps, authors, or custom fields for filtering.

### Document Loaders (The Abstraction)

LangChain implements loaders for dozens of data sources. The interface is always the same:

```python
loader = SomeLoader(source)
documents = loader.load()  # Always returns List[Document]
```

| Loader | Source | Example |
|--------|--------|---------|
| `TextLoader` | Plain text files | `.txt`, `.md` |
| `PyPDFLoader` | PDF documents | `.pdf` (page per Document) |
| `NotionDirectoryLoader` | Notion exports | Notion workspace |
| `SlackChatLoader` | Slack messages | Chat history |
| `YoutubeLoader` | YouTube transcripts | Video captions |
| `GoogleDriveLoader` | Google Drive | Any Drive file |
| `CSVLoader` | CSV files | Row per Document |

**The power of this abstraction**: Your downstream code (splitting, embedding, retrieval) doesn't care where the data came from. Swap `TextLoader` for `PyPDFLoader` and the rest of the pipeline is unchanged.

### Under the Hood (Source Code)

Eden dives into LangChain's source code to show how simple it is:

```python
# TextLoader source code (simplified):
class TextLoader:
    def load(self):
        with open(self.file_path) as f:
            text = f.read()
        return [Document(page_content=text, metadata={"source": self.file_path})]
```

Even the WhatsApp loader is just: open file → regex to extract sender/date/text → format → return as Document. The pattern is always the same.

---

## Deep Dive: Embeddings

### What is an Embedding?

An embedding model is a **black box** that:
- **Input**: Text (word, sentence, paragraph)
- **Output**: A vector (list of numbers, e.g., 1536 dimensions)

The key property: **semantically similar texts produce vectors that are close together** in vector space.

> **Important distinction**: An embedding IS a vector representation, but not all vectors are embeddings. An embedding specifically captures semantic meaning learned through training.

### Example

```
"I want to order an extra large coffee"  →  [0.23, -0.45, 0.89, ...]
"I'll have a tall coffee"                →  [0.21, -0.43, 0.91, ...]  ← CLOSE!
"quiero pedir café extra grande"         →  [0.22, -0.44, 0.90, ...]  ← CLOSE! (cross-language)
"The stock market crashed today"         →  [0.78,  0.12, -0.56, ...] ← FAR!
```

The embedding model doesn't care about language — it captures **meaning**. Two sentences with the same semantic intent will be close in vector space even if one is in English and the other in Spanish.

### How It Enables RAG

When a user asks "How tall is the Burj Khalifa?", we embed this question into a vector. In the vector space, this vector will be close to the Wikipedia paragraph about the Burj Khalifa's height (if we've embedded and stored it). We retrieve that paragraph, inject it as context, and the LLM can answer accurately even if it wasn't trained on this information.

### Distance Metrics

| Metric | Use Case | How It Works |
|--------|----------|--------------|
| **Cosine Similarity** | Most common, default in Pinecone | Measures angle between vectors (0° = identical) |
| **Euclidean Distance** | When magnitude matters | Straight-line distance in vector space |
| **Dot Product** | Fast computation | Combination of angle and magnitude |

### Embedding Models

| Model | Provider | Dimensions | Notes |
|-------|----------|-----------|-------|
| `text-embedding-3-small` | OpenAI | 512–1536 | Good balance of cost/quality |
| `text-embedding-ada-002` | OpenAI | 1536 | Legacy but still widely used |
| `text-embedding-3-large` | OpenAI | 256–3072 | Highest quality, most expensive |

**Rule of thumb**: Longer vectors = more semantic information captured, but higher storage/compute cost.

### LangChain's Uniform Embedding Interface

LangChain provides **one interface** for all embedding providers. Whether you use OpenAI, Cohere, Hugging Face, or any other provider — the API is the same:

```python
# Swap providers by changing one import + one class name. Downstream code stays identical.
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()  # Uses text-embedding-ada-002 by default

# OR:
from langchain_community.embeddings import CohereEmbeddings
embeddings = CohereEmbeddings()

# The rest of your code doesn't change:
vector = embeddings.embed_query("What is Pinecone?")
```

This is the same pattern as `init_chat_model()` for LLMs — abstract the provider, keep the interface consistent.

### ⚠️ Critical Rule: Same Model for Ingestion AND Retrieval

The embedding model used to embed your documents MUST be the same model used to embed the user's query. If you embed documents with `text-embedding-3-small` (1536 dimensions) but query with `text-embedding-ada-002` (also 1536 dimensions), the vectors live in different vector spaces and similarity search will return garbage.

**If you change your embedding model, you must re-embed all your documents.**

---

## Deep Dive: Vector Databases

### What is a Vector Database?

A specialized database that:
1. **Stores** high-dimensional vectors alongside metadata
2. **Indexes** vectors for fast nearest-neighbor search
3. **Retrieves** the top-k most similar vectors to a query vector

### Why not just use a regular database?

Regular databases search by exact match (`WHERE name = 'laptop'`). Vector databases search by **semantic similarity** — "find me documents that mean something similar to this question."

### Pinecone (Used in This Course)

- **Managed**: Cloud-based, no infrastructure to manage
- **Serverless**: Pay per query, scales automatically
- **Free tier**: Sufficient for learning and prototyping

#### Pinecone Index Configuration

| Setting | Our Value | Why |
|---------|-----------|-----|
| **Dimensions** | 1536 | Must match embedding model output size |
| **Metric** | Cosine | Default, works well for text similarity |
| **Type** | Dense | Standard for text embeddings |
| **Mode** | Serverless | Cost-effective for development |
| **Cloud/Region** | AWS us-east-1 | For production: same region as your app to avoid egress costs |

> **Production tip**: Deploy your vector store in the same cloud region as your RAG application. Cross-region calls add latency and egress costs.

#### Setting Up Pinecone

1. Go to [pinecone.io](https://pinecone.io) → Create account (free tier)
2. Create an index: set dimensions=1536, metric=cosine, type=dense
3. Generate an API key
4. Add to `.env`:
   ```bash
   PINECONE_API_KEY=pcsk_...        # LangChain looks for this exact env var name
   INDEX_NAME=medium-blogs-embeddings-index
   ```

> **Note**: `PINECONE_API_KEY` is the exact environment variable name that LangChain's Pinecone integration expects. If you name it differently, the integration won't auto-detect it.

#### Data Structure in Pinecone

Each record contains:
```json
{
  "id": "unique-vector-id",
  "values": [0.23, -0.45, 0.89, ...],  // The embedding vector (1536 numbers)
  "metadata": {
	"text": "The actual chunk content...",
	"source": "mediumblog.txt"            // Where this chunk came from
  }
}
```

The `metadata` field enables **filtering** in production — restrict searches to specific documents, date ranges, or categories.

#### What `from_documents()` Does Internally

Eden dives into LangChain's source to demystify `PineconeVectorStore.from_documents()`:

```python
# Simplified internal logic:
def from_documents(texts, embeddings, index_name):
	for batch in batched(texts):
		vectors = embeddings.embed_documents([t.page_content for t in batch])
		index.upsert(vectors_with_metadata)
```

What you get for free by using LangChain instead of calling Pinecone directly:
- **Batching**: Splits large document sets into API-friendly batches
- **Rate limit handling**: Retries with backoff when API limits are hit
- **Threading/AsyncIO**: Concurrent embedding calls for faster ingestion
- **Universal interface**: Same code works with Chroma, Weaviate, Qdrant — swap one class name

> *"Can we write this logic ourselves? Yes. But LangChain handles batching, rate limits, async, and gives one interface for all vector stores."* — Eden

### Other Vector Databases

| Database | Type | Key Differentiator |
|----------|------|-------|
| **Pinecone** | Managed (Cloud) | Fully managed, serverless, simple API, enterprise security. Used in this course. |
| **Chroma** | Open Source (Local) | In-memory or persistent mode. Great for prototyping/POC. Python/JS SDKs. |
| **Weaviate** | Open Source / Cloud | Supports multi-modal (text, images, audio). Integrates with HuggingFace, OpenAI, LangChain. |
| **Qdrant** | Open Source / Cloud | Rust-based, high performance. Supports nested filtering, disk storage, scalar quantization. |
| **Milvus** | Open Source | Designed for trillion-scale. Separates storage/compute for horizontal scaling. |
| **FAISS** | Library (Meta) | Not a database — a similarity search library. Fast but no persistence/scaling. |
| **pgvector** | PostgreSQL Extension | Add vector search to existing Postgres. Good when you already use Postgres. |

### How to Choose a Vector Database (Production Criteria)

| Factor | What to Evaluate |
|--------|-----------------|
| **Scalability** | Can it handle your data volume growth without performance loss? |
| **Performance** | Query latency, throughput under load, multi-dimensional search speed |
| **Security** | Encryption, access controls, compliance (SOC2, HIPAA, GDPR) |
| **Cost** | Pricing model (per query, per vector, per GB) — LLM APIs already expensive |
| **Deployment** | Cloud-only vs self-hosted vs hybrid — matches your infra preferences? |
| **Integration** | Native LangChain support? SDKs in your language? |

---

## Deep Dive: Chunking

### Why Chunk?

Even with 1M token context windows, chunking is critical because of **garbage in, garbage out**: sending irrelevant context degrades LLM answers, wastes tokens, and adds cost.

### Chunking Parameters

| Parameter | What It Controls | Our Value |
|-----------|-----------------|-----------|
| `chunk_size` | Maximum characters per chunk | 1000 |
| `chunk_overlap` | Characters shared between adjacent chunks | 0 |
| `separator` | Character(s) to split on | `\n\n` (double newline) |

### The Chunk Size Trade-Off

| Too Small | Too Large |
|-----------|-----------|
| Chunks lose context/meaning | Includes irrelevant information |
| Need more chunks to cover a topic | Wastes tokens and increases cost |
| More retrieval calls | Worse "needle in haystack" effect |

**Rule of thumb**: A chunk should make sense if you read it as a human. If it's gibberish in isolation, it's too small.

### Chunk Overlap

When `chunk_overlap > 0`, adjacent chunks share some text. This helps when an answer spans a chunk boundary — the overlapping text ensures both chunks contain the complete context.

```
Chunk 1: [A B C D E F G H]
Chunk 2:         [G H I J K L M N]   ← G,H appear in both (overlap = 2)
```

### Important: Chunks Are Still Documents

After splitting, each chunk is still a LangChain `Document` object — it inherits the metadata from the parent document (including `source`). So every chunk knows where it came from, enabling source attribution in answers.

### Chunk Sizes Are Approximate

Because splitting happens on separator characters (`\n\n`), some chunks may exceed `chunk_size`. If a paragraph is longer than 1000 characters and there's no separator within it, LangChain will produce a chunk larger than the limit. This is expected and usually not a problem with modern context windows (32K–1M tokens), but be aware of it.

### Advanced Chunking Strategies (Production)

| Strategy | When to Use |
|----------|-------------|
| **Character-based** | Simple text documents |
| **Token-based** | When you need precise token counts |
| **Recursive** | Structured documents (tries headers, then paragraphs, then sentences) |
| **Semantic** | When meaning boundaries don't align with formatting |
| **Code-aware** | Source code (respects function/class boundaries) |

---

## Deep Dive: The Retriever

### What is `as_retriever()`?

Calling `vectorstore.as_retriever()` returns a **VectorStoreRetriever** object — a LangChain Runnable that wraps the vector store's search functionality.

```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
```

- `k=3` means: return the top 3 most relevant chunks
- The retriever has an `invoke()` method (it's a Runnable)
- Under the hood, it calls the vendor's similarity search implementation

### Inheritance Chain

Eden traces through the source code to show the class hierarchy:

```
VectorStoreRetriever → BaseRetriever → Runnable
```

Because it's a **Runnable**, the retriever has:
- `invoke()` — synchronous retrieval
- `ainvoke()` — async retrieval
- `batch()` — multiple queries at once
- It can be piped (`|`) in LCEL chains

Internally, `invoke()` calls `get_relevant_documents()`, which each vector store vendor implements differently (Pinecone uses their SDK, Chroma uses theirs, etc.).

### How Retrieval Works (Step by Step)

1. User asks: *"What is Pinecone in machine learning?"*
2. The query string is embedded into a vector using the same embedding model
3. The vector is sent to Pinecone for similarity search
4. Pinecone returns the top-k vectors (with their metadata/text)
5. LangChain wraps them as `Document` objects with `page_content` and `metadata`

### Why k=3?

- More chunks = more context = better coverage **but** also more noise
- Fewer chunks = more focused **but** might miss relevant info
- `k=3` is a practical starting point; tune based on your data and prompt size

---

## Production Considerations

### Why Not Use an Agent for RAG?

LangChain's documentation shows an "Agentic RAG" approach where retrieval is wrapped as a tool and a ReAct agent decides when to call it. **Eden explicitly advises against this for production:**

| Agentic RAG (Not Recommended) | Deterministic RAG (Recommended) |
|------|------|
| LLM *decides* whether to search | Always searches the knowledge base |
| Can skip searches when needed | Predictable behavior every time |
| Can answer off-topic questions | Stays within business logic bounds |
| 2 LLM calls per query (reason + answer) | 1 LLM call per query |
| Higher latency and cost | Lower latency and cost |
| Vulnerable to jailbreaking | Controlled pipeline |

**When IS Agentic RAG appropriate?** When you genuinely need the agent to decide between multiple tools (web search, calculator, database, etc.) — not when retrieval should always happen.

### The Better Architecture (Coming Later)

The course later covers **Agentic RAG with LangGraph** — based on research papers with:
- Hallucination checking
- Answer relevance validation
- Deterministic graph-based control flow
- Much more robust than a simple ReAct agent

### LangSmith Tracing

Without LCEL, each step (retrieval, formatting, LLM call) appears as separate traces — hard to debug.

With LCEL, the entire pipeline appears as **one unified trace** showing:
- Input question
- Each step's duration
- The bottleneck (usually the LLM call)
- Final output

This is critical for production debugging and optimization.

---

## Interview Q&A Anchors

**Q: What is RAG and why do we need it?**

> **A:** RAG stands for Retrieval Augmented Generation. It solves the problem of LLMs not knowing about private or recent data. Instead of stuffing the entire document into the prompt (which hits token limits, costs more, adds latency, and reduces accuracy), RAG retrieves only the relevant chunks, augments the prompt with them, and generates a grounded answer. The three letters map directly to the pipeline: **R**etrieve relevant chunks → **A**ugment the prompt with context → **G**enerate the answer.

**Q: What are the four problems with stuffing the entire document into the prompt?**

> **A:** (1) **Token limit** — exceeding the context window causes request failure. (2) **Needle in a haystack** — longer prompts reduce LLM accuracy (proven by research). (3) **Cost** — more tokens = higher API bills. (4) **Latency** — more tokens = slower response times.

**Q: What is an embedding and why is it useful for RAG?**

> **A:** An embedding is a vector (list of numbers) that represents text in a high-dimensional space where semantically similar texts are close together. This enables us to convert a user's question into a vector, then use mathematical distance calculations to find the stored chunks whose vectors are closest — meaning they're semantically most relevant to the question.

**Q: What is a vector database and how is it different from a regular database?**

> **A:** A vector database stores high-dimensional vectors and provides fast nearest-neighbor similarity search. Unlike regular databases that do exact-match queries (`WHERE id = 5`), vector databases find items that are *semantically similar* to a query vector — essential for finding relevant document chunks in RAG.

**Q: Why do we split documents into chunks even with 1M token context windows?**

> **A:** Because of "garbage in, garbage out." Even if the context window is large enough, sending irrelevant text (1) costs more tokens, (2) proven by research to degrade answer quality (needle in haystack problem), and (3) increases latency. Chunking lets us send only the relevant portions.

**Q: What is the difference between the naive RAG implementation and the LCEL version?**

> **A:** Both produce the same answer. The naive version calls each step manually (retrieve → format → prompt → LLM) as separate function calls. The LCEL version composes them into a single chain using the `|` operator, which gives you: built-in streaming (`.stream()`), async support (`.ainvoke()`), batch processing (`.batch()`), better composability with other chains, and unified LangSmith tracing where all steps appear under one trace.

**Q: What does `RunnablePassthrough.assign()` do?**

> **A:** It passes the input dictionary through unchanged while computing and adding new keys. In our RAG chain, the input is `{"question": "..."}`. `RunnablePassthrough.assign(context=...)` keeps the `question` key and adds a `context` key computed by the retrieval sub-chain. The output `{"question": "...", "context": "..."}` then feeds into the prompt template.

**Q: Why does Eden recommend deterministic RAG over agentic RAG for production?**

> **A:** In production (e.g., customer support), you always want to query the knowledge base — there's no decision to make. Wrapping retrieval as a tool inside a ReAct agent adds: an extra LLM call (to decide whether to search), latency, cost, and vulnerability to jailbreaking. It also means the agent might skip the search or answer off-topic questions. Deterministic RAG guarantees predictable behavior, lower cost, and stays within business logic.

**Q: What is the `Document` object in LangChain and why does it have a `metadata` field?**

> **A:** A LangChain `Document` has two attributes: `page_content` (the text) and `metadata` (a dict). The `metadata.source` field provides **provenance** — it tells users where the answer came from (which file, which page). In production, you can add custom metadata (timestamps, categories, access control) for filtering during retrieval.

**Q: Why does LangChain auto-convert Python functions to `RunnableLambda` in LCEL chains?**

> **A:** When you pipe a regular Python function (like `format_docs`) in an LCEL chain, LangChain automatically wraps it in a `RunnableLambda`. This gives it the full Runnable interface (`invoke()`, `stream()`, `batch()`, `ainvoke()`) so it can participate in the chain's streaming and async capabilities without you writing boilerplate.

**Q: What happens if you change your embedding model after ingestion?**

> **A:** The vectors become incompatible. Each embedding model maps text to a different vector space. Even if two models produce the same dimensionality (e.g., both 1536), the vector spaces are different. You must **re-embed all documents** with the new model. This is why embedding model choice is an important upfront decision.

**Q: What is the `length_function` parameter in text splitters and when would you change it?**

> **A:** By default, chunk size is measured in characters using `len()`. But if you need precise control over token counts (important for cost optimization), you can provide a custom function that counts tokens using the model's tokenizer (e.g., `tiktoken` for OpenAI). This ensures chunks never exceed a specific token count.

**Q: Why is LangChain's one-interface pattern important for vector stores specifically?**

> **A:** Vector stores differ widely in their APIs, authentication, query syntax, and data formats. LangChain abstracts this behind a common interface (`from_documents()`, `as_retriever()`, `similarity_search()`). This means you can prototype with Chroma (free, local) and deploy with Pinecone (managed, scalable) by changing one class name — no rewrite needed.

---

## References

- [LangChain RAG Tutorial](https://docs.langchain.com/)
- [LangChain Python Build Overview](https://docs.langchain.com/oss/python/build-overview)
- [LangChain API Reference](https://reference.langchain.com/python/langchain)
- [What is RAG? (Medium)](https://medium.com/@drjulija/what-is-retrieval-augmented-generation-rag-938e4f6e03d1)
- [Needle in a Haystack Research](https://arxiv.org/abs/2307.03172)
- [Pinecone Documentation](https://docs.pinecone.io/)
