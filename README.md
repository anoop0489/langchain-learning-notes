# LangChain & LangGraph Learning Journey 🦜🔗

This repository documents my journey mastering LangChain, LangGraph, and Agentic AI for production-grade applications. It follows a structured, project-based approach, mapping Python/LangChain "magic" to standard OOP (C#/Java) principles.

## 🗂️ Project Modules (Course Roadmap)

### 1. Fundamentals & Ice Breaker 🧊
*Folder: [`/ice_breaker`](./ice_breaker)*
The foundational concepts of LangChain, LCEL, and observability.
* [00. Environment Setup (`uv`, `.env`)](./ice_breaker/00_Environment_Setup.md)
* [01. Fundamentals & LCEL (PromptTemplates, Model Switching)](./ice_breaker/01_LangChain_Fundamentals.md)
* [02. Chat Model Architecture (Message Roles, Statelessness)](./ice_breaker/02_Chat_Models_Architecture.md)
* [03. LangSmith Tracing & Observability](./ice_breaker/03_LangSmith_Tracing.md)

### 2. The ReAct Agent Architecture 🤖
*Folder: [`/react-langchain`](./react-langchain)*
Moving from linear chains to autonomous AI Agents that can "think" and use tools.
* [04. AI Agents, Tools & Structured Output (Pydantic)](./react-langchain/04_AI_Agents_and_Tools.md)

### 3. Vector Databases & Retrieval 📚
*Folders: `/intro-to-vector-dbs`, `/vectorstore-in-memory`*
* *(Coming Soon: Embeddings, Vector Stores, Pinecone, FAISS)*

### 4. RAG Applications (Retrieval-Augmented Generation) 🔍
*Folder: `/documentation-helper`*
* *(Coming Soon: Building a chatbot over Python package docs using advanced RAG and Streamlit)*

### 5. LangGraph & Advanced Flow Engineering 🕸️
*Folders: `/langgraph-course`, `/reflection-agent`, `/reflexion-agent`*
* *(Coming Soon: Moving from basic AgentExecutors to complex, stateful graph architectures)*

### 6. Agentic RAG & Code Interpreters 💻
*Folders: `/agentic-rag`, `/code-interperter`*
* *(Coming Soon: Corrective/Adaptive RAG and lightweight code execution assistants)*

### 7. Model Context Protocol (MCP) 🔌
*Folder: `/mcpcrashcourse`*
* *(Coming Soon: Integrating standard MCP servers and clients)*

---

## 📖 Reference Guides

| Guide | Description | 
| :--- | :--- | 
| [Python OOP for C#/Java Devs](./reference-guides/Python_to_CSharp_Glossary.md) | Translates LangChain's Python architecture (kwargs, operator overloading, factory methods) into strict C#/Java OOP terminology. |

## 🛠️ Tech Stack
- **Languages/Tools:** Python (managed via `uv`), Pydantic
- **Frameworks:** LangChain, LangGraph
- **Observability:** LangSmith
- **Models:** OpenAI (GPT-4o), Ollama (Local LLMs)
- **APIs:** Tavily (Search)

---
*Created as part of my preparation for Senior AI Engineer roles.*