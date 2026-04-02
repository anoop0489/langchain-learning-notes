# LangChain & LangGraph Learning Journey 🦜🔗

This repository documents my journey mastering LangChain, LangGraph, and Agentic AI for production-grade applications. It follows a structured, project-based approach, mapping Python/LangChain "magic" to standard OOP (C#/Java) principles.

## 🗂️ Project Modules (Course Roadmap)

### 1. Fundamentals & Ice Breaker 🧊
*Folder: [`/01-langchain-fundamentals`](./01-langchain-fundamentals)*
The foundational concepts of LangChain, LCEL, and observability.
* [00. Environment Setup (`uv`, `.env`)](./00-environment-setup/00_Environment_Setup.md)
* [01. Fundamentals & LCEL (PromptTemplates, Model Switching)](./01-langchain-fundamentals/01_LangChain_Fundamentals.md)
* [02. Chat Model Architecture (Message Roles, Statelessness)](./01-langchain-fundamentals/02_Chat_Models_Architecture.md)
* [03. LangSmith Tracing & Observability](./01-langchain-fundamentals/03_LangSmith_Tracing.md)

### 2. Intro to AI Agents 🤖
*Folder: [`/02-intro-to-ai-agents`](./02-intro-to-ai-agents)*
Moving from linear chains to autonomous AI Agents that can "think" and use tools.
* [04. AI Agents, Tools & Structured Output (Pydantic)](./02-intro-to-ai-agents/04_AI_Agents_and_Tools.md)

### 3. Agents Under The Hood 🕵️‍♂️
*Folder: [`/04-agents-under-the-hood`](./04-agents-under-the-hood)*
Peeling back LangChain's abstractions to understand the raw ReACT loop, state machines, and local LLM execution.
* [05. The ReACT Architecture & Local LLMs](./04-agents-under-the-hood/05_ReACT_Architecture.md)

### 4. Vector Databases & Retrieval 📚
*Folder: `/05-vector-dbs`*
* *(Coming Soon: Embeddings, Vector Stores, Pinecone, FAISS)*

### 5. RAG Applications (Retrieval-Augmented Generation) 🔍
*Folder: `/06-documentation-helper`*
* *(Coming Soon: Building a chatbot over Python package docs using advanced RAG and Streamlit)*

### 6. LangGraph & Advanced Flow Engineering 🕸️
*Folders: `/07-langgraph-course`*
* *(Coming Soon: Moving from basic AgentExecutors to complex, stateful graph architectures)*

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

### ⚖️ Disclaimer
* **Personal Project:** This repository is a personal portfolio and learning sandbox. The opinions, code, and architectural patterns expressed here are strictly my own and do not reflect the views, policies, or intellectual property of my current or former employers.
* **Educational Use Only:** The code provided in this repository is for educational and demonstrative purposes. It is not intended for production use without further security, scaling, and testing audits.
* **Liability:** All code and notes are provided "as-is" without warranty of any kind. I assume no liability for any direct or indirect damages, data loss, or system failures resulting from the use of this material.