# LangChain & LangGraph Learning Journey 🦜🔗

This repository documents my journey mastering LangChain, LangGraph, and Agentic AI for production-grade applications. It follows a structured, project-based approach, mapping Python/LangChain "magic" to standard OOP (C#/Java) principles.

## 🗂️ Project Modules (Course Roadmap)

### Section 1: Introduction 🚀
*Folder: [`/01-introduction`](./01-introduction)*
Environment setup, tooling, and project initialization.
* [00. Environment Setup (`uv`, `.env`, API Keys, Ollama)](./01-introduction/00_Environment_Setup.md)

### Section 2: The GIST of LangChain 🧊
*Folder: [`/02-gist-of-langchain`](./02-gist-of-langchain)*
The foundational concepts of LangChain, LCEL, and observability.
* [01. Fundamentals & LCEL (PromptTemplates, Model Switching)](./02-gist-of-langchain/01_LangChain_Fundamentals.md)
* [02. Chat Model Architecture (Message Roles, Statelessness)](./02-gist-of-langchain/02_Chat_Models_Architecture.md)
* [03. LangSmith Tracing & Observability](./02-gist-of-langchain/03_LangSmith_Tracing.md)

### Section 3: The GIST of AI Agents 🤖
*Folder: [`/03-gist-of-ai-agents`](./03-gist-of-ai-agents)*
Moving from linear chains to autonomous AI Agents that can "think" and use tools.
* [04. AI Agents, Tools & Structured Output (Pydantic)](./03-gist-of-ai-agents/04_AI_Agents_and_Tools.md)

### Sections 4–7: Agents Under The Hood 🕵️‍♂️
*Folder: [`/04-07-agents-under-the-hood`](./04-07-agents-under-the-hood)*
Peeling back LangChain's abstractions layer by layer — from framework magic to raw regex.
* [05. The ReACT Architecture & Local LLMs](./04-07-agents-under-the-hood/05_ReACT_Architecture.md) *(Section 4)*
* [06. Agents Under the Hood - Tool Calling & Raw Loops](./04-07-agents-under-the-hood/06_Agents_Under_The_Hood.md) *(Sections 5–7)*

### Section 8: Function Calling 🔧
*Folder: [`/08-function-calling`](./08-function-calling)*
Theory of function calling -- why it replaced the ReAct prompt and how it works.
* [08. Function Calling (Theory)](./08-function-calling/08_Function_Calling.md)

### Section 9: The GIST of RAG 📚
*Folder: `/09-gist-of-rag`*
* *(Coming Soon: Embeddings, Vector Databases, & Retrieval)*

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