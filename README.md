# LangChain Learning Journey 🦜🔗

This repository documents my journey mastering LangChain for production-grade AI applications. Each file represents a core concept, complete with code examples, technical breakdowns, and interview-ready explanations.

## 📚 Table of Contents

| Topic | Description | Key Concepts |
| :--- | :--- | :--- |
| [00. Environment Setup](./00_Environment_Setup.md) | Modern Python setup using `uv` for speed and isolation. | `uv init`, `.env`, API Security |
| [01. Fundamentals & LCEL](./01_LangChain_Fundamentals.md) | The "Hello World" Chain, PromptTemplates, and Model Switching. | `ChatPromptTemplate`, `LCEL`, `StrOutputParser`, `ChatOllama` |
| [02. Chat Model Architecture](./02_Chat_Models_Architecture.md) | Deep dive into Message Roles, Statelessness, and the "Context Window". | `SystemMessage`, `HumanMessage`, `AIMessage` |
| [03. LangSmith Tracing](./03_LangSmith_Tracing.md) | Debugging, Observability, and Latency tracking for LLM applications. | `LANGCHAIN_TRACING_V2`, `Callbacks` |

## 🛠️ Tech Stack
- **Python** (Managed via `uv`)
- **LangChain**
- **LangSmith** (Observability & Tracing)
- **OpenAI (GPT-4o)**
- **Ollama** (Local LLMs)

---
*Created as part of my preparation for Senior AI Engineer roles.*
