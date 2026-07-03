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
*Folder: [`/09-gist-of-rag`](./09-gist-of-rag)*
Retrieval Augmented Generation — from theory to a fully working pipeline with Pinecone.
* [09. RAG Theory & Concepts (Embeddings, Vector DBs, Chunking)](./09-gist-of-rag/09_RAG_Theory_And_Concepts.md)
* [10. RAG Implementation (Ingestion & Retrieval Pipelines)](./09-gist-of-rag/10_RAG_Implementation.md)

### Section 10: Documentation Assistant 🤖💬
*Folder: [`/10-documentation-assistant`](./10-documentation-assistant)*
Building a production documentation helper — web crawling, agentic RAG, Streamlit UI, and memory.
* [11. Doc Assistant Theory & Concepts (Tavily, Agents, Streamlit, Memory)](./10-documentation-assistant/11_DocAssistant_Theory_And_Concepts.md)
* [12. Doc Assistant Implementation (Crawl → Agent → Chat UI)](./10-documentation-assistant/12_DocAssistant_Implementation.md)

### Section 11: The GIST of LLMs & Prompt Engineering 🧠
*Folder: [`/11-llms-and-prompt-engineering`](./11-llms-and-prompt-engineering)*
The theoretical foundations — what LLMs are, how prompting techniques evolved, and the shift to context engineering.
* [13. LLMs, Prompting & Context Engineering (Zero-Shot → ReAct → Context Engineering)](./11-llms-and-prompt-engineering/13_LLMs_Prompting_And_Context_Engineering.md)

### Section 12: LLM Applications in Production 🚀
*Folder: [`/12-llm-apps-in-production`](./12-llm-apps-in-production)*
The capstone — production challenges, the LLM landscape, privacy, open-source vs managed, CAIR framework, and AI FOMO.
* [14. LLM Apps in Production (Challenges, Landscape, Privacy, CAIR, Strategic Decisions)](./12-llm-apps-in-production/14_LLM_Apps_In_Production.md)

### Section 13: LangGraph Fundamentals 🕸️
*Folder: [`/13-langgraph-fundamentals`](./13-langgraph-fundamentals)*
LangGraph orchestration runtime — prerequisites, state graphs, persistence, and agentic workflows.
* [15. LangGraph Prerequisites (Modern LangChain Patterns: ToolRuntime, Command, Middleware, StateGraph, Checkpointers)](./13-langgraph-fundamentals/15_LangGraph_Prerequisites.md)
* [16. Introduction to LangGraph (Autonomy Spectrum, Flow Engineering, Core Components)](./13-langgraph-fundamentals/16_LangGraph_Introduction.md)
* [17. LangGraph Terminology (Definitions, Flow Diagrams, State Updates, HIL, Syntax Reference)](./13-langgraph-fundamentals/17_LangGraph_Terminology.md)

### Section 14: Reflection Agent 🪞
*Folder: [`/14-reflection-agent`](./14-reflection-agent)*
Building a reflection agent that iteratively improves tweets through generate → critique → revise cycles.
* [18. Reflection Agent (Generate/Reflect Loop, Self-Improvement, Prompt Tricks)](./14-reflection-agent/18_Reflection_Agent.md)

### Section 15: Reflexion Agent 🔬
*Folder: [`/15-reflexion-agent`](./15-reflexion-agent)*
Tool-augmented self-improvement — extending reflection with Tavily search, structured output, and citations.
* [19. Reflexion Agent (Tool-Augmented Self-Critique, Structured Output, Citations)](./15-reflexion-agent/19_Reflexion_Agent.md)

### Section 16: Agentic RAG 🔍🤖
*Folder: [`/16-agentic-rag`](./16-agentic-rag)*
Adaptive retrieval with self-correction — routing, document grading, hallucination detection, and web search fallback.
* [20. Agentic RAG Theory (Adaptive RAG, Self-RAG, Routing, Grading, Hallucination Detection)](./16-agentic-rag/20_Agentic_RAG.md)
* [21. Agentic RAG Implementation (ChromaDB, Tavily, LangGraph Conditional Edges)](./16-agentic-rag/21_Agentic_RAG_Implementation.md)
* [22. Production Optimisation (Rerankers, Fewer LLM Calls, Cost/Latency Guide)](./16-agentic-rag/22_Production_Optimisation.md)

### Section 17: Introduction to MCP 🔌
*Folder: [`/17-introduction-to-mcp`](./17-introduction-to-mcp)*
Model Context Protocol — the universal standard for exposing tools, resources, and context to LLMs.
* [23. MCP Theory & Concepts (Protocol, Transports, Tools/Resources/Prompts, Interceptors)](./17-introduction-to-mcp/23_MCP_Theory_And_Concepts.md)



## 📖 Reference Guides

| Guide | Description | 
| :--- | :--- | 
| [Python OOP for C#/Java Devs](./reference-guides/Python_to_CSharp_Glossary.md) | Translates LangChain's Python architecture (kwargs, operator overloading, factory methods) into strict C#/Java OOP terminology. |
| [RAG Architecture Decisions](./reference-guides/RAG_Architecture_Decisions.md) | Deterministic vs Agentic vs Conversational RAG — cost breakdowns, memory strategies, system prompt optimization, and when to use which. |
| [Production Patterns](./reference-guides/Production_Patterns.md) | Deployment (FastAPI, containers), resilience (retries, fallbacks, circuit breakers), security (prompt injection, PII), and observability. |

## 🛠️ Tech Stack
- **Languages/Tools:** Python (managed via `uv`), Pydantic
- **Frameworks:** LangChain, LangGraph
- **Observability:** LangSmith
- **Models:** OpenAI (GPT-4o), Ollama (Local LLMs)
- **Vector Stores:** Pinecone (cloud), ChromaDB (local)
- **APIs:** Tavily (Search)

---
*Created as part of my preparation for Senior AI Engineer roles.*

### ⚖️ Disclaimer
* **Personal Project:** This repository is a personal portfolio and learning sandbox. The opinions, code, and architectural patterns expressed here are strictly my own and do not reflect the views, policies, or intellectual property of my current or former employers.
* **Educational Use Only:** The code provided in this repository is for educational and demonstrative purposes. It is not intended for production use without further security, scaling, and testing audits.
* **Liability:** All code and notes are provided "as-is" without warranty of any kind. I assume no liability for any direct or indirect damages, data loss, or system failures resulting from the use of this material.