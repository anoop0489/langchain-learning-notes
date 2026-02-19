# 03. LangSmith Tracing & Observability 🕵️‍♂️

## 1. Core Concept: The "Black Box" Problem
In production, LLM chains are often "black boxes." You send text in, and you get text out. But when something goes wrong (e.g., the bot gives a rude answer, hallucinates, or takes 15 seconds to reply), you need to know exactly what happened inside the box. You need to know:
* **Exact Input:** What was the *exact* formatted prompt sent to the model after variables were injected?
* **Latency:** Which specific step slowed down the response?
* **Token Usage:** How much did this specific run cost?

**LangSmith** is the observability platform (the "Debugger") built by LangChain that solves this by tracing every single step of your chain.

## 2. Environment Setup
To enable tracing, you do not need to change your core LangChain Python code. You simply set environment variables that LangChain automatically detects.

### A. Get your API Key
1. Go to [smith.langchain.com](https://smith.langchain.com/) and create an account.
2. Navigate to **Settings** (gear icon) -> **API Keys** -> **Create API Key**.

### B. Update your `.env` file
Add these exact lines to the `.env` file we created in Module 00:

```bash
# 1. Enable Tracing (The master switch)
LANGCHAIN_TRACING_V2=true

# 2. Your LangSmith API Key
LANGCHAIN_API_KEY=lsv2_pt_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# 3. Project Name (Groups your traces together in the UI)
LANGCHAIN_PROJECT=Course_First_Project

```

## 3. Code Implementation

*This script demonstrates how tracing works invisibly. As long as `load_dotenv()` runs and the variables are set, the trace is sent to LangSmith automatically.*

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Load Environment Variables (Triggers LangSmith automatically if set)
load_dotenv()

def main():
    # 2. Verify Tracing Status (Good practice for development)
    if os.getenv("LANGCHAIN_TRACING_V2") == "true":
        print("✅ LangSmith Tracing is ENABLED. Traces will be logged to your dashboard.")
    else:
        print("⚠️ Tracing is DISABLED. Set LANGCHAIN_TRACING_V2=true in .env")

    # 3. Define the Chain Components
    information = "LangChain is a framework for developing applications powered by language models."
    
    chat_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI tutor. Summarize the following concept in exactly one sentence."),
        ("human", "{information}")
    ])
    
    # We use a chat model here, but any model works with LangSmith
    llm = ChatOpenAI(temperature=0, model="gpt-4o")
    
    # The Output Parser cleans up the AIMessage into a raw string
    parser = StrOutputParser()

    # 4. Build the LCEL Chain
    chain = chat_template | llm | parser

    # 5. Execution
    # When you call .invoke(), LangChain spins up a background thread to send the trace to LangSmith.
    print("\nExecuting Chain...")
    response = chain.invoke(input={"information": information})
    
    print("\n--- RESPONSE ---")
    print(response)
    print("\n👉 Now go check [https://smith.langchain.com](https://smith.langchain.com) to see the trace of this run!")

if __name__ == "__main__":
    main()

```

## 4. Technical Deep Dive (Under the Hood)

### A. How Tracing Works: The Callback System

How does LangChain log data without you writing `logger.info()` everywhere? It uses a **Callback Handler Architecture**.

* When you initialize a model or chain, LangChain looks for the `LANGCHAIN_TRACING_V2` environment variable.
* If it is set to `true`, LangChain automatically injects a `LangChainTracer` into the callback manager.
* As your chain runs, events (like `on_chain_start`, `on_llm_start`, `on_llm_end`) are fired.
* The Tracer catches these events and sends HTTP requests to the LangSmith API asynchronously.

### B. Key Metrics to Watch in the LangSmith UI

When you open a trace in LangSmith, a Senior Engineer looks for:

1. **P50 / P99 Latency:** If your P99 latency is 10 seconds, 1% of your users are experiencing terrible performance. You can click the trace to see if the LLM or your local code is the bottleneck.
2. **Tokens per Run:** This directly correlates to your OpenAI/Anthropic bill. LangSmith calculates exact costs per run.
3. **Exact Prompt Inputs:** Often, bugs are caused by variables not injecting correctly into the `{information}` placeholder. Tracing shows you the final, compiled string sent to the API.

---

## 5. Interview Q&A Anchors

**Q: How do you debug and monitor an LLM application in production?**

> **A:** I use an observability platform like **LangSmith** (or Datadog/Phoenix). By enabling `LANGCHAIN_TRACING_V2`, I can capture the execution path of every request. This allows me to inspect the exact prompt that was rendered, track token usage for cost analysis, and identify if a hallucination was caused by bad context retrieval or a model failure.

**Q: Does enabling tracing slow down the user's application?**

> **A:** Minimally. LangChain's tracing is designed to be **non-blocking**. It uses background threads to batch and send trace data to the LangSmith API asynchronously. The main execution thread (which the user is waiting on) is not significantly delayed by the logging process.

**Q: What is the architectural difference between standard "Logging" and LLM "Tracing"?**

> **A:** **Logging** usually captures discrete, disconnected events (e.g., "Error on line 50"). **Tracing** captures the *hierarchical relationships* and *latency* between steps. For example, in an Agent loop `Thought -> Action -> Observation`, a trace visually nests these steps, showing exactly which tool was called and how long it took, which a flat log file cannot easily represent.
