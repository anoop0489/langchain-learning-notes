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

Core Concept: Deconstructing the Code

To master LangChain, you must understand the underlying Python concepts driving it. 

* **Class:** A blueprint or template for creating objects. (e.g., `ChatOpenAI` is the blueprint for a chat model).
* **Object (Instance):** A specific realization of that blueprint. (e.g., `llm = ChatOpenAI()` creates a specific AI object you can interact with).
* **Method:** A function that belongs to a Class or Object. (e.g., `.invoke()` is an action the object can perform).
* **Tuple:** An unchangeable list in Python, written with parentheses `("system", "You are a tutor")`.
* **String:** Plain text, written in quotes `"Hello"`.


*This script demonstrates how tracing works invisibly. As long as `load_dotenv()` runs and the variables are set, the trace is sent to LangSmith automatically.*

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. LOAD ENVIRONMENT VARIABLES
load_dotenv() 
# Technical Definition: A Function. It reads key-value pairs from a .env file and adds them to os.environ.
# Why: This keeps secrets (API Keys) out of your code. It also triggers LangSmith tracing 
# if LANGCHAIN_TRACING_V2=true is found in your environment.

def main():
    # 2. DEFINE THE DATA
    information = "LangChain is a framework for developing applications powered by language models."
    # Technical Definition: A standard Python string variable representing our dynamic user data.

    # 3. DEFINE THE MESSAGE TEMPLATE (THE "BLUEPRINT")
    # CRITICAL: We use TUPLES here, not Classes/Objects.
    messages = [
        ("system", "You are a helpful AI tutor. Summarize the following concept in exactly one sentence."),
        ("human", "{information}"),
    ]
    # Technical Distinction:
    # - ("human", "{information}") is a SCHEMA (a template tuple). It is NOT a HumanMessage object yet.
    # - Using an Object like 'HumanMessage(content="{information}")' would fail here because 
    #   it "locks" the text in as literal string data before the variable can be injected.
    
    # 4. INSTANTIATE THE CHAT PROMPT TEMPLATE
    chat_template = ChatPromptTemplate.from_messages(messages)
    # Technical Definition: 'ChatPromptTemplate' is the Class. '.from_messages()' is a Class Method.
    # 'chat_template' is the Object. 
    # Why: It acts as the "Engine" that will later combine your blueprint tuples with your string data.

    # 5. INSTANTIATE THE CHAT MODEL (THE "ENGINE")
    llm = ChatOpenAI(temperature=0, model="gpt-4o")
    # Technical Definition: 'ChatOpenAI' is the Class; 'llm' is the Object (Instance).
    # - Temperature=0: Sets the model to be 'deterministic' (consistent, not creative).
    # - Model="gpt-4o": Specifies the version of the neural network to be used.

    # 6. INSTANTIATE THE OUTPUT PARSER (THE "CLEANER")
    parser = StrOutputParser()
    # Technical Definition: 'StrOutputParser' is the Class. 'parser' is the Object.
    # Why: It intercepts the AI's complex 'AIMessage' object and extracts only the 'content' string. 
    # Without this, you get a JSON-like object back.

    # 7. BUILD THE LCEL CHAIN (THE "PIPELINE")
    chain = chat_template | llm | parser
    # Technical Definition: LCEL (LangChain Expression Language) uses the pipe operator '|'.
    # Python Secret: The "|" symbol is actually a Python feature called "Operator Overloading". 
    # It triggers a hidden dunder (double underscore) method called `__or__`. 
    # It tells Python to pass the output of the left object into the input of the right object.

    

    # 8. EXECUTION (THE "START" BUTTON)
    response = chain.invoke(input={"information": information})
    # Technical Definition: .invoke() is the Method that triggers the chain.
    # Behind the scenes: 
    #   1. LangChain finds the "{information}" placeholder in your tuple.
    #   2. It injects the 'information' string.
    #   3. It finally creates the 'HumanMessage' OBJECT and sends it to the AI.
    #   4. It sends a 'trace' to LangSmith asynchronously for debugging.

    # 9. OUTPUT
    print(response)

if __name__ == "__main__":
    main()

```

## 4. Quick Reference Dictionary

| Concept | Technical Status | GitHub Note for Developers |
| --- | --- | --- |
| **`HumanMessage`** | **Class** | The "blueprint" imported from `langchain_core`. |
| **`HumanMessage(content="Hi")`** | **Object / Instance** | A final, "inked" message. Do not use this inside templates because the variable placeholders won't work. |
| **`("human", "{var}")`** | **Template Schema** | A "penciled" instruction (Tuple). Use this inside `ChatPromptTemplate` so LangChain can swap `{var}` for real data during `.invoke()`. |
| **`LANGCHAIN_TRACING_V2`** | **Env Variable** | The current standard for enabling LangSmith (replaces the legacy LANGSMITH_TRACING). |
| **` | ` (Pipe Operator)** | **Dunder Method (`__or__`)** |

## 5. Technical Deep Dive (Under the Hood)

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

## 6. Interview Q&A Anchors

**Q: How do you debug and monitor an LLM application in production?**

> **A:** I use an observability platform like **LangSmith** (or Datadog/Phoenix). By enabling `LANGCHAIN_TRACING_V2`, I can capture the execution path of every request. This allows me to inspect the exact prompt that was rendered, track token usage for cost analysis, and identify if a hallucination was caused by bad context retrieval or a model failure.

**Q: Does enabling tracing slow down the user's application?**

> **A:** Minimally. LangChain's tracing is designed to be **non-blocking**. It uses background threads to batch and send trace data to the LangSmith API asynchronously. The main execution thread (which the user is waiting on) is not significantly delayed by the logging process.

**Q: What is the architectural difference between standard "Logging" and LLM "Tracing"?**

> **A:** **Logging** usually captures discrete, disconnected events (e.g., "Error on line 50"). **Tracing** captures the *hierarchical relationships* and *latency* between steps. For example, in an Agent loop `Thought -> Action -> Observation`, a trace visually nests these steps, showing exactly which tool was called and how long it took, which a flat log file cannot easily represent.
