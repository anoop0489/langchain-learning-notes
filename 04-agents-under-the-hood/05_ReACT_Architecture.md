# 04. Agents Under The Hood: The ReACT Architecture 🤖

*Based on Section 4: Agents Under The Hood (Lectures 24 - 27)*

## 🎯 What You Will Learn
* The history and theory behind the **ReACT** (Reason + Act) algorithm.
* The anatomy of an Agent Loop: **Thought -> Action -> Observation**.
* How the "Agent Scratchpad" acts as the state memory for the loop.
* Setting up local, open-weights LLMs using **Ollama** (`qwen` model).
* The Course Roadmap: Peeling back LangChain's abstractions from "Magic" (Layer 0) to "Raw Regex" (Layer 3).

## 📦 Dependency Setup
We are adding formatting tools (`black`, `isort`) and local LLM support (`langchain-ollama`). Run this in your terminal:
```bash
uv add langchain langchain-ollama langchain-openai python-dotenv black isort
```

---

## 1. Core Architectural Concepts

### The ReACT Algorithm (Reason + Act)
First published in a 2023 Princeton/Google research paper, ReACT is the foundational algorithm powering tools like Devin, GitHub Copilot Workspace, and advanced AI agents. 

An LLM by itself can only generate text. It cannot *do* things. ReACT solves this by forcing the LLM into a strict loop where it must **Reason** about what to do next, **Act** by requesting a specific tool, and then wait for an **Observation** (the result of the tool) before thinking again.

### C#/Java Analogy: The State Machine
* **Traditional Script:** Linear execution. `Step 1 -> Step 2 -> Step 3`.
* **ReACT Agent:** A **State Machine** wrapped in a `while(!state.IsFinished)` loop. 
  * The **LLM** acts as the `CpuContext`. It analyzes the current `State` and returns a `Command` (e.g., `ExecuteTool("GetPrice")`).
  * Your **Python/C# Code** is the runtime. It executes the C# method, appends the return value to the `State` history, and loops back to feed the new `State` to the LLM.

---

## 2. Component Overview: The E-Commerce Agent

In this section, Eden has us build an e-commerce bot to answer questions like: *"What is the price of a laptop with the gold discount?"*

To prove how agents work, we will give the agent two distinct tools (which you can think of as C# Interfaces):
1. `get_product_price(product)` -> Returns the base price of the item.
2. `apply_discount(price, tier)` -> Returns the final price after applying a Bronze, Silver, or Gold discount.

### The ReACT Loop: Step-by-Step Breakdown
Here is exactly what happens inside the `while` loop when a user asks for that discounted laptop:

**Iteration 1**
1. **Thought (LLM):** "The user wants a discounted price. First, I need to know the base price of a laptop. I have a tool called `get_product_price`."
2. **Action (LLM):** `CALL: get_product_price(product="laptop")`
3. **Observation (Python):** Our local Python code intercepts this, runs the actual database query, and returns `"$1000"` to the agent's memory.

**Iteration 2**
1. **Thought (LLM):** "Okay, the laptop is $1000. Now I need to apply the 'gold' discount. I have a tool called `apply_discount`."
2. **Action (LLM):** `CALL: apply_discount(price=1000, tier="gold")`
3. **Observation (Python):** Our Python code calculates it and returns `"$850"`.

**Iteration 3**
1. **Thought (LLM):** "The final price is $850. I have all the information I need to answer the user. I do not need any more tools."
2. **Action (LLM):** `RETURN_ANSWER: "The price of the laptop with the gold discount is $850."`
3. **Loop Ends.**

---

## 🧠 3. The Abstraction Layers Roadmap

To give us the deepest possible understanding of AI Agents, Eden will guide us backward through the abstraction layers:

* **Layer 0 (The LangChain Magic):** Using `create_agent()`. You pass tools, it works, but it's a black box.
* **Layer 1 (LangChain Primitives):** We write the `while` loop ourselves, but use LangChain's `bind_tools` and `ToolMessage` objects to handle the complex formatting.
* **Layer 2 (Raw Function Calling):** We write the raw JSON schemas for the tools without LangChain, sending pure API requests to the LLM.
* **Layer 3 (Pure ReACT / No Function Calling):** We go back to 2023. No JSON schemas. We force the LLM to output text formatted exactly like `Action: ToolName`. We use **Regular Expressions (Regex)** in Python to parse the string, run the tool, and feed it back. 

---

## 💻 4. Environment Setup (Ollama & Git)

### A. Code Repository Sync
If you are following Eden's GitHub repo, checkout the starting branch for this section:
```bash
git checkout -b project/agents-under-the-hood <commit-hash>
```

### B. Ollama (Local LLM Engine)
* **What it is:** A lightweight framework that allows you to run large language models directly on your local CPU/GPU without paying OpenAI API fees.
* **C# Analogy:** Think of Ollama as **Docker Desktop for LLMs**. You "pull" an image (the model weights) and "run" it, exposing a local REST API (`localhost:11434`) that LangChain can talk to.

To run the local models, install Ollama and run:
```bash
# 1. Download the Qwen model (lightweight and supports tool calling)
ollama pull qwen2.5:1.5b

# 2. Test it in the CLI
ollama run qwen2.5:1.5b
# Type "Hi" to verify it responds, then type "/bye" to exit.

# 3. Start the Local Server
ollama serve
```
*(Leave the `ollama serve` terminal window open. Your Python code will now route its API requests to this local server instead of OpenAI).*

---

## 5. Quick Reference Dictionary

| Concept | Definition | OOP Equivalent |
| --- | --- | --- |
| **ReACT** | Reason + Act. The theoretical algorithm for agent loops. | A State Machine inside a `while` loop. |
| **Thought** | The LLM's internal reasoning about what to do next. | CPU evaluating the `CurrentState`. |
| **Action** | The LLM deciding to invoke a specific tool. | Emitting an `ICommand` interface. |
| **Observation** | The raw string result returned by the executed tool. | The return value of the executed C# method. |
| **Scratchpad** | The running history of all Thoughts, Actions, and Observations appended to the prompt. | The `List<StateHistory>` object. |

---

## ⚠️ Production Notes (What Breaks & How to Fix It)

* **Context Window Exhaustion:** The "Agent Scratchpad" grows with every loop iteration. If your agent gets stuck in a loop of calling tools incorrectly, the scratchpad will eventually exceed the LLM's token limit, causing an API crash.
  * **The Fix:** Implement a strict `max_iterations` counter (e.g., force quit if the loop runs more than 5 times).
* **Local Model Limitations (Ollama):** Lightweight models like Qwen 1.5B are fast, but they are not GPT-4o. They might occasionally hallucinate the JSON arguments or forget to include required tool parameters.
  * **The Fix:** Your Python code must include `try/catch` blocks around tool execution. If a tool fails, you must catch the error and feed it *back* to the LLM as an Observation (e.g., `"Observation: Error - missing argument 'tier'."`) so the LLM can correct its mistake in the next loop.

---

## 6. Interview Q&A Anchors

**Q: Explain the ReACT algorithm in the context of AI Agents.**
> **A:** ReACT stands for Reason and Act. It is an iterative loop where the LLM is prompted to first analyze the current state (Thought), choose a specific predefined function to execute along with its arguments (Action), and wait. The application code then executes that function and returns the result (Observation) back to the LLM. This loop continues until the LLM determines it has enough information to formulate a final response to the user.

**Q: How does the LLM "remember" what tools it has already called during an execution loop?**
> **A:** Through the Agent Scratchpad. Because LLMs are inherently stateless, the orchestrating framework (like LangChain or custom Python code) must append the history of all previous Thoughts, Actions, and Observations to the system prompt on every single iteration of the loop.

**Q: What is the benefit of testing Agents locally with Ollama instead of OpenAI?**
> **A:** Cost and privacy. Agent loops are heavily iterative; a single complex user request might result in 10-15 API calls as the agent reasons and acts. Using a local open-weights model via Ollama during development prevents racking up massive token charges on OpenAI, while keeping proprietary data entirely on the local machine.
