# 06. Deconstructing Agents: LangChain Primitives (Layer 1) 🧅

*Based on Section 5: Agents Under The Hood - LangChain Tool Calling*

## 🎯 What You Will Learn
* The three layers of Agent abstraction: LangChain Primitives, Raw JSON Calling, and Raw ReAct.
* How the `@tool` decorator uses reflection to auto-generate JSON schemas.
* How to manually build an Agent `while` loop (replacing the `AgentExecutor` black box).
* The critical role of `ToolMessage` and the `tool_call_id` in maintaining state.

---

## 1. The Abstraction Hierarchy 

Every AI agent follows the same core ReAct loop: **Reason -> Execute Tool -> Observe -> Repeat**. What changes is *how much of that loop you have to write yourself*.

Eden breaks this down into three distinct architectural layers. We are starting at Layer 1.

| Layer | Implementation | C#/Java Analogy |
| :--- | :--- | :--- |
| **Layer 1: LangChain Primitives** | We write the `while` loop. LangChain handles the JSON serialization via `@tool` and `bind_tools()`. | **Micro-ORMs (Dapper).** You write the SQL (the loop), but the framework handles the object mapping for you. |
| **Layer 2: Raw Function Calling** | We discard LangChain. We manually write the JSON schemas and invoke the raw API (e.g., Ollama SDK). | **Raw ADO.NET / JDBC.** You write raw `SqlCommand` strings and map the `DataReader` entirely by hand. |
| **Layer 3: Pure ReAct (Regex)** | No JSON APIs allowed. We prompt the LLM to output a specific string format and parse it with Regex. | **Raw TCP Sockets.** Building your own database protocol parser over a raw network stream. |

---

## 2. The Complete Layer 1 Code (Heavily Annotated for Beginners)

Read this script top-to-bottom. The comments explain exactly what LangChain is doing behind the scenes at every step.

```python
import os
from dotenv import load_dotenv

# 1. Load environment variables (like your LANGSMITH_API_KEY) from the .env file
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"


# ==========================================
# PART 1: DEFINING THE TOOLS
# ==========================================

# @tool is a LangChain Magic Keyword (Decorator).
# LLMs cannot read Python code. They only read JSON. 
# @tool automatically reads the data types (product: str) and the docstring ("""Look up...""")
# and converts them into a JSON Instruction Manual so the LLM knows this tool exists.
@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f"    >> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    print(f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


# ==========================================
# PART 2: THE AGENT LOOP (THE STATE MACHINE)
# ==========================================

# @traceable tells LangSmith to record everything that happens inside this function.
# You can log into LangSmith's website later to see exactly what the LLM was thinking.
@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    
    # 1. Group our tools together
    tools = [get_product_price, apply_discount]
    
    # 2. Create a lookup dictionary: {"get_product_price": <actual_function_code>}
    # We need this later so when the LLM says "Run get_product_price", we can find the code.
    tools_dict = {t.name: t for t in tools}

    # 3. Setup the AI Model (Connecting to our local Ollama server)
    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    
    # 4. Attach the tools to the AI
    # This does NOT run the tools. It just attaches the JSON Instruction Manuals
    # we created earlier to the LLM, saying: "You are allowed to use these if you need them."
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("=" * 60)

    # 5. Create the "Agent Scratchpad" (The Conversation History)
    # The LLM has no memory. We must send it the ENTIRE history every single time we talk to it.
    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool and a discount tool.\n\n"
                "STRICT RULES:\n"
                "1. NEVER guess or assume any product price. MUST call get_product_price first.\n"
                "2. Only call apply_discount AFTER you have received a price.\n"
            )
        ),
        HumanMessage(content=question),
    ]

    # 6. START THE LOOP
    # We use a 'for' loop instead of 'while True' as a Circuit Breaker.
    # If the AI gets confused and loops forever, this forces it to stop at 10 to save money/CPU.
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        # STEP A: Let the AI Think. We send it the history and wait for a response.
        ai_message = llm_with_tools.invoke(messages)

        # Look to see if the AI asked to use a tool
        tool_calls = ai_message.tool_calls

        # STEP B: The Exit Condition
        # If tool_calls is empty, the AI didn't need a tool. It figured out the final answer!
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content

        # STEP C: Process the Tool Request
        # The AI asked for a tool. Let's get the details of what it wants.
        tool_call = tool_calls[0] 
        tool_name = tool_call.get("name")      # e.g., "get_product_price"
        tool_args = tool_call.get("args", {})  # e.g., {"product": "laptop"}
        tool_call_id = tool_call.get("id")     # A unique ID receipt for this specific request

        print(f"  [Tool Selected] {tool_name} with args: {tool_args}")

        # STEP D: Execute the Python Code
        # Look up the string name in our dictionary to find the actual Python function
        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")

        # Actually run the python code using the arguments the AI provided
        observation = tool_to_use.invoke(tool_args)

        print(f"  [Tool Result] {observation}")

        # STEP E: Update the Memory (Scratchpad)
        # First, save the AI's request to use the tool into history
        messages.append(ai_message)
        
        # Second, save the actual result of the tool into history.
        # We MUST use ToolMessage, and we MUST provide the exact same tool_call_id.
        # This tells the AI: "Here is the result for that specific tool you just asked for."
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )

        # The loop now restarts from the top, sending this updated history back to the AI!

    print("ERROR: Max iterations reached without a final answer")
    return None

if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")

```

---

## 3. Component Deep Dive

### A. The `@tool` Decorator (Reflection & Schema Generation)

In standard Python, a function is just a block of code. But LLMs cannot read Python. They require strict JSON schemas (like a Swagger/OpenAPI spec) to understand what tools exist.
**C#/Java Analogy:** Think of `@tool` exactly like C# `[Route]` or `[Function]` attributes. At runtime, LangChain uses reflection to inspect the function's signature and docstring, automatically compiling it into a JSON payload to send to the LLM.

### B. Binding Tools to the LLM (`bind_tools`)

**OOP Concept:** **Dependency Injection / Protocol Binding.** `bind_tools` does not execute anything. It simply modifies the HTTP client. It tells the LLM, *"When I send you a prompt, I am also attaching this list of available interfaces you are allowed to call."*

### C. The Tool Execution (Service Locator)

When the LLM returns a `tool_name` string, we must execute the corresponding function.

```python
tool_to_use = tools_dict.get(tool_name) 
observation = tool_to_use.invoke(tool_args)

```

**C#/Java Analogy:** This is a classic **Service Locator Pattern**. The LLM returns a string containing the name of the method. We look up that string in our registry (`tools_dict`), grab the actual function pointer, and invoke it using the LLM's arguments.

### D. The ToolMessage (State Reconciliation)

This is the critical handoff. We append the result of our local code execution to the `messages` array. We **must** wrap it in a `ToolMessage` and pass the exact `tool_call_id` that the LLM originally generated. This acts as the cryptographic "receipt" that proves to the LLM that we executed the exact tool it asked for.

---

## ⚠️ Production Notes (What Breaks & How to Fix It)

* **Tool ID Mismatches:** When an LLM requests a tool, it generates a unique ID (e.g., `call_abc123`). When you construct the `ToolMessage` to return the observation, the `tool_call_id` must match perfectly. If you forget this ID, or map it incorrectly, the LLM API will immediately throw an HTTP 400 Bad Request error because its state history is corrupted.
* **Infinite Loops:** Because LLMs are non-deterministic, they can get confused and call the same tool endlessly. You must **never** use a `while True:` loop in production AI. Always use a capped `for` loop (`MAX_ITERATIONS`) to act as a circuit breaker and prevent infinite API billing.
* **Stringification of Output:** The `ToolMessage(content=str(observation))` requires a string. If your local Python function returns a complex object, a C# `struct`, or a nested Dictionary, you must serialize it to JSON/String before appending it to the message history.

---

## 6. Interview Q&A Anchors

**Q: In LangChain, what is the exact purpose of the `ToolMessage` class?**

> **A:** `ToolMessage` is the state-reconciliation object used in the ReAct loop. When the LLM issues a tool call, the application executes the local code. The raw result of that code must be converted to a string, wrapped in a `ToolMessage`, and tagged with the LLM's original `tool_call_id`. This allows the LLM to explicitly link its request to the application's response in the context window.

**Q: Why do we write a `for` loop with a `MAX_ITERATIONS` constant instead of a `while` loop for our Agent?**

> **A:** As a strict safeguard against infinite execution loops. If the LLM hallucinates the wrong parameters, the tool might return an error string. The LLM might then stubbornly retry the exact same bad parameters indefinitely. Capping the loop ensures the execution safely aborts, protecting system resources and API budgets.

**Q: How does LangChain translate your backend code into something the LLM can trigger?**

> **A:** Through reflection and schema generation. By using the `@tool` decorator, LangChain parses the Python function's type hints and docstrings to generate an OpenAPI/JSON Schema. We inject this schema into the API payload via `bind_tools()`. The LLM reads this JSON schema, and when it requires the tool, it returns a matching JSON object requesting execution.