# 02. Chat Models: The Message Protocol & Architecture 🧠

*Based on Section 1: LangChain Basics (Lectures 9 - 11)*

## 🎯 What You Will Learn
* The architectural shift from Completion Models to Chat Models.
* The "Chat Markup Language" (System, Human, AI Roles).
* How to handle the inherently stateless nature of LLM APIs.
* Raw message construction vs. PromptTemplates.

## 📦 Dependency Setup
Run this exact command in your terminal using `uv`:
```bash
uv add langchain-core langchain-openai python-dotenv
```

---

## 1. Core Architectural Concepts

### Completion vs. Chat
This module defines the architectural shift from **Completion Models** (Legacy) to **Chat Models** (Modern).

* **Completion Models (LLMs):** Text-In $\to$ Text-Out. You give it a string ("The sky is"), and it completes it (" blue"). It has no concept of "talking."
* **Chat Models:** Messages-In $\to$ Message-Out. You give it a **List of Messages** with specific **Roles** (System, Human, AI). It outputs an `AIMessage`.

### C#/Java Analogy
* **Completion Model:** A pure static string manipulation function: `string response = StringUtils.CompleteText("The sky is ");`
* **Chat Model:** A REST API Endpoint that takes a JSON Array of Message DTOs. Because HTTP is fundamentally stateless, you must pass the entire session history array on every single request to simulate "memory".

---

## 2. Technical Deep Dive (Components)

### A. The Three Roles (The "Chat Markup Language")

Modern models (GPT-4, Claude, Gemini) are trained on structured dialogue, not just raw text. They expect inputs tagged with specific tokens indicating *who* is speaking.

| Role | Class | Purpose | Production Note |
| --- | --- | --- | --- |
| **System** | `SystemMessage` | Meta-instructions. Defines **who** the AI is and **how** it behaves. | **Security Critical.** This is where you place guardrails (e.g., "Do not reveal user PII"). |
| **Human** | `HumanMessage` | The user's input. | In production, strictly separate this from System instructions to prevent "Prompt Injection." |
| **AI** | `AIMessage` | The model's output. | Used to feed conversation history back into the model so it remembers the context. |

### B. "Stateless" Architecture

A critical concept for interviews: **LLMs are stateless.**

* The model does NOT remember you.
* When you send the second question ("How do I handle missing values?"), the model has already forgotten the first question.
* **Solution:** You must re-send the **entire chain** of messages (System + Human + AI + Human) every single time you call `.invoke()`. This is why the `messages` list in the code below is so important.

### C. `LLM` vs. `ChatModel` in LangChain

LangChain has two distinct class types:

1. **`LLM` classes (e.g., `OpenAI`):** Accept a string, return a string. (Deprecated for most modern uses).
2. **`ChatModel` classes (e.g., `ChatOpenAI`):** Accept `List[BaseMessage]`, return `AIMessage`.
* *Why it matters:* Even if a model is technically a text completion engine under the hood, the `ChatModel` wrapper handles the complex tokenization required to format the "System/Human" structure correctly for the API.

---

## 💻 Dual Examples

### 1. Course Project Implementation (General Chat API)
Eden Marco demonstrates passing a hardcoded array of `SystemMessage`, `HumanMessage`, and `AIMessage` objects directly to `llm.invoke()` to prove that memory is just an illusion created by appending previous outputs to the input list.

### 2. Generic / Real-World Implementation (Customer Support Bot)
In a production C# backend, you don't hardcode messages. You retrieve the user's `SessionId`, query Redis or a SQL database to reconstruct the `List<BaseMessage>` of their last 10 interactions, append their new `HumanMessage`, and send the entire payload to OpenAI to generate the next response.

---

## 3. Code Implementation: Raw Message Structures

*This script demonstrates how to interact with a Chat Model using raw message objects, revealing what actually happens inside a `ChatPromptTemplate`.*

```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# 1. Load Config
load_dotenv()

# 2. Initialize the Chat Model
llm = ChatOpenAI(temperature=0.7, model="gpt-4o")

# 3. Define a "Conversation History" (The Context)
# Unlike completion models, we don't just send a query.
# We send a structured list representing the state of the conversation.
messages = [
    # SYSTEM: The "God Mode" instruction. Sets behavior/persona.
    SystemMessage(content="You are a sarcastic senior engineer who loves Python."),
    
    # HUMAN: The user's first input.
    HumanMessage(content="I am writing a script to parse CSVs."),
    
    # AI: A 'fake' history. We inject this to give the model memory of what it 'said'.
    AIMessage(content="Oh, thrilling. Another CSV parser. Groundbreaking work."),
    
    # HUMAN: The user's follow-up question.
    HumanMessage(content="Hey, be nice! How do I handle missing values with pandas?")
]

# 4. Execution
# We pass the LIST of messages, not a single string.
response = llm.invoke(messages)

# 5. Output Analysis
print(f"Role: {type(response).__name__}") # Expect: AIMessage
print(f"Content: {response.content}")
```

### Code Breakdown:
1. **Line 15 (`messages = [...]`):** We are manually constructing the state array. In a real application, this array is built dynamically.
2. **Line 17 (`SystemMessage`):** The global configuration for the LLM. It dictates the rules of engagement for the entire session.
3. **Line 23 (`AIMessage`):** We are "faking" memory here by hardcoding what the AI supposedly said previously. This is exactly how actual memory modules work under the hood—they just append the last API response to this list before the next loop.
4. **Line 31 (`llm.invoke(messages)`):** Notice we pass the `List` object, not a concatenated string. LangChain handles converting these Python objects into the specific JSON schema OpenAI expects.

---

## ⚠️ Production Notes (What Breaks & How to Fix It)

* **Context Window Exhaustion:** If you continuously append messages to the list, you will eventually exceed the model's token limit (e.g., 128k tokens) and the API will throw an HTTP 400 error.
  * **The Fix:** Implement a "Sliding Window" (only send the last N messages), or use an LLM to periodically summarize the oldest messages into a single dense context string.
* **Role Contamination (Prompt Injection):** If user input is accidentally passed as a `SystemMessage`, the user gains "God Mode" over your application.
  * **The Fix:** Strictly type your inputs. Dynamic user strings must *always* and *only* be wrapped in a `HumanMessage` class.

---

## 4. Interview Q&A Anchors

**Q: What is the difference between a SystemMessage and a HumanMessage?**
> **A:** A `SystemMessage` is a high-level instruction that sets the behavior, persona, and constraints of the AI (the "Director's notes"). A `HumanMessage` is the dynamic input from the end-user. Separating them is crucial for security, as it allows the developer to define hard constraints that the user theoretically cannot override.

**Q: Why do we pass a list of messages instead of a single string?**
> **A:** Because modern models are fine-tuned on conversation data. Passing a list allows the model to distinguish between instructions (System), past context (AI), and current input (Human). It also enables "few-shot prompting" where we provide examples of good interactions (User/AI pairs) to guide the model's performance.

**Q: If I use `ChatOpenAI`, does it remember my previous messages automatically?**
> **A:** No. The model is stateless. The developer (or the LangChain `Memory` module) must manage the list of past messages and re-send the full history with every new request.

---