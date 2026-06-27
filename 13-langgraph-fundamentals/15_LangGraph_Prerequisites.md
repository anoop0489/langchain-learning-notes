# 15. LangGraph Prerequisites — Modern LangChain Patterns You Need First 🧱

*Pre-requisite knowledge for LangGraph, sourced from official LangChain docs (2025) + validated independently*

> ⚠️ **Why this exists:** Eden's course covers LangChain fundamentals (Sections 1-12), but LangGraph introduces newer patterns — `ToolRuntime`, `StateGraph`, `Command`, `init_chat_model`, middleware, and more — that didn't exist when the course was recorded. This doc bridges the gap.

---

## 📋 Table of Contents

| # | Topic | What You'll Learn |
|---|-------|-------------------|
| 1 | [LangGraph Overview](#1-langgraph-overview) | What LangGraph is, why it exists, how it relates to LangChain |
| 2 | [StateGraph & MessagesState](#2-stategraph--messagesstate) | The core graph primitives — nodes, edges, state |
| 3 | [init_chat_model — Universal Model Factory](#3-init_chat_model--universal-model-factory) | Provider-agnostic model initialization, parameters, retries |
| 4 | [Modern Tool Patterns](#4-modern-tool-patterns) | `@tool` decorator, reserved argument names, advanced schemas |
| 5 | [ToolRuntime — Access Context](#5-toolruntime--access-context) | State, context, store, stream writer, execution info |
| 6 | [Tool Return Values & Command](#6-tool-return-values--command) | String, object, multimodal, `Command` for state updates |
| 7 | [Structured Output](#7-structured-output) | `with_structured_output`, Pydantic, TypedDict, JSON Schema |
| 8 | [Middleware & Dynamic Tool Selection](#8-middleware--dynamic-tool-selection) | `wrap_model_call`, `wrap_tool_call`, filtering tools at runtime |
| 9 | [Streaming & Batch](#9-streaming--batch) | `.stream()`, `.astream_events()`, `.batch()`, chunk accumulation |
| 10 | [Persistence & Checkpointers](#10-persistence--checkpointers) | Durable execution, `thread_id`, resume after failures |
| 11 | [Interview Q&A Anchors](#11-interview-qa-anchors) | Quick-fire answers for all prerequisites |

---

## 📚 Key Definitions

| Term | Quick Recall | Full Definition |
|------|-------------|-----------------|
| **LangGraph** | Orchestration runtime for stateful agents | A low-level framework for building long-running, stateful workflows with persistence, human-in-the-loop, and streaming. Built on top of LangChain but usable independently. |
| **StateGraph** | A directed graph where nodes transform shared state | The core LangGraph primitive — you define nodes (functions), edges (transitions), and a shared state schema. Think of it as a state machine for AI workflows. |
| **MessagesState** | Pre-built state with a `messages` list | A convenience TypedDict with `messages: Annotated[list, add_messages]` — the default state for chatbot-style graphs. |
| **ToolRuntime** | Injected context object for tools | A parameter automatically injected into tools that provides access to state, context, store, stream writer, and execution info — hidden from the LLM schema. |
| **Command** | A tool return type that updates graph state | Allows tools to write back to graph state (e.g., set user preferences) rather than just returning text to the model. |
| **init_chat_model** | Provider-agnostic model factory | A single function that initializes any chat model from any provider using `"provider:model_name"` syntax. |
| **Middleware** | Hooks that wrap model/tool calls | Functions decorated with `@wrap_model_call` or `@wrap_tool_call` that intercept and modify requests/responses — used for dynamic tool selection, error handling, caching. |
| **Checkpointer** | Persistence layer for graph state | Saves graph state after each node execution so workflows can resume after crashes, handle long-running tasks, and support human-in-the-loop interrupts. |
| **thread_id** | Conversation scope identifier | A unique ID passed in `config["configurable"]["thread_id"]` that scopes a conversation's message history and checkpoints. Like a session ID. |
| **Reducer** | Conflict resolution for state updates | A function that determines how to merge concurrent updates to the same state field (e.g., `add_messages` appends instead of replacing). |

---

## 1. LangGraph Overview

### What Is LangGraph?

LangGraph is a **low-level orchestration runtime** for building stateful, long-running AI agents and workflows. It sits *above* LangChain (which provides model/tool integrations) but *below* high-level agent harnesses.

```
┌─────────────────────────────────────────────────────┐
│  Deep Agents SDK (Harness — planning, subagents)    │  ← Highest level
├─────────────────────────────────────────────────────┤
│  LangGraph (Runtime — state, persistence, HIL)      │  ← This section
├─────────────────────────────────────────────────────┤
│  LangChain (Framework — models, tools, agents)      │  ← Sections 1-12
├─────────────────────────────────────────────────────┤
│  LangSmith (Platform — tracing, eval, deploy)       │  ← Observability
└─────────────────────────────────────────────────────┘
```

### Why LangGraph Exists (Problems It Solves)

| Problem | LangChain Alone | LangGraph Solution |
|---------|----------------|-------------------|
| Agent crashes mid-task | Lost all progress, start over | **Checkpointers** persist state — resume from last node |
| Need human approval before action | Not supported natively | **Interrupts** pause graph, wait for human, resume |
| Agent loop runs forever | Hard to detect/stop | **Recursion limits** + state inspection at each step |
| Multiple agents collaborating | Manual orchestration | **Subgraphs** — compose agents as nodes in a parent graph |
| Conversation memory across sessions | Manual DB integration | **Thread-scoped persistence** — automatic with `thread_id` |
| Need different logic paths | Simple if/else in chains | **Conditional edges** — route based on state |

### C# Analogy

Think of LangGraph as **Durable Functions (Azure)** or **Temporal.io workflows** for AI:
- Each **node** = an activity/step function
- The **state** = the workflow context object
- **Checkpointers** = the durable execution storage
- **Interrupts** = the `WaitForExternalEvent` / human approval pattern
- **Conditional edges** = the orchestrator's `switch` logic

### The Hello World

```python
from langgraph.graph import StateGraph, MessagesState, START, END

def mock_llm(state: MessagesState):
	return {"messages": [{"role": "ai", "content": "hello world"}]}

graph = StateGraph(MessagesState)
graph.add_node(mock_llm)
graph.add_edge(START, "mock_llm")
graph.add_edge("mock_llm", END)
app = graph.compile()

result = app.invoke({"messages": [{"role": "user", "content": "hi!"}]})
```

---

## 2. StateGraph & MessagesState

### StateGraph — The Core Primitive

A `StateGraph` is a directed graph where:
- **Nodes** are Python functions that receive state and return state updates
- **Edges** define the execution flow between nodes
- **State** is a shared TypedDict that flows through the entire graph

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

# Step 1: Define your state schema
class MyState(TypedDict):
	messages: Annotated[list, add_messages]  # Reducer: append, don't replace
	user_name: str                            # Custom field

# Step 2: Define node functions (receive state, return partial updates)
def greet(state: MyState):
	name = state.get("user_name", "stranger")
	return {"messages": [{"role": "ai", "content": f"Hello, {name}!"}]}

# Step 3: Build the graph
graph = StateGraph(MyState)
graph.add_node("greet", greet)
graph.add_edge(START, "greet")
graph.add_edge("greet", END)

# Step 4: Compile and run
app = graph.compile()
result = app.invoke({"messages": [], "user_name": "Anoop"})
```

### MessagesState — The Pre-Built Default

For chatbot-style agents, LangGraph provides `MessagesState` so you don't have to define the messages field yourself:

```python
from langgraph.graph import MessagesState

# This is equivalent to:
# class MessagesState(TypedDict):
#     messages: Annotated[list[AnyMessage], add_messages]
```

### Reducers — Why `Annotated[list, add_messages]`?

A **reducer** tells LangGraph how to merge a node's return value into existing state. Without it, returning `{"messages": [...]}` would *replace* the entire list. With `add_messages`, it *appends*.

| Reducer | Behavior | Use Case |
|---------|----------|----------|
| `add_messages` | Appends new messages to existing list | Chat history (default) |
| None (no annotation) | Replaces the field entirely | Simple values like `user_name` |
| Custom function | Your logic for merging | Counters, sets, complex objects |

### C# Analogy

| LangGraph | C# Equivalent |
|-----------|---------------|
| `StateGraph` | A state machine (`Stateless` library) or Durable Functions orchestrator |
| `TypedDict` state | A POCO/DTO class with public properties |
| Node function | An Activity function in Durable Functions |
| `add_messages` reducer | `List<T>.AddRange()` vs assignment |
| `START` / `END` | Entry point / terminal state |
| Conditional edge | `switch` statement in orchestrator |

---

## 3. init_chat_model — Universal Model Factory

### The Problem It Solves

Before `init_chat_model`, switching providers meant changing imports and class names:

```python
# Old way — tightly coupled to provider
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatOpenAI(model="gpt-4o")  # ← Locked to OpenAI
```

### The Modern Way

```python
from langchain.chat_models import init_chat_model

# Provider-agnostic — switch by changing a string
model = init_chat_model("gpt-4o")                          # OpenAI (auto-detected)
model = init_chat_model("anthropic:claude-sonnet-4-6")     # Anthropic
model = init_chat_model("google_genai:gemini-2.5-flash")   # Google
model = init_chat_model("ollama:qwen3:1.7b")              # Local Ollama
```

### Key Parameters

| Parameter | Type | Purpose |
|-----------|------|---------|
| `model` | str | Model name, optionally prefixed with `"provider:"` |
| `temperature` | float | Randomness (0 = deterministic, 1 = creative) |
| `max_tokens` | int | Maximum output length |
| `timeout` | int | Seconds before request times out |
| `max_retries` | int | Retry count for failures (default: 6) |
| `api_key` | str | Provider API key (usually from env var) |

### Connection Resilience (Built-In Retries)

LangChain chat models automatically retry with exponential backoff:

```python
model = init_chat_model(
	"gpt-4o",
	max_retries=10,   # Increase for unreliable networks (default: 6)
	timeout=120,      # Seconds before timeout
)
```

**What gets retried:** Network errors, rate limits (429), server errors (5xx)
**What doesn't:** Client errors (401 unauthorized, 404 not found)

### Configurable Models (Runtime Provider Switching)

```python
# No model specified at creation — pick at runtime
configurable_model = init_chat_model(temperature=0)

# Same code, different model each time
configurable_model.invoke("hello", config={"configurable": {"model": "gpt-4o"}})
configurable_model.invoke("hello", config={"configurable": {"model": "claude-sonnet-4-6"}})
```

### C# Analogy

`init_chat_model` is like a **factory pattern** with dependency injection:

```csharp
// C# equivalent concept
IChatModel model = ChatModelFactory.Create("openai:gpt-4o", new ModelOptions {
	Temperature = 0,
	MaxRetries = 10,
	Timeout = TimeSpan.FromSeconds(120)
});
```

Or like `HttpClientFactory` — you don't `new HttpClient()` directly, you use the factory for configuration, resilience, and lifecycle management.

---

## 4. Modern Tool Patterns

### The `@tool` Decorator

The standard way to create tools in LangChain:

```python
from langchain.tools import tool

@tool
def get_weather(location: str) -> str:
	"""Get the weather at a location."""  # ← Docstring becomes the tool description
	return f"It's sunny in {location}."
```

**Key rules:**
- Type hints are **required** (they define the input schema the LLM sees)
- The docstring is the tool description the model reads to decide when to use it
- Prefer `snake_case` names (some providers reject spaces/special chars)

### Custom Name & Description

```python
@tool("web_search", description="Search the internet for current information. Use for any factual queries.")
def search(query: str) -> str:
	"""Search the web."""
	return f"Results for: {query}"
```

### Advanced Schema with Pydantic

For complex inputs with validation and descriptions:

```python
from pydantic import BaseModel, Field
from typing import Literal

class WeatherInput(BaseModel):
	"""Input for weather queries."""
	location: str = Field(description="City name or coordinates")
	units: Literal["celsius", "fahrenheit"] = Field(default="celsius")

@tool(args_schema=WeatherInput)
def get_weather(location: str, units: str = "celsius") -> str:
	"""Get current weather."""
	return f"22°C in {location}"
```

### ⚠️ Reserved Argument Names

These parameter names **cannot** be used as tool arguments — they cause runtime errors:

| Reserved Name | Why It's Reserved |
|--------------|-------------------|
| `config` | Used internally to pass `RunnableConfig` to tools |
| `runtime` | Reserved for `ToolRuntime` parameter (state, context, store access) |

```python
# ❌ BAD — will crash at runtime
@tool
def bad_tool(config: str) -> str:  # "config" is reserved!
	return config

# ✅ GOOD — use a different name
@tool
def good_tool(configuration: str) -> str:
	return configuration
```

### C# Analogy

| LangChain Tool | C# Equivalent |
|---------------|---------------|
| `@tool` decorator | `[Description("...")]` attribute on a method |
| `args_schema` (Pydantic) | A DTO class with `[Required]` and `[Description]` attributes |
| Reserved `config`/`runtime` | Like `HttpContext` in ASP.NET — injected by the framework, not by the caller |
| Tool name | The method name registered in a DI container |

---

## 5. ToolRuntime — Access Context

### What Is ToolRuntime?

`ToolRuntime` is an **automatically injected** parameter that gives your tool access to the execution context. The LLM never sees it — it's hidden from the schema.

```python
from langchain.tools import tool, ToolRuntime

@tool
def my_tool(query: str, runtime: ToolRuntime) -> str:
	"""Search something."""
	# LLM only sees: my_tool(query: str)
	# But your code has access to runtime.state, runtime.context, etc.
	return f"Results for {query}"
```

### What ToolRuntime Provides

```
┌────────────────────────────────────────────────────────────┐
│                      ToolRuntime                             │
├────────────────────────────────────────────────────────────┤
│  .state            → Short-term memory (current messages,   │
│                      custom fields for this conversation)   │
│  .context          → Immutable config (user_id, session)    │
│  .store            → Long-term memory (persists across      │
│                      conversations — like a database)        │
│  .stream_writer    → Emit real-time progress updates        │
│  .execution_info   → thread_id, run_id, attempt number     │
│  .server_info      → Assistant/graph ID (LangGraph Server)  │
│  .tool_call_id     → Unique ID for this specific call       │
│  .config           → Full RunnableConfig                    │
└────────────────────────────────────────────────────────────┘
```

### 5.1 State (Short-Term Memory)

State is the **current conversation's data** — messages, custom fields, counters. It exists only for the duration of a thread.

```python
from langchain.tools import tool, ToolRuntime

@tool
def get_message_count(runtime: ToolRuntime) -> str:
	"""Get the number of messages in the conversation."""
	messages = runtime.state["messages"]
	return f"There are {len(messages)} messages in this conversation."

@tool
def get_user_preference(pref_name: str, runtime: ToolRuntime) -> str:
	"""Get a user preference from the current session state."""
	preferences = runtime.state.get("user_preferences", {})
	return preferences.get(pref_name, "Not set")
```

### 5.2 Context (Immutable Per-Run Configuration)

Context carries **per-invocation data** like user ID, session info, or permissions. It's set when you call `.invoke()` and cannot be changed during execution.

```python
from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime

@dataclass
class UserContext:
	user_id: str
	role: str  # "admin", "editor", "viewer"

@tool
def get_account_info(runtime: ToolRuntime[UserContext]) -> str:
	"""Get the current user's account information."""
	user_id = runtime.context.user_id  # ← Typed access
	role = runtime.context.role
	return f"User {user_id} with role {role}"

# When invoking the agent:
# agent.invoke(..., context=UserContext(user_id="abc123", role="admin"))
```

**Key distinction:**
- `thread_id` scopes the **conversation** (message history, checkpoints)
- `context` carries **per-run data** your tools read at invocation time

### 5.3 Store (Long-Term Memory)

Store is **persistent storage** that survives across conversations. Unlike state (which is per-thread), store data persists indefinitely.

```python
from langchain.tools import tool, ToolRuntime

@tool
def save_user_preference(key: str, value: str, runtime: ToolRuntime) -> str:
	"""Save a user preference that persists across conversations."""
	store = runtime.store
	user_id = runtime.context.user_id

	# Namespace pattern: ("users", user_id) + key
	store.put(("preferences", user_id), key, {"value": value})
	return f"Saved {key}={value} for future conversations."

@tool
def get_user_preference(key: str, runtime: ToolRuntime) -> str:
	"""Retrieve a previously saved user preference."""
	store = runtime.store
	user_id = runtime.context.user_id

	item = store.get(("preferences", user_id), key)
	return item.value["value"] if item else "No preference saved."
```

| Memory Type | Scope | Lifetime | Analogy |
|-------------|-------|----------|---------|
| **State** | Per-thread (conversation) | Until thread is deleted | `HttpContext.Items` / session state |
| **Context** | Per-invocation | Single request | `HttpContext.User` / JWT claims |
| **Store** | Global (cross-thread) | Persistent (until deleted) | Database / Redis cache |

### 5.4 Stream Writer (Real-Time Progress)

Emit updates to the client while a tool is executing:

```python
@tool
def long_running_search(query: str, runtime: ToolRuntime) -> str:
	"""Search multiple databases (takes time)."""
	writer = runtime.stream_writer

	writer("Searching primary database...")
	# ... actual work ...
	writer("Searching secondary database...")
	# ... actual work ...
	writer("Compiling results...")

	return "Found 42 matching documents."
```

### 5.5 Execution Info

Access thread/run identity and retry state:

```python
@tool
def debug_tool(runtime: ToolRuntime) -> str:
	"""Log execution identity."""
	info = runtime.execution_info
	return f"Thread: {info.thread_id}, Run: {info.run_id}, Attempt: {info.node_attempt}"
```

### C# Analogy for ToolRuntime

| ToolRuntime Component | C# Equivalent |
|----------------------|---------------|
| `runtime.state` | `HttpContext.Items` or scoped service state |
| `runtime.context` | `HttpContext.User.Claims` (JWT claims, injected per-request) |
| `runtime.store` | `IDistributedCache` / EF Core `DbContext` |
| `runtime.stream_writer` | `IServerSentEventsWriter` / SignalR `IHubContext.SendAsync()` |
| `runtime.execution_info` | `Activity.Current` (OpenTelemetry trace/span info) |
| `runtime.tool_call_id` | Correlation ID from a message broker |

---

## 6. Tool Return Values & Command

### Return a String (Most Common)

The model sees this as text and decides what to do next:

```python
@tool
def get_weather(city: str) -> str:
	"""Get weather for a city."""
	return f"It is currently sunny in {city}."
# Model sees: ToolMessage(content="It is currently sunny in Boston.")
```

### Return an Object (Structured Data)

The model can reason over specific fields:

```python
@tool
def get_weather_data(city: str) -> dict:
	"""Get structured weather data."""
	return {"city": city, "temperature_c": 22, "conditions": "sunny"}
```

### Return Multimodal Content

Tools can return images, audio, etc. to vision-capable models:

```python
@tool
def capture_screenshot() -> list[dict]:
	"""Capture a screenshot."""
	return [
		{"type": "text", "text": "Screenshot of the current page:"},
		{"type": "image", "url": "https://example.com/page.png"},
	]
```

### Return a Command (Update Graph State)

**This is the key new pattern.** When a tool needs to *write back* to the graph's state (not just return text to the model), it returns a `Command`:

```python
from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.types import Command

@tool
def set_language(language: str, runtime: ToolRuntime) -> Command:
	"""Set the user's preferred response language."""
	return Command(
		update={
			"preferred_language": language,       # ← Updates graph state field
			"messages": [
				ToolMessage(
					content=f"Language set to {language}.",
					tool_call_id=runtime.tool_call_id,  # ← Required correlation ID
				)
			],
		}
	)
```

**Why Command exists:**
- Normal string returns → model sees text, state unchanged
- Command → state is mutated (e.g., setting preferences, flags, counters)
- The `ToolMessage` inside `Command` is optional but recommended so the model knows the action succeeded

### Return Direct (Skip Model Post-Processing)

When the tool's output IS the final answer — no further LLM reasoning needed:

```python
@tool(return_direct=True)
def fetch_order_status(order_id: str) -> str:
	"""Fetch the current status of a customer order."""
	return f"Order {order_id} is shipped and will arrive in 2 days."
# Agent returns this directly to the user — no extra LLM call
```

**Use when:** The tool output is the complete, user-ready answer (e.g., database lookup).
**Don't use when:** The result needs summarization, further reasoning, or chaining.

### C# Analogy

| Tool Return | C# Equivalent |
|-------------|---------------|
| String return | An endpoint returning `Ok("result")` |
| Dict return | An endpoint returning `Ok(new { temp = 22 })` |
| `Command` | An endpoint that also writes to `HttpContext.Items` or dispatches a domain event |
| `return_direct=True` | A short-circuit middleware that returns a cached response without hitting the controller |

---

## 7. Structured Output

### The Problem

LLMs return free-form text. But you often need **structured data** (JSON matching a schema):

```python
# Without structured output:
response = model.invoke("Extract the movie title and year from: 'Inception came out in 2010'")
# → "The movie is Inception and it was released in 2010."  ← Free text, hard to parse
```

### The Solution: `with_structured_output`

```python
from pydantic import BaseModel, Field

class Movie(BaseModel):
	"""A movie with details."""
	title: str = Field(description="The title of the movie")
	year: int = Field(description="The year released")
	director: str = Field(description="The director")

model_with_structure = model.with_structured_output(Movie)
response = model_with_structure.invoke("Tell me about Inception")
# → Movie(title="Inception", year=2010, director="Christopher Nolan")  ← Typed object!
```

### Three Schema Formats

| Format | Best For | Validation |
|--------|----------|-----------|
| **Pydantic BaseModel** | Production code — rich validation, descriptions, nesting | Automatic |
| **TypedDict** | Simple cases — no runtime validation needed | Manual |
| **JSON Schema** (dict) | Max control, interoperability with other systems | Manual |

### Methods (How It Works Under the Hood)

| Method | How It Constrains | Provider Support |
|--------|-------------------|-----------------|
| `json_schema` | Provider's native structured output feature | OpenAI, Anthropic, Google |
| `function_calling` | Forces a tool call matching the schema | Most providers |
| `json_mode` | Generates valid JSON (schema in prompt) | Legacy — less reliable |

### C# Analogy

`with_structured_output` is like using `System.Text.Json` deserialization with a schema contract:

```csharp
// C# equivalent concept
var movie = await chatClient.GetStructuredResponseAsync<Movie>(
	"Tell me about Inception",
	JsonSchema.FromType<Movie>()
);
// movie.Title → "Inception", movie.Year → 2010
```

Or in Semantic Kernel: `kernel.InvokeAsync<Movie>(...)` with a schema constraint.

---

## 8. Middleware & Dynamic Tool Selection

### What Is Middleware?

Middleware are **hooks** that wrap model calls or tool calls, letting you intercept and modify behavior. Two types:

| Decorator | Wraps | Use Cases |
|-----------|-------|-----------|
| `@wrap_model_call` | The LLM invocation | Dynamic tool filtering, model switching, logging |
| `@wrap_tool_call` | Tool execution | Error handling, retries, authorization |

### wrap_model_call — Dynamic Tool Selection

Filter which tools the model sees based on state, permissions, or context:

```python
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse

@wrap_model_call
def permission_filter(request: ModelRequest, handler) -> ModelResponse:
	"""Only show admin tools to admin users."""
	user_role = request.runtime.context.role

	if user_role != "admin":
		# Filter out admin-only tools
		safe_tools = [t for t in request.tools if not t.name.startswith("admin_")]
		request = request.override(tools=safe_tools)

	return handler(request)  # ← Pass to next middleware or model
```

### wrap_tool_call — Error Handling

Catch tool exceptions and return friendly messages:

```python
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage
from langchain.tools.tool_node import ToolCallRequest

@wrap_tool_call
def handle_errors(request: ToolCallRequest, handler) -> ToolMessage:
	"""Convert exceptions into ToolMessages the model can handle."""
	try:
		return handler(request)
	except Exception as e:
		return ToolMessage(
			content=f"Tool error: {e}. Please try a different approach.",
			tool_call_id=request.tool_call["id"],
		)
```

### Dynamic Model Selection

Switch models at runtime based on complexity:

```python
from langchain_openai import ChatOpenAI

basic = ChatOpenAI(model="gpt-4o-mini")
advanced = ChatOpenAI(model="gpt-4o")

@wrap_model_call
def route_by_complexity(request: ModelRequest, handler) -> ModelResponse:
	"""Use advanced model for complex conversations."""
	if len(request.state["messages"]) > 10:
		return handler(request.override(model=advanced))
	return handler(request.override(model=basic))
```

### C# Analogy

| Middleware Concept | C# Equivalent |
|-------------------|---------------|
| `@wrap_model_call` | ASP.NET middleware pipeline / `DelegatingHandler` |
| `@wrap_tool_call` | Exception filter / `IAsyncActionFilter` |
| `handler(request)` | `next(context)` in the middleware pipeline |
| `request.override(...)` | Mutating `HttpContext` before passing downstream |
| Dynamic tool filtering | Authorization policy removing endpoints from route table |

---

## 9. Streaming & Batch

### Stream (Real-Time Token Output)

```python
# Synchronous streaming
for chunk in model.stream("Why do parrots talk?"):
	print(chunk.text, end="", flush=True)

# Async streaming
async for chunk in model.astream("Why do parrots talk?"):
	print(chunk.text, end="", flush=True)
```

### Chunk Accumulation

Chunks can be summed to reconstruct the full message:

```python
full = None
for chunk in model.stream("What color is the sky?"):
	full = chunk if full is None else full + chunk
# full is now equivalent to what .invoke() would return
```

### Streaming Events (Advanced)

Filter by event type for fine-grained control:

```python
async for event in model.astream_events("Hello"):
	if event["event"] == "on_chat_model_stream":
		print(event["data"]["chunk"].text, end="")
	elif event["event"] == "on_chat_model_end":
		print(f"\n\nFull: {event['data']['output'].text}")
```

### Batch (Parallel Requests)

```python
responses = model.batch([
	"Why do parrots talk?",
	"How do airplanes fly?",
	"What is quantum computing?"
])

# Control parallelism
responses = model.batch(
	list_of_inputs,
	config={"max_concurrency": 5}
)
```

### C# Analogy

| LangChain | C# Equivalent |
|-----------|---------------|
| `.stream()` | `IAsyncEnumerable<T>` / `await foreach` |
| `.astream()` | `IAsyncEnumerable<T>` with `async` context |
| Chunk accumulation (`full + chunk`) | `StringBuilder.Append()` pattern |
| `.batch()` | `Task.WhenAll(requests.Select(r => client.SendAsync(r)))` |
| `max_concurrency` | `SemaphoreSlim` / `Parallel.ForEachAsync(maxDegreeOfParallelism)` |
| `astream_events()` | Server-Sent Events / `IObservable<T>` (Rx) |

---

## 10. Persistence & Checkpointers

### The Problem

Without persistence, if an agent crashes mid-task or needs human approval, all progress is lost.

### The Solution: Checkpointers

A **checkpointer** saves the graph's state after each node execution:

```python
from langgraph.checkpoint.memory import MemorySaver

# In-memory (development/testing)
checkpointer = MemorySaver()

app = graph.compile(checkpointer=checkpointer)

# Every invocation needs a thread_id to scope the conversation
result = app.invoke(
	{"messages": [{"role": "user", "content": "hello"}]},
	config={"configurable": {"thread_id": "conversation-123"}}
)
```

### thread_id — Conversation Scope

`thread_id` is how LangGraph knows which conversation you're continuing:

```python
# First turn
app.invoke({"messages": [...]}, config={"configurable": {"thread_id": "abc"}})

# Second turn — same thread_id, so it has the previous messages
app.invoke({"messages": [{"role": "user", "content": "follow up"}]},
		   config={"configurable": {"thread_id": "abc"}})

# Different conversation — different thread_id
app.invoke({"messages": [...]}, config={"configurable": {"thread_id": "xyz"}})
```

### Production Checkpointers

| Checkpointer | Use Case |
|--------------|----------|
| `MemorySaver` | Development, testing (lost on restart) |
| `PostgresSaver` | Production — durable, scalable |
| `SqliteSaver` | Single-machine production |

### Human-in-the-Loop (Interrupts)

Checkpointers enable **interrupts** — pausing a graph to wait for human input:

```python
# Graph pauses before "dangerous_action" node
app = graph.compile(
	checkpointer=checkpointer,
	interrupt_before=["dangerous_action"]
)

# Invoke — graph runs until the interrupt point, then stops
result = app.invoke({"messages": [...]}, config={"configurable": {"thread_id": "abc"}})
# result.status → "interrupted"

# Human reviews, then resumes
app.invoke(None, config={"configurable": {"thread_id": "abc"}})  # None = continue
```

### C# Analogy

| LangGraph | C# Equivalent |
|-----------|---------------|
| Checkpointer | Azure Durable Functions `IDurableOrchestrationContext` |
| `thread_id` | `instanceId` in Durable Functions |
| `MemorySaver` | In-memory state (test double) |
| `PostgresSaver` | Azure Table Storage / SQL persistence provider |
| `interrupt_before` | `WaitForExternalEvent("approval")` in Durable Functions |
| Resume after interrupt | `RaiseEventAsync("approval", payload)` |

---

## 11. Interview Q&A Anchors

**Q: What is LangGraph and how does it differ from LangChain?**
> **A:** LangChain provides model/tool integrations and agent abstractions (the "what" — models, prompts, tools). LangGraph is the orchestration runtime (the "how" — state management, persistence, human-in-the-loop, streaming). You can use LangGraph without LangChain, but they complement each other. Think of LangChain as the .NET SDK libraries and LangGraph as the ASP.NET runtime.

**Q: What is `ToolRuntime` and why does it exist?**
> **A:** `ToolRuntime` is an injected parameter that gives tools access to execution context — state (conversation messages), context (user identity), store (persistent memory), and stream writer (progress updates). It's hidden from the LLM schema so the model doesn't see it. It exists because tools often need more than just their arguments — they need to know who's calling, what the conversation history is, and where to persist data.

**Q: What are the reserved argument names for LangChain tools?**
> **A:** `config` and `runtime` are reserved. Using them as tool parameter names causes runtime errors because LangChain uses them internally for dependency injection. If you need to accept "config" data, name the parameter something else like `configuration` or `settings`.

**Q: What's the difference between State, Context, and Store in ToolRuntime?**
> **A:** State is short-term memory scoped to the current conversation thread (messages, custom fields). Context is immutable per-invocation configuration (user ID, permissions) — think JWT claims. Store is long-term persistent memory that survives across conversations — think database. A tool might read the user's context to know who they are, check state for conversation history, and write to store to remember preferences for next time.

**Q: What is a `Command` return type and when would you use it?**
> **A:** `Command` is a tool return type that updates graph state beyond just returning text. Use it when a tool needs to set flags, preferences, or counters in the graph's state — not just return information to the model. For example, a "set_language" tool should update the `preferred_language` field in state so subsequent nodes can use it, not just tell the model "language was set."

**Q: How does `init_chat_model` improve over direct class instantiation?**
> **A:** It's a provider-agnostic factory — you switch providers by changing a string (`"openai:gpt-4o"` → `"anthropic:claude-sonnet-4-6"`) without changing imports. It also supports runtime configurability (pick model per-request via `config`), consistent parameter handling across providers, and built-in retry/timeout configuration.

**Q: What is a Checkpointer and why is `thread_id` required?**
> **A:** A Checkpointer persists graph state after each node execution, enabling resume-after-crash, human-in-the-loop interrupts, and multi-turn conversations. `thread_id` is the scope key — it tells the checkpointer which conversation's state to load/save. Without it, the graph wouldn't know which user's conversation to resume. It's like a session ID in web apps.

**Q: How does middleware work in LangChain agents?**
> **A:** Middleware wraps model calls (`@wrap_model_call`) or tool calls (`@wrap_tool_call`) with interceptor logic. `wrap_model_call` can filter tools, switch models, or log requests before they hit the LLM. `wrap_tool_call` can catch errors, add retries, or gate execution. They compose — multiple middleware run in order, each calling `handler(request)` to pass to the next one. It's the same pipeline pattern as ASP.NET middleware.

**Q: What does `return_direct=True` do on a tool?**
> **A:** It short-circuits the agent loop — the tool's output goes directly to the user without another LLM call. Use it when the tool output IS the final answer (e.g., a database lookup returning a formatted status). Don't use it when the output needs summarization or further reasoning.

**Q: How does structured output work in LangChain?**
> **A:** `model.with_structured_output(Schema)` constrains the model to output data matching your schema (Pydantic model, TypedDict, or JSON Schema). Under the hood, it uses the provider's native structured output API or forces a tool call matching the schema. The result is a typed object you can use directly in code — no regex parsing needed.

---

## References

- [LangGraph Overview](https://docs.langchain.com/oss/python/langgraph/overview)
- [LangChain Models](https://docs.langchain.com/oss/python/langchain/models)
- [LangChain Tools](https://docs.langchain.com/oss/python/langchain/tools)
- [LangGraph Persistence](https://docs.langchain.com/oss/python/langgraph/persistence)
- [LangGraph Interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts)
- [LangChain Middleware](https://docs.langchain.com/oss/python/langchain/middleware)
- [LangChain Structured Output](https://docs.langchain.com/oss/python/langchain/structured-output)
- [LangChain Messages](https://docs.langchain.com/oss/python/langchain/messages)
