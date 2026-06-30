# 19. Reflexion Agent — Tool-Augmented Self-Improvement with LangGraph

> **Context:** Section 15, Chapters 103-110. Eden builds a Reflexion agent that extends the basic reflection pattern (Section 14) with tool use — fetching real-time data via Tavily to ground revisions in facts and citations.

---

## Table of Contents

| # | Section | What You'll Learn |
|---|---------|-------------------|
| 1 | [What Is a Reflexion Agent?](#1-what-is-a-reflexion-agent) | How it extends basic reflection with tools and structured output |
| 2 | [Architecture Overview](#2-architecture-overview) | Three nodes, one cycle, tool-augmented revision |
| 3 | [The Structured Output Trick](#3-the-structured-output-trick) | Forcing the LLM to produce answer + critique + queries in one call |
| 4 | [The Graph Structure](#4-the-graph-structure) | Nodes, edges, state, stop condition |
| 5 | [Advanced Prompt Engineering](#5-advanced-prompt-engineering) | How the prompt forces self-critique and improvement |
| 6 | [The ToolNode Name-Matching Trick](#6-the-toolnode-name-matching-trick) | Why tools are named after Pydantic schemas |
| 7 | [Comparison: Reflection vs Reflexion](#7-comparison-reflection-vs-reflexion) | Side-by-side with Section 14 |
| 8 | [Code Walkthrough](#8-code-walkthrough) | Step-by-step through all 4 files |
| 9 | [LangGraph Workflow Patterns](#9-langgraph-workflow-patterns) | Prompt chaining, parallelization, routing, orchestrator-worker, evaluator-optimizer |
| 10 | [ToolNode Deep Dive](#10-toolnode-deep-dive) | How ToolNode works, input/output formats, error handling, ToolRuntime |
| 11 | [Interview Q&A Anchors](#interview-qa-anchors) | Quick-fire answers |

---

## Key Definitions

| Term | Quick Recall | Full Definition |
|------|-------------|----------------|
| **Reflexion** | Reflection + tools + citations | An agent architecture from the Reflexion paper (Shinn et al., 2023) that combines self-critique with tool execution to iteratively improve answers using external data. |
| **Structured Output** | LLM forced to return a schema | Using `tool_choice` to force the LLM to always return data matching a Pydantic model — no freeform text allowed. |
| **AnswerQuestion** | Schema for initial response | Pydantic model: answer + self-critique (Reflection) + search queries. |
| **ReviseAnswer** | Schema for revised response | Extends AnswerQuestion with a `references` field for citations. |
| **tool_choice** | Force a specific tool call | Parameter that tells the LLM "you MUST call this tool" — eliminates the option to respond with plain text. |
| **ToolNode** | Prebuilt node that executes tools | LangGraph's built-in node that looks up tools by name and executes them, returning ToolMessages. |
| **Actor Prompt** | Shared template for both chains | One prompt template with a variable `{first_instruction}` — instantiated differently for draft vs revise. |
| **Prompt Chaining** | Linear A → B → C | Each LLM call processes the output of the previous one. No branching, no cycles. |
| **Parallelization** | Multiple LLM calls at once | Independent subtasks run simultaneously, results aggregated. |
| **Routing** | LLM picks a branch | LLM classifies input and routes to specialized handlers. No cycles. |
| **Orchestrator-Worker** | Plan → spawn → synthesize | One LLM plans subtasks, dynamically spawns workers via Send API, collects results. |
| **Evaluator-Optimizer** | Generate ↔ Evaluate loop | One LLM generates, another evaluates. Cycle until quality threshold met. |
| **Send API** | Dynamic worker spawning | LangGraph API that creates node executions at runtime based on state content. |
| **ToolRuntime** | State injection for tools | Lets tools access graph state and run-scoped context that the LLM didn't generate. |

---

## 1. What Is a Reflexion Agent?

The Reflexion agent extends the basic reflection pattern (Section 14) with **three key additions**:

1. **Tools** — Uses Tavily search to fetch real-time data
2. **Structured output** — Forces the LLM to return answer + critique + search queries in one call
3. **Citations** — Revised answers must include numbered references to source URLs

```
┌─────────────────────────────────────────────────────────────┐
│              REFLEXION vs REFLECTION                         │
│                                                             │
│  Section 14 (Reflection):                                   │
│    Generate ←→ Reflect (pure LLM-to-LLM, no tools)         │
│                                                             │
│  Section 15 (Reflexion):                                    │
│    Draft → Search → Revise → Search → Revise → END         │
│    (LLM + tools + citations + structured output)            │
└─────────────────────────────────────────────────────────────┘
```

**Paper:** [Reflexion: Language Agents with Verbal Reinforcement Learning (Shinn et al., 2023)](https://arxiv.org/pdf/2303.11366)

**LangChain blog:** [Reflection Agents](https://www.langchain.com/blog/reflection-agents)

---

## 2. Architecture Overview

### Mermaid Diagram (from Eden's project)

```mermaid
---
config:
  flowchart:
	curve: linear
---
graph LR;
	__start__([__start__]):::first
	draft(draft)
	execute_tools(execute_tools)
	revise(revise)
	__end__([__end__]):::last
	__start__ --> draft;
	draft --> execute_tools;
	execute_tools --> revise;
	revise -.-> execute_tools;
	revise -.-> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```

### ASCII Equivalent

```
┌───────┐    ┌───────────────┐    ┌────────┐
│ DRAFT │───▶│ EXECUTE_TOOLS │───▶│ REVISE │
└───────┘    └───────────────┘    └────┬───┘
					▲                   │
					│    (≤ MAX_ITER)   │
					└───────────────────┘
										│ (> MAX_ITER)
										▼
									 [END]
```

### What Each Node Does

| Node | Input | Output | Purpose |
|------|-------|--------|---------|
| **draft** | User question | AIMessage with tool_calls (AnswerQuestion) | Initial answer + critique + search queries |
| **execute_tools** | AIMessage with search_queries | ToolMessage with search results | Fetch real-time data from Tavily |
| **revise** | All messages so far | AIMessage with tool_calls (ReviseAnswer) | Improved answer + new critique + new queries + citations |

---

## 3. The Structured Output Trick

The most powerful technique in this agent: **the LLM is forced to self-critique as part of its output format**.

```python
class AnswerQuestion(BaseModel):
	answer: str           # The actual ~250 word answer
	reflection: Reflection  # Self-critique (missing + superfluous)
	search_queries: List[str]  # Queries to improve the answer

class ReviseAnswer(AnswerQuestion):
	references: List[str]  # Citations from search results
```

**Why this works:**
- `tool_choice="AnswerQuestion"` means the LLM CANNOT return plain text
- It MUST fill in every field including `reflection` and `search_queries`
- The self-critique isn't optional — it's structurally enforced
- The search queries bridge the LLM's self-awareness gap with real data

**The revision chain sees:**
1. The original answer (from draft)
2. The critique (from the same draft response)
3. The search results (from Tavily)
4. And must produce a BETTER answer + NEW critique + NEW queries + CITATIONS

---

## 4. The Graph Structure

### State

Uses `MessagesState` directly — just a growing list of messages:

```python
from langgraph.graph import MessagesState
builder = StateGraph(MessagesState)
```

### Stop Condition

Counts `ToolMessage` instances to track iterations:

```python
MAX_ITERATIONS = 2

def event_loop(state: MessagesState):
	count_tool_visits = sum(isinstance(item, ToolMessage) for item in state["messages"])
	if count_tool_visits > MAX_ITERATIONS:
		return END
	return "execute_tools"
```

### Edges

| From | To | Type | Condition |
|------|-----|------|-----------|
| START | draft | Fixed | Always |
| draft | execute_tools | Fixed | Always (draft always produces search queries) |
| execute_tools | revise | Fixed | Always |
| revise | execute_tools or END | Conditional | ToolMessage count > MAX_ITERATIONS → END |

---

## 5. Advanced Prompt Engineering

Eden highlights that making the LLM **actually incorporate critique** is harder than generating the critique. Key techniques:

### Shared Actor Prompt

Both chains use the same template with different `{first_instruction}`:

```python
"""You are expert researcher.
Current time: {time}

1. {first_instruction}
2. Reflect and critique your answer. Be severe to maximize improvement.
3. Recommend search queries to research information and improve your answer."""
```

### First Responder Instruction
```
"Provide a detailed ~250 word answer."
```

### Revisor Instruction
```
"Revise your previous answer using the new information.
- You should use the previous critique to add important information...
- You MUST include numerical citations...
- Add a References section...
- Make SURE it is not more than 250 words."
```

### Why "Be severe to maximize improvement"?

Without this directive, the LLM tends to say "great answer, no issues!" — which defeats the purpose of self-critique. The prompt explicitly instructs severity to force meaningful feedback.

---

## 6. The ToolNode Name-Matching Trick

The most subtle implementation detail:

```python
# In tool_executor.py
execute_tools = ToolNode([
	StructuredTool.from_function(run_queries, name=AnswerQuestion.__name__),
	StructuredTool.from_function(run_queries, name=ReviseAnswer.__name__),
])
```

**Why name tools after schemas?**

When you use `tool_choice="AnswerQuestion"`, the LLM's response has:
```json
{"tool_calls": [{"name": "AnswerQuestion", "args": {"answer": "...", "search_queries": [...]}}]}
```

`ToolNode` looks up tools BY NAME. So it needs a tool named `"AnswerQuestion"` to handle that call. Both tools run the exact same `run_queries` function — the name is just for routing.

---

## 7. Comparison: Reflection vs Reflexion

| Aspect | Section 14 (Reflection) | Section 15 (Reflexion) |
|--------|------------------------|------------------------|
| **Tools** | None | Tavily search |
| **Output format** | Freeform text (AIMessage) | Structured (Pydantic schema via tool_choice) |
| **Critique mechanism** | Separate reflect node | Built into the same response (Reflection field) |
| **Citations** | None | Required (References list) |
| **Stop condition** | Message count > 6 | ToolMessage count > MAX_ITERATIONS |
| **Nodes** | 2 (generate, reflect) | 3 (draft, execute_tools, revise) |
| **Grounding** | None (pure LLM) | Real-time web data |
| **Use case** | Tweet improvement | Research articles, factual content |
| **Quality driver** | LLM-to-LLM feedback | External data + self-critique |
| **Code complexity** | ~60 lines | ~100 lines (across 4 files) |

---

## 8. Code Walkthrough

### File Structure

```
15-reflexion-agent/
├── 19_Reflexion_Agent.md          # This theory file
└── src/
	├── schemas.py                 # Pydantic models (AnswerQuestion, ReviseAnswer)
	├── chains.py                  # LLM chains (first_responder, revisor)
	├── tool_executor.py           # Tavily search via ToolNode
	└── main.py                    # Graph definition + execution
```

### Execution Flow

```
Step 1: User sends question
		→ state: [HumanMessage("Write about AI-Powered SOC...")]

Step 2: draft_node runs first_responder chain
		→ state: [..., AIMessage(tool_calls=[{name: "AnswerQuestion",
				   args: {answer: "...", reflection: {...}, search_queries: [...]}}])]

Step 3: execute_tools (ToolNode) extracts search_queries, calls Tavily
		→ state: [..., ToolMessage(content="[search results...]")]

Step 4: revise_node runs revisor chain (sees answer + critique + search results)
		→ state: [..., AIMessage(tool_calls=[{name: "ReviseAnswer",
				   args: {answer: "...", reflection: {...}, search_queries: [...], references: [...]}}])]

Step 5: event_loop checks ToolMessage count (1 ≤ 2) → route to execute_tools

Step 6: execute_tools runs NEW search queries from revise step
		→ state: [..., ToolMessage(content="[new search results...]")]

Step 7: revise_node runs again (sees all prior context + new data)
		→ state: [..., AIMessage(tool_calls=[{name: "ReviseAnswer", args: {...}}])]

Step 8: event_loop checks ToolMessage count (2 ≤ 2) → route to execute_tools

Step 9: execute_tools runs yet another set of queries
		→ state: [..., ToolMessage(content="[more results...]")]

Step 10: revise_node runs final time
		 → state: [..., AIMessage(tool_calls=[{name: "ReviseAnswer", args: {...}}])]

Step 11: event_loop checks ToolMessage count (3 > 2) → route to END ✅
```

---

## 9. LangGraph Workflow Patterns

> Eden discussed these official LangGraph patterns during this section to show where the Reflexion agent fits in the broader landscape. These are from the [official LangGraph docs](https://langchain-ai.github.io/langgraph/).

### Pattern Taxonomy

| Pattern | Flow Shape | Cycles? | LLM Decides Flow? | Our Examples |
|---------|-----------|---------|-------------------|------|
| **Prompt Chaining** | A → B → C (linear) | ❌ | ❌ | RAG pipeline (Section 9) |
| **Parallelization** | START → [A, B, C] → Aggregator → END | ❌ | ❌ | — |
| **Routing** | Router → one-of-many paths → END | ❌ | ✅ (picks branch) | LangChain routers |
| **Orchestrator-Worker** | Orchestrator → spawn N workers → Synthesizer | ❌ | ✅ (plans subtasks) | — |
| **Evaluator-Optimizer** | Generate ←→ Evaluate (cycle) | ✅ | ✅ (decides quality) | Reflection Agent (Section 14) |
| **Agent (ReAct)** | Think → Act → Observe (cycle) | ✅ | ✅ (picks tools) | ReAct Agent (Section 13) |
| **Reflexion** | Draft → Search → Revise (cycle) | ✅ | ✅ (generates queries) | This section |

### Prompt Chaining

Each LLM call processes the output of the previous call. No branching, no cycles.

```
START → generate_joke → [check_punchline] → improve_joke → polish_joke → END
                              │
                              └─── (pass) ──→ END
```

**When to use:** Well-defined tasks with verifiable intermediate steps (translation, content verification).

### Parallelization

Multiple LLM calls run simultaneously, results aggregated.

```
         ┌─── call_llm_1 (joke) ───┐
START ───┼─── call_llm_2 (story) ──┼──→ aggregator → END
         └─── call_llm_3 (poem) ───┘
```

**When to use:** Independent subtasks that can run concurrently (speed), or running the same task multiple times (confidence).

**LangGraph feature:** Multiple edges from START to different nodes automatically run in parallel.

### Routing

LLM classifies input and routes to specialized handlers. No cycles.

```
START → router_llm → {story: llm_1, joke: llm_2, poem: llm_3} → END
```

**Key technique:** Use `with_structured_output(Route)` to get a Pydantic enum from the LLM, then route based on the field value.

### Orchestrator-Worker

One LLM plans subtasks, spawns workers dynamically, synthesizes results.

```
START → orchestrator (plans sections) → [Send("worker", section) for each] → synthesizer → END
```

**Key LangGraph feature:** The `Send` API lets you dynamically create worker nodes at runtime — you don't need to know how many workers you'll need at graph construction time.

```python
from langgraph.types import Send

def assign_workers(state):
    return [Send("llm_call", {"section": s}) for s in state["sections"]]
```

### Evaluator-Optimizer

Generate → evaluate → (if not good enough) → generate again. Exactly what our Reflection Agent does.

```
START → generator → evaluator → {Accepted: END, Rejected: generator}
```

**When to use:** Tasks with clear success criteria that need iteration (translation quality, code correctness, joke humor).

### Where Reflexion Fits

The Reflexion agent is a **combination** of:
- **Evaluator-Optimizer** (self-critique drives revision)
- **Agent** (uses tools to fetch data)
- **Prompt Chaining** (structured progression from draft to final)

---

## 10. ToolNode Deep Dive

> `ToolNode` is a prebuilt LangGraph node that handles tool execution. Eden uses it in this project and the concepts apply to all LangGraph agents.

### What ToolNode Does

| Input | Process | Output |
|-------|---------|--------|
| `AIMessage` with `tool_calls` | Looks up tool by name → invokes with args | `ToolMessage(content=result)` |

### Key Features

| Feature | Description |
|---------|-------------|
| **Name-based routing** | Matches `tool_call.name` to registered tool names |
| **Parallel execution** | If AIMessage has multiple tool_calls, executes all in parallel |
| **Error handling** | Configurable: catch errors and return them as ToolMessage, or propagate |
| **State injection** | Via `ToolRuntime` — tools can access graph state and run-scoped context |

### Input Formats

```python
# 1. Graph state (most common in LangGraph)
{"messages": [AIMessage(tool_calls=[{"name": "search", "args": {...}, "id": "abc"}])]}

# 2. Message list
[AIMessage(tool_calls=[...])]

# 3. Direct tool calls (for testing)
[{"name": "search", "args": {"query": "test"}, "id": "1", "type": "tool_call"}]
```

### Error Handling Options

```python
# Default: catches invocation errors, propagates execution errors
ToolNode(tools)

# Catch ALL errors, return as ToolMessage
ToolNode(tools, handle_tool_errors=True)

# Custom error message
ToolNode(tools, handle_tool_errors="Something went wrong, please try again.")

# Custom handler function
ToolNode(tools, handle_tool_errors=lambda e: f"Error: {str(e)}")

# Disable error handling (exceptions propagate)
ToolNode(tools, handle_tool_errors=False)
```

### ToolRuntime — Accessing State From Tools

Tools normally only receive the args the LLM generates. To access **graph state** or **run-scoped context**, use `ToolRuntime`:

```python
from langchain.tools import ToolRuntime, tool

class State(MessagesState):
    user_id: str

@tool
def get_user_info(runtime: ToolRuntime[None, State]) -> str:
    """Look up user information."""
    # Access graph state that the LLM didn't generate
    user_id = runtime.state["user_id"]
    return f"User {user_id}"
```

**Important:** Tools can only access state values passed to the ToolNode. When ToolNode is a direct graph node, it receives the full state. If you invoke it manually from another node, pass the full state explicitly.

### How It's Used in This Project

```python
# In tool_executor.py — tools named to match LLM's tool_choice
execute_tools = ToolNode([
    StructuredTool.from_function(run_queries, name="AnswerQuestion"),
    StructuredTool.from_function(run_queries, name="ReviseAnswer"),
])
```

The LLM calls `tool_choice="AnswerQuestion"` → ToolNode finds tool named `"AnswerQuestion"` → executes `run_queries` → returns ToolMessage with search results.

---

## Interview Q&A Anchors

**Q: What is a Reflexion agent and how does it differ from basic reflection?**
> **A:** A Reflexion agent combines self-critique with tool execution. Unlike basic reflection (LLM-to-LLM feedback only), reflexion fetches external data via tools to ground improvements in real facts. The LLM produces structured output containing answer, critique, AND search queries in one call — forcing self-awareness into the output format itself.

**Q: Why use `tool_choice` to force structured output?**
> **A:** `tool_choice="AnswerQuestion"` eliminates the LLM's option to respond with freeform text. It MUST return data matching the Pydantic schema, including the self-critique and search queries. This makes the reflection loop deterministic — every response always contains actionable feedback and queries for the next iteration.

**Q: How does the ToolNode know which function to call?**
> **A:** ToolNode matches tool calls by name. Since the LLM uses `tool_choice="AnswerQuestion"`, its response has `tool_calls[0].name == "AnswerQuestion"`. We register a `StructuredTool` with that exact name. Both `AnswerQuestion` and `ReviseAnswer` tools execute the same search function — the name is just a routing mechanism.

**Q: What's the stop condition in a Reflexion agent?**
> **A:** It counts `ToolMessage` instances in the state. Each tool execution adds one ToolMessage, so counting them equals counting iterations. When the count exceeds `MAX_ITERATIONS`, the conditional edge routes to END instead of back to execute_tools.

**Q: When would you use Reflexion over basic Reflection?**
> **A:** Use Reflexion when factual accuracy matters — research articles, technical documentation, reports that need citations. Basic reflection is sufficient for style/quality improvements (tweets, emails) where external data isn't needed. Reflexion costs more (tool calls + more LLM tokens) but produces grounded, verifiable output.

**Q: What are the main LangGraph workflow patterns?**
> **A:** Five core patterns: (1) Prompt chaining — linear sequence of LLM calls. (2) Parallelization — multiple independent calls at once. (3) Routing — LLM classifies input and routes to specialized handlers. (4) Orchestrator-worker — one LLM plans subtasks, spawns workers dynamically via Send API. (5) Evaluator-optimizer — generate/evaluate cycle until quality threshold met. Agents add tool use and cycles on top of these.

**Q: What is ToolNode and when would you use it vs writing your own tool execution?**
> **A:** ToolNode is a prebuilt LangGraph node that handles tool execution: name-based routing, parallel execution, error handling, and state injection via ToolRuntime. Use it when you want standard tool execution behavior. Write your own when you need custom routing logic, specialized error recovery, or non-standard argument transformation.

**Q: How does the Send API enable dynamic parallelism?**
> **A:** The Send API lets you create worker nodes at runtime based on state content. Instead of defining a fixed number of parallel branches at graph construction time, you return `[Send("worker", input) for input in dynamic_list]` from a conditional edge. Each Send spawns an independent execution of the target node with its own input, and results are collected via a shared state key with a reducer.

---

## Runnable Scripts

→ [`src/main.py`](./src/main.py) — Full agent execution
→ [`src/chains.py`](./src/chains.py) — Standalone chain test

---

## References

- [Reflexion Paper (Shinn et al., 2023)](https://arxiv.org/pdf/2303.11366) — The original paper from Northeastern, MIT, and Princeton
- [LangChain Blog — Reflection Agents](https://www.langchain.com/blog/reflection-agents) — LangChain team's implementation guide
- [LangGraph Reflexion Tutorial](https://langchain-ai.github.io/langgraph/tutorials/reflexion/reflexion/) — Official LangGraph tutorial
- [Tavily Search](https://tavily.com/) — Search engine optimized for LLM applications
- [LangGraph Workflows and Agents Guide](https://langchain-ai.github.io/langgraph/concepts/workflows-agents/) — Official patterns documentation (prompt chaining, parallelization, routing, orchestrator-worker, evaluator-optimizer)
- [ToolNode API Reference](https://reference.langchain.com/python/langgraph.prebuilt/tool_node/ToolNode) — Prebuilt node for tool execution
