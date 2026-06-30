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
| 9 | [Interview Q&A Anchors](#interview-qa-anchors) | Quick-fire answers |

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
