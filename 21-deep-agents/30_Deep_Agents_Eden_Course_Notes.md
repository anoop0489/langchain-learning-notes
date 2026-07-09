# 30. Deep Agents — Eden's Course Notes (Chapters 147–152)

> **Context:** Section 21, Chapters 147–152. Eden introduces **Deep Agents** — agents built for *long-horizon* tasks (deep research, coding). He taxonomizes agents (shallow vs deep), defines what makes an agent "deep", and breaks down the four capabilities every deep agent shares: **planning, subagents, a file system, and a detailed system prompt**. These notes validate and expand Eden's transcript with accurate technical detail.

---

## The Core Idea

> **Remember this, forget the rest.** A **deep agent** is an agent that can complete *complex, long-running tasks* (minutes → hours → days) without drowning in its own context. It achieves this through **context engineering**: a planning tool to stay on track, subagents to do heavy work in isolation, and a file system to offload results out of the context window. The bottleneck isn't the LLM's reasoning — it's how you *manage the context* you feed it.

**The technique in one sentence:**

> "Shallow (ReAct) agents bloat their context until they rot. Deep agents keep the main context lean by planning explicitly, delegating to isolated subagents, and writing intermediate work to a file system."

**Why this section matters:**

```
LLMs are improving GRADUALLY now (not exponential leaps).
So where does the "magic" of Claude Code / Cursor / Devin come from?

   ┌──────────────────────────────────────────────┐
   │  THE APPLICATION LAYER                        │  ← this is the frontier
   │  (how we harness the LLM: deep agent design)  │
   ├──────────────────────────────────────────────┤
   │  THE LLM (getting better, but gradually)      │
   └──────────────────────────────────────────────┘

Innovation today is driven by the application layer —
the deep-agent harness we build ON TOP of the model.
```

---

## Table of Contents

| # | Section | What You'll Learn |
|---|---------|-------------------|
| 1 | [Introduction — What This Section Covers (Ch. 147)](#1-introduction--what-this-section-covers-ch-147) | Long-horizon tasks and why deep agents matter |
| 2 | [Taxonomy: Shallow, Deep & Coding Agents (Ch. 148)](#2-taxonomy-shallow-deep--coding-agents-ch-148) | Why the ReAct agent is "shallow" and what makes an agent "deep" |
| 3 | [The Four Characteristics of a Deep Agent (Ch. 148)](#3-the-four-characteristics-of-a-deep-agent-ch-148) | Planning, subagents, file system, system prompt |
| 4 | [Characteristic 1 — Dynamic To-Do Lists (Ch. 149)](#4-characteristic-1--dynamic-to-do-lists-ch-149) | Explicit planning tools vs implicit chain-of-thought |
| 5 | [Characteristic 2 — Subagents & Hierarchical Delegation (Ch. 150)](#5-characteristic-2--subagents--hierarchical-delegation-ch-150) | Spawning specialized workers in isolated context |
| 6 | [Subagent Context Flow (Ch. 151)](#6-subagent-context-flow-ch-151) | How delegation keeps the main context lean |
| 7 | [Characteristic 3 — File Systems (Ch. 152)](#7-characteristic-3--file-systems-ch-152) | Persisting and selecting context to fight context rot |
| 8 | [Context Engineering — The Unifying Theory](#8-context-engineering--the-unifying-theory) | Write, Select, Compress, Isolate |
| 9 | [Interview Q&A Anchors](#interview-qa-anchors) | Quick-fire answers |
| 10 | [References](#references) | Blogs, tools, and docs Eden mentions |

---

## Key Definitions

| Term | Quick Recall | Full Definition |
|------|-------------|----------------|
| **Deep agent** | Agent for complex, long-running tasks | An agent capable of long-horizon work (deep research, coding) that can run for minutes/hours/days, pause for user input, and resume. It stays reliable through context engineering. |
| **Shallow agent** | Plain ReAct agent, limited depth | Eden's term for a vanilla ReAct/tool-calling agent. Fine for short tasks with few iterations, but its context bloats on complex tasks, causing degraded quality. |
| **Long-horizon task** | Many-step, long-running goal | A complex objective needing many iterations of gather → process → deduce (e.g., "implement this feature", "research this topic") — not a one-shot like "book me a flight". |
| **Context rot** | Degradation as context grows | The umbrella problem where a bloated context window degrades LLM performance — via confusion, contradiction, or pollution — pushing the agent "off the rails". |
| **Context bloat** | Ever-growing token accumulation | In a ReAct loop, every decision + tool result gets appended to the context. More iterations → bigger context → higher cost, higher latency, worse quality. |
| **Planning tool** | Explicit dynamic to-do list | A tool (usually a markdown to-do list) the agent actively updates with `pending`/`in_progress`/`completed` statuses — explicit planning, not implicit chain-of-thought. |
| **Subagent** | Specialized worker in isolated context | A fresh instance the deep agent spawns with its own system prompt and tools. It runs an independent loop and returns **one** condensed result, keeping heavy work out of the main context. |
| **Hierarchical delegation** | Main agent → specialized subagents | The pattern where a deep agent breaks work down and hands focused subtasks to specialized subagents (like delegating a job to someone with the right skills and tools). |
| **Context isolation** | Subagent work doesn't touch main context | A subagent's intermediate tokens live in *its own* window; only the final artifact returns — so it never pollutes the main agent's attention. |
| **File system** | Offload & retrieve context on disk | Tools (`ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`) that let the agent persist intermediate results and retrieve only what's relevant — an *interface* you can back with any store. |
| **Context engineering** | Getting the *right* info into context | The discipline of structuring, retrieving, and prioritizing information so the context window holds exactly what the task needs — more decisive than the prompt itself. |
| **Coding agent** | Deep agent specialized for code | A subset of deep agents tailored for software work (Claude Code, Cursor, Devin, Gemini CLI) — they write code, run tests, open browsers, take screenshots. |
| **Deep research agent** | Deep agent specialized for research | A deep agent that performs thorough multi-source research (Perplexity, ChatGPT/Claude "research", open-source GPT Researcher). |

---

## 1. Introduction — What This Section Covers (Ch. 147)

Eden opens by framing **deep agents** as agents that can *act on long-horizon tasks* — the hard, multi-step, long-running kind. The canonical examples are **coding agents** (Claude Code, Cursor CLI) and **deep research agents**, plus **LangChain's Deep Agents harness** we'll dissect.

The section has three goals:

1. **Taxonomize agents** — where do "deep agents" sit in the agent landscape?
2. **Define a deep agent** — what high-level characteristics must it have?
3. **Review LangChain's Deep Agents harness** — first as *users*, then dissect the *code* to see how the LangChain team built it.

> 💡 **Why it's exciting:** Understanding the deep-agent harness gives you a glimpse of how state-of-the-art coding agents like Claude Code work under the hood.

---

## 2. Taxonomy: Shallow, Deep & Coding Agents (Ch. 148)

### The landscape of agents

Eden maps the world of agents as one big umbrella:

```
						┌──────────────────────────────────┐
						│        DOMAIN OF AGENTS           │
						│                                  │
						│   ┌──────────────────────────┐    │
						│   │  SHALLOW AGENTS           │    │
						│   │  • ReAct agent            │    │
						│   │  • Agentic apps (hybrid   │    │
						│   │    RAG: LLM routes steps) │    │
						│   └──────────────────────────┘    │
						│                                  │
						│   ┌──────────────────────────┐    │
						│   │  DEEP AGENTS              │    │
						│   │  • Deep research agents   │    │
						│   │  • Coding agents ◄─subset │    │
						│   └──────────────────────────┘    │
						└──────────────────────────────────┘
```

### Why the ReAct agent is "shallow"

The **ReAct agent** is where the whole agent paradigm started — an LLM decides whether to use a tool, executes it, observes the result, and loops until it has an answer. Eden calls it a **shallow agent**, not as an insult but as a *classification*: it can't go deep on complex tasks. Two reasons:

1. **The architecture bloats context.** Every iteration appends the decision + tool result back into the prompt. One iteration is fine; many iterations grow the context relentlessly.
2. **The reality of LLMs.** Context windows are finite, and a bloated window causes **context rot** — confusion, contradiction, pollution.

```
ReAct loop on a COMPLEX task:

 Turn 1:  [prompt + tool result]                      → 10K tokens
 Turn 2:  [prompt + result 1 + result 2]              → 30K tokens
 Turn 3:  [... + result 3]                            → 55K tokens
 Turn 5:  [... everything so far ...]                 → 100K tokens
														▲
						  context keeps growing ────────┘
   Result: ↑ cost, ↑ latency, ↓ quality (context rot → off the rails)
```

> ⚠️ **Transcript correction ("cloud code" → Claude Code):** Throughout these chapters the speech-to-text renders **Claude Code** as "cloud code". The tool is **Claude Code**, Anthropic's coding agent. All references below use the correct name.

> **Important nuance — shallow ≠ bad.** Eden stresses the ReAct architecture is *"the basis for basically everything"* and is genuinely good for **shallow tasks** with few iterations. Many production agents are ReAct agents, and for many use cases that's *enough*. The problem only appears on **deep/complex** tasks that need long, iterative gathering and processing.

### What makes an agent "deep"

Deep agents perform **long-horizon tasks**: complex, iteration-heavy, long-running (minutes → hours → days), able to **pause for user input and resume**. Eden's examples:

| Category | Examples | Notes |
|----------|----------|-------|
| **Deep research agents** | Perplexity, ChatGPT "research", Claude "research", **GPT Researcher** (open source) | Almost every vendor now ships a research feature that triggers a deep-agent run. |
| **Coding agents** | **Claude Code** (leading, per Eden), Devin, Cursor, Gemini CLI | A **subset** of deep agents tailored for coding — they code, run tests, open browsers, take screenshots, like a real engineer. |

> ✅ **Transcript validation (GPT Researcher):** GPT Researcher is a real, widely-used open-source deep research agent (created by Assaf Elovic). Eden notes he reviews it in depth in his LangGraph course.

> 💡 **The bigger point.** LLMs are now improving *gradually*, not exponentially. The **application layer** — how we *harness* LLMs into deep agents — is what's driving innovation today. Building a working, capable application from a prompt with minimal human intervention was unthinkable five years ago; the deep-agent application layer is what makes it real now.

---

## 3. The Four Characteristics of a Deep Agent (Ch. 148)

Eden credits the **LangChain team** with coining "Deep Agents" and articulating the architecture after analyzing many implementations. Almost every deep agent you meet implements **four ideas**:

![Deep Agents Components](./assets/Deep%20Agents%20Components.png)

| # | Characteristic | What it does | Solves |
|---|----------------|--------------|--------|
| 1 | **Planning tool** | Explicit, dynamic to-do list the agent updates as it works | Staying on track over many steps |
| 2 | **Subagents** | Spawn specialized workers in **isolated** context | Scaling work without bloating the main context |
| 3 | **File system** | Write intermediate results / share state to disk | Offloading context out of the window |
| 4 | **Detailed system prompt** | A large, carefully engineered system prompt | Orchestrating all of the above reliably |

> ⚠️ **Transcript correction (numbering):** Eden says *"thirdly"* twice — once for the file system and once for the system prompt. The fourth characteristic is the **detailed (Eden: "monstrous") system prompt**. So the list is **planning → subagents → file system → system prompt**.

The common thread across all four: **context engineering** and smart **context management** — solving context bloat/accumulation so the agent can scale and run long-horizon tasks with quality results. The next chapters unpack characteristics 1–3 in detail.

---

## 4. Characteristic 1 — Dynamic To-Do Lists (Ch. 149)

Every deep agent has a **planning tool**. Crucially, this is **explicit** planning — *not* the implicit chain-of-thought reasoning LLMs do internally. It's typically a **markdown to-do list** the agent actively reviews and updates *between* executions.

```
┌─ AGENT'S TO-DO LIST (markdown, highly dynamic) ─────────────┐
│  [x] Analyze existing auth module          (completed)      │
│  [x] Identify token-refresh bug            (completed)      │
│  [»] Write failing test for refresh path   (in_progress)    │
│  [ ] Implement fix                         (pending)        │
│  [ ] Run full test suite                   (pending)        │
└─────────────────────────────────────────────────────────────┘
		 ▲ updated after each step; user can influence it too
```

Key properties:

- **Status tracking** — tasks are `pending`, `in_progress`, or `completed`.
- **Dynamic** — the agent re-reads and rewrites the plan continuously as it learns.
- **Smarter failure handling** — a failed task is **not** blindly retried (unlike the raw ReAct loop); the plan lets the agent re-strategize.
- **Steerable** — you, the user, can influence the to-do list.

> **Claude Code example.** Claude Code's planning tool is *internal* — you can't edit it directly, but you can watch it execute. Eden references an X (Twitter) post by **Boris Cherny** (creator of Claude Code) showing the `update_todo` call that continuously rewrites the list.

> 💡 **Why it works — it's just how humans work.** Faced with something complex, we break it down and track what's done. It shows progress and (Eden jokes) gives a little dopamine hit when you check a box. The idea is intuitive; the power is in making the LLM do it *explicitly*.

> 🔷 **C# analogy:** Think of a persisted `List<WorkItem>` where `WorkItem` has a `Status { Pending, InProgress, Completed }` enum — a structured, observable task list the process reads and mutates as it runs, rather than keeping the plan only in "memory" (chain-of-thought).

---

## 5. Characteristic 2 — Subagents & Hierarchical Delegation (Ch. 150)

The second characteristic is **subagents**, which enable **hierarchical delegation**. The deep agent can **spawn new instances** — but as *specialized* subagents for focused tasks. Each subagent has:

- its **own system prompt**,
- its **own description**,
- its **own set of tools**.

```
					 ┌─────────────────┐
					 │   MAIN AGENT     │
					 │  (orchestrator)  │
					 └───┬─────────┬────┘
			  delegates  │         │  delegates
						 ▼         ▼
		  ┌────────────────┐   ┌────────────────┐
		  │  SUBAGENT A     │   │  SUBAGENT B     │
		  │  own prompt     │   │  own prompt     │
		  │  own tools      │   │  own tools      │
		  │  own ReAct loop │   │  own ReAct loop │
		  └───────┬─────────┘   └───────┬────────┘
				  │ 1 condensed         │ 1 condensed
				  │ result              │ result
				  └──────────┬──────────┘
							 ▼
					 back to MAIN AGENT
			  (intermediate work stays isolated)
```

### Eden's real-world analogy (the ceiling hatch)

Eden isn't a handyman. He had a noisy ceiling hatch — rain hitting fiberglass echoed through the house. The fix: lay synthetic grass over the fiberglass so raindrops land softly. But it needed a special diagonal ladder and skills he doesn't have. So he **delegated** to his father-in-law:

| Real-world element | Subagent equivalent |
|--------------------|---------------------|
| A message describing the task | The **prompt** the main agent sends the subagent |
| Father-in-law's skills | The subagent's **system prompt** |
| His own box cutter & ladder | The subagent's **tools** |
| Eden wasn't even there while he worked | **Context isolation** — the main agent doesn't see the intermediate work |
| "Everything's fixed" at the end | The single **condensed result** returned |

The analogy nails it: you delegate to someone with the right **skills** and **tools**, you explain **what to do**, they work in **isolation**, and you just get the **final result**.

### What subagents give you

- **Isolated work** — each runs in its **own context window**, never polluting the main agent's context.
- **Specialization** — different system prompts and tools per subagent → higher-quality, focused work.
- **Independent loops** — each subagent runs its own tool-calling/ReAct loop internally and returns **only** the final response (no intermediate results leak back).
- **Parallel execution** — subagents can run concurrently, dramatically improving depth, efficiency, and quality.

> **Claude Code example.** Claude Code spawns an **exploration subagent** to find authentication patterns (searching the codebase) *while the main agent keeps running* — exactly this architecture in production.

> 🔷 **C# analogy:** A subagent is like `Task.Run` into a **fresh DI scope** — the child does its work with its own dependencies (tools) and configuration (system prompt), shares no mutable state with the parent, and returns a single result object. The parent never sees the child's local variables (intermediate context).

---

## 6. Subagent Context Flow (Ch. 151)

This chapter is the *why* behind subagents: **context compression through isolation**.

### The flow

```
MAIN AGENT THREAD (must stay lean)
  │  every message here grows the main token count
  │
  │  ── generates a NEW prompt ──►  ┌───────────────────────────┐
  │     (the ONLY context the       │  SUBAGENT (fresh context) │
  │      subagent will ever see)    │  • sees ONLY that prompt  │
  │                                 │  • runs independently     │
  │                                 │  • invokes its own tools  │
  │  ◄── ONE condensed response ─── │  • no history awareness   │
  │      (the "artifact")           └───────────────────────────┘
  │
  ▼  main context grew by only the small artifact, not the heavy work
```

Two critical insights:

1. **The subagent only sees the prompt the main agent writes for it** — not the whole conversation. So the main agent's prompt *engineering* determines subagent quality. Eden notes you can deliberately **shape this prompt** to make the subagent's job easier and get better results.
2. **Only one condensed response returns.** Every subagent spawn starts from a **fresh context** with just that prompt, does the heavy lifting, and hands back a compact artifact.

### Why it matters — the token budget

LLMs have **finite** token limits (today ~200K, up to ~1M; future maybe 2M/10M — but always finite). Approaching the limit is bad on every axis:

| Problem near the limit | Consequence |
|------------------------|-------------|
| Exceed the limit | The request **fails** outright |
| More tokens per call | **Higher cost** (you pay per token) |
| More tokens to process | **Higher latency** (slower responses) |
| Too much irrelevant context | **Context pollution** → worse answers |

Without subagents, a single Claude Code instance accumulates tokens every turn (Eden's illustration: ~10K → ~30K → ~100K by turn 5), eventually forcing a `/compact`, a `/clear`, or a fresh instance.

```
WITHOUT subagents (one instance):        WITH a subagent:
  turn 1  ██                10K            main:   ██        (stays lean)
  turn 2  █████             30K            subagent runs in ITS OWN window
  turn 5  ████████████     100K            main += ███  (only the ~15–20K
		  ▲ forces /compact or /clear                artifact: summary + code)
```

A subagent might internally burn a lot of tokens, but it only returns a **condensed ~15–20K-token artifact** (e.g., a summary + the code it changed). That heavy work is **never accumulated** into the main agent's context — an elegant context-engineering win. And because each subagent has a **tailor-made system prompt**, it often solves its focused task *better* than the generalist main agent would.

---

## 7. Characteristic 3 — File Systems (Ch. 152)

The third characteristic: deep agents have access to a **file system** with tools to search, read, write, update, and delete files — giving them full control to **manage context** on disk.

### The file operations

**Claude Code's** file tools and the **Deep Agents** interface line up almost exactly:

| Operation | Claude Code | Deep Agents interface |
|-----------|-------------|-----------------------|
| List directory | — | `ls` |
| Read file content | Read | `read_file` |
| Create / overwrite | Write | `write_file` |
| Precise string replace | Edit | `edit_file` |
| Find files by pattern | Glob | `glob` |
| Search within files | Grep | `grep` |

> **It's an *interface*, not a hardcoded disk.** This is the key design point. Deep Agents **exposes an interface**, so you can back it with anything — local disk, a **Firestore** database on Google Cloud, **DynamoDB** on AWS, etc. Fully pluggable. (In doc [29 — Theory & Concepts](./29_Deep_Agents_Theory_And_Concepts.md#52-virtual-filesystem) these are the *backends*: `StateBackend`, `FilesystemBackend`, `StoreBackend`, or custom.)

### Why file systems fight context rot

As a conversation grows, context grows, and you hit context rot (contradiction, confusion, noise) → bad LLM responses. Eden references a **LangChain blog** illustration of the core context-engineering challenge:

```
		┌───────────────────────────────────────────────┐
		│  BLUE = ALL available context                 │
		│  (codebase, docs, web, files, databases —     │
		│   potentially massive)                        │
		│                                              │
		│        ┌───────────────────────────┐         │
		│        │  RED = what the agent      │         │
		│        │  actually PULLS into the   │         │
		│        │  context window            │         │
		│        │      ┌───────────────┐     │         │
		│        │      │ GREEN = what   │     │         │
		│        │      │ the task       │     │         │
		│        │      │ actually NEEDS │     │         │
		│        │      └───────────────┘     │         │
		│        └───────────────────────────┘         │
		└───────────────────────────────────────────────┘

  GOAL: make RED as small as possible while still covering GREEN.
```

The failure modes:

| Scenario | What goes wrong | In the diagram |
|----------|-----------------|----------------|
| **Under-retrieval** | Agent misses relevant info it needs | Red covers only part of green |
| **Over-retrieval** | Agent pulls too much noise, diluting the signal | Red is way too big |
| **Misaligned retrieval** | Agent looks in the wrong places entirely | Red doesn't overlap green |
| **Context window limit** | Red is finite — can't hold everything | Red is capped in size |

The **sweet spot**: the smallest possible red circle that still covers green — and this selection happens **on almost every iteration**. That's why *how you structure, retrieve, and prioritize* information matters **more than the prompt itself**. Even the best reasoning model gives wrong answers with the wrong context.

### The file system as the context engine

The file system is the **mechanism** that lets the agent hit that green sweet spot. It implements **two** context-engineering principles:

| Principle | How the file system does it |
|-----------|-----------------------------|
| **Write context** | Persist temporary files, intermediate results, web-retrieved data to **persistent storage** — so it *doesn't* pollute the context window |
| **Select context** | Retrieve only what's relevant using `glob` (find files by pattern) and `grep` (search contents by regex) |

> 🔷 **C# analogy:** The file system is like moving large objects out of the hot request path into a **cache / persistent store** (`IDistributedCache`, blob storage), keeping only *keys/pointers* in memory, then fetching precisely what you need on demand — the same "keep the working set small, page in what's relevant" instinct.

---

## 8. Context Engineering — The Unifying Theory

Eden mentions "writing" and "selecting" context. Those are two of the **four context-engineering strategies** popularized by the LangChain team — worth knowing all four because the deep-agent characteristics map cleanly onto them:

| Strategy | What it means | Deep-agent mechanism |
|----------|---------------|----------------------|
| **Write** | Persist context outside the window | **File system** (`write_file`) + the planning to-do list |
| **Select** | Pull only relevant context back in | **File system** (`glob`, `grep`, `read_file`) |
| **Compress** | Shrink context you keep | **Summarization**; a subagent's condensed artifact |
| **Isolate** | Split context across boundaries | **Subagents** (each in its own window) |

```
   THE FOUR DEEP-AGENT CHARACTERISTICS  ─── all serve ───►  CONTEXT ENGINEERING

   Planning tool ──────────► keeps the agent on-track (Write: externalize the plan)
   Subagents ──────────────► Isolate + Compress (heavy work off the main context)
   File system ────────────► Write + Select (persist, then retrieve precisely)
   System prompt ──────────► orchestrates all of the above
```

> **The one-line takeaway.** Deep agents aren't "smarter models" — they're **better context managers**. Every characteristic exists to keep the right information (green) in a lean context window (small red) so a finite-context LLM can complete long-horizon work.

---

## Interview Q&A Anchors

**Q: What is a deep agent, and how is it different from a shallow (ReAct) agent?**
> **A:** A deep agent handles complex, long-horizon tasks — long-running (minutes to days), many iterations, able to pause for user input and resume. A shallow agent is a vanilla ReAct/tool-calling loop: great for short tasks with few iterations, but on complex tasks its context bloats every turn, causing context rot and degraded quality. Deep agents add planning, subagents, and a file system specifically to manage context so they can go deep.

**Q: Why does the ReAct architecture struggle with complex tasks?**
> **A:** Because every iteration appends the decision plus the tool result back into the prompt, so the context grows relentlessly. On a long task this leads to context rot — confusion, contradiction, and pollution — plus higher cost and latency since you pay per token and process more of them each call. It's not that ReAct is bad; it's the basis for everything and ideal for shallow tasks. It just doesn't scale to deep, iteration-heavy work.

**Q: What are the four characteristics of a deep agent?**
> **A:** (1) A planning tool — an explicit, dynamic to-do list the agent updates as it works; (2) subagents — specialized workers spawned in isolated context; (3) a file system — to write intermediate results and select relevant context; and (4) a detailed system prompt that orchestrates it all. All four exist to solve one problem: context management for long-horizon tasks.

**Q: Why use an explicit planning tool instead of chain-of-thought?**
> **A:** Chain-of-thought reasoning is implicit and ephemeral — it lives in the token stream and gets buried as context grows. An explicit planning tool is a persisted, dynamic to-do list with pending/in-progress/completed statuses that the agent re-reads and updates between steps, and that the user can steer. It keeps the agent on track over many iterations and, unlike raw ReAct, lets it re-strategize on failure instead of blindly retrying.

**Q: How do subagents keep the main context lean?**
> **A:** The main agent writes a fresh prompt for the subagent — that prompt is the only context the subagent sees. The subagent runs its own independent loop, burning whatever tokens it needs in its own isolated window, and returns a single condensed artifact (e.g., a summary plus changed code). The heavy intermediate work never accumulates in the main agent's context, so the main thread stays lean and avoids hitting the token limit.

**Q: What's the significance of the subagent prompt?**
> **A:** Since the subagent only ever sees the prompt the main agent generates, that prompt fully determines the subagent's quality — it has no awareness of the broader conversation. This means you can deliberately shape the prompt to make the subagent's job easier, and give the subagent a tailored system prompt and tool set so it solves its focused task better than the generalist main agent would.

**Q: Why do deep agents need a file system?**
> **A:** To fight context rot by managing context on disk rather than in the window. It implements two context-engineering principles: Write (persist temporary results, intermediate work, and retrieved data to storage so they don't pollute the context) and Select (use glob and grep to retrieve only what's relevant). Importantly it's an interface, so the backend can be local disk, Firestore, DynamoDB, or anything else.

**Q: Explain the blue/red/green context-engineering diagram.**
> **A:** Blue is all available context (codebase, docs, web, databases — potentially massive). Red is what the agent actually pulls into the window. Green is what the task actually needs. The goal is to make red as small as possible while still covering green. Failure modes are under-retrieval (red misses green), over-retrieval (red is bloated with noise), misaligned retrieval (red doesn't overlap green), and the hard context-window limit on red's size.

**Q: Why is the "application layer" the frontier of AI innovation right now?**
> **A:** LLMs are improving gradually rather than in exponential leaps, so the big wins increasingly come from how we harness them — the deep-agent application layer built on top. By combining planning, subagents, file systems, and strong context engineering, we can build agents that automate reasoning-heavy human work (like coding a whole feature) that seemed impossible a few years ago. The design of the harness, not just the model, is what pushes the boundary.

**Q: What are the four context-engineering strategies, and how do they map to deep agents?**
> **A:** Write (externalize context — the file system and planning list), Select (retrieve only relevant context — glob/grep/read_file), Compress (shrink what you keep — summarization and a subagent's condensed artifact), and Isolate (split context across boundaries — subagents in their own windows). Eden explicitly covers Write and Select via the file system; subagents cover Isolate and Compress. Together they keep a finite context window focused on what the task needs.

---

## References

- [LangChain — Deep Agents Overview](https://docs.langchain.com/oss/python/deepagents/) — the harness Eden dissects
- [LangChain Blog — *Deep Agents*](https://blog.langchain.com/deep-agents/) — where the term and architecture were articulated
- [LangChain Blog — *Context Engineering for Agents*](https://blog.langchain.com/context-engineering-for-agents/) — the Write/Select/Compress/Isolate framework and the blue/red/green diagram
- [Claude Code](https://www.anthropic.com/claude-code) — Anthropic's leading coding agent (Boris Cherny, creator)
- [GPT Researcher](https://github.com/assafelovic/gpt-researcher) — popular open-source deep research agent (Assaf Elovic)
- [Cursor](https://cursor.com/) · [Devin](https://devin.ai/) · [Gemini CLI](https://github.com/google-gemini/gemini-cli) — other coding agents Eden mentions
- [Perplexity](https://www.perplexity.ai/) — deep research feature example
- Companion: [29. Deep Agents — Theory & Concepts](./29_Deep_Agents_Theory_And_Concepts.md) — the official-docs deep dive on the harness
