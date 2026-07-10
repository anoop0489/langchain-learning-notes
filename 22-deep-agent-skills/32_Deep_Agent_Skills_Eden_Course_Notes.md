# 32. Deep Agent Skills — Eden's Course Notes (Chapters 153–157)

> **Context:** Section 22, Chapters 153–157. Eden dissects **Agent Skills** — reusable packages of domain knowledge (a `SKILL.md` file plus supporting resources) that an agent loads *on demand*. He does this through **three layers of abstraction**: (1) *using* skills via the Deep Agents CLI, (2) *tracing* what actually reaches the LLM in LangSmith, and (3) *reading the source* of `skills.py` to see how **progressive disclosure** is implemented. Because Deep Agents is **open source**, we can verify the exact mechanism — unlike Claude Code, Cursor CLI, Gemini CLI, or Manus, which are closed. These notes validate and correct Eden's transcript.

---

## The Core Idea

> **Remember this, forget the rest.** A **skill** is a folder with a `SKILL.md` file (YAML frontmatter + instructions) and optional supporting files. The agent harness loads only the **metadata** (name, description, location) into the system prompt at startup — *not* the full skill. The LLM then decides *when* a skill applies and *which* files to read. This two-step "show the index, read the body only when needed" pattern is **progressive disclosure**, and it's what keeps the context window lean while making rich capabilities available.

**The technique in one sentence:**

> "Load skill *metadata* eagerly so the LLM knows what exists; load skill *content* lazily so the context stays small — and let the LLM choose which files to open."

**Why this matters:** Skills are the same mechanism across Claude Code, Gemini CLI, Manus, and Deep Agents. Learning it once from open-source code means you understand all of them.

---

## Table of Contents

| # | Section | What You'll Learn |
|---|---------|-------------------|
| 1 | [The Three Layers of Understanding (Ch. 153)](#1-the-three-layers-of-understanding-ch-153) | The learning roadmap: use → trace → source |
| 2 | [What Is a Skill?](#2-what-is-a-skill) | Anatomy of a `SKILL.md` package |
| 3 | [Layer 1 — Using Skills in the Deep Agents CLI (Ch. 154)](#3-layer-1--using-skills-in-the-deep-agents-cli-ch-154) | Installing & invoking a skill (Remotion example) |
| 4 | [Layer 2 — Tracing Skills with LangSmith (Ch. 155)](#4-layer-2--tracing-skills-with-langsmith-ch-155) | What actually reaches the LLM, step by step |
| 5 | [The Two Middlewares (Recap, Ch. 156)](#5-the-two-middlewares-recap-ch-156) | `before_agent` discovery vs `wrap_model_call` injection |
| 6 | [Layer 3 — Inside `skills.py` (Ch. 157)](#6-layer-3--inside-skillspy-ch-157) | The source code behind progressive disclosure |
| 7 | [The Full Progressive Disclosure Flow](#7-the-full-progressive-disclosure-flow) | End-to-end diagram |
| 8 | [C# Analogy](#8-c-analogy) | Lazy loading / metadata index mapping |
| 9 | [Interview Q&A Anchors](#interview-qa-anchors) | Quick-fire answers |
| 10 | [References](#references) | Docs, repo, and tools Eden mentions |

---

## Key Definitions

| Term | Quick Recall | Full Definition |
|------|-------------|----------------|
| **Skill** | A reusable knowledge package | A directory containing a `SKILL.md` file (YAML frontmatter + instructions) plus optional supporting files (scripts, rules, assets, templates). Follows the [Agent Skills standard](https://agentskills.io/). |
| **`SKILL.md`** | The skill's entry point / index | Markdown file whose frontmatter (name, description, when-to-use) is the metadata loaded eagerly; its body + linked files are read lazily. |
| **Progressive disclosure** | Show the index, read the body on demand | Load only skill *metadata* into context at startup; load full skill *content* only when the LLM decides a task needs it. |
| **Skill discovery** | Find what skills exist | The `before_agent` step that scans sources, parses frontmatter, and stores skill metadata in agent state. Runs **once per session**. |
| **Skill injection** | Put the skills index in the prompt | The `wrap_model_call` step that appends the `SKILLS_SYSTEM_PROMPT` (skill list + locations + instructions) to the system prompt **before every LLM call**. |
| **Agent harness** | Scaffolding around the LLM loop | Deep Agents — the open-source implementation whose `skills.py` we can read (unlike closed Claude Code / Cursor / Gemini CLI). |
| **Backend** | Where skills physically live | Pluggable storage (local filesystem, or cloud like Firestore/Bigtable). Skills are read through the backend, so they can be cloud-based. |

---

## 1. The Three Layers of Understanding (Ch. 153)

Eden structures the whole section as **peeling back layers of abstraction**, from most abstract (using) to most concrete (source code):

```
┌─────────────────────────────────────────────────────────────┐
│ LAYER 1 — USE (most abstract)                                │
│   Add & run skills as a user via the Deep Agents CLI.        │
├─────────────────────────────────────────────────────────────┤
│ LAYER 2 — TRACE                                              │
│   Inspect the LLM calls in LangSmith. See what context       │
│   is actually sent and when skills load.                     │
├─────────────────────────────────────────────────────────────┤
│ LAYER 3 — SOURCE (most concrete)                            │
│   Read skills.py in the Deep Agents repo. See the exact      │
│   code that implements progressive disclosure.               │
└─────────────────────────────────────────────────────────────┘
```

> 💡 **Why open source matters:** Claude Code, Cursor CLI, Gemini CLI, and Manus are **closed source** — you can't see how their skills work. LangChain **Deep Agents** is open source, so this section can verify every claim against real code. The skill mechanism is ~800 lines in one file (`skills.py`).

After all three layers, you'll know **how to use** skills, **how the agent runs** them, and **how to implement** the mechanism yourself.

---

## 2. What Is a Skill?

A skill is just a **directory** on disk (or in a backend). Its structure, using the Remotion example Eden installs:

```
remotion-best-practices/
├── SKILL.md              ← frontmatter (name, description) + instructions ← the INDEX
├── rules/                ← markdown reference files, loaded on demand
│   ├── gifs.md
│   ├── animations.md
│   ├── compositions.md
│   └── timings.md
└── assets/               ← TypeScript files the agent can execute or use as inspiration
```

**`SKILL.md`** has two parts:

| Part | Loaded when? | Purpose |
|------|-------------|---------|
| **Frontmatter (YAML)** — name, description, when-to-use | **Eagerly** (at startup, into system prompt) | The "index card" the LLM scans to decide if the skill applies |
| **Body + linked files** (`rules/*.md`, `assets/*`) | **Lazily** (only when the LLM reads them) | The full instructions and resources |

> 🎯 **Key design insight (Ch. 157):** The `SKILL.md` should act like an **index** — accessible and easy for the LLM to navigate — because *the LLM alone decides which linked files to read*. A well-structured index → better file selection → better results.

---

## 3. Layer 1 — Using Skills in the Deep Agents CLI (Ch. 154)

Eden installs and runs a skill entirely as a **user**, no code.

### Step 1 — Install the Deep Agents CLI

Installed via `uv` (or a script). This exposes two executables: **`deepagents`** and **`deepagents-cli`**. Run `deepagents --help` to see options.

### Step 2 — Configure an LLM

On first run, the CLI reports **"no credentials configured"** — you must set an LLM. Eden uses Anthropic:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."   # any provider works
```

> ⚠️ **Transcript correction:** The transcript's speech-to-text renders "Deep Agents CLI" inconsistently and mishears some commands. The two executables are **`deepagents`** and **`deepagents-cli`**; the env var is **`ANTHROPIC_API_KEY`**. On Windows PowerShell (this repo's environment) use `$env:ANTHROPIC_API_KEY = "..."` instead of `export`.

### Step 3 — Install a skill

Eden installs the **Remotion** skill (Remotion = an open-source package for creating videos with React) using the community **`skills`** installer:

```bash
npx skills add remotion-dev/skills
```

The installer asks **two questions**:

| Question | Options | Eden's choice |
|----------|---------|---------------|
| **Which agent format?** | `.agents/skills` (universal), `.claude/skills` (Claude Code), `skills/` (OpenCode), … | `.agents/skills` — the **universal** format most agents (incl. Deep Agents) read |
| **Project or global?** | Project (current dir) vs Global (home dir) | **Global** — so the skill is available to all agents on the machine |

Before installing, the installer shows a **risk assessment** (e.g., agent-safety verdict, Socket alerts, Snyk risk). Eden reviews and installs.

> ⚠️ **Transcript correction:** The transcript garbles the security scanners as "Socket," "SNCC," and "agent say." These are almost certainly **Socket** and **Snyk** (supply-chain security scanners) plus an agent-safety check. Verify any skill's source before installing — skills can ship executable scripts.

> ⚠️ **Transcript note:** At recording time, Deep Agents was **not yet listed** in the installer's agent list, but it **does** support the universal `.agents/skills` directory. So Eden picks the universal option deliberately.

### Step 4 — Verify & run

Restart the CLI and ask *"which skills do you have?"*. Three skills appear:

| Skill | Origin | Purpose |
|-------|--------|---------|
| **skill-creator** | Built-in (LangChain default) | Helps create new skills (like Anthropic's / Codex's skill-creator) |
| **find-skills** | Built-in (LangChain default) | A skill for finding other skills |
| **remotion-best-practices** | Just installed | Best practices for building Remotion videos |

Then Eden prompts: *"Create a brand new Remotion video about agent skills."* The agent:
1. Reads `SKILL.md` for `remotion-best-practices` (the skill is now disclosed into context),
2. Reads several supporting files,
3. Scaffolds a new Remotion project (portrait aspect ratio), writes React/TypeScript code, installs dependencies, builds scenes, and renders the video.

The model shown at work is **Claude Sonnet 4** (bottom-right of the CLI).

> ⚠️ **Transcript correction:** The transcript says "Anthropic Claude Sonnet 4.6." As of these notes there is **no Claude Sonnet 4.6**; the current Sonnet is **Claude Sonnet 4.5** (`claude-sonnet-4-5`). The official Deep Agents docs also use placeholder future model strings like `claude-sonnet-4-6` — treat the *format* `provider:model` as normative, not the exact version.

**Takeaway of Layer 1:** Installing a skill is a one-liner, and the harness performs **dynamic disclosure** — it only loads the full skill *when you ask for the task that needs it*.

---

## 4. Layer 2 — Tracing Skills with LangSmith (Ch. 155)

Now Eden enables tracing to see **exactly what reaches the LLM**.

### Enabling tracing

Set these environment variables (the docs example is slightly outdated):

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY="lsv2_..."
export LANGSMITH_PROJECT="my-deepagent-execution"   # any name you like
```

> ⚠️ **Transcript / docs correction:** Eden finds that the documented variables **didn't work** — the `/trace` command errored saying LangSmith tracing wasn't configured. The **correct** variables are **`LANGSMITH_TRACING=true`** and **`LANGSMITH_API_KEY`** (the older `LANGCHAIN_TRACING` / `LANGCHAIN_API_KEY` names are the legacy form the docs still showed). This matches this repo's convention of overriding `LANGSMITH_PROJECT` per section. Verify in-CLI with `/trace`.

### What the trace reveals

**Prompt "hello" (no skill needed):**
- The system prompt already contains an **"available skills"** section listing the three skills' **metadata only** (name, description, location) — *not* their full content.
- It also contains a **"how to use skills"** instruction block explaining progressive disclosure:
  > *"Skills follow a progressive disclosure pattern. You see their name and description above, but you read the full instructions only when needed."*
  Steps the prompt gives the LLM: **(1) recognize when a skill applies → (2) read the skill's full instructions → (3) follow them → (4) access supporting files.**
- Because "hello" needs no skill, **no skill content is loaded** — only metadata was present.

**Prompt "create a GIF on agent skills with Remotion" (skill needed):**
1. The LLM sees the metadata, **recognizes** the Remotion skill applies, and calls the **`read_file`** tool on `remotion-best-practices/SKILL.md`.
2. That content enters context. On the next call, the LLM chooses to read **more** files — `animations.md`, `compositions.md`, `timings.md` — and later text-animation / sequencing rules.
3. **Notably, the LLM did *not* read `gifs.md`** even though the task was a GIF — proving **file selection is the LLM's responsibility**, not the harness's. (Eden flags this as both the beauty and the risk of the design.)
4. With richer context, the agent renders the MP4, then converts it to a GIF.

> 🔑 **The critical distinction Eden draws:**
> - **`before_agent` (discovery)** merely *finds* the skills and stores their metadata in state. It does **not** inject into the prompt.
> - **`wrap_model_call` (injection)** is what actually *appends the `skills-system` section to the system prompt* before each call.
> Both run, but only the second one writes the skills index into the LLM's context.

---

## 5. The Two Middlewares (Recap, Ch. 156)

Eden recaps with the **ReAct agent loop** and shows where skills hook in.

**The standard loop** (see `assets/ReAct_Standard_Agent_Loop.png`):

```
   ┌──────────► LLM call (with question) ──────────┐
   │                    │                          │
   │             decide: tool?                     │
   │             ┌──────┴───────┐                  │
   │            yes             no                 │
   │             │               └──► final answer │
   │      execute tool ──────────────────► reason ─┘
```

**With the skills middleware added** (see `assets/ReAct_Loop_With_Skills_Middleware.png`):

| # | Hook | When it runs | What it does |
|---|------|-------------|--------------|
| 1 | **`before_agent` middleware** (discovery) | **Once**, when the session loads | Loads all available skills into agent memory/state: names + locations. This is *skill discovery*. |
| 2 | **`wrap_model_call` middleware** (injection) | **Before every LLM call** | Appends the *skills-system appendix* to the system prompt: the full list of skills, their locations, and progressive-disclosure instructions. |

After injection, the decision to load a skill (or use a normal tool) belongs to the **LLM**. This is the exact behavior seen in the Layer 2 traces, and it mirrors how Claude Code / Gemini CLI / Manus behave.

---

## 6. Layer 3 — Inside `skills.py` (Ch. 157)

The deepest layer: the real source at
[`libs/deepagents/deepagents/middleware/skills.py`](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/middleware/skills.py) (~800 lines).

### 6.1 `before_agent` — discovery (runs once per session)

```
before_agent(state):
	if state already has skill metadata:  return   # skip — load once per session
	sources = [project-level paths, user-level paths, other backends...]
	for source in sources:
		skills = list_skills(source)   # scan subdirs, read files, parse YAML frontmatter
	metadata = { skill_name: skill_data, ... }
	state.SkillsState = list(metadata.values())     # store in agent state
```

Key facts from the code:

- **Loads once per session.** If skill metadata already exists in state, it's skipped.
- **Source order matters:** skills are loaded in source order, and **later sources override earlier ones** if they share the same skill name. (An explicit LangChain engineering choice — Eden notes it's neither obviously good nor bad, just a design decision to be aware of.)
- **`backend` abstraction:** skills are read through a pluggable backend. For the CLI it's the **local filesystem**, but it could be **Firestore, Bigtable**, or any cloud store — enabling **cloud-based skills**.
- **`list_skills`** does the "ugly but unremarkable" work: scan subdirectories, read content, parse YAML frontmatter, return metadata. *No sophisticated logic — just filesystem traversal + parsing.*
- The result is a **dictionary** `{name → metadata}`, whose **values** become a list stored in state. In the LangSmith trace this appears as the `skill-metadata` field on `skills-middleware before_agent`, containing the three configured skills.

### 6.2 `wrap_model_call` → `modify_request` — injection (runs before every call)

```
wrap_model_call(request):
	request = modify_request(request)   # inject skills into the system prompt
	return call_model(request)

modify_request(request):
	metadata = state skill metadata (names + SKILL.md paths + descriptions)
	skills_section = SKILLS_SYSTEM_PROMPT.format(
		skill_locations = <paths>,
		skills_list     = <names + descriptions>,
	)
	request.system_prompt += skills_section
	return request
```

Key facts:

- `wrap_model_call` **takes the outgoing request and modifies it**, then sends the modified version.
- `modify_request` pulls skill metadata (names, descriptions, `SKILL.md` locations) from state and fills the **`SKILLS_SYSTEM_PROMPT`** template — boilerplate containing the progressive-disclosure instructions plus placeholders `skill_locations` and `skills_list`.
- This appended block is the **`skills-system`** section visible in the trace's system prompt — present on skill-needing calls, absent before injection.
- **The LLM decides everything downstream.** The harness only *prepares and presents* the metadata; choosing which files to progressively disclose is entirely the LLM's job.

> 💡 **Eden's "most beautiful thing":** The elegance isn't clever code — most of `skills.py` is filesystem traversal + YAML parsing. It's the **division of responsibility**: the harness makes metadata *present and easy to navigate*, and the *LLM* owns the decision of which files to open. Good `SKILL.md` "index" design is therefore what drives good results.

---

## 7. The Full Progressive Disclosure Flow

Putting all three layers together:

```
SESSION START
   │
   ▼
[before_agent middleware]  ── DISCOVERY (once) ─────────────────┐
   • scan sources via backend                                   │
   • parse SKILL.md frontmatter                                 │
   • store {name → metadata} in agent state                     │
																│
   ┌────────────────────────────────────────────────────────────┘
   │
   ▼   (repeats every turn)
[wrap_model_call → modify_request]  ── INJECTION (every call) ──┐
   • fill SKILLS_SYSTEM_PROMPT (list + locations + how-to)       │
   • append to system prompt                                     │
   ┌─────────────────────────────────────────────────────────────┘
   │
   ▼
LLM CALL
   • sees skill METADATA only (small context)
   • recognizes a skill applies?
		│no ──► answer / use normal tool
		│yes
		▼
   read_file(SKILL.md)  ← DISCLOSURE of full instructions
		▼
   LLM chooses which supporting files to read  ← LLM's responsibility
		▼
   read_file(rules/animations.md), read_file(assets/...), ...
		▼
   execute the task (e.g., render the Remotion video)
```

**One-line summary:** *Metadata eager, content lazy, file-selection delegated to the LLM.*

---

## 8. C# Analogy

The progressive-disclosure pattern maps cleanly onto lazy-loading idioms a .NET engineer already knows:

| Deep Agents concept | C# / .NET analogy |
|---------------------|-------------------|
| Skill metadata loaded at startup | An **index / manifest** (e.g., `IReadOnlyList<SkillDescriptor>`) with just names + descriptions + paths |
| Full skill content loaded on demand | **`Lazy<T>`** — the body is only materialized when first accessed |
| `SKILL.md` frontmatter as the "index card" | Assembly **metadata / attributes** you reflect over before deciding to load the type |
| `backend` abstraction (filesystem/Firestore/Bigtable) | A **repository / provider interface** (`ISkillBackend`) with swappable implementations |
| `before_agent` discovery (runs once) | A **singleton cache** populated once, guarded by a "already loaded?" check |
| `wrap_model_call` injection (every call) | **Middleware / a delegating handler** that decorates the outgoing request each time |
| LLM chooses which files to read | The **consumer** decides which lazy members to dereference; the provider just exposes them |

> ⚠️ Analogy is for intuition only — the LLM's file choice is probabilistic, not deterministic like `Lazy<T>` access.

---

## Interview Q&A Anchors

**Q: What is an agent "skill" and what's in it?**
> A skill is a directory following the Agent Skills standard, centered on a `SKILL.md` file. The frontmatter holds metadata (name, description, when-to-use); the body and linked files (rules, scripts, assets) hold the full instructions. The metadata is loaded eagerly into the prompt; the content is read lazily only when a task needs it.

**Q: Explain progressive disclosure.**
> It's a two-tier loading strategy: the harness injects only skill *metadata* (name + description + location) into the system prompt so the LLM knows what's available, and the full skill *content* is read on demand via `read_file` when the LLM decides a skill applies. This keeps the context window small while making many rich skills available.

**Q: In Deep Agents, which component actually puts skills into the prompt?**
> Two middlewares split the job. `before_agent` runs once per session and *discovers* skills, storing their metadata in agent state — it does **not** touch the prompt. `wrap_model_call` (via `modify_request`) runs before **every** LLM call and *injects* the `SKILLS_SYSTEM_PROMPT` (list, locations, instructions) into the system prompt. Injection is the one that reaches the LLM.

**Q: Who decides which skill files get read?**
> The **LLM**, not the harness. The harness only makes metadata present and navigable. That's why `SKILL.md` should be structured like a clear index — the model uses it to choose which supporting files to progressively disclose. Eden even shows the LLM skipping the obvious `gifs.md` for a GIF task, underscoring that selection is model-driven.

**Q: Why does Deep Agents being open source matter here?**
> Claude Code, Cursor CLI, Gemini CLI, and Manus implement the same skill concept but are closed source. Deep Agents' `skills.py` (~800 lines) lets us verify the exact mechanism — discovery, injection, and delegation — so we understand the pattern used across all of them.

**Q: What does the `backend` abstraction buy you?**
> Skills are read through a pluggable backend, so they don't have to live on the local filesystem — they can be served from Firestore, Bigtable, or any custom store. This makes cloud-hosted, centrally managed skills possible without changing the disclosure logic.

**Q: How do skills differ from memory in Deep Agents?**
> Skills use progressive disclosure — metadata loaded eagerly, content lazily — via `SKILL.md`. Memory uses `AGENTS.md` files that are **always** fully loaded. Skills are on-demand capabilities; memory is persistent, always-on context.

---

## References

- [Deep Agents Overview (official docs)](https://docs.langchain.com/oss/python/deepagents/overview) — prerequisite: [31. Official Overview Reference](./31_Deep_Agents_Official_Overview_Reference.md)
- [Deep Agents `skills.py` source](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/middleware/skills.py) — the ~800-line implementation Eden reviews
- [Deep Agents repository](https://github.com/langchain-ai/deepagents/)
- [Agent Skills standard](https://agentskills.io/)
- [Remotion docs](https://www.remotion.dev/docs) · [Remotion repo](https://github.com/remotion-dev/remotion) · [Remotion AI skills](https://www.remotion.dev/docs/ai/skills)
- Companion: [29. Deep Agents — Theory & Concepts](../21-deep-agents/29_Deep_Agents_Theory_And_Concepts.md) · [30. Deep Agents — Eden's Course Notes](../21-deep-agents/30_Deep_Agents_Eden_Course_Notes.md)
