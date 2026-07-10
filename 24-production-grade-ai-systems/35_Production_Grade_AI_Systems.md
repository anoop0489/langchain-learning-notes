# 35. Production-Grade AI Agents — The Complete Engineering Guide

> **Context:** Section 24. This is a **comprehensive, opinionated field guide** to everything that matters when you build and operate a production-grade AI agent. It is written from first principles — *"if I were the engineer responsible for shipping this, what would I have to get right?"* — and pulls together architecture, retrieval, memory, evaluation, safety, ops, cost, and organizational concerns into one place. It is intentionally broad: use the Table of Contents to jump to what you need. The goal is that you can **design, defend, and operate** a real system, and answer any interview question on the topic with depth.

> 💡 **The thesis.** Building an agent that *works in a demo* is easy. Building one that is **reliable, safe, observable, affordable, and trusted** at scale is where most of the effort goes — and, as a rough rule of thumb, only a small fraction of that effort is the "AI" part. It's engineering discipline applied to a non-deterministic component. (The "20% logic / 80% platform" split below is a mental model, not a measured statistic.)

---

## Table of Contents

| # | Section | What You'll Learn |
|---|---------|-------------------|
| 1 | [First Principles: What Makes AI Systems Hard](#1-first-principles-what-makes-ai-systems-hard) | Why non-determinism changes everything |
| 2 | [The Production Agent Stack](#2-the-production-agent-stack) | The full layered architecture |
| 3 | [Choosing an Architecture](#3-choosing-an-architecture) | Workflow vs agent, single vs multi-agent |
| 4 | [Models & the Model Layer](#4-models--the-model-layer) | Selection, routing, fallbacks, fine-tuning |
| 5 | [Prompting & Context Engineering](#5-prompting--context-engineering) | Prompts, context windows, context rot |
| 6 | [Tools & Function Calling](#6-tools--function-calling) | Designing reliable tool interfaces |
| 7 | [Retrieval (RAG → Semantic Search + Ranking)](#7-retrieval-rag--semantic-search--ranking) | The retrieval subsystem in depth |
| 8 | [Memory & State](#8-memory--state) | Short-term, long-term, persistence |
| 9 | [Orchestration & Control Flow](#9-orchestration--control-flow) | Loops, HITL, durability, interrupts |
| 10 | [The AI Gateway](#10-the-ai-gateway) | Central control plane for model traffic |
| 11 | [Observability & Tracing](#11-observability--tracing) | Monitoring built for agents |
| 12 | [Evaluation (Evals)](#12-evaluation-evals) | Offline, online, LLM-as-judge |
| 13 | [Reliability & Resilience](#13-reliability--resilience) | Retries, fallbacks, idempotency |
| 14 | [Safety, Security & Guardrails](#14-safety-security--guardrails) | Injection, PII, permissions, abuse |
| 15 | [Cost, Latency & Scale](#15-cost-latency--scale) | Making it fast and affordable |
| 16 | [User Trust & Product Design (CAIR)](#16-user-trust--product-design-cair) | Perceived reliability |
| 17 | [Feedback Loops & Continuous Improvement](#17-feedback-loops--continuous-improvement) | Learning from real usage |
| 18 | [Deployment & MLOps/LLMOps](#18-deployment--mlopsllmops) | CI/CD, versioning, rollout |
| 19 | [Data, Privacy & Governance](#19-data-privacy--governance) | Compliance, residency, retention |
| 20 | [Team, Process & Anti-Patterns](#20-team-process--anti-patterns) | How to actually ship |
| 21 | [Production Readiness Checklist](#21-production-readiness-checklist) | The pre-launch gate |
| 22 | [Interview Q&A Anchors](#interview-qa-anchors) | Quick-fire answers |
| 23 | [References](#references) | Docs & related notes |

---

## Key Definitions

| Term | Quick Recall | Full Definition |
|------|-------------|----------------|
| **Agent** | LLM in a loop with tools | A system where an LLM decides *which actions to take* (tool calls) and iterates on results until a goal is met. |
| **Workflow** | Predefined steps | A deterministic pipeline where control flow is coded, and the LLM fills specific steps. Lower risk than a full agent. |
| **AI Gateway** | Control plane for models | Single entry point enforcing guardrails, permissions, routing, fallbacks, cost/quota control across providers. |
| **Observability** | Trace intent, not clicks | AI-specific monitoring: full run traces, per-step tokens/latency, and success measured against *user intent*. |
| **Eval** | Automated quality gate | A test suite (offline datasets + online metrics, often LLM-as-judge) that gates every release. |
| **CAIR** | User trust metric | *Confidence in AI Results* = `Value ÷ (Risk × Effort to fix)`; mostly a product-design lever. |
| **Context engineering** | Curate the window | The discipline of putting *exactly the right* tokens in the context window — no more, no less. |
| **HITL** | Human-in-the-loop | A pause point where a human approves/edits before a high-stakes action executes. |
| **Guardrail** | Input/output safety check | Programmatic checks around the model (injection filters, PII redaction, output schema validation, moderation). |

---

## 1. First Principles: What Makes AI Systems Hard

Traditional software is **deterministic**: same input → same output, and you test it exhaustively. An LLM-based system is fundamentally different, and every production concern flows from these properties:

1. **Non-determinism.** The same prompt can yield different outputs across calls. Setting `temperature=0` *reduces* variance but does **not** guarantee identical outputs (mixture-of-experts routing, floating-point/batching effects, and silent provider updates all contribute). You cannot rely on exact-match tests; you need *evaluation* (statistical/semantic) instead of assertions.
2. **Probabilistic correctness.** The model is *usually* right, not *always* right. Design assuming it will be wrong some percentage of the time — and make wrong answers cheap to detect and recover from.
3. **Unbounded input space.** Users type free-form natural language. You cannot enumerate all inputs, so you cannot enumerate all failure modes.
4. **Opaque reasoning.** The model is a black box; you can't step through its "logic." This makes observability and explainability essential, not optional.
5. **Cost and latency scale with tokens.** Every decision (more context, more steps, bigger model) has a direct dollar and millisecond cost.
6. **The model is a moving target.** Providers deprecate, update, and re-tune models. Your "tested" behavior can change under you.
7. **Adversarial surface.** Natural-language input means prompt injection and jailbreaks are a first-class security threat with no perfect defense.

> **Mindset shift:** You are not "calling a function." You are integrating a *stochastic, adversarially-exposed, externally-controlled component* into a system that must still be reliable. Everything below is how you tame that.

---

## 2. The Production Agent Stack

A production agent is a **custom core wrapped in a standard platform**:

```text
   ┌──────────────────────────────────────────────────────────┐
   │  PRODUCT / UX LAYER                                       │  trust, previews, approvals,
   │  (approval gates, previews, streaming, feedback UI)       │  transparency, undo
   ├──────────────────────────────────────────────────────────┤
   │  GUARDRAILS (input + output)                             │  injection filter, PII, moderation,
   │                                                          │  schema validation
   ├──────────────────────────────────────────────────────────┤
   │  AGENT / ORCHESTRATION LOGIC          ← the "custom" 20% │  planning, control flow,
   │  (LangGraph state machine, HITL, loops)                  │  tool selection
   ├───────────────┬───────────────┬──────────────┬───────────┤
   │   MEMORY      │   RETRIEVAL    │    TOOLS     │  AI GATEWAY│  ← the "standard" 80%
   │ (state/store) │ (search+rank)  │ (APIs/MCP)   │ (routing)  │
   ├───────────────┴───────────────┴──────────────┴───────────┤
   │  OBSERVABILITY + EVALS + COST TRACKING (cross-cutting)    │
   ├──────────────────────────────────────────────────────────┤
   │  INFRA (compute, queues, vector DB, cache, secrets)      │
   └──────────────────────────────────────────────────────────┘
```

The **agent logic** is your differentiation. Everything around it is the same problem every team solves — so lean on frameworks (LangChain/LangGraph) and managed services for the platform, and spend your creativity on the core.

---

## 3. Choosing an Architecture

Not everything should be a fully autonomous agent. **Match the pattern to the risk and complexity.**

| Pattern | When to use | Trade-off |
|---------|-------------|-----------|
| **Single LLM call** | Simple transform (classify, extract, summarize) | Cheapest, most predictable; limited capability. |
| **Chain / Workflow** | Known, fixed sequence of steps | Deterministic control flow, easy to test; can't adapt. |
| **Router** | Branch to different handlers by intent | Predictable; needs good intent classification. |
| **Single agent (tool loop)** | Open-ended tasks needing tools + iteration | Flexible; less predictable, must bound the loop. |
| **Multi-agent** | Distinct specialized roles / parallel subtasks | Powerful; adds coordination cost, latency, and failure modes. |

**Guiding rules:**
- **Prefer the least-agentic option that solves the problem.** Every degree of autonomy adds unpredictability, cost, and attack surface.
- **Workflows for reliability, agents for flexibility.** If the steps are known, hard-code them and let the LLM fill nodes — don't make the model re-derive the plan every time.
- **Multi-agent is not free.** Only reach for it when roles are genuinely distinct or you need parallelism; otherwise it multiplies latency and error propagation.
- **Bound every loop.** Max iterations, max tool calls, max wall-clock, max cost per run.

---

## 4. Models & the Model Layer

**Selection criteria** (evaluate on *your* task, not benchmarks alone):
- Capability (reasoning depth, instruction following, tool-calling quality)
- Context window size vs your real context needs
- Latency and throughput (tokens/sec, time-to-first-token)
- Cost per input/output token
- Modality (text, vision, audio) needs
- Data/privacy terms (does the provider train on your data? residency?)
- Open-weight (self-host) vs API (managed) — control vs convenience

**Practical model strategy:**
- **Tiered routing.** Use small/cheap models for easy steps (classification, routing, extraction) and reserve frontier models for hard reasoning. This is often the single biggest cost lever.
- **Fallbacks.** Always have a secondary provider/model for when the primary is rate-limited or down.
- **Abstract the provider.** Never hard-code one model in business logic — go through the gateway (Section 10) so you can swap without code changes.
- **Fine-tuning vs prompting vs RAG.** Prefer prompting + RAG first (fast, cheap, updatable). Fine-tune only when you need consistent format/style/tone or to compress a large stable instruction set — not to inject *knowledge* (that's what RAG is for).
- **Pin and test versions.** Treat a model version like a dependency; when the provider updates it, re-run evals before trusting it.

---

## 5. Prompting & Context Engineering

The context window is the model's entire working memory for a call. **Context engineering** — deciding exactly what goes in it — is the highest-leverage skill.

**Prompt structure that holds up in production:**
- Clear **role/system** instruction; keep it stable and versioned.
- **Explicit output contract** (format, schema, constraints) — and validate it.
- **Few-shot examples** for tricky formats/edge cases (but watch token cost).
- **Delimit** untrusted/user/retrieved content clearly so it can't be confused with instructions.
- **Instructions near the end** for long contexts (recency bias helps).

**Context window pitfalls:**
- **"Lost in the middle."** Models attend best to the start and end; critical info buried mid-context is often ignored. Place key facts strategically.
- **Context rot / dilution.** More tokens ≠ better. Irrelevant context degrades output (garbage in, garbage out) and raises cost/latency. Retrieve and include *only* what's needed.
- **Token budgeting.** Reserve room for the response and for tool-call round-trips; don't fill the window to the brim.

**Manage prompts like code:** version them, review changes, and gate prompt edits behind evals — a one-word prompt change can regress quality.

---

## 6. Tools & Function Calling

Tools are how an agent affects the world. Their design largely determines reliability.

**Designing good tools:**
- **Clear, descriptive names + docstrings/schemas.** The model chooses tools from these; ambiguity causes wrong calls.
- **Narrow, single-purpose tools** beat mega-tools with many modes.
- **Strong input schemas** (Pydantic/JSON Schema) with validation; reject bad args early.
- **Structured, informative results** — including actionable errors the model can recover from ("date must be YYYY-MM-DD" beats "400").
- **Idempotency** for anything with side effects, so retries don't double-charge/double-send.
- **Least privilege.** Give the agent the narrowest capability that works; gate destructive actions behind confirmation.

**Operational concerns:**
- **Timeouts and error handling** on every tool — a hung tool hangs the agent.
- **Return errors to the model** so it can retry/adapt, but cap retries to avoid loops.
- **MCP (Model Context Protocol)** is emerging as a standard way to expose tools/resources to agents across systems — worth adopting for interoperability.
- **Guard side effects.** Distinguish read-only tools (safe to auto-run) from write/destructive tools (require approval).

---

## 7. Retrieval (RAG → Semantic Search + Ranking)

Retrieval quality is frequently the biggest lever on answer quality. Naive RAG (embed → top-k cosine → stuff) is the *start*, not the finish.

**The full retrieval pipeline:**
1. **Ingestion & chunking.** Chunk by *semantic structure* (headings, paragraphs) not fixed length; tune size/overlap; attach rich **metadata** (source, section, timestamp, tenant, permissions).
2. **Embeddings.** Choose a model matched to your domain/language; keep embedding version consistent across index and queries.
3. **Retrieval.** **Hybrid** (dense vector + sparse/BM25 keyword) beats either alone — semantic recall plus exact-term precision.
4. **Reranking.** A cross-encoder/reranker reorders candidates by true relevance so the *best* few land in the limited prompt budget.
5. **Metadata filtering & access control.** Filter by tenant/permissions/recency *before* results reach the model — critical for security and correctness.
6. **Query transformation.** Rewrite/expand queries, decompose multi-part questions, handle follow-ups (coreference) before retrieval.
7. **Grounding & citations.** Return sources so answers are verifiable and to reduce hallucination.

**Operate retrieval as its own monitored subsystem:** track recall, precision, and groundedness/faithfulness; iterate on chunking, embeddings, and rerankers like any service.

**Advanced patterns:** parent-document retrieval, contextual retrieval (prepend chunk context before embedding), graph/structured retrieval, and indexing with a record manager + incremental cleanup to avoid duplicate/costly re-embedding.

---

## 8. Memory & State

Memory turns a stateless LLM into something continuous and personal.

| Type | What it is | Backed by |
|------|-----------|-----------|
| **Short-term (thread) memory** | Current conversation state | LangGraph **state + checkpointer + `thread_id`** so a thread can resume. |
| **Long-term memory** | Durable facts across sessions/users | A **store** (namespace + key): preferences, learned corrections, profiles. |

**Strategies to stay within budget (and improve quality):**
- **Save-all** — simplest; fine for short chats.
- **Trimming** — drop oldest messages by token/message count.
- **Summarization** — compress history into a running summary + keep recent turns.
- **Semantic/long-term recall** — store facts in a vector/store and retrieve relevant ones on demand.

**Production concerns:**
- **Persist deliberately** — in-memory saver for dev; **Postgres / Redis / MongoDB** checkpointers for prod durability.
- **Tenant isolation** — namespace long-term memory rigorously; "cross-company context" is powerful but a data-leak risk.
- **Right-size** — even huge context windows aren't a reason to dump everything; it costs more, is slower, and dilutes quality.

> Deeper treatment: **[33. Memory & Context Reference](../23-langchain-glossary/33_Memory_And_Context_Reference.md)**.

---

## 9. Orchestration & Control Flow

How the agent *runs* is as important as what it decides.

- **State machine over free-form loop.** A graph (LangGraph `StateGraph`) with explicit nodes/edges is far more debuggable and controllable than an opaque while-loop. You can inspect, checkpoint, and resume.
- **Bounded execution.** Enforce max steps, max tool calls, max cost, and wall-clock timeouts. Detect and break loops (e.g., repeated identical tool calls).
- **Human-in-the-loop (HITL).** For high-stakes actions, pause and require approval/edit before executing. LangGraph **interrupts** + checkpointers make this durable.
- **Durability & resumability.** Checkpoint state so a crashed/long-running run resumes instead of restarting — essential for long tasks and reliability.
- **Streaming.** Stream tokens and intermediate steps to the UI for perceived speed and transparency.
- **Concurrency.** Parallelize independent branches (fan-out/fan-in); be careful with shared state and reducers.

---

## 10. The AI Gateway

A single choke point every model call passes through — the "API gateway" of the LLM era. Centralize what you don't want scattered across services:

- **Guardrails & permissions** — who may call which model with which data.
- **Prompt security** — injection/jailbreak filtering, PII redaction before egress.
- **Smart routing** — route by use case/scale/cost/availability across providers and model tiers, with automatic **fallbacks** when a model throttles or fails.
- **Resilience** — retries, timeouts, circuit breakers.
- **Cost & quota control** — budget caps, per-tenant quotas, caching, cheaper models for cheap tasks.
- **Central observability & audit** — one place to log every model interaction.

> **Design rule:** Assume any single model will fail, throttle, or be deprecated. Route through the gateway so business logic never hard-codes a provider.

---

## 11. Observability & Tracing

Standard APM (status codes, page views) is insufficient because agents are **non-deterministic, multi-step, and driven by natural-language intent**.

**Instrument:**

| Signal | Why |
|--------|-----|
| Full run traces (prompts, tool I/O, sub-agent calls) | Debug *why* an action happened, and replay runs. |
| Tokens per step/run | Cost attribution, runaway-loop detection. |
| Latency per step (incl. time-to-first-token) | Find the slow model/tool in a chain. |
| Success vs *user intent* | Did we actually solve the request? |
| Model + prompt + code version | Correlate regressions to releases. |
| Tool success/error rates | Spot flaky integrations. |
| User feedback (thumbs, edits) | Ground truth for quality. |

**Principle:** if you can't **replay a full agent run**, you can't operate it. LangSmith is purpose-built (trace trees, per-step tokens/latency, dataset+eval integration); OpenTelemetry-based tracing is the vendor-neutral counterpart.

---

## 12. Evaluation (Evals)

Because you can't unit-test non-determinism, **evals are your quality gate and the heart of iteration.**

**Types:**
- **Offline evals** — run candidate prompts/models/agents against a **curated dataset** of critical cases before shipping; gate releases on results (AI's CI).
- **Online evals** — measure quality on live traffic (sampling, user feedback, guard metrics).
- **Component vs end-to-end** — evaluate retrieval (recall/precision/groundedness), individual tools, *and* the full task outcome.

**Methods:**
- **Reference-based** — compare to golden answers (exact match, semantic similarity).
- **LLM-as-judge** — a model scores outputs on rubrics (helpfulness, correctness, faithfulness). Powerful but needs its own validation and calibration against humans.
- **Human review** — the ultimate ground truth; sample and label continuously to grow your eval set.

**Practices:** build the dataset from *real failures*, version it, track metrics over time, and never ship a prompt/model/agent change that regresses core cases.

---

## 13. Reliability & Resilience

Engineering discipline around a fallible component:

- **Retries with backoff** on transient errors (rate limits, timeouts) — with jitter and caps.
- **Fallbacks** — secondary model/provider, degraded-but-useful responses, cached answers.
- **Circuit breakers** — stop hammering a failing dependency; fail fast and recover.
- **Timeouts everywhere** — per model call, per tool, per run.
- **Idempotency** — for side-effecting operations so retries are safe.
- **Graceful degradation** — when AI is unavailable, degrade to a safe default rather than a hard error.
- **Rate limiting & backpressure** — protect your own system and downstream providers.
- **Dead-letter / audit** — capture failed runs for later analysis and replay.

---

## 14. Safety, Security & Guardrails

Natural-language input makes this a first-class, adversarial concern with **no perfect defense** — defend in depth.

- **Prompt injection & jailbreaks.** Untrusted content (web pages, docs, tool outputs, user input) can carry instructions. Mitigate: clearly delimit/label untrusted text, never let retrieved content override system instructions, constrain tool permissions, and treat model output feeding tools as untrusted.
- **Input guardrails** — moderation, injection detection, PII detection/redaction before the model sees or emits data.
- **Output guardrails** — schema/format validation, content moderation, groundedness checks, sanitize anything passed to downstream systems (SQL, shell, HTML → injection risks).
- **Least-privilege tools & data access** — scope every capability; gate destructive actions behind HITL; enforce per-tenant/per-user authorization on retrieval and tools.
- **Secrets & keys** — never in prompts or code; use a secrets manager; rotate.
- **Data exfiltration** — an injected instruction may try to make the agent leak data via a tool; restrict egress and monitor.
- **Abuse & cost attacks** — rate-limit, quota, and detect prompt-bombing that runs up your bill.
- **Auditability** — log security-relevant events for incident response and compliance.

---

## 15. Cost, Latency & Scale

- **Right-size the model per task** — cheap models for easy steps; frontier models only where needed.
- **Control context size** — trimming/summarization/retrieval precision cut cost *and* latency and often improve quality.
- **Cache** — semantic/exact prompt caching, embedding caching, retrieval caching; use provider prompt-caching where available.
- **Stream** — token-by-token streaming for perceived latency; stream intermediate steps too.
- **Parallelize** — fan out independent LLM/tool calls (map-reduce style) instead of serial chains.
- **Batch** — batch embeddings and offline jobs.
- **Reduce round-trips** — fewer, well-designed tool calls beat many chatty ones.
- **Budget & monitor** — per-request and per-tenant cost tracking with alerts; a runaway loop can be very expensive fast.

---

## 16. User Trust & Product Design (CAIR)

Reliability has two halves that most teams conflate:

| | **Technical reliability** | **Perceived reliability** |
|--|---------------------------|----------------------------|
| Question | Does it stay up and behave correctly? | Do *users feel* they can trust it? |
| Levers | Retries, fallbacks, timeouts, tested tools | Explainability, transparency, feedback, product design |

A technically flawless agent that *feels* like an unpredictable black box won't get adopted. The best lens for the second column is **CAIR — Confidence in AI Results** (Assaf Elovic; see **[14. LLM Apps in Production](../12-llm-apps-in-production/14_LLM_Apps_In_Production.md)**):

```text
			  Value
CAIR = ─────────────────────
		Risk  ×  Effort to fix
```

CAIR is mostly a **product-design** metric — raise it by lowering risk and fix-effort, not only by improving the model:

| Pattern | Effect |
|---------|--------|
| **Preview / dry-run** before applying changes | ↓ Risk |
| **Approval gates** (HITL) for high-stakes actions | ↓ Risk |
| **Undo / rollback** | ↓ Effort to fix |
| **Explainability** — show *how* an action was reached | ↑ Trust on errors |
| **Transparency** — show *what* tools/data were used, with citations | ↑ Trust |
| **Confidence signals / graceful "I don't know"** | ↑ Trust vs confident wrongness |

> An AI feature can jump from medium to high adoption **without changing the model at all** — just by adding a preview so users approve changes before they take effect.

---

## 17. Feedback Loops & Continuous Improvement

Systems that improve from real usage win over time. The loop:

1. **Capture** — thumbs up/down, edits, corrections, and implicit signals (did they accept the suggestion?).
2. **Store** — as eval data, and/or as durable memory.
3. **Learn** — grow your eval dataset from real failures; adjust prompts/retrieval; update long-term memory.
4. **Re-inject** — feed learnings back into future runs.

**A lean feedback loop you can ship in a day:** keep a **Markdown file** of learned preferences/corrections (per-user or per-product), empty at first. When a user gives natural-language feedback, the **agent** (not the human) updates the file, and it's injected into every future task via **middleware** that runs before the LLM/tool call.

```text
[User feedback in natural language]
		  │
		  ▼
   Middleware hook (before LLM / before tool call)
		  │
		  ├─► read feedback.md ──► inject into context
		  └─► agent updates feedback.md with new learnings
		  │
		  ▼
   Agent runs the task with accumulated feedback
```

Two rules: **the agent owns the file, not the human** (users only speak natural language), and **implement it as middleware** so core logic is untouched. It's a no-vector-DB, no-fine-tune long-term-memory pattern.

---

## 18. Deployment & MLOps/LLMOps

- **Version everything** — prompts, models, tools, agent graph, retrieval config, and eval datasets. A change to any can shift behavior.
- **CI/CD with eval gates** — run offline evals in the pipeline; block merges/deploys that regress core cases.
- **Progressive rollout** — canary/shadow/A-B new prompts or models against live traffic before full rollout.
- **Rollback plan** — be able to instantly revert a prompt/model change (config-driven, not code-deploy).
- **Environment parity** — dev/staging/prod with representative data.
- **Reproducibility** — pin model versions; log full inputs/outputs to reconstruct any run. Provider `seed` parameters help but are best-effort only (see Section 1), so don't rely on them for exact reproduction.
- **Infra** — async workers/queues for long runs, autoscaling for spiky LLM latency, managed vector DB, caches, secrets manager.

---

## 19. Data, Privacy & Governance

- **Provider data terms** — know whether prompts are used for training; prefer zero-retention/enterprise terms for sensitive data.
- **PII handling** — minimize, redact, or tokenize before sending to models; classify data sensitivity.
- **Data residency & sovereignty** — pick regions/providers that meet legal requirements; consider self-hosted/open-weight models for strict cases.
- **Retention & deletion** — define how long traces, memory, and logs live; support user data-deletion (GDPR/CCPA).
- **Access control & audit** — enforce authz on data and tools; keep audit logs.
- **Compliance** — map to relevant frameworks (SOC 2, HIPAA, GDPR, EU AI Act) early, not after launch.
- **Bias, fairness & content policy** — evaluate for harmful/biased outputs; document limitations.

---

## 20. Team, Process & Anti-Patterns

**Process that works:**
- Start with the **simplest thing** (single call/workflow); add agency only when justified.
- **Build the eval set before scaling** — it's your source of truth and your regression net.
- **Instrument from day one** — you can't improve what you can't see.
- **Ship narrow, then widen** — a reliable narrow scope beats a broad flaky one.
- **Keep humans in the loop** for high-stakes actions until evals justify autonomy.

**Common anti-patterns:**
- ❌ Over-engineering to multi-agent when a workflow would do.
- ❌ "Vibes-based" evaluation (eyeballing a few outputs) instead of a real eval set.
- ❌ Dumping everything into context because the window is big.
- ❌ Hard-coding one model/provider.
- ❌ No bounds on the agent loop (cost/latency/infinite-loop risk).
- ❌ Trusting model output that feeds tools/SQL/shell without validation.
- ❌ Treating prompts as throwaway strings instead of versioned, evaluated artifacts.
- ❌ Fine-tuning to add knowledge (use RAG) or to fix a prompt you didn't iterate on.

---

## 21. Production Readiness Checklist

- [ ] **Architecture**: least-agentic pattern that solves the problem; loops bounded (steps/cost/time).
- [ ] **Models**: tiered routing, fallbacks, provider abstracted, versions pinned + eval-gated.
- [ ] **Prompts/context**: versioned, reviewed, eval-gated; context curated (no dumping).
- [ ] **Tools**: schemas + validation, idempotency, least privilege, timeouts, actionable errors.
- [ ] **Retrieval**: hybrid + rerank, metadata/permission filtering, groundedness measured.
- [ ] **Memory**: durable checkpointer backend, trimming/summarization, tenant isolation.
- [ ] **Orchestration**: state-machine, HITL on high-stakes, durable/resumable, streaming.
- [ ] **AI Gateway**: routing, fallbacks, rate-limit handling, budget caps, central audit.
- [ ] **Observability**: full replayable traces, per-step tokens/latency, intent-based success.
- [ ] **Evals**: offline dataset gating releases + online metrics; grown from real failures.
- [ ] **Reliability**: retries/backoff, circuit breakers, timeouts, graceful degradation.
- [ ] **Security**: injection defenses, input/output guardrails, PII, least privilege, secrets mgmt.
- [ ] **Cost/latency**: right-sizing, caching, streaming, parallelism, per-tenant budget alerts.
- [ ] **Trust/UX**: previews, approvals, undo, explainability, transparency, citations (CAIR).
- [ ] **Feedback**: mechanism to capture and re-apply user corrections.
- [ ] **Deploy/Ops**: version everything, CI eval gates, canary rollout, instant rollback.
- [ ] **Data/Governance**: data terms, residency, retention/deletion, compliance mapped.

---

## Interview Q&A Anchors

**Q: What separates a production-grade agent from a demo?**
> **A:** The demo is the agent logic; production is the platform around it — observability, an AI gateway, memory, retrieval, evals, guardrails, reliability, and trust-oriented UX. The clever reasoning is ~20% of the work; the standard infrastructure that keeps it observable, safe, affordable, and trusted is the other 80%.

**Q: When would you *not* build a full agent?**
> **A:** Whenever a simpler pattern solves it. If the steps are known, use a workflow and let the LLM fill specific nodes — it's deterministic, testable, cheaper, and safer. Reach for agency (open-ended tool loops) only when the task genuinely requires the model to decide the path, and always bound the loop.

**Q: How do you test a non-deterministic system?**
> **A:** With evals, not assertions. Build a curated dataset of real/critical cases; run offline evals (reference-based + LLM-as-judge, validated against humans) to gate every release; add online evals and user feedback on live traffic; and evaluate components (retrieval, tools) as well as end-to-end outcomes.

**Q: Why is observability different for agents, and what do you capture?**
> **A:** Agents are non-deterministic, multi-step, and intent-driven, so status codes aren't enough. You capture full replayable run traces (prompts, tool I/O, sub-agents), per-step tokens/latency, model/prompt versions, and success measured against user intent. If you can't replay a run, you can't operate it.

**Q: What's your defense against prompt injection?**
> **A:** Defense in depth: delimit/label all untrusted content, never let retrieved/tool content override system instructions, enforce least-privilege tools and per-tenant authorization, add input/output guardrails (moderation, PII, schema validation), treat model output feeding tools/SQL/shell as untrusted, and restrict/monitor egress. There's no perfect fix, so you also detect and contain.

**Q: How do you control cost and latency?**
> **A:** Tiered model routing, ruthless context curation (retrieve only what's needed), caching (prompt/embedding/retrieval), streaming for perceived speed, parallelizing independent calls, bounding the loop, and per-tenant budget monitoring with alerts.

**Q: How do you make users *trust* an agent?**
> **A:** Separate technical from perceived reliability. For perceived trust, use CAIR (Value ÷ (Risk × Effort to fix)) and lower risk/fix-effort via product design: previews, approval gates, undo, plus explainability, transparency, and citations. You often raise adoption without changing the model at all.

**Q: RAG or fine-tuning?**
> **A:** RAG for knowledge (fresh, updatable, cite-able, cheaper) — it's the default for grounding in your data. Fine-tune for consistent format/style/behavior or to compress stable instructions, not to inject facts. Most production systems are prompting + RAG first, fine-tuning only when a measured need remains.

---

## References

- **LangSmith (agent observability & evals)** — https://docs.langchain.com/langsmith
- **LangGraph (orchestration, HITL, persistence)** — https://langchain-ai.github.io/langgraph/
- **LangChain middleware** — https://docs.langchain.com/oss/python/langchain/middleware
- **LangChain memory concepts** — https://docs.langchain.com/oss/python/concepts/memory
- **Anthropic — Building effective agents** — https://www.anthropic.com/research/building-effective-agents
- **Model Context Protocol (MCP)** — https://modelcontextprotocol.io
- **OWASP Top 10 for LLM Applications** — https://owasp.org/www-project-top-10-for-large-language-model-applications/
- Related in this repo: [14. LLM Apps in Production — CAIR framework](../12-llm-apps-in-production/14_LLM_Apps_In_Production.md)
- Related in this repo: [33. Memory & Context Reference](../23-langchain-glossary/33_Memory_And_Context_Reference.md)
- Related in this repo: [Production Patterns reference guide](../reference-guides/Production_Patterns.md)
- Related in this repo: [RAG Architecture Decisions](../reference-guides/RAG_Architecture_Decisions.md)
