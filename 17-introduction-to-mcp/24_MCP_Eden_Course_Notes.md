# 24. MCP — Eden's Course Notes (Chapters 125–129)

> **Context:** Section 17, Chapters 125–129. Eden introduces MCP (Model Context Protocol) — why it exists, how tool calling actually works under the hood, the full architecture, and the server components. These notes validate and expand on Eden's transcript with accurate technical details.

---

## The Core Idea

> **Remember this, forget the rest.** MCP decouples tool execution from the agent. Instead of every AI application implementing its own tool integrations, you write tools once in an MCP server and any MCP-compatible client can use them. It's the same principle as adding a layer of abstraction — and it turns an N×M integration problem into an N+M one.

**The technique in one sentence:**

> "Write tools once, expose via MCP, connect from anywhere — Cursor, Claude Desktop, your own agent, whatever supports the protocol."

**The evolution that led to MCP:**

```
Level 1 — Custom tool integrations (pre-MCP):
  Each agent implements its own tool code.
  Want to support 5 apps? Write 5 integrations.
  (N clients × M tools = N×M custom integrations)

Level 2 — Framework tools (LangChain era):
  LangChain provides pre-built tool wrappers.
  Still coupled to the LangChain runtime.
  (Tools execute inside your application process)

Level 3 — MCP (current):
  Tools live in external servers.
  Any client speaks the protocol.
  Tools execute in the SERVER, not the client.
  (N + M integrations. Decoupled. Scalable.)
```

---

## Table of Contents

| # | Section | What You'll Learn |
|---|---------|-------------------|
| 1 | [Why MCP? — The Integration Problem (Ch. 125)](#1-why-mcp--the-integration-problem-ch-125) | The N×M integration explosion and how MCP solves it |
| 2 | [How LLMs Actually Use Tools (Ch. 126)](#2-how-llms-actually-use-tools-ch-126) | LLMs are token generators — tool calling is application-layer behaviour |
| 3 | [MCP Architecture & Protocol Flow (Ch. 127)](#3-mcp-architecture--protocol-flow-ch-127) | Host, Client, Server, and the full request lifecycle |
| 4 | [The Gist of the Protocol — Full Interaction (Ch. 128)](#4-the-gist-of-the-protocol--full-interaction-ch-128) | Step-by-step flow from user query to tool execution and response |
| 5 | [MCP Servers — The Three Interfaces (Ch. 129)](#5-mcp-servers--the-three-interfaces-ch-129) | Tools, Resources, Prompts, and the ecosystem |
| 6 | [Key Difference: LangChain vs MCP Tool Execution](#6-key-difference-langchain-vs-mcp-tool-execution) | Where tools run and why decoupling matters |
| 7 | [Interview Q&A Anchors](#interview-qa-anchors) | Quick-fire answers |

---

## Key Definitions

| Term | Quick Recall | Full Definition |
|------|-------------|----------------|
| **MCP** | USB-C for AI — universal tool protocol | An open standard that lets any AI application connect to any tool server without custom integration code. Write once, connect everywhere. |
| **MCP Host** | The AI app (Cursor, Claude Desktop, your agent) | The user-facing application that contains MCP client(s). Manages permissions, routes tool calls, and presents results to users. |
| **MCP Client** | Lives inside the host, talks to one server | A component within the host that maintains a 1:1 connection with a single MCP server. One host can have many clients (one per server). |
| **MCP Server** | Exposes tools/resources/prompts | A process that wraps functionality and exposes it via the MCP protocol. Can run locally (stdio) or remotely (HTTP). |
| **Tool (MCP)** | Model-controlled executable function | A function the LLM can decide to invoke. Executes on the SERVER, not in the host application. |
| **Resource (MCP)** | Application-controlled data | Data exposed to the AI system — can be static (PDFs, JSON) or dynamic (live DB queries). Application decides when to read, not the LLM. |
| **Prompt (MCP)** | User-controlled interaction template | Predefined templates for common interactions. Users invoke them to standardize complex workflows. |
| **Sampling** | Server requests the host to generate a completion | Allows the MCP server to ask the host's LLM to generate text — enables recursive agentic patterns but has security implications. |
| **N×M → N+M** | The integration math MCP solves | Without MCP: N clients × M tools = N×M integrations. With MCP: N clients + M servers = N+M. |
| **Tool calling** | LLM outputs function invocation instead of text | Ad hoc behaviour added via system prompts — the LLM generates structured text (function name + args) that the application parses and executes. |
| **Decoupled execution** | Tools run in server, not in your agent | MCP separates orchestration (agent decides WHAT to call) from execution (server RUNS the tool). Enables independent scaling, deployment, monitoring. |
| **Dynamic tool discovery** | Agent learns new tools at runtime | Because initialization can happen periodically, agents can receive new tools without redeployment. |

---

## 1. Why MCP? — The Integration Problem (Ch. 125)

### The Problem: Custom Integrations Don't Scale

Eden's core argument: if you build an AI agent with Slack, Gmail, and DB tools hardcoded for Cursor, then Windsurf wants the same capabilities, you must rewrite the integration. Add VS Code, Claude Desktop, Lovable, Bolt — you're writing the same tool code over and over.

```
┌─────────────────────────────────────────────────────────────┐
│ WITHOUT MCP (N × M integrations)                             │
│                                                             │
│  Cursor  ──custom──► Slack tool                             │
│  Cursor  ──custom──► Gmail tool                             │
│  Cursor  ──custom──► DB tool                                │
│  Windsurf──custom──► Slack tool (rewrite!)                  │
│  Windsurf──custom──► Gmail tool (rewrite!)                  │
│  Windsurf──custom──► DB tool   (rewrite!)                   │
│  Claude  ──custom──► Slack tool (rewrite again!)            │
│                                                             │
│  3 apps × 3 tools = 9 custom integrations                   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ WITH MCP (N + M integrations)                                │
│                                                             │
│  Cursor   ─┐                    ┌─ Slack MCP Server         │
│  Windsurf ─┼── MCP protocol ──►├─ Gmail MCP Server         │
│  Claude   ─┘                    └─ DB MCP Server            │
│                                                             │
│  3 apps + 3 servers = 6 integrations (each done ONCE)       │
└─────────────────────────────────────────────────────────────┘
```

### The Core Principle

> "If you want to solve a problem, add another layer of abstraction."

MCP is that abstraction layer. It standardizes:
- How tools are **discovered** (server tells client what's available)
- How tools are **invoked** (standard protocol message format)
- How results are **returned** (clean, LLM-friendly responses)

### The Network Effect

Eden compares MCP adoption to social media — value grows exponentially with users:
- More MCP servers exist → more reason for clients to support MCP
- More clients support MCP → more reason to write MCP servers
- As of 2025, thousands of community-built MCP servers exist

---

## 2. How LLMs Actually Use Tools (Ch. 126)

### The Fundamental Truth

> **LLMs are token generators. Nothing more.**

They cannot:
- Execute code
- Make HTTP requests
- Access databases
- Search the web

All of these "capabilities" are **application-layer features** that software engineers build around the LLM.

### How Tool Calling Works Under the Hood

```
┌──────────────────────────────────────────────────────────────┐
│ TOOL CALLING IS NOT MAGIC — IT'S A SYSTEM PROMPT TRICK       │
│                                                              │
│ Step 1: Application wraps user query + available tools       │
│         in a specially crafted prompt                         │
│                                                              │
│ Step 2: LLM generates one of two things:                     │
│         (a) A direct text answer, OR                         │
│         (b) A structured tool call (function name + args)    │
│                                                              │
│ Step 3: Application PARSES the output                        │
│         - If (a) → return to user                            │
│         - If (b) → execute the tool, feed result back to LLM │
│                                                              │
│ Step 4: LLM generates final answer using tool result         │
└──────────────────────────────────────────────────────────────┘
```

**Concrete example:**

```
User: "What's the weather in London?"

System prompt (simplified):
  "You have these tools: get_weather(city: str). 
   If you need real-time data, output a tool call instead of guessing."

LLM output (not a natural answer, but a structured call):
  {"tool": "get_weather", "args": {"city": "London"}}

Application: parses this → calls actual weather API → gets "15°C, cloudy"

Second LLM call:
  "User asked: What's the weather in London?
   Tool result: 15°C, cloudy.
   Generate a natural answer."

LLM output: "The weather in London is currently 15°C and cloudy."
```

### Key Points Eden Emphasises

1. **Tool calling is probabilistic** — LLMs are statistical, so they don't pick the right tool 100% of the time. But for production use, it works "most of the time" and is "pretty good for agentic applications."

2. **Each vendor implements tool calling differently** — OpenAI, Anthropic, Google all have slightly different formats for the tool call output. But they all follow the same principle: fancy system prompt → structured output → parse → execute.

3. **The ReAct prompt** (covered in Section 4) was an early version of this — Thought/Action/Observation loop encoded in a text prompt.

> ⚠️ **Transcript correction:** Eden says "tool calling does not work 100%" — this is accurate. In practice, tool selection accuracy varies by model: GPT-4o and Claude 3.5+ are highly reliable (>95% correct tool selection on well-described tools), while smaller models may require more explicit descriptions or few-shot examples.

### Why This Matters for MCP

MCP doesn't change how tool calling works in the LLM. It changes **where the tool execution happens** and **how tools are discovered**:

| Without MCP | With MCP |
|-------------|----------|
| Tools defined in your code | Tools defined in external servers |
| Tools discovered at compile time | Tools discovered at runtime (initialization) |
| Tools execute in your process | Tools execute in the server process |
| Change tools → redeploy agent | Change tools → redeploy server only |

---

## 3. MCP Architecture & Protocol Flow (Ch. 127)

### The Components Diagram

![MCP Components](./assets/MCP%20Components.png)

### The Three-Layer Architecture

```
┌─────────────────────────────────────────────────────────┐
│ MCP HOST (left side)                                     │
│ Examples: Claude Desktop, Cursor, Windsurf, your agent   │
│                                                         │
│  ┌─────────────┐  ┌─────────────┐                      │
│  │ MCP Client  │  │ MCP Client  │  (1:1 with servers)  │
│  └──────┬──────┘  └──────┬──────┘                      │
└─────────┼────────────────┼──────────────────────────────┘
		  │ MCP Protocol   │ MCP Protocol
		  ▼                ▼
   ┌──────────────┐ ┌──────────────┐
   │ MCP Server 1 │ │ MCP Server 2 │  (right side)
   │              │ │              │
   │ Exposes:     │ │ Exposes:     │
   │ • Tools      │ │ • Tools      │
   │ • Resources  │ │ • Resources  │
   │ • Prompts    │ │ • Prompts    │
   └──────────────┘ └──────────────┘
		  │                │
		  ▼                ▼
   ┌──────────────┐ ┌──────────────┐
   │ External     │ │ External     │
   │ Systems:     │ │ Systems:     │
   │ APIs, DBs,   │ │ Files, PDFs, │
   │ Web services │ │ Knowledge    │
   └──────────────┘ └──────────────┘
```

### Critical Rule: 1:1 Client-Server Connection

> **One MCP client connects to exactly one MCP server.** If your host needs to talk to 3 servers, it needs 3 clients.

This is a deliberate design choice for simplicity and isolation. Each client handles its own:
- Connection lifecycle
- Error handling
- Tool namespace

### What MCP Standardizes

Eden lists the protocol methods that servers must implement:

| Method | Purpose |
|--------|---------|
| `list_tools` | Returns available tools with descriptions and schemas |
| `call_tool` | Executes a specific tool with given arguments |
| `list_resources` | Returns available data resources |
| `list_resource_templates` | Returns URI templates for dynamic resources |
| `list_prompts` | Returns available prompt templates |
| `get_prompt` | Returns a specific prompt with given parameters |
| `progress_notification` | Reports execution progress for long-running tools |

### The USB-C Analogy

Eden uses the USB-C analogy (consistent with official MCP docs):
- MCP server = external device (keyboard, monitor, phone)
- MCP host = computer with USB-C port
- MCP protocol = the USB-C standard

Just like you can plug any USB-C device into any USB-C port, you can connect any MCP server to any MCP host.

> ⚠️ **Transcript correction:** Eden says "NCP" multiple times in the transcript — this is a speech-to-text error. It's always **MCP** (Model Context Protocol).

---

## 4. The Gist of the Protocol — Full Interaction (Ch. 128)

This is the **most important chapter** — it shows the complete lifecycle from app startup to user response.

### Phase 1: Initialization (Before User Interaction)

```
┌─────────┐                           ┌────────────┐
│  HOST   │                           │ MCP SERVER │
│  (App)  │                           │ (Weather)  │
│         │                           │            │
│ Client ─┼── 1. Initialize ────────► │            │
│         │◄── 2. Acknowledge ─────── │            │
│         │                           │            │
│ Client ─┼── 3. List tools ────────► │            │
│         │◄── 4. Available tools ─── │            │
│         │     [forecast, alerts]     │            │
└─────────┘                           └────────────┘
```

**This happens when you launch the app** — before any user types anything. The client:
1. Establishes connection with the server
2. Server acknowledges and capabilities are exchanged
3. Client asks "what tools do you have?"
4. Server responds with tool definitions (name, description, input schema)

### Phase 2: User Query → Tool Execution → Response

```
┌──────┐   ┌─────────┐   ┌─────┐   ┌────────────┐
│ USER │   │  HOST   │   │ LLM │   │ MCP SERVER │
└──┬───┘   └────┬────┘   └──┬──┘   └─────┬──────┘
   │            │            │            │
   │ 1. Query  │            │            │
   ├───────────►│            │            │
   │            │ 2. Query   │            │
   │            │  + tools   │            │
   │            ├───────────►│            │
   │            │            │            │
   │            │ 3. Tool    │            │
   │            │   call     │            │
   │            │◄───────────┤            │
   │            │                         │
   │            │ 4. Execute tool         │
   │            ├────────────────────────►│
   │            │                         │
   │            │ 5. Tool result          │
   │            │◄────────────────────────┤
   │            │                         │
   │            │ 6. Query   │            │
   │            │  + result  │            │
   │            ├───────────►│            │
   │            │            │            │
   │            │ 7. Final   │            │
   │            │   answer   │            │
   │            │◄───────────┤            │
   │            │            │            │
   │ 8. Answer │            │            │
   │◄───────────┤            │            │
```

**Step by step:**

| Step | What Happens | Who Does It |
|------|--------------|-------------|
| 1 | User sends query (e.g., "What's the weather in California?") | User → Host |
| 2 | Host sends query **+ available tools** to LLM (the "fancy system prompt") | Host → LLM |
| 3 | LLM responds with a tool call: `forecast(location="California")` | LLM → Host |
| 4 | Host sends tool call to MCP server via protocol | Client → Server |
| 5 | **Server executes the tool** and returns result | Server → Client |
| 6 | Host sends original query + tool result to LLM | Host → LLM |
| 7 | LLM generates natural language final answer | LLM → Host |
| 8 | Host returns answer to user | Host → User |

### The Critical Insight: Where Tools Execute

> **In LangChain (vanilla):** Tools execute INSIDE your application process.
> **In MCP:** Tools execute IN THE SERVER — a separate process.

This decoupling is the architectural win:

| Concern | LangChain (in-process) | MCP (server-side) |
|---------|----------------------|-------------------|
| **Scaling** | Scale the entire agent to scale one tool | Scale the server independently |
| **Deployment** | Redeploy agent to update a tool | Redeploy server only — agent is untouched |
| **Monitoring** | All tools share agent's monitoring | Each server has its own monitoring/logging |
| **Security** | Tool code runs with agent's permissions | Server can have restricted permissions |
| **Debugging** | Tools interleaved with agent logic | Clear separation — tool logs live in server |
| **Dynamic updates** | Must redeploy to add/change tools | Re-initialize client → new tools appear |

### Dynamic Tool Calling

Eden highlights a powerful advantage: because initialization can happen **periodically** (not just at startup), your agent can discover new tools at runtime:

```
Time 0:00 — Agent initializes → Server has tools: [forecast, alerts]
Time 1:00 — New tool deployed to server: [forecast, alerts, radar]  
Time 2:00 — Agent re-initializes → Now has tools: [forecast, alerts, radar]
			No agent redeployment needed!
```

---

## 5. MCP Servers — The Three Interfaces (Ch. 129)

### What MCP Servers Expose

MCP servers are wrappers that federate access to external systems through **three interfaces**:

| Interface | Controlled By | What It Is | Example |
|-----------|---------------|-----------|---------|
| **Tools** | The MODEL decides when to use | Executable functions | `get_weather(city)`, `send_email(to, body)`, `query_db(sql)` |
| **Resources** | The APPLICATION decides when to read | Data (static or dynamic) | PDF documents, database records, config files, live API data |
| **Prompts** | The USER decides when to invoke | Interaction templates | "Summarize this document", "Review this PR for security" |

### Tools — Model-Controlled Functions

- The AI decides when and how to call them
- Can implement **any** functionality (read data, write data, call APIs)
- You control what to expose — don't expose `delete_all_emails` if you don't want the AI to use it

### Resources — Application-Controlled Data

Two types:
- **Static** — files, PDFs, images, JSON documents
- **Dynamic** — live database queries, real-time API responses

The key difference from tools: resources don't DO anything. They provide **context**. The application decides when to read them, not the LLM.

### Prompts — User-Controlled Templates

Predefined templates that standardize complex interactions. The user chooses to invoke them.

Example: A "code review" prompt template that structures how the AI should review code, what to look for, what format to use.

### How to Get MCP Servers

| Approach | Description | When to Use |
|----------|-------------|-------------|
| **Build manually** | Write Python/Node.js code (~100-200 lines) | Custom internal tools |
| **AI-generated** | Use Cursor/generators to scaffold | Quick prototypes |
| **Community servers** | Clone from GitHub, modify if needed | Standard integrations (thousands exist) |
| **Official vendor servers** | Maintained by the company (Stripe, Cloudflare, etc.) | Production third-party integrations |

> ⚠️ **Eden's important warning:** Don't reinvent the wheel! Before building an MCP server for a third-party service, check if they already maintain one. Stripe, Cloudflare, GitHub, and many others already have official MCP servers. Use those — they're maintained, tested, and free.

### Server Execution Modes

| Mode | Transport | Use Case |
|------|-----------|----------|
| **Local** | stdio (stdin/stdout) | Development, private tools, single machine |
| **Remote** | HTTP (streamable-http) | Production, shared access, team tools |
| **Docker** | Container | Isolation, reproducibility, deployment |

### Sampling — Server-Initiated LLM Calls

Eden mentions a fourth capability: **sampling**. This allows the MCP server to request the host to generate a completion.

```
Normal flow:  User → Host → LLM → "call tool" → Server executes → result
Sampling:     Server → Host → LLM → "generate text for me" → result back to server
```

This enables recursive patterns (server needs AI reasoning as part of its execution) but introduces:
- **Security risk** — server can prompt-inject the host's LLM
- **Privacy risk** — server can extract information via crafted prompts
- **Cost risk** — server can trigger unbounded LLM calls

### Composability

> "Any application can be BOTH an MCP client and server."

This enables multi-layered architectures where specialized agents chain together:

```
Agent A (client) → MCP Server B (which is also a client) → MCP Server C
```

### The Future (Eden's Predictions)

| Feature | Status (2025) | What It Enables |
|---------|---------------|-----------------|
| **Registry & discovery** | In development | Central API to find MCP servers (like npm for tools) |
| **Verified servers** | In development | Prevents supply-chain attacks (malicious fake servers) |
| **`.well-known` endpoints** | Proposed | Websites expose MCP capabilities like `robots.txt` for search engines |
| **OAuth 2.0 auth** | Partially implemented | Secure access with standard auth flows |
| **Session tokens** | Proposed | Persistent authenticated connections |

> ⚠️ **Transcript correction:** Eden mentions "server sent events" (SSE) as a transport option. As of the MCP spec (2025-03-26), SSE transport is **deprecated** in favour of **streamable HTTP**. New implementations should use `transport="http"` (streamable-http), not SSE. The `langchain-mcp-adapters` library uses `"http"` as the transport key.

---

## 6. Key Difference: LangChain vs MCP Tool Execution

Eden repeatedly emphasises this distinction. Here's the comparison:

```
┌─────────────────────────────────────────────────────────────┐
│ LANGCHAIN REACT AGENT (vanilla)                              │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ YOUR APPLICATION PROCESS                             │    │
│  │                                                     │    │
│  │  Agent (orchestration)                              │    │
│  │    ↓                                                │    │
│  │  Tool execution (RUNS HERE — same process)          │    │
│  │    ↓                                                │    │
│  │  Result → back to agent                             │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  Everything is coupled. Scale agent = scale tools.          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ MCP INTEGRATION                                              │
│                                                             │
│  ┌───────────────────────────┐  ┌───────────────────────┐  │
│  │ YOUR APPLICATION PROCESS  │  │ MCP SERVER (separate)  │  │
│  │                           │  │                        │  │
│  │  Agent (orchestration)    │  │  Tool execution        │  │
│  │    ↓                      │  │  (RUNS HERE)           │  │
│  │  MCP Client ──protocol──► │  │                        │  │
│  │    ↓                      │  │                        │  │
│  │  Result ◄── protocol ──── │  │                        │  │
│  └───────────────────────────┘  └───────────────────────┘  │
│                                                             │
│  Decoupled. Scale/deploy/monitor independently.             │
└─────────────────────────────────────────────────────────────┘
```

**Both can coexist.** Eden notes you can technically make "dummy tools" in LangChain that just forward requests to an external service — but MCP **standardizes** this pattern with:
- One protocol everyone agrees on
- Automatic tool discovery
- No custom serialization/deserialization code

### When to Use Which

| Scenario | Use LangChain tools | Use MCP servers |
|----------|-------------------|-----------------|
| Quick prototype, single developer | ✅ Simpler, fewer moving parts | Overkill |
| Tools need to be shared across apps | Requires copying code | ✅ Write once, use everywhere |
| Team with multiple agents | Duplicated tool code | ✅ Central tool servers |
| Need independent scaling | Must scale entire agent | ✅ Scale servers independently |
| Need dynamic tool updates | Requires redeployment | ✅ Re-initialize picks up new tools |
| Simple internal tools (< 5) | ✅ Less infrastructure | More setup than needed |

---

## MCP Server Discovery Resources

Eden mentions the site [mcp.so](https://mcp.so/) for discovering community MCP servers. Additional directories:

| Resource | URL | Description |
|----------|-----|-------------|
| mcp.so | https://mcp.so/ | Community directory of MCP servers |
| Anthropic's official list | https://github.com/modelcontextprotocol/servers | Official + community servers |
| MCP Hub | https://mcphub.io/ | Searchable registry |

---

## Interview Q&A Anchors

**Q: What is MCP and why does it exist?**
> **A:** MCP (Model Context Protocol) standardizes how AI applications expose tools, resources, and prompts to LLMs. It solves the N×M integration problem — without it, every app (Cursor, Claude, Windsurf) needs custom code for every tool. With MCP, tools are written once in a server and any MCP-compatible client can use them. Think USB-C: one standard connector, all devices work.

**Q: How does tool calling actually work in LLMs?**
> **A:** LLMs are pure token generators — they can't execute code. Tool calling is application-layer behaviour: the system prompt includes available tools as structured descriptions, the LLM outputs a structured tool call (function name + arguments) instead of a natural language answer, and the application parses this output, executes the real function, then feeds the result back to the LLM for final answer generation.

**Q: What's the difference between how LangChain and MCP handle tool execution?**
> **A:** In LangChain (vanilla), tools execute inside your agent process — same runtime, same deployment, same scaling. With MCP, tools execute in a separate server process. The agent handles orchestration (deciding WHAT to call), the MCP server handles execution (RUNNING the tool). This decoupling enables independent scaling, deployment, monitoring, and dynamic tool updates without agent redeployment.

**Q: Explain the 1:1 client-server relationship in MCP.**
> **A:** Each MCP client connects to exactly one MCP server. If your host application needs to use tools from 3 different servers (weather, email, database), it creates 3 separate MCP clients, one per server. This isolation ensures each connection has its own lifecycle management, error handling, and namespace — a failure in one server doesn't cascade to others.

**Q: What are the three primitives an MCP server exposes?**
> **A:** Tools (model-controlled executable functions the LLM decides to call), Resources (application-controlled data like files or DB records that provide context), and Prompts (user-controlled templates for standardizing complex interactions). Tools are the most common — the LLM sees their name/description and decides when to invoke them.

**Q: What happens during MCP initialization — before any user interaction?**
> **A:** When the application starts, each MCP client establishes a connection with its server. The server acknowledges and sends its capabilities. The client then asks for available tools (and resources/prompts), and the server returns their definitions including names, descriptions, and input schemas. Only after this handshake can the agent include these tools in LLM prompts. This is why MCP clients can discover new tools by re-initializing periodically.

**Q: What is "sampling" in MCP and why is it a security concern?**
> **A:** Sampling allows an MCP server to request the host application's LLM to generate a completion. This enables recursive agentic patterns where the tool itself needs AI reasoning. The security concern is that a malicious server could craft prompts to extract private information from the host's context, trigger unwanted actions via the LLM, or cause unbounded API costs by generating excessive completions.

**Q: Why does Eden say "don't reinvent the wheel" for MCP servers?**
> **A:** Major companies (Stripe, Cloudflare, GitHub) maintain official MCP servers for their products. They're motivated to do so because it increases product adoption. Before building your own integration, check if an official one exists — it'll be better maintained, tested, and secure. Only build custom servers for your own internal tools or when no official option exists.

---

## References

- [MCP Official Documentation](https://modelcontextprotocol.io/introduction)
- [MCP Specification (2025-03-26)](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports)
- [Uber Eats MCP Server (Eden's example)](https://github.com/ericzakariasson/uber-eats-mcp-server)
- [mcp.so — Community server directory](https://mcp.so/)
- [Official MCP Servers Repository](https://github.com/modelcontextprotocol/servers)
- [LangChain MCP Documentation](https://docs.langchain.com/oss/python/langchain/mcp)
