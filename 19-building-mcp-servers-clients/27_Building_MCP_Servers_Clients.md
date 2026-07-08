# 27. Building MCP Servers & Clients — Eden Course Notes

> **Context:** Section 19, Chapters 134–141. Eden builds custom MCP servers (stdio + SSE) and connects them using LangChain's `MultiServerMCPClient`. This is Step 2 of the MCP learning path — writing your own servers and clients instead of using pre-built ones.

---

## The Core Idea

> **Remember this, forget the rest.** The `langchain-mcp-adapters` package is the bridge between MCP servers and LangGraph agents. It converts MCP tools → LangChain tools with one function call (`load_mcp_tools`), then you use them in a standard ReAct agent. The key insight: MCP and LangChain both have "tools" — this package translates between the two formats so you don't have to.

**What this section builds:**

```
┌──────────────────────────────────────────────────────────────┐
│ SECTION 19 GOAL                                              │
│                                                              │
│  Custom MCP SERVERS          LangChain MCP CLIENT            │
│  ┌────────────────┐         ┌─────────────────────┐         │
│  │ math_server.py │──stdio─►│                     │         │
│  │ (add/multiply) │         │ MultiServerMCPClient │         │
│  └────────────────┘         │         OR          │         │
│  ┌────────────────┐         │ load_mcp_tools()    │         │
│  │ weather_server │──SSE───►│                     │         │
│  │ (get_weather)  │         └─────────┬───────────┘         │
│  └────────────────┘                   │                      │
│                                       ▼                      │
│                              ┌─────────────────┐             │
│                              │ LangGraph Agent │             │
│                              │ (create_react_  │             │
│                              │  agent)         │             │
│                              └─────────────────┘             │
│                                                              │
│  Result: A LangGraph agent that uses MCP servers as tools.   │
└──────────────────────────────────────────────────────────────┘
```

---

## Table of Contents

| # | Section | What You'll Learn |
|---|---------|-------------------|
| 1 | [The Learning Path (Ch. 134)](#1-the-learning-path-ch-134) | Why we build servers before clients |
| 2 | [Project Setup (Ch. 135)](#2-project-setup-ch-135) | Dependencies and boilerplate |
| 3 | [Building MCP Servers (Ch. 136)](#3-building-mcp-servers-ch-136) | Math server (stdio) + Weather server (SSE) |
| 4 | [LangChain vs MCP Tools — The Conceptual Bridge (Ch. 139)](#4-langchain-vs-mcp-tools--the-conceptual-bridge-ch-139) | Why the adapter exists, similarities and differences |
| 5 | [Building the MCP Client (Ch. 137–138, 140–141)](#5-building-the-mcp-client-ch-137138-140141) | Connecting servers to a LangGraph agent |
| 6 | [The Full Working Example](#6-the-full-working-example) | Complete, corrected source code |
| 7 | [Interview Q&A Anchors](#7-interview-qa-anchors) | Quick-fire answers |

---

## Key Definitions

| Term | Quick Recall | Full Definition |
|------|-------------|----------------|
| **langchain-mcp-adapters** | Bridge MCP ↔ LangChain | A package that converts MCP tools into LangChain-compatible tools, so LangGraph agents can call any MCP server's tools natively. |
| **MultiServerMCPClient** | One client → many servers | A convenience class that manages connections to multiple MCP servers and returns all their tools as a unified list. Handles session lifecycle internally. |
| **load_mcp_tools()** | MCP session → LangChain tools | Takes an active MCP `ClientSession`, queries the server for its tool list, and wraps each one as a LangChain `Tool` object. |
| **FastMCP** | Easy server creation | A decorator-based API from the `mcp` SDK for defining tools, resources, and prompts. `@mcp.tool()` is all you need. |
| **StdioServerParameters** | How to launch a subprocess server | Specifies the command and args to start an MCP server as a child process. Client and server communicate via stdin/stdout. |
| **create_react_agent** | Pre-built ReAct agent | LangGraph's convenience function that creates a tool-calling agent with automatic loop handling. Just provide an LLM and tools. |

---

## 1. The Learning Path (Ch. 134)

Eden structures the progression:

```
Section 18 (previous): Pre-built server + pre-built client
  → Learned the protocol without writing code

Section 19 (this one): Build custom servers + build custom client
  → Step 2a: Implement MCP servers (math + weather)
  → Step 2b: Implement MCP client using langchain-mcp-adapters
  → Step 2c: Connect everything into a LangGraph agent
```

> ⚠️ **Transcript correction:** Eden repeatedly says "NCP" throughout the transcript. This is a speech-to-text error — he means **MCP** (Model Context Protocol) in every instance.

---

## 2. Project Setup (Ch. 135)

### Dependencies

```bash
uv add langchain-mcp-adapters langgraph langchain-openai python-dotenv
```

| Package | Purpose |
|---------|---------|
| `langchain-mcp-adapters` | The bridge package (also installs `mcp` SDK automatically) |
| `langgraph` | Agent framework (provides `create_react_agent`) |
| `langchain-openai` | OpenAI LLM integration |
| `python-dotenv` | Load `.env` file |

### Environment Variables

```env
OPENAI_API_KEY=sk-...

# Optional: LangSmith tracing (set LANGCHAIN_TRACING_V2=false to disable)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_...
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_PROJECT=mcp-test
```

---

## 3. Building MCP Servers (Ch. 136)

Eden builds two minimal MCP servers to demonstrate both transport types.

### Server 1: Math Server (stdio transport)

```python
# servers/math_server.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Math")

@mcp.tool()
def add(a: int, b: int) -> int:
	"""Add two numbers"""
	return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
	"""Multiply two numbers"""
	return a * b

if __name__ == "__main__":
	mcp.run(transport="stdio")
```

**How to run:** You don't run it manually — the **client** launches it as a subprocess via `StdioServerParameters`. The client sends stdin, reads stdout.

### Server 2: Weather Server (SSE transport)

```python
# servers/weather_server.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Weather")

@mcp.tool()
async def get_weather(location: str) -> str:
	"""Get weather for location."""
	return "Hot as hell"

if __name__ == "__main__":
	mcp.run(transport="sse")
```

**How to run:** You start it manually — it runs as an HTTP server on port 8000:

```bash
uv run servers/weather_server.py
# Output: Running on http://localhost:8000
```

### The Key Difference Between Transports

| | stdio | SSE/HTTP |
|---|-------|----------|
| **Who starts the server?** | The client launches it as a subprocess | You start it manually (or deploy it) |
| **Communication** | Via stdin/stdout pipes | Via HTTP requests |
| **Deployment** | Local only (same machine) | Can be anywhere (local, cloud, enterprise) |
| **Use case** | Dev tools, local integrations | Shared services, team-wide access |

> ⚠️ **Transcript correction:** Eden says "SSH" multiple times when he means **SSE** (Server-Sent Events). This is a speech-to-text error. SSE is a web technology for streaming HTTP; SSH is an unrelated secure shell protocol.

> ⚠️ **Important note on SSE:** As of the MCP spec (2025-03-26), SSE transport is deprecated in favour of **streamable HTTP**. Eden's code uses `transport="sse"` which still works but new projects should prefer `transport="http"`. The `langchain-mcp-adapters` package supports both.

### Eden's File Naming Bug

Eden initially named his file `math.py`, which **collides with Python's built-in `math` module**. This causes import errors like `AttributeError: module 'math' has no attribute 'tool'`. Always name it `math_server.py`.

---

## 4. LangChain vs MCP Tools — The Conceptual Bridge (Ch. 139)

This is the key conceptual chapter — why does `langchain-mcp-adapters` exist at all?

### The Similarity

Both LangChain and MCP have the concept of **tools** — functions with:
- A **name** (identifier)
- A **description** (tells the LLM when to use it)
- **Arguments** (typed inputs)
- A **return value** (output)

Both systems inject tool descriptions into the LLM's prompt so the model can decide which tool to call.

### The Difference

| Aspect | LangChain Tools | MCP Tools |
|--------|----------------|-----------|
| **Defined where?** | In your Python code (same process) | On a separate MCP server (potentially remote) |
| **Exposed to?** | A specific LLM via `bind_tools()` | An AI application (Cursor, Claude, custom client) via MCP protocol |
| **Additional capabilities** | Tools only | Tools + Resources + Prompts |
| **Communication** | Direct function call (in-process) | Over protocol (stdio/HTTP — cross-process or cross-network) |

### What langchain-mcp-adapters Does

```
┌──────────────────────────────────────────────────────────────┐
│ WITHOUT the adapter:                                         │
│                                                              │
│  MCP Server ──(MCP protocol)──► MCP Client ──► ???           │
│  (has tools)                    (gets tools)   (Can't use    │
│                                                 with         │
│                                                 LangGraph)   │
├──────────────────────────────────────────────────────────────┤
│ WITH the adapter:                                            │
│                                                              │
│  MCP Server ──(MCP protocol)──► MCP Client                   │
│  (has tools)                    (gets tools)                 │
│                                       │                      │
│                                       ▼                      │
│                              load_mcp_tools()                │
│                                       │                      │
│                                       ▼                      │
│                              LangChain Tool objects           │
│                                       │                      │
│                                       ▼                      │
│                              LangGraph Agent                  │
│                              (uses them natively)            │
└──────────────────────────────────────────────────────────────┘
```

### The Key Value Proposition

> You can leverage **any MCP server that someone else wrote** (hundreds exist in the MCP ecosystem) in your LangGraph agent without rewriting the tools in LangChain format. The adapter does the translation automatically.

---

## 5. Building the MCP Client (Ch. 137–138, 140–141)

### The SingleServer Pattern (stdio only)

When you connect to just one server:

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

server_params = StdioServerParameters(
	command="python",
	args=["/full/absolute/path/to/servers/math_server.py"],
)

async with stdio_client(server_params) as (read, write):
	async with ClientSession(read_stream=read, write_stream=write) as session:
		await session.initialize()
		tools = await load_mcp_tools(session)

		agent = create_react_agent(llm, tools)
		result = await agent.ainvoke(
			{"messages": [HumanMessage(content="What is 54 + 2 * 3?")]}
		)
		print(result["messages"][-1].content)
```

### The MultiServer Pattern (stdio + SSE)

When you connect to multiple servers at once:

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

client = MultiServerMCPClient(
	{
		"math": {
			"command": "python",
			"args": ["/full/absolute/path/to/servers/math_server.py"],
			"transport": "stdio",
		},
		"weather": {
			"url": "http://localhost:8000/sse",
			"transport": "sse",
		}
	}
)
tools = await client.get_tools()

agent = create_react_agent(llm, tools)
math_result = await agent.ainvoke({"messages": "what's (3 + 5) x 12?"})
weather_result = await agent.ainvoke({"messages": "what is the weather in NYC?"})
```

### Critical Lesson: Absolute Paths

> ⚠️ Just like with Claude Desktop (Section 18), you **must use full absolute paths** for the server script in `StdioServerParameters`. Relative paths will fail because the subprocess doesn't inherit your working directory.

---

## 6. The Full Working Example

The complete, corrected source code is in [`src/`](src/). See:

- [`src/servers/math_server.py`](src/servers/math_server.py) — Math MCP server (stdio)
- [`src/servers/weather_server.py`](src/servers/weather_server.py) — Weather MCP server (SSE)
- [`src/mcp_client_single.py`](src/mcp_client_single.py) — Single-server client (stdio only)
- [`src/mcp_client_multi.py`](src/mcp_client_multi.py) — Multi-server client (stdio + SSE)

### Project Structure

```
19-building-mcp-servers-clients/
├── 26_LangChain_MCP_Adapters_Reference.md   # Official docs reference
├── 27_Building_MCP_Servers_Clients.md        # This file (Eden notes)
└── src/
	├── servers/
	│   ├── __init__.py
	│   ├── math_server.py                    # stdio transport
	│   └── weather_server.py                 # SSE transport
	├── mcp_client_single.py                  # Single server (stdio) demo
	└── mcp_client_multi.py                   # Multi server (stdio + SSE) demo
```

### Running the Examples

```bash
# Terminal 1: Start the SSE weather server
uv run src/servers/weather_server.py

# Terminal 2: Run the multi-server client
uv run src/mcp_client_multi.py
```

---

## 7. Interview Q&A Anchors

**Q: What does `langchain-mcp-adapters` do and why is it needed?**
> **A:** It converts MCP tools into LangChain-compatible tools so LangGraph agents can use them natively. Without it, MCP tools speak the MCP protocol and LangChain tools speak the LangChain interface — they're incompatible. The adapter translates tool definitions (name, description, args schema) and handles tool invocation over the MCP protocol transparently.

**Q: What's the difference between `load_mcp_tools()` and `MultiServerMCPClient`?**
> **A:** `load_mcp_tools()` works with a single active `ClientSession` — you manage the connection lifecycle yourself. `MultiServerMCPClient` manages connections to multiple servers simultaneously, handles session creation/cleanup, and returns a unified tool list from all servers. Use `MultiServerMCPClient` when connecting to 2+ servers; use `load_mcp_tools()` for fine-grained session control.

**Q: Why does the math server use stdio while the weather server uses SSE/HTTP?**
> **A:** To demonstrate both transport types. Stdio is simpler — the client launches the server as a subprocess and communicates via pipes (local only). SSE/HTTP runs the server independently on a port — it can be deployed remotely and shared across multiple clients. In production, HTTP transport is preferred for shared services; stdio is for local dev tools.

**Q: How do LangChain tools and MCP tools relate conceptually?**
> **A:** Both are function interfaces with a name, description, typed arguments, and return value. Both inject tool descriptions into the LLM's prompt for tool-calling decisions. The difference is scope: LangChain tools are in-process Python functions; MCP tools live on separate servers and communicate over a protocol. `langchain-mcp-adapters` bridges this gap by wrapping MCP tools in the LangChain interface.

**Q: What happens if you name your MCP server file `math.py`?**
> **A:** It collides with Python's built-in `math` module. When Python imports dependencies, it may resolve `import math` to your file instead of the standard library, causing `AttributeError`. Always use descriptive names like `math_server.py` to avoid shadowing built-in modules.

**Q: What's the session lifecycle in MultiServerMCPClient?**
> **A:** By default, it creates a new MCP session for each tool invocation (stateless pattern). For stateful servers that need to maintain context across calls, you can explicitly open a session with `async with client.session("server_name") as session:` and load tools within that context.

---

## References

- [langchain-mcp-adapters GitHub](https://github.com/langchain-ai/langchain-mcp-adapters)
- [MCP Python SDK (FastMCP)](https://github.com/modelcontextprotocol/python-sdk)
- [LangGraph create_react_agent](https://langchain-ai.github.io/langgraph/reference/prebuilt/#create_react_agent)
- [MCP Transport Specification](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports)
