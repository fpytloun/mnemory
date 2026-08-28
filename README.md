<p align="center">
  <img src="files/banner.png" alt="mnemory banner" />
</p>

# mnemory

Give your AI agents persistent memory. mnemory is a self-hosted [MCP](https://modelcontextprotocol.io/) server that adds personalization and long-term memory to any AI assistant — Claude Code, ChatGPT, Open WebUI, Cursor, or any MCP-compatible client.

**Plug and play.** Connect mnemory and your agent immediately starts remembering user preferences, facts, decisions, and context across conversations. No system prompt changes needed.

**Self-hosted and secure.** Your data stays on your infrastructure. No cloud dependencies, no third-party access to your memories.

**Intelligent.** Uses a unified LLM pipeline for fact extraction, deduplication, and contradiction resolution in a single call. Memories are semantically searchable, automatically categorized, and expire naturally when no longer relevant.

## Features

- **Zero config** — `uvx mnemory`, connect your MCP client, done. Works out of the box with any OpenAI-compatible API.
- **Intelligent extraction** — A single LLM call extracts facts, classifies metadata, and deduplicates against existing memories.
- **Contradiction resolution** — "I drive a Skoda" + later "I bought a Tesla" = automatic update, not a duplicate.
- **Two-tier memory** — Fast searchable summaries in a vector store + detailed artifact storage (reports, code, research) retrieved on demand.
- **AI-powered search** — Multi-query semantic search with temporal awareness. Ask "What did I decide last week about the database?" and it finds the right memories.
- **Memory health checks** — Built-in three-phase consistency checker (fsck) detects duplicates, contradictions, quality issues, and prompt injection. Run manually or on a schedule with auto-fix.
- **10+ client support** — Claude Code, ChatGPT, Open WebUI, OpenClaw, Cursor, Windsurf, Cline, OpenCode, and more. Native plugins available for automatic recall/remember.
- **Built-in management UI** — Dashboard, semantic search, memory browser with full CRUD, relationship graph visualization, and health check interface. No extra tools needed.
- **Production ready** — Qdrant for vectors, S3/MinIO for artifacts, API key or Cognis JWT authentication, per-user isolation, Kubernetes-friendly stateless HTTP.
- **Secure by default** — API key or Cognis JWT authentication with session-level identity binding, per-user memory isolation, anti-injection safeguards in extraction prompts.
- **REST API + MCP** — Dual interface with the same backend. 16 MCP tools + full REST API with OpenAPI spec. Build plugins, integrations, or use directly.
- **Prometheus monitoring** — Built-in `/metrics` endpoint with operation counters and memory gauges. Pre-built Grafana dashboard included.

## Quick Start

mnemory needs an OpenAI-compatible API key for LLM and embeddings. It picks up `OPENAI_API_KEY` from your environment automatically.

```bash
uvx mnemory
```

If `uvx` reports `No module named 'mcp.server.fastmcp'`, update uv. Older uv
versions on Alpine can ignore mnemory's MCP v1 dependency constraint. Until
you update uv, run:

```bash
uvx --with "mcp<2" mnemory
```

That's it. mnemory starts on `http://localhost:8050/mcp`, stores data in `~/.mnemory/`.

Now connect your client — for **Claude Code**, add to your MCP config:

```json
{
  "mcpServers": {
    "mnemory": {
      "type": "streamable-http",
      "url": "http://localhost:8050/mcp",
      "headers": {
        "X-Agent-Id": "claude-code"
      }
    }
  }
}
```

Start a new conversation. Memory works automatically.

Also available via [Docker, pip, or production setup with Qdrant + S3](docs/deployment.md). See the [full quick start guide](docs/quickstart.md) for more clients and options.

## Screenshots

<p align="center">
  <img src="files/screenshots/ui-dashboard.jpg" alt="Dashboard — memory totals, breakdowns, and operation counts" width="800" />
  <br><em>Dashboard with memory breakdowns by type, category, and role</em>
</p>

<p align="center">
  <img src="files/screenshots/ui-search.jpg" alt="Search — semantic and AI-powered search with filters" width="800" />
  <br><em>Semantic search and AI-powered find with filters</em>
</p>

<p align="center">
  <img src="files/screenshots/ui-graph.jpg" alt="Graph — D3.js force-directed memory relationship visualization" width="800" />
  <br><em>Memory relationship graph visualization</em>
</p>

See [all screenshots and UI features](docs/management-ui.md) including memory browser, health checks, and artifact management.

## Supported Clients

mnemory works with any MCP-compatible client. Some clients also have dedicated plugins for automatic recall/remember.

| Client | MCP | Plugin | Setup Guide |
|---|---|---|---|
| Claude Code | Yes | Yes ([hooks](integrations/claude-code/)) | [Guide](docs/clients/claude-code.md) |
| ChatGPT | Yes (MCP connector) | -- | [Guide](docs/clients/chatgpt.md) |
| Claude Desktop | Yes | -- | [Guide](docs/clients/claude-desktop.md) |
| Hermes Agent | Yes | Yes ([plugin](integrations/hermes/)) | [Guide](docs/clients/hermes.md) |
| Open WebUI | Yes | Yes ([filter](integrations/openwebui/)) | [Guide](docs/clients/open-webui.md) |
| OpenCode | Yes | Yes ([plugin](integrations/opencode/)) | [Guide](docs/clients/opencode.md) |
| OpenClaw | Yes | Yes ([plugin](integrations/openclaw/)) | [Guide](docs/clients/openclaw.md) |
| Cursor | Yes | -- | [Guide](docs/clients/cursor.md) |
| Windsurf | Yes | -- | [Guide](docs/clients/windsurf.md) |
| Cline | Yes | -- | [Guide](docs/clients/cline.md) |
| Continue.dev | Yes | -- | [Guide](docs/clients/continue.md) |
| Codex CLI | Yes | -- | [Guide](docs/clients/codex.md) |

**MCP** = works via Model Context Protocol (LLM-driven tool calls). **Plugin** = dedicated integration with automatic recall/remember (no LLM tool-calling needed).

## How It Works

**Storing:** You share information naturally. mnemory extracts individual facts, classifies them (type, category, importance), checks for duplicates and contradictions against existing memories, and stores them as searchable vectors — all in a single LLM call.

**Searching:** Ask a question and mnemory generates multiple search queries covering different angles and associations, runs them in parallel, and reranks results by relevance. Temporal-aware — "what did I decide last week?" just works.

**Recalling:** At conversation start, your agent loads pinned memories (core facts, preferences, identity) plus recent context. During conversation, relevant memories are found automatically based on what you're discussing.

**Maintaining:** Memories have configurable TTL — context expires in 7 days, episodic memories in 90. Frequently accessed memories stay alive (reinforcement). The built-in health checker detects and fixes duplicates, contradictions, and quality issues.

Learn more in the [architecture docs](docs/architecture.md).

## Benchmark

Evaluated on the [LoCoMo](https://github.com/snap-research/locomo) benchmark — 10 multi-session dialogues with 1540 QA questions across 4 categories:

| System | single_hop | multi_hop | temporal | open_domain | Overall |
|---|---|---|---|---|---|
| **mnemory** | **63.1** | **53.1** | **74.8** | **78.2** | **73.2** |
| mnemory (gpt-oss-120b) | 66.3 | 59.4 | 68.5 | 73.8 | 70.5 |
| Memobase | 70.9 | 52.1 | 85.0 | 77.2 | 75.8 |
| Mem0-Graph | 65.7 | 47.2 | 58.1 | 75.7 | 68.4 |
| Mem0 | 67.1 | 51.2 | 55.5 | 72.9 | 66.9 |
| Zep | 61.7 | 41.4 | 49.3 | 76.6 | 66.0 |
| LangMem | 62.2 | 47.9 | 23.4 | 71.1 | 58.1 |

Configuration: `gpt-5-mini` for extraction, `text-embedding-3-small` for vectors. `gpt-oss-120b` via Groq is a budget alternative at ~5x lower cost with comparable quality. See [configuration docs](docs/configuration.md#model-selection) for model options and [`benchmarks/`](benchmarks/) for reproduction.

## Documentation

| Document | Description |
|---|---|
| [Quick Start](docs/quickstart.md) | Get running in 5 minutes with any client |
| [Configuration](docs/configuration.md) | All environment variables — LLM, storage, server, memory behavior |
| [Memory Model](docs/memory-model.md) | Types, categories, importance, TTL, roles, scoping, sub-agents |
| [MCP Tools](docs/mcp-tools.md) | 16 MCP tools — memory CRUD, search, artifacts |
| [REST API](docs/rest-api.md) | Full REST API, fsck pipeline, recall/remember endpoints |
| [Architecture](docs/architecture.md) | System diagram, detailed flows for storing/searching/recalling |
| [Management UI](docs/management-ui.md) | Screenshots, features, access, UI development |
| [Monitoring](docs/monitoring.md) | Prometheus metrics, Grafana dashboard |
| [Deployment](docs/deployment.md) | Production setup, Docker, authentication, Kubernetes |
| [Development](docs/development.md) | Building, testing, linting, contributing |
| [Client Guides](docs/clients/) | Per-client setup instructions (10 clients) |
| [System Prompts](docs/system-prompts/) | Templates for personality agents and custom setups |

## License

Apache 2.0
