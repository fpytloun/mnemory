# MCP Tools

mnemory exposes 17 tools via the [Model Context Protocol](https://modelcontextprotocol.io/). These are available to any MCP-compatible client.

## Session Initialization

| Tool | Description |
|---|---|
| `initialize_memory` | **Start here.** Returns behavioral instructions + core memories. Use for clients that don't inject MCP server instructions (e.g., Open WebUI). |

## Memory Operations

| Tool | Description |
|---|---|
| `add_memory` | Store a memory with optional metadata, `infer` flag, `role`, `event_date`, and `labels` |
| `add_memories` | Batch-add multiple memories in a single call (supports `labels` per item) |
| `search_memories` | Semantic search with type/category/role/date/labels filters, importance reranking |
| `find_memories` | AI-powered search: generates multiple queries, searches, and reranks by relevance to your question. Temporal-aware — resolves "last week", "in 2023", etc. |
| `ask_memories` | Ask a question and get a human-readable answer synthesized from relevant memories. Uses `find_memories` internally, then generates a natural language answer. Most expensive (3 LLM calls). |
| `get_core_memories` | Load pinned + recent context at conversation start. Use for clients that inject MCP server instructions (e.g., Claude Code). |
| `get_recent_memories` | Get recent activity from the last N days with scope filter (user/agent/all) |
| `list_memories` | List all/filtered memories (supports `labels` and `memory_layer` filters) |
| `update_memory` | Create an immutable successor revision. Supports `expected_revision` and `idempotency_key`. |
| `delete_memory` | Retract by default. Set `privacy_erase=true` only to erase the lineage and artifacts. |
| `delete_all_memories` | Delete all memories in scope |
| `list_categories` | List categories with counts for discoverability |

## Artifact Operations

| Tool | Description |
|---|---|
| `save_artifact` | Attach detailed content to a memory |
| `get_artifact` | Retrieve artifact content (text paginated, binary ≤1 MB inline) |
| `get_artifact_url` | Generate a signed download URL for direct browser access |
| `list_artifacts` | List artifacts on a memory |
| `delete_artifact` | Remove an artifact |

## Tool Usage Patterns

### Storing Memories

- **`infer=true`** (default): The server runs a single LLM call that extracts facts, classifies metadata (type, categories, importance), and deduplicates against existing memories. Always use this for normal operation.
- **`infer=false`**: Skips the LLM call — content is embedded and stored directly. This bypasses fact extraction, deduplication, and contradiction resolution. Intended for server-side maintenance and bulk data import only, not for routine memory storage by agents.
- **`add_memories`** (batch): Processes multiple memories in a single call, avoiding per-item round-trip latency.
- **`labels`**: Client-provided key-value metadata that bypasses LLM extraction entirely. When using `infer=true`, labels are inherited by all extracted facts. See [Memory Model — Labels](memory-model.md#labels) for details.

### Searching

Three search tools are available:

- **`search_memories`**: Fast single-query vector search. Use for simple lookups and routine memory recall. Preferred for most cases.
- **`find_memories`**: AI-powered multi-query search. Takes a natural language question, generates multiple targeted searches following associations (e.g., "dogs" -> pets, partner, house, lifestyle), and uses AI to rerank results by relevance. Use for complex, multi-faceted questions where a single search query wouldn't capture all relevant context. Slower (2 extra LLM calls) but higher quality for complex queries.
- **`ask_memories`**: Ask a question and get a human-readable answer. Uses `find_memories` internally to locate relevant memories, then passes them to an LLM to generate a natural language prose answer. Most expensive operation (3 LLM calls: query generation + reranking + answer generation). Use when you need a synthesized answer rather than raw memory results. Set `include_memories=true` to also receive the supporting memories used to generate the answer.

### Artifacts

For detailed content too long for fast memory (research reports, analysis, code, data), save it as an artifact attached to a memory. The memory holds the searchable summary; the artifact holds the full details. Search results show which memories have artifacts — fetch them with `get_artifact` when you need the details.

**Binary size limit:** Binary artifacts larger than 1 MB cannot be returned inline via `get_artifact`. For large binary artifacts, use `get_artifact_url` to generate a short-lived signed download URL that can be opened in a browser or embedded in `<img>` tags.

### Role Parameter

The `role` parameter tells the server who the memory is about:

- **`role="user"`** (default for `add_memory`): Facts about the user — preferences, personal info, context, decisions.
- **`role="assistant"`**: Facts about the agent — identity, personality, capabilities, knowledge. Requires `agent_id` to be set.

The REST `POST /api/remember` endpoint supports a third option:

- **`role=null`** (default for `remember`): **Auto mode** — extracts facts from ALL participants (user and assistant turns). Each fact is attributed to the correct role automatically. Assistant facts are silently dropped if no `agent_id` is set in the session.

When searching or listing, `role` filters results by who the memory is about. Search automatically returns both agent-specific and shared user memories.
