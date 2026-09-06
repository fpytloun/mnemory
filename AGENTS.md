# AGENTS.md — Coding Agent Instructions for mnemory

## Project Overview

**mnemory** is a self-hosted, two-tier memory system for AI agents, exposed as an MCP (Model Context Protocol) server over Streamable HTTP. It provides persistent memory with intelligent fact extraction, deduplication, and semantic search.

- **Language**: Python 3.11+
- **Framework**: MCP SDK (`mcp.server.fastmcp.FastMCP`) with Starlette for HTTP
- **Core dependencies**: [qdrant-client](https://github.com/qdrant/qdrant-client) for vector storage, [openai](https://github.com/openai/openai-python) for LLM and embeddings
- **License**: Apache 2.0
- **Repository**: https://github.com/fpytloun/mnemory

## Architecture

```
mnemory/
├── server.py              # MCP server entry point, 16 tool definitions, health endpoint, auth middleware
├── config.py              # Configuration from environment variables (dataclasses)
├── categories.py          # Predefined category registry, validation, matching logic
├── memory.py              # Business logic layer (orchestrates vector + artifact stores)
├── llm.py                 # OpenAI-compatible LLM client with structured output support
├── embeddings.py          # OpenAI-compatible embedding client with batch support
├── prompts.py             # Unified extraction+classification+dedup prompt templates, query generation and rerank prompts for find_memories
├── ttl.py                 # TTL (Time-To-Live) utility functions for memory expiration
├── instructions.py        # Configurable MCP server instructions (passive/proactive/personality/managed modes)
├── session.py             # Server-side memory session tracking (MemorySession + SessionStore)
├── metrics.py             # Prometheus metrics collection, operation counters, Qdrant gauge aggregation
├── api/
│   ├── __init__.py        # FastAPI app factory, session store instance
│   ├── deps.py            # Auth + identity FastAPI dependency (reads contextvars)
│   ├── schemas.py         # Pydantic request/response models for all endpoints
│   ├── memories.py        # Memory CRUD + artifact + category REST endpoints
│   ├── recall.py          # POST /api/recall — combined initialize + search
│   ├── remember.py        # POST /api/remember — fire-and-forget memory storage
│   ├── sessions.py        # GET /api/sessions — session summary endpoints for UI
│   └── ui.py              # GET /api/whoami + GET /api/stats — management UI endpoints
├── ui/
│   ├── __init__.py        # Package marker
│   ├── tailwind.config.js # Tailwind CSS config with brand colors
│   ├── src/input.css      # Tailwind directives + component classes
│   └── static/            # Pre-built UI assets (HTML, JS modules: api/app/metrics/search/memories/sessions/graph, CSS, vendored libs)
└── storage/
    ├── vector.py          # Direct Qdrant vector store (insert, search, update, delete)
    ├── artifact.py        # Artifact store abstraction (S3 and filesystem backends)
    └── session.py         # Session persistence backends (SQLite, Redis, in-memory) + serialization
```

### Layer responsibilities

| Layer | File | Responsibility |
|---|---|---|
| **Transport** | `server.py` | MCP tool definitions, HTTP routing, auth, serialization |
| **REST API** | `api/` | FastAPI sub-app with OpenAPI spec, CRUD + intelligence endpoints |
| **Business Logic** | `memory.py` | Validation, reranking, core memory assembly, artifact lifecycle |
| **Categories** | `categories.py` | Category validation, prefix matching, counting |
| **TTL** | `ttl.py` | Expiration calculation, decay detection, reinforcement metadata |
| **Vector Storage** | `storage/vector.py` | Direct Qdrant client for all vector operations |
| **Artifact Storage** | `storage/artifact.py` | S3/MinIO and filesystem backends for binary artifacts |
| **Management UI** | `ui/`, `api/ui.py` | Built-in web UI — dashboard, search, browse/CRUD, graph (Alpine.js + Tailwind + Chart.js + D3.js) |
| **Instructions** | `instructions.py` | Configurable MCP server instructions (passive/proactive/personality modes) |
| **Sessions** | `session.py` | Server-side memory session tracking with write-through cache |
| **Session Storage** | `storage/session.py` | Session persistence backends (SQLite, Redis, in-memory) |
| **Metrics** | `metrics.py` | Prometheus metrics collection, operation counters, Qdrant gauge aggregation |
| **Configuration** | `config.py` | Environment variable parsing into dataclass configs |

### Key design decisions

1. **Direct Qdrant + OpenAI**: Uses Qdrant directly for all vector operations and OpenAI-compatible APIs for LLM and embeddings. A single unified LLM call handles fact extraction, per-fact classification, and deduplication simultaneously.

2. **Two-tier memory**: Fast memory (vector store, max 1000 chars, searchable) + slow memory (artifact store, up to 10MB, retrieved on demand). Artifacts are always attached to a parent fast memory.

3. **Configurable backends**: Vector store uses Qdrant (local embedded mode for dev, remote for production). Artifact store supports S3/MinIO (production) or local filesystem (dev). Session store supports SQLite (default), Redis (clustered), or in-memory (tests). Selected via `QDRANT_HOST`, `ARTIFACT_BACKEND`, and `SESSION_BACKEND` env vars.

4. **Official MCP SDK**: Uses `from mcp.server.fastmcp import FastMCP` (the official `mcp` package), NOT the standalone `fastmcp` package. Starlette is used directly for custom routes and middleware.

5. **Stateless HTTP**: The server runs in stateless mode (`stateless_http=True, json_response=True`) for Kubernetes compatibility.

6. **Stateless deployment assumption**: Production deployments may have no persistent local filesystem (Kubernetes stateless pods). The only guaranteed persistent stores are Qdrant (always present) and optionally S3/Redis. Never store application state (migration tracking, caches, metadata) on the local filesystem unless it's explicitly configured as a backend (e.g., `ARTIFACT_BACKEND=filesystem` for dev). Use Qdrant or the configured session/artifact backend for any state that must survive restarts.

7. **Hybrid search**: Search uses both dense vectors (semantic similarity via OpenAI embeddings) and sparse vectors (BM25 keyword matching via FastEmbed) fused with Qdrant's native RRF (Reciprocal Rank Fusion). FastEmbed is a required dependency. Importance reranking is applied in Python post-processing after RRF fusion (FormulaQuery is used only as a fallback when hybrid search fails at query time, see [qdrant/qdrant#6836](https://github.com/qdrant/qdrant/issues/6836)).

## Build / Run / Test

### Local development

```bash
# Install with all optional dependencies
pip install -e ".[all,dev]"

# Run with minimal config (uses OPENAI_API_KEY, data in ~/.mnemory/)
export LLM_API_KEY=sk-your-key
mnemory

# Or run the module directly
python -m mnemory.server
```

### Docker build

```bash
# Build for linux/amd64 (for Kubernetes deployment)
docker buildx build --platform linux/amd64 -t genunix/mnemory:latest .

# Push
docker push genunix/mnemory:latest
```

### Tests

```bash
# Unit tests (fast, no API key needed, default pytest run)
pytest tests/

# E2e tests (require LLM_API_KEY or OPENAI_API_KEY, use real LLM + embedded Qdrant)
pytest -m e2e -v

# Both together
pytest -m '' -v
```

### E2e tests

End-to-end tests in `tests/test_e2e.py` exercise the full pipeline (extraction → storage → search) with real LLM calls and embedded Qdrant. They are **excluded from the default `pytest` run** via `addopts = "-m 'not e2e'"` in `pyproject.toml`.

**When to run e2e tests:**
- After changing `prompts.py` (extraction prompts, dedup logic)
- After changing `memory.py` (business logic, add/search/update/delete)
- After changing `storage/vector.py` or `storage/artifact.py`
- After changing `sanitize.py` (anti-injection, boundary tags)
- Before releases

**Requirements:**
- `LLM_API_KEY` or `OPENAI_API_KEY` environment variable set (auto-skips if missing)
- ~5 minutes runtime (LLM calls + embedding generation)
- `pytest-timeout` installed (120s per test, 180s for `find_memories`)

**Writing e2e tests:**
- Each test class uses a unique `user_id` for isolation (shared session-scoped `MemoryService`)
- Use fuzzy assertions (keyword presence, count ranges) — LLM output is non-deterministic
- Use `infer=False` for deterministic setup data, `infer=True` for testing extraction
- `role="assistant"` requires `infer=True` (anti-injection safeguard) — cannot bypass with `infer=False`
- Use `time.sleep(0.5)` after writes before reads (Qdrant indexing)
- Helper functions: `_mem_text()`, `_all_text()`, `_assert_count_between()`, `_assert_any_contains()`

### Linting

```bash
ruff check mnemory/
ruff format mnemory/
```

## Code Conventions

### Style

- Python 3.11+ features (type unions with `|`, `from __future__ import annotations`)
- Type hints on all function signatures
- Docstrings on all public classes and methods
- `logging` module for all output (never `print()`)
- f-strings for string formatting
- `json.dumps()` for all MCP tool return values (tools return `str`)

### Error handling

- MCP tools catch all exceptions and return JSON error objects: `{"error": true, "message": "..."}`
- Validation errors (`ValueError`) are caught separately from internal errors
- Internal errors are logged with `logger.exception()` for stack traces
- Never let exceptions propagate to the MCP framework

### Configuration

- All config via environment variables (no config files)
- Dataclass-based config objects in `config.py`
- `load_config()` validates required fields at startup
- Defaults are sensible for local development

### Memory metadata

Custom metadata is stored as flat fields in the Qdrant payload alongside standard fields. Our custom fields:

| Field | Type | Description |
|---|---|---|
| `memory_type` | str | preference, fact, episodic, procedural, context |
| `categories` | list[str] | Category tags |
| `importance` | str | low, normal, high, critical |
| `pinned` | bool | Whether to include in core memories |
| `role` | str | "user" or "assistant" — who the memory is about. Always stored as one of these two values. |
| `artifacts` | list[dict] | Artifact metadata (id, filename, content_type, size, created_at) |
| `event_date` | str\|None | ISO 8601 UTC datetime of when the event occurred (None = not set). Used to anchor relative time references during extraction and for temporal queries at search time. |
| `created_at_utc` | str | Our own UTC timestamp |
| `ttl_days` | int\|None | Original TTL setting in days (None = permanent) |
| `expires_at` | str\|None | ISO 8601 expiration timestamp (None = never expires) |
| `decayed_at` | str\|None | When memory entered decayed state (None = active) |
| `labels` | dict\|None | Client-provided key-value metadata (e.g., project, topic, conversation_id). Bypasses LLM extraction. Filterable in search/list queries. |
| `last_accessed_at` | str\|None | Last time returned in search |
| `access_count` | int | Number of times accessed in search |
| `checked_at` | str\|None | ISO 8601 UTC timestamp of last fsck maintenance check (None = unchecked) |
| `memory_layer` | str | "raw" (from remember) or "consolidated" (from add_memory, consolidation, or pre-existing). Controls recall ranking and dedup scope. Absent = treated as "consolidated" for backward compatibility. **Exception**: the consolidation service treats absent `memory_layer` as "raw" when fetching memories for consolidation, since pre-existing memories linked to sessions were created by the remember pipeline before the two-layer system. |
| `superseded_by` | str\|None | ID of the consolidated memory that replaced this raw memory (None = not superseded). Set by the consolidation service. |
| `derived_from` | list[str]\|None | IDs of source raw memories (on consolidated memories produced by consolidation). |
| `lineage_id` | str | Stable assertion lineage. The first revision ID is the lineage ID. |
| `revision` | int | Monotonic revision number within a lineage. |
| `revision_state` | str | `pending`, `active`, `superseded`, `source`, `retracted`, or `aborted`. |
| `supersedes` | str\|None | Previous semantic revision ID in the same lineage. |
| `revision_successor_id` | str\|None | One-time backlink to the successor revision. |
| `revision_operation_id` | str\|None | Durable operation and audit record ID. |
| `provenance_quality` | str | `exact` for new provenance or `legacy_batch` for coarse legacy links. |
| `validation_state` | str | Reserved validation state: `unverified`, `confirmed`, or `disputed`. |

Normal search and list operations return active revisions. History, provenance,
recovery, and privacy-erasure operations can read all revision states.

`delete_memory` retracts a memory and preserves its content and artifacts.
Only the explicit privacy-erasure path physically deletes a lineage.

### Adding a new MCP tool

1. Define the tool function in `server.py` with `@mcp.tool()` decorator
2. Write a detailed docstring (this becomes the tool description for LLMs)
3. Add business logic in `memory.py` (keep `server.py` thin)
4. Return JSON string via `json.dumps()`
5. Wrap in try/except returning error JSON on failure
6. Update `instructions.py` if the tool changes the usage workflow
7. Update `docs/mcp-tools.md` tool table

### Adding a new storage backend

1. Implement the backend class in the appropriate `storage/` file
2. Follow the existing protocol/interface pattern
3. Add configuration fields to `config.py`
4. Add the backend selection logic in the store's `__init__`
5. Add optional dependency to `pyproject.toml`
6. Update `docs/configuration.md` configuration table

### Data migrations

Migrations run automatically on startup before the server accepts requests. State is tracked in a separate `_mnemory_meta` Qdrant collection (not the filesystem) to support stateless Kubernetes deployments.

- Migration classes live in `mnemory/migration.py`
- Each migration has a unique ID (e.g., `001_add_sparse_vectors`), a `description`, and a `run()` method
- `MigrationRunner` loads applied migrations from `_mnemory_meta`, runs pending ones in order, and records completion
- Migrations must be **idempotent** — safe to re-run if interrupted mid-way or if multiple instances start simultaneously
- Migrations must log progress clearly (batch N/M, points processed, elapsed time, estimated total time) since they may take minutes for large instances
- Progress checkpoints are saved to `_mnemory_meta` after each batch for resumable migrations
- The health endpoint returns 503 during migration (Kubernetes readiness probe integration)
- New migrations are added to the `get_migrations()` registry in `migration.py`
- After adding a migration, document it in the changelog with performance expectations — startup migration may take a long time for large instances

## Important Notes

- **Artifact metadata**: Artifact references (id, filename, size, etc.) are stored in the fast memory's metadata in the vector store. The actual content is in S3/filesystem. Deleting a memory should also delete its artifacts.

- **Role parameter**: The `role` parameter controls which extraction prompt is used. On `add_memory`, it defaults to `"user"`. On `remember()`, it defaults to `None` (auto mode). When `role="assistant"`, content is passed to the agent-specific extraction prompt (`_AGENT_SYSTEM_PROMPT` in `prompts.py`), focusing on the assistant's identity, personality, capabilities, actions, and conclusions. When `role="user"`, the user extraction prompt is used. When `role=None` (auto mode, `remember` only), the auto extraction prompt (`_AUTO_REMEMBER_EXTRACTION_SYSTEM_PROMPT`) extracts facts from all participants, attributing each fact to the correct role via a per-fact `role` field in the LLM output. Assistant facts are silently dropped if `agent_id` is not set. The `role` is stored in metadata for filtering in search/list and for section organization in `get_core_memories`. Requires `agent_id` when set to `"assistant"`. The `_trusted` parameter (internal-only, never exposed in MCP/REST) allows `infer=False` with `role="assistant"` for trusted server-side callers like the consolidation service. The `ALLOW_CLIENT_INFER` config option controls whether client-facing tools can use `infer=False` at all.

- **Sub-agents**: Agent IDs support colon-separated namespacing (e.g., `openwebui:bob`). The session validation in `_resolve_agent_id()` allows any `agent_id` that starts with `session_agent_id + ":"`. Sub-agents are fully independent — no memory inheritance from the parent. The `_is_sub_agent()` helper in `server.py` encapsulates the prefix check. `verify_memory_access()` in `memory.py` also allows access to sub-agent memories from the parent session.

- **Two-layer memory system**: Memories have a `memory_layer` field: `raw` (provisional evidence from `remember` endpoint) or `consolidated` (durable knowledge from `add_memory`, consolidation, or pre-existing memories). The `remember` endpoint writes `memory_layer=raw` and its dedup only searches against other raw memories (never modifies consolidated). Session summaries are persisted in the `_mnemory_sessions` Qdrant collection for use by the future consolidation service. The extraction prompt uses a 6-category model (Topic, Exploration, Decision, Fact, Action, Preference/Workflow) to guide what to extract.

- **Session timezone**: The `X-Timezone` HTTP header sets a per-session timezone (IANA name like `Europe/Prague`). Stored in `_session_timezone` ContextVar, accessed via `_get_session_timezone()`. Used by `add_memory()` and `find_memories()` — overrides `DEFAULT_TIMEZONE` env var for naive `event_date` parsing and for computing "today's date" in temporal query generation. Priority chain: explicit tz in event_date string > `X-Timezone` header > `DEFAULT_TIMEZONE` env > server local timezone.
