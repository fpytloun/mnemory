# Architecture

## System Overview

```
+-----------------------------------------------------------------+
|                      Client Layer                                |
|                                                                  |
|  Open WebUI Filter    Open WebUI OpenAPI Tool    Claude Code Plugin |
|  (auto recall/remember)  (LLM-driven ops)       (auto recall)   |
|  Claude Code / OpenCode / Cursor (MCP)                          |
+----------+------------------+----------------------+-------------+
           | REST              | REST (OpenAPI)        | MCP
           v                  v                       v
+-----------------------------------------------------------------+
|                    mnemory                                       |
|                                                                  |
|  FastAPI (/api/)                  MCP (/mcp)                    |
|  +- POST /api/recall              16 MCP tools                  |
|  +- POST /api/remember                                          |
|  +- GET  /api/sessions                                          |
|  +- POST /api/memories            OpenAPI spec at /api/docs     |
|  +- POST /api/memories/search                                   |
|  +- POST /api/memories/find                                     |
|  +- GET  /api/memories/core                                     |
|  +- ... (full CRUD)                                             |
|  +- GET  /api/categories                                        |
|                                                                  |
|  +----------------------------------------------------------+   |
|  |  Shared: MemoryService, VectorStore, LLMClient           |   |
|  +----------------------------------------------------------+   |
|                                                                  |
|  +-----------------+  +-----------------------+                  |
|  |  Fast Memory     |  |  Slow Memory          |                 |
|  |  (Qdrant)        |  |  (S3/MinIO or FS)     |                 |
|  |  Searchable      |  |  Detailed artifacts   |                 |
|  |  facts/summaries |  |  retrieved on demand   |                |
|  +--------+---------+  +-----------+-----------+                 |
+-----------+------------------------+-----------------------------+
            |                        |
            v                        v
     +------------+          +--------------+
     |   Qdrant   |          |  S3 / MinIO  |
     |            |          |  or local FS |
     +------------+          +--------------+
            |
            v
     +------------+
     |  LiteLLM   |
     | or OpenAI  |
     |(LLM+embed) |
     +------------+
```

## How It Works

### Storing a Memory

1. You call `add_memory(content="I drive a Skoda Octavia 2019", categories=["vehicles"])`
2. Content is embedded and similar existing memories are retrieved
3. A single LLM call extracts facts, classifies them, and deduplicates against existing memories
4. If new: creates embedding, stores in Qdrant with metadata
5. If update: merges with existing memory (e.g., "User now drives a Tesla Model 3")

With `infer=false`, the LLM call is skipped — the content is embedded and stored directly. This is much faster (single embedding call vs. LLM + embedding).

**Temporal anchoring:** Pass `event_date` to anchor relative time references. For example, if the content says "I bought it yesterday" and `event_date="2023-05-08"`, the extraction resolves "yesterday" to May 7, 2023 — not today's date. The `event_date` is also stored as metadata for temporal search queries.

### Searching

**search_memories** — fast single-query hybrid search:

1. You call `search_memories(query="what car do I have")`
2. Query is embedded into both dense (OpenAI) and sparse (BM25) vectors
3. Qdrant runs both searches in parallel and fuses results via Reciprocal Rank Fusion (RRF)
4. Results filtered by category/type/date range if specified
5. Importance-based score boost applied in post-processing (small additive bonus based on importance level)
6. Results below score threshold filtered out
7. Returns top N results with artifact indicators

**find_memories** — AI-powered multi-query search:

1. You call `find_memories(question="What do I think about dogs? Should I buy one?")`
2. LLM generates multiple search queries covering different angles and associations (e.g., dogs, pets, partner, house, garden, lifestyle). Temporal-aware — knows today's date to resolve "last week", "recently", etc.
3. Each query runs through `search_memories` independently. When the question names a time period, each query also runs once scoped to that period. This only adds candidates; it never removes undated or out-of-range memories
4. Results are merged and deduplicated across all queries
5. LLM reranks all results by relevance to the original question, using `event_date` metadata for temporal scoring
6. Returns top N results with scores on the same 0.0-1.0 scale

### Core Memories

1. At conversation start, LLM calls `get_core_memories()`
2. Fetches all pinned memories in a single query, organized by `role`:
   - **Agent Identity** (`role=assistant`, fact/preference): name, personality
   - **Agent Knowledge** (`role=assistant`, other types): researched conclusions
   - **Agent Instructions** (`role=user`): user preferences specific to this agent
3. Fetches all pinned user memories (facts, preferences) — shared across agents
4. Fetches top-N non-pinned memories by importance (`CORE_TOP_MEMORIES`, default 10) at or above `CORE_MIN_IMPORTANCE` (default `normal`). These are added after pinned memories in each section.
5. All memories within each section are sorted by importance (critical > high > normal > low), then by recency
6. Fetches recent context memories from last N days (configurable)
7. Each section is capped at `CORE_MAX_PER_SECTION` memories (default 25). Pinned memories are included first, then top-N fills up to the limit.
8. If output exceeds `MAX_CORE_CONTEXT_LENGTH`, recent context entries are trimmed one by one (least important first). Main sections are never truncated.
9. Returns structured text injected into conversation context

### Two-Layer Memory System

Memories have a `memory_layer` field that distinguishes provisional evidence from durable knowledge:

- **Raw** (`memory_layer=raw`): Created by the `remember` endpoint. Provisional, granular, may contain noise. Immediately available for recall and context compaction.
- **Consolidated** (`memory_layer=consolidated`): Created by `add_memory` (manual/MCP), the consolidation service, or pre-existing memories. Durable, canonical knowledge. Preferred in recall ranking.

**Why two layers?** The single extraction call in `remember` cannot perfectly distinguish durable knowledge from conversational noise. Instead of trying to make extraction perfect, the system accepts imperfect raw extraction and adds a background consolidation layer that synthesizes durable knowledge from raw memories + their session summaries.

**Session summaries** are persisted in a separate `_mnemory_sessions` Qdrant collection. Updated on every `remember` call, they capture the narrative arc of conversations — what was discussed, concluded, decided. The consolidation service uses these summaries alongside raw memories to reconstruct conclusions that no single raw memory captured.

**Consolidation flow:**

1. `remember` writes raw memories + updates session summary in `_mnemory_sessions`
2. After a session goes idle (configurable threshold), the consolidation service:
   - Reads the session summary + linked raw memories
   - LLM synthesizes durable knowledge (decisions, preferences, facts, actions)
   - Stores consolidated memories with `derived_from` links to source raw memories
   - Marks raw memories as superseded (except those with artifacts)
3. If the user continues the conversation after consolidation, new raw memories reset the session to idle, triggering re-consolidation on the next check cycle
4. Re-consolidation is append-only: the LLM receives both new raw memories and previously consolidated memories as context, produces new consolidated outputs, and the new consolidated memory IDs are appended alongside previous ones
5. Fsck is an automated maintenance engine for durable memory coherency. Auto-fsck runs incrementally (only changed memories), with a four-phase pipeline: security scan → duplicate/contradiction detection → content quality → metadata normalization. Manual fsck runs a full scan. LLM budget is bounded per run. Raw memories stay in the session/consolidation lifecycle; garbage collection of superseded raw runs independently of fsck
6. Recall ranking boosts consolidated memories and penalizes raw/superseded ones

**Extraction model:** The extraction prompt uses a 6-category model to guide what to extract:

| Category | What to look for | Memory type |
|---|---|---|
| Topic | What the user is working on | `context` |
| Exploration | What was investigated, with outcome | `episodic` |
| Decision | Conclusions, agreements, rejected approaches | `fact` or `episodic` |
| Fact | Stable biographical info or project state | `fact` or `context` |
| Action | What was actually done (code, deployments, etc.) | `episodic` |
| Preference/Workflow | Likes, dislikes, habits, procedures | `preference` or `procedural` |

### Artifacts

1. Agent does deep research, stores summary: `add_memory(content="Researched washing machines. Samsung WW90T best for price/quality.")`
2. Attaches full report: `save_artifact(memory_id="...", content="<5000 word report>", filename="research.md")`
3. Later, search finds the summary with `has_artifacts: true`
4. Agent fetches details: `get_artifact(memory_id="...", artifact_id="...")`

## Layer Responsibilities

| Layer | File | Responsibility |
|---|---|---|
| **Transport** | `server.py` | MCP tool definitions, HTTP routing, auth, serialization |
| **REST API** | `api/` | FastAPI sub-app with OpenAPI spec, CRUD + intelligence endpoints |
| **Business Logic** | `memory.py` | Validation, reranking, core memory assembly, artifact lifecycle |
| **Categories** | `categories.py` | Category validation, prefix matching, counting |
| **TTL** | `ttl.py` | Expiration calculation, decay detection, reinforcement metadata |
| **Vector Storage** | `storage/vector.py` | Direct Qdrant client for all vector operations |
| **Artifact Storage** | `storage/artifact.py` | S3/MinIO and filesystem backends for binary artifacts |
| **Management UI** | `ui/`, `api/ui.py` | Built-in web UI — dashboard, search, browse/CRUD, graph |
| **Instructions** | `instructions.py` | Configurable MCP server instructions (passive/proactive/personality modes) |
| **Sessions** | `session.py` | Server-side memory session tracking with write-through cache |
| **Session Storage** | `storage/session.py` | Session persistence backends (SQLite, Redis, in-memory) |
| **Metrics** | `metrics.py` | Prometheus metrics collection, operation counters, Qdrant gauge aggregation |
| **Configuration** | `config.py` | Environment variable parsing into dataclass configs |
