# REST API

mnemory exposes a full REST API alongside the MCP server. The FastAPI sub-app is mounted at `/api/` with auto-generated OpenAPI spec at `/api/openapi.json` and Swagger UI at `/api/docs`.

Both MCP and REST share the same `MemoryService` backend and authentication middleware. Authentication may use API keys (`Authorization: Bearer <key>` / `X-API-Key`) or Cognis-issued ES256 JWTs (`Authorization: Bearer <jwt>` with `aud=mnemory`). The OpenAPI document and Swagger UI are public metadata for client import; documented API operations still require authentication.

## Memory CRUD

| Endpoint | Method | Description |
|---|---|---|
| `/api/memories` | POST | Add a memory |
| `/api/memories/batch` | POST | Batch add memories |
| `/api/memories/search` | POST | Semantic search |
| `/api/memories/find` | POST | AI-powered multi-query search |
| `/api/memories/core` | GET | Core memories (pinned + recent) |
| `/api/memories/recent` | GET | Recent memories |
| `/api/memories` | GET | List memories |
| `/api/memories/by-ids` | POST | Fetch memories by ID list (batch) |
| `/api/memories/{id}` | PUT | Create a successor revision |
| `/api/memories/{id}` | DELETE | Retract without erasing history |
| `/api/memories/{id}/privacy` | DELETE | Explicitly erase a lineage and unreferenced artifacts |
| `/api/memories/{id}/history` | GET | Get immutable revisions and audit records |
| `/api/memories/{id}/links` | GET | Get exact supersession and derivation links |
| `/api/memories/{id}/artifacts` | POST | Save artifact |
| `/api/memories/{id}/artifacts` | GET | List artifacts |
| `/api/memories/{id}/artifacts/{aid}` | GET | Get artifact (JSON with base64 for binary) |
| `/api/memories/{id}/artifacts/{aid}/download-token` | POST | Generate a signed download token |
| `/api/memories/{id}/artifacts/{aid}/raw` | GET | Download raw artifact bytes (token or API key auth) |
| `/api/memories/{id}/artifacts/{aid}` | DELETE | Delete artifact |
| `/api/categories` | GET | List categories |

Artifact deletion removes the reference from the selected revision. Immutable
history can retain the artifact bytes. Use lineage privacy erasure when the
artifact content must be physically removed from all revisions.

## Revision preconditions

The four REST mutation classes accept an optional `If-Match` header:

- `PUT /api/memories/{id}`
- `DELETE /api/memories/{id}`
- `POST /api/memories/{id}/artifacts`
- `DELETE /api/memories/{id}/artifacts/{aid}`

The value is the positive integer `revision` returned by memory and mutation
responses. A bare value (`If-Match: 3`) or strong quoted value
(`If-Match: "3"`) is valid. Weak entity tags, lists, wildcards, snapshot
hashes, zero, negative values, and other strings return HTTP 400.

Memory update bodies and memory delete queries retain `expected_revision`.
When a request supplies both forms, their integer values must match. A
conflict returns HTTP 409 before mutation. A valid but stale revision also
returns HTTP 409. Artifact mutation responses retain `lineage_id`, `revision`,
`artifact_revision`, and replay state. Signed artifact download URLs are not
affected by this contract.

## Fsck operation recovery

`POST /api/fsck/audit` starts a bounded audit for the exact memory IDs and
revisions of one existing fsck operation. The request accepts at most 20
targets. The audit does not search a neighbor corpus, stamp `checked_at`, or
change memory metadata. Its completed journal record contains fingerprints and
counts, but no memory content.

`POST /api/fsck/operations/{operation_id}/re-evaluate` compares an old fsck
operation with a newer completed exact audit bound to that operation. A normal
or historically rehydrated check is rejected. The default
`terminalize=false` is mutation-free and does not apply actions. With explicit
`terminalize=true`, the endpoint marks the old journal entry `superseded`.
Applying the fresh issue remains a separate explicit request.

## Failed-session diagnostics and retry

`GET /api/sessions/failed/diagnostics` returns aggregate failure metadata.
`GET /api/sessions/failed/retry-eligibility` returns authorized, content-free
candidate metadata. Eligibility validates every linked raw-memory revision for
existence, exact scope, raw layer, active state, and supersession. Diagnostics
include stable ineligibility reason counts.

`POST /api/sessions/failed/retry` defaults to
`dry_run=true`, which creates no operation record and changes no session.

Mutation requires `LEGACY_FAILED_RETRY_ENABLED=true`, an idempotency key, and
an explicit list of no more than ten eligible sessions. Retries run
sequentially and stop on configured failure, revision, or timeout conditions.
The execution path repeats linked-input validation immediately before it
creates or claims the retry journal.

`GET /api/memories/core` supports:

- `recent_days` — how many days of recent context to include
- `include_stats` — when `true`, also return structured stats for the assembled core context

Example response with `include_stats=true`:

```json
{
  "text": "## User Facts\n- User lives in Prague\n...",
  "stats": {
    "memory_count": 4,
    "char_count": 1840,
    "estimated_tokens": 460,
    "by_type": {"fact": 2, "preference": 1, "episodic": 1},
    "by_role": {"user": 4},
    "by_section": {"user_facts": 2, "user_preferences": 1, "recent_user_activity": 1},
    "section_labels": {"user_facts": "User Facts", "user_preferences": "User Preferences", "recent_user_activity": "User Activity"},
    "sections": {"user_facts": ["mem_1", "mem_2"], "user_preferences": ["mem_3"], "recent_user_activity": ["mem_4"]},
    "memory_ids": ["mem_1", "mem_2", "mem_3", "mem_4"]
  }
}
```

Core memories exclude `memory_layer=raw` by default, so only consolidated (and legacy no-layer) memories are included in the default injected context.

## Session Summaries

Persistent session summaries from the remember endpoint, used by the consolidation service.

| Endpoint | Method | Description |
|---|---|---|
| `/api/sessions` | GET | List session summaries with pagination, search, and sorting (`offset`, `limit`, `consolidation_state`, `q`, `sort_by`, `sort_dir`) |
| `/api/sessions/{id}` | GET | Get a single session summary |
| `/api/sessions/{id}` | DELETE | Delete a session summary (optional `delete_memories=true` to also delete linked raw memories) |
| `/api/sessions/{id}/consolidate` | POST | Trigger consolidation for a specific session (returns result) |

`GET /api/sessions` returns:

```json
{
  "sessions": [
    {
      "session_id": "ses_...",
      "summary": "...",
      "created_at": "2026-03-28T10:00:00+00:00",
      "updated_at": "2026-03-28T10:05:00+00:00",
      "consolidation_state": "idle"
    }
  ],
  "total": 123,
  "offset": 0,
  "limit": 25,
  "has_more": true,
  "total_truncated": false
}
```

- `q`: case-insensitive substring match against the stored session summary text and session ID
- `sort_by`: `updated_at` (default) or `created_at`
- `sort_dir`: `desc` (default) or `asc`
- `total`: count after all filters/search are applied, before paging
- `total_truncated`: true if the server hit its internal safety cap while scanning very large session sets

## Memory Check (fsck)

Built-in memory consistency checker. Runs a three-phase pipeline to detect quality issues and suggest fixes:

| Endpoint | Method | Description |
|---|---|---|
| `/api/fsck` | POST | Start a memory check (runs in background) |
| `/api/fsck/audit` | POST | Start an exact-target, audit-only check |
| `/api/fsck/auto-run` | POST | Trigger an immediate auto-fsck run for the current user |
| `/api/fsck/{id}` | GET | Poll check status, progress, and results |
| `/api/fsck/{id}/apply` | POST | Apply selected fixes from a completed check |

### Four-Phase Pipeline

1. **Security scan** (instant) — regex-based detection of prompt injection patterns, confirmed via LLM re-evaluation
2. **Duplicate detection** — vector similarity clustering + LLM evaluation of each cluster for duplicates and contradictions
3. **Content quality** — LLM batch evaluation for broken, meaningless, or unsalvageable memories
4. **Metadata normalization** — LLM batch evaluation for wrong memory_type, categories, importance, pinned, or role

Phases 2–4 run LLM calls in parallel (`FSCK_LLM_CONCURRENCY`, default 4 workers) for ~4x speedup on large memory sets. Results are cached with configurable TTL (`FSCK_CACHE_TTL`, default 24 hours) so you can review issues in the UI and apply fixes without re-running the check.

By default, fsck focuses on durable memories (consolidated memories plus legacy memories without a `memory_layer` field). Raw provisional memories are excluded unless the caller explicitly opts in with `include_raw=true`.

**Incremental mode:** Auto-fsck defaults to incremental processing — only memories changed since the last maintenance run are checked. Manual fsck runs a full scan. The `checked_at` metadata field tracks when each memory was last checked. LLM budget is bounded by `FSCK_MAX_LLM_CALLS` (default 200 per run); phases execute in priority order (security → duplicates → content quality → metadata normalization), so under budget pressure earlier phases take precedence and later phases catch up in subsequent runs.

**Issue types:** `duplicate`, `contradiction`, `quality`, `reclassify`, `security`

**Workflow:** Start check → poll until completed → review issues in UI → select and apply fixes. Each fix is a set of actions (update content, update metadata, delete memory). Auto-fsck applies qualifying fixes automatically based on confidence and severity thresholds.

## Intelligence Layer

Two high-level endpoints designed for plugin-driven automatic memory management:

### POST /api/recall

Combined initialize + search. Call on each user message.

```json
{
  "session_id": null,
  "query": "Should I buy a dog?",
  "include_instructions": true,
  "managed": true,
  "score_threshold": 0.5,
  "labels": {"project": "myapp"}
}
```

- First call (no `session_id`): creates session, returns instructions + core memories + `find_memories` results
- Subsequent calls: returns only NEW relevant memories via fast `search_memories` (no LLM), filtering out already-returned IDs
- `score_threshold`: optional per-request minimum score (0.0-1.0) for search results, applied on top of server's `SEARCH_SCORE_THRESHOLD`. Prevents context bloat from weak matches on follow-up messages.
- `labels`: optional label filter — only return memories matching these key-value pairs. See [Memory Model — Labels](memory-model.md#labels).
- Graceful degradation: `find_memories` fails -> `search_memories` -> core memories only -> empty response

### POST /api/remember

Fire-and-forget memory storage. Call after each exchange.

```json
{
  "session_id": "sess_abc123",
  "messages": [
    {"role": "user", "content": "I just moved to Berlin"},
    {"role": "assistant", "content": "That's exciting!"}
  ],
  "role": null,
  "labels": {"conversation_id": "conv-123"}
}
```

The `role` parameter controls the extraction point of view:

| Value | Behavior |
|---|---|
| `null` (default) | **Auto mode** — extracts facts from ALL participants. Each fact is attributed to the correct role (`user` or `assistant`). Assistant facts are silently dropped if no `agent_id` is set in the session. |
| `"user"` | Extracts only user facts, suppresses assistant content. |
| `"assistant"` | Extracts only assistant facts (identity, recommendations, conclusions). Requires `agent_id` via `X-Agent-Id` header. |

- Returns `{"accepted": true}` immediately, processes in background
- Reuses the same extraction pipeline as `add_memory(infer=True)` — no extra LLM calls
- `labels`: optional key-value metadata attached to all extracted memories. Bypasses LLM — stored as-is. See [Memory Model — Labels](memory-model.md#labels).
- Stored memory IDs are added to the session to prevent echo on next recall
- Rate limited per user (configurable via `REMEMBER_RATE_LIMIT`)

## Memory Sessions

The recall/remember endpoints use server-side sessions (`MemorySession`) to track which memories the client already has. This prevents context bloat by only returning new memories on subsequent calls.

- Sessions are created on first recall and expire after idle timeout (default 24 hours, configurable via `MEMORY_SESSION_TTL`)
- Both recall and remember calls reset the idle timer (auto-prolong), so active conversations never expire
- Sessions are persisted to a pluggable backend (SQLite by default, Redis for clustered deployments) and survive server restarts
- Losing a session is harmless — next recall creates a new one
- Periodic background sweep cleans up expired sessions from both the in-memory cache and the backend

### Session Persistence

Sessions use a write-through cache: an in-memory dict for fast reads, with all mutations written to the backend for durability. On cache miss (e.g., after restart), sessions are loaded lazily from the backend.

| Backend | Use case | Config |
|---|---|---|
| `sqlite` (default) | Single-node, local development, `uvx mnemory` | `SESSION_PATH` (default `~/.mnemory/sessions.db`) |
| `redis` | Clustered deployments, multiple replicas | `REDIS_URL` (e.g., `redis://host:6379/0`) |
| `memory` | Tests, no persistence needed | No additional config |

Auto-detection: if `REDIS_URL` is set and `SESSION_BACKEND` is not explicitly set, defaults to `redis`. See [Configuration](configuration.md) for all session env vars.

## Periodic Maintenance

mnemory includes a built-in background maintenance service that automatically runs memory consistency checks and applies fixes on a configurable schedule.

Enable it by setting `FSCK_AUTO_INTERVAL` to a non-zero number of hours:

```bash
FSCK_AUTO_INTERVAL=24          # Run every 24 hours
FSCK_AUTO_MIN_CONFIDENCE=0.95  # Only apply fixes with >=95% confidence
FSCK_AUTO_MIN_SEVERITY=medium  # Only apply medium or high severity fixes
```

**How it works:**

1. The maintenance loop sleeps for the configured interval, then wakes up
2. All users in the collection are enumerated via a Qdrant scroll
3. For each user, a full fsck check is run (security scan -> duplicate detection -> quality check)
4. Issues that meet both the confidence and severity thresholds are automatically applied
5. Results are recorded in Prometheus metrics (`mnemory_autofsck_*` counters + last-run timestamp)
6. Per-user errors are isolated — one failing user does not stop the run

The loop is **sleep-first**: it waits the full interval before the first run, so startup is not impacted. Auto-applied fixes are visible in the [management UI](management-ui.md) (Check tab status banner) and in the Grafana dashboard (Auto-fsck row).
