# Configuration Reference

All configuration is via environment variables. Defaults are optimized for local development — just set `OPENAI_API_KEY` and run.

Data is stored in `~/.mnemory/` by default. Override with `DATA_DIR` env var. In Docker, `DATA_DIR` is set to `/data` for volume mounting.

## LLM & Embeddings

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | | OpenAI API key (also used as fallback for `LLM_API_KEY`) |
| `OPENAI_API_BASE` | | OpenAI-compatible API base URL (also used as fallback for `LLM_BASE_URL`) |
| `LLM_API_KEY` | (falls back to `OPENAI_API_KEY`) | API key for LLM provider |
| `LLM_BASE_URL` | (falls back to `OPENAI_API_BASE`, then `https://api.openai.com/v1`) | OpenAI-compatible API base URL |
| `LLM_MODEL` | `gpt-5.4-mini` | LLM model for fact extraction and deduplication |
| `LLM_REASONING_EFFORT` | | Reasoning effort for LLM (none/minimal/low/medium/high). Models that don't support it auto-skip. |
| `EMBED_MODEL` | `text-embedding-3-small` | Embedding model |
| `EMBED_BASE_URL` | (falls back to `LLM_BASE_URL`) | Separate base URL for embeddings |
| `EMBED_API_KEY` | (falls back to `LLM_API_KEY`) | Separate API key for embedding provider |
| `EMBED_DIMS` | `1536` | Embedding dimensions |

### Model Selection

The default `gpt-5.4-mini` provides a good balance of quality and cost. Any OpenAI-compatible model with structured output (JSON schema) support works.

| Model | Provider | Input $/M | Output $/M | LoCoMo Score | Notes |
|---|---|---|---|---|---|
| `gpt-5.4-mini` | OpenAI | $0.75 | $4.50 | 73.2* | Default. Supports prompt caching (50% input discount) and batch API (50% off) |
| `gpt-oss-120b` | Groq | $0.15 | $0.60 | 70.5 | Recommended budget option. ~5x cheaper than gpt-5.4-mini |
| `gpt-oss-120b` | OpenAI | $0.039 | $0.19 | 70.5 | Same model, cheapest option via OpenAI direct |
| `gpt-5-mini` | OpenAI | $0.25 | $2.00 | 73.2 | Previous default, still fully supported |

*LoCoMo score measured with `gpt-5-mini`. `gpt-5.4-mini` is expected to match or exceed this.

**`gpt-oss-120b` setup** requires a separate embedding provider since gpt-oss models don't serve embeddings:

```bash
LLM_MODEL=gpt-oss-120b
LLM_BASE_URL=https://api.groq.com/openai/v1   # or https://api.openai.com/v1
LLM_API_KEY=gsk-your-groq-key
EMBED_API_KEY=sk-your-openai-key               # embeddings stay on OpenAI
EMBED_BASE_URL=https://api.openai.com/v1
```

**Not recommended:** `gpt-oss-20b` fails security/injection e2e tests and has lower extraction quality (78/80 vs 80/80 for `gpt-oss-120b`). The smaller model is too aggressive in refusing to extract facts from content that contains boundary tag characters, and occasionally fails to generate session summaries.

## Data Storage

| Variable | Default | Description |
|---|---|---|
| `DATA_DIR` | `~/.mnemory` | Base data directory (all paths below default to subdirectories of this) |
| `QDRANT_HOST` | | Qdrant host (empty = local embedded mode, set for remote Qdrant) |
| `QDRANT_PORT` | `6333` | Qdrant port |
| `QDRANT_PATH` | `{DATA_DIR}/qdrant` | Local embedded Qdrant storage path (used when `QDRANT_HOST` is not set) |
| `QDRANT_API_KEY` | | Qdrant API key (optional) |
| `QDRANT_COLLECTION` | `mnemory` | Qdrant collection name |
| `ARTIFACT_BACKEND` | `filesystem` | Artifact backend: `filesystem` (local) or `s3` (production) |
| `ARTIFACT_PATH` | `{DATA_DIR}/artifacts` | Local filesystem path for artifacts |
| `S3_ENDPOINT` | `http://localhost:9000` | S3/MinIO endpoint URL |
| `S3_ACCESS_KEY` | (required for s3) | S3 access key |
| `S3_SECRET_KEY` | (required for s3) | S3 secret key |
| `S3_BUCKET` | `mnemory` | S3 bucket name |
| `S3_REGION` | | S3 region (optional) |

## Server

| Variable | Default | Description |
|---|---|---|
| `MCP_HOST` | `0.0.0.0` | Listen host |
| `MCP_PORT` | `8050` | Listen port |
| `MCP_API_KEY` | | Single API key for authentication (empty = no auth) |
| `MCP_API_KEYS` | | JSON dict mapping API keys to user IDs (see [Authentication](#authentication) below) |
| `MNEMORY_JWT_PUBLIC_KEY` | | Path to a Cognis ES256 public key PEM for JWT validation |
| `MNEMORY_JWKS_URL` | | JWKS URL for Cognis-issued JWT validation (mutually exclusive with `MNEMORY_JWT_PUBLIC_KEY`) |
| `INSTRUCTION_MODE` | `proactive` | LLM behavioral instructions: `passive`, `proactive`, or `personality` (see [Instruction Modes](#instruction-modes) below) |
| `ENABLE_DELETE_ALL` | `false` | Enable the `delete_all_memories` tool (destructive, disabled by default) |
| `ENABLE_METRICS` | `true` | Enable the `/metrics` Prometheus endpoint |
| `METRICS_CACHE_TTL` | `600` | Cache TTL in seconds for Qdrant gauge aggregation on `/metrics` |
| `MGMT_PORT` | | Management port for `/health` and `/metrics` (see [Management Port](#management-port) below) |
| `MGMT_HOST` | (falls back to `MCP_HOST`) | Bind host for the management port |
| `SERVER_BASE_URL` | | Base URL for generated download URLs (e.g., `https://mnemory.example.com`). If not set, URLs are relative paths. |
| `DOWNLOAD_TOKEN_TTL` | `3600` | Default lifetime in seconds for artifact download tokens (1 hour) |
| `DOWNLOAD_TOKEN_MAX_TTL` | `86400` | Maximum allowed lifetime in seconds for download tokens (24 hours) |
| `LOG_LEVEL` | `INFO` | Logging level |

## Memory Behavior

| Variable | Default | Description |
|---|---|---|
| `MAX_MEMORY_LENGTH` | `1000` | Max characters for fast memory content |
| `MAX_ARTIFACT_SIZE` | `10485760` | Max bytes per artifact (10MB) |
| `MAX_CORE_CONTEXT_LENGTH` | `4000` | Max characters for recent context section in get_core_memories. Only affects graceful per-entry trimming of recent context — main sections are never truncated. |
| `DEFAULT_RECENT_DAYS` | `7` | Default days for recent context in core memories |
| `RECENT_LIMIT_USER` | `25` | Max recent user memories to include |
| `RECENT_LIMIT_AGENT` | `25` | Max recent agent memories to include |
| `AUTO_CLASSIFY` | `true` | Auto-classify memory metadata (type, categories, importance, pinned) via LLM when not provided |
| `CLASSIFY_CACHE_TTL` | `300` | TTL in seconds for the category cache used during auto-classification |
| `CORE_MEMORIES_CACHE_TTL` | `300` | TTL in seconds for the core memories cache (get_core_memories). Set to 0 to disable. Invalidated on memory mutations. |
| `CORE_TOP_MEMORIES` | `10` | Max non-pinned memories to include in core sections by importance. Set to 0 to disable (only pinned memories). |
| `CORE_MAX_PER_SECTION` | `25` | Max memories per section in core memories (Agent Identity, User Facts, etc.). Pinned memories are included first, then top-N fills up to the limit. Set to 0 for unlimited. |
| `CORE_MIN_IMPORTANCE` | `normal` | Minimum importance for non-pinned memories in core sections. Options: low, normal, high, critical |
| `TTL_FACT` | (none) | Default TTL in days for `fact` memories (empty = permanent) |
| `TTL_PREFERENCE` | (none) | Default TTL in days for `preference` memories (empty = permanent) |
| `TTL_EPISODIC` | `90` | Default TTL in days for `episodic` memories |
| `TTL_PROCEDURAL` | `60` | Default TTL in days for `procedural` memories |
| `TTL_CONTEXT` | `7` | Default TTL in days for `context` memories |
| `TRACK_MEMORY_ACCESS` | `true` | Update last_accessed_at and reset TTL on search/recall |
| `SEARCH_SCORE_THRESHOLD` | `0.30` | Minimum score for dense-only search results (0.0-1.0). Only used as fallback when hybrid search fails at query time |
| `SEARCH_SCORE_THRESHOLD_HYBRID` | `0.0` | Minimum score for hybrid (RRF) search results. RRF score range depends on Qdrant's k constant (default k=1 gives ~0.1-1.0, similar to cosine; k=60 gives ~0.01-0.03). Default 0.0 disables threshold filtering |
| `DEDUP_SIMILARITY_THRESHOLD` | `0.4` | Minimum similarity for deduplication matching during memory ingestion |
| `SEARCH_SIMILARITY_WEIGHT` | `0.9` | Weight for cosine similarity in search ranking (remainder goes to importance). Default 0.9 = 90% similarity, 10% importance |
| `SEARCH_SPARSE_MODEL` | `Qdrant/bm25` | BM25 sparse embedding model for hybrid search. Default is recommended |
| `SEARCH_KEYWORD_WEIGHT` | `0.2` | **Deprecated.** Ignored — replaced by BM25 hybrid search. Kept for backward compatibility |
| `DEFAULT_TIMEZONE` | | Default IANA timezone for naive `event_date` values (e.g., `Europe/Prague`). Empty = server local timezone. Can be overridden per session via `X-Timezone` header |
| `FIND_MEMORIES_QUERIES` | `5` | Maximum number of search queries the LLM generates for `find_memories` (may return fewer or zero) |
| `FIND_LLM_MODEL` | | Override LLM model for find/ask pipeline — query generation, reranking, answer generation (empty = use main `LLM_MODEL`). Smaller models like `gpt-4.1-nano` work well for these structured tasks |
| `FIND_REASONING_EFFORT` | `low` | Reasoning effort for find/ask LLM calls. Defaults to `low` since query generation and reranking are simple structured tasks. Set empty to inherit `LLM_REASONING_EFFORT` |
| `MAX_INPUT_LENGTH` | `400000` | Max chars for input to `add_memory(infer=True)` and `remember()`. ~100k tokens |
| `MEMORY_SESSION_TTL` | `86400` | Default session idle TTL in seconds (24 hours). Each access (recall/remember) resets the timer |
| `MEMORY_SESSION_SWEEP_INTERVAL` | `300` | Interval in seconds between session cleanup sweeps (5 minutes) |
| `SESSION_BACKEND` | `sqlite` | Session persistence backend: `memory` (no persistence), `sqlite` (local file), `redis` (clustered). Auto-detects `redis` if `REDIS_URL` is set |
| `SESSION_PATH` | `{DATA_DIR}/sessions.db` | SQLite database path for session persistence (only used when `SESSION_BACKEND=sqlite`) |
| `REDIS_URL` | | Redis connection URL for session persistence (e.g., `redis://host:6379/0`). Required when `SESSION_BACKEND=redis`. When set without explicit `SESSION_BACKEND`, auto-selects `redis` |
| `RECALL_MAX_RESULTS` | `10` | Max search results returned by recall endpoint |
| `REMEMBER_RATE_LIMIT` | `10` | Max remember requests per minute per user. 0 = no limit |
| `FSCK_CACHE_TTL` | `86400` | How long memory check results are cached in seconds (24 hours) |
| `FSCK_LLM_CONCURRENCY` | `4` | Max concurrent LLM calls during memory check. Set to 1 for sequential |
| `FSCK_LLM_MODEL` | | Override LLM model for memory check (empty = use main `LLM_MODEL`) |
| `FSCK_REASONING_EFFORT` | `medium` | Reasoning effort for memory check LLM calls (empty = use main `LLM_REASONING_EFFORT`). Defaults to `medium` for better accuracy |
| `FSCK_AUTO_INTERVAL` | `0` | Interval in hours between automatic background memory checks. `0` = disabled |
| `FSCK_AUTO_MIN_CONFIDENCE` | `0.95` | Minimum confidence score (0.0-1.0) for a fix to be auto-applied |
| `FSCK_AUTO_MIN_SEVERITY` | `medium` | Minimum severity for a fix to be auto-applied. Options: `low`, `medium`, `high` |
| `FSCK_MAX_MEMORIES` | `5000` | Maximum memories per check run. If a user has more durable memories, a random sample is taken. `0` = no limit |
| `FSCK_MAX_LLM_CALLS` | `200` | Maximum LLM calls per check run. When budget is exhausted, the pipeline stops gracefully; unchecked memories are picked up in the next incremental auto-fsck run. `0` = no limit |
| `LABELS_MAX_FIELDS` | `20` | Max number of label keys per memory |
| `LABELS_MAX_KEY_LENGTH` | `64` | Max length of a label key (alphanumeric + underscore only) |
| `LABELS_MAX_VALUE_LENGTH` | `1000` | Max length of a string label value |
| `LABELS_INDEXES` | | Comma-separated list of label keys to index in Qdrant for fast filtering (e.g., `project,topic,conversation_id`). Only needed for remote Qdrant with large datasets |
| `ALLOW_CLIENT_INFER` | `true` | Whether client-facing tools (MCP, REST) can use `infer=False`. When `false`, `infer` is always forced to `true` for client requests. Internal callers (consolidation, remember) are unaffected |
| `CONSOLIDATION_LLM_MODEL` | | Optional separate LLM model for consolidation. Falls back to `FSCK_LLM_MODEL`, then `LLM_MODEL`. Consolidation benefits from a stronger model |
| `CONSOLIDATION_REASONING_EFFORT` | | Reasoning effort for consolidation LLM calls. Falls back to `FSCK_REASONING_EFFORT`, then `LLM_REASONING_EFFORT` |
| `CONSOLIDATION_CHECK_INTERVAL` | `300` | How often the consolidation loop checks for idle sessions (seconds). Separate from idle threshold which controls eligibility |
| `CONSOLIDATION_IDLE_THRESHOLD` | `3600` | Seconds before a session is eligible for within-session consolidation (1 hour) |
| `CONSOLIDATION_BATCH_SIZE` | `100` | Max raw memories per consolidation LLM call. Sessions with more are split into time-based batches |
| `CONSOLIDATION_MAX_RAW_PER_USER` | `500` | Max raw memories to process per user during cross-session consolidation |
| `CONSOLIDATION_MAX_CLUSTERS` | `20` | Max clusters to evaluate during cross-session consolidation |
| `CONSOLIDATION_RAW_RETENTION_DAYS` | `30` | Days to retain superseded raw memories before garbage collection. Artifact-bearing memories are always retained |
| `RECALL_RAW_PENALTY` | `0.05` | Score penalty for raw-layer memories in search results |
| `RECALL_SUPERSEDED_PENALTY` | `0.15` | Score penalty for superseded raw memories in search results |

## Instruction Modes

The `INSTRUCTION_MODE` env var controls how aggressively the LLM uses memory tools. These instructions are sent via the MCP protocol and injected into the LLM's system prompt by supporting clients.

| Mode | Description |
|---|---|
| `passive` | Soft guidance — use memory when asked or clearly relevant. Minimal behavioral directives. |
| `proactive` | **Default.** Always search before answering, proactively store new information, treat memory as primary context. Plug-and-play magic. |
| `personality` | Proactive + identity development. The agent can develop and maintain its own personality, knowledge, and "soul" through `role=assistant` memories. |

For most setups, `proactive` (default) is the right choice — it makes memory work automatically without any system prompt configuration. Use `personality` when all connected agents should develop their own identity. Use `passive` for manual control.

To activate personality behavior for a **specific agent** while keeping the server in `proactive` mode, add the personality snippet to that agent's system prompt instead. See [system-prompts/](system-prompts/) for templates.

## Authentication

mnemory supports two authentication modes:

- **Standalone mode** — API keys via `MCP_API_KEY` / `MCP_API_KEYS`
- **Cognis integration mode** — ES256 JWTs signed by Cognis via `Authorization: Bearer <jwt>`

JWT validation is optional and additive. If both JWT validation and API keys are configured, mnemory tries JWT validation first for Bearer tokens and falls back to API key matching if JWT validation fails.

### Cognis JWT Validation

Configure one verifier source:

```bash
MNEMORY_JWT_PUBLIC_KEY=/path/to/cognis-public.pem
# or
MNEMORY_JWKS_URL=https://cognis.example.com/.well-known/jwks.json
```

- Verifies `iss="cognis"`
- Requires `aud` to include `"mnemory"`
- Binds `user_id` from JWT `sub`
- Uses JWT `agent_id` when present, otherwise falls back to `X-Agent-Id`
- Rejects requests when JWT `agent_id` and `X-Agent-Id` disagree

### Session-Level Identity (`MCP_API_KEYS`)

Map API keys to user IDs so the LLM doesn't need to pass `user_id` in every tool call:

```bash
MCP_API_KEYS='{"mnm-key-for-filip": "filip", "mnm-shared-service-key": "*"}'
```

- `"key": "username"` — authenticates AND binds `user_id=username` to the session
- `"key": "*"` — authenticates only (wildcard), `user_id` must come from identity headers or tool parameter

**Identity resolution priority (user_id):**
1. Cognis JWT `sub` claim — most secure, cannot be overridden
2. API key mapping (non-wildcard) — cannot be overridden
3. `X-User-Id` HTTP header — explicit identity header
4. `X-OpenWebUI-User-Email` HTTP header — automatic Open WebUI integration
5. Tool parameter — backward compatible fallback

**Agent ID** is set via the `X-Agent-Id` HTTP header per client connection:
- Open WebUI: `X-Agent-Id: open-webui`
- Claude Code: `X-Agent-Id: claude-code`

**Timezone** is set via the `X-Timezone` HTTP header per client connection. This overrides the `DEFAULT_TIMEZONE` env var for the session, affecting how naive `event_date` values are interpreted. Use an IANA timezone name (e.g., `Europe/Prague`, `America/New_York`).

`MCP_API_KEY` (single key) is kept for backward compatibility — it authenticates but does not bind to a user. If both `MCP_API_KEYS` and `MCP_API_KEY` are set, `MCP_API_KEYS` is checked first, then `MCP_API_KEY` as fallback.

When mnemory runs in JWT-only mode without API keys, artifact download tokens keep their existing ephemeral signing behavior across restarts. This is acceptable for Stage 0 Cognis integration, but production deployments that need stable raw artifact URLs should continue to configure an API key until a dedicated download-token signing secret is introduced.

## Management Port

By default, `/health` and `/metrics` are served on the main port (`MCP_PORT`) and go through standard API key authentication. Set `MGMT_PORT` to run unauthenticated copies of these endpoints on a separate port — useful for Kubernetes probes and Prometheus scraping:

```bash
MGMT_PORT=9090          # /health and /metrics on port 9090, no auth
MGMT_HOST=127.0.0.1     # Optional: bind management to localhost only
```

When `MGMT_PORT` is set and differs from `MCP_PORT`:
- `/health` and `/metrics` are served on `MGMT_PORT` without auth
- The main port (`MCP_PORT`) still serves authenticated `/health` and `/metrics`
- All MCP and REST API endpoints remain on the main port with auth

When `MGMT_PORT` is not set (default):
- `/health` and `/metrics` are on the main port with standard auth
- Kubernetes probes and Prometheus must send a valid API key
