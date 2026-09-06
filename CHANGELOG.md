# Changelog

## [1.13.0] — 2026-09-06

### Added

- **Immutable memory revisions** — Semantic updates create successor points in the existing Qdrant memory collection.
- **Provenance and audit** — Added exact derivation links, history APIs, and the `_mnemory_operations` audit collection.
- **Stale-write protection** — Update and retraction APIs accept expected revisions and lineage-scoped idempotency keys.
- **Evidence confirmation** — Inferred deduplication can confirm an active revision from new independent user evidence.
- **Optional slow decay** — Disabled-by-default gradual ranking decay uses access and validation timestamps.
- **Recoverable fsck journal** — Fsck operations retain deterministic checkpoints and terminal failure or skip outcomes.
- **Legacy failure controls** — Added content-free diagnostics and default-disabled bounded retry controls.

### Changed

- **Delete semantics** — Ordinary deletion now creates a retraction tombstone and preserves history. Explicit privacy deletion erases the full lineage.
- **Consolidation evidence** — Consolidation records sources per output and retains raw evidence instead of physically deleting it.
- **Migration 005** — Startup backfills revision metadata without changing existing IDs, text, vectors, or legacy payload fields.
- **Migration 006** — Startup adds validation indexes and neutral legacy validation projections without changing content or vectors.
- **Recall reinforcement** — Access timestamps and TTL refresh now apply only after final result delivery.
- **Migration 007** — Failed sessions without diagnostics are classified as legacy metadata without retrying them.

## [1.12.2] — 2026-08-23

### Bug Fixes

- **Consolidation recovery isolation** — Continued recovery and consolidation for other sessions when one session recovery fails.
- **Metrics and maintenance Qdrant scans** — Reduced Qdrant scan load, refreshed metrics asynchronously, and kept complete metric snapshots available during refresh.

## [1.12.1] — 2026-05-07

### Bug Fixes

- **Shared-agent owner isolation** — Isolated shared-agent owner memories so shared-agent recall and access checks do not cross owner boundaries ([`fbf960d`](https://github.com/fpytloun/mnemory/commit/fbf960d)).
- **Management UI header overflow** — Prevented dashboard header controls from overflowing and kept desktop headers on a single line ([`646c4f5`](https://github.com/fpytloun/mnemory/commit/646c4f5), [`27c0542`](https://github.com/fpytloun/mnemory/commit/27c0542)).

## [1.12.0] — 2026-04-23

### Added

- **Owner-scoped shared-agent recall** — Added `owner_id`-aware recall, search, recent-memory, and listing flows so shared agents can load owner-shared agent memories without leaking unrelated personal memory. This includes dedicated core-memory assembly for shared-agent sessions ([`3fce5a6`](https://github.com/fpytloun/mnemory/commit/3fce5a6)).

### Bug Fixes

- **Legacy shared-memory backfill** — Backfilled legacy owner IDs during migration and search fallback paths so pre-existing shared memories continue to resolve correctly after the new owner-scoped sharing model landed ([`87ca90f`](https://github.com/fpytloun/mnemory/commit/87ca90f)).

### Testing

- **Owner fallback coverage** — Tightened unit and migration coverage for legacy owner fallback and shared-agent recall edge cases ([`b4d8001`](https://github.com/fpytloun/mnemory/commit/b4d8001)).

## [1.11.1] — 2026-04-14

### Bug Fixes

- **Authenticated health endpoints on the main port** — Kept authenticated health endpoints available on the main server port so health checks continue to work correctly behind authenticated deployments and proxies ([`5f785a6`](https://github.com/fpytloun/mnemory/commit/5f785a6))

## [1.11.0] — 2026-04-01

### Added

- **Hermes Agent plugin** (`integrations/hermes/`): New plugin integration for the Hermes Agent framework with automatic memory recall and capture ([`6ebe8b3`](https://github.com/fpytloun/mnemory/commit/6ebe8b3))
- **Exchange token endpoint**: `POST /api/auth/exchange` for cross-service SSO token exchange ([`2222e8b`](https://github.com/fpytloun/mnemory/commit/2222e8b))
- **Cognis session cookie auth**: Support `cognis_session` cookie for cross-service SSO alongside existing API key and JWT auth ([`3be1824`](https://github.com/fpytloun/mnemory/commit/3be1824))
- **Separate embedding API key**: New `EMBED_API_KEY` config for using a different provider for embeddings than for LLM ([`b5af1c4`](https://github.com/fpytloun/mnemory/commit/b5af1c4))

### Improved

- **Dashboard category legend** — The By Category chart now uses a right-side scrollable legend so large project/category sets stay readable ([`66beccb`](https://github.com/fpytloun/mnemory/commit/66beccb))
- **Graph filters** — The Graph tab now supports category and memory-layer filters in addition to the existing type toggles ([`66beccb`](https://github.com/fpytloun/mnemory/commit/66beccb))
- **Core context preview** — The User tab now shows a read-only preview of the exact default `get_core_memories` context an agent receives, including agent-specific scope when an agent is selected ([`66beccb`](https://github.com/fpytloun/mnemory/commit/66beccb))
- **Core memories exclude raw layer** — `get_core_memories`, initialize/recall flows, and the User tab preview now omit provisional raw memories so only consolidated knowledge is injected by default. Structured stats added to core memories output ([`84a4001`](https://github.com/fpytloun/mnemory/commit/84a4001))
- **User tab stats and cleanup mode** — The User tab now shows approximate token usage plus compact core-memory stats, and advanced mode can load the actual backing memories as editable cards for cleanup ([`66beccb`](https://github.com/fpytloun/mnemory/commit/66beccb))
- **Fsck redesign** — Fsck redesigned as an automated maintenance engine with incremental processing, improved metadata normalization accuracy, and incremental re-checking ([`b04fa11`](https://github.com/fpytloun/mnemory/commit/b04fa11), [`deb5eeb`](https://github.com/fpytloun/mnemory/commit/deb5eeb))
- **Prompt caching** — Extraction prompts restructured for OpenAI prompt caching, placing stable system instructions before variable content ([`b4cdd56`](https://github.com/fpytloun/mnemory/commit/b4cdd56))

### Bug Fixes

- **Fact-vs-episodic misclassification** — Reduced misclassification of facts as episodic memories and vice versa through improved extraction prompt guidance ([`da270b9`](https://github.com/fpytloun/mnemory/commit/da270b9))
- **Cross-user session leakage in OpenWebUI** — Prevented session leakage in multi-user deployments where one user's memories could be visible to another ([`313f97e`](https://github.com/fpytloun/mnemory/commit/313f97e))
- **Auth flow hardening** — Hardened authentication flow and fsck incremental checks against edge cases ([`9a02bb7`](https://github.com/fpytloun/mnemory/commit/9a02bb7))
- **Fsck issue cards** — Affected memories are now included in fsck issue cards, and raw memories are excluded from checks ([`c60a8d0`](https://github.com/fpytloun/mnemory/commit/c60a8d0))

### Documentation

- **Default model update** — Updated default model reference to `gpt-5.4-mini` and documented `gpt-oss-120b` as a recommended budget alternative with comparable benchmark results ([`8df307d`](https://github.com/fpytloun/mnemory/commit/8df307d))
- **Infer=true guidance** — Clarified guidance for when to use `infer=true` vs `infer=false` in memory storage instructions ([`3f41e3d`](https://github.com/fpytloun/mnemory/commit/3f41e3d))

### Testing

- **Boundary tag injection test** — Softened boundary tag injection e2e test assertions for cautious models that may refuse injection-like content ([`0948b46`](https://github.com/fpytloun/mnemory/commit/0948b46))

## [1.10.0] — 2026-03-28

### Improved

- **Sessions tab browsing** -- The top-level Sessions list now supports server-side pagination, search over session summaries/IDs, last-activity vs created-date sorting, asc/desc ordering, page-size selection, and accurate total counts with truncation warnings for very large result sets.
- **Dashboard layer visibility** -- The Dashboard now shows raw vs consolidated memory totals and a dedicated memory-layer chart, including per-user filtered views.

### Bug Fixes

- **Management UI tab regression** -- Fixed broken HTML nesting in the Sessions tab that prevented the User, Graph, and Check tabs from rendering correctly.
- **Check tab agent filter** -- Fixed agent loading in the Check tab after `/api/stats` started returning `agents` as a flat string list.

## [1.9.1] — 2026-03-28

### Added

- **Root redirect** -- Navigating to `/` now redirects to `/ui/` for better discoverability of the management UI.
- **Cognis JWT authentication** -- ES256 JWT validation alongside existing API key auth for Cognis controller integration.
- **`POST /api/memories/by-ids`** -- New endpoint to fetch multiple memories by their IDs in a single request (max 500 IDs). Returns user-scoped results with silent skip for missing IDs.

### Improved

- **Sessions tab pagination** -- Linked memories and consolidated memories in expanded sessions are now paginated (10 per page) with "Show More / Show Less" controls, preventing large sessions from overwhelming the page.
- **Sessions tab performance** -- Memory loading now uses the batch endpoint instead of fetching all memories and filtering client-side, drastically reducing API payload and load time.
- **Compact session layout** -- Tighter spacing and consolidated/raw memory counts in expanded session details.

## [1.9.0] — 2026-03-26

### Two-Layer Memory System

- **Raw and consolidated memory layers**: Memories now have a `memory_layer` field — `raw` (provisional evidence from the `remember` endpoint) or `consolidated` (durable knowledge from `add_memory`, consolidation, or pre-existing memories). The `remember` endpoint writes `memory_layer=raw` and its dedup only searches against other raw memories, never modifying consolidated knowledge. This fixes the C1 dedup isolation problem where remember calls could overwrite manually stored memories ([`0638e93`](https://github.com/fpytloun/mnemory/commit/0638e93))
- **Session summary persistence**: Session summaries are now persisted to a dedicated `_mnemory_sessions` Qdrant collection. Summaries are updated on each `remember` call and survive server restarts. New REST endpoints for listing and querying session summaries ([`0638e93`](https://github.com/fpytloun/mnemory/commit/0638e93))
- **6-category extraction model**: Extraction prompts enhanced with a 6-category model (Topic, Exploration, Decision, Fact, Action, Preference/Workflow) to guide what to extract from conversations, improving extraction quality and classification accuracy ([`0638e93`](https://github.com/fpytloun/mnemory/commit/0638e93))
- **Layer-aware recall ranking**: Search results are reranked by memory layer — raw memories are penalized (-0.05) and superseded raw memories are further penalized (-0.15), ensuring consolidated knowledge surfaces first ([`944807d`](https://github.com/fpytloun/mnemory/commit/944807d))
- **`memory_layer` filter**: `list_memories` MCP tool and REST endpoint accept a `memory_layer` filter (raw/consolidated) for browsing by layer ([`944807d`](https://github.com/fpytloun/mnemory/commit/944807d))
- **Data migrations**: Automatic migrations for `_mnemory_sessions` collection creation and `memory_layer` payload index ([`0638e93`](https://github.com/fpytloun/mnemory/commit/0638e93))

### Within-Session Consolidation Service

- **ConsolidationService**: New background service that consolidates raw memories into durable knowledge after sessions go idle. State machine (idle → consolidating → consolidated/failed) with output validation and crash recovery. Wired into the existing `MaintenanceService` periodic loop ([`944807d`](https://github.com/fpytloun/mnemory/commit/944807d))
- **Role-separated consolidation**: User and assistant memories are consolidated independently with separate LLM calls and role-specific system prompts. User consolidation focuses on decisions, preferences, facts, and goals. Assistant consolidation focuses on implementations, deployments, research findings, and recommendations ([`edfec56`](https://github.com/fpytloun/mnemory/commit/edfec56))
- **Append-only re-consolidation**: When new raw memories arrive after consolidation, the session resets to idle and becomes eligible for re-consolidation. New consolidated memories are appended alongside existing ones (no data loss). Previous consolidated memories are passed as read-only context to prevent duplication ([`1c4da6a`](https://github.com/fpytloun/mnemory/commit/1c4da6a))
- **Time-based batching**: Large sessions are split into batches (`CONSOLIDATION_BATCH_SIZE`, default 100) with accumulated context from prior batches to prevent cross-batch duplication ([`1c4da6a`](https://github.com/fpytloun/mnemory/commit/1c4da6a))
- **Manual consolidation trigger**: `POST /api/sessions/{id}/consolidate` endpoint for on-demand consolidation with 409 race guard and failed-state reset. Fire-and-forget (returns 202 immediately, runs in background) ([`da514a4`](https://github.com/fpytloun/mnemory/commit/da514a4), [`b3cb51f`](https://github.com/fpytloun/mnemory/commit/b3cb51f))
- **Session delete**: `DELETE /api/sessions/{id}` endpoint with optional `delete_memories=true` to also delete linked raw memories ([`007e730`](https://github.com/fpytloun/mnemory/commit/007e730))
- **Dedicated consolidation model**: `CONSOLIDATION_LLM_MODEL` and `CONSOLIDATION_REASONING_EFFORT` config options (falls back to `FSCK_LLM_MODEL`, then `LLM_MODEL`) for using a stronger model for synthesis while keeping extraction lightweight ([`9a11e3b`](https://github.com/fpytloun/mnemory/commit/9a11e3b))
- **GC for superseded raw memories**: Fsck now garbage-collects superseded raw memories without artifacts ([`944807d`](https://github.com/fpytloun/mnemory/commit/944807d))
- **Consolidation metrics**: New Prometheus counters and gauges — runs, produced, superseded, duration, validation failures, pending sessions ([`944807d`](https://github.com/fpytloun/mnemory/commit/944807d))

### Security

- **`_trusted` bypass for internal callers**: New internal-only `_trusted` parameter on `add_memory` allows `infer=False` with `role='assistant'` for trusted server-side callers (consolidation, fsck). Never exposed in MCP tools or REST API ([`2e06181`](https://github.com/fpytloun/mnemory/commit/2e06181))
- **`ALLOW_CLIENT_INFER` config**: When set to `false`, client-facing tools always force `infer=True`, preventing untrusted clients from bypassing LLM extraction. Internal callers unaffected. Default: `true` ([`2e06181`](https://github.com/fpytloun/mnemory/commit/2e06181))

### Integrations

- **OpenClaw plugin** (`@fpytloun/openclaw-mnemory`): New standalone plugin for automatic memory recall/capture in OpenClaw. Ships as npm package with 15 tools, CLI commands, config schema, two-phase recall (init at session start + per-turn topical search), and comprehensive test suite ([`b2147ac`](https://github.com/fpytloun/mnemory/commit/b2147ac), [`c2ed398`](https://github.com/fpytloun/mnemory/commit/c2ed398))
- **OpenCode plugin rewrite** (`@fpytloun/opencode-mnemory`): Rewritten from single-file plugin (382 lines) to multi-file npm package (~2750 lines). Adds per-turn semantic search (old plugin only recalled at session start), 16 custom tools via plugin API (eliminates separate MCP server config), prompt injection escaping, session cleanup with AbortController, and unit tests ([`c6d9e20`](https://github.com/fpytloun/mnemory/commit/c6d9e20))
- **OpenWebUI filter improvements**: Fixed static context (instructions + core memories) lost on subsequent turns by caching from first turn. Fixed pending session handling, tool stripping for both `tool_ids` and `tools` fields. Added debug valve for diagnostic status messages. Bumped to v0.2.0 ([`e3a3f72`](https://github.com/fpytloun/mnemory/commit/e3a3f72), [`9d750ce`](https://github.com/fpytloun/mnemory/commit/9d750ce))

### Management UI

- **Sessions tab**: New tab for viewing persistent session summaries with memory cards, session detail modal, consolidation status, and "Consolidate Now" button for idle/failed sessions. Agent filter dropdown, delete with optional raw memory cleanup ([`b044e27`](https://github.com/fpytloun/mnemory/commit/b044e27), [`da514a4`](https://github.com/fpytloun/mnemory/commit/da514a4), [`007e730`](https://github.com/fpytloun/mnemory/commit/007e730))
- **Memory layer filter**: Browse and Search tabs now include a layer filter dropdown (All/Raw/Consolidated) with `raw`, `superseded`, and `consolidated` badges on memory cards ([`da514a4`](https://github.com/fpytloun/mnemory/commit/da514a4), [`b3cb51f`](https://github.com/fpytloun/mnemory/commit/b3cb51f))
- **Expanded memory details**: `superseded_by` and `derived_from` fields shown in expanded memory details ([`da514a4`](https://github.com/fpytloun/mnemory/commit/da514a4))

### Prompt & Extraction Improvements

- **Stronger summary prompts**: Per-turn summaries now explicitly capture assistant work (implementations, designs, research, deployments). Trivial exchanges produce minimal output instead of inflated "User asked..." statements ([`e5fccbe`](https://github.com/fpytloun/mnemory/commit/e5fccbe))
- **Broader agent extraction**: Agent extraction prompt expanded to also extract actions with lasting impact, conclusions/decisions, and recommendations — not just identity facts ([`c62e8a0`](https://github.com/fpytloun/mnemory/commit/c62e8a0))
- **Conservative consolidation**: Consolidation prompts require the "valuable in FUTURE conversations" test. Preferences require explicit statement or repeated pattern. Empty output allowed for sessions with no durable knowledge ([`853ac3f`](https://github.com/fpytloun/mnemory/commit/853ac3f))
- **Category enforcement**: Consolidated memories now reliably have categories via three-layer fix — prompt instruction, schema `minItems=1` constraint, and code fallback inheriting from source raw memories ([`99e6b0a`](https://github.com/fpytloun/mnemory/commit/99e6b0a), [`17f12b8`](https://github.com/fpytloun/mnemory/commit/17f12b8))
- **TTL and event_date in consolidation**: Consolidation output includes memory_type TTL guidance and event_date support for episodic memories ([`25091c6`](https://github.com/fpytloun/mnemory/commit/25091c6))

### Bug Fixes

- **Null text in dedup response**: Handle `{"text": null}` in LLM dedup decisions that caused `AttributeError` on `.strip()` ([`5fdaa33`](https://github.com/fpytloun/mnemory/commit/5fdaa33))
- **Deterministic session point IDs**: Session IDs are now converted to UUID5 for Qdrant compatibility (session ID strings are not valid Qdrant point IDs) ([`a681c22`](https://github.com/fpytloun/mnemory/commit/a681c22))
- **Corrupted session entries**: Stricter filtering requires both `session_id` and `summary` in payload, with deduplication by session_id ([`d1fe2fa`](https://github.com/fpytloun/mnemory/commit/d1fe2fa), [`8debe74`](https://github.com/fpytloun/mnemory/commit/8debe74))
- **Consolidation validation**: Lowered ratio threshold from 0.2 to 0.05 and made validation non-blocking (warns instead of failing) to prevent valid consolidation output from being rejected ([`61efb45`](https://github.com/fpytloun/mnemory/commit/61efb45))
- **VectorStore method**: Fixed `AttributeError` where consolidation code called non-existent `get()` method instead of `get_by_id()` ([`6e08d93`](https://github.com/fpytloun/mnemory/commit/6e08d93))

### Testing

- **Consolidation unit tests**: Comprehensive test suite for the consolidation service — role-separated prompts, state machine, re-consolidation, batch processing, validation, and raw memory fetching ([`944807d`](https://github.com/fpytloun/mnemory/commit/944807d), various fixes)
- **Two-layer e2e tests**: End-to-end tests for memory layer assignment, C1 dedup isolation, session summary persistence, and 6-category extraction model ([`dcf7b80`](https://github.com/fpytloun/mnemory/commit/dcf7b80))

### Documentation

- **Two-layer memory system docs**: Updated architecture, memory model, configuration, monitoring, REST API, MCP tools, and management UI documentation for the new two-layer system and consolidation service ([`2bbb041`](https://github.com/fpytloun/mnemory/commit/2bbb041))
- **OpenClaw client guide**: New `docs/clients/openclaw.md` setup guide ([`b2147ac`](https://github.com/fpytloun/mnemory/commit/b2147ac))

### CI

- **npm publishing workflow**: New GitHub Actions workflow for publishing OpenCode and OpenClaw plugins as npm packages on tag push (`opencode-v*`, `openclaw-v*`) ([`de73102`](https://github.com/fpytloun/mnemory/commit/de73102))
- **Selective Python publishing**: Python publish workflow now skips non-mnemory release tags ([`23d7c0a`](https://github.com/fpytloun/mnemory/commit/23d7c0a))

## [1.8.3] — 2026-03-23

### Bug Fixes

- **Reject unsupported categories in `add_memory`**: User-provided categories are now validated immediately against the predefined set before entering the LLM pipeline. Previously, invalid categories (e.g., "professional", "coding") were silently downgraded to an empty list via the `_validate_metadata()` fallback, making those memories invisible in the UI's category filters. Now returns a clear error listing valid categories, matching the existing behavior of `update_memory()`. LLM-extracted categories are also filtered against the predefined set during parsing, preventing hallucinated categories from entering the pipeline and reducing unnecessary LLM retry calls.

  **Note:** Memories previously stored with invalid categories may still have empty category lists. Use the built-in health checker (fsck) to detect and re-classify affected memories.

## [1.8.2] — 2026-03-20

### OpenWebUI Filter

- **Fixed prompt cache invalidation**: Recalled memories are now appended after the last user message instead of inserted before it. The previous insertion strategy shifted the cached prefix boundary on every turn, causing OpenAI prompt cache misses (~10% hit rate). The new append strategy preserves the entire conversation history as a stable prefix, restoring cache hit rates to ~70% — matching the behavior without the filter. Measured savings: ~60% reduction in input token cost per request ([`41dc285`](https://github.com/fpytloun/mnemory/commit/41dc285))
- **First-turn initialization**: Filter now correctly loads instructions and core memories on the first message of a conversation, using the `/api/recall` endpoint with `include_instructions=True` and `managed=True` ([`78c638b`](https://github.com/fpytloun/mnemory/commit/78c638b))

### Instructions

- **Composable instruction blocks with managed mode**: Refactored instruction system into composable blocks (`_INTRO_*`, `_RECALL_*`, `_STORAGE_*`, `_GUIDANCE_*`) that can be combined per mode. New `managed` flag for plugin/filter clients that handle recall and storage automatically — produces minimal instructions telling the LLM not to duplicate automatic operations. Supports `managed=True` with optional `instruction_mode` override for personality/proactive guidance within managed contexts ([`75dec50`](https://github.com/fpytloun/mnemory/commit/75dec50))

## [1.8.1] — 2026-03-16

### Bug Fixes

- **MCP tools no longer block the event loop**: All 19 MCP tool functions converted from synchronous to `async def` with `asyncio.to_thread()`. Previously, blocking LLM calls (28–40s) ran directly on the asyncio event loop, preventing `/health` from responding and causing Kubernetes liveness probe failures. The server now remains responsive during all MCP operations ([`80db248`](https://github.com/fpytloun/mnemory/commit/80db248))
- **TOCTOU race in `update_memory` and `delete_memory`**: Access control check (`verify_memory_access`) and the mutating operation are now executed atomically in a single thread call, eliminating a race window introduced by the async conversion ([`80db248`](https://github.com/fpytloun/mnemory/commit/80db248))

### Performance

- **Bounded thread pool for MCP tool execution**: A configurable `ThreadPoolExecutor` (env var `MCP_THREAD_POOL_SIZE`, default `min(32, cpu_count + 4)`) is set as the event loop's default executor. Multiple MCP tool calls can now execute concurrently instead of serializing on the event loop ([`80db248`](https://github.com/fpytloun/mnemory/commit/80db248))
- **Thread-safe service initialization**: `_get_service()` uses double-checked locking to prevent race conditions when multiple `asyncio.to_thread()` calls trigger lazy initialization concurrently ([`80db248`](https://github.com/fpytloun/mnemory/commit/80db248))

### Observability

- **Thread pool Prometheus metrics**: Two new gauges — `mnemory_thread_pool_max_workers` (configured pool size) and `mnemory_thread_pool_threads` (spawned thread count) — for monitoring thread pool utilization ([`80db248`](https://github.com/fpytloun/mnemory/commit/80db248))

## [1.8.0] — 2026-03-14

### Client-Provided Labels

- **Labels metadata on memories**: Memories now support optional key-value labels (e.g., `project`, `topic`, `conversation_id`) that bypass LLM extraction and are stored exactly as provided. Labels are inherited by all facts extracted during `infer=True`. Pass `labels` to `add_memory`, `add_memories`, `remember`, and `update_memory`. Filter by labels in `search_memories`, `find_memories`, `ask_memories`, and `list_memories` with AND logic (all must match). List values use any-of matching within a single key ([`18d18b1`](https://github.com/fpytloun/mnemory/commit/18d18b1), [`fa7e566`](https://github.com/fpytloun/mnemory/commit/fa7e566))
- **Labels in REST API**: All REST endpoints (add, search, find, ask, list, update, remember, recall) accept and return labels. Pydantic schemas updated ([`18d18b1`](https://github.com/fpytloun/mnemory/commit/18d18b1), [`fa7e566`](https://github.com/fpytloun/mnemory/commit/fa7e566))
- **Labels in UI**: Management UI supports labels input (JSON), display as badges, client-side filtering, and edit modal support ([`18d18b1`](https://github.com/fpytloun/mnemory/commit/18d18b1))
- **Labels validation**: Keys must be alphanumeric + underscore, starting with letter or underscore. Values can be str, int, float, bool, or list[str]. Max 20 labels per memory (configurable). Reserved keys (memory_type, user_id, etc.) are rejected ([`fa7e566`](https://github.com/fpytloun/mnemory/commit/fa7e566))
- **Labels config**: New env vars `LABELS_MAX_FIELDS` (default 20), `LABELS_MAX_KEY_LENGTH` (default 64), `LABELS_MAX_VALUE_LENGTH` (default 256), `LABELS_INDEXES` (optional Qdrant payload indexes for label keys) ([`fa7e566`](https://github.com/fpytloun/mnemory/commit/fa7e566))

### Prometheus Timing Instrumentation

- **LLM, embedding, and Qdrant latency histograms**: 3 new Prometheus histograms (`mnemory_llm_duration_seconds`, `mnemory_embedding_duration_seconds`, `mnemory_qdrant_duration_seconds`) with operation labels. All 13 LLM call sites, embedding operations, sparse embedding, and all Qdrant vector operations are instrumented ([`937d112`](https://github.com/fpytloun/mnemory/commit/937d112))
- **find_memories pipeline timing**: Per-step timing breakdown (query_gen, embed, search, rerank) logged at INFO level ([`937d112`](https://github.com/fpytloun/mnemory/commit/937d112))
- **Grafana dashboard**: 6 new latency panels (LLM p95 by operation, call rate, embedding p95, Qdrant p95, LLM p50 vs p95 by model) ([`937d112`](https://github.com/fpytloun/mnemory/commit/937d112))

### Token Usage Optimization

- **Compact MCP tool docstrings**: Reduced tool description overhead from ~5,500 to ~1,500 tokens per request by moving detailed parameter guidance from docstrings to server instructions. Estimated savings: ~4,000 prompt tokens per LLM request ([`79818c6`](https://github.com/fpytloun/mnemory/commit/79818c6))
- **`include_instructions` flag on `initialize_memory`**: Set to `false` when the client already has instructions via MCP instructions field, avoiding duplicate instructions in context ([`79818c6`](https://github.com/fpytloun/mnemory/commit/79818c6))
- **OpenWebUI filter: strip redundant MCP tools**: Filter now strips `initialize_memory`, `get_core_memories`, `get_recent_memories` from tool_ids (saving ~800 tokens) and injects memory context before the last user message for better prompt caching ([`41f7e8f`](https://github.com/fpytloun/mnemory/commit/41f7e8f))

### Bug Fixes

- **Sub-agent memories now visible in MCP search tools**: The `search_memories`, `find_memories`, `ask_memories`, and `list_memories` MCP tools silently ignored the `agent_id` tool parameter when `X-Agent-Id` header was set, making sub-agent memories (e.g., `openwebui:miroslav`) invisible in search results. Now properly resolves the `agent_id` parameter, honoring sub-agent scope while blocking cross-agent access ([`3a7de1f`](https://github.com/fpytloun/mnemory/commit/3a7de1f))

### Performance

- **Eliminated redundant embedding API call in dual-scope search**: `search_memories_dual_scope()` now pre-computes the dense embedding once and reuses it across both sub-searches, halving embedding latency for dual-scope searches ([`3a7de1f`](https://github.com/fpytloun/mnemory/commit/3a7de1f))

### Observability

- **Granular search mode tracking**: Vector store tracks `hybrid`, `dense-formula`, and `dense-simple` modes, logged at INFO level with duration and result count ([`3a7de1f`](https://github.com/fpytloun/mnemory/commit/3a7de1f))
- **Search summary logging**: Both search methods log mode, threshold, result count, and filtered count at INFO level ([`3a7de1f`](https://github.com/fpytloun/mnemory/commit/3a7de1f))
- **Warnings for dense-only fallback**: WARNING-level logs when sparse vector is unavailable or hybrid search fails ([`3a7de1f`](https://github.com/fpytloun/mnemory/commit/3a7de1f))

### Documentation

- **Corrected RRF score range**: Qdrant's default RRF k=1 produces scores ~0.1-1.0, not ~0.01-0.03 as previously documented for k=60 ([`3a7de1f`](https://github.com/fpytloun/mnemory/commit/3a7de1f))

## [1.7.0] — 2026-03-05

### Secure Artifact Download Tokens

- **HMAC-signed download tokens**: Replaced long-lived API keys in URLs with short-lived, stateless HMAC-signed tokens for artifact raw access. Tokens are scoped to a specific artifact, expire after a configurable TTL (default 1 hour, max 24 hours), and are verified without server-side storage. Browser-embedded `<img src="...?token=...">` and download links now use these tokens instead of API keys ([`29b2223`](https://github.com/fpytloun/mnemory/commit/29b2223))
- **New MCP tool `get_artifact_url`**: Generates a signed download URL for direct browser access to artifacts. Returns URL, expiry, content type, filename, and size. Use when artifacts are binary or larger than 1 MB ([`29b2223`](https://github.com/fpytloun/mnemory/commit/29b2223))
- **New REST endpoint `POST /api/memories/{id}/artifacts/{aid}/download-token`**: Authenticated endpoint that generates a download token with optional custom TTL. Returns the token and a ready-to-use URL ([`29b2223`](https://github.com/fpytloun/mnemory/commit/29b2223))
- **1 MB binary inline cap on `get_artifact`**: Binary artifacts larger than 1 MB are no longer returned inline via the MCP `get_artifact` tool. Instead, a guidance message directs the client to use `get_artifact_url`. Text artifacts are unaffected ([`29b2223`](https://github.com/fpytloun/mnemory/commit/29b2223))
- **Removed `key` query parameter**: The `key` query parameter for passing API keys in URLs has been removed from the auth middleware. All browser-embedded artifact access must use download tokens ([`29b2223`](https://github.com/fpytloun/mnemory/commit/29b2223))
- **New config vars**: `SERVER_BASE_URL` (for full URL generation), `DOWNLOAD_TOKEN_TTL` (default 3600s), `DOWNLOAD_TOKEN_MAX_TTL` (default 86400s) ([`29b2223`](https://github.com/fpytloun/mnemory/commit/29b2223))

### Binary Artifact Support

- **Fixed binary artifact retrieval**: Binary artifacts (images, PDFs, etc.) now return full content as a single base64 blob instead of being truncated by text pagination. Added `load_raw()` method for direct byte access and `GET /api/memories/{id}/artifacts/{aid}/raw` endpoint for browser rendering ([`ff89fc7`](https://github.com/fpytloun/mnemory/commit/ff89fc7))
- **Increased max artifact size to 10 MB**: `MAX_ARTIFACT_SIZE` default raised from 100 KB to 10 MB (10485760 bytes) ([`ff89fc7`](https://github.com/fpytloun/mnemory/commit/ff89fc7))
- **UI binary rendering**: Management UI now detects binary artifacts — images render inline via `<img>`, other binary types show download links. Text artifacts retain paginated display ([`ff89fc7`](https://github.com/fpytloun/mnemory/commit/ff89fc7))

### New MCP Tool: ask_memories

- **`ask_memories` tool**: Ask a natural language question and get a human-readable prose answer synthesized from stored memories. Uses `find_memories` internally for retrieval, then passes results to an LLM for answer generation. Set `include_memories=true` to also receive the supporting memories. Most expensive operation (3 LLM calls) — use when you need a synthesized answer rather than raw results ([`db2fd19`](https://github.com/fpytloun/mnemory/commit/db2fd19))

### Performance

- **Parallelized find/ask search queries**: `find_memories` and `ask_memories` now run generated search queries concurrently using `ThreadPoolExecutor` instead of sequentially, reducing latency by ~400–1200 ms depending on query count ([`407926d`](https://github.com/fpytloun/mnemory/commit/407926d))
- **Configurable find/ask LLM model**: New `FIND_LLM_MODEL` env var allows using a faster/cheaper model for query generation and reranking in the find/ask pipeline ([`407926d`](https://github.com/fpytloun/mnemory/commit/407926d))

### Bug Fixes

- **Core memories no longer hard-truncated**: Replaced hard character truncation with per-section entry limits (`CORE_MAX_PER_SECTION`). Main sections are never truncated — only the recent context section is gracefully trimmed per-entry. Improved agent memory guidance in core memories output ([`856a564`](https://github.com/fpytloun/mnemory/commit/856a564))
- **Code analysis no longer misclassified as permanent facts**: Extraction prompts updated to prevent transient technical observations and code analysis conclusions from being stored as permanent `fact` type memories ([`1c51431`](https://github.com/fpytloun/mnemory/commit/1c51431))
- **Session-specific assistant memories filtered from extraction**: Extraction prompts now filter out session-specific assistant memories (e.g., "I concluded X in this session") that should not persist across conversations ([`7ec1b3d`](https://github.com/fpytloun/mnemory/commit/7ec1b3d))

## [1.6.1] — 2026-03-04

### Bug Fixes

- **Episodic memories now always have `event_date`**: Episodic memories (decisions, intents, goals, interactions) created via the remember endpoint or add_memory were stored without `event_date` because extraction prompts instructed the LLM to set null for facts without explicit temporal anchors. Updated all 5 extraction prompt sections and ~18 examples to clarify that episodic events should default to today's date. Added a code-level fallback in `_execute_action()` that auto-sets `event_date` to today (UTC) for episodic memories when neither the LLM nor the caller provides one ([`f2d038f`](https://github.com/fpytloun/mnemory/commit/f2d038f))

## [1.6.0] — 2026-03-04

### Hybrid Search

- **BM25 sparse vectors + RRF fusion**: Search now uses Qdrant's native hybrid search — dense vectors (semantic similarity) and BM25 sparse vectors (keyword matching) fused with Reciprocal Rank Fusion (RRF). Replaces the old Python-side keyword boost post-processing. Results are more accurate for queries containing specific terms, names, or identifiers ([`276521a`](https://github.com/fpytloun/mnemory/commit/276521a))
- **fastembed is now a required dependency**: `fastembed>=0.4.0` moved from optional to main dependencies. The BM25 model (~50MB) is pre-downloaded in the Docker image at build time ([`276521a`](https://github.com/fpytloun/mnemory/commit/276521a))
- **Importance reranking**: Post-RRF importance boost applied in Python (weighted by `IMPORTANCE_WEIGHTS`), replacing the previous `FormulaQuery` approach which is unreliable inside Qdrant Prefetch ([qdrant/qdrant#6836](https://github.com/qdrant/qdrant/issues/6836)) ([`276521a`](https://github.com/fpytloun/mnemory/commit/276521a))
- **New config vars**: `SEARCH_SPARSE_MODEL` (default `Qdrant/bm25`), `SEARCH_SCORE_THRESHOLD_HYBRID` (default `0.0` — RRF scores are ~0.016–0.033, not 0–1 like cosine) ([`276521a`](https://github.com/fpytloun/mnemory/commit/276521a))
- **Deprecated**: `SEARCH_KEYWORD_WEIGHT` is no longer used (keyword matching is now handled by BM25 sparse vectors natively in Qdrant)
- **Dense-only fallback**: If hybrid search fails at query time (e.g., Qdrant version mismatch), the system gracefully falls back to dense-only search ([`3199a47`](https://github.com/fpytloun/mnemory/commit/3199a47))
- **Python 3.14 not supported**: fastembed's BM25 model depends on `py_rust_stemmers`, a Rust extension that segfaults on Python 3.14 due to a PyO3/C API incompatibility ([qdrant/fastembed#576](https://github.com/qdrant/fastembed/issues/576)). Use Python 3.11–3.13 (3.13 recommended). The Docker image uses `python:3.13-slim` ([`9f2b920`](https://github.com/fpytloun/mnemory/commit/9f2b920))

### Data Migrations

- **Automatic migration framework**: New `MigrationRunner` in `mnemory/migration.py` — idempotent, resumable with batch checkpointing, state tracked in `_mnemory_meta` Qdrant collection. Health endpoint returns 503 during migration. Existing instances get sparse vectors added via `001_add_sparse_vectors` on first startup. **Note: migration may take several minutes for large instances** (processes all existing memories in batches of 100) ([`276521a`](https://github.com/fpytloun/mnemory/commit/276521a))
- **Collection recreation fallback**: Migration handles Qdrant rejecting sparse vector addition by recreating the collection with correct config ([`4b4e7a7`](https://github.com/fpytloun/mnemory/commit/4b4e7a7))

### Session Persistence

- **Persistent session storage**: Sessions now survive server restarts via pluggable backends — SQLite (default for local/single-node), Redis (for clustered deployments), or in-memory (tests). Uses a write-through cache pattern: in-memory dict for fast reads, backend for durability, lazy loading on cache miss ([`eb3eddf`](https://github.com/fpytloun/mnemory/commit/eb3eddf))
- **24-hour session TTL**: Default `MEMORY_SESSION_TTL` increased from 1 hour to 24 hours. Both recall and remember calls reset the idle timer (auto-prolong), so active conversations never expire ([`eb3eddf`](https://github.com/fpytloun/mnemory/commit/eb3eddf))
- **Remember auto-prolong**: The remember endpoint now touches the session to reset its idle timeout, preventing session expiry during active conversations that only use remember (not recall) ([`eb3eddf`](https://github.com/fpytloun/mnemory/commit/eb3eddf))
- **Redis optional dependency**: Added `redis>=5.0.0` as optional dependency, installable via `pip install mnemory[redis]` ([`eb3eddf`](https://github.com/fpytloun/mnemory/commit/eb3eddf))
- **New env vars**: `SESSION_BACKEND` (sqlite/redis/memory), `SESSION_PATH` (SQLite DB path), `REDIS_URL` (Redis connection). Auto-detects Redis when `REDIS_URL` is set ([`eb3eddf`](https://github.com/fpytloun/mnemory/commit/eb3eddf))

### Remember Pipeline

- **Two-stage extract+dedup pipeline**: Remember pipeline rewritten to separate fact extraction from deduplication, with session context tracking for conversation continuity across multiple remember calls ([`b089a48`](https://github.com/fpytloun/mnemory/commit/b089a48))
- **Three-way role semantics with auto mode**: Remember pipeline now supports `role=None` (auto mode) that extracts facts from all conversation participants, attributing each to the correct role. Assistant facts require `agent_id` ([`521b8ba`](https://github.com/fpytloun/mnemory/commit/521b8ba))

### Role Handling

- **Role extraction fix for `role=assistant`**: Agent memories with `infer=True` now correctly use the agent extraction prompt ([`ff25699`](https://github.com/fpytloun/mnemory/commit/ff25699))
- **Role metadata written on UPDATE actions**: Dedup UPDATE actions now correctly persist the role field in metadata ([`942568a`](https://github.com/fpytloun/mnemory/commit/942568a))

### Core Memories

- **Dedup core memories against search results**: `get_core_memories` results are deduplicated against search results, and recent memories are filtered by importance ([`969b786`](https://github.com/fpytloun/mnemory/commit/969b786))

### Fsck Improvements

- **Detect and fix wrong role metadata**: Quality phase now detects memories with incorrect `role` field and proposes fixes ([`23bc415`](https://github.com/fpytloun/mnemory/commit/23bc415))

### Prompt & Extraction Improvements

- **Softer dedup language**: Reduced over-suppression of new memories by softening dedup prompt guidance ([`ea3407e`](https://github.com/fpytloun/mnemory/commit/ea3407e))
- **Restored inline examples**: Re-added extraction examples and rules that were dropped during the remember pipeline rewrite ([`bbf5f8d`](https://github.com/fpytloun/mnemory/commit/bbf5f8d))
- **Improved extraction reliability**: Better remember extraction reliability and classification ([`b378182`](https://github.com/fpytloun/mnemory/commit/b378182))

## [1.5.0] — 2026-02-28

### Batch Operations

- **Batch delete API and MCP tool**: New `delete_memories` MCP tool and `DELETE /api/memories/batch` endpoint for deleting multiple memories in a single call ([`94698a9`](https://github.com/fpytloun/mnemory/commit/94698a9))
- **Bulk selection in UI**: Browse tab now supports select-all, shift-click range selection, and batch delete for managing large memory sets ([`c9637c9`](https://github.com/fpytloun/mnemory/commit/c9637c9))

### Memory Intelligence

- **Context hint in remember pipeline**: Plugins can pass a context hint (e.g., working directory, current topic) to the remember endpoint, improving extraction relevance for project-scoped work ([`7a00f72`](https://github.com/fpytloun/mnemory/commit/7a00f72))
- **Dual-scope deduplication**: Agent-scoped memories are now deduplicated against both agent-specific and shared user memories, preventing cross-scope duplicates ([`9220a10`](https://github.com/fpytloun/mnemory/commit/9220a10))
- **Improved memory_type classification**: Better prompt guidance for classifying goals, intents, and decisions — goals/intents are now correctly classified as `episodic` rather than `fact`, with expanded e2e test coverage ([`d6c3b8d`](https://github.com/fpytloun/mnemory/commit/d6c3b8d))
- **Clarified extraction guidance**: Fact-extraction prompts refined to better distinguish stable facts from transient observations ([`be30809`](https://github.com/fpytloun/mnemory/commit/be30809))

### Fsck / Health Checks

- **Artifact-aware fsck**: Split actions are skipped and delete actions trigger warnings for memories with attached artifacts, preventing accidental data loss ([`4562a36`](https://github.com/fpytloun/mnemory/commit/4562a36))
- **Auto-fsck UI**: Check tab now shows scheduling status banner with interval, next-run countdown, run-now button, and last auto-fsck results display ([`590374b`](https://github.com/fpytloun/mnemory/commit/590374b))
- **Auto-fsck race condition fix**: Resolved issue where UI showed auto-applied issues as unapplied due to a timing race between apply and UI refresh ([`2e6e488`](https://github.com/fpytloun/mnemory/commit/2e6e488))

### Management UI

- **Refresh button**: Manual refresh button added to the dashboard header ([`12d3dde`](https://github.com/fpytloun/mnemory/commit/12d3dde))

### Integrations

- **MNEMORY_INCLUDE_ASSISTANT option**: Claude Code and OpenCode plugins can now include assistant messages in remember flows, enabling the server to extract agent identity and capability facts ([`1df5e63`](https://github.com/fpytloun/mnemory/commit/1df5e63))
- **OpenCode child session handling**: OpenCode plugin skips memory recall/remember for child/subtask sessions to avoid noise from automated sub-agent work ([`6439dd4`](https://github.com/fpytloun/mnemory/commit/6439dd4))

### Documentation

- **Docs reorganization**: README split into dedicated documentation pages under `docs/` — architecture, configuration, deployment, development, management UI, MCP tools, memory model, monitoring, and REST API ([`9475191`](https://github.com/fpytloun/mnemory/commit/9475191))

## [1.4.0] — 2026-02-27

### Periodic Maintenance (Auto-fsck)

- **Background maintenance service**: New `MaintenanceService` (`mnemory/maintenance.py`) runs automatic memory consistency checks on a configurable schedule. Sleep-first loop (waits the full interval before the first run), per-user error isolation, and `asyncio.to_thread()` for sync LLM work ([`d7f5139`](https://github.com/fpytloun/mnemory/commit/d7f5139))
- **Auto-apply with thresholds**: Fixes that meet configurable confidence (`FSCK_AUTO_MIN_CONFIDENCE`, default 0.95) and severity (`FSCK_AUTO_MIN_SEVERITY`, default `medium`) thresholds are applied automatically. Issues below either threshold are skipped ([`d7f5139`](https://github.com/fpytloun/mnemory/commit/d7f5139))
- **New config vars**: `FSCK_AUTO_INTERVAL` (hours, default 0 = disabled), `FSCK_AUTO_MIN_CONFIDENCE` (0.0–1.0), `FSCK_AUTO_MIN_SEVERITY` (low/medium/high) ([`d7f5139`](https://github.com/fpytloun/mnemory/commit/d7f5139))
- **User enumeration**: New `VectorStore.list_user_ids()` scrolls all Qdrant points to enumerate distinct users for the maintenance loop ([`d7f5139`](https://github.com/fpytloun/mnemory/commit/d7f5139))

### Metrics

- **Auto-fsck Prometheus metrics**: Four new counters (`mnemory_autofsck_runs_total`, `mnemory_autofsck_issues_found_total`, `mnemory_autofsck_fixes_applied_total`, `mnemory_autofsck_fixes_failed_total`) and a gauge (`mnemory_autofsck_last_run_timestamp`), all labeled by `user_id` ([`d7f5139`](https://github.com/fpytloun/mnemory/commit/d7f5139))
- **Stats API extension**: `GET /api/stats` now includes an `autofsck` section with enabled flag, config, per-user run history, and aggregated totals ([`d7f5139`](https://github.com/fpytloun/mnemory/commit/d7f5139))

### Management UI

- **Auto-fsck status banner**: Check tab shows a teal status banner when auto-fsck is enabled, displaying interval, confidence, severity, last run age, and fixes applied ([`d7f5139`](https://github.com/fpytloun/mnemory/commit/d7f5139))
- **Last Auto-Check stat card**: Dashboard shows a 7th stat card with the age of the last auto-fsck run (conditional on auto-fsck being enabled) ([`d7f5139`](https://github.com/fpytloun/mnemory/commit/d7f5139))

### Grafana Dashboard

- **Auto-fsck row**: New "Auto-fsck" row with four panels — Last Auto-Check age (color thresholds: green <1h, yellow <24h, red >24h), Issues Found, Fixes Applied, and an Activity timeseries showing fixes applied and issues found per hour ([`d7f5139`](https://github.com/fpytloun/mnemory/commit/d7f5139))

### Fsck Improvements

- **Apply idempotency**: `apply_check` tracks applied issue IDs and skips re-applying on repeated calls ([`d7a9b8a`](https://github.com/fpytloun/mnemory/commit/d7a9b8a))
- **Background cleanup task**: `FsckStore` now has `start_cleanup_task()` / `stop_cleanup_task()` for periodic sweep of expired checks, wired into server lifespan ([`d7a9b8a`](https://github.com/fpytloun/mnemory/commit/d7a9b8a))
- **Apply validations**: Category and memory type values in update actions are validated and stripped if invalid, preventing bad metadata from being written ([`d7a9b8a`](https://github.com/fpytloun/mnemory/commit/d7a9b8a))
- **Ownership checks**: Apply actions verify memory ownership before mutating, skipping memories belonging to other users ([`d7a9b8a`](https://github.com/fpytloun/mnemory/commit/d7a9b8a))
- **Scroll pagination**: `scroll_with_vectors` now paginates correctly for large collections ([`d7a9b8a`](https://github.com/fpytloun/mnemory/commit/d7a9b8a))
- **FSCK_LLM_MODEL / FSCK_REASONING_EFFORT**: Override the LLM model and reasoning effort specifically for fsck checks, independent of the main `LLM_MODEL` ([`df4be81`](https://github.com/fpytloun/mnemory/commit/df4be81))
- **LLM security re-evaluation**: Regex-flagged memories are re-evaluated by the LLM before being reported as security issues, reducing false positives ([`75febcf`](https://github.com/fpytloun/mnemory/commit/75febcf))
- **Confidence field**: All fsck issues now carry a `confidence` score (0.0–1.0) from the LLM, surfaced in the UI and used for auto-apply threshold filtering ([`75febcf`](https://github.com/fpytloun/mnemory/commit/75febcf))

### Core Memories

- **Top-N non-pinned memories**: `get_core_memories` now includes the top-N most important non-pinned memories per section, configurable via `CORE_TOP_MEMORIES` (default 10) and `CORE_MIN_IMPORTANCE` (default `normal`) ([`40c085e`](https://github.com/fpytloun/mnemory/commit/40c085e))

### Documentation

- **Check UI screenshots**: Three new screenshots for the Check tab (progress, results, detail) added to README ([`4b87ca9`](https://github.com/fpytloun/mnemory/commit/4b87ca9))
- **Periodic Maintenance section**: README documents the auto-fsck feature with configuration example and how-it-works description ([`d7f5139`](https://github.com/fpytloun/mnemory/commit/d7f5139))

## [1.3.0] — 2026-02-25

### Management UI

- **Built-in web UI**: Full management UI at `/ui` — dashboard, search, browse/CRUD, and graph visualization. Alpine.js + Tailwind CSS + Chart.js + D3.js, all vendored as static files with zero external dependencies ([`14d1546`](https://github.com/fpytloun/mnemory/commit/14d1546))
- **Dashboard**: Memory totals, breakdowns by type/category/role (Chart.js donut charts), operation counts, per-user filtering, server version display, auto-refresh with manual refresh button ([`66dbf1e`](https://github.com/fpytloun/mnemory/commit/66dbf1e))
- **Search tab**: Semantic search (`search_memories`) and AI-powered find (`find_memories`) with filters — memory type, role, agent ID, categories multi-select, "has artifacts" toggle, include decayed, result limit. Sort by score, newest, or oldest ([`66dbf1e`](https://github.com/fpytloun/mnemory/commit/66dbf1e), [`33d2a7f`](https://github.com/fpytloun/mnemory/commit/33d2a7f), [`7cb089f`](https://github.com/fpytloun/mnemory/commit/7cb089f))
- **Browse tab**: Full CRUD with server-side sorting (newest/oldest via Qdrant `order_by`), Add Memory modal, inline edit modal (content, type, categories, importance, pinned, TTL), delete with confirmation, and artifact management (upload, view with pagination, delete) ([`bbeb69c`](https://github.com/fpytloun/mnemory/commit/bbeb69c), [`33d2a7f`](https://github.com/fpytloun/mnemory/commit/33d2a7f))
- **Graph tab**: D3.js force-directed visualization of memory relationships (shared categories = edges, node size = importance, color = memory type). Type filter checkboxes, node limit slider, click-to-select detail panel ([`14d1546`](https://github.com/fpytloun/mnemory/commit/14d1546))
- **Card layout**: Two-zone badge row — left: type + importance + `#category` tags; right: agent badge + artifacts indicator + assistant chip + pinned star ([`7cb089f`](https://github.com/fpytloun/mnemory/commit/7cb089f))
- **Agent filtering**: Agent ID dropdown filter in both Search and Browse tabs, with agent badges (robot icon + name) visible on cards without expanding ([`7cb089f`](https://github.com/fpytloun/mnemory/commit/7cb089f))
- **Session persistence**: API key and selected user stored in localStorage (persists across tab closes) ([`8171f02`](https://github.com/fpytloun/mnemory/commit/8171f02))

### Backend Improvements

- **Server-side sorting**: `GET /api/memories` accepts `sort` query param (newest/oldest). Uses Qdrant `order_by` with automatic datetime payload index creation on startup ([`33d2a7f`](https://github.com/fpytloun/mnemory/commit/33d2a7f), [`ff83be4`](https://github.com/fpytloun/mnemory/commit/ff83be4))
- **Index management**: `_ensure_indexes()` extracted as separate method, runs on every startup (not just collection creation). Existing collections now get new indexes automatically ([`ff83be4`](https://github.com/fpytloun/mnemory/commit/ff83be4))
- **Ordering fallback**: `get_all()` gracefully falls back to Python-side sorting if Qdrant `order_by` fails (e.g., missing index on older collections) ([`ff83be4`](https://github.com/fpytloun/mnemory/commit/ff83be4))
- **agent_id metadata fix**: `format_memory_item()` now injects top-level `agent_id` back into metadata dict, fixing agent badges not appearing in API responses ([`7cb089f`](https://github.com/fpytloun/mnemory/commit/7cb089f))
- **OpenAPI sanitization**: Schema cleanup for OpenAI function-calling compatibility ([`7ce33a9`](https://github.com/fpytloun/mnemory/commit/7ce33a9))
- **Non-conversational message filtering**: `remember()` ignores tool calls, system messages, and other non-conversational content ([`9f08f38`](https://github.com/fpytloun/mnemory/commit/9f08f38))
- **Runtime version loading**: Package version loaded via `importlib.metadata` at runtime with dev fallback ([`2640aad`](https://github.com/fpytloun/mnemory/commit/2640aad))

### Extraction & Search

- **English extraction**: Extraction prompts now require English output regardless of input language ([`a567984`](https://github.com/fpytloun/mnemory/commit/a567984))
- **Third-party guidance**: Extraction prompts include guidance for handling third-party information mentioned in conversations ([`a567984`](https://github.com/fpytloun/mnemory/commit/a567984))

### Testing

- **End-to-end test suite**: Comprehensive e2e tests exercising the full pipeline (extraction → storage → search) with real LLM calls and embedded Qdrant. Excluded from default `pytest` run via `-m 'not e2e'` ([`4b7353e`](https://github.com/fpytloun/mnemory/commit/4b7353e))

### Documentation

- **Client setup guides**: Dedicated docs for Claude Code, ChatGPT, Claude Desktop, Open WebUI, OpenCode, Cursor, Windsurf, Cline, Continue.dev, and Codex CLI ([`e283c48`](https://github.com/fpytloun/mnemory/commit/e283c48))
- **UI screenshots**: Dashboard, search, browse, and graph screenshots added to README ([`47defec`](https://github.com/fpytloun/mnemory/commit/47defec))

## [1.2.0] — 2026-02-24

### Monitoring

- **Prometheus metrics endpoint**: New `/metrics` endpoint with operation counters (`mnemory_operations_total`) and Qdrant-backed gauges for memory counts by type, category, role, decay, and pinned status. Configurable cache TTL and optional `prometheus_client` dependency ([`3951649`](https://github.com/fpytloun/mnemory/commit/3951649))
- **Management port**: New `MGMT_PORT` config to serve `/health` and `/metrics` on a separate port without authentication — designed for Kubernetes probes and Prometheus scraping ([`3951649`](https://github.com/fpytloun/mnemory/commit/3951649))
- **Grafana dashboard**: Pre-built dashboard with memory overview, operation rates, and breakdowns by type/category/role with user and agent filtering ([`f48c909`](https://github.com/fpytloun/mnemory/commit/f48c909))

### Security

- **Prompt injection safeguards**: All 6 LLM prompt builders hardened with Unicode boundary tags (`⟨user_input⟩`, `⟨existing_memories⟩`) and anti-injection preamble. Content within tags is escaped to prevent breakout ([`95f61bb`](https://github.com/fpytloun/mnemory/commit/95f61bb))
- **Injection detection logging**: Common prompt injection patterns (role overrides, instruction manipulation, boundary tag abuse) are detected and logged at WARNING level without blocking storage ([`95f61bb`](https://github.com/fpytloun/mnemory/commit/95f61bb))
- **Category name validation**: `project:<name>` categories validated against a safe character regex to prevent prompt format injection ([`95f61bb`](https://github.com/fpytloun/mnemory/commit/95f61bb))
- **Role validation**: REST `remember` endpoint and Pydantic schemas enforce an allowlist of valid message roles ([`95f61bb`](https://github.com/fpytloun/mnemory/commit/95f61bb))
- **Per-memory boundary tags**: Core memories output wraps each memory in `⟨memory_item⟩` tags to prevent cross-memory poisoning and section header forgery ([`5cb8daf`](https://github.com/fpytloun/mnemory/commit/5cb8daf))
- **Agent identity gating**: `role="assistant"` memories now require `infer=True` — direct injection of arbitrary agent personality via `infer=False` is blocked with `ValueError` ([`5cb8daf`](https://github.com/fpytloun/mnemory/commit/5cb8daf))
- **Input length validation**: `infer=False` path now has the same 400K character cap as `infer=True`. Pydantic `max_length` constraints added to 10 request fields across all schemas ([`5cb8daf`](https://github.com/fpytloun/mnemory/commit/5cb8daf))
- **Output escaping**: Markdown headers in memory text are escaped in `format_memory_item()` and `_format_memories()` to prevent section forgery in search results and core memories ([`5cb8daf`](https://github.com/fpytloun/mnemory/commit/5cb8daf))
- **Error sanitization**: `initialize_memory` error messages truncated to 200 chars with newlines stripped to prevent information leakage ([`5cb8daf`](https://github.com/fpytloun/mnemory/commit/5cb8daf))

## [1.1.0] — 2026-02-24

### REST API & Intelligence Layer

- **REST API foundation**: Full memory CRUD + artifact operations as FastAPI sub-app mounted at `/api/`, with auto-generated OpenAPI spec at `/api/docs` ([`ceb07f1`](https://github.com/fpytloun/mnemory/commit/ceb07f1))
- **Recall endpoint** (`POST /api/recall`): Combined initialize + search in one call with server-side session tracking to avoid context bloat. First call returns instructions + core memories + AI-powered search; subsequent calls return only new memories ([`1747af6`](https://github.com/fpytloun/mnemory/commit/1747af6))
- **Remember endpoint** (`POST /api/remember`): Fire-and-forget memory storage from conversation turns with rate limiting and session deduplication ([`1747af6`](https://github.com/fpytloun/mnemory/commit/1747af6))
- **Configurable search modes**: Per-request `search_mode` (find vs search) and `score_threshold` on recall endpoint ([`2c240de`](https://github.com/fpytloun/mnemory/commit/2c240de), [`1f1e259`](https://github.com/fpytloun/mnemory/commit/1f1e259), [`13b3dde`](https://github.com/fpytloun/mnemory/commit/13b3dde))
- **Cross-agent access control**: Dual-scope search and agent memory isolation in REST endpoints ([`7997e29`](https://github.com/fpytloun/mnemory/commit/7997e29))
- **Input validation and session bounds**: Request validation, response models, and `_MAX_KNOWN_IDS` cap on session tracking ([`bfd90a2`](https://github.com/fpytloun/mnemory/commit/bfd90a2))

### Plugins

- **Open WebUI filter**: Automatic recall on inlet, remember on outlet, with configurable status messages and `show_status` user valve ([`75bb955`](https://github.com/fpytloun/mnemory/commit/75bb955), [`6a43998`](https://github.com/fpytloun/mnemory/commit/6a43998), [`444b2d8`](https://github.com/fpytloun/mnemory/commit/444b2d8))
- **OpenCode plugin**: TypeScript plugin with system prompt injection via `experimental.chat.system.transform`, pre-fetched recall with async caching, fire-and-forget remember, and compaction persistence ([`f658c48`](https://github.com/fpytloun/mnemory/commit/f658c48), [`1592071`](https://github.com/fpytloun/mnemory/commit/1592071))
- **Managed instructions**: New `build_managed_instructions()` for plugin-driven setups — tells the LLM what's automatic and what's available for explicit use ([`1747af6`](https://github.com/fpytloun/mnemory/commit/1747af6))

### Memory Intelligence

- **Context-aware query generation**: `find_memories` accepts an optional `context` hint (e.g., working directory) and injects the user's `project:*` categories into the query generation prompt for more targeted search ([`8247906`](https://github.com/fpytloun/mnemory/commit/8247906))
- **Temporal reasoning**: `event_date` parameter on `add_memory` anchors relative time references ("yesterday" resolves correctly). `find_memories` generates date-aware queries using session timezone ([`d537d62`](https://github.com/fpytloun/mnemory/commit/d537d62), [`9ecdc64`](https://github.com/fpytloun/mnemory/commit/9ecdc64))
- **Improved extraction quality**: Better extraction rules, conversation-aware prompts that filter assistant noise, and LLM-backed metadata validation with retry ([`737f7b2`](https://github.com/fpytloun/mnemory/commit/737f7b2), [`a3f0e48`](https://github.com/fpytloun/mnemory/commit/a3f0e48), [`f9c1df5`](https://github.com/fpytloun/mnemory/commit/f9c1df5))

### Artifacts

- **Auto-artifact creation**: LLM-guided for `infer=True` (extraction prompt decides via `store_artifact` flag), size-based for `infer=False` (content exceeding `MAX_MEMORY_LENGTH` auto-saved). New `remember()` method for conversation processing ([`08a998c`](https://github.com/fpytloun/mnemory/commit/08a998c))
- **M:N artifact linking**: Artifacts decoupled from individual memories — one artifact can be referenced by multiple memories. Storage path changed from `{user_id}/{memory_id}/{artifact_id}/` to `{user_id}/{artifact_id}/` ([`cee1f6a`](https://github.com/fpytloun/mnemory/commit/cee1f6a))
- **Artifact garbage collection**: Deleting a memory checks if other memories still reference the same artifact; orphaned artifacts are cleaned up ([`cee1f6a`](https://github.com/fpytloun/mnemory/commit/cee1f6a))

### LLM & Provider Compatibility

- **Provider adaptation**: Automatic handling of parameter incompatibilities across OpenAI-compatible providers (reasoning_effort, json_schema support) ([`7c585b7`](https://github.com/fpytloun/mnemory/commit/7c585b7))
- **Robust error handling**: Retry logic and graceful handling of `BadRequestError` for chat completions ([`f2c11fa`](https://github.com/fpytloun/mnemory/commit/f2c11fa))

### Benchmarks

- **LoCoMo benchmark suite**: Evaluation on 1540 QA questions across 4 categories (single_hop, multi_hop, temporal, open_domain). Overall score: 73.2 with `gpt-5-mini` ([`26798ff`](https://github.com/fpytloun/mnemory/commit/26798ff), [`7c976ca`](https://github.com/fpytloun/mnemory/commit/7c976ca))

### Bug Fixes

- Serialize local embedded Qdrant write operations to prevent concurrency issues ([`3efc793`](https://github.com/fpytloun/mnemory/commit/3efc793))
- Handle session compaction with fresh recall and re-injection ([`bdef0e0`](https://github.com/fpytloun/mnemory/commit/bdef0e0))
- Configurable `request_timeout` in Open WebUI filter instead of hardcoded 10s ([`444b2d8`](https://github.com/fpytloun/mnemory/commit/444b2d8))

### Breaking Changes

- **Artifact storage path changed**: `{user_id}/{memory_id}/{artifact_id}/{filename}` → `{user_id}/{artifact_id}/{filename}`. Existing artifacts at the old path become inaccessible. Re-create affected artifacts or manually move them.

## [1.0.0] — 2025-02-02

Initial release. MCP server with persistent two-tier memory (Qdrant + S3/filesystem), unified LLM extraction/classification/deduplication pipeline, semantic search with importance reranking, TTL-based memory decay, role-based scoping, and configurable behavioral instructions.
