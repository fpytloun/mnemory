# Monitoring

mnemory exposes a `/metrics` endpoint in [Prometheus text exposition format](https://prometheus.io/docs/instrumenting/exposition_formats/), enabled by default (`ENABLE_METRICS=true`).

## Endpoint Access

By default, `/metrics` and `/health` are served on the main port with standard API key authentication. When `MGMT_PORT` is set, the same endpoints are also exposed on the management port without auth. For production, scrape and probe the management port to avoid sending credentials and to prevent duplicate ingestion (see [Management Port](configuration.md#management-port)).

## Available Metrics

### Counters (in-memory, reset on restart)

| Metric | Labels | Description |
|---|---|---|
| `mnemory_operations_total` | `operation`, `user_id`, `agent_id` | Total MCP/REST operations. Operations: `add_memory`, `add_memories`, `search_memories`, `find_memories`, `update_memory`, `delete_memory`, `delete_all`, `get_core_memories`, `get_recent_memories`, `list_memories`, `save_artifact`, `get_artifact`, `list_artifacts`, `delete_artifact`, `list_categories`, `recall`, `remember`, `initialize_memory`, `fsck_check`, `fsck_apply` |
| `mnemory_remember_facts_extracted_total` | `user_id` | Facts extracted by remember pipeline (pre-dedup) |
| `mnemory_remember_facts_stored_total` | `user_id` | Facts stored by remember pipeline (post-dedup) |
| `mnemory_remember_dedup_decisions_total` | `user_id`, `action` | Remember dedup decisions (add, update, delete, skip) |
| `mnemory_remember_empty_extractions_total` | `user_id` | Remember calls that produced zero facts |
| `mnemory_consolidation_runs_total` | `user_id`, `type` | Consolidation runs (session, cross_session) |
| `mnemory_consolidation_memories_produced_total` | `user_id` | Consolidated memories produced |
| `mnemory_consolidation_memories_superseded_total` | `user_id` | Raw memories superseded by consolidation |
| `mnemory_consolidation_validation_failures_total` | `user_id` | Consolidation output validation failures |

### Histograms (in-memory, reset on restart)

| Metric | Labels | Description |
|---|---|---|
| `mnemory_consolidation_duration_seconds` | `user_id`, `type` | Consolidation run duration (buckets: 1, 5, 10, 30, 60, 120, 300s) |

### Gauges (from Qdrant, cached)

Refreshed asynchronously after the configurable cache TTL expires (`METRICS_CACHE_TTL`, default 600s). Stale values remain available during refresh.

| Metric | Labels | Description |
|---|---|---|
| `mnemory_memories_total` | `user_id`, `agent_id`, `memory_type`, `role` | Total memories by all key dimensions |
| `mnemory_memories_decayed_total` | `user_id`, `agent_id` | Decayed (expired) memories |
| `mnemory_memories_pinned_total` | `user_id`, `agent_id` | Pinned memories |
| `mnemory_memories_by_category_total` | `user_id`, `category` | Memories per category |
| `mnemory_memories_with_artifacts_total` | `user_id`, `agent_id` | Memories with artifacts attached |
| `mnemory_active_sessions` | -- | Active memory sessions (recall/remember) |
| `mnemory_info` | `version`, `vector_backend`, `artifact_backend` | Server metadata (always 1) |
| `mnemory_consolidation_sessions_pending` | `user_id` | Sessions awaiting consolidation |

## Prometheus Scrape Config

```yaml
scrape_configs:
  - job_name: mnemory
    scrape_interval: 60s
    static_configs:
      - targets: ['mnemory:9090']  # MGMT_PORT
```

## Grafana Dashboard

A pre-built Grafana dashboard is available in [`integrations/grafana/`](../integrations/grafana/). Import `dashboard.json` into Grafana for an overview of memories, operations, and breakdowns by type/category/role with user and agent filtering.
