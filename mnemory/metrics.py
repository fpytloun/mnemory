"""Prometheus metrics collection for mnemory.

Exposes two types of metrics:
- **Counters** (in-memory, reset on restart): operation counts per user/agent
- **Gauges** (from Qdrant, cached): memory counts by type/category/status

The MetricsCollector is a module-level singleton, initialized lazily.
When metrics are disabled (ENABLE_METRICS=false), get_collector() returns None
and all instrumentation calls are no-ops.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

logger = logging.getLogger(__name__)

# ── Module-level singleton ────────────────────────────────────────────

_collector: MetricsCollector | None = None
_collector_initialized = False


def get_collector() -> MetricsCollector | None:
    """Get the metrics collector singleton.

    Returns None if metrics are disabled. Safe to call before init
    (returns None).
    """
    return _collector


def init_collector(
    vector_store: Any,
    session_store: Any,
    config: Any,
) -> MetricsCollector | None:
    """Initialize the metrics collector singleton.

    Called once during server startup. Returns None if metrics are
    disabled in config.

    Args:
        vector_store: VectorStore instance for Qdrant aggregation.
        session_store: SessionStore instance for active session count.
        config: Config instance for metadata and cache TTL.
    """
    global _collector, _collector_initialized

    if _collector_initialized:
        return _collector

    _collector_initialized = True

    if not config.server.enable_metrics:
        logger.info("Prometheus metrics disabled (ENABLE_METRICS=false)")
        return None

    _collector = MetricsCollector(
        vector_store=vector_store,
        session_store=session_store,
        config=config,
    )
    logger.info(
        "Prometheus metrics enabled (cache_ttl=%ds)",
        config.server.metrics_cache_ttl,
    )
    return _collector


# ── MetricsCollector ──────────────────────────────────────────────────


class MetricsCollector:
    """Collects and exposes Prometheus metrics for mnemory.

    Uses a custom CollectorRegistry to avoid polluting the global
    default registry. Gauges from Qdrant are cached with a configurable
    TTL to avoid expensive scroll operations on every scrape.
    """

    def __init__(
        self,
        vector_store: Any,
        session_store: Any,
        config: Any,
    ):
        self._vector_store = vector_store
        self._session_store = session_store
        self._config = config
        self._cache_ttl = config.server.metrics_cache_ttl

        # Cache state
        self._cache_lock = threading.Lock()
        self._exposition_lock = threading.RLock()
        self._last_success_at: float = 0.0
        self._retry_after: float = 0.0
        self._refresh_in_flight = False

        # Optional reference to MaintenanceService for schedule info.
        self._maintenance: Any = None

        # Custom registry (not the global default)
        self._registry = CollectorRegistry()

        # ── Info gauge (static metadata) ──────────────────────────
        self._info = Gauge(
            "mnemory_info",
            "mnemory server metadata (always 1)",
            ["version", "vector_backend", "artifact_backend"],
            registry=self._registry,
        )
        from mnemory import __version__

        self._info.labels(
            version=__version__,
            vector_backend="qdrant" if config.vector.is_remote else "qdrant-local",
            artifact_backend=config.artifact.backend,
        ).set(1)

        # ── Operation counter (in-memory) ─────────────────────────
        self._operations = Counter(
            "mnemory_operations_total",
            "Total number of MCP/REST operations",
            ["operation", "user_id", "agent_id"],
            registry=self._registry,
        )

        # ── Active sessions gauge (live) ──────────────────────────
        self._active_sessions = Gauge(
            "mnemory_active_sessions",
            "Number of active memory sessions",
            registry=self._registry,
        )

        # ── Memory gauges (from Qdrant, cached) ───────────────────
        self._memories_total = Gauge(
            "mnemory_memories_total",
            "Total memories by dimensions",
            ["user_id", "agent_id", "memory_type", "role"],
            registry=self._registry,
        )
        self._memories_decayed = Gauge(
            "mnemory_memories_decayed_total",
            "Total currently expired, non-pinned memories",
            ["user_id", "agent_id"],
            registry=self._registry,
        )
        self._memories_pinned = Gauge(
            "mnemory_memories_pinned_total",
            "Total pinned memories",
            ["user_id", "agent_id"],
            registry=self._registry,
        )
        self._memories_by_category = Gauge(
            "mnemory_memories_by_category_total",
            "Total memories by category",
            ["user_id", "category"],
            registry=self._registry,
        )
        self._memories_by_layer = Gauge(
            "mnemory_memories_by_layer_total",
            "Total memories by memory layer",
            ["user_id", "memory_layer"],
            registry=self._registry,
        )
        self._memories_with_artifacts = Gauge(
            "mnemory_memories_with_artifacts_total",
            "Total memories that have artifacts attached",
            ["user_id", "agent_id"],
            registry=self._registry,
        )

        # ── Auto-fsck counters (in-memory) ────────────────────────
        self._autofsck_runs = Counter(
            "mnemory_autofsck_runs_total",
            "Total number of automatic fsck runs",
            ["user_id"],
            registry=self._registry,
        )
        self._autofsck_issues_found = Counter(
            "mnemory_autofsck_issues_found_total",
            "Total issues found by automatic fsck runs",
            ["user_id"],
            registry=self._registry,
        )
        self._autofsck_fixes_applied = Counter(
            "mnemory_autofsck_fixes_applied_total",
            "Total fixes applied by automatic fsck runs",
            ["user_id"],
            registry=self._registry,
        )
        self._autofsck_fixes_failed = Counter(
            "mnemory_autofsck_fixes_failed_total",
            "Total fixes that failed during automatic fsck runs",
            ["user_id"],
            registry=self._registry,
        )
        self._fsck_operation_outcomes = Counter(
            "mnemory_fsck_operation_outcomes_total",
            "Fsck operation journal outcomes",
            ["outcome"],
            registry=self._registry,
        )
        self._fsck_recovery = Counter(
            "mnemory_fsck_recovery_total",
            "Fsck operation recovery outcomes",
            ["outcome"],
            registry=self._registry,
        )
        self._fsck_operations_nonterminal = Gauge(
            "mnemory_fsck_operations_nonterminal",
            "Current nonterminal fsck operation count",
            registry=self._registry,
        )
        self._fsck_operations_stranded = Gauge(
            "mnemory_fsck_operations_stranded",
            "Current stranded fsck operations by bounded age bucket",
            ["age_bucket"],
            registry=self._registry,
        )
        self._session_retry_batches = Counter(
            "mnemory_session_retry_batches_total",
            "Bounded failed-session retry batch outcomes",
            ["outcome"],
            registry=self._registry,
        )
        self._session_retry_sessions = Counter(
            "mnemory_session_retry_sessions_total",
            "Failed-session retry outcomes",
            ["outcome"],
            registry=self._registry,
        )
        self._session_retry_stops = Counter(
            "mnemory_session_retry_stops_total",
            "Failed-session retry stop reasons",
            ["reason"],
            registry=self._registry,
        )
        self._failed_sessions = Gauge(
            "mnemory_failed_sessions",
            "Current failed sessions by bounded classification",
            ["legacy", "failure_class"],
            registry=self._registry,
        )
        self._failed_sessions_age = Gauge(
            "mnemory_failed_sessions_age",
            "Current failed sessions by age bucket",
            ["age_bucket"],
            registry=self._registry,
        )
        self._failed_sessions_size = Gauge(
            "mnemory_failed_sessions_size",
            "Current failed sessions by raw-memory count bucket",
            ["size_bucket"],
            registry=self._registry,
        )
        self._failed_sessions_eligible = Gauge(
            "mnemory_failed_sessions_eligible",
            "Current legacy failed sessions eligible for bounded retry",
            registry=self._registry,
        )

        # ── Remember quality counters (in-memory) ─────────────────
        self._remember_extracted = Counter(
            "mnemory_remember_facts_extracted_total",
            "Total facts extracted by remember pipeline (pre-dedup)",
            ["user_id"],
            registry=self._registry,
        )
        self._remember_stored = Counter(
            "mnemory_remember_facts_stored_total",
            "Total facts stored by remember pipeline (post-dedup)",
            ["user_id"],
            registry=self._registry,
        )
        self._remember_dedup_decisions = Counter(
            "mnemory_remember_dedup_decisions_total",
            "Remember dedup decisions by action type",
            ["user_id", "action"],
            registry=self._registry,
        )
        self._remember_empty = Counter(
            "mnemory_remember_empty_extractions_total",
            "Remember calls that produced zero extracted facts",
            ["user_id"],
            registry=self._registry,
        )
        self._confirmations = Counter(
            "mnemory_confirmations_total",
            "Independent-evidence confirmation outcomes",
            ["outcome"],
            registry=self._registry,
        )
        self._evidence_requests = Counter(
            "mnemory_evidence_requests_total",
            "Synchronous trusted evidence request outcomes",
            ["outcome", "reason"],
            registry=self._registry,
        )

        # ── Consolidation counters (in-memory) ────────────────────
        self._consolidation_runs = Counter(
            "mnemory_consolidation_runs_total",
            "Total consolidation runs by type",
            ["user_id", "type"],
            registry=self._registry,
        )
        self._consolidation_produced = Counter(
            "mnemory_consolidation_memories_produced_total",
            "Total consolidated memories produced",
            ["user_id"],
            registry=self._registry,
        )
        self._consolidation_superseded = Counter(
            "mnemory_consolidation_memories_superseded_total",
            "Total raw memories superseded by consolidation",
            ["user_id"],
            registry=self._registry,
        )
        self._consolidation_validation_failures = Counter(
            "mnemory_consolidation_validation_failures_total",
            "Consolidation output validation failures",
            ["user_id"],
            registry=self._registry,
        )
        self._consolidation_duration = Histogram(
            "mnemory_consolidation_duration_seconds",
            "Consolidation run duration in seconds",
            ["user_id", "type"],
            buckets=[1, 5, 10, 30, 60, 120, 300],
            registry=self._registry,
        )
        self._consolidation_pending = Gauge(
            "mnemory_consolidation_sessions_pending",
            "Number of sessions awaiting consolidation",
            ["user_id"],
            registry=self._registry,
        )

        # ── Latency histograms ────────────────────────────────────
        self._llm_duration = Histogram(
            "mnemory_llm_duration_seconds",
            "LLM call duration in seconds",
            ["operation", "model"],
            buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
            registry=self._registry,
        )
        self._embedding_duration = Histogram(
            "mnemory_embedding_duration_seconds",
            "Embedding call duration in seconds",
            ["type", "method"],
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
            registry=self._registry,
        )
        self._qdrant_duration = Histogram(
            "mnemory_qdrant_duration_seconds",
            "Qdrant operation duration in seconds",
            ["operation"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
            registry=self._registry,
        )

        # ── Auto-fsck last-run timestamp gauge ────────────────────
        self._autofsck_last_run = Gauge(
            "mnemory_autofsck_last_run_timestamp",
            "Unix timestamp of the last automatic fsck run per user",
            ["user_id"],
            registry=self._registry,
        )

        # ── Thread pool gauges ────────────────────────────────────
        self._thread_pool_size = Gauge(
            "mnemory_thread_pool_max_workers",
            "Maximum number of threads in the MCP tool thread pool",
            registry=self._registry,
        )
        self._thread_pool_threads = Gauge(
            "mnemory_thread_pool_threads",
            "Number of spawned threads in the MCP tool thread pool (idle + busy)",
            registry=self._registry,
        )

        # Reference to the thread pool executor (set during startup)
        self._thread_pool: Any = None

    # ── Public API ────────────────────────────────────────────────

    def set_maintenance_service(self, maintenance: Any) -> None:
        """Set a reference to the MaintenanceService for schedule info.

        Called once during startup after both MetricsCollector and
        MaintenanceService are created.
        """
        self._maintenance = maintenance

    def set_thread_pool(self, executor: Any) -> None:
        """Set a reference to the MCP tool thread pool for metrics.

        Called once during startup after the ThreadPoolExecutor is created.
        """
        self._thread_pool = executor
        if hasattr(executor, "_max_workers"):
            self._thread_pool_size.set(executor._max_workers)
        self.collect_gauges()

    def record_operation(
        self,
        operation: str,
        user_id: str,
        agent_id: str | None = None,
    ) -> None:
        """Increment the operation counter.

        Args:
            operation: Operation name (e.g., "add_memory", "search_memories").
            user_id: User who performed the operation.
            agent_id: Agent scope (None → empty string label).
        """
        self._operations.labels(
            operation=operation,
            user_id=user_id,
            agent_id=agent_id or "",
        ).inc()

    def observe_llm_duration(
        self,
        operation: str,
        model: str,
        duration: float,
    ) -> None:
        """Record an LLM call duration.

        Args:
            operation: LLM operation name (e.g., "extract", "query_gen",
                "rerank", "answer", "classify", "dedup", "shorten",
                "reclassify", "compact", "fsck_security", "fsck_dedup",
                "fsck_quality").
            model: Model name (e.g., "gpt-5-mini").
            duration: Duration in seconds.
        """
        self._llm_duration.labels(operation=operation, model=model).observe(duration)

    def observe_embedding_duration(
        self,
        embed_type: str,
        method: str,
        duration: float,
    ) -> None:
        """Record an embedding call duration.

        Args:
            embed_type: "dense" or "sparse".
            method: "embed" or "embed_batch".
            duration: Duration in seconds.
        """
        self._embedding_duration.labels(type=embed_type, method=method).observe(
            duration
        )

    def observe_qdrant_duration(
        self,
        operation: str,
        duration: float,
    ) -> None:
        """Record a Qdrant operation duration.

        Args:
            operation: Qdrant operation name (e.g., "search", "insert",
                "update", "delete", "scroll", "retrieve").
            duration: Duration in seconds.
        """
        self._qdrant_duration.labels(operation=operation).observe(duration)

    def record_autofsck_run(
        self,
        *,
        user_id: str,
        issues_found: int,
        fixes_applied: int,
        fixes_failed: int,
    ) -> None:
        """Record the result of one automatic fsck run for a user.

        Args:
            user_id: User the check was run for.
            issues_found: Number of qualifying issues found.
            fixes_applied: Number of fixes successfully applied.
            fixes_failed: Number of fixes that failed.
        """
        self._autofsck_runs.labels(user_id=user_id).inc()
        self._autofsck_issues_found.labels(user_id=user_id).inc(issues_found)
        self._autofsck_fixes_applied.labels(user_id=user_id).inc(fixes_applied)
        self._autofsck_fixes_failed.labels(user_id=user_id).inc(fixes_failed)
        self._autofsck_last_run.labels(user_id=user_id).set(time.time())

    def record_remember_extraction(
        self,
        *,
        user_id: str,
        facts_extracted: int,
        facts_stored: int,
        dedup_actions: dict[str, int] | None = None,
    ) -> None:
        """Record remember pipeline quality metrics.

        Args:
            user_id: User who triggered the remember call.
            facts_extracted: Number of facts extracted by Stage 1 (pre-dedup).
            facts_stored: Number of facts actually stored after Stage 2.
            dedup_actions: Optional dict of action -> count (add, update, delete, skip).
        """
        self._remember_extracted.labels(user_id=user_id).inc(facts_extracted)
        self._remember_stored.labels(user_id=user_id).inc(facts_stored)
        if facts_extracted == 0:
            self._remember_empty.labels(user_id=user_id).inc()
        if dedup_actions:
            for action, count in dedup_actions.items():
                self._remember_dedup_decisions.labels(
                    user_id=user_id, action=action.lower()
                ).inc(count)

    def record_confirmation(self, outcome: str) -> None:
        """Record a bounded-cardinality confirmation outcome."""
        if outcome not in {"accepted", "replayed", "rejected", "recovered"}:
            outcome = "rejected"
        self._confirmations.labels(outcome=outcome).inc()

    def record_fsck_operation(self, outcome: str) -> None:
        """Record one bounded fsck journal outcome."""
        if outcome not in {"committed", "skipped", "failed", "superseded"}:
            outcome = "failed"
        self._fsck_operation_outcomes.labels(outcome=outcome).inc()

    def record_fsck_recovery(self, outcome: str) -> None:
        """Record one bounded fsck recovery outcome."""
        if outcome not in {"committed", "skipped", "failed", "superseded"}:
            outcome = "failed"
        self._fsck_recovery.labels(outcome=outcome).inc()

    def record_session_retry(
        self,
        *,
        batch_outcome: str,
        succeeded: int,
        failed: int,
        stop_reason: str | None,
    ) -> None:
        """Record bounded failed-session retry outcomes."""
        self._session_retry_batches.labels(outcome=batch_outcome).inc()
        if succeeded:
            self._session_retry_sessions.labels(outcome="succeeded").inc(succeeded)
        if failed:
            self._session_retry_sessions.labels(outcome="failed").inc(failed)
        if stop_reason:
            self._session_retry_stops.labels(reason=stop_reason).inc()

    def record_evidence(self, outcome: str, reason: str = "none") -> None:
        """Record a bounded-cardinality trusted evidence outcome."""
        allowed_outcomes = {
            "accepted",
            "replayed",
            "recovered",
            "rejected",
            "skipped",
        }
        allowed_reasons = {
            "none",
            "auth",
            "schema",
            "claim_active",
            "lease_lost",
            "stale_target",
            "recovery",
            "conflict",
            "timeout",
            "server_error",
        }
        self._evidence_requests.labels(
            outcome=outcome if outcome in allowed_outcomes else "rejected",
            reason=reason if reason in allowed_reasons else "server_error",
        ).inc()

    def record_consolidation_run(
        self,
        *,
        user_id: str,
        run_type: str,  # "session" or "cross_session"
        memories_produced: int = 0,
        memories_superseded: int = 0,
        duration_seconds: float = 0.0,
        validation_failed: bool = False,
    ) -> None:
        """Record consolidation run metrics.

        Args:
            user_id: User the consolidation was run for.
            run_type: "session" for within-session, "cross_session" for cross.
            memories_produced: Number of consolidated memories created.
            memories_superseded: Number of raw memories marked superseded.
            duration_seconds: Total duration of the consolidation run.
            validation_failed: Whether output validation failed.
        """
        self._consolidation_runs.labels(user_id=user_id, type=run_type).inc()
        self._consolidation_produced.labels(user_id=user_id).inc(memories_produced)
        self._consolidation_superseded.labels(user_id=user_id).inc(memories_superseded)
        self._consolidation_duration.labels(user_id=user_id, type=run_type).observe(
            duration_seconds
        )
        if validation_failed:
            self._consolidation_validation_failures.labels(user_id=user_id).inc()

    def collect_gauges(self) -> None:
        """Refresh gauge values from Qdrant (with caching).

        Scrolls all points in the collection (without vectors),
        aggregates by dimensions, and updates gauge values. Results
        are cached for ``metrics_cache_ttl`` seconds.

        Also updates the active sessions gauge and thread pool gauge
        (always fresh).
        """
        # Active sessions — always fresh (cheap)
        self._active_sessions.set(self._session_store.active_count)

        # Thread pool spawned threads — always fresh (cheap).
        # ThreadPoolExecutor._threads is the set of all spawned worker
        # threads (idle + busy). This is the best approximation without
        # instrumenting each task submission.
        if self._thread_pool is not None:
            spawned = len(getattr(self._thread_pool, "_threads", set()))
            self._thread_pool_threads.set(spawned)

        # Submit one asynchronous refresh when stale. Callers keep using the
        # last complete gauge snapshot while it runs.
        now = time.monotonic()
        with self._cache_lock:
            if self._refresh_in_flight:
                return
            if self._last_success_at and now - self._last_success_at < self._cache_ttl:
                return
            if now < self._retry_after or self._thread_pool is None:
                return
            self._refresh_in_flight = True
        try:
            future = self._thread_pool.submit(self._refresh_gauges_from_qdrant)
            future.add_done_callback(self._finish_gauge_refresh)
        except Exception:
            with self._cache_lock:
                self._refresh_in_flight = False
                self._retry_after = time.monotonic() + self._cache_ttl
            logger.warning("Failed to submit metrics refresh", exc_info=True)

    def _finish_gauge_refresh(self, future: Any) -> None:
        """Update cache timing after an asynchronous refresh finishes."""
        completed_at = time.monotonic()
        error = future.exception()
        with self._cache_lock:
            self._refresh_in_flight = False
            if error is None:
                self._last_success_at = completed_at
                self._retry_after = 0.0
            else:
                self._retry_after = completed_at + self._cache_ttl
        if error is not None:
            logger.warning(
                "Failed to refresh metrics from Qdrant",
                exc_info=(type(error), error, error.__traceback__),
            )

    def generate_metrics(self) -> bytes:
        """Generate Prometheus text exposition format output."""
        self.collect_gauges()
        with self._exposition_lock:
            return generate_latest(self._registry)

    @property
    def content_type(self) -> str:
        """Prometheus content type header value."""
        return CONTENT_TYPE_LATEST

    def get_stats_json(self) -> dict:
        """Return an atomic JSON snapshot for the management UI."""
        self.collect_gauges()
        with self._exposition_lock:
            return self._get_stats_json_unlocked()

    def _get_stats_json_unlocked(self) -> dict:
        """Return metrics as a JSON-serializable dict for the management UI.

        Calls collect_gauges() to ensure freshness, then reads the internal
        prometheus_client gauge/counter values to build a structured response.
        Reuses the cached Qdrant aggregation — no extra scroll.
        """
        from mnemory import __version__

        totals = {
            "memories": 0,
            "raw": 0,
            "consolidated": 0,
            "pinned": 0,
            "decayed": 0,
            "with_artifacts": 0,
        }
        by_type: dict[str, int] = defaultdict(int)
        by_category: dict[str, int] = defaultdict(int)
        by_layer: dict[str, int] = defaultdict(int)
        by_role: dict[str, int] = defaultdict(int)
        users: set[str] = set()
        agents: set[str] = set()
        by_user: dict[str, dict] = {}

        def _ensure_user(uid: str) -> dict:
            if uid not in by_user:
                by_user[uid] = {
                    "total": 0,
                    "raw": 0,
                    "consolidated": 0,
                    "pinned": 0,
                    "decayed": 0,
                    "with_artifacts": 0,
                    "by_type": {},
                    "by_category": {},
                    "by_layer": {},
                    "by_role": {},
                }
            return by_user[uid]

        # Read _memories_total gauge: labels are (user_id, agent_id, memory_type, role)
        try:
            for labels, metric in self._memories_total._metrics.items():
                uid, aid, mtype, role = labels
                count = int(metric._value.get())
                totals["memories"] += count
                by_type[mtype] += count
                by_role[role] += count
                users.add(uid)
                if aid:
                    agents.add(aid)
                u = _ensure_user(uid)
                u["total"] += count
                u["by_type"][mtype] = u["by_type"].get(mtype, 0) + count
                u["by_role"][role] = u["by_role"].get(role, 0) + count
        except Exception:
            logger.debug("Failed to read _memories_total gauge", exc_info=True)

        # Read pinned gauge: labels are (user_id, agent_id)
        try:
            for labels, metric in self._memories_pinned._metrics.items():
                uid, aid = labels
                count = int(metric._value.get())
                totals["pinned"] += count
                users.add(uid)
                _ensure_user(uid)["pinned"] += count
        except Exception:
            logger.debug("Failed to read _memories_pinned gauge", exc_info=True)

        # Read decayed gauge: labels are (user_id, agent_id)
        try:
            for labels, metric in self._memories_decayed._metrics.items():
                uid, aid = labels
                count = int(metric._value.get())
                totals["decayed"] += count
                users.add(uid)
                _ensure_user(uid)["decayed"] += count
        except Exception:
            logger.debug("Failed to read _memories_decayed gauge", exc_info=True)

        # Read with_artifacts gauge: labels are (user_id, agent_id)
        try:
            for labels, metric in self._memories_with_artifacts._metrics.items():
                uid, aid = labels
                count = int(metric._value.get())
                totals["with_artifacts"] += count
                users.add(uid)
                _ensure_user(uid)["with_artifacts"] += count
        except Exception:
            logger.debug("Failed to read _memories_with_artifacts gauge", exc_info=True)

        # Read by_category gauge: labels are (user_id, category)
        try:
            for labels, metric in self._memories_by_category._metrics.items():
                uid, cat = labels
                count = int(metric._value.get())
                by_category[cat] += count
                users.add(uid)
                u = _ensure_user(uid)
                u["by_category"][cat] = u["by_category"].get(cat, 0) + count
        except Exception:
            logger.debug("Failed to read _memories_by_category gauge", exc_info=True)

        # Read by_layer gauge: labels are (user_id, memory_layer)
        try:
            for labels, metric in self._memories_by_layer._metrics.items():
                uid, layer = labels
                count = int(metric._value.get())
                by_layer[layer] += count
                if layer in {"raw", "consolidated"}:
                    totals[layer] += count
                users.add(uid)
                u = _ensure_user(uid)
                u["by_layer"][layer] = u["by_layer"].get(layer, 0) + count
                if layer in {"raw", "consolidated"}:
                    u[layer] += count
        except Exception:
            logger.debug("Failed to read _memories_by_layer gauge", exc_info=True)

        # Read operations counter: labels are (operation, user_id, agent_id)
        operations: dict[str, dict] = {}
        try:
            for labels, metric in self._operations._metrics.items():
                op, uid, aid = labels
                count = int(metric._value.get())
                if op not in operations:
                    operations[op] = {"total": 0, "by_user": {}}
                operations[op]["total"] += count
                if uid:
                    operations[op]["by_user"][uid] = (
                        operations[op]["by_user"].get(uid, 0) + count
                    )
        except Exception:
            logger.debug("Failed to read _operations counter", exc_info=True)

        # Sort operations by total descending
        operations = dict(
            sorted(operations.items(), key=lambda x: x[1]["total"], reverse=True)
        )

        # Determine vector/artifact backend from info gauge
        vector_backend = "unknown"
        artifact_backend = "unknown"
        try:
            for labels, metric in self._info._metrics.items():
                version_label, vb, ab = labels
                vector_backend = vb
                artifact_backend = ab
                break
        except Exception:
            pass

        # Build autofsck section
        autofsck_cfg = self._config.memory
        autofsck_enabled = autofsck_cfg.fsck_auto_interval > 0
        autofsck_by_user: dict[str, dict] = {}

        try:
            for labels, metric in self._autofsck_runs._metrics.items():
                (uid,) = labels
                if uid not in autofsck_by_user:
                    autofsck_by_user[uid] = {
                        "runs": 0,
                        "issues_found": 0,
                        "fixes_applied": 0,
                        "fixes_failed": 0,
                        "last_run": None,
                    }
                autofsck_by_user[uid]["runs"] = int(metric._value.get())
        except Exception:
            logger.debug("Failed to read _autofsck_runs counter", exc_info=True)

        try:
            for labels, metric in self._autofsck_issues_found._metrics.items():
                (uid,) = labels
                if uid in autofsck_by_user:
                    autofsck_by_user[uid]["issues_found"] = int(metric._value.get())
        except Exception:
            logger.debug("Failed to read _autofsck_issues_found counter", exc_info=True)

        try:
            for labels, metric in self._autofsck_fixes_applied._metrics.items():
                (uid,) = labels
                if uid in autofsck_by_user:
                    autofsck_by_user[uid]["fixes_applied"] = int(metric._value.get())
        except Exception:
            logger.debug(
                "Failed to read _autofsck_fixes_applied counter", exc_info=True
            )

        try:
            for labels, metric in self._autofsck_fixes_failed._metrics.items():
                (uid,) = labels
                if uid in autofsck_by_user:
                    autofsck_by_user[uid]["fixes_failed"] = int(metric._value.get())
        except Exception:
            logger.debug("Failed to read _autofsck_fixes_failed counter", exc_info=True)

        try:
            for labels, metric in self._autofsck_last_run._metrics.items():
                (uid,) = labels
                if uid in autofsck_by_user:
                    ts = metric._value.get()
                    autofsck_by_user[uid]["last_run"] = ts if ts > 0 else None
        except Exception:
            logger.debug("Failed to read _autofsck_last_run gauge", exc_info=True)

        autofsck_totals = {
            "runs": sum(u["runs"] for u in autofsck_by_user.values()),
            "issues_found": sum(u["issues_found"] for u in autofsck_by_user.values()),
            "fixes_applied": sum(u["fixes_applied"] for u in autofsck_by_user.values()),
            "fixes_failed": sum(u["fixes_failed"] for u in autofsck_by_user.values()),
        }

        return {
            "version": __version__,
            "vector_backend": vector_backend,
            "artifact_backend": artifact_backend,
            "active_sessions": self._session_store.active_count,
            "users": sorted(users),
            "agents": sorted(agents),
            "totals": totals,
            "by_type": dict(sorted(by_type.items(), key=lambda x: x[1], reverse=True)),
            "by_category": dict(
                sorted(by_category.items(), key=lambda x: x[1], reverse=True)
            ),
            "by_layer": {
                "raw": by_layer.get("raw", 0),
                "consolidated": by_layer.get("consolidated", 0),
            },
            "by_role": dict(by_role),
            "by_user": {
                uid: {
                    **udata,
                    "by_type": dict(
                        sorted(
                            udata["by_type"].items(), key=lambda x: x[1], reverse=True
                        )
                    ),
                    "by_category": dict(
                        sorted(
                            udata["by_category"].items(),
                            key=lambda x: x[1],
                            reverse=True,
                        )
                    ),
                    "by_layer": {
                        "raw": udata["by_layer"].get("raw", 0),
                        "consolidated": udata["by_layer"].get("consolidated", 0),
                    },
                    "by_role": dict(udata["by_role"]),
                }
                for uid, udata in sorted(by_user.items())
            },
            "operations": operations,
            "autofsck": {
                "enabled": autofsck_enabled,
                "interval_hours": autofsck_cfg.fsck_auto_interval,
                "min_confidence": autofsck_cfg.fsck_auto_min_confidence,
                "min_severity": autofsck_cfg.fsck_auto_min_severity,
                "by_user": autofsck_by_user,
                "totals": autofsck_totals,
                "next_run_at": (
                    self._maintenance.next_run_at
                    if self._maintenance is not None
                    else None
                ),
                "running": (
                    self._maintenance.is_running
                    if self._maintenance is not None
                    else False
                ),
                "last_check_ids": (
                    self._maintenance.last_check_ids
                    if self._maintenance is not None
                    else {}
                ),
            },
        }

    # ── Internal ──────────────────────────────────────────────────

    def _refresh_gauges_from_qdrant(self) -> None:
        """Scroll all Qdrant points and aggregate into gauge values.

        Clears existing gauge label sets before updating to avoid
        stale labels from deleted users/agents/types.
        """
        client = self._vector_store._client
        collection = self._vector_store.collection_name

        # Aggregation accumulators
        # (user_id, agent_id, memory_type, role) -> count
        total: dict[tuple[str, str, str, str], int] = defaultdict(int)
        # (user_id, agent_id) -> count
        decayed: dict[tuple[str, str], int] = defaultdict(int)
        pinned: dict[tuple[str, str], int] = defaultdict(int)
        with_artifacts: dict[tuple[str, str], int] = defaultdict(int)
        # (user_id, category) -> count
        by_category: dict[tuple[str, str], int] = defaultdict(int)
        # (user_id, memory_layer) -> count
        by_layer: dict[tuple[str, str], int] = defaultdict(int)

        # Scroll all points (no vectors, selected payload fields only)
        offset = None
        batch_size = 2048
        payload_fields = [
            "user_id",
            "agent_id",
            "memory_type",
            "role",
            "memory_layer",
            "expires_at",
            "pinned",
            "artifacts",
            "categories",
            "revision_state",
        ]

        while True:
            points, next_offset = client.scroll(
                collection_name=collection,
                scroll_filter=None,
                limit=batch_size,
                offset=offset,
                with_payload=payload_fields,
                with_vectors=False,
            )

            for point in points:
                payload = point.payload or {}
                if payload.get("revision_state", "active") != "active":
                    continue
                user_id = payload.get("user_id", "unknown")
                agent_id = payload.get("agent_id", "") or ""
                memory_type = payload.get("memory_type", "unknown")
                role = payload.get("role", "user")
                memory_layer = payload.get("memory_layer") or "consolidated"

                # Total by dimensions
                total[(user_id, agent_id, memory_type, role)] += 1

                # Preserve the metric series name, but count actual expiry.
                expires_at = payload.get("expires_at")
                if expires_at and not payload.get("pinned"):
                    try:
                        expires_dt = datetime.fromisoformat(expires_at)
                        if expires_dt.tzinfo is None:
                            expires_dt = expires_dt.replace(tzinfo=timezone.utc)
                        if expires_dt < datetime.now(timezone.utc):
                            decayed[(user_id, agent_id)] += 1
                    except (TypeError, ValueError):
                        logger.debug("Ignoring invalid expires_at value %r", expires_at)

                # Pinned
                if payload.get("pinned"):
                    pinned[(user_id, agent_id)] += 1

                # Has artifacts
                artifacts = payload.get("artifacts")
                if artifacts and len(artifacts) > 0:
                    with_artifacts[(user_id, agent_id)] += 1

                # Categories
                categories = payload.get("categories") or []
                for cat in categories:
                    by_category[(user_id, cat)] += 1

                by_layer[(user_id, memory_layer)] += 1

            if next_offset is None:
                break
            offset = next_offset

        from qdrant_client.models import FieldCondition, Filter, MatchValue

        nonterminal = 0
        stranded = defaultdict(int)
        now = datetime.now(timezone.utc)
        stranded_after = self._config.memory.fsck_stranded_after_seconds
        retry_max_memories = self._config.memory.legacy_failed_retry_max_raw_memories
        if client.collection_exists("_mnemory_operations"):
            offset = None
            while True:
                points, next_offset = client.scroll(
                    collection_name="_mnemory_operations",
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(
                                key="operation_kind", match=MatchValue(value="fsck")
                            )
                        ]
                    ),
                    limit=512,
                    offset=offset,
                    with_payload=["status", "created_at_utc", "updated_at_utc"],
                    with_vectors=False,
                )
                for point in points:
                    payload = point.payload or {}
                    if payload.get("status") not in {"planned", "applying", "failed"}:
                        continue
                    nonterminal += 1
                    value = payload.get("updated_at_utc") or payload.get(
                        "created_at_utc"
                    )
                    try:
                        age = (now - datetime.fromisoformat(str(value))).total_seconds()
                    except (TypeError, ValueError):
                        age = 0
                    if age < stranded_after:
                        continue
                    bucket = (
                        "lt_1h"
                        if age < 3600
                        else "1h_24h"
                        if age < 86400
                        else "24h_7d"
                        if age < 604800
                        else "gt_7d"
                    )
                    stranded[bucket] += 1
                if next_offset is None:
                    break
                offset = next_offset

        failed_classes = defaultdict(int)
        failed_ages = defaultdict(int)
        failed_sizes = defaultdict(int)
        failed_eligible = 0
        if client.collection_exists("_mnemory_sessions"):
            offset = None
            while True:
                points, next_offset = client.scroll(
                    collection_name="_mnemory_sessions",
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(
                                key="consolidation_state",
                                match=MatchValue(value="failed"),
                            )
                        ]
                    ),
                    limit=512,
                    offset=offset,
                    with_payload=[
                        "legacy_failure",
                        "failure_class",
                        "last_error_at",
                        "updated_at",
                        "memory_ids",
                        "session_revision",
                        "consolidation_token",
                        "consolidation_plan",
                    ],
                    with_vectors=False,
                )
                for point in points:
                    payload = point.payload or {}
                    legacy = bool(payload.get("legacy_failure"))
                    failure_class = payload.get("failure_class") or "unknown"
                    failed_classes[(str(legacy).lower(), failure_class)] += 1
                    value = payload.get("last_error_at") or payload.get("updated_at")
                    try:
                        age_days = (now - datetime.fromisoformat(str(value))).days
                    except (TypeError, ValueError):
                        age_days = -1
                    age_bucket = (
                        "unknown"
                        if age_days < 0
                        else "lt_7d"
                        if age_days < 7
                        else "7d_30d"
                        if age_days < 30
                        else "30d_90d"
                        if age_days < 90
                        else "gt_90d"
                    )
                    failed_ages[age_bucket] += 1
                    count = len(payload.get("memory_ids") or [])
                    size_bucket = (
                        "1_5"
                        if count <= 5
                        else "6_20"
                        if count <= 20
                        else "21_100"
                        if count <= 100
                        else "gt_100"
                    )
                    failed_sizes[size_bucket] += 1
                    if (
                        legacy
                        and 0 < count <= retry_max_memories
                        and payload.get("session_revision") is not None
                        and not payload.get("consolidation_token")
                        and not payload.get("consolidation_plan")
                    ):
                        failed_eligible += 1
                if next_offset is None:
                    break
                offset = next_offset

        with self._exposition_lock:
            # Clear and replace all Qdrant-backed labels as one snapshot.
            self._memories_total._metrics.clear()
            self._memories_decayed._metrics.clear()
            self._memories_pinned._metrics.clear()
            self._memories_by_category._metrics.clear()
            self._memories_by_layer._metrics.clear()
            self._memories_with_artifacts._metrics.clear()
            self._fsck_operations_nonterminal.set(nonterminal)
            self._fsck_operations_stranded._metrics.clear()
            for bucket, count in stranded.items():
                self._fsck_operations_stranded.labels(age_bucket=bucket).set(count)
            self._failed_sessions._metrics.clear()
            for (legacy, failure_class), count in failed_classes.items():
                self._failed_sessions.labels(
                    legacy=legacy, failure_class=failure_class
                ).set(count)
            self._failed_sessions_age._metrics.clear()
            for bucket, count in failed_ages.items():
                self._failed_sessions_age.labels(age_bucket=bucket).set(count)
            self._failed_sessions_size._metrics.clear()
            for bucket, count in failed_sizes.items():
                self._failed_sessions_size.labels(size_bucket=bucket).set(count)
            self._failed_sessions_eligible.set(failed_eligible)

            for (uid, aid, mtype, role), count in total.items():
                self._memories_total.labels(
                    user_id=uid,
                    agent_id=aid,
                    memory_type=mtype,
                    role=role,
                ).set(count)

            for (uid, aid), count in decayed.items():
                self._memories_decayed.labels(
                    user_id=uid,
                    agent_id=aid,
                ).set(count)

            for (uid, aid), count in pinned.items():
                self._memories_pinned.labels(
                    user_id=uid,
                    agent_id=aid,
                ).set(count)

            for (uid, cat), count in by_category.items():
                self._memories_by_category.labels(
                    user_id=uid,
                    category=cat,
                ).set(count)

            for (uid, layer), count in by_layer.items():
                self._memories_by_layer.labels(
                    user_id=uid,
                    memory_layer=layer,
                ).set(count)

            for (uid, aid), count in with_artifacts.items():
                self._memories_with_artifacts.labels(
                    user_id=uid,
                    agent_id=aid,
                ).set(count)

        logger.debug(
            "Metrics refreshed: %d label sets across %d points",
            len(total)
            + len(decayed)
            + len(pinned)
            + len(by_category)
            + len(by_layer)
            + len(with_artifacts),
            sum(total.values()),
        )
