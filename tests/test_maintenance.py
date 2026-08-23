"""Tests for mnemory.maintenance — periodic auto-fsck maintenance service.

Covers:
- MaintenanceService: disabled when interval=0, start/stop lifecycle
- MaintenanceService._meets_thresholds: all confidence/severity combinations
- MaintenanceService._run_for_user: per-user error isolation, metrics recording
- VectorStore.list_user_ids: mocked Qdrant scroll, pagination, deduplication
- MetricsCollector.record_autofsck_run: counter increments, last-run timestamp
- MetricsCollector.get_stats_json: autofsck section structure
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from mnemory.maintenance import (
    _SEVERITY_ORDER,
    MaintenanceService,
    gc_superseded_raw,
)

# ── Helpers ──────────────────────────────────────────────────────────


def _make_config(
    *,
    interval: int = 0,
    min_confidence: float = 0.95,
    min_severity: str = "medium",
    consolidation_raw_retention_days: int = 30,
    consolidation_idle_threshold: int = 3600,
    consolidation_check_interval: int = 300,
):
    """Build a minimal mock Config for MaintenanceService."""
    cfg = MagicMock()
    cfg.memory.fsck_auto_interval = interval
    cfg.memory.fsck_auto_min_confidence = min_confidence
    cfg.memory.fsck_auto_min_severity = min_severity
    cfg.memory.consolidation_raw_retention_days = consolidation_raw_retention_days
    cfg.memory.consolidation_idle_threshold = consolidation_idle_threshold
    cfg.memory.consolidation_check_interval = consolidation_check_interval
    return cfg


def _make_issue(*, confidence: float | None = 1.0, severity: str = "high"):
    """Build a minimal mock FsckIssue."""
    issue = MagicMock()
    issue.issue_id = f"issue-{id(issue)}"
    issue.confidence = confidence
    issue.severity = severity
    return issue


def _make_fsck_service(
    *,
    user_ids: list[str] | None = None,
    check_status: str = "completed",
    issues: list | None = None,
    apply_result: dict | None = None,
):
    """Build a minimal mock FsckService."""
    fsck = MagicMock()

    # list_user_ids on the vector store
    fsck._vector.list_user_ids.return_value = user_ids or []

    # start_check returns a FsckCheck-like object
    check = MagicMock()
    check.check_id = "check-abc"
    check.status = check_status
    check.issues = issues or []
    fsck.start_check.return_value = check
    fsck.get_check.return_value = check

    # apply_check returns counts
    fsck.apply_check.return_value = apply_result or {
        "applied": 0,
        "skipped": 0,
        "failed": 0,
        "details": [],
    }

    # gc_superseded_raw returns deleted count
    fsck.gc_superseded_raw.return_value = {"deleted": 0}

    return fsck


# ── TestMaintenanceService ────────────────────────────────────────────


class TestMaintenanceService:
    """Tests for MaintenanceService lifecycle and logic."""

    def test_disabled_when_interval_zero(self):
        """Service should not start a task when interval=0."""

        async def _run():
            cfg = _make_config(interval=0)
            fsck = _make_fsck_service()
            svc = MaintenanceService(config=cfg, fsck=fsck)

            await svc.start()
            assert svc._task is None

            await svc.stop()  # no-op, should not raise

        asyncio.run(_run())

    def test_consolidation_uses_one_global_scan_and_recovers_first(self):
        """Maintenance uses one session scan and retries reset idle sessions."""

        async def _run():
            old = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
            consolidation = MagicMock()
            consolidation._sessions.scan_consolidation_candidates.return_value = [
                {
                    "session_id": "completed",
                    "user_id": "alice",
                    "updated_at": old,
                    "memory_ids": ["m1"],
                    "consolidation_state": "consolidating",
                    "consolidated_memory_ids": ["c1"],
                },
                {
                    "session_id": "retry",
                    "user_id": "alice",
                    "updated_at": old,
                    "memory_ids": ["m2"],
                    "consolidation_state": "consolidating",
                },
                {
                    "session_id": "idle",
                    "user_id": "bob",
                    "updated_at": old,
                    "memory_ids": ["m3"],
                    "consolidation_state": "idle",
                },
            ]
            consolidation.recover_session.side_effect = [True, False]
            result = MagicMock()
            result.state = "consolidated"
            consolidation.consolidate_session.return_value = result
            service = MaintenanceService(
                config=_make_config(),
                fsck=_make_fsck_service(),
                consolidation=consolidation,
            )

            await service._run_consolidation()

            consolidation._sessions.scan_consolidation_candidates.assert_called_once_with()
            consolidation._vector.list_user_ids.assert_not_called()
            assert [
                call.args[0]
                for call in consolidation.consolidate_session.call_args_list
            ] == ["retry", "idle"]

        asyncio.run(_run())

    def test_consolidation_isolates_recovery_errors_per_session(self):
        """One recovery error does not block other sessions for the user."""

        async def _run():
            old = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
            consolidation = MagicMock()
            consolidation._sessions.scan_consolidation_candidates.return_value = [
                {
                    "session_id": "broken",
                    "user_id": "alice",
                    "updated_at": old,
                    "memory_ids": ["m1"],
                    "consolidation_state": "consolidating",
                },
                {
                    "session_id": "retry",
                    "user_id": "alice",
                    "updated_at": old,
                    "memory_ids": ["m2"],
                    "consolidation_state": "consolidating",
                },
                {
                    "session_id": "idle",
                    "user_id": "alice",
                    "updated_at": old,
                    "memory_ids": ["m3"],
                    "consolidation_state": "idle",
                },
            ]
            consolidation.recover_session.side_effect = [
                RuntimeError("recovery failed"),
                False,
            ]
            result = MagicMock()
            result.state = "consolidated"
            consolidation.consolidate_session.return_value = result
            service = MaintenanceService(
                config=_make_config(),
                fsck=_make_fsck_service(),
                consolidation=consolidation,
            )

            await service._run_consolidation()

            assert [
                call.args[0]
                for call in consolidation.consolidate_session.call_args_list
            ] == ["retry", "idle"]

        asyncio.run(_run())

    def test_start_creates_task_when_enabled(self):
        """Service should create a background task when interval > 0."""

        async def _run():
            cfg = _make_config(interval=24)
            fsck = _make_fsck_service()
            svc = MaintenanceService(config=cfg, fsck=fsck)

            await svc.start()
            assert svc._task is not None
            assert not svc._task.done()

            await svc.stop()
            assert svc._task is None

        asyncio.run(_run())

    def test_stop_cancels_task(self):
        """stop() should cancel the running task cleanly."""

        async def _run():
            cfg = _make_config(interval=999)  # very long interval — loop will sleep
            fsck = _make_fsck_service()
            svc = MaintenanceService(config=cfg, fsck=fsck)

            await svc.start()
            task = svc._task
            assert task is not None

            await svc.stop()
            assert task.cancelled() or task.done()
            assert svc._task is None

        asyncio.run(_run())

    # ── _meets_thresholds ─────────────────────────────────────────

    def test_meets_thresholds_both_pass(self):
        """Issue with high confidence and high severity should pass."""
        issue = _make_issue(confidence=0.99, severity="high")
        assert MaintenanceService._meets_thresholds(issue, 0.95, "medium") is True

    def test_meets_thresholds_confidence_too_low(self):
        """Issue with confidence below threshold should fail."""
        issue = _make_issue(confidence=0.80, severity="high")
        assert MaintenanceService._meets_thresholds(issue, 0.95, "medium") is False

    def test_meets_thresholds_severity_too_low(self):
        """Issue with severity below threshold should fail."""
        issue = _make_issue(confidence=0.99, severity="low")
        assert MaintenanceService._meets_thresholds(issue, 0.95, "medium") is False

    def test_meets_thresholds_exact_confidence(self):
        """Issue with confidence exactly at threshold should pass."""
        issue = _make_issue(confidence=0.95, severity="medium")
        assert MaintenanceService._meets_thresholds(issue, 0.95, "medium") is True

    def test_meets_thresholds_exact_severity(self):
        """Issue with severity exactly at threshold should pass."""
        issue = _make_issue(confidence=1.0, severity="medium")
        assert MaintenanceService._meets_thresholds(issue, 0.95, "medium") is True

    def test_meets_thresholds_none_confidence(self):
        """None confidence should be treated as 0.0 (fails any positive threshold)."""
        issue = _make_issue(confidence=None, severity="high")
        assert MaintenanceService._meets_thresholds(issue, 0.95, "medium") is False

    def test_meets_thresholds_zero_confidence_threshold(self):
        """Zero confidence threshold should pass any confidence including None."""
        issue = _make_issue(confidence=None, severity="high")
        assert MaintenanceService._meets_thresholds(issue, 0.0, "low") is True

    def test_meets_thresholds_severity_order(self):
        """Severity ordering: high > medium > low."""
        assert _SEVERITY_ORDER["high"] > _SEVERITY_ORDER["medium"]
        assert _SEVERITY_ORDER["medium"] > _SEVERITY_ORDER["low"]

    def test_meets_thresholds_all_severity_levels(self):
        """Test all severity combinations against 'medium' threshold."""
        cfg_min = "medium"
        assert (
            MaintenanceService._meets_thresholds(
                _make_issue(confidence=1.0, severity="high"), 0.0, cfg_min
            )
            is True
        )
        assert (
            MaintenanceService._meets_thresholds(
                _make_issue(confidence=1.0, severity="medium"), 0.0, cfg_min
            )
            is True
        )
        assert (
            MaintenanceService._meets_thresholds(
                _make_issue(confidence=1.0, severity="low"), 0.0, cfg_min
            )
            is False
        )

    # ── _run_for_user ─────────────────────────────────────────────

    def test_run_for_user_applies_qualifying_fixes(self):
        """Qualifying issues should be applied and metrics recorded."""

        async def _run():
            issue = _make_issue(confidence=0.99, severity="high")
            fsck = _make_fsck_service(
                user_ids=["filip"],
                issues=[issue],
                apply_result={"applied": 1, "skipped": 0, "failed": 0, "details": []},
            )
            collector = MagicMock()
            cfg = _make_config(interval=24, min_confidence=0.95, min_severity="medium")
            svc = MaintenanceService(config=cfg, fsck=fsck, collector=collector)

            await svc._run_for_user("filip")

            fsck.apply_check.assert_called_once()
            collector.record_autofsck_run.assert_called_once_with(
                user_id="filip",
                issues_found=1,
                fixes_applied=1,
                fixes_failed=0,
            )

        asyncio.run(_run())

    def test_run_for_user_no_qualifying_fixes(self):
        """Issues below threshold should not trigger apply_check."""

        async def _run():
            issue = _make_issue(
                confidence=0.50, severity="low"
            )  # below both thresholds
            fsck = _make_fsck_service(
                user_ids=["filip"],
                issues=[issue],
            )
            collector = MagicMock()
            cfg = _make_config(interval=24, min_confidence=0.95, min_severity="medium")
            svc = MaintenanceService(config=cfg, fsck=fsck, collector=collector)

            await svc._run_for_user("filip")

            fsck.apply_check.assert_not_called()
            collector.record_autofsck_run.assert_called_once_with(
                user_id="filip",
                issues_found=0,
                fixes_applied=0,
                fixes_failed=0,
            )

        asyncio.run(_run())

    def test_run_for_user_failed_check_skips_apply(self):
        """If check ends with status != completed, apply should be skipped."""

        async def _run():
            fsck = _make_fsck_service(check_status="failed")
            collector = MagicMock()
            cfg = _make_config(interval=24)
            svc = MaintenanceService(config=cfg, fsck=fsck, collector=collector)

            await svc._run_for_user("filip")

            fsck.apply_check.assert_not_called()
            collector.record_autofsck_run.assert_not_called()

        asyncio.run(_run())

    def test_run_auto_fsck_per_user_error_isolation(self):
        """An error for one user should not prevent processing other users."""

        async def _run():
            fsck = _make_fsck_service(user_ids=["alice", "bob"])
            collector = MagicMock()
            cfg = _make_config(interval=24)
            svc = MaintenanceService(config=cfg, fsck=fsck, collector=collector)

            call_count = 0

            async def _run_for_user_side_effect(user_id):
                nonlocal call_count
                call_count += 1
                if user_id == "alice":
                    raise RuntimeError("simulated failure for alice")
                # bob succeeds normally

            svc._run_for_user = _run_for_user_side_effect

            # Should not raise even though alice fails
            await svc._run_auto_fsck()
            assert call_count == 2

        asyncio.run(_run())

    def test_run_auto_fsck_no_users(self):
        """When no users exist, run should complete without errors."""

        async def _run():
            fsck = _make_fsck_service(user_ids=[])
            cfg = _make_config(interval=24)
            svc = MaintenanceService(config=cfg, fsck=fsck)

            # Should not raise
            await svc._run_auto_fsck()

        asyncio.run(_run())

    def test_no_collector_does_not_raise(self):
        """Service should work fine when no collector is provided."""

        async def _run():
            issue = _make_issue(confidence=0.99, severity="high")
            fsck = _make_fsck_service(
                issues=[issue],
                apply_result={"applied": 1, "skipped": 0, "failed": 0, "details": []},
            )
            cfg = _make_config(interval=24)
            svc = MaintenanceService(config=cfg, fsck=fsck, collector=None)

            # Should not raise
            await svc._run_for_user("filip")

        asyncio.run(_run())


# ── TestListUserIds ───────────────────────────────────────────────────


class TestListUserIds:
    """Tests for VectorStore.list_user_ids."""

    def _make_point(self, user_id: str):
        """Create a mock Qdrant point with a user_id payload."""
        point = MagicMock()
        point.payload = {"user_id": user_id}
        return point

    def _make_store(self):
        """Create a mock VectorStore (without spec to allow private attrs)."""
        store = MagicMock()
        store.collection_name = "mnemory"
        return store

    def test_returns_sorted_unique_user_ids(self):
        """Should return sorted, deduplicated user_ids."""
        from mnemory.storage.vector import VectorStore

        store = self._make_store()
        points = [
            self._make_point("charlie"),
            self._make_point("alice"),
            self._make_point("alice"),  # duplicate
            self._make_point("bob"),
        ]
        store._client.scroll.return_value = (points, None)  # no next page

        result = VectorStore.list_user_ids(store)
        assert result == ["alice", "bob", "charlie"]

    def test_handles_pagination(self):
        """Should follow pagination until next_offset is None."""
        from mnemory.storage.vector import VectorStore

        store = self._make_store()
        page1 = [self._make_point("alice"), self._make_point("bob")]
        page2 = [self._make_point("charlie")]

        store._client.scroll.side_effect = [
            (page1, "cursor-1"),
            (page2, None),
        ]

        result = VectorStore.list_user_ids(store)
        assert result == ["alice", "bob", "charlie"]
        assert store._client.scroll.call_count == 2

    def test_empty_collection(self):
        """Empty collection should return empty list."""
        from mnemory.storage.vector import VectorStore

        store = self._make_store()
        store._client.scroll.return_value = ([], None)

        result = VectorStore.list_user_ids(store)
        assert result == []

    def test_skips_points_without_user_id(self):
        """Points with missing user_id should be ignored."""
        from mnemory.storage.vector import VectorStore

        store = self._make_store()
        point_with = self._make_point("alice")
        point_without = MagicMock()
        point_without.payload = {}  # no user_id key

        store._client.scroll.return_value = ([point_with, point_without], None)

        result = VectorStore.list_user_ids(store)
        assert result == ["alice"]


# ── TestMetricsAutofsck ───────────────────────────────────────────────


class TestMetricsAutofsck:
    """Tests for MetricsCollector auto-fsck metrics."""

    def _make_collector(
        self,
        *,
        interval: int = 24,
        min_confidence: float = 0.95,
        min_severity: str = "medium",
    ):
        """Create a MetricsCollector with a mock config and minimal dependencies."""
        from mnemory.metrics import MetricsCollector

        config = MagicMock()
        config.server.enable_metrics = True
        config.server.metrics_cache_ttl = 60
        config.vector.is_remote = False
        config.artifact.backend = "filesystem"
        config.memory.fsck_auto_interval = interval
        config.memory.fsck_auto_min_confidence = min_confidence
        config.memory.fsck_auto_min_severity = min_severity

        vector_store = MagicMock()
        session_store = MagicMock()
        session_store.active_count = 0

        return MetricsCollector(
            vector_store=vector_store,
            session_store=session_store,
            config=config,
        )

    def test_record_autofsck_run_increments_counters(self):
        """record_autofsck_run should increment all counters."""
        collector = self._make_collector()

        collector.record_autofsck_run(
            user_id="filip",
            issues_found=5,
            fixes_applied=3,
            fixes_failed=1,
        )

        # Verify counters via Prometheus internal state
        runs = collector._autofsck_runs.labels(user_id="filip")._value.get()
        issues = collector._autofsck_issues_found.labels(user_id="filip")._value.get()
        applied = collector._autofsck_fixes_applied.labels(user_id="filip")._value.get()
        failed = collector._autofsck_fixes_failed.labels(user_id="filip")._value.get()

        assert runs == 1
        assert issues == 5
        assert applied == 3
        assert failed == 1

    def test_record_autofsck_run_sets_last_run_timestamp(self):
        """record_autofsck_run should set the last_run gauge to current time."""
        collector = self._make_collector()
        before = time.time()

        collector.record_autofsck_run(
            user_id="filip",
            issues_found=0,
            fixes_applied=0,
            fixes_failed=0,
        )

        after = time.time()
        ts = collector._autofsck_last_run.labels(user_id="filip")._value.get()
        assert before <= ts <= after

    def test_record_autofsck_run_accumulates(self):
        """Multiple calls should accumulate counter values."""
        collector = self._make_collector()

        collector.record_autofsck_run(
            user_id="filip", issues_found=2, fixes_applied=1, fixes_failed=0
        )
        collector.record_autofsck_run(
            user_id="filip", issues_found=3, fixes_applied=2, fixes_failed=1
        )

        issues = collector._autofsck_issues_found.labels(user_id="filip")._value.get()
        applied = collector._autofsck_fixes_applied.labels(user_id="filip")._value.get()
        failed = collector._autofsck_fixes_failed.labels(user_id="filip")._value.get()

        assert issues == 5
        assert applied == 3
        assert failed == 1

    def test_get_stats_json_autofsck_section_disabled(self):
        """get_stats_json should include autofsck.enabled=False when interval=0."""
        collector = self._make_collector(interval=0)

        # Patch collect_gauges to avoid Qdrant scroll
        collector.collect_gauges = MagicMock()

        stats = collector.get_stats_json()
        assert "autofsck" in stats
        assert stats["autofsck"]["enabled"] is False
        assert stats["autofsck"]["interval_hours"] == 0

    def test_get_stats_json_autofsck_section_enabled(self):
        """get_stats_json should include correct autofsck config when enabled."""
        collector = self._make_collector(
            interval=24, min_confidence=0.90, min_severity="high"
        )
        collector.collect_gauges = MagicMock()

        stats = collector.get_stats_json()
        af = stats["autofsck"]
        assert af["enabled"] is True
        assert af["interval_hours"] == 24
        assert af["min_confidence"] == 0.90
        assert af["min_severity"] == "high"

    def test_get_stats_json_autofsck_by_user(self):
        """get_stats_json should include per-user autofsck data after a run."""
        collector = self._make_collector(interval=24)
        collector.collect_gauges = MagicMock()

        collector.record_autofsck_run(
            user_id="filip", issues_found=3, fixes_applied=2, fixes_failed=0
        )

        stats = collector.get_stats_json()
        af = stats["autofsck"]
        assert "filip" in af["by_user"]
        u = af["by_user"]["filip"]
        assert u["runs"] == 1
        assert u["issues_found"] == 3
        assert u["fixes_applied"] == 2
        assert u["fixes_failed"] == 0
        assert u["last_run"] is not None

    def test_get_stats_json_autofsck_totals(self):
        """get_stats_json autofsck.totals should aggregate across users."""
        collector = self._make_collector(interval=24)
        collector.collect_gauges = MagicMock()

        collector.record_autofsck_run(
            user_id="alice", issues_found=2, fixes_applied=1, fixes_failed=0
        )
        collector.record_autofsck_run(
            user_id="bob", issues_found=4, fixes_applied=3, fixes_failed=1
        )

        stats = collector.get_stats_json()
        totals = stats["autofsck"]["totals"]
        assert totals["runs"] == 2
        assert totals["issues_found"] == 6
        assert totals["fixes_applied"] == 4
        assert totals["fixes_failed"] == 1


def test_gc_uses_lightweight_metadata_scan_and_preserves_eligibility() -> None:
    vector = MagicMock()
    old = (datetime.now(timezone.utc) - timedelta(days=31)).isoformat()
    vector.scroll_gc_metadata.return_value = [
        {
            "id": "eligible",
            "metadata": {
                "memory_layer": "raw",
                "superseded_by": "new",
                "artifacts": [],
                "created_at_utc": old,
            },
        },
        {
            "id": "artifact",
            "metadata": {
                "memory_layer": "raw",
                "superseded_by": "new",
                "artifacts": [{"id": "a1"}],
                "created_at_utc": old,
            },
        },
    ]

    assert gc_superseded_raw(vector, "filip") == {"deleted": 1}
    vector.scroll_gc_metadata.assert_called_once_with(user_id="filip")
    vector.scroll_with_vectors.assert_not_called()
    vector.delete.assert_called_once_with("eligible")


def test_gc_metadata_scan_requests_exact_fields_without_vectors() -> None:
    from mnemory.storage.vector import VectorStore

    point = MagicMock()
    point.id = "memory-1"
    point.payload = {"memory_layer": "raw"}
    store = VectorStore.__new__(VectorStore)
    store._client = MagicMock()
    store._config = MagicMock()
    store._config.vector.collection_name = "mnemory"
    store._client.scroll.return_value = ([point], None)

    assert store.scroll_gc_metadata(user_id="filip") == [
        {"id": "memory-1", "metadata": {"memory_layer": "raw"}}
    ]
    kwargs = store._client.scroll.call_args.kwargs
    assert kwargs["with_vectors"] is False
    assert kwargs["with_payload"] == [
        "memory_layer",
        "superseded_by",
        "artifacts",
        "created_at_utc",
    ]
