"""Tests for asynchronous Qdrant gauge refreshes."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

from mnemory.metrics import MetricsCollector


def _make_collector(*, ttl: int = 600) -> MetricsCollector:
    config = MagicMock()
    config.server.metrics_cache_ttl = ttl
    config.vector.is_remote = False
    config.artifact.backend = "filesystem"
    config.memory.fsck_auto_interval = 0
    vector = MagicMock()
    vector.collection_name = "mnemory"
    vector._client.scroll.return_value = ([], None)
    sessions = MagicMock()
    sessions.active_count = 2
    return MetricsCollector(vector, sessions, config)


def test_metrics_cache_ttl_default_is_600(monkeypatch) -> None:
    from mnemory.config import ServerConfig

    monkeypatch.delenv("METRICS_CACHE_TTL", raising=False)
    assert ServerConfig().metrics_cache_ttl == 600


def test_refresh_request_uses_exact_payload_and_page_size() -> None:
    collector = _make_collector()

    collector._refresh_gauges_from_qdrant()

    kwargs = collector._vector_store._client.scroll.call_args.kwargs
    assert kwargs["limit"] == 2048
    assert kwargs["with_vectors"] is False
    assert kwargs["with_payload"] == [
        "user_id",
        "agent_id",
        "memory_type",
        "role",
        "memory_layer",
        "decayed_at",
        "pinned",
        "artifacts",
        "categories",
    ]


def test_stale_callers_submit_one_refresh_without_waiting() -> None:
    collector = _make_collector()
    started = threading.Event()
    release = threading.Event()

    def refresh() -> None:
        started.set()
        release.wait(timeout=2)

    collector._refresh_gauges_from_qdrant = refresh
    with ThreadPoolExecutor(max_workers=1) as executor:
        collector.set_thread_pool(executor)
        assert started.wait(timeout=1)

        before = time.monotonic()
        for _ in range(10):
            collector.collect_gauges()
        elapsed = time.monotonic() - before

        assert elapsed < 0.2
        assert executor._work_queue.qsize() == 0
        release.set()


def test_success_ttl_starts_when_refresh_completes() -> None:
    collector = _make_collector(ttl=10)
    started = threading.Event()
    release = threading.Event()

    def refresh() -> None:
        started.set()
        release.wait(timeout=2)

    collector._refresh_gauges_from_qdrant = refresh
    with ThreadPoolExecutor(max_workers=1) as executor:
        collector.set_thread_pool(executor)
        assert started.wait(timeout=1)
        assert collector._last_success_at == 0
        release.set()
    assert collector._last_success_at > 0


def test_failure_keeps_stale_gauges_and_sets_retry_backoff() -> None:
    collector = _make_collector(ttl=10)
    collector._memories_total.labels(
        user_id="filip",
        agent_id="",
        memory_type="fact",
        role="user",
    ).set(3)

    def fail() -> None:
        raise RuntimeError("Qdrant unavailable")

    collector._refresh_gauges_from_qdrant = fail
    with ThreadPoolExecutor(max_workers=1) as executor:
        collector.set_thread_pool(executor)

    assert collector._retry_after > time.monotonic()
    assert len(collector._memories_total._metrics) == 1


def test_gauge_replacement_is_atomic_for_both_expositions() -> None:
    collector = _make_collector()
    collector._memories_total.labels(
        user_id="old",
        agent_id="",
        memory_type="fact",
        role="user",
    ).set(1)
    point = MagicMock()
    point.payload = {
        "user_id": "new",
        "memory_type": "fact",
        "role": "user",
        "memory_layer": "consolidated",
    }
    collector._vector_store._client.scroll.return_value = ([point], None)

    replacement_started = threading.Event()
    release_replacement = threading.Event()
    original_labels = collector._memories_total.labels

    def blocking_labels(*args, **kwargs):
        replacement_started.set()
        release_replacement.wait(timeout=2)
        return original_labels(*args, **kwargs)

    collector._memories_total.labels = blocking_labels
    refresh = threading.Thread(target=collector._refresh_gauges_from_qdrant)
    refresh.start()
    assert replacement_started.wait(timeout=1)

    results: dict[str, object] = {}
    prometheus_reader = threading.Thread(
        target=lambda: results.setdefault("prometheus", collector.generate_metrics())
    )
    json_reader = threading.Thread(
        target=lambda: results.setdefault("json", collector.get_stats_json())
    )
    prometheus_reader.start()
    json_reader.start()
    time.sleep(0.05)
    assert prometheus_reader.is_alive()
    assert json_reader.is_alive()

    release_replacement.set()
    refresh.join(timeout=1)
    prometheus_reader.join(timeout=1)
    json_reader.join(timeout=1)

    output = results["prometheus"]
    assert isinstance(output, bytes)
    assert b'user_id="new"' in output
    assert b'user_id="old"' not in output
    stats = results["json"]
    assert isinstance(stats, dict)
    assert stats["users"] == ["new"]
    assert stats["totals"]["memories"] == 1
