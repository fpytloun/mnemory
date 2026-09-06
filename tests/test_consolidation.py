"""Unit tests for the consolidation service."""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from mnemory.consolidation import ConsolidationService


class TestGlobalConsolidationCandidateScan:
    """Tests for maintenance candidate discovery across all users."""

    def test_follows_all_cursors_and_requests_only_required_payload(self):
        from mnemory.storage.vector import SessionSummaryStore

        store = SessionSummaryStore.__new__(SessionSummaryStore)
        store._client = MagicMock()

        def point(index: int) -> MagicMock:
            value = MagicMock()
            value.payload = {
                "session_id": f"s{index}",
                "user_id": f"u{index % 2}",
                "consolidation_state": "idle",
            }
            return value

        store._client.scroll.side_effect = [
            ([point(index) for index in range(120)], "cursor-1"),
            ([point(index) for index in range(120, 263)], None),
        ]

        sessions = store.scan_consolidation_candidates()

        assert len(sessions) == 263
        assert store._client.scroll.call_count == 2
        first = store._client.scroll.call_args_list[0].kwargs
        second = store._client.scroll.call_args_list[1].kwargs
        assert first["with_vectors"] is False
        assert first["with_payload"] == [
            "session_id",
            "user_id",
            "owner_id",
            "agent_id",
            "memory_ids",
            "updated_at",
            "consolidation_state",
            "consolidated_memory_ids",
            "session_revision",
            "consolidation_token",
            "consolidation_plan",
            "session_update_token",
        ]
        assert second["offset"] == "cursor-1"


def _make_service(vector_mock=None):
    """Create a ConsolidationService with mocked dependencies."""
    service = ConsolidationService.__new__(ConsolidationService)
    service._vector = vector_mock or MagicMock()
    service._llm = MagicMock()
    service._embedding = MagicMock()
    service._memory = MagicMock()
    service._sessions = MagicMock()
    service._sessions.finalize_consolidation.return_value = "consolidated"
    service._collector = None
    service._config = MagicMock()
    return service


def _wire_mock_session_cas(store) -> None:
    """Make a mocked Qdrant client apply payload CAS writes in place."""
    point = store._client.retrieve.return_value[0]

    def set_payload(*, payload, **kwargs):
        point.payload.update(payload)

    def delete_payload(*, keys, **kwargs):
        for key in keys:
            point.payload.pop(key, None)

    store._client.set_payload.side_effect = set_payload
    store._client.delete_payload.side_effect = delete_payload


class TestFetchRawMemories:
    """Tests for ConsolidationService._fetch_raw_memories."""

    def test_includes_memories_without_layer(self):
        """Memories without memory_layer field should be treated as raw."""
        vector = MagicMock()
        vector.get_by_id.return_value = {
            "id": "mem-1",
            "memory": "User likes Python",
            "user_id": "user-1",
            "owner_id": "user-1",
            "metadata": {
                "memory_type": "preference",
                # No memory_layer field
            },
        }
        service = _make_service(vector)
        result = service._fetch_raw_memories(["mem-1"], "user-1")
        assert len(result) == 1
        assert result[0]["id"] == "mem-1"

    def test_includes_explicit_raw(self):
        """Memories with memory_layer='raw' should be included."""
        vector = MagicMock()
        vector.get_by_id.return_value = {
            "id": "mem-2",
            "memory": "User prefers dark mode",
            "user_id": "user-1",
            "owner_id": "user-1",
            "metadata": {"memory_layer": "raw"},
        }
        service = _make_service(vector)
        result = service._fetch_raw_memories(["mem-2"], "user-1")
        assert len(result) == 1

    def test_excludes_consolidated(self):
        """Memories with memory_layer='consolidated' should be excluded."""
        vector = MagicMock()
        vector.get_by_id.return_value = {
            "id": "mem-3",
            "memory": "User lives in Prague",
            "user_id": "user-1",
            "owner_id": "user-1",
            "metadata": {"memory_layer": "consolidated"},
        }
        service = _make_service(vector)
        result = service._fetch_raw_memories(["mem-3"], "user-1")
        assert len(result) == 0

    def test_excludes_superseded(self):
        """Superseded raw memories should be excluded."""
        vector = MagicMock()
        vector.get_by_id.return_value = {
            "id": "mem-4",
            "memory": "User likes Java",
            "user_id": "user-1",
            "owner_id": "user-1",
            "metadata": {
                "memory_layer": "raw",
                "superseded_by": "consolidated-1",
            },
        }
        service = _make_service(vector)
        result = service._fetch_raw_memories(["mem-4"], "user-1")
        assert len(result) == 0

    def test_skips_missing_memories(self):
        """Memory IDs that don't exist should be skipped."""
        vector = MagicMock()
        vector.get_by_id.return_value = None
        service = _make_service(vector)
        result = service._fetch_raw_memories(["nonexistent"], "user-1")
        assert len(result) == 0

    def test_handles_exceptions(self):
        """Exceptions from vector.get_by_id should be caught and skipped."""
        vector = MagicMock()
        vector.get_by_id.side_effect = [
            Exception("Connection error"),
            {
                "id": "mem-5",
                "memory": "User likes Rust",
                "user_id": "user-1",
                "owner_id": "user-1",
                "metadata": {"memory_layer": "raw"},
            },
        ]
        service = _make_service(vector)
        result = service._fetch_raw_memories(["bad-id", "mem-5"], "user-1")
        assert len(result) == 1
        assert result[0]["id"] == "mem-5"

    def test_mixed_memories(self):
        """Test with a mix of raw, consolidated, superseded, and missing."""
        vector = MagicMock()
        vector.get_by_id.side_effect = [
            # raw, no layer field (should include)
            {
                "id": "m1",
                "memory": "fact 1",
                "user_id": "user-1",
                "owner_id": "user-1",
                "metadata": {},
            },
            # explicit raw (should include)
            {
                "id": "m2",
                "memory": "fact 2",
                "user_id": "user-1",
                "owner_id": "user-1",
                "metadata": {"memory_layer": "raw"},
            },
            # consolidated (should exclude)
            {
                "id": "m3",
                "memory": "fact 3",
                "user_id": "user-1",
                "owner_id": "user-1",
                "metadata": {"memory_layer": "consolidated"},
            },
            # superseded raw (should exclude)
            {
                "id": "m4",
                "memory": "fact 4",
                "user_id": "user-1",
                "owner_id": "user-1",
                "metadata": {"memory_layer": "raw", "superseded_by": "c1"},
            },
            # missing (should skip)
            None,
        ]
        service = _make_service(vector)
        result = service._fetch_raw_memories(["m1", "m2", "m3", "m4", "m5"], "user-1")
        assert len(result) == 2
        assert {r["id"] for r in result} == {"m1", "m2"}

    def test_deduplicates_memory_ids(self):
        """Duplicate memory IDs should be fetched only once."""
        vector = MagicMock()
        vector.get_by_id.return_value = {
            "id": "m1",
            "memory": "fact 1",
            "user_id": "user-1",
            "owner_id": "user-1",
            "metadata": {"memory_layer": "raw"},
        }
        service = _make_service(vector)
        result = service._fetch_raw_memories(["m1", "m1", "m1", "m1"], "user-1")
        assert len(result) == 1
        assert result[0]["id"] == "m1"
        # get_by_id should be called only once despite 4 duplicate IDs
        assert vector.get_by_id.call_count == 1

    def test_rejects_cross_tenant_session_memory_id(self):
        """A poisoned session must not read or mutate another tenant's source."""
        vector = MagicMock()
        vector.get_by_id.return_value = {
            "id": "foreign",
            "memory": "Private foreign fact",
            "user_id": "other-user",
            "owner_id": "other-owner",
            "metadata": {"memory_layer": "raw", "revision_state": "active"},
        }
        service = _make_service(vector)

        result = service._fetch_raw_memories(
            ["foreign"],
            "user-1",
            "owner-1",
            "agent-1",
        )

        assert result == []


class TestReConsolidationStateReset:
    """Tests for re-consolidation state reset in SessionSummaryStore.upsert()."""

    def test_state_resets_to_idle_on_new_memories_after_consolidation(self):
        """When new memories arrive after consolidation, state should reset to idle."""
        from mnemory.storage.vector import SessionSummaryStore

        store = SessionSummaryStore.__new__(SessionSummaryStore)
        store._client = MagicMock()

        # Simulate existing consolidated session
        store._client.retrieve.return_value = [
            MagicMock(
                payload={
                    "session_id": "ses-1",
                    "user_id": "user-1",
                    "summary": "old summary",
                    "turn_count": 5,
                    "memory_ids": ["m1", "m2"],
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "updated_at": "2026-01-01T00:00:00+00:00",
                    "consolidation_state": "consolidated",
                }
            )
        ]
        _wire_mock_session_cas(store)

        store.upsert(
            session_id="ses-1",
            user_id="user-1",
            summary="new summary",
            new_memory_ids=["m3"],
        )

        # Verify the upsert was called with consolidation_state="idle"
        payload = store._client.retrieve.return_value[0].payload
        assert payload["consolidation_state"] == "idle"

    def test_state_preserved_when_no_new_memories(self):
        """When no new memories arrive, consolidated state should be preserved."""
        from mnemory.storage.vector import SessionSummaryStore

        store = SessionSummaryStore.__new__(SessionSummaryStore)
        store._client = MagicMock()

        store._client.retrieve.return_value = [
            MagicMock(
                payload={
                    "session_id": "ses-1",
                    "user_id": "user-1",
                    "summary": "old summary",
                    "turn_count": 5,
                    "memory_ids": ["m1"],
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "updated_at": "2026-01-01T00:00:00+00:00",
                    "consolidation_state": "consolidated",
                }
            )
        ]
        _wire_mock_session_cas(store)

        store.upsert(
            session_id="ses-1",
            user_id="user-1",
            summary="updated summary",
            new_memory_ids=None,
        )

        payload = store._client.retrieve.return_value[0].payload
        assert payload["consolidation_state"] == "consolidated"

    def test_idle_state_stays_idle_with_new_memories(self):
        """Idle state should remain idle when new memories arrive."""
        from mnemory.storage.vector import SessionSummaryStore

        store = SessionSummaryStore.__new__(SessionSummaryStore)
        store._client = MagicMock()

        store._client.retrieve.return_value = [
            MagicMock(
                payload={
                    "session_id": "ses-1",
                    "user_id": "user-1",
                    "summary": "summary",
                    "turn_count": 3,
                    "memory_ids": ["m1"],
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "updated_at": "2026-01-01T00:00:00+00:00",
                    "consolidation_state": "idle",
                }
            )
        ]
        _wire_mock_session_cas(store)

        store.upsert(
            session_id="ses-1",
            user_id="user-1",
            summary="summary",
            new_memory_ids=["m2"],
        )

        payload = store._client.retrieve.return_value[0].payload
        assert payload["consolidation_state"] == "idle"

    def test_failed_state_resets_and_clears_error_on_new_memories(self):
        from mnemory.storage.vector import SessionSummaryStore

        store = SessionSummaryStore.__new__(SessionSummaryStore)
        store._client = MagicMock()
        store._client.retrieve.return_value = [
            MagicMock(
                payload={
                    "session_id": "ses-1",
                    "user_id": "user-1",
                    "summary": "summary",
                    "turn_count": 3,
                    "memory_ids": ["m1"],
                    "consolidation_state": "failed",
                    "attempt_count": 2,
                    "last_error_code": "TimeoutError",
                    "last_error_at": "2026-01-01T00:00:00+00:00",
                }
            )
        ]
        _wire_mock_session_cas(store)

        store.upsert(
            session_id="ses-1",
            user_id="user-1",
            summary="summary",
            new_memory_ids=["m2"],
        )

        payload = store._client.retrieve.return_value[0].payload
        assert payload["consolidation_state"] == "idle"
        assert payload["attempt_count"] == 2
        assert "last_error_code" not in payload
        assert "last_error_at" not in payload


def test_consolidation_claim_has_one_cross_replica_winner() -> None:
    """A unique claimant token must make one filtered session claim win."""
    from mnemory.storage.vector import SessionSummaryStore

    client = QdrantClient(location=":memory:")
    client.create_collection(
        collection_name=SessionSummaryStore.COLLECTION,
        vectors_config=VectorParams(size=1, distance=Distance.COSINE),
    )
    first = SessionSummaryStore(client)
    second = SessionSummaryStore(client)
    first.upsert(
        session_id="ses-race",
        user_id="user-1",
        summary="summary",
        new_memory_ids=["m1"],
    )
    barrier = threading.Barrier(2)
    results: list[bool] = []

    def claim(store: SessionSummaryStore, token: str) -> None:
        barrier.wait()
        results.append(
            store.claim_consolidation(
                "ses-race",
                expected_revision=1,
                token=token,
                attempt_count=1,
            )
        )

    threads = [
        threading.Thread(target=claim, args=(first, "claim-a")),
        threading.Thread(target=claim, args=(second, "claim-b")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(results) == [False, True]


def test_expired_retry_session_claim_transfers_once() -> None:
    from mnemory.storage.vector import SessionSummaryStore

    client = QdrantClient(location=":memory:")
    client.create_collection(
        collection_name=SessionSummaryStore.COLLECTION,
        vectors_config=VectorParams(size=1, distance=Distance.COSINE),
    )
    first = SessionSummaryStore(client)
    second = SessionSummaryStore(client)
    first.upsert(
        session_id="ses-retry",
        user_id="user-1",
        summary="summary",
        new_memory_ids=["m1"],
    )
    session = first.get("ses-retry")
    assert session is not None
    point_id = session["_point_id"]
    client.set_payload(
        collection_name=SessionSummaryStore.COLLECTION,
        payload={"consolidation_state": "failed"},
        points=[point_id],
        wait=True,
    )
    reset = first.reset_failed_for_retry(
        "ses-retry",
        point_id=point_id,
        expected_revision=1,
        retry_token="retry-token",
        retry_claimant="worker-1",
        retry_lease_seconds=60,
    )
    assert reset is not None
    client.set_payload(
        collection_name=SessionSummaryStore.COLLECTION,
        payload={
            "retry_lease_expires_at": (
                datetime.now(timezone.utc) - timedelta(seconds=1)
            ).isoformat()
        },
        points=[point_id],
        wait=True,
    )

    transferred = second.transfer_retry_claim(
        "ses-retry",
        point_id=point_id,
        retry_token="retry-token",
        previous_claimant="worker-1",
        retry_claimant="worker-2",
        retry_lease_seconds=60,
    )

    assert transferred is not None
    assert transferred["retry_claimant"] == "worker-2"
    assert not first.renew_retry_claim(
        "ses-retry",
        point_id=point_id,
        retry_token="retry-token",
        retry_claimant="worker-1",
        retry_lease_seconds=60,
    )


def test_consolidation_output_stops_before_write_after_lease_loss() -> None:
    service = _make_service()
    guard = MagicMock(side_effect=RuntimeError("lease lost"))

    with pytest.raises(RuntimeError, match="lease lost"):
        service._store_consolidated(
            [
                {
                    "text": "fact",
                    "memory_id": "memory-1",
                    "operation_id": "operation-1",
                    "role": "user",
                }
            ],
            user_id="user-1",
            owner_id="owner-1",
            agent_id=None,
            mutation_guard=guard,
        )

    service._memory.add_memory.assert_not_called()


def test_consolidation_output_rechecks_claim_at_memory_sink() -> None:
    service = _make_service()
    guard = MagicMock(side_effect=[None, RuntimeError("lease transferred")])
    writes: list[str] = []

    def add_memory(*args, _mutation_guard, **kwargs):
        _mutation_guard()
        writes.append("stored")
        return {"results": [{"id": "memory-1"}]}

    service._memory.add_memory.side_effect = add_memory

    with pytest.raises(RuntimeError, match="lease transferred"):
        service._store_consolidated(
            [
                {
                    "text": "fact",
                    "memory_id": "memory-1",
                    "operation_id": "operation-1",
                    "role": "user",
                }
            ],
            user_id="user-1",
            owner_id="owner-1",
            agent_id=None,
            mutation_guard=guard,
        )

    assert writes == []


def test_session_ids_are_isolated_by_tenant_and_agent_scope() -> None:
    """The same external session ID must map to separate scoped points."""
    from mnemory.storage.vector import SessionSummaryStore

    client = QdrantClient(location=":memory:")
    client.create_collection(
        collection_name=SessionSummaryStore.COLLECTION,
        vectors_config=VectorParams(size=1, distance=Distance.COSINE),
    )
    store = SessionSummaryStore(client)
    store.upsert(
        session_id="ses-scope",
        user_id="user-1",
        owner_id="owner-1",
        agent_id="agent-1",
        summary="private",
        new_memory_ids=["m1"],
    )

    store.upsert(
        session_id="ses-scope",
        user_id="user-2",
        owner_id="owner-2",
        agent_id="agent-2",
        summary="separate",
        new_memory_ids=["m2"],
    )

    first = store.get(
        "ses-scope",
        user_id="user-1",
        owner_id="owner-1",
        agent_id="agent-1",
    )
    second = store.get(
        "ses-scope",
        user_id="user-2",
        owner_id="owner-2",
        agent_id="agent-2",
    )
    assert first["memory_ids"] == ["m1"]
    assert second["memory_ids"] == ["m2"]
    assert first["_point_id"] != second["_point_id"]


def test_session_listing_filters_owner_and_sibling_agent() -> None:
    from mnemory.storage.vector import SessionSummaryStore

    client = QdrantClient(location=":memory:")
    client.create_collection(
        collection_name=SessionSummaryStore.COLLECTION,
        vectors_config=VectorParams(size=1, distance=Distance.COSINE),
    )
    store = SessionSummaryStore(client)
    for owner_id, agent_id, memory_id in (
        ("owner-1", "parent:alpha", "m1"),
        ("owner-1", "parent:beta", "m2"),
        ("owner-2", "parent:alpha", "m3"),
    ):
        store.upsert(
            session_id=f"ses-{memory_id}",
            user_id="user-1",
            owner_id=owner_id,
            agent_id=agent_id,
            summary=f"summary {memory_id}",
            new_memory_ids=[memory_id],
        )

    sessions = store.list_for_user(
        "user-1",
        owner_id="owner-1",
        session_agent_id="parent:alpha",
    )

    assert [session["session_id"] for session in sessions] == ["ses-m1"]


def test_candidate_scan_merges_append_only_memory_links() -> None:
    from mnemory.storage.vector import SessionSummaryStore

    client = QdrantClient(location=":memory:")
    client.create_collection(
        collection_name=SessionSummaryStore.COLLECTION,
        vectors_config=VectorParams(size=1, distance=Distance.COSINE),
    )
    store = SessionSummaryStore(client)
    store.upsert(
        session_id="ses-scan",
        user_id="user-1",
        summary="summary",
        new_memory_ids=["m1"],
    )
    store._write_session_append_events(
        session_id="ses-scan",
        user_id="user-1",
        owner_id="user-1",
        agent_id=None,
        memory_ids=["m2"],
    )

    sessions = store.scan_consolidation_candidates()

    assert len(sessions) == 1
    assert set(sessions[0]["memory_ids"]) == {"m1", "m2"}


def test_legacy_session_append_keeps_one_projection() -> None:
    from mnemory.storage.vector import SessionSummaryStore, _session_point_id

    client = QdrantClient(location=":memory:")
    client.create_collection(
        collection_name=SessionSummaryStore.COLLECTION,
        vectors_config=VectorParams(size=1, distance=Distance.COSINE),
    )
    client.upsert(
        collection_name=SessionSummaryStore.COLLECTION,
        points=[
            PointStruct(
                id=_session_point_id("ses-legacy"),
                vector=[0.0],
                payload={
                    "session_id": "ses-legacy",
                    "user_id": "user-1",
                    "owner_id": "user-1",
                    "summary": "old",
                    "memory_ids": ["m1"],
                    "turn_count": 1,
                    "session_revision": 1,
                    "consolidation_state": "idle",
                },
            )
        ],
        wait=True,
    )
    store = SessionSummaryStore(client)

    store.upsert(
        session_id="ses-legacy",
        user_id="user-1",
        summary="new",
        new_memory_ids=["m2"],
    )

    points, _ = client.scroll(
        collection_name=SessionSummaryStore.COLLECTION,
        with_payload=True,
        with_vectors=False,
    )
    assert len(points) == 1
    assert set(store.get("ses-legacy")["memory_ids"]) == {"m1", "m2"}


def test_deleted_session_does_not_restore_append_events() -> None:
    from mnemory.storage.vector import SessionSummaryStore

    client = QdrantClient(location=":memory:")
    client.create_collection(
        collection_name=SessionSummaryStore.COLLECTION,
        vectors_config=VectorParams(size=1, distance=Distance.COSINE),
    )
    store = SessionSummaryStore(client)
    store.upsert(
        session_id="ses-delete",
        user_id="user-1",
        summary="old",
        new_memory_ids=["m1"],
    )
    session = store.get("ses-delete")
    store.delete("ses-delete", point_id=session["_point_id"])
    store.upsert(
        session_id="ses-delete",
        user_id="user-1",
        summary="new",
        new_memory_ids=["m2"],
    )

    assert store.get("ses-delete")["memory_ids"] == ["m2"]


def test_session_read_timeout_never_replaces_projection() -> None:
    from mnemory.storage.vector import SessionSummaryStore

    store = SessionSummaryStore.__new__(SessionSummaryStore)
    store._client = MagicMock()
    store._client.get_collections.return_value.collections = []
    store._client.retrieve.side_effect = TimeoutError("qdrant timeout")

    with pytest.raises(TimeoutError, match="qdrant timeout"):
        store.upsert(
            session_id="ses-timeout",
            user_id="user-1",
            summary="new",
            new_memory_ids=[],
        )

    store._client.upsert.assert_not_called()


def test_concurrent_session_appends_preserve_both_ids() -> None:
    """The session CAS loop must retain concurrent raw-memory links."""
    from mnemory.storage.vector import SessionSummaryStore

    client = QdrantClient(location=":memory:")
    client.create_collection(
        collection_name=SessionSummaryStore.COLLECTION,
        vectors_config=VectorParams(size=1, distance=Distance.COSINE),
    )
    first = SessionSummaryStore(client)
    second = SessionSummaryStore(client)
    first.upsert(
        session_id="ses-append",
        user_id="user-1",
        summary="base",
        new_memory_ids=["m1"],
    )
    barrier = threading.Barrier(2)

    def append(store: SessionSummaryStore, memory_id: str) -> None:
        barrier.wait()
        store.upsert(
            session_id="ses-append",
            user_id="user-1",
            summary=memory_id,
            new_memory_ids=[memory_id],
        )

    threads = [
        threading.Thread(target=append, args=(first, "m2")),
        threading.Thread(target=append, args=(second, "m3")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    session = first.get("ses-append")
    assert set(session["memory_ids"]) == {"m1", "m2", "m3"}
    assert session["session_revision"] == 3


def test_stale_session_append_preserves_consolidation_claim() -> None:
    """A claim between session read and CAS must not be overwritten."""
    from mnemory.storage.vector import SessionSummaryStore

    client = QdrantClient(location=":memory:")
    client.create_collection(
        collection_name=SessionSummaryStore.COLLECTION,
        vectors_config=VectorParams(size=1, distance=Distance.COSINE),
    )
    writer = SessionSummaryStore(client)
    claimant = SessionSummaryStore(client)
    writer.upsert(
        session_id="ses-interleave",
        user_id="user-1",
        summary="base",
        new_memory_ids=["m1"],
    )
    reached_write = threading.Event()
    release_write = threading.Event()
    original_set_payload = client.set_payload
    delayed_once = False

    def delayed_set_payload(**kwargs):
        nonlocal delayed_once
        if "session_update_token" in kwargs.get("payload", {}) and not delayed_once:
            delayed_once = True
            reached_write.set()
            assert release_write.wait(timeout=5)
        return original_set_payload(**kwargs)

    client.set_payload = delayed_set_payload
    thread = threading.Thread(
        target=writer.upsert,
        kwargs={
            "session_id": "ses-interleave",
            "user_id": "user-1",
            "summary": "new",
            "new_memory_ids": ["m2"],
        },
    )
    thread.start()
    assert reached_write.wait(timeout=5)
    assert claimant.claim_consolidation(
        "ses-interleave",
        expected_revision=1,
        token="claim-1",
        attempt_count=1,
    )
    release_write.set()
    thread.join(timeout=5)

    session = claimant.get("ses-interleave")
    assert session["consolidation_state"] == "consolidating"
    assert session["consolidation_token"] == "claim-1"
    assert set(session["memory_ids"]) == {"m1", "m2"}
    assert session["session_revision"] == 2


def test_store_consolidated_propagates_partial_write_failure() -> None:
    """A failed output write must retain the recovery plan and active sources."""
    service = _make_service()
    service._memory.add_memory.side_effect = [
        {"results": [{"id": "out-1"}]},
        RuntimeError("injected output failure"),
    ]
    facts = [
        {
            "memory_id": "out-1",
            "operation_id": "op-1",
            "source_session_id": "ses-1",
            "derived_from": ["raw-1"],
            "text": "User likes Python",
            "memory_type": "preference",
            "categories": ["technical"],
            "importance": "normal",
            "pinned": False,
            "role": "user",
        },
        {
            "memory_id": "out-2",
            "operation_id": "op-1",
            "source_session_id": "ses-1",
            "derived_from": ["raw-2"],
            "text": "User likes Rust",
            "memory_type": "preference",
            "categories": ["technical"],
            "importance": "normal",
            "pinned": False,
            "role": "user",
        },
    ]

    with pytest.raises(RuntimeError, match="injected output failure"):
        service._store_consolidated(
            facts,
            user_id="user-1",
            agent_id=None,
        )


def test_store_consolidated_preserves_distinct_owner() -> None:
    service = _make_service()
    service._memory.add_memory.return_value = {"results": [{"id": "out-1"}]}
    facts = [
        {
            "memory_id": "out-1",
            "operation_id": "op-1",
            "source_session_id": "ses-1",
            "derived_from": ["raw-1"],
            "text": "Owner-scoped fact",
            "memory_type": "fact",
            "categories": ["personal"],
            "importance": "normal",
            "pinned": False,
            "role": "user",
        }
    ]

    service._store_consolidated(
        facts,
        user_id="subject-1",
        owner_id="owner-1",
        agent_id=None,
    )

    assert service._memory.add_memory.call_args.kwargs["owner_id"] == "owner-1"


def test_consolidation_carries_evidence_only_as_tombstones() -> None:
    service = _make_service()
    service._memory._config.memory.validation_max_score_roots = 3
    service._memory.vector.get_by_id.side_effect = [
        {
            "metadata": {
                "validation_eligible": True,
                "validation_state": "confirmed",
                "evidence_root_ids": ["trusted-1", "trusted-2"],
            }
        },
        {
            "metadata": {
                "validation_eligible": False,
                "validation_state": "unverified",
                "evidence_root_ids": ["untrusted-1", "untrusted-2"],
            }
        },
    ]
    service._memory.add_memory.return_value = {"results": [{"id": "out-1"}]}

    service._store_consolidated(
        [
            {
                "memory_id": "out-1",
                "operation_id": "op-1",
                "source_session_id": "ses-1",
                "derived_from": ["raw-trusted", "raw-untrusted"],
                "text": "User lives in Prague",
                "memory_type": "fact",
                "categories": ["personal"],
                "importance": "normal",
                "pinned": False,
                "role": "user",
            }
        ],
        user_id="user-1",
        agent_id=None,
    )

    revision_metadata = service._memory.add_memory.call_args.kwargs[
        "_revision_metadata"
    ]
    assert revision_metadata["evidence_root_ids"] == []
    assert revision_metadata["consumed_evidence_root_ids"] == [
        "trusted-1",
        "trusted-2",
        "untrusted-1",
        "untrusted-2",
    ]
    assert revision_metadata["validation_count"] == 0
    assert revision_metadata["validation_state"] == "unverified"


def test_recovery_keeps_newer_session_generation_idle() -> None:
    """Recovery of an old plan must not strand newer raw memories."""
    service = _make_service()
    service._store_consolidated = MagicMock(return_value=["out-1"])
    service._sessions.get.return_value = {
        "session_id": "ses-1",
        "user_id": "user-1",
        "owner_id": "user-1",
        "session_revision": 2,
    }
    service._sessions.finalize_consolidation.return_value = "idle"
    service._memory.revisions.operations.get.return_value = {"operation_id": "op-1"}
    session = {
        "session_id": "ses-1",
        "user_id": "user-1",
        "owner_id": "user-1",
        "session_revision": 1,
        "memory_ids": ["raw-1"],
        "consolidation_token": "token-1",
        "consolidation_plan": [
            {
                "memory_id": "out-1",
                "derived_from": ["raw-1"],
            }
        ],
    }

    assert service.recover_session(session) is True
    service._sessions.finalize_consolidation.assert_called_once_with(
        "ses-1",
        expected_revision=1,
        token="token-1",
        consolidated_memory_ids=["out-1"],
        point_id=None,
    )


def test_descriptive_memory_types_are_normalized() -> None:
    from mnemory.consolidation import _normalize_consolidation_memory_type

    assert _normalize_consolidation_memory_type("goal") == "episodic"
    assert _normalize_consolidation_memory_type("workflow") == "procedural"
    assert _normalize_consolidation_memory_type("unexpected-label") == "episodic"


def test_legacy_session_point_id_is_diagnosed(caplog) -> None:
    from mnemory.storage.vector import SessionSummaryStore

    store = SessionSummaryStore.__new__(SessionSummaryStore)
    store._client = MagicMock()
    point = MagicMock()
    point.id = "00000000-0000-0000-0000-000000000001"
    point.payload = {
        "session_id": "ses-legacy",
        "user_id": "user-1",
        "memory_ids": ["m1"],
        "consolidation_state": "idle",
    }
    store._client.scroll.return_value = ([point], None)

    with caplog.at_level("WARNING"):
        sessions = store.scan_consolidation_candidates()

    assert sessions[0]["session_id"] == "ses-legacy"
    assert "Legacy session point id detected" in caplog.text


class TestConsolidationPromptWithPrevious:
    """Tests for build_consolidation_prompt with previous_consolidated."""

    def test_prompt_includes_previous_section(self):
        """When previous_consolidated is provided, prompt should include the section."""
        from mnemory.prompts import build_consolidation_prompt

        messages, schema = build_consolidation_prompt(
            summary="Test summary",
            role="user",
            raw_memories=[
                {
                    "id": "m1",
                    "memory": "User likes Python",
                    "metadata": {
                        "memory_type": "preference",
                        "role": "user",
                        "importance": "normal",
                        "categories": [],
                    },
                },
            ],
            previous_consolidated=[
                {
                    "id": "c1",
                    "memory": "User prefers Python for coding",
                    "metadata": {
                        "memory_type": "preference",
                        "role": "user",
                        "importance": "normal",
                    },
                },
            ],
        )

        user_msg = messages[1]["content"]
        assert "Previously Consolidated Memories" in user_msg
        assert "User prefers Python for coding" in user_msg
        assert "Do NOT duplicate" in user_msg

    def test_prompt_without_previous(self):
        """When no previous_consolidated, prompt should not include the section."""
        from mnemory.prompts import build_consolidation_prompt

        messages, schema = build_consolidation_prompt(
            summary="Test summary",
            role="user",
            raw_memories=[
                {
                    "id": "m1",
                    "memory": "User likes Python",
                    "metadata": {
                        "memory_type": "preference",
                        "role": "user",
                        "importance": "normal",
                        "categories": [],
                    },
                },
            ],
        )

        user_msg = messages[1]["content"]
        assert "Previously Consolidated" not in user_msg

    def test_user_role_uses_user_system_prompt(self):
        """role='user' should use the user-focused system prompt."""
        from mnemory.prompts import build_consolidation_prompt

        messages, _ = build_consolidation_prompt(
            summary="Test",
            role="user",
            raw_memories=[
                {
                    "id": "m1",
                    "memory": "User likes X",
                    "metadata": {
                        "memory_type": "preference",
                        "importance": "normal",
                        "categories": [],
                    },
                },
            ],
        )

        system_msg = messages[0]["content"]
        assert "USER" in system_msg
        assert "User decided to" in system_msg or "User prefers" in system_msg

    def test_assistant_role_uses_assistant_system_prompt(self):
        """role='assistant' should use the assistant-focused system prompt."""
        from mnemory.prompts import build_consolidation_prompt

        messages, _ = build_consolidation_prompt(
            summary="Test",
            role="assistant",
            raw_memories=[
                {
                    "id": "m1",
                    "memory": "Assistant implemented X",
                    "metadata": {
                        "memory_type": "episodic",
                        "importance": "high",
                        "categories": [],
                    },
                },
            ],
        )

        system_msg = messages[0]["content"]
        assert "ASSISTANT" in system_msg
        assert (
            "Assistant implemented" in system_msg or "Assistant deployed" in system_msg
        )

    def test_schema_has_no_role_field(self):
        """The output schema should not have a role field (role is implicit)."""
        from mnemory.prompts import build_consolidation_prompt

        _, schema = build_consolidation_prompt(
            summary="Test",
            role="user",
            raw_memories=[
                {
                    "id": "m1",
                    "memory": "Test",
                    "metadata": {
                        "memory_type": "fact",
                        "importance": "normal",
                        "categories": [],
                    },
                },
            ],
        )

        item_props = schema["schema"]["properties"]["memories"]["items"]["properties"]
        assert "role" not in item_props
        item_required = schema["schema"]["properties"]["memories"]["items"]["required"]
        assert "role" not in item_required

    def test_prompt_with_no_raw_memories(self):
        """Prompt should work with empty raw memories (summary-only extraction)."""
        from mnemory.prompts import build_consolidation_prompt

        messages, schema = build_consolidation_prompt(
            summary="User decided to use PostgreSQL for billing. Assistant recommended Redis for caching.",
            role="user",
            raw_memories=[],
        )

        user_msg = messages[1]["content"]
        assert "(no raw memories)" in user_msg
        assert "PostgreSQL" in user_msg
        # System prompt should mention summary-only extraction
        system_msg = messages[0]["content"]
        assert "summary" in system_msg.lower()

    def test_prompt_includes_other_role_context(self):
        """When other_role_consolidated is provided, prompt should include cross-role section."""
        from mnemory.prompts import build_consolidation_prompt

        messages, _ = build_consolidation_prompt(
            summary="Test summary",
            role="assistant",
            raw_memories=[
                {
                    "id": "m1",
                    "memory": "Assistant committed changes as e3a3f72",
                    "metadata": {
                        "memory_type": "episodic",
                        "importance": "normal",
                        "categories": ["technical"],
                    },
                },
            ],
            other_role_consolidated=[
                {"text": "User committed the mnemory changes as commit e3a3f72."},
            ],
        )

        user_msg = messages[1]["content"]
        assert "Already Consolidated" in user_msg
        assert "user role" in user_msg
        assert "e3a3f72" in user_msg
        assert "Do NOT duplicate" in user_msg

    def test_prompt_without_other_role_context(self):
        """When no other_role_consolidated, prompt should not include cross-role section."""
        from mnemory.prompts import build_consolidation_prompt

        messages, _ = build_consolidation_prompt(
            summary="Test summary",
            role="user",
            raw_memories=[
                {
                    "id": "m1",
                    "memory": "User likes Python",
                    "metadata": {
                        "memory_type": "preference",
                        "importance": "normal",
                        "categories": [],
                    },
                },
            ],
        )

        user_msg = messages[1]["content"]
        assert "Already Consolidated" not in user_msg

    def test_prompt_merge_instruction_is_conservative(self):
        """System prompt should instruct to keep separate memories when details differ."""
        from mnemory.prompts import build_consolidation_prompt

        messages, _ = build_consolidation_prompt(
            summary="Test",
            role="user",
            raw_memories=[
                {
                    "id": "m1",
                    "memory": "Test",
                    "metadata": {
                        "memory_type": "fact",
                        "importance": "normal",
                        "categories": [],
                    },
                },
            ],
        )

        system_msg = messages[0]["content"]
        assert "truly redundant" in system_msg
        assert "SEPARATE" in system_msg

    def test_prompt_category_reclassification_instruction(self):
        """System prompt should instruct to reclassify wrong categories."""
        from mnemory.prompts import build_consolidation_prompt

        messages, _ = build_consolidation_prompt(
            summary="Test",
            role="user",
            raw_memories=[
                {
                    "id": "m1",
                    "memory": "Test",
                    "metadata": {
                        "memory_type": "fact",
                        "importance": "normal",
                        "categories": [],
                    },
                },
            ],
        )

        system_msg = messages[0]["content"]
        assert "Reclassify categories" in system_msg
        assert 'use "home" rather than "project:home"' in system_msg

    def test_prompt_distinguishes_recalled_from_new_decision(self):
        """User prompt should avoid turning recalled knowledge into a new decision."""
        from mnemory.prompts import build_consolidation_prompt

        messages, _ = build_consolidation_prompt(
            summary="User recalled a prior pruning plan for the mirobalan tree.",
            role="user",
            raw_memories=[],
        )

        system_msg = messages[0]["content"]
        assert "Distinguish NEW decisions from recalled prior knowledge" in system_msg
        assert "merely recalls or restates" in system_msg

    def test_prompt_prefers_implementation_over_same_session_recommendation(self):
        """Assistant prompt should prefer implementation over same-session recommendation."""
        from mnemory.prompts import build_consolidation_prompt

        messages, _ = build_consolidation_prompt(
            summary="Assistant recommended and then implemented the same fix.",
            role="assistant",
            raw_memories=[],
        )

        system_msg = messages[0]["content"]
        assert "prefer the IMPLEMENTATION memory" in system_msg
        assert "Do not emit both a recommendation and an implementation" in system_msg

    def test_prompt_enforces_one_memory_one_takeaway(self):
        """Prompt should discourage combining multiple durable takeaways."""
        from mnemory.prompts import build_consolidation_prompt

        messages, _ = build_consolidation_prompt(
            summary="User approved the change and stated an ongoing goal.",
            role="user",
            raw_memories=[],
        )

        system_msg = messages[0]["content"]
        assert "One memory = one durable takeaway" in system_msg


class TestConsolidationAlwaysRunsBothPasses:
    """Tests for the always-run-both-passes behavior in consolidation."""

    def test_user_pass_runs_with_no_user_raw_memories(self):
        """User consolidation should run even with 0 user raw memories."""
        import json

        service = _make_service()
        service._config.memory.consolidation_batch_size = 100

        # Mock LLM to return a user decision extracted from summary
        service._llm.generate.return_value = json.dumps(
            {
                "memories": [
                    {
                        "text": "User decided to prune the maple gently",
                        "memory_type": "episodic",
                        "categories": ["home"],
                        "importance": "normal",
                        "pinned": False,
                        "event_date": "2026-03-24",
                    }
                ]
            }
        )

        facts, ids_map = service._consolidate_role(
            session_id="ses-1",
            raw_memories=[],  # No user raw memories
            role="user",
            summary="User decided to prune the maple more gently. Assistant recommended a conservative pruning strategy.",
            artifact_ids=set(),
            previous_consolidated=[],
            session_date="2026-03-24",
        )

        # LLM should have been called (summary-only extraction)
        assert service._llm.generate.called
        assert len(facts) == 1
        assert facts[0]["text"] == "User decided to prune the maple gently"
        assert facts[0]["role"] == "user"

    def test_output_resolves_only_its_declared_source_aliases(self):
        """Each output must retain only the raw revisions that support it."""
        import json

        service = _make_service()
        service._config.memory.consolidation_batch_size = 100
        service._llm.generate.return_value = json.dumps(
            {
                "memories": [
                    {
                        "text": "User likes Python",
                        "memory_type": "preference",
                        "categories": ["technical"],
                        "importance": "normal",
                        "pinned": False,
                        "event_date": None,
                        "source_ids": ["S0"],
                    }
                ]
            }
        )
        raw_memories = [
            {
                "id": "raw-1",
                "memory": "User likes Python",
                "metadata": {"memory_type": "preference"},
            },
            {
                "id": "raw-2",
                "memory": "User lives in Prague",
                "metadata": {"memory_type": "fact"},
            },
        ]

        facts, _ = service._consolidate_role(
            session_id="ses-1",
            raw_memories=raw_memories,
            role="user",
            summary="User likes Python and lives in Prague.",
            artifact_ids=set(),
            previous_consolidated=[],
            session_date="2026-03-24",
        )

        assert facts[0]["derived_from"] == ["raw-1"]

    def test_assistant_pass_receives_user_consolidated_context(self):
        """Assistant consolidation should receive user facts as cross-role context."""
        import json

        service = _make_service()
        service._config.memory.consolidation_batch_size = 100

        service._llm.generate.return_value = json.dumps(
            {
                "memories": [
                    {
                        "text": "Assistant recommended conservative maple pruning",
                        "memory_type": "episodic",
                        "categories": ["home"],
                        "importance": "normal",
                        "pinned": False,
                        "event_date": "2026-03-24",
                    }
                ]
            }
        )

        user_facts = [
            {"text": "User decided to prune the maple gently"},
        ]

        facts, _ = service._consolidate_role(
            session_id="ses-1",
            raw_memories=[
                {
                    "id": "m1",
                    "memory": "Assistant recommended conservative pruning",
                    "metadata": {
                        "memory_type": "episodic",
                        "role": "assistant",
                        "importance": "normal",
                        "categories": ["home"],
                    },
                },
            ],
            role="assistant",
            summary="Test summary",
            artifact_ids=set(),
            previous_consolidated=[],
            session_date="2026-03-24",
            other_role_consolidated=user_facts,
        )

        # Verify the LLM was called with a prompt containing cross-role context
        call_args = service._llm.generate.call_args
        messages = call_args[0][0]
        user_msg = messages[1]["content"]
        assert "Already Consolidated" in user_msg
        assert "User decided to prune the maple gently" in user_msg

    def test_assistant_pass_skipped_without_agent_id(self):
        """When agent_id is None, assistant pass should not run."""
        service = _make_service()
        service._config.memory.consolidation_batch_size = 100

        # Set up session with no agent_id
        service._sessions.get.return_value = {
            "session_id": "ses-1",
            "user_id": "user-1",
            "agent_id": None,
            "memory_ids": ["m1"],
            "summary": "Test summary with enough content to be substantive for the test",
            "created_at": "2026-03-24T00:00:00+00:00",
        }
        service._sessions.list_for_user.return_value = []

        # Only user raw memories
        service._vector.get_by_id.return_value = {
            "id": "m1",
            "memory": "User likes Python",
            "user_id": "user-1",
            "owner_id": "user-1",
            "metadata": {
                "memory_layer": "raw",
                "role": "user",
                "memory_type": "preference",
                "importance": "normal",
                "categories": ["technical"],
            },
        }

        import json

        service._llm.generate.return_value = json.dumps({"memories": []})
        service._memory.add_memory.return_value = {"results": []}

        service.consolidate_session("ses-1")

        # LLM should be called exactly once (user pass only, no assistant pass)
        assert service._llm.generate.call_count == 1
