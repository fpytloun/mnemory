"""Tests for the data migration framework."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mnemory.migration import (
    _TEMP_COLLECTION,
    META_COLLECTION,
    AddSparseVectorsMigration,
    BackfillOwnerIdMigration,
    ClassifyLegacyFailedSessionsMigration,
    Migration,
    MigrationRunner,
    get_migrations,
)

# ── Helpers ───────────────────────────────────────────────────────────


class DummyMigration(Migration):
    """Simple migration for testing the runner."""

    id = "test_001"
    description = "Test migration"

    def __init__(self):
        self.run_count = 0
        self.last_progress = None

    def run(self, client, *, progress, state_callback, state):
        self.run_count += 1
        self.last_progress = progress


class FailingMigration(Migration):
    """Migration that always raises."""

    id = "test_fail"
    description = "Failing migration"

    def run(self, client, *, progress, state_callback, state):
        raise RuntimeError("Migration failed")


def test_legacy_failed_session_migration_is_classification_only() -> None:
    migration = ClassifyLegacyFailedSessionsMigration()
    client = MagicMock()
    point = MagicMock()
    point.id = "session-1"
    client.scroll.side_effect = [([point], None)]
    state = {}

    migration.run(
        client,
        progress=None,
        state_callback=MagicMock(),
        state=state,
    )

    payload = client.set_payload.call_args.kwargs["payload"]
    assert payload == {
        "failure_schema_version": 1,
        "legacy_failure": True,
        "failure_class": "legacy_unknown",
    }
    assert "consolidation_state" not in payload
    assert "attempt_count" not in payload
    assert client.set_payload.call_args.kwargs["points"] == ["session-1"]


def _mock_client(existing_collections=None, state_payload=None):
    """Create a mock Qdrant client with configurable state."""
    client = MagicMock()

    # Mock get_collections
    collections = []
    if existing_collections:
        for name in existing_collections:
            col = MagicMock()
            col.name = name
            collections.append(col)
    client.get_collections.return_value.collections = collections

    # Mock retrieve for state loading
    if state_payload is not None:
        point = MagicMock()
        point.payload = state_payload
        client.retrieve.return_value = [point]
    else:
        client.retrieve.return_value = []

    return client


# ── MigrationRunner ──────────────────────────────────────────────────


class TestMigrationRunnerMetaCollection:
    """Test _mnemory_meta collection creation."""

    def test_creates_meta_collection_if_missing(self):
        """Should create _mnemory_meta when it doesn't exist."""
        client = _mock_client(existing_collections=[])
        runner = MigrationRunner(client, [])
        runner.run_pending()
        client.create_collection.assert_called_once()
        args = client.create_collection.call_args
        assert args[1]["collection_name"] == META_COLLECTION

    def test_skips_creation_if_meta_exists(self):
        """Should not create _mnemory_meta if it already exists."""
        client = _mock_client(existing_collections=[META_COLLECTION])
        runner = MigrationRunner(client, [])
        runner.run_pending()
        client.create_collection.assert_not_called()

    def test_skips_creation_with_other_collections(self):
        """Should create meta even when other collections exist."""
        client = _mock_client(existing_collections=["memories"])
        runner = MigrationRunner(client, [])
        runner.run_pending()
        client.create_collection.assert_called_once()


class TestMigrationRunnerStateManagement:
    """Test state loading and saving."""

    def test_loads_empty_state_when_no_point(self):
        """Should return empty state when no meta point exists."""
        client = _mock_client(existing_collections=[META_COLLECTION])
        migration = DummyMigration()
        runner = MigrationRunner(client, [migration])
        runner.run_pending()
        # Migration should have run (no applied state)
        assert migration.run_count == 1

    def test_loads_existing_state(self):
        """Should load state from existing meta point."""
        client = _mock_client(
            existing_collections=[META_COLLECTION],
            state_payload={"applied": ["test_001"]},
        )
        migration = DummyMigration()
        runner = MigrationRunner(client, [migration])
        runner.run_pending()
        # Migration should be skipped (already applied)
        assert migration.run_count == 0

    def test_saves_state_after_migration(self):
        """Should persist state after successful migration."""
        client = _mock_client(existing_collections=[META_COLLECTION])
        migration = DummyMigration()
        runner = MigrationRunner(client, [migration])
        runner.run_pending()

        # Should have called upsert at least twice:
        # 1. Mark as running
        # 2. Mark as complete
        assert client.upsert.call_count >= 2

        # Last upsert should contain the migration ID in applied
        last_call = client.upsert.call_args
        point = last_call[1]["points"][0]
        assert migration.id in point.payload["applied"]
        assert "last_run" in point.payload


class TestMigrationRunnerExecution:
    """Test migration execution logic."""

    def test_runs_pending_migration(self):
        """Should run a migration that hasn't been applied."""
        client = _mock_client(existing_collections=[META_COLLECTION])
        migration = DummyMigration()
        runner = MigrationRunner(client, [migration])
        runner.run_pending()
        assert migration.run_count == 1

    def test_skips_applied_migration(self):
        """Should skip a migration that's already applied."""
        client = _mock_client(
            existing_collections=[META_COLLECTION],
            state_payload={"applied": ["test_001"]},
        )
        migration = DummyMigration()
        runner = MigrationRunner(client, [migration])
        runner.run_pending()
        assert migration.run_count == 0

    def test_runs_multiple_in_order(self):
        """Should run multiple migrations in order."""
        client = _mock_client(existing_collections=[META_COLLECTION])
        m1 = DummyMigration()
        m1.id = "test_001"
        m2 = DummyMigration()
        m2.id = "test_002"
        runner = MigrationRunner(client, [m1, m2])
        runner.run_pending()
        assert m1.run_count == 1
        assert m2.run_count == 1

    def test_skips_first_runs_second(self):
        """Should skip applied migration but run the next one."""
        client = _mock_client(
            existing_collections=[META_COLLECTION],
            state_payload={"applied": ["test_001"]},
        )
        m1 = DummyMigration()
        m1.id = "test_001"
        m2 = DummyMigration()
        m2.id = "test_002"
        runner = MigrationRunner(client, [m1, m2])
        runner.run_pending()
        assert m1.run_count == 0
        assert m2.run_count == 1

    def test_reruns_interrupted_migration(self):
        """Should re-run a migration that was interrupted (running marker)."""
        client = _mock_client(
            existing_collections=[META_COLLECTION],
            state_payload={"applied": ["test_001:running"]},
        )
        migration = DummyMigration()
        runner = MigrationRunner(client, [migration])
        runner.run_pending()
        # Should re-run since it was interrupted
        assert migration.run_count == 1

    def test_passes_progress_checkpoint(self):
        """Should pass progress checkpoint to resumed migration."""
        checkpoint = {"offset": "abc-123", "processed": 50}
        client = _mock_client(
            existing_collections=[META_COLLECTION],
            state_payload={
                "applied": ["test_001:running"],
                "test_001_progress": checkpoint,
            },
        )
        migration = DummyMigration()
        runner = MigrationRunner(client, [migration])
        runner.run_pending()
        assert migration.last_progress == checkpoint

    def test_no_progress_for_fresh_migration(self):
        """Fresh migration should receive None progress."""
        client = _mock_client(existing_collections=[META_COLLECTION])
        migration = DummyMigration()
        runner = MigrationRunner(client, [migration])
        runner.run_pending()
        assert migration.last_progress is None


class TestMigrationRunnerFailure:
    """Test failure handling."""

    def test_failure_removes_running_marker(self):
        """Failed migration should remove running marker from state."""
        client = _mock_client(existing_collections=[META_COLLECTION])
        migration = FailingMigration()

        runner = MigrationRunner(client, [migration])
        with pytest.raises(RuntimeError, match="Migration failed"):
            runner.run_pending()

        # Last upsert should NOT contain the running marker
        last_call = client.upsert.call_args
        point = last_call[1]["points"][0]
        assert "test_fail:running" not in point.payload.get("applied", [])
        assert "test_fail" not in point.payload.get("applied", [])

    def test_failure_propagates_exception(self):
        """Migration failure should propagate the exception."""
        client = _mock_client(existing_collections=[META_COLLECTION])
        migration = FailingMigration()
        runner = MigrationRunner(client, [migration])
        with pytest.raises(RuntimeError, match="Migration failed"):
            runner.run_pending()

    def test_failure_stops_subsequent_migrations(self):
        """Failure in one migration should stop subsequent ones."""
        client = _mock_client(existing_collections=[META_COLLECTION])
        m1 = FailingMigration()
        m2 = DummyMigration()
        m2.id = "test_002"
        runner = MigrationRunner(client, [m1, m2])
        with pytest.raises(RuntimeError):
            runner.run_pending()
        assert m2.run_count == 0


class TestMigrationRunnerOptimisticLock:
    """Test optimistic locking for multi-pod safety."""

    def test_running_marker_set_before_execution(self):
        """Should set running marker before executing migration."""
        client = _mock_client(existing_collections=[META_COLLECTION])
        upsert_payloads = []

        def capture_upsert(**kwargs):
            point = kwargs["points"][0]
            upsert_payloads.append(dict(point.payload))

        client.upsert.side_effect = capture_upsert

        migration = DummyMigration()
        runner = MigrationRunner(client, [migration])
        runner.run_pending()

        # First upsert should have the running marker
        assert "test_001:running" in upsert_payloads[0]["applied"]
        # Last upsert should have the final ID (not running)
        assert "test_001" in upsert_payloads[-1]["applied"]
        assert "test_001:running" not in upsert_payloads[-1]["applied"]


# ── AddSparseVectorsMigration helpers ────────────────────────────────


def _sparse_collection_info(points_count: int = 0):
    """Create a mock collection info WITH sparse config (fast path)."""
    from qdrant_client.models import Distance, VectorParams

    info = MagicMock()
    info.points_count = points_count
    info.config.params.sparse_vectors = {"bm25": MagicMock()}
    info.config.params.vectors = VectorParams(size=1536, distance=Distance.COSINE)
    return info


def _dense_only_collection_info(points_count: int = 0):
    """Create a mock collection info WITHOUT sparse config (recreation path)."""
    from qdrant_client.models import Distance, VectorParams

    info = MagicMock()
    info.points_count = points_count
    info.config.params.sparse_vectors = {}
    info.config.params.vectors = VectorParams(size=1536, distance=Distance.COSINE)
    return info


def _mock_sparse():
    """Create a mock sparse client that's available."""
    sparse = MagicMock()
    sparse.available = True
    return sparse


# ── AddSparseVectorsMigration: fast path ─────────────────────────────


class TestAddSparseVectorsMigration:
    """Test the BM25 sparse vector migration (fast path).

    These tests cover the case where update_collection() succeeds and
    the collection has sparse config — sparse vectors are added to
    existing points via update_vectors().
    """

    def test_skips_when_sparse_unavailable(self):
        """Should skip gracefully when fastembed is not installed."""
        client = MagicMock()
        sparse = MagicMock()
        sparse.available = False

        migration = AddSparseVectorsMigration("memories", sparse)
        migration.run(
            client,
            progress=None,
            state_callback=MagicMock(),
            state={},
        )

        # Should not call update_collection or scroll
        client.update_collection.assert_not_called()
        client.scroll.assert_not_called()

    def test_skips_when_sparse_is_none(self):
        """Should skip gracefully when sparse client is None."""
        client = MagicMock()
        migration = AddSparseVectorsMigration("memories", None)
        migration.run(
            client,
            progress=None,
            state_callback=MagicMock(),
            state={},
        )
        client.update_collection.assert_not_called()

    def test_adds_sparse_config_to_collection(self):
        """Should call update_collection with sparse vector config."""
        client = MagicMock()
        sparse = _mock_sparse()

        # update_collection succeeds → sparse config exists
        client.get_collection.return_value = _sparse_collection_info(0)

        migration = AddSparseVectorsMigration("memories", sparse)
        migration.run(
            client,
            progress=None,
            state_callback=MagicMock(),
            state={},
        )

        client.update_collection.assert_called_once()
        kwargs = client.update_collection.call_args[1]
        assert kwargs["collection_name"] == "memories"
        assert "bm25" in kwargs["sparse_vectors_config"]

    def test_skips_empty_collection(self):
        """Should skip scrolling when collection is empty."""
        client = MagicMock()
        sparse = _mock_sparse()

        client.get_collection.return_value = _sparse_collection_info(0)

        migration = AddSparseVectorsMigration("memories", sparse)
        migration.run(
            client,
            progress=None,
            state_callback=MagicMock(),
            state={},
        )

        client.scroll.assert_not_called()
        sparse.embed_batch.assert_not_called()

    def test_processes_single_batch(self):
        """Should process a single batch of points."""
        from qdrant_client.models import SparseVector

        client = MagicMock()
        sparse = _mock_sparse()

        client.get_collection.return_value = _sparse_collection_info(2)

        # Mock scroll: one batch, then done
        point1 = MagicMock()
        point1.id = "uuid-1"
        point1.payload = {"data": "User likes cats"}
        point2 = MagicMock()
        point2.id = "uuid-2"
        point2.payload = {"data": "User works at Acme"}
        client.scroll.return_value = ([point1, point2], None)

        # Mock sparse embeddings
        sv1 = SparseVector(indices=[1, 2], values=[0.5, 0.3])
        sv2 = SparseVector(indices=[3, 4], values=[0.7, 0.1])
        sparse.embed_batch.return_value = [sv1, sv2]

        state = {}
        state_callback = MagicMock()

        migration = AddSparseVectorsMigration("memories", sparse)
        migration.run(
            client,
            progress=None,
            state_callback=state_callback,
            state=state,
        )

        # Should have scrolled once
        client.scroll.assert_called_once()
        # Should have embedded the texts
        sparse.embed_batch.assert_called_once_with(
            ["User likes cats", "User works at Acme"]
        )
        # Should have updated vectors
        client.update_vectors.assert_called_once()
        point_vectors = client.update_vectors.call_args[1]["points"]
        assert len(point_vectors) == 2
        assert point_vectors[0].id == "uuid-1"
        assert point_vectors[1].id == "uuid-2"

    def test_processes_multiple_batches(self):
        """Should process multiple batches with pagination."""
        from qdrant_client.models import SparseVector

        client = MagicMock()
        sparse = _mock_sparse()

        client.get_collection.return_value = _sparse_collection_info(3)

        # Mock scroll: two batches
        p1 = MagicMock()
        p1.id = "uuid-1"
        p1.payload = {"data": "text 1"}
        p2 = MagicMock()
        p2.id = "uuid-2"
        p2.payload = {"data": "text 2"}
        p3 = MagicMock()
        p3.id = "uuid-3"
        p3.payload = {"data": "text 3"}

        client.scroll.side_effect = [
            ([p1, p2], "next-offset"),
            ([p3], None),
        ]

        sv = SparseVector(indices=[1], values=[0.5])
        sparse.embed_batch.return_value = [sv, sv]

        state = {}
        migration = AddSparseVectorsMigration("memories", sparse)
        migration.run(
            client,
            progress=None,
            state_callback=MagicMock(),
            state=state,
        )

        assert client.scroll.call_count == 2
        assert client.update_vectors.call_count == 2

    def test_saves_checkpoint_after_each_batch(self):
        """Should save progress checkpoint after each batch."""
        from qdrant_client.models import SparseVector

        client = MagicMock()
        sparse = _mock_sparse()

        client.get_collection.return_value = _sparse_collection_info(2)

        p1 = MagicMock()
        p1.id = "uuid-1"
        p1.payload = {"data": "text"}
        client.scroll.side_effect = [
            ([p1], "next-offset"),
            ([], None),
        ]

        sv = SparseVector(indices=[1], values=[0.5])
        sparse.embed_batch.return_value = [sv]

        state = {}
        state_callback = MagicMock()

        migration = AddSparseVectorsMigration("memories", sparse)
        migration.run(
            client,
            progress=None,
            state_callback=state_callback,
            state=state,
        )

        # State callback should have been called with progress
        assert state_callback.call_count >= 1
        progress_key = f"{migration.id}_progress"
        assert progress_key in state
        # After completion, the last checkpoint should have offset=None
        last_progress = state[progress_key]
        assert last_progress["processed"] >= 1

    def test_resumes_from_checkpoint(self):
        """Should resume from checkpoint offset."""
        from qdrant_client.models import SparseVector

        client = MagicMock()
        sparse = _mock_sparse()

        client.get_collection.return_value = _sparse_collection_info(10)

        p1 = MagicMock()
        p1.id = "uuid-5"
        p1.payload = {"data": "text"}
        client.scroll.return_value = ([p1], None)

        sv = SparseVector(indices=[1], values=[0.5])
        sparse.embed_batch.return_value = [sv]

        state = {}
        progress = {"offset": "resume-offset", "processed": 4}

        migration = AddSparseVectorsMigration("memories", sparse)
        migration.run(
            client,
            progress=progress,
            state_callback=MagicMock(),
            state=state,
        )

        # Should have scrolled starting from the resume offset
        scroll_kwargs = client.scroll.call_args[1]
        assert scroll_kwargs["offset"] == "resume-offset"

    def test_handles_missing_payload_data(self):
        """Should handle points with missing or empty data field."""
        from qdrant_client.models import SparseVector

        client = MagicMock()
        sparse = _mock_sparse()

        client.get_collection.return_value = _sparse_collection_info(2)

        p1 = MagicMock()
        p1.id = "uuid-1"
        p1.payload = {}  # No "data" field
        p2 = MagicMock()
        p2.id = "uuid-2"
        p2.payload = None  # None payload
        client.scroll.return_value = ([p1, p2], None)

        sv = SparseVector(indices=[1], values=[0.5])
        sparse.embed_batch.return_value = [sv, sv]

        migration = AddSparseVectorsMigration("memories", sparse)
        migration.run(
            client,
            progress=None,
            state_callback=MagicMock(),
            state={},
        )

        # Should have embedded empty strings for missing data
        sparse.embed_batch.assert_called_once_with(["", ""])

    def test_skips_update_when_no_sparse_vectors(self):
        """Should skip update_vectors when embed_batch returns None."""
        client = MagicMock()
        sparse = _mock_sparse()

        client.get_collection.return_value = _sparse_collection_info(1)

        p1 = MagicMock()
        p1.id = "uuid-1"
        p1.payload = {"data": "text"}
        client.scroll.return_value = ([p1], None)

        # embed_batch returns None (shouldn't happen, but defensive)
        sparse.embed_batch.return_value = None

        migration = AddSparseVectorsMigration("memories", sparse)
        migration.run(
            client,
            progress=None,
            state_callback=MagicMock(),
            state={},
        )

        client.update_vectors.assert_not_called()


# ── AddSparseVectorsMigration: recreation path ──────────────────────


class TestAddSparseVectorsMigrationRecreation:
    """Test collection recreation when update_collection() can't add sparse config.

    This covers the case where remote Qdrant rejects adding new named
    vectors to an existing collection (server limitation ≤1.16).
    """

    def _make_client_no_sparse(self, points_count: int = 2):
        """Create a mock client that rejects update_collection for sparse config.

        Simulates remote Qdrant where:
        - update_collection() raises (can't add new vector names)
        - get_collection() returns no sparse config
        - Collection has points that need migrating
        """
        import httpx
        from qdrant_client.http.exceptions import UnexpectedResponse
        from qdrant_client.models import SparseVector

        client = MagicMock()

        # update_collection raises (remote Qdrant limitation)
        client.update_collection.side_effect = UnexpectedResponse(
            status_code=400,
            reason_phrase="Bad Request",
            content=b'{"status":{"error":"Wrong input"}}',
            headers=httpx.Headers(),
        )

        # get_collection returns different info for different collections
        original_info = _dense_only_collection_info(points_count)
        temp_info = _sparse_collection_info(points_count)

        def get_collection_side_effect(name):
            if name == _TEMP_COLLECTION:
                return temp_info
            return original_info

        client.get_collection.side_effect = get_collection_side_effect

        # get_collections: original exists, temp does not (initially)
        orig_col = MagicMock()
        orig_col.name = "memories"
        client.get_collections.return_value.collections = [orig_col]

        # Mock scroll: return points with dense vectors
        p1 = MagicMock()
        p1.id = "uuid-1"
        p1.payload = {"data": "User likes cats"}
        p1.vector = [0.1] * 1536
        p2 = MagicMock()
        p2.id = "uuid-2"
        p2.payload = {"data": "User works at Acme"}
        p2.vector = [0.2] * 1536

        points = [p1, p2][:points_count]
        client.scroll.return_value = (points, None)

        # Mock sparse embeddings
        sv = SparseVector(indices=[1, 2], values=[0.5, 0.3])
        sparse = _mock_sparse()
        sparse.embed_batch.return_value = [sv] * points_count

        return client, sparse

    def test_detects_missing_sparse_config(self):
        """Should detect when sparse config is missing after update_collection."""
        client, sparse = self._make_client_no_sparse(0)

        # Empty collection — should detect missing sparse and skip
        original_info = _dense_only_collection_info(0)
        client.get_collection.side_effect = None
        client.get_collection.return_value = original_info

        migration = AddSparseVectorsMigration("memories", sparse)
        migration.run(
            client,
            progress=None,
            state_callback=MagicMock(),
            state={},
        )

        # update_collection was attempted
        client.update_collection.assert_called_once()
        # No update_vectors (fast path not used — no sparse config)
        client.update_vectors.assert_not_called()

    def test_recreation_creates_temp_collection(self):
        """Should create temp collection with dense + sparse config."""
        client, sparse = self._make_client_no_sparse(2)

        migration = AddSparseVectorsMigration("memories", sparse)
        migration.run(
            client,
            progress=None,
            state_callback=MagicMock(),
            state={},
        )

        # Should have created temp collection
        create_calls = client.create_collection.call_args_list
        temp_creates = [
            c for c in create_calls if c[1].get("collection_name") == _TEMP_COLLECTION
        ]
        assert len(temp_creates) >= 1
        # Temp collection should have sparse config
        assert "bm25" in temp_creates[0][1]["sparse_vectors_config"]

    def test_recreation_copies_points_to_temp(self):
        """Should copy all points from original to temp with sparse vectors."""
        client, sparse = self._make_client_no_sparse(2)

        migration = AddSparseVectorsMigration("memories", sparse)
        migration.run(
            client,
            progress=None,
            state_callback=MagicMock(),
            state={},
        )

        # Should have upserted points into temp collection
        upsert_calls = client.upsert.call_args_list
        temp_upserts = [
            c for c in upsert_calls if c[1].get("collection_name") == _TEMP_COLLECTION
        ]
        assert len(temp_upserts) >= 1
        # Points should have been upserted
        points = temp_upserts[0][1]["points"]
        assert len(points) == 2
        assert points[0].id == "uuid-1"
        assert points[1].id == "uuid-2"

    def test_recreation_deletes_original(self):
        """Should delete the original collection during recreation."""
        client, sparse = self._make_client_no_sparse(2)

        migration = AddSparseVectorsMigration("memories", sparse)
        migration.run(
            client,
            progress=None,
            state_callback=MagicMock(),
            state={},
        )

        # Should have deleted original collection
        delete_calls = client.delete_collection.call_args_list
        assert any(
            c.args == ("memories",) or c[0] == ("memories",) for c in delete_calls
        )

    def test_recreation_recreates_original_with_sparse(self):
        """Should recreate original collection with sparse config."""
        client, sparse = self._make_client_no_sparse(2)

        migration = AddSparseVectorsMigration("memories", sparse)
        migration.run(
            client,
            progress=None,
            state_callback=MagicMock(),
            state={},
        )

        # Should have created original collection with sparse config
        create_calls = client.create_collection.call_args_list
        orig_creates = [
            c for c in create_calls if c[1].get("collection_name") == "memories"
        ]
        assert len(orig_creates) >= 1
        assert "bm25" in orig_creates[0][1]["sparse_vectors_config"]

    def test_recreation_copies_back_from_temp(self):
        """Should copy points from temp back to recreated original."""
        client, sparse = self._make_client_no_sparse(2)

        migration = AddSparseVectorsMigration("memories", sparse)
        migration.run(
            client,
            progress=None,
            state_callback=MagicMock(),
            state={},
        )

        # Should have upserted into original collection (copy back)
        upsert_calls = client.upsert.call_args_list
        orig_upserts = [
            c for c in upsert_calls if c[1].get("collection_name") == "memories"
        ]
        assert len(orig_upserts) >= 1

    def test_recreation_deletes_temp_on_success(self):
        """Should delete temp collection after successful recreation."""
        client, sparse = self._make_client_no_sparse(2)

        migration = AddSparseVectorsMigration("memories", sparse)
        migration.run(
            client,
            progress=None,
            state_callback=MagicMock(),
            state={},
        )

        # Should have deleted temp collection
        delete_calls = client.delete_collection.call_args_list
        assert any(
            c.args == (_TEMP_COLLECTION,) or c[0] == (_TEMP_COLLECTION,)
            for c in delete_calls
        )

    def test_recreation_saves_phase_checkpoints(self):
        """Should save phase checkpoints during recreation."""
        client, sparse = self._make_client_no_sparse(2)

        state = {}
        state_callback = MagicMock()

        migration = AddSparseVectorsMigration("memories", sparse)
        migration.run(
            client,
            progress=None,
            state_callback=state_callback,
            state=state,
        )

        # State callback should have been called multiple times
        assert state_callback.call_count >= 2
        # Should have saved phase transitions
        progress_key = f"{migration.id}_progress"
        assert progress_key in state

    def test_recreation_aborts_on_count_mismatch(self):
        """Should abort if temp collection has fewer points than original."""
        client, sparse = self._make_client_no_sparse(2)

        # Make temp collection report fewer points than original
        original_info = _dense_only_collection_info(2)
        temp_info = _sparse_collection_info(1)  # Only 1 point copied

        def get_collection_side_effect(name):
            if name == _TEMP_COLLECTION:
                return temp_info
            return original_info

        client.get_collection.side_effect = get_collection_side_effect

        migration = AddSparseVectorsMigration("memories", sparse)
        with pytest.raises(RuntimeError, match="temp collection has 1"):
            migration.run(
                client,
                progress=None,
                state_callback=MagicMock(),
                state={},
            )

        # Should NOT have deleted original (safety)
        delete_calls = [
            c
            for c in client.delete_collection.call_args_list
            if c.args == ("memories",) or c[0] == ("memories",)
        ]
        assert len(delete_calls) == 0

    def test_recreation_resumes_copy_to_temp(self):
        """Should resume copy-to-temp phase from checkpoint."""
        client, sparse = self._make_client_no_sparse(2)

        # Simulate resuming from copy_to_temp phase
        progress = {
            "phase": "copy_to_temp",
            "offset": "resume-offset",
            "processed": 1,
        }

        migration = AddSparseVectorsMigration("memories", sparse)
        migration.run(
            client,
            progress=progress,
            state_callback=MagicMock(),
            state={},
        )

        # Should have scrolled from the resume offset
        scroll_calls = client.scroll.call_args_list
        assert len(scroll_calls) >= 1
        # First scroll should use the resume offset
        first_scroll = scroll_calls[0]
        assert first_scroll[1].get("offset") == "resume-offset"

    def test_recreation_resumes_copy_back(self):
        """Should resume copy-back phase from checkpoint."""
        client, sparse = self._make_client_no_sparse(2)

        # For copy_back phase, temp collection must exist and have points
        temp_col = MagicMock()
        temp_col.name = _TEMP_COLLECTION
        orig_col = MagicMock()
        orig_col.name = "memories"
        client.get_collections.return_value.collections = [temp_col]

        # Simulate resuming from copy_back phase
        progress = {
            "phase": "copy_back",
            "offset": None,
            "processed": 0,
        }

        migration = AddSparseVectorsMigration("memories", sparse)
        migration.run(
            client,
            progress=progress,
            state_callback=MagicMock(),
            state={},
        )

        # Should have created original collection (it was deleted in phase 1)
        create_calls = [
            c
            for c in client.create_collection.call_args_list
            if c[1].get("collection_name") == "memories"
        ]
        assert len(create_calls) >= 1

    def test_recreation_skips_existing_temp(self):
        """Should reuse existing temp collection (from interrupted migration)."""
        client, sparse = self._make_client_no_sparse(2)

        # Temp collection already exists
        temp_col = MagicMock()
        temp_col.name = _TEMP_COLLECTION
        orig_col = MagicMock()
        orig_col.name = "memories"
        client.get_collections.return_value.collections = [orig_col, temp_col]

        migration = AddSparseVectorsMigration("memories", sparse)
        migration.run(
            client,
            progress=None,
            state_callback=MagicMock(),
            state={},
        )

        # Should NOT have created temp collection (it already exists)
        temp_creates = [
            c
            for c in client.create_collection.call_args_list
            if c[1].get("collection_name") == _TEMP_COLLECTION
        ]
        assert len(temp_creates) == 0

    def test_recreation_generates_sparse_vectors(self):
        """Should generate sparse vectors during copy to temp."""
        client, sparse = self._make_client_no_sparse(2)

        migration = AddSparseVectorsMigration("memories", sparse)
        migration.run(
            client,
            progress=None,
            state_callback=MagicMock(),
            state={},
        )

        # Should have called embed_batch with the memory texts
        sparse.embed_batch.assert_called_with(["User likes cats", "User works at Acme"])

    def test_fast_path_when_update_succeeds(self):
        """Should use fast path when update_collection succeeds."""
        from qdrant_client.models import SparseVector

        client = MagicMock()
        sparse = _mock_sparse()

        # update_collection succeeds (no exception)
        # get_collection returns sparse config
        client.get_collection.return_value = _sparse_collection_info(1)

        p1 = MagicMock()
        p1.id = "uuid-1"
        p1.payload = {"data": "text"}
        client.scroll.return_value = ([p1], None)

        sv = SparseVector(indices=[1], values=[0.5])
        sparse.embed_batch.return_value = [sv]

        migration = AddSparseVectorsMigration("memories", sparse)
        migration.run(
            client,
            progress=None,
            state_callback=MagicMock(),
            state={},
        )

        # Should use update_vectors (fast path), not upsert (recreation)
        client.update_vectors.assert_called_once()
        # Should NOT create temp collection
        temp_creates = [
            c
            for c in client.create_collection.call_args_list
            if c[1].get("collection_name") == _TEMP_COLLECTION
        ]
        assert len(temp_creates) == 0


# ── get_migrations registry ──────────────────────────────────────────


class TestGetMigrations:
    """Test the migration registry."""

    def test_returns_list(self):
        """Should return a list of migrations."""
        sparse = MagicMock()
        result = get_migrations("memories", sparse)
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_first_migration_is_sparse_vectors(self):
        """First migration should be AddSparseVectorsMigration."""
        sparse = MagicMock()
        result = get_migrations("memories", sparse)
        assert isinstance(result[0], AddSparseVectorsMigration)
        assert result[0].id == "001_add_sparse_vectors"

    def test_passes_collection_name(self):
        """Should pass collection name to migrations."""
        sparse = MagicMock()
        result = get_migrations("custom_collection", sparse)
        assert result[0]._collection_name == "custom_collection"

    def test_passes_sparse_client(self):
        """Should pass sparse client to migrations."""
        sparse = MagicMock()
        result = get_migrations("memories", sparse)
        assert result[0]._sparse is sparse

    def test_includes_owner_id_backfill_migration(self):
        """Registry should include the owner_id backfill migration."""
        sparse = MagicMock()
        result = get_migrations("memories", sparse)
        assert any(
            isinstance(migration, BackfillOwnerIdMigration) for migration in result
        )


class TestBackfillOwnerIdMigration:
    """Tests for legacy owner_id backfill."""

    def test_backfills_owner_id_from_user_id(self):
        client = _mock_client(existing_collections=[META_COLLECTION])
        point_a = MagicMock()
        point_a.id = "mem-a"
        point_a.payload = {"user_id": "alice@example.com"}
        point_b = MagicMock()
        point_b.id = "mem-b"
        point_b.payload = {"user_id": "bob@example.com"}
        client.scroll.side_effect = [([point_a, point_b], None)]

        migration = BackfillOwnerIdMigration("memories")
        state_callback = MagicMock()
        state: dict[str, object] = {}

        migration.run(
            client,
            progress=None,
            state_callback=state_callback,
            state=state,
        )

        assert client.set_payload.call_count == 2
        first = client.set_payload.call_args_list[0].kwargs
        second = client.set_payload.call_args_list[1].kwargs
        assert first["payload"] == {"owner_id": "alice@example.com"}
        assert first["points"] == ["mem-a"]
        assert second["payload"] == {"owner_id": "bob@example.com"}
        assert second["points"] == ["mem-b"]
        assert state_callback.called

    def test_skips_points_without_valid_user_id(self):
        client = _mock_client(existing_collections=[META_COLLECTION])
        point = MagicMock()
        point.id = "mem-invalid"
        point.payload = {"user_id": "   "}
        client.scroll.side_effect = [([point], None)]

        migration = BackfillOwnerIdMigration("memories")
        migration.run(
            client,
            progress=None,
            state_callback=MagicMock(),
            state={},
        )

        client.set_payload.assert_not_called()

    def test_backfill_paginates_and_updates_progress(self):
        client = _mock_client(existing_collections=[META_COLLECTION])
        first = MagicMock()
        first.id = "mem-a"
        first.payload = {"user_id": "alice@example.com"}
        second = MagicMock()
        second.id = "mem-b"
        second.payload = {"user_id": "bob@example.com"}
        client.scroll.side_effect = [([first], "next-token"), ([second], None)]

        migration = BackfillOwnerIdMigration("memories")
        state_callback = MagicMock()
        state: dict[str, object] = {}

        migration.run(
            client,
            progress=None,
            state_callback=state_callback,
            state=state,
        )

        assert client.scroll.call_count == 2
        assert client.scroll.call_args_list[1].kwargs["offset"] == "next-token"
        assert client.set_payload.call_count == 2
        assert state["004_backfill_owner_id_progress"] == {
            "offset": None,
            "processed": 2,
            "updated": 2,
        }
        assert state_callback.call_count == 2
