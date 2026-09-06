"""Tests for immutable Qdrant memory revisions."""

from __future__ import annotations

import hashlib
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from mnemory.memory import MemoryService
from mnemory.revisions import (
    _TRANSITION_FIELDS,
    EVIDENCE_OPERATION_KIND,
    OPERATIONS_COLLECTION,
    EvidenceClaimActiveError,
    EvidenceConflictError,
    EvidenceLeaseLostError,
    RevisionConflictError,
    RevisionOperationStore,
    RevisionService,
    _operation_point_id,
    canonical_fingerprint,
)

MEMORY_ID = "11111111-1111-4111-8111-111111111111"


def _service() -> RevisionService:
    client = QdrantClient(location=":memory:")
    client.create_collection(
        collection_name="memories",
        vectors_config=VectorParams(size=2, distance=Distance.COSINE),
    )
    vector = SimpleNamespace(
        _client=client,
        _config=SimpleNamespace(vector=SimpleNamespace(is_remote=False)),
        collection_name="memories",
        embedding=MagicMock(embed=MagicMock(return_value=[0.2, 0.8])),
        _point_to_memory=lambda point: {
            "id": str(point.id),
            "memory": (point.payload or {}).get("data", ""),
            "user_id": (point.payload or {}).get("user_id"),
            "owner_id": (point.payload or {}).get("owner_id"),
            "agent_id": (point.payload or {}).get("agent_id"),
            "metadata": dict(point.payload or {}),
        },
    )
    return RevisionService(vector)


def _insert(
    service: RevisionService,
    *,
    memory_id: str = MEMORY_ID,
    agent_id: str | None = None,
) -> None:
    payload = {
        "data": "old",
        "hash": "old-hash",
        "user_id": "user-1",
        "owner_id": "owner-1",
        "role": "user",
        "memory_type": "fact",
        "categories": ["personal"],
        "importance": "normal",
        "pinned": False,
        "artifacts": [{"id": "artifact-1"}],
        **RevisionService.initial_metadata(memory_id),
    }
    if agent_id:
        payload["agent_id"] = agent_id
    service._client.upsert(
        collection_name="memories",
        points=[PointStruct(id=memory_id, vector=[1.0, 0.0], payload=payload)],
        wait=True,
    )


def test_mark_source_requires_the_planned_revision() -> None:
    service = _service()
    _insert(service)

    with pytest.raises(RevisionConflictError):
        service.mark_source(
            [MEMORY_ID],
            operation_id="fsck:check-1:issue-1",
            user_id="user-1",
            owner_id="owner-1",
            expected_revisions={MEMORY_ID: 2},
        )

    point = service._read_point(MEMORY_ID)
    assert point is not None
    assert point.payload["revision_state"] == "active"


def test_mark_source_verifies_its_transition_identity() -> None:
    service = _service()
    _insert(service)
    guard = MagicMock()

    service.mark_source(
        [MEMORY_ID],
        operation_id="fsck:check-1:issue-1",
        user_id="user-1",
        owner_id="owner-1",
        expected_revisions={MEMORY_ID: 1},
        mutation_guard=guard,
    )

    point = service._read_point(MEMORY_ID)
    assert point is not None
    assert point.payload["revision_state"] == "source"
    assert point.payload["revision_operation_id"] == "fsck:check-1:issue-1"
    guard.assert_called_once()


def test_update_creates_immutable_successor_and_history() -> None:
    service = _service()
    _insert(service)

    result = service.revise(
        MEMORY_ID,
        user_id="user-1",
        owner_id="owner-1",
        session_agent_id=None,
        changes={"data": "new"},
        expected_revision=1,
        idempotency_key="request-1",
    )

    assert result["revision"] == 2
    original = service._read_point(MEMORY_ID)
    successor = service._read_point(result["revision_id"])
    assert original.payload["data"] == "old"
    assert original.payload["revision_state"] == "superseded"
    assert successor.payload["data"] == "new"
    assert successor.payload["revision_state"] == "active"
    assert successor.payload["supersedes"] == MEMORY_ID
    history = service.history(
        result["revision_id"],
        user_id="user-1",
        owner_id="owner-1",
        session_agent_id=None,
    )
    assert [item["metadata"]["revision"] for item in history["revisions"]] == [1, 2]
    assert history["operations"][0]["status"] == "committed"


def test_idempotent_retry_returns_same_successor() -> None:
    service = _service()
    _insert(service)
    kwargs = {
        "user_id": "user-1",
        "owner_id": "owner-1",
        "session_agent_id": None,
        "changes": {"data": "new"},
        "expected_revision": 1,
        "idempotency_key": "request-1",
    }

    first = service.revise(MEMORY_ID, **kwargs)
    second = service.revise(MEMORY_ID, **kwargs)

    assert second["revision_id"] == first["revision_id"]
    assert second["replayed"] is True
    assert len(service._lineage_points(MEMORY_ID)) == 2


def test_stale_expected_revision_is_rejected() -> None:
    service = _service()
    _insert(service)
    service.revise(
        MEMORY_ID,
        user_id="user-1",
        owner_id="owner-1",
        session_agent_id=None,
        changes={"data": "new"},
        expected_revision=1,
        idempotency_key="request-1",
    )

    with pytest.raises(RevisionConflictError) as exc:
        service.revise(
            MEMORY_ID,
            user_id="user-1",
            owner_id="owner-1",
            session_agent_id=None,
            changes={"data": "other"},
            expected_revision=1,
            idempotency_key="request-2",
        )
    assert exc.value.current_revision == 2


def test_retraction_preserves_content_and_artifacts() -> None:
    service = _service()
    _insert(service)

    result = service.retract(
        MEMORY_ID,
        user_id="user-1",
        owner_id="owner-1",
        session_agent_id=None,
        expected_revision=1,
        idempotency_key="delete-1",
    )

    point = service._read_point(MEMORY_ID)
    assert result["status"] == "retracted"
    assert point.payload["data"] == "old"
    assert point.payload["artifacts"] == [{"id": "artifact-1"}]
    assert point.payload["revision_state"] == "retracted"


def test_tenant_authorization_fails_closed() -> None:
    service = _service()
    _insert(service)

    with pytest.raises(ValueError, match="Cannot access memory"):
        service.revise(
            MEMORY_ID,
            user_id="other",
            owner_id="owner-1",
            session_agent_id=None,
            changes={"data": "new"},
        )


def test_concurrent_expected_revision_has_one_winner() -> None:
    service = _service()
    _insert(service)
    barrier = threading.Barrier(2)
    results: list[dict] = []
    errors: list[Exception] = []

    def revise(key: str, text: str) -> None:
        barrier.wait()
        try:
            results.append(
                service.revise(
                    MEMORY_ID,
                    user_id="user-1",
                    owner_id="owner-1",
                    session_agent_id=None,
                    changes={"data": text},
                    expected_revision=1,
                    idempotency_key=key,
                )
            )
        except Exception as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=revise, args=("race-1", "one")),
        threading.Thread(target=revise, args=("race-2", "two")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(results) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], RevisionConflictError)
    states = [
        point.payload["revision_state"] for point in service._lineage_points(MEMORY_ID)
    ]
    assert states.count("active") == 1
    assert all(state in {"active", "superseded", "aborted"} for state in states)


def test_retry_finalizes_after_crash_between_activation_and_finalize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service()
    _insert(service)
    original_write = service.operations.write
    failed = False

    def crash_once(token: str, payload: dict) -> str:
        nonlocal failed
        if payload.get("status") == "activated" and not failed:
            failed = True
            raise RuntimeError("injected crash")
        return original_write(token, payload)

    monkeypatch.setattr(service.operations, "write", crash_once)
    kwargs = {
        "user_id": "user-1",
        "owner_id": "owner-1",
        "session_agent_id": None,
        "changes": {"data": "new"},
        "expected_revision": 1,
        "idempotency_key": "request-1",
    }
    with pytest.raises(RuntimeError, match="injected crash"):
        service.revise(MEMORY_ID, **kwargs)

    monkeypatch.setattr(service.operations, "write", original_write)
    result = service.revise(MEMORY_ID, **kwargs)

    assert result["replayed"] is True
    states = [
        point.payload["revision_state"] for point in service._lineage_points(MEMORY_ID)
    ]
    assert states.count("active") == 1
    assert states.count("superseded") == 1


def test_implicit_key_recovers_after_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service()
    _insert(service)
    original_write = service.operations.write
    failed = False

    def crash_once(token: str, payload: dict) -> str:
        nonlocal failed
        if payload.get("status") == "activated" and not failed:
            failed = True
            raise RuntimeError("injected crash")
        return original_write(token, payload)

    monkeypatch.setattr(service.operations, "write", crash_once)
    kwargs = {
        "user_id": "user-1",
        "owner_id": "owner-1",
        "session_agent_id": None,
        "changes": {"data": "new"},
        "expected_revision": 1,
    }
    with pytest.raises(RuntimeError, match="injected crash"):
        service.revise(MEMORY_ID, **kwargs)

    monkeypatch.setattr(service.operations, "write", original_write)
    result = service.revise(MEMORY_ID, **kwargs)

    assert result["replayed"] is True
    assert result["revision"] == 2


def test_implicit_key_binds_to_source_revision() -> None:
    service = _service()
    _insert(service)

    second = service.revise(
        MEMORY_ID,
        user_id="user-1",
        owner_id="owner-1",
        session_agent_id=None,
        changes={"data": "new"},
    )
    third = service.revise(
        second["revision_id"],
        user_id="user-1",
        owner_id="owner-1",
        session_agent_id=None,
        changes={"data": "old"},
    )

    assert third["revision"] == 3
    assert service._read_point(third["revision_id"]).payload["data"] == "old"


def test_retraction_recovers_crash_after_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service()
    _insert(service)
    original_write = service.operations.write
    failed = False

    def crash_once(token: str, payload: dict) -> str:
        nonlocal failed
        if payload.get("status") == "claimed" and not failed:
            failed = True
            raise RuntimeError("injected retract crash")
        return original_write(token, payload)

    monkeypatch.setattr(service.operations, "write", crash_once)
    kwargs = {
        "user_id": "user-1",
        "owner_id": "owner-1",
        "session_agent_id": None,
        "expected_revision": 1,
        "idempotency_key": "delete-1",
    }
    with pytest.raises(RuntimeError, match="injected retract crash"):
        service.retract(MEMORY_ID, **kwargs)

    monkeypatch.setattr(service.operations, "write", original_write)
    result = service.retract(MEMORY_ID, **kwargs)
    assert result["status"] == "retracted"
    assert result["replayed"] is True


def test_history_filters_inaccessible_prior_agent_revision() -> None:
    service = _service()
    _insert(service, agent_id="parent:alpha")
    result = service.revise(
        MEMORY_ID,
        user_id="user-1",
        owner_id="owner-1",
        session_agent_id="parent",
        changes={"agent_id": None},
        expected_revision=1,
        idempotency_key="share-1",
    )

    history = service.history(
        result["revision_id"],
        user_id="user-1",
        owner_id="owner-1",
        session_agent_id="parent:beta",
    )

    assert [item["id"] for item in history["revisions"]] == [result["revision_id"]]
    assert history["operations"] == []
    with pytest.raises(ValueError, match="Cannot access memory"):
        service.links(
            MEMORY_ID,
            user_id="user-1",
            owner_id="owner-1",
            session_agent_id="parent:beta",
        )


def test_privacy_erase_claim_blocks_revision_and_is_retryable() -> None:
    service = _service()
    _insert(service)

    plan = service.prepare_privacy_erase(
        MEMORY_ID,
        user_id="user-1",
        owner_id="owner-1",
        session_agent_id=None,
    )
    replay = service.prepare_privacy_erase(
        MEMORY_ID,
        user_id="user-1",
        owner_id="owner-1",
        session_agent_id=None,
    )

    assert replay["operation_id"] == plan["operation_id"]
    with pytest.raises(RevisionConflictError, match="being erased"):
        service.revise(
            MEMORY_ID,
            user_id="user-1",
            owner_id="owner-1",
            session_agent_id=None,
            changes={"data": "resurrected"},
        )
    memory = MemoryService.__new__(MemoryService)
    memory.revisions = service
    memory.artifact = MagicMock()
    with pytest.raises(RevisionConflictError, match="being erased"):
        memory.save_artifact(
            MEMORY_ID,
            user_id="user-1",
            owner_id="owner-1",
            content="late artifact",
        )
    memory.artifact.save.assert_not_called()

    result = service.finalize_privacy_erase(replay)
    assert result["status"] == "erased"
    assert service._read_point(MEMORY_ID) is None
    assert service.operations.find_privacy_erase(MEMORY_ID) is None


def test_privacy_erase_retries_after_artifact_failure() -> None:
    revisions = _service()
    _insert(revisions)
    memory = MemoryService.__new__(MemoryService)
    memory.revisions = revisions
    memory.vector = revisions._vector
    memory.vector.artifact_has_references_outside = MagicMock(return_value=False)
    memory.artifact = MagicMock()
    memory.artifact.delete_by_id.side_effect = [
        RuntimeError("artifact backend unavailable"),
        None,
    ]
    memory._get_user_lock = lambda key: threading.Lock()
    memory._core_cache = MagicMock()
    memory._category_cache = MagicMock()

    with pytest.raises(RuntimeError, match="artifact backend unavailable"):
        memory.privacy_erase_memory(
            MEMORY_ID,
            user_id="user-1",
            owner_id="owner-1",
        )

    assert revisions._read_point(MEMORY_ID) is not None
    result = memory.privacy_erase_memory(
        MEMORY_ID,
        user_id="user-1",
        owner_id="owner-1",
    )
    assert result["status"] == "erased"
    assert revisions._read_point(MEMORY_ID) is None
    assert memory.artifact.delete_by_id.call_count == 2


def test_privacy_recovery_checks_agent_scope_after_points_are_deleted() -> None:
    service = _service()
    _insert(service, agent_id="parent:alpha")
    plan = service.prepare_privacy_erase(
        MEMORY_ID,
        user_id="user-1",
        owner_id="owner-1",
        session_agent_id="parent",
    )
    service._client.delete(
        collection_name="memories",
        points_selector=[MEMORY_ID],
        wait=True,
    )

    with pytest.raises(ValueError, match="not found"):
        service.prepare_privacy_erase(
            MEMORY_ID,
            user_id="user-1",
            owner_id="owner-1",
            session_agent_id="parent:beta",
        )
    assert service.operations.get(plan["operation_token"]) is not None


def test_history_resolves_audit_by_revision_operation_id() -> None:
    service = _service()
    _insert(service)
    token = "consolidation-token"
    operation_id = service.operations.write(
        token,
        {
            "status": "committed",
            "operation_kind": "consolidation",
            "actor_kind": "consolidation",
            "user_id": "user-1",
            "owner_id": "owner-1",
            "lineage_id": "session:ses-1",
        },
    )
    service._client.set_payload(
        collection_name="memories",
        payload={"revision_operation_id": operation_id},
        points=[MEMORY_ID],
        wait=True,
    )

    history = service.history(
        MEMORY_ID,
        user_id="user-1",
        owner_id="owner-1",
        session_agent_id=None,
    )

    assert [operation["operation_id"] for operation in history["operations"]] == [
        operation_id
    ]


def test_artifact_reads_require_active_owner_and_agent_scope() -> None:
    service = _service()
    _insert(service, agent_id="parent:alpha")
    service._client.set_payload(
        collection_name="memories",
        payload={"artifacts": [{"id": "artifact-1"}]},
        points=[MEMORY_ID],
        wait=True,
    )
    memory = MemoryService.__new__(MemoryService)
    memory.revisions = service
    memory.artifact = MagicMock()

    with pytest.raises(ValueError, match="Cannot access memory"):
        memory.list_artifacts(
            MEMORY_ID,
            user_id="user-1",
            owner_id="other-owner",
            session_agent_id="parent:alpha",
        )
    with pytest.raises(ValueError, match="Cannot access memory"):
        memory.get_artifact(
            MEMORY_ID,
            "artifact-1",
            user_id="user-1",
            owner_id="owner-1",
            session_agent_id="parent:beta",
        )


def test_artifact_reads_allow_exact_authorized_source_revision() -> None:
    service = _service()
    _insert(service, agent_id="parent:alpha")
    service._client.set_payload(
        collection_name="memories",
        payload={"revision_state": "source"},
        points=[MEMORY_ID],
        wait=True,
    )
    memory = MemoryService.__new__(MemoryService)
    memory.revisions = service
    memory.artifact = MagicMock()
    memory.artifact.load.return_value = {"content": "artifact body"}

    assert memory.list_artifacts(
        MEMORY_ID,
        user_id="user-1",
        owner_id="owner-1",
        session_agent_id="parent:alpha",
    ) == [{"id": "artifact-1"}]
    assert memory.get_artifact(
        MEMORY_ID,
        "artifact-1",
        user_id="user-1",
        owner_id="owner-1",
        session_agent_id="parent:alpha",
    ) == {"content": "artifact body"}
    with pytest.raises(ValueError, match="Cannot access memory"):
        memory.list_artifacts(
            MEMORY_ID,
            user_id="user-1",
            owner_id="owner-1",
            session_agent_id="parent:beta",
        )


def test_artifact_save_rejects_stale_alias_before_blob_write() -> None:
    revisions = _service()
    _insert(revisions)
    successor = revisions.revise(
        MEMORY_ID,
        user_id="user-1",
        owner_id="owner-1",
        session_agent_id=None,
        changes={"data": "updated"},
    )
    memory = MemoryService.__new__(MemoryService)
    memory.revisions = revisions
    memory.artifact = MagicMock()

    with pytest.raises(RevisionConflictError):
        memory.save_artifact(
            MEMORY_ID,
            user_id="user-1",
            owner_id="owner-1",
            content="body",
            expected_revision=1,
        )

    memory.artifact.save.assert_not_called()
    assert successor["revision"] == 2


def test_artifact_delete_rejects_stale_alias_without_mutation() -> None:
    revisions = _service()
    _insert(revisions)
    revisions._client.set_payload(
        collection_name="memories",
        payload={"artifacts": [{"id": "artifact-1"}]},
        points=[MEMORY_ID],
        wait=True,
    )
    revisions.revise(
        MEMORY_ID,
        user_id="user-1",
        owner_id="owner-1",
        session_agent_id=None,
        changes={"data": "updated"},
    )
    memory = MemoryService.__new__(MemoryService)
    memory.revisions = revisions
    memory.vector = MagicMock()
    memory.artifact = MagicMock()

    with pytest.raises(RevisionConflictError):
        memory.delete_artifact(
            MEMORY_ID,
            "artifact-1",
            user_id="user-1",
            owner_id="owner-1",
            expected_revision=1,
        )

    memory.artifact.delete_by_id.assert_not_called()


def test_artifact_precondition_retry_is_idempotent() -> None:
    service = _service()
    _insert(service)
    artifact = {"id": "artifact-1", "filename": "note.md"}

    first = service.set_artifacts(
        MEMORY_ID,
        user_id="user-1",
        owner_id="owner-1",
        session_agent_id=None,
        artifact=artifact,
        operation_kind="artifact_save",
        idempotency_key="artifact-1",
        expected_revision=1,
    )
    second = service.set_artifacts(
        MEMORY_ID,
        user_id="user-1",
        owner_id="owner-1",
        session_agent_id=None,
        artifact=artifact,
        operation_kind="artifact_save",
        idempotency_key="artifact-1",
        expected_revision=1,
    )

    assert first["revision"] == 1
    assert first["artifact_revision"] == 1
    assert second["replayed"] is True
    assert second["operation_id"] == first["operation_id"]


def test_operations_collection_creation_race_verifies_schema() -> None:
    barrier = threading.Barrier(2)

    class RacingClient:
        def __init__(self) -> None:
            self.created = False
            self.lock = threading.Lock()
            self.schema_checks = 0

        def get_collections(self):
            with self.lock:
                created = self.created
            if not created:
                barrier.wait(timeout=2)
            names = [SimpleNamespace(name=OPERATIONS_COLLECTION)] if created else []
            return SimpleNamespace(collections=names)

        def create_collection(self, **_kwargs):
            with self.lock:
                if self.created:
                    raise RuntimeError("already exists")
                self.created = True

        def get_collection(self, name):
            assert name == OPERATIONS_COLLECTION
            self.schema_checks += 1
            return SimpleNamespace(
                config=SimpleNamespace(
                    params=SimpleNamespace(
                        vectors=VectorParams(size=1, distance=Distance.COSINE)
                    )
                )
            )

    client = RacingClient()
    errors: list[Exception] = []

    def initialize() -> None:
        try:
            RevisionOperationStore(client, is_remote=False)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=initialize) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert errors == []
    assert client.created is True
    assert client.schema_checks == 1


def test_concurrent_artifact_saves_keep_both_references() -> None:
    service = _service()
    _insert(service)
    barrier = threading.Barrier(2)
    errors: list[Exception] = []

    def save(artifact_id: str) -> None:
        barrier.wait()
        try:
            service.set_artifacts(
                MEMORY_ID,
                user_id="user-1",
                owner_id="owner-1",
                session_agent_id=None,
                artifact={"id": artifact_id},
                operation_kind="artifact_save",
                idempotency_key=artifact_id,
            )
        except Exception as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=save, args=("artifact-1",)),
        threading.Thread(target=save, args=("artifact-2",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    point = service._read_point(MEMORY_ID)
    assert {item["id"] for item in point.payload["artifacts"]} == {
        "artifact-1",
        "artifact-2",
    }


def test_artifact_audit_failure_preserves_link_and_replays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service()
    _insert(service)
    original_write = service.operations.write
    failed = False

    def fail_commit_once(token: str, payload: dict) -> str:
        nonlocal failed
        if payload.get("status") == "committed" and not failed:
            failed = True
            raise RuntimeError("injected audit failure")
        return original_write(token, payload)

    monkeypatch.setattr(service.operations, "write", fail_commit_once)
    kwargs = {
        "user_id": "user-1",
        "owner_id": "owner-1",
        "session_agent_id": None,
        "artifact": {"id": "artifact-1"},
        "operation_kind": "artifact_save",
        "idempotency_key": "artifact-1",
    }
    with pytest.raises(RuntimeError, match="injected audit failure"):
        service.set_artifacts(MEMORY_ID, **kwargs)
    assert service._read_point(MEMORY_ID).payload["artifacts"] == [{"id": "artifact-1"}]

    monkeypatch.setattr(service.operations, "write", original_write)
    service.current(
        MEMORY_ID,
        user_id="user-1",
        owner_id="owner-1",
        session_agent_id=None,
    )
    replay = service.set_artifacts(MEMORY_ID, **kwargs)
    assert replay["replayed"] is True
    assert replay["lineage_id"] == MEMORY_ID
    assert replay["revision"] == 1
    assert replay["artifact_revision"] == 1


@pytest.mark.parametrize(
    ("initial_artifacts", "kwargs", "expected_artifacts"),
    [
        (
            [],
            {
                "artifact": {"id": "artifact-1"},
                "operation_kind": "artifact_save",
                "idempotency_key": "artifact-1",
            },
            [{"id": "artifact-1"}],
        ),
        (
            [{"id": "artifact-1"}],
            {
                "remove_artifact_id": "artifact-1",
                "operation_kind": "artifact_delete",
                "idempotency_key": "artifact-1",
            },
            [],
        ),
    ],
)
def test_prepared_artifact_operation_retries_before_claim(
    monkeypatch: pytest.MonkeyPatch,
    initial_artifacts: list[dict],
    kwargs: dict,
    expected_artifacts: list[dict],
) -> None:
    service = _service()
    _insert(service)
    service._client.set_payload(
        collection_name="memories",
        payload={"artifacts": initial_artifacts},
        points=[MEMORY_ID],
        wait=True,
    )
    original_claim = service._claim
    failed = False

    def fail_claim_once(**claim_kwargs):
        nonlocal failed
        if not failed:
            failed = True
            raise RuntimeError("injected pre-claim crash")
        return original_claim(**claim_kwargs)

    monkeypatch.setattr(service, "_claim", fail_claim_once)
    common = {
        "user_id": "user-1",
        "owner_id": "owner-1",
        "session_agent_id": None,
    }
    with pytest.raises(RuntimeError, match="injected pre-claim crash"):
        service.set_artifacts(MEMORY_ID, **common, **kwargs)

    replay = service.set_artifacts(MEMORY_ID, **common, **kwargs)
    assert replay["lineage_id"] == MEMORY_ID
    assert replay["revision"] == 1
    assert replay["artifact_revision"] == 1
    assert service._read_point(MEMORY_ID).payload["artifacts"] == expected_artifacts


def test_artifact_delete_retries_physical_cleanup() -> None:
    revisions = _service()
    _insert(revisions)
    revisions._client.set_payload(
        collection_name="memories",
        payload={"artifacts": [{"id": "artifact-1"}]},
        points=[MEMORY_ID],
        wait=True,
    )
    memory = MemoryService.__new__(MemoryService)
    memory.revisions = revisions
    memory.vector = revisions._vector
    memory.vector.artifact_has_references_outside = MagicMock(return_value=False)
    memory.artifact = MagicMock()
    memory.artifact.delete_by_id.side_effect = [
        RuntimeError("artifact backend unavailable"),
        None,
    ]

    with pytest.raises(RuntimeError, match="artifact backend unavailable"):
        memory.delete_artifact(
            MEMORY_ID,
            "artifact-1",
            user_id="user-1",
            owner_id="owner-1",
        )

    result = memory.delete_artifact(
        MEMORY_ID,
        "artifact-1",
        user_id="user-1",
        owner_id="owner-1",
    )
    assert result["status"] == "deleted"
    assert memory.artifact.delete_by_id.call_count == 2


def test_revision_migration_preserves_legacy_content_and_vector() -> None:
    from mnemory.migration import AddRevisionFoundationMigration

    client = QdrantClient(location=":memory:")
    client.create_collection(
        collection_name="memories",
        vectors_config=VectorParams(size=2, distance=Distance.COSINE),
    )
    source_id = MEMORY_ID
    target_id = "22222222-2222-4222-8222-222222222222"
    client.upsert(
        collection_name="memories",
        points=[
            PointStruct(
                id=source_id,
                vector=[1.0, 0.0],
                payload={
                    "data": "legacy source",
                    "hash": "legacy-hash",
                    "user_id": "user-1",
                    "owner_id": "owner-1",
                    "memory_layer": "raw",
                    "superseded_by": target_id,
                    "custom_field": "preserve-me",
                },
            ),
            PointStruct(
                id=target_id,
                vector=[0.0, 1.0],
                payload={
                    "data": "legacy output",
                    "user_id": "user-1",
                    "owner_id": "owner-1",
                    "derived_from": [source_id],
                },
            ),
        ],
        wait=True,
    )
    migration = AddRevisionFoundationMigration("memories")

    migration.run(
        client,
        progress=None,
        state_callback=lambda state: None,
        state={},
    )

    source = client.retrieve(
        collection_name="memories",
        ids=[source_id],
        with_payload=True,
        with_vectors=True,
    )[0]
    assert source.payload["data"] == "legacy source"
    assert source.payload["hash"] == "legacy-hash"
    assert source.payload["custom_field"] == "preserve-me"
    assert source.payload["superseded_by"] == target_id
    assert source.payload["revision_state"] == "source"
    assert source.payload["provenance_quality"] == "legacy_batch"
    assert source.vector == [1.0, 0.0]
    assert {item.name for item in client.get_collections().collections} == {
        "memories",
        "_mnemory_operations",
    }


def _evidence_candidate(
    service: RevisionService,
    memory_id: str,
    *,
    layer: str = "raw",
    derived_from: list[str] | None = None,
    text: str = "User lives in Prague",
    include_fact_hash: bool = True,
) -> dict[str, str]:
    content_hash = hashlib.sha256(text.encode()).hexdigest()
    fact_hash = service._normalized_fact_hash(text)
    payload = {
        "data": text,
        "hash": content_hash,
        "fact_hash": fact_hash,
        "user_id": "user-1",
        "owner_id": "owner-1",
        "role": "user",
        "agent_id": None,
        "memory_layer": layer,
        "lineage_id": memory_id,
        "revision": 1,
        "revision_state": "active",
        "derived_from": derived_from if derived_from is not None else [],
    }
    if not include_fact_hash:
        payload.pop("fact_hash")
    service._client.upsert(
        collection_name="memories",
        points=[PointStruct(id=memory_id, vector=[1.0, 0.0], payload=payload)],
        wait=True,
    )
    return {
        "candidate_id": memory_id,
        "lineage_id": memory_id,
        "revision": 1,
        "revision_id": memory_id,
        "content_hash": content_hash,
        "fact_hash": fact_hash,
    }


@pytest.mark.parametrize(
    ("layer", "target_id", "unrelated_id"),
    [
        (
            "raw",
            "17171717-1717-4171-8171-171717171717",
            "18181818-1818-4181-8181-181818181818",
        ),
        (
            "consolidated",
            "19191919-1919-4191-8191-191919191919",
            "20202020-2020-4202-8202-202020202020",
        ),
    ],
)
def test_legacy_fact_hash_target_recovers_once(
    monkeypatch,
    layer: str,
    target_id: str,
    unrelated_id: str,
) -> None:
    service = _service()
    candidate = _evidence_candidate(
        service,
        target_id,
        layer=layer,
        derived_from=["raw-source"] if layer == "consolidated" else None,
        include_fact_hash=False,
    )
    unrelated = _evidence_candidate(
        service,
        unrelated_id,
        layer=layer,
        text="User lives in Prague",
        include_fact_hash=False,
    )
    plan = service.plan_evidence(
        [candidate],
        user_id="user-1",
        owner_id="owner-1",
        evidence_root_id=f"legacy-fact-{layer}",
    )
    assert plan[0]["action"] == "CONFIRM"
    assert plan[0]["target_id"] == candidate["candidate_id"]
    parent = service.operations.seal_evidence_plan(
        user_id="user-1",
        owner_id="owner-1",
        evidence_root_id=f"legacy-fact-{layer}",
        request_fingerprint=f"legacy-fact-request-{layer}",
        targets=[
            {
                "ordinal": 0,
                "target_id": candidate["candidate_id"],
                "action": "CONFIRM",
                "lineage_id": candidate["lineage_id"],
                "revision_id": candidate["revision_id"],
                "revision": candidate["revision"],
                "content_hash": candidate["content_hash"],
                "fact_hash": candidate["fact_hash"],
            }
        ],
    )
    service.operations.claim_evidence_plan(
        parent["operation_id"],
        request_fingerprint=f"legacy-fact-request-{layer}",
        epoch=1,
        nonce=f"legacy-parent-{layer}",
    )
    confirm_args = {
        "user_id": "user-1",
        "owner_id": "owner-1",
        "session_agent_id": None,
        "evidence_root_id": parent["evidence_root_id"],
        "source_kind": "evidence_plan",
        "source_fingerprint": f"legacy-fact-request-{layer}",
        "ttl_multiplier": 1.0,
        "max_score_roots": 3,
        "idempotency_key": f"{parent['operation_id']}:0",
        "expected_revision_id": candidate["revision_id"],
        "expected_lineage_id": candidate["lineage_id"],
        "expected_content_hash": candidate["content_hash"],
        "expected_fact_hash": candidate["fact_hash"],
        "parent_operation_id": parent["operation_id"],
        "parent_epoch": 1,
        "parent_nonce": f"legacy-parent-{layer}",
    }
    original_set_payload = service._client.set_payload
    crashed = False

    def crash_projection(**kwargs):
        nonlocal crashed
        if (
            not crashed
            and kwargs.get("collection_name") == "memories"
            and "evidence_root_ids" in (kwargs.get("payload") or {})
        ):
            crashed = True
            raise RuntimeError("legacy fact recovery interruption")
        return original_set_payload(**kwargs)

    monkeypatch.setattr(service._client, "set_payload", crash_projection)
    with pytest.raises(RuntimeError, match="legacy fact recovery interruption"):
        service.confirm(candidate["candidate_id"], **confirm_args)
    monkeypatch.setattr(service._client, "set_payload", original_set_payload)

    crashed_target = service._payload(
        service._read_point(candidate["candidate_id"], with_vectors=False)
    )
    child_operation_id = _operation_point_id(
        crashed_target["transition_operation_token"]
    )
    service._client.set_payload(
        collection_name=OPERATIONS_COLLECTION,
        payload={"child_claim_deadline_utc": "2000-01-01T00:00:00+00:00"},
        points=[child_operation_id],
        wait=True,
    )
    service._client.set_payload(
        collection_name=OPERATIONS_COLLECTION,
        payload={"claim_deadline_utc": "2000-01-01T00:00:00+00:00"},
        points=[parent["operation_id"]],
        wait=True,
    )
    service.operations.claim_evidence_plan(
        parent["operation_id"],
        request_fingerprint=f"legacy-fact-request-{layer}",
        epoch=2,
        nonce=f"legacy-parent-recovered-{layer}",
    )
    confirm_args.update(
        parent_epoch=2,
        parent_nonce=f"legacy-parent-recovered-{layer}",
    )
    recovered = service.confirm(candidate["candidate_id"], **confirm_args)
    checkpointed = service.operations.checkpoint_evidence_plan(
        parent["operation_id"],
        request_fingerprint=f"legacy-fact-request-{layer}",
        epoch=2,
        nonce=f"legacy-parent-recovered-{layer}",
        checkpoints=[
            {
                "ordinal": 0,
                "target_id": candidate["candidate_id"],
                "action": "CONFIRM",
                "status": "confirmed",
                "result": recovered,
            }
        ],
    )
    committed = service.operations.commit_evidence_plan(
        parent["operation_id"],
        request_fingerprint=f"legacy-fact-request-{layer}",
        epoch=2,
        nonce=f"legacy-parent-recovered-{layer}",
    )
    target = service._payload(
        service._read_point(candidate["candidate_id"], with_vectors=False)
    )
    unrelated_target = service._payload(
        service._read_point(unrelated["candidate_id"], with_vectors=False)
    )
    assert recovered["status"] == "confirmed"
    assert target["validation_count"] == 1
    assert checkpointed["checkpoints"][0]["status"] == "confirmed"
    assert committed["status"] == "committed"
    assert not any(field in target for field in _TRANSITION_FIELDS)
    assert not any(field in unrelated_target for field in _TRANSITION_FIELDS)
    assert unrelated_target.get("validation_count", 0) == 0


def test_evidence_planner_is_read_only_and_strict() -> None:
    service = _service()
    raw = _evidence_candidate(service, "44444444-4444-4444-8444-444444444444")
    consolidated = _evidence_candidate(
        service,
        "55555555-5555-4555-8555-555555555555",
        layer="consolidated",
        derived_from=["raw-a", "raw-b"],
    )

    plan = service.plan_evidence(
        [consolidated, raw],
        user_id="user-1",
        owner_id="owner-1",
        evidence_root_id="event-1",
    )

    assert [item["action"] for item in plan] == ["CONFIRM", "SKIP"]
    assert plan[0]["target_id"] == "44444444-4444-4444-8444-444444444444"
    assert plan[1]["reason"] == "candidate_not_equivalent"
    assert service.operations.get_by_id("raw-evidence") is None


def test_evidence_journal_claim_checkpoint_commit_and_generic_guard() -> None:
    service = _service()
    raw = _evidence_candidate(service, "66666666-6666-4666-8666-666666666666")
    plan = service.operations.seal_evidence_plan(
        user_id="user-1",
        owner_id="owner-1",
        evidence_root_id="event-1",
        request_fingerprint="request-1",
        targets=[{"ordinal": 0, "target_id": raw["candidate_id"], "action": "CONFIRM"}],
    )
    replay = service.operations.seal_evidence_plan(
        user_id="user-1",
        owner_id="owner-1",
        evidence_root_id="event-1",
        request_fingerprint="request-1",
        targets=[],
    )
    assert replay["operation_id"] == plan["operation_id"]
    with pytest.raises(EvidenceConflictError):
        service.operations.seal_evidence_plan(
            user_id="user-1",
            owner_id="owner-1",
            evidence_root_id="event-1",
            request_fingerprint="request-2",
            targets=[],
        )
    with pytest.raises(ValueError, match="evidence-specific"):
        service.operations.write(
            "generic-evidence",
            {"operation_kind": EVIDENCE_OPERATION_KIND},
        )

    claimed = service.operations.claim_evidence_plan(
        plan["operation_id"],
        request_fingerprint="request-1",
        epoch=1,
        nonce="nonce-1",
    )
    assert claimed["status"] == "claimed"
    with pytest.raises(EvidenceClaimActiveError):
        service.operations.claim_evidence_plan(
            plan["operation_id"],
            request_fingerprint="request-1",
            epoch=2,
            nonce="nonce-2",
        )
    checkpointed = service.operations.checkpoint_evidence_plan(
        plan["operation_id"],
        request_fingerprint="request-1",
        epoch=1,
        nonce="nonce-1",
        checkpoints=[{"ordinal": 0, "status": "confirmed"}],
    )
    assert checkpointed["checkpoints"][0]["status"] == "confirmed"
    committed = service.operations.commit_evidence_plan(
        plan["operation_id"],
        request_fingerprint="request-1",
        epoch=1,
        nonce="nonce-1",
    )
    assert committed["status"] == "committed"


def test_evidence_seal_is_single_winner_under_local_concurrency() -> None:
    service = _service()
    store = service.operations
    barrier = threading.Barrier(2)
    results: list[dict] = []
    errors: list[Exception] = []

    def seal() -> None:
        barrier.wait()
        try:
            results.append(
                store.seal_evidence_plan(
                    user_id="user-1",
                    owner_id="owner-1",
                    evidence_root_id="event-race",
                    request_fingerprint="request-race",
                    targets=[],
                )
            )
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=seal) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert len({item["operation_id"] for item in results}) == 1


def test_evidence_consumption_survives_successor_revision() -> None:
    service = _service()
    _evidence_candidate(service, MEMORY_ID, text="User lives in Prague")
    service._client.set_payload(
        collection_name="memories",
        payload={"evidence_root_ids": ["origin"]},
        points=[MEMORY_ID],
        wait=True,
    )
    service.confirm(
        MEMORY_ID,
        user_id="user-1",
        owner_id="owner-1",
        session_agent_id=None,
        evidence_root_id="event-1",
        source_kind="evidence",
        source_fingerprint="event-1",
        ttl_multiplier=1.0,
        max_score_roots=3,
    )
    normal_payload = service._read_point(MEMORY_ID).payload
    assert not any(field in normal_payload for field in _TRANSITION_FIELDS)
    successor = service.revise(
        MEMORY_ID,
        user_id="user-1",
        owner_id="owner-1",
        session_agent_id=None,
        changes={"data": "User lives in Berlin"},
    )
    result = service.confirm(
        successor["revision_id"],
        user_id="user-1",
        owner_id="owner-1",
        session_agent_id=None,
        evidence_root_id="event-1",
        source_kind="evidence",
        source_fingerprint="event-1",
        ttl_multiplier=1.0,
        max_score_roots=3,
    )
    assert result["status"] == "skipped"
    successor_payload = service._read_point(successor["revision_id"]).payload
    assert successor_payload["evidence_root_ids"] == []
    assert "event-1" in successor_payload["consumed_evidence_root_ids"]
    confirmed = service.confirm(
        successor["revision_id"],
        user_id="user-1",
        owner_id="owner-1",
        session_agent_id=None,
        evidence_root_id="event-2",
        source_kind="evidence",
        source_fingerprint="event-2",
        ttl_multiplier=1.0,
        max_score_roots=3,
    )
    assert confirmed["status"] == "confirmed"
    assert (
        service._read_point(successor["revision_id"]).payload["validation_count"] == 1
    )


def test_apply_evidence_plan_resumes_embedded_checkpoints() -> None:
    revisions = _service()
    candidate = _evidence_candidate(revisions, "77777777-7777-4777-8777-777777777777")
    second_candidate = _evidence_candidate(
        revisions, "88888888-8888-4888-8888-888888888888", text="User uses Rust"
    )
    memory = MemoryService.__new__(MemoryService)
    memory.revisions = revisions
    memory._config = SimpleNamespace(
        memory=SimpleNamespace(
            validation_ttl_multiplier=1.0,
            validation_max_score_roots=3,
        )
    )
    plan = memory.plan_evidence(
        [candidate, second_candidate],
        user_id="user-1",
        owner_id="owner-1",
        evidence_root_id="event-2",
    )
    sealed = memory.seal_evidence_plan(plan, request_fingerprint="request-2")
    revisions.operations.claim_evidence_plan(
        sealed["operation_id"],
        request_fingerprint="request-2",
        epoch=1,
        nonce="nonce-2",
    )
    revisions.operations.checkpoint_evidence_plan(
        sealed["operation_id"],
        request_fingerprint="request-2",
        epoch=1,
        nonce="nonce-2",
        checkpoints=[{"ordinal": 0, "status": "skipped"}],
    )
    result = memory.apply_evidence_plan(
        sealed["operation_id"],
        request_fingerprint="request-2",
        epoch=1,
        nonce="nonce-2",
        user_id="user-1",
        owner_id="owner-1",
    )
    assert result["status"] == "committed"
    assert result["result"]["checkpoints"][0]["status"] == "skipped"
    assert result["result"]["checkpoints"][1]["status"] == "confirmed"


def test_evidence_seal_conflicting_fingerprints_have_one_parent_winner() -> None:
    service = _service()
    barrier = threading.Barrier(2)
    results: list[dict] = []
    errors: list[Exception] = []

    def seal(fingerprint: str) -> None:
        barrier.wait()
        try:
            results.append(
                service.operations.seal_evidence_plan(
                    user_id="user-1",
                    owner_id="owner-1",
                    evidence_root_id="event-conflict",
                    request_fingerprint=fingerprint,
                    targets=[],
                )
            )
        except Exception as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=seal, args=("request-a",)),
        threading.Thread(target=seal, args=("request-b",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(results) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], EvidenceConflictError)


def test_evidence_takeover_invalidates_old_lease() -> None:
    service = _service()
    plan = service.operations.seal_evidence_plan(
        user_id="user-1",
        owner_id="owner-1",
        evidence_root_id="event-takeover",
        request_fingerprint="request-takeover",
        targets=[],
    )
    service.operations.claim_evidence_plan(
        plan["operation_id"],
        request_fingerprint="request-takeover",
        epoch=1,
        nonce="old-nonce",
    )
    service._client.set_payload(
        collection_name=OPERATIONS_COLLECTION,
        payload={"claim_deadline_utc": "2000-01-01T00:00:00+00:00"},
        points=[plan["operation_id"]],
        wait=True,
    )
    service.operations.claim_evidence_plan(
        plan["operation_id"],
        request_fingerprint="request-takeover",
        epoch=2,
        nonce="new-nonce",
    )
    with pytest.raises(EvidenceLeaseLostError):
        service.operations.verify_evidence_claim(
            plan["operation_id"],
            request_fingerprint="request-takeover",
            epoch=1,
            nonce="old-nonce",
        )


def test_child_takeover_fences_paused_old_target_write() -> None:
    service = _service()
    candidate = _evidence_candidate(
        service, "eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee", text="User lives in Prague"
    )
    parent = service.operations.seal_evidence_plan(
        user_id="user-1",
        owner_id="owner-1",
        evidence_root_id="event-child-fence",
        request_fingerprint="request-child-fence",
        targets=[
            {
                "ordinal": 0,
                "target_id": candidate["candidate_id"],
                "action": "CONFIRM",
                "lineage_id": candidate["lineage_id"],
                "revision_id": candidate["revision_id"],
                "revision": candidate["revision"],
                "content_hash": candidate["content_hash"],
                "fact_hash": candidate["fact_hash"],
            }
        ],
    )
    service.operations.claim_evidence_plan(
        parent["operation_id"],
        request_fingerprint="request-child-fence",
        epoch=1,
        nonce="parent-old",
    )
    child_token = canonical_fingerprint(["child-operation-token", "child-fence", "0"])
    service.operations.write(
        child_token,
        {
            "status": "prepared",
            "operation_kind": "confirm",
            "parent_operation_id": parent["operation_id"],
            "evidence_child_key": f"{parent['operation_id']}:0",
            "target_revision_id": candidate["candidate_id"],
        },
    )
    old = service.operations.claim_evidence_child(
        child_token,
        parent_operation_id=parent["operation_id"],
        ordinal=0,
        epoch=1,
        nonce="child-old",
    )
    service._client.set_payload(
        collection_name=OPERATIONS_COLLECTION,
        payload={
            "child_claim_deadline_utc": "2000-01-01T00:00:00+00:00",
            "child_claim_epoch": 1,
        },
        points=[_operation_point_id(child_token)],
        wait=True,
    )
    new = service.operations.claim_evidence_child(
        child_token,
        parent_operation_id=parent["operation_id"],
        ordinal=0,
        epoch=2,
        nonce="child-new",
    )
    assert old["child_transition_token"] != new["child_transition_token"]
    with pytest.raises(RevisionConflictError):
        service._claim(
            point_id=candidate["candidate_id"],
            payload=service._payload(service._read_point(candidate["candidate_id"])),
            token=old["child_transition_token"],
            kind="confirm",
            successor_id=None,
            expected_child_token=old["child_transition_token"],
        )


def test_child_takeover_after_parent_check_blocks_old_projection(monkeypatch) -> None:
    service = _service()
    candidate = _evidence_candidate(service, "ffffffff-ffff-4fff-8fff-ffffffffffff")
    parent = service.operations.seal_evidence_plan(
        user_id="user-1",
        owner_id="owner-1",
        evidence_root_id="event-pause-projection",
        request_fingerprint="request-pause-projection",
        targets=[],
    )
    service.operations.claim_evidence_plan(
        parent["operation_id"],
        request_fingerprint="request-pause-projection",
        epoch=1,
        nonce="parent-old",
    )
    captured: dict[str, str] = {}
    original_child_claim = service.operations.claim_evidence_child

    def capture_child(*args, **kwargs):
        captured["token"] = args[0] if args else kwargs["token"]
        return original_child_claim(*args, **kwargs)

    monkeypatch.setattr(service.operations, "claim_evidence_child", capture_child)
    original_set_payload = service._client.set_payload
    takeover_done = False

    def pause_before_old_claim(**kwargs):
        nonlocal takeover_done
        payload = kwargs.get("payload") or {}
        if (
            not takeover_done
            and kwargs.get("collection_name") == "memories"
            and payload.get("transition_token")
        ):
            takeover_done = True
            child_token = captured["token"]
            service._client.set_payload(
                collection_name=OPERATIONS_COLLECTION,
                payload={"child_claim_deadline_utc": "2000-01-01T00:00:00+00:00"},
                points=[_operation_point_id(child_token)],
                wait=True,
            )
            original_child_claim(
                child_token,
                parent_operation_id=parent["operation_id"],
                ordinal=0,
                epoch=2,
                nonce="child-new",
            )
        return original_set_payload(**kwargs)

    monkeypatch.setattr(service._client, "set_payload", pause_before_old_claim)
    with pytest.raises(EvidenceClaimActiveError):
        service.confirm(
            candidate["candidate_id"],
            user_id="user-1",
            owner_id="owner-1",
            session_agent_id=None,
            evidence_root_id=parent["evidence_root_id"],
            source_kind="evidence_plan",
            source_fingerprint="request-pause-projection",
            ttl_multiplier=1.0,
            max_score_roots=3,
            idempotency_key=f"{parent['operation_id']}:0",
            expected_revision_id=candidate["revision_id"],
            expected_lineage_id=candidate["lineage_id"],
            expected_content_hash=candidate["content_hash"],
            expected_fact_hash=candidate["fact_hash"],
            parent_operation_id=parent["operation_id"],
            parent_epoch=1,
            parent_nonce="parent-old",
        )
    target = service._payload(service._read_point(candidate["candidate_id"]))
    assert "transition_token" not in target
    assert target["evidence_child_fence"] != captured.get("old_fence")


def test_retry_same_child_repairs_fence_after_takeover_crash(monkeypatch) -> None:
    service = _service()
    candidate = _evidence_candidate(service, "14141414-1414-4141-8141-141414141414")
    parent = service.operations.seal_evidence_plan(
        user_id="user-1",
        owner_id="owner-1",
        evidence_root_id="event-repair-window",
        request_fingerprint="request-repair-window",
        targets=[],
    )
    service.operations.claim_evidence_plan(
        parent["operation_id"],
        request_fingerprint="request-repair-window",
        epoch=1,
        nonce="parent-old",
    )
    child_token = canonical_fingerprint(["repair-window-child"])
    service.operations.write(
        child_token,
        {
            "status": "prepared",
            "operation_kind": "confirm",
            "parent_operation_id": parent["operation_id"],
            "evidence_child_key": f"{parent['operation_id']}:0",
            "target_revision_id": candidate["candidate_id"],
        },
    )
    old = service.operations.claim_evidence_child(
        child_token,
        parent_operation_id=parent["operation_id"],
        ordinal=0,
        epoch=1,
        nonce="child-old",
    )
    service._client.set_payload(
        collection_name=OPERATIONS_COLLECTION,
        payload={"child_claim_deadline_utc": "2000-01-01T00:00:00+00:00"},
        points=[_operation_point_id(child_token)],
        wait=True,
    )
    service._client.set_payload(
        collection_name=OPERATIONS_COLLECTION,
        payload={"claim_deadline_utc": "2000-01-01T00:00:00+00:00"},
        points=[parent["operation_id"]],
        wait=True,
    )
    service.operations.claim_evidence_plan(
        parent["operation_id"],
        request_fingerprint="request-repair-window",
        epoch=2,
        nonce="parent-new",
    )
    original_set_payload = service._client.set_payload
    crashed = False

    def crash_after_child(**kwargs):
        nonlocal crashed
        if (
            not crashed
            and kwargs.get("collection_name") == "memories"
            and "evidence_child_fence" in (kwargs.get("payload") or {})
        ):
            crashed = True
            raise RuntimeError("crash after child takeover")
        return original_set_payload(**kwargs)

    monkeypatch.setattr(service._client, "set_payload", crash_after_child)
    with pytest.raises(RuntimeError, match="crash after child takeover"):
        service.operations.claim_evidence_child(
            child_token,
            parent_operation_id=parent["operation_id"],
            ordinal=0,
            epoch=2,
            nonce="child-new",
        )
    monkeypatch.setattr(service._client, "set_payload", original_set_payload)
    repaired = service.operations.claim_evidence_child(
        child_token,
        parent_operation_id=parent["operation_id"],
        ordinal=0,
        epoch=2,
        nonce="child-new",
    )
    assert repaired["child_transition_token"] != old["child_transition_token"]
    target = service._payload(service._read_point(candidate["candidate_id"]))
    assert target["evidence_child_fence"] == repaired["child_transition_token"]
    service._claim(
        point_id=candidate["candidate_id"],
        payload=service._payload(service._read_point(candidate["candidate_id"])),
        token=repaired["child_transition_token"],
        kind="confirm",
        successor_id=None,
        expected_child_token=repaired["child_transition_token"],
    )
    target_after_claim = service._payload(
        service._read_point(candidate["candidate_id"])
    )
    assert target_after_claim["transition_token"] == repaired["child_transition_token"]
    assert target_after_claim["transition_operation_token"]


def test_confirm_recovers_child_crash_window_exactly_once(monkeypatch) -> None:
    service = _service()
    candidate = _evidence_candidate(service, "15151515-1515-4151-8151-151515151515")
    parent = service.operations.seal_evidence_plan(
        user_id="user-1",
        owner_id="owner-1",
        evidence_root_id="event-exact-recovery",
        request_fingerprint="request-exact-recovery",
        targets=[
            {
                "ordinal": 0,
                "target_id": candidate["candidate_id"],
                "action": "CONFIRM",
                "lineage_id": candidate["lineage_id"],
                "revision_id": candidate["revision_id"],
                "revision": candidate["revision"],
                "content_hash": candidate["content_hash"],
                "fact_hash": candidate["fact_hash"],
            }
        ],
    )
    service.operations.claim_evidence_plan(
        parent["operation_id"],
        request_fingerprint="request-exact-recovery",
        epoch=1,
        nonce="parent-1",
    )
    original_repair = service.operations._repair_child_target_fence
    crashed = False

    def crash_once(record):
        nonlocal crashed
        if not crashed:
            crashed = True
            raise RuntimeError("crash between child claim and target fence")
        return original_repair(record)

    monkeypatch.setattr(service.operations, "_repair_child_target_fence", crash_once)
    confirm_args = {
        "user_id": "user-1",
        "owner_id": "owner-1",
        "session_agent_id": None,
        "evidence_root_id": parent["evidence_root_id"],
        "source_kind": "evidence_plan",
        "source_fingerprint": "request-exact-recovery",
        "ttl_multiplier": 1.0,
        "max_score_roots": 3,
        "idempotency_key": f"{parent['operation_id']}:0",
        "expected_revision_id": candidate["revision_id"],
        "expected_lineage_id": candidate["lineage_id"],
        "expected_content_hash": candidate["content_hash"],
        "expected_fact_hash": candidate["fact_hash"],
        "parent_operation_id": parent["operation_id"],
        "parent_epoch": 1,
        "parent_nonce": "parent-1",
    }
    with pytest.raises(RuntimeError, match="between child claim"):
        service.confirm(candidate["candidate_id"], **confirm_args)
    monkeypatch.setattr(
        service.operations, "_repair_child_target_fence", original_repair
    )
    operation_points, _ = service._client.scroll(
        collection_name=OPERATIONS_COLLECTION,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="parent_operation_id",
                    match=MatchValue(value=parent["operation_id"]),
                )
            ]
        ),
        limit=10,
        with_payload=True,
        with_vectors=False,
    )
    assert len(operation_points) == 1
    child_operation = dict(operation_points[0].payload)
    service._client.set_payload(
        collection_name=OPERATIONS_COLLECTION,
        payload={"child_claim_deadline_utc": "2000-01-01T00:00:00+00:00"},
        points=[operation_points[0].id],
        wait=True,
    )
    service._client.set_payload(
        collection_name=OPERATIONS_COLLECTION,
        payload={"claim_deadline_utc": "2000-01-01T00:00:00+00:00"},
        points=[parent["operation_id"]],
        wait=True,
    )
    service.operations.claim_evidence_plan(
        parent["operation_id"],
        request_fingerprint="request-exact-recovery",
        epoch=2,
        nonce="parent-2",
    )
    confirm_args.update(parent_epoch=2, parent_nonce="parent-2")
    result = service.confirm(candidate["candidate_id"], **confirm_args)
    assert result["status"] == "confirmed"
    assert (
        service._payload(service._read_point(candidate["candidate_id"]))[
            "validation_count"
        ]
        == 1
    )
    child_after = service.operations.get(child_operation["operation_token"])
    assert child_after is not None
    assert child_after["status"] == "committed"


def test_crash_after_target_claim_recovers_projection_once(monkeypatch) -> None:
    service = _service()
    candidate = _evidence_candidate(service, "12121212-1212-4121-8121-121212121212")
    parent = service.operations.seal_evidence_plan(
        user_id="user-1",
        owner_id="owner-1",
        evidence_root_id="event-crash-projection",
        request_fingerprint="request-crash-projection",
        targets=[
            {
                "ordinal": 0,
                "target_id": candidate["candidate_id"],
                "action": "CONFIRM",
                "lineage_id": candidate["lineage_id"],
                "revision_id": candidate["revision_id"],
                "revision": candidate["revision"],
                "content_hash": candidate["content_hash"],
                "fact_hash": candidate["fact_hash"],
            }
        ],
    )
    service.operations.claim_evidence_plan(
        parent["operation_id"],
        request_fingerprint="request-crash-projection",
        epoch=1,
        nonce="parent-crash",
    )
    original_set_payload = service._client.set_payload
    crashed = False

    def crash_projection(**kwargs):
        nonlocal crashed
        if (
            not crashed
            and kwargs.get("collection_name") == "memories"
            and "evidence_root_ids" in (kwargs.get("payload") or {})
        ):
            crashed = True
            raise RuntimeError("crash after target claim")
        return original_set_payload(**kwargs)

    monkeypatch.setattr(service._client, "set_payload", crash_projection)
    with pytest.raises(RuntimeError, match="crash after target claim"):
        service.confirm(
            candidate["candidate_id"],
            user_id="user-1",
            owner_id="owner-1",
            session_agent_id=None,
            evidence_root_id=parent["evidence_root_id"],
            source_kind="evidence_plan",
            source_fingerprint="request-crash-projection",
            ttl_multiplier=1.0,
            max_score_roots=3,
            idempotency_key=f"{parent['operation_id']}:0",
            expected_revision_id=candidate["revision_id"],
            expected_lineage_id=candidate["lineage_id"],
            expected_content_hash=candidate["content_hash"],
            expected_fact_hash=candidate["fact_hash"],
            parent_operation_id=parent["operation_id"],
            parent_epoch=1,
            parent_nonce="parent-crash",
        )
    monkeypatch.setattr(service._client, "set_payload", original_set_payload)
    target_after_crash = service._payload(
        service._read_point(candidate["candidate_id"])
    )
    child_operation_id = _operation_point_id(
        target_after_crash["transition_operation_token"]
    )
    service._client.set_payload(
        collection_name=OPERATIONS_COLLECTION,
        payload={"child_claim_deadline_utc": "2000-01-01T00:00:00+00:00"},
        points=[child_operation_id],
        wait=True,
    )
    service._client.set_payload(
        collection_name=OPERATIONS_COLLECTION,
        payload={
            "claim_deadline_utc": "2000-01-01T00:00:00+00:00",
            "status": "claimed",
        },
        points=[parent["operation_id"]],
        wait=True,
    )
    service.operations.claim_evidence_plan(
        parent["operation_id"],
        request_fingerprint="request-crash-projection",
        epoch=2,
        nonce="parent-recovered",
    )
    recovered_result = service.confirm(
        candidate["candidate_id"],
        user_id="user-1",
        owner_id="owner-1",
        session_agent_id=None,
        evidence_root_id=parent["evidence_root_id"],
        source_kind="evidence_plan",
        source_fingerprint="request-crash-projection",
        ttl_multiplier=1.0,
        max_score_roots=3,
        idempotency_key=f"{parent['operation_id']}:0",
        expected_revision_id=candidate["revision_id"],
        expected_lineage_id=candidate["lineage_id"],
        expected_content_hash=candidate["content_hash"],
        expected_fact_hash=candidate["fact_hash"],
        parent_operation_id=parent["operation_id"],
        parent_epoch=2,
        parent_nonce="parent-recovered",
    )
    service.operations.checkpoint_evidence_plan(
        parent["operation_id"],
        request_fingerprint="request-crash-projection",
        epoch=2,
        nonce="parent-recovered",
        checkpoints=[
            {
                "ordinal": 0,
                "target_id": candidate["candidate_id"],
                "action": "CONFIRM",
                "status": "confirmed",
                "result": recovered_result,
            }
        ],
    )
    committed = service.operations.commit_evidence_plan(
        parent["operation_id"],
        request_fingerprint="request-crash-projection",
        epoch=2,
        nonce="parent-recovered",
    )
    service.current(
        candidate["candidate_id"],
        user_id="user-1",
        owner_id="owner-1",
        session_agent_id=None,
    )
    target = service._payload(service._read_point(candidate["candidate_id"]))
    assert target["validation_count"] == 1
    assert committed["status"] == "committed"
    assert not any(field in target for field in _TRANSITION_FIELDS)
    operation_points, _ = service._client.scroll(
        collection_name=OPERATIONS_COLLECTION,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="parent_operation_id",
                    match=MatchValue(value=parent["operation_id"]),
                )
            ]
        ),
        limit=10,
        with_payload=True,
        with_vectors=False,
    )
    assert len(operation_points) == 1
    assert operation_points[0].payload["status"] == "committed"


def test_two_child_takeovers_repair_old_target_fence_once(monkeypatch) -> None:
    service = _service()
    candidate = _evidence_candidate(service, "16161616-1616-4161-8161-161616161616")
    parent = service.operations.seal_evidence_plan(
        user_id="user-1",
        owner_id="owner-1",
        evidence_root_id="event-two-takeovers",
        request_fingerprint="request-two-takeovers",
        targets=[
            {
                "ordinal": 0,
                "target_id": candidate["candidate_id"],
                "action": "CONFIRM",
                "lineage_id": candidate["lineage_id"],
                "revision_id": candidate["revision_id"],
                "revision": candidate["revision"],
                "content_hash": candidate["content_hash"],
                "fact_hash": candidate["fact_hash"],
            }
        ],
    )
    service.operations.claim_evidence_plan(
        parent["operation_id"],
        request_fingerprint="request-two-takeovers",
        epoch=1,
        nonce="parent-1",
    )
    confirm_args = {
        "user_id": "user-1",
        "owner_id": "owner-1",
        "session_agent_id": None,
        "evidence_root_id": parent["evidence_root_id"],
        "source_kind": "evidence_plan",
        "source_fingerprint": "request-two-takeovers",
        "ttl_multiplier": 1.0,
        "max_score_roots": 3,
        "idempotency_key": f"{parent['operation_id']}:0",
        "expected_revision_id": candidate["revision_id"],
        "expected_lineage_id": candidate["lineage_id"],
        "expected_content_hash": candidate["content_hash"],
        "expected_fact_hash": candidate["fact_hash"],
        "parent_operation_id": parent["operation_id"],
        "parent_epoch": 1,
        "parent_nonce": "parent-1",
    }
    original_set_payload = service._client.set_payload
    crashed = False

    def crash_projection(**kwargs):
        nonlocal crashed
        if (
            not crashed
            and kwargs.get("collection_name") == "memories"
            and "evidence_root_ids" in (kwargs.get("payload") or {})
        ):
            crashed = True
            raise RuntimeError("first recovery interruption")
        return original_set_payload(**kwargs)

    monkeypatch.setattr(service._client, "set_payload", crash_projection)
    with pytest.raises(RuntimeError, match="first recovery interruption"):
        service.confirm(candidate["candidate_id"], **confirm_args)
    monkeypatch.setattr(service._client, "set_payload", original_set_payload)

    target_after_first = service._payload(
        service._read_point(candidate["candidate_id"], with_vectors=False)
    )
    first_fence = target_after_first["evidence_child_fence"]
    child_operation_id = _operation_point_id(
        target_after_first["transition_operation_token"]
    )
    service._client.set_payload(
        collection_name=OPERATIONS_COLLECTION,
        payload={"child_claim_deadline_utc": "2000-01-01T00:00:00+00:00"},
        points=[child_operation_id],
        wait=True,
    )
    service._client.set_payload(
        collection_name=OPERATIONS_COLLECTION,
        payload={"claim_deadline_utc": "2000-01-01T00:00:00+00:00"},
        points=[parent["operation_id"]],
        wait=True,
    )
    service.operations.claim_evidence_plan(
        parent["operation_id"],
        request_fingerprint="request-two-takeovers",
        epoch=2,
        nonce="parent-2",
    )
    confirm_args.update(parent_epoch=2, parent_nonce="parent-2")
    original_repair = service.operations._repair_child_target_fence
    takeover_crashed = False

    def crash_takeover(record):
        nonlocal takeover_crashed
        if not takeover_crashed:
            takeover_crashed = True
            raise RuntimeError("second recovery interruption")
        return original_repair(record)

    monkeypatch.setattr(
        service.operations, "_repair_child_target_fence", crash_takeover
    )
    with pytest.raises(RuntimeError, match="second recovery interruption"):
        service.confirm(candidate["candidate_id"], **confirm_args)
    monkeypatch.setattr(
        service.operations, "_repair_child_target_fence", original_repair
    )
    target_after_second = service._payload(
        service._read_point(candidate["candidate_id"], with_vectors=False)
    )
    assert target_after_second["evidence_child_fence"] == first_fence

    service._client.set_payload(
        collection_name=OPERATIONS_COLLECTION,
        payload={"child_claim_deadline_utc": "2000-01-01T00:00:00+00:00"},
        points=[child_operation_id],
        wait=True,
    )
    service._client.set_payload(
        collection_name=OPERATIONS_COLLECTION,
        payload={"claim_deadline_utc": "2000-01-01T00:00:00+00:00"},
        points=[parent["operation_id"]],
        wait=True,
    )
    service.operations.claim_evidence_plan(
        parent["operation_id"],
        request_fingerprint="request-two-takeovers",
        epoch=3,
        nonce="parent-3",
    )
    confirm_args.update(parent_epoch=3, parent_nonce="parent-3")
    recovered_result = service.confirm(candidate["candidate_id"], **confirm_args)
    service.operations.checkpoint_evidence_plan(
        parent["operation_id"],
        request_fingerprint="request-two-takeovers",
        epoch=3,
        nonce="parent-3",
        checkpoints=[
            {
                "ordinal": 0,
                "target_id": candidate["candidate_id"],
                "action": "CONFIRM",
                "status": "confirmed",
                "result": recovered_result,
            }
        ],
    )
    committed = service.operations.commit_evidence_plan(
        parent["operation_id"],
        request_fingerprint="request-two-takeovers",
        epoch=3,
        nonce="parent-3",
    )
    target = service._payload(
        service._read_point(candidate["candidate_id"], with_vectors=False)
    )
    assert target["validation_count"] == 1
    assert committed["status"] == "committed"
    assert not any(field in target for field in _TRANSITION_FIELDS)
    child = service.operations.get(target_after_first["transition_operation_token"])
    assert child is not None
    assert child["status"] == "committed"


def test_evidence_commit_requires_complete_terminal_checkpoints() -> None:
    service = _service()
    plan = service.operations.seal_evidence_plan(
        user_id="user-1",
        owner_id="owner-1",
        evidence_root_id="event-complete",
        request_fingerprint="request-complete",
        targets=[
            {"ordinal": 0, "target_id": "one", "action": "SKIP"},
            {"ordinal": 1, "target_id": "two", "action": "SKIP"},
        ],
    )
    service.operations.claim_evidence_plan(
        plan["operation_id"],
        request_fingerprint="request-complete",
        epoch=1,
        nonce="complete-nonce",
    )
    with pytest.raises(EvidenceConflictError):
        service.operations.commit_evidence_plan(
            plan["operation_id"],
            request_fingerprint="request-complete",
            epoch=1,
            nonce="complete-nonce",
            result={
                "checkpoints": [
                    {
                        "ordinal": 0,
                        "target_id": "one",
                        "action": "SKIP",
                        "status": "skipped",
                    }
                ]
            },
        )


def test_planner_prefers_one_consolidated_descendant() -> None:
    service = _service()
    raw = _evidence_candidate(
        service, "99999999-9999-4999-8999-999999999999", text="User lives in Prague"
    )
    consolidated = _evidence_candidate(
        service,
        "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        layer="consolidated",
        derived_from=[raw["candidate_id"]],
        text="User lives in Prague",
    )
    plan = service.plan_evidence(
        [
            {
                "candidate_ids": [raw["candidate_id"], consolidated["candidate_id"]],
                "text": "User lives in Prague",
                "fact_hash": consolidated["fact_hash"],
            }
        ],
        user_id="user-1",
        owner_id="owner-1",
        evidence_root_id="event-canonical",
    )
    assert len(plan) == 1
    assert plan[0]["action"] == "CONFIRM"
    assert plan[0]["target_id"] == consolidated["candidate_id"]


@pytest.mark.parametrize(
    ("decision", "expected"),
    [
        (True, "CONFIRM"),
        (False, "SKIP"),
    ],
)
def test_planner_uses_full_bidirectional_semantic_decision(
    decision: bool, expected: str
) -> None:
    service = _service()
    candidate = _evidence_candidate(
        service,
        "cccccccc-cccc-4ccc-8ccc-cccccccccccc",
        text="The user lives in Prague",
    )
    calls: list[tuple[str, str, str | None]] = []

    def semantic(source: str, target: str, hint: str | None) -> bool:
        calls.append((source, target, hint))
        return decision

    plan = service.operations.plan_evidence(
        [
            {
                "candidate_id": candidate["candidate_id"],
                "source_text": "I reside in Prague.",
                "assertion_text": "The user lives in Prague",
                "content_hash": candidate["content_hash"],
                "fact_hash": candidate["fact_hash"],
            }
        ],
        user_id="user-1",
        owner_id="owner-1",
        evidence_root_id="event-semantic",
        semantic_equivalence=semantic,
    )
    assert plan[0]["action"] == expected
    if expected == "CONFIRM":
        assert (
            plan[0]["source_hash"]
            == hashlib.sha256("The user lives in Prague".encode()).hexdigest()
        )
    assert calls == [
        ("I reside in Prague.", "The user lives in Prague", "The user lives in Prague")
    ]


@pytest.mark.parametrize(
    ("source", "target", "decision"),
    [
        ("I do not live in Prague.", "The user lives in Prague", False),
        ("Alice lives in Prague.", "The user lives in Prague", False),
        ("I reside in Prague.", "The user lives in Prague", True),
        ("I live in Prague and use Rust.", "The user lives in Prague", False),
    ],
)
def test_semantic_planner_preserves_full_equivalence_decision(
    source: str, target: str, decision: bool
) -> None:
    service = _service()
    candidate = _evidence_candidate(
        service,
        "dddddddd-dddd-4ddd-8ddd-dddddddddddd",
        text=target,
    )
    plan = service.operations.plan_evidence(
        [
            {
                "candidate_id": candidate["candidate_id"],
                "source_text": source,
                "assertion_text": target,
                "content_hash": candidate["content_hash"],
                "fact_hash": candidate["fact_hash"],
            }
        ],
        user_id="user-1",
        owner_id="owner-1",
        evidence_root_id="event-semantic-matrix",
        semantic_equivalence=lambda *_: decision,
    )
    assert plan[0]["action"] == ("CONFIRM" if decision else "SKIP")


def test_sealed_target_revised_before_resume_becomes_terminal_skip() -> None:
    revisions = _service()
    candidate = _evidence_candidate(
        revisions, "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb", text="User lives in Prague"
    )
    memory = MemoryService.__new__(MemoryService)
    memory.revisions = revisions
    memory._config = SimpleNamespace(
        memory=SimpleNamespace(
            validation_ttl_multiplier=1.0,
            validation_max_score_roots=3,
        )
    )
    plan = memory.plan_evidence(
        [candidate],
        user_id="user-1",
        owner_id="owner-1",
        evidence_root_id="event-stale",
    )
    sealed = memory.seal_evidence_plan(plan, request_fingerprint="request-stale")
    revisions.operations.claim_evidence_plan(
        sealed["operation_id"],
        request_fingerprint="request-stale",
        epoch=1,
        nonce="execution-1",
    )
    revisions.revise(
        candidate["candidate_id"],
        user_id="user-1",
        owner_id="owner-1",
        session_agent_id=None,
        changes={"data": "User lives in Berlin"},
    )
    result = memory.apply_evidence_plan(
        sealed["operation_id"],
        request_fingerprint="request-stale",
        epoch=1,
        nonce="execution-1",
        user_id="user-1",
        owner_id="owner-1",
    )
    assert result["status"] == "committed"
    assert result["result"]["checkpoints"][0]["status"] == "skipped"
    assert result["result"]["checkpoints"][0]["reason"] == "stale_target"
