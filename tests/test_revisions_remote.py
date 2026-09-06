"""Remote Qdrant concurrency checks for revision claims."""

from __future__ import annotations

import hashlib
import os
import threading
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from mnemory.revisions import (
    _TRANSITION_FIELDS,
    OPERATIONS_COLLECTION,
    EvidenceLeaseLostError,
    RevisionConflictError,
    RevisionService,
    _operation_point_id,
    canonical_fingerprint,
)

QDRANT_URL = os.environ.get("MNEMORY_TEST_QDRANT_URL")
pytestmark = pytest.mark.skipif(
    not QDRANT_URL,
    reason="MNEMORY_TEST_QDRANT_URL is not configured",
)


def _remote_service(client: QdrantClient, collection_name: str) -> RevisionService:
    vector = SimpleNamespace(
        _client=client,
        _config=SimpleNamespace(vector=SimpleNamespace(is_remote=True)),
        collection_name=collection_name,
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


def test_remote_qdrant_allows_one_expected_revision_winner() -> None:
    """Strong filtered claims must produce one active successor."""
    collection_name = f"mnemory_revision_test_{uuid.uuid4().hex}"
    memory_id = str(uuid.uuid4())
    first_client = QdrantClient(url=QDRANT_URL)
    second_client = QdrantClient(url=QDRANT_URL)
    first_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=2, distance=Distance.COSINE),
    )
    first = _remote_service(first_client, collection_name)
    second = _remote_service(second_client, collection_name)
    first_client.upsert(
        collection_name=collection_name,
        points=[
            PointStruct(
                id=memory_id,
                vector=[1.0, 0.0],
                payload={
                    "data": "old",
                    "hash": "old",
                    "user_id": "user-1",
                    "owner_id": "owner-1",
                    **RevisionService.initial_metadata(memory_id),
                },
            )
        ],
        wait=True,
    )
    barrier = threading.Barrier(2)
    results: list[dict] = []
    errors: list[Exception] = []

    def revise(service: RevisionService, key: str, value: str) -> None:
        barrier.wait()
        try:
            results.append(
                service.revise(
                    memory_id,
                    user_id="user-1",
                    owner_id="owner-1",
                    session_agent_id=None,
                    changes={"data": value},
                    expected_revision=1,
                    idempotency_key=key,
                )
            )
        except Exception as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=revise, args=(first, "request-a", "one")),
        threading.Thread(target=revise, args=(second, "request-b", "two")),
    ]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(results) == 1
        assert len(errors) == 1
        assert isinstance(errors[0], RevisionConflictError)
        points = first._lineage_points(memory_id)
        states = [point.payload["revision_state"] for point in points]
        assert states.count("active") == 1
    finally:
        first_client.delete_collection(collection_name)
        if first_client.collection_exists(OPERATIONS_COLLECTION):
            first_client.delete_collection(OPERATIONS_COLLECTION)


def test_remote_qdrant_child_takeover_fences_old_target_write() -> None:
    """A paused child cannot claim a target after a remote child takeover."""
    collection_name = f"mnemory_evidence_child_fence_test_{uuid.uuid4().hex}"
    client = QdrantClient(url=QDRANT_URL)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=2, distance=Distance.COSINE),
    )
    target_id = str(uuid.uuid4())
    service = _remote_service(client, collection_name)
    try:
        client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=target_id,
                    vector=[1.0, 0.0],
                    payload={
                        "data": "User lives in Prague",
                        "hash": "target-hash",
                        "fact_hash": "fact-hash",
                        "user_id": "user-1",
                        "owner_id": "owner-1",
                        "role": "user",
                        "memory_layer": "raw",
                        **RevisionService.initial_metadata(target_id),
                    },
                )
            ],
            wait=True,
        )
        parent = service.operations.seal_evidence_plan(
            user_id="user-1",
            owner_id="owner-1",
            evidence_root_id="remote-child-root",
            request_fingerprint="remote-child-request",
            targets=[],
        )
        service.operations.claim_evidence_plan(
            parent["operation_id"],
            request_fingerprint="remote-child-request",
            epoch=1,
            nonce="parent-1",
        )
        child = canonical_fingerprint(["remote-child", "0"])
        service.operations.write(
            child,
            {
                "status": "prepared",
                "operation_kind": "confirm",
                "parent_operation_id": parent["operation_id"],
                "evidence_child_key": f"{parent['operation_id']}:0",
                "target_revision_id": target_id,
            },
        )
        old = service.operations.claim_evidence_child(
            child,
            parent_operation_id=parent["operation_id"],
            ordinal=0,
            epoch=1,
            nonce="child-1",
        )
        client.set_payload(
            collection_name=OPERATIONS_COLLECTION,
            payload={"child_claim_deadline_utc": "2000-01-01T00:00:00+00:00"},
            points=[_operation_point_id(child)],
            wait=True,
        )
        new = service.operations.claim_evidence_child(
            child,
            parent_operation_id=parent["operation_id"],
            ordinal=0,
            epoch=2,
            nonce="child-2",
        )
        assert old["child_transition_token"] != new["child_transition_token"]
        with pytest.raises(RevisionConflictError):
            service._claim(
                point_id=target_id,
                payload={
                    "user_id": "user-1",
                    "owner_id": "owner-1",
                    "revision": 1,
                    "agent_id": None,
                },
                token=old["child_transition_token"],
                kind="confirm",
                successor_id=None,
                expected_child_token=old["child_transition_token"],
            )
    finally:
        client.delete_collection(collection_name)
        if client.collection_exists(OPERATIONS_COLLECTION):
            client.delete_collection(OPERATIONS_COLLECTION)


def test_remote_qdrant_fsck_audit_binding_has_one_winner() -> None:
    """A check ID must bind to only one immutable audit payload."""
    collection_name = f"mnemory_fsck_audit_test_{uuid.uuid4().hex}"
    first_client = QdrantClient(url=QDRANT_URL)
    second_client = QdrantClient(url=QDRANT_URL)
    first_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=2, distance=Distance.COSINE),
    )
    first = _remote_service(first_client, collection_name)
    second = _remote_service(second_client, collection_name)
    check_id = str(uuid.uuid4())
    barrier = threading.Barrier(2)
    results: list[dict] = []
    errors: list[Exception] = []

    def create(service: RevisionService, basis: str) -> None:
        payload = {
            "audit_check_id": check_id,
            "user_id": "user-1",
            "owner_id": "owner-1",
            "agent_id": None,
            "basis_operation_id": basis,
            "basis_operation_fingerprint": basis,
            "target_revisions": [
                {
                    "memory_id": "memory-1",
                    "revision": 1,
                    "agent_id": None,
                }
            ],
            "target_revision_ids": ["memory-1"],
            "target_state": "complete",
            "target_snapshot_fingerprint": "snapshot-1",
            "issue_signatures": [],
            "summary": {"total": 0},
            "created_at_utc": "2026-01-02T00:00:00+00:00",
            "completed_at_utc": "2026-01-02T00:00:01+00:00",
        }
        barrier.wait()
        try:
            results.append(service.operations.create_fsck_audit(payload))
        except Exception as exc:
            errors.append(exc)

    try:
        threads = [
            threading.Thread(target=create, args=(first, "basis-1")),
            threading.Thread(target=create, args=(second, "basis-2")),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        stored = first.operations.get_fsck_audit(check_id)
        assert stored is not None
        assert stored["basis_operation_id"] in {"basis-1", "basis-2"}
        assert len(results) == 1
        assert len(errors) == 1
        assert isinstance(errors[0], ValueError)
    finally:
        first_client.delete_collection(collection_name=collection_name)
        if first_client.collection_exists(OPERATIONS_COLLECTION):
            first_client.delete_collection(OPERATIONS_COLLECTION)


def test_remote_qdrant_evidence_seal_is_conditionally_idempotent() -> None:
    """The conditional parent insert has one durable winner across clients."""
    collection_name = f"mnemory_evidence_test_{uuid.uuid4().hex}"
    first_client = QdrantClient(url=QDRANT_URL)
    second_client = QdrantClient(url=QDRANT_URL)
    first_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=2, distance=Distance.COSINE),
    )
    first = _remote_service(first_client, collection_name)
    second = _remote_service(second_client, collection_name)
    barrier = threading.Barrier(2)
    results: list[dict] = []
    errors: list[Exception] = []

    def seal(store) -> None:
        barrier.wait()
        try:
            results.append(
                store.operations.seal_evidence_plan(
                    user_id="user-1",
                    owner_id="owner-1",
                    evidence_root_id="remote-event",
                    request_fingerprint="remote-request",
                    targets=[],
                )
            )
        except Exception as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=seal, args=(first,)),
        threading.Thread(target=seal, args=(second,)),
    ]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert errors == []
        assert len({item["operation_id"] for item in results}) == 1
    finally:
        first_client.delete_collection(collection_name)
        if first_client.collection_exists(OPERATIONS_COLLECTION):
            first_client.delete_collection(OPERATIONS_COLLECTION)


def test_remote_qdrant_evidence_conflicting_fingerprints_have_one_winner() -> None:
    """The root-derived parent ID rejects a concurrent different request."""
    collection_name = f"mnemory_evidence_conflict_test_{uuid.uuid4().hex}"
    first_client = QdrantClient(url=QDRANT_URL)
    second_client = QdrantClient(url=QDRANT_URL)
    first_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=2, distance=Distance.COSINE),
    )
    first = _remote_service(first_client, collection_name)
    second = _remote_service(second_client, collection_name)
    barrier = threading.Barrier(2)
    results: list[dict] = []
    errors: list[Exception] = []

    def seal(store: RevisionService, fingerprint: str) -> None:
        barrier.wait()
        try:
            results.append(
                store.operations.seal_evidence_plan(
                    user_id="user-1",
                    owner_id="owner-1",
                    evidence_root_id="remote-conflicting-root",
                    request_fingerprint=fingerprint,
                    targets=[],
                )
            )
        except Exception as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=seal, args=(first, "remote-request-a")),
        threading.Thread(target=seal, args=(second, "remote-request-b")),
    ]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(results) == 1
        assert len(errors) == 1
        assert isinstance(errors[0], RevisionConflictError)
    finally:
        first_client.delete_collection(collection_name)
        if first_client.collection_exists(OPERATIONS_COLLECTION):
            first_client.delete_collection(OPERATIONS_COLLECTION)


def test_remote_qdrant_evidence_checkpoint_recovery_and_takeover() -> None:
    """A lost response can resume; an expired worker cannot mutate."""
    collection_name = f"mnemory_evidence_recovery_test_{uuid.uuid4().hex}"
    client = QdrantClient(url=QDRANT_URL)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=2, distance=Distance.COSINE),
    )
    service = _remote_service(client, collection_name)
    try:
        plan = service.operations.seal_evidence_plan(
            user_id="user-1",
            owner_id="owner-1",
            evidence_root_id="remote-recovery-root",
            request_fingerprint="remote-recovery-request",
            targets=[],
        )
        claimed = service.operations.claim_evidence_plan(
            plan["operation_id"],
            request_fingerprint="remote-recovery-request",
            epoch=1,
            nonce="remote-old",
        )
        client.set_payload(
            collection_name=OPERATIONS_COLLECTION,
            payload={"claim_deadline_utc": "2000-01-01T00:00:00+00:00"},
            points=[plan["operation_id"]],
            wait=True,
        )
        service.operations.claim_evidence_plan(
            plan["operation_id"],
            request_fingerprint="remote-recovery-request",
            epoch=2,
            nonce="remote-new",
        )
        with pytest.raises(EvidenceLeaseLostError):
            service.operations.verify_evidence_claim(
                claimed["operation_id"],
                request_fingerprint="remote-recovery-request",
                epoch=1,
                nonce="remote-old",
            )
        resumed = service.operations.checkpoint_evidence_plan(
            plan["operation_id"],
            request_fingerprint="remote-recovery-request",
            epoch=2,
            nonce="remote-new",
            checkpoints=[],
        )
        assert resumed["status"] == "claimed"
    finally:
        client.delete_collection(collection_name)
        if client.collection_exists(OPERATIONS_COLLECTION):
            client.delete_collection(OPERATIONS_COLLECTION)


def test_remote_qdrant_planner_suppresses_raw_ancestor_for_consolidated_target() -> (
    None
):
    """Remote planning selects one target for an ancestry family."""
    collection_name = f"mnemory_evidence_ancestry_test_{uuid.uuid4().hex}"
    client = QdrantClient(url=QDRANT_URL)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=2, distance=Distance.COSINE),
    )
    raw_id = str(uuid.uuid4())
    consolidated_id = str(uuid.uuid4())
    text = "User lives in Prague"
    payload_base = {
        "data": text,
        "hash": hashlib.sha256(text.encode()).hexdigest(),
        "fact_hash": hashlib.sha256(text.casefold().encode()).hexdigest(),
        "user_id": "user-1",
        "owner_id": "owner-1",
        "role": "user",
        "memory_layer": "raw",
        **RevisionService.initial_metadata(raw_id),
    }
    try:
        client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(id=raw_id, vector=[1.0, 0.0], payload=payload_base),
                PointStruct(
                    id=consolidated_id,
                    vector=[1.0, 0.0],
                    payload={
                        **payload_base,
                        "memory_layer": "consolidated",
                        "lineage_id": consolidated_id,
                        "derived_from": [raw_id],
                    },
                ),
            ],
            wait=True,
        )
        service = _remote_service(client, collection_name)
        plan = service.operations.plan_evidence(
            [
                {
                    "candidate_id": raw_id,
                    "text": text,
                    "fact_hash": payload_base["fact_hash"],
                },
                {
                    "candidate_id": consolidated_id,
                    "text": text,
                    "fact_hash": payload_base["fact_hash"],
                },
            ],
            user_id="user-1",
            owner_id="owner-1",
            evidence_root_id="remote-ancestry-root",
        )
        confirms = [item for item in plan if item["action"] == "CONFIRM"]
        assert len(confirms) == 1
        assert confirms[0]["target_id"] == consolidated_id
    finally:
        client.delete_collection(collection_name)
        if client.collection_exists(OPERATIONS_COLLECTION):
            client.delete_collection(OPERATIONS_COLLECTION)


def test_remote_qdrant_same_child_retry_repairs_interrupted_fence() -> None:
    """A child update lost before target fencing converges on retry."""
    collection_name = f"mnemory_evidence_repair_test_{uuid.uuid4().hex}"
    client = QdrantClient(url=QDRANT_URL)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=2, distance=Distance.COSINE),
    )
    target_id = str(uuid.uuid4())
    service = _remote_service(client, collection_name)
    try:
        client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=target_id,
                    vector=[1.0, 0.0],
                    payload={
                        "data": "User lives in Prague",
                        "user_id": "user-1",
                        "owner_id": "owner-1",
                    },
                )
            ],
            wait=True,
        )
        parent = service.operations.seal_evidence_plan(
            user_id="user-1",
            owner_id="owner-1",
            evidence_root_id="remote-repair-root",
            request_fingerprint="remote-repair-request",
            targets=[],
        )
        child_token = canonical_fingerprint(["remote-repair-child"])
        service.operations.write(
            child_token,
            {
                "status": "prepared",
                "operation_kind": "confirm",
                "parent_operation_id": parent["operation_id"],
                "evidence_child_key": f"{parent['operation_id']}:0",
                "target_revision_id": target_id,
            },
        )
        service.operations.claim_evidence_child(
            child_token,
            parent_operation_id=parent["operation_id"],
            ordinal=0,
            epoch=1,
            nonce="child-old",
        )
        client.set_payload(
            collection_name=OPERATIONS_COLLECTION,
            payload={"child_claim_deadline_utc": "2000-01-01T00:00:00+00:00"},
            points=[_operation_point_id(child_token)],
            wait=True,
        )
        original_repair = service.operations._repair_child_target_fence
        crashed = False

        def crash_once(record):
            nonlocal crashed
            if not crashed:
                crashed = True
                raise RuntimeError("remote fence repair crash")
            return original_repair(record)

        service.operations._repair_child_target_fence = crash_once
        with pytest.raises(RuntimeError, match="remote fence repair crash"):
            service.operations.claim_evidence_child(
                child_token,
                parent_operation_id=parent["operation_id"],
                ordinal=0,
                epoch=2,
                nonce="child-new",
            )
        service.operations._repair_child_target_fence = original_repair
        repaired = service.operations.claim_evidence_child(
            child_token,
            parent_operation_id=parent["operation_id"],
            ordinal=0,
            epoch=2,
            nonce="child-new",
        )
        target = client.retrieve(
            collection_name=collection_name,
            ids=[target_id],
            with_payload=True,
            with_vectors=False,
            consistency="all",
        )[0]
        assert (
            target.payload["evidence_child_fence"] == repaired["child_transition_token"]
        )
    finally:
        client.delete_collection(collection_name)
        if client.collection_exists(OPERATIONS_COLLECTION):
            client.delete_collection(OPERATIONS_COLLECTION)


def test_remote_qdrant_crash_after_target_claim_recovers_once() -> None:
    """A remote target claim remains recoverable through its child operation."""
    collection_name = f"mnemory_evidence_crash_test_{uuid.uuid4().hex}"
    client = QdrantClient(url=QDRANT_URL)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=2, distance=Distance.COSINE),
    )
    target_id = str(uuid.uuid4())
    service = _remote_service(client, collection_name)
    text = "User lives in Prague"
    target = {
        "candidate_id": target_id,
        "lineage_id": target_id,
        "revision": 1,
        "revision_id": target_id,
        "content_hash": hashlib.sha256(text.encode()).hexdigest(),
        "fact_hash": hashlib.sha256(text.casefold().encode()).hexdigest(),
    }
    try:
        client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=target_id,
                    vector=[1.0, 0.0],
                    payload={
                        "data": text,
                        "hash": target["content_hash"],
                        "fact_hash": target["fact_hash"],
                        "user_id": "user-1",
                        "owner_id": "owner-1",
                        "role": "user",
                        "memory_layer": "raw",
                        **RevisionService.initial_metadata(target_id),
                    },
                )
            ],
            wait=True,
        )
        parent = service.operations.seal_evidence_plan(
            user_id="user-1",
            owner_id="owner-1",
            evidence_root_id="remote-crash-root",
            request_fingerprint="remote-crash-request",
            targets=[
                {
                    "ordinal": 0,
                    "target_id": target_id,
                    "action": "CONFIRM",
                    "lineage_id": target_id,
                    "revision_id": target_id,
                    "revision": 1,
                    "content_hash": target["content_hash"],
                    "fact_hash": target["fact_hash"],
                }
            ],
        )
        service.operations.claim_evidence_plan(
            parent["operation_id"],
            request_fingerprint="remote-crash-request",
            epoch=1,
            nonce="remote-parent",
        )
        original_set_payload = client.set_payload
        crashed = False

        def crash_projection(**kwargs):
            nonlocal crashed
            if (
                not crashed
                and kwargs.get("collection_name") == collection_name
                and "evidence_root_ids" in (kwargs.get("payload") or {})
            ):
                crashed = True
                raise RuntimeError("remote crash after target claim")
            return original_set_payload(**kwargs)

        client.set_payload = crash_projection
        with pytest.raises(RuntimeError, match="remote crash after target claim"):
            service.confirm(
                target_id,
                user_id="user-1",
                owner_id="owner-1",
                session_agent_id=None,
                evidence_root_id=parent["evidence_root_id"],
                source_kind="evidence_plan",
                source_fingerprint="remote-crash-request",
                ttl_multiplier=1.0,
                max_score_roots=3,
                idempotency_key=f"{parent['operation_id']}:0",
                expected_revision_id=target_id,
                expected_lineage_id=target_id,
                expected_content_hash=target["content_hash"],
                expected_fact_hash=target["fact_hash"],
                parent_operation_id=parent["operation_id"],
                parent_epoch=1,
                parent_nonce="remote-parent",
            )
        client.set_payload = original_set_payload
        crashed_target = service._read_point(target_id, with_vectors=False)
        child_operation_id = _operation_point_id(
            crashed_target.payload["transition_operation_token"]
        )
        client.set_payload(
            collection_name=OPERATIONS_COLLECTION,
            payload={"child_claim_deadline_utc": "2000-01-01T00:00:00+00:00"},
            points=[child_operation_id],
            wait=True,
        )
        client.set_payload(
            collection_name=OPERATIONS_COLLECTION,
            payload={"claim_deadline_utc": "2000-01-01T00:00:00+00:00"},
            points=[parent["operation_id"]],
            wait=True,
        )
        service.operations.claim_evidence_plan(
            parent["operation_id"],
            request_fingerprint="remote-crash-request",
            epoch=2,
            nonce="remote-parent-recovered",
        )
        recovered_result = service.confirm(
            target_id,
            user_id="user-1",
            owner_id="owner-1",
            session_agent_id=None,
            evidence_root_id=parent["evidence_root_id"],
            source_kind="evidence_plan",
            source_fingerprint="remote-crash-request",
            ttl_multiplier=1.0,
            max_score_roots=3,
            idempotency_key=f"{parent['operation_id']}:0",
            expected_revision_id=target_id,
            expected_lineage_id=target_id,
            expected_content_hash=target["content_hash"],
            expected_fact_hash=target["fact_hash"],
            parent_operation_id=parent["operation_id"],
            parent_epoch=2,
            parent_nonce="remote-parent-recovered",
        )
        checkpoint = {
            "ordinal": 0,
            "target_id": target_id,
            "action": "CONFIRM",
            "status": "confirmed",
            "result": recovered_result,
        }
        service.operations.checkpoint_evidence_plan(
            parent["operation_id"],
            request_fingerprint="remote-crash-request",
            epoch=2,
            nonce="remote-parent-recovered",
            checkpoints=[checkpoint],
        )
        committed = service.operations.commit_evidence_plan(
            parent["operation_id"],
            request_fingerprint="remote-crash-request",
            epoch=2,
            nonce="remote-parent-recovered",
        )
        service.current(
            target_id,
            user_id="user-1",
            owner_id="owner-1",
            session_agent_id=None,
        )
        recovered = service._read_point(target_id, with_vectors=False)
        assert recovered is not None
        assert recovered.payload["validation_count"] == 1
        assert committed["status"] == "committed"
    finally:
        client.set_payload = (
            original_set_payload
            if "original_set_payload" in locals()
            else client.set_payload
        )
        client.delete_collection(collection_name)
        if client.collection_exists(OPERATIONS_COLLECTION):
            client.delete_collection(OPERATIONS_COLLECTION)


def test_remote_qdrant_two_child_takeovers_repair_old_fence_once() -> None:
    """Two remote takeover crashes converge on one confirmation."""
    collection_name = f"mnemory_evidence_takeover_test_{uuid.uuid4().hex}"
    client = QdrantClient(url=QDRANT_URL)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=2, distance=Distance.COSINE),
    )
    target_id = str(uuid.uuid4())
    service = _remote_service(client, collection_name)
    text = "User lives in Prague"
    content_hash = hashlib.sha256(text.encode()).hexdigest()
    fact_hash = hashlib.sha256(text.casefold().encode()).hexdigest()
    original_set_payload = client.set_payload
    try:
        client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=target_id,
                    vector=[1.0, 0.0],
                    payload={
                        "data": text,
                        "hash": content_hash,
                        "fact_hash": fact_hash,
                        "user_id": "user-1",
                        "owner_id": "owner-1",
                        "role": "user",
                        "memory_layer": "raw",
                        **RevisionService.initial_metadata(target_id),
                    },
                )
            ],
            wait=True,
        )
        parent = service.operations.seal_evidence_plan(
            user_id="user-1",
            owner_id="owner-1",
            evidence_root_id="remote-two-takeovers",
            request_fingerprint="remote-two-takeovers",
            targets=[
                {
                    "ordinal": 0,
                    "target_id": target_id,
                    "action": "CONFIRM",
                    "lineage_id": target_id,
                    "revision_id": target_id,
                    "revision": 1,
                    "content_hash": content_hash,
                    "fact_hash": fact_hash,
                }
            ],
        )
        service.operations.claim_evidence_plan(
            parent["operation_id"],
            request_fingerprint="remote-two-takeovers",
            epoch=1,
            nonce="remote-parent-1",
        )
        confirm_args = {
            "user_id": "user-1",
            "owner_id": "owner-1",
            "session_agent_id": None,
            "evidence_root_id": parent["evidence_root_id"],
            "source_kind": "evidence_plan",
            "source_fingerprint": "remote-two-takeovers",
            "ttl_multiplier": 1.0,
            "max_score_roots": 3,
            "idempotency_key": f"{parent['operation_id']}:0",
            "expected_revision_id": target_id,
            "expected_lineage_id": target_id,
            "expected_content_hash": content_hash,
            "expected_fact_hash": fact_hash,
            "parent_operation_id": parent["operation_id"],
            "parent_epoch": 1,
            "parent_nonce": "remote-parent-1",
        }
        crashed = False

        def crash_projection(**kwargs):
            nonlocal crashed
            if (
                not crashed
                and kwargs.get("collection_name") == collection_name
                and "evidence_root_ids" in (kwargs.get("payload") or {})
            ):
                crashed = True
                raise RuntimeError("first remote recovery interruption")
            return original_set_payload(**kwargs)

        client.set_payload = crash_projection
        with pytest.raises(RuntimeError, match="first remote recovery interruption"):
            service.confirm(target_id, **confirm_args)
        client.set_payload = original_set_payload

        target_after_first = service._read_point(target_id, with_vectors=False)
        assert target_after_first is not None
        first_fence = target_after_first.payload["evidence_child_fence"]
        child_operation_token = target_after_first.payload["transition_operation_token"]
        child_operation_id = _operation_point_id(child_operation_token)
        client.set_payload(
            collection_name=OPERATIONS_COLLECTION,
            payload={"child_claim_deadline_utc": "2000-01-01T00:00:00+00:00"},
            points=[child_operation_id],
            wait=True,
        )
        client.set_payload(
            collection_name=OPERATIONS_COLLECTION,
            payload={"claim_deadline_utc": "2000-01-01T00:00:00+00:00"},
            points=[parent["operation_id"]],
            wait=True,
        )
        service.operations.claim_evidence_plan(
            parent["operation_id"],
            request_fingerprint="remote-two-takeovers",
            epoch=2,
            nonce="remote-parent-2",
        )
        confirm_args.update(parent_epoch=2, parent_nonce="remote-parent-2")
        original_repair = service.operations._repair_child_target_fence
        takeover_crashed = False

        def crash_takeover(record):
            nonlocal takeover_crashed
            if not takeover_crashed:
                takeover_crashed = True
                raise RuntimeError("second remote recovery interruption")
            return original_repair(record)

        service.operations._repair_child_target_fence = crash_takeover
        with pytest.raises(RuntimeError, match="second remote recovery interruption"):
            service.confirm(target_id, **confirm_args)
        service.operations._repair_child_target_fence = original_repair
        target_after_second = service._read_point(target_id, with_vectors=False)
        assert target_after_second is not None
        assert target_after_second.payload["evidence_child_fence"] == first_fence

        client.set_payload(
            collection_name=OPERATIONS_COLLECTION,
            payload={"child_claim_deadline_utc": "2000-01-01T00:00:00+00:00"},
            points=[child_operation_id],
            wait=True,
        )
        client.set_payload(
            collection_name=OPERATIONS_COLLECTION,
            payload={"claim_deadline_utc": "2000-01-01T00:00:00+00:00"},
            points=[parent["operation_id"]],
            wait=True,
        )
        service.operations.claim_evidence_plan(
            parent["operation_id"],
            request_fingerprint="remote-two-takeovers",
            epoch=3,
            nonce="remote-parent-3",
        )
        confirm_args.update(parent_epoch=3, parent_nonce="remote-parent-3")
        recovered_result = service.confirm(target_id, **confirm_args)
        service.operations.checkpoint_evidence_plan(
            parent["operation_id"],
            request_fingerprint="remote-two-takeovers",
            epoch=3,
            nonce="remote-parent-3",
            checkpoints=[
                {
                    "ordinal": 0,
                    "target_id": target_id,
                    "action": "CONFIRM",
                    "status": "confirmed",
                    "result": recovered_result,
                }
            ],
        )
        committed = service.operations.commit_evidence_plan(
            parent["operation_id"],
            request_fingerprint="remote-two-takeovers",
            epoch=3,
            nonce="remote-parent-3",
        )
        recovered = service._read_point(target_id, with_vectors=False)
        assert recovered is not None
        assert recovered.payload["validation_count"] == 1
        assert committed["status"] == "committed"
        assert not any(field in recovered.payload for field in _TRANSITION_FIELDS)
        child = service.operations.get(child_operation_token)
        assert child is not None
        assert child["status"] == "committed"
    finally:
        client.set_payload = original_set_payload
        client.delete_collection(collection_name)
        if client.collection_exists(OPERATIONS_COLLECTION):
            client.delete_collection(OPERATIONS_COLLECTION)


@pytest.mark.parametrize("layer", ["raw", "consolidated"])
def test_remote_qdrant_legacy_fact_hash_target_recovers_once(layer: str) -> None:
    """Legacy targets without fact_hash use derived verification remotely."""
    collection_name = f"mnemory_legacy_fact_test_{uuid.uuid4().hex}"
    client = QdrantClient(url=QDRANT_URL)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=2, distance=Distance.COSINE),
    )
    target_id = str(uuid.uuid4())
    unrelated_id = str(uuid.uuid4())
    service = _remote_service(client, collection_name)
    text = "User lives in Prague"
    content_hash = hashlib.sha256(text.encode()).hexdigest()
    fact_hash = hashlib.sha256(text.casefold().encode()).hexdigest()
    original_set_payload = client.set_payload
    try:
        client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=target_id,
                    vector=[1.0, 0.0],
                    payload={
                        "data": text,
                        "hash": content_hash,
                        "user_id": "user-1",
                        "owner_id": "owner-1",
                        "role": "user",
                        "memory_layer": layer,
                        **RevisionService.initial_metadata(target_id),
                    },
                ),
                PointStruct(
                    id=unrelated_id,
                    vector=[0.0, 1.0],
                    payload={
                        "data": text,
                        "hash": content_hash,
                        "user_id": "user-1",
                        "owner_id": "owner-1",
                        "role": "user",
                        "memory_layer": layer,
                        **RevisionService.initial_metadata(unrelated_id),
                    },
                ),
            ],
            wait=True,
        )
        parent = service.operations.seal_evidence_plan(
            user_id="user-1",
            owner_id="owner-1",
            evidence_root_id=f"remote-legacy-fact-{layer}",
            request_fingerprint=f"remote-legacy-fact-request-{layer}",
            targets=[
                {
                    "ordinal": 0,
                    "target_id": target_id,
                    "action": "CONFIRM",
                    "lineage_id": target_id,
                    "revision_id": target_id,
                    "revision": 1,
                    "content_hash": content_hash,
                    "fact_hash": fact_hash,
                }
            ],
        )
        service.operations.claim_evidence_plan(
            parent["operation_id"],
            request_fingerprint=f"remote-legacy-fact-request-{layer}",
            epoch=1,
            nonce=f"remote-legacy-parent-{layer}",
        )
        confirm_args = {
            "user_id": "user-1",
            "owner_id": "owner-1",
            "session_agent_id": None,
            "evidence_root_id": parent["evidence_root_id"],
            "source_kind": "evidence_plan",
            "source_fingerprint": f"remote-legacy-fact-request-{layer}",
            "ttl_multiplier": 1.0,
            "max_score_roots": 3,
            "idempotency_key": f"{parent['operation_id']}:0",
            "expected_revision_id": target_id,
            "expected_lineage_id": target_id,
            "expected_content_hash": content_hash,
            "expected_fact_hash": fact_hash,
            "parent_operation_id": parent["operation_id"],
            "parent_epoch": 1,
            "parent_nonce": f"remote-legacy-parent-{layer}",
        }
        crashed = False

        def crash_projection(**kwargs):
            nonlocal crashed
            if (
                not crashed
                and kwargs.get("collection_name") == collection_name
                and "evidence_root_ids" in (kwargs.get("payload") or {})
            ):
                crashed = True
                raise RuntimeError("remote legacy fact recovery interruption")
            return original_set_payload(**kwargs)

        client.set_payload = crash_projection
        with pytest.raises(
            RuntimeError, match="remote legacy fact recovery interruption"
        ):
            service.confirm(target_id, **confirm_args)
        client.set_payload = original_set_payload

        crashed_target = service._read_point(target_id, with_vectors=False)
        assert crashed_target is not None
        child_operation_id = _operation_point_id(
            crashed_target.payload["transition_operation_token"]
        )
        client.set_payload(
            collection_name=OPERATIONS_COLLECTION,
            payload={"child_claim_deadline_utc": "2000-01-01T00:00:00+00:00"},
            points=[child_operation_id],
            wait=True,
        )
        client.set_payload(
            collection_name=OPERATIONS_COLLECTION,
            payload={"claim_deadline_utc": "2000-01-01T00:00:00+00:00"},
            points=[parent["operation_id"]],
            wait=True,
        )
        service.operations.claim_evidence_plan(
            parent["operation_id"],
            request_fingerprint=f"remote-legacy-fact-request-{layer}",
            epoch=2,
            nonce=f"remote-legacy-parent-recovered-{layer}",
        )
        confirm_args.update(
            parent_epoch=2,
            parent_nonce=f"remote-legacy-parent-recovered-{layer}",
        )
        recovered = service.confirm(target_id, **confirm_args)
        service.operations.checkpoint_evidence_plan(
            parent["operation_id"],
            request_fingerprint=f"remote-legacy-fact-request-{layer}",
            epoch=2,
            nonce=f"remote-legacy-parent-recovered-{layer}",
            checkpoints=[
                {
                    "ordinal": 0,
                    "target_id": target_id,
                    "action": "CONFIRM",
                    "status": "confirmed",
                    "result": recovered,
                }
            ],
        )
        committed = service.operations.commit_evidence_plan(
            parent["operation_id"],
            request_fingerprint=f"remote-legacy-fact-request-{layer}",
            epoch=2,
            nonce=f"remote-legacy-parent-recovered-{layer}",
        )
        target = service._read_point(target_id, with_vectors=False)
        unrelated = service._read_point(unrelated_id, with_vectors=False)
        assert target is not None
        assert unrelated is not None
        assert recovered["status"] == "confirmed"
        assert target.payload["validation_count"] == 1
        assert committed["status"] == "committed"
        assert not any(field in target.payload for field in _TRANSITION_FIELDS)
        assert not any(field in unrelated.payload for field in _TRANSITION_FIELDS)
        assert unrelated.payload.get("validation_count", 0) == 0
    finally:
        client.set_payload = original_set_payload
        client.delete_collection(collection_name)
        if client.collection_exists(OPERATIONS_COLLECTION):
            client.delete_collection(OPERATIONS_COLLECTION)
