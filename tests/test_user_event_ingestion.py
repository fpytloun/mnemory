"""Contract tests for trusted shared user-event ingestion."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from jwt import InvalidTokenError
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from mnemory.api.evidence import derive_evidence_root
from mnemory.api.schemas import EvidenceRememberRequest, UserEventRememberRequest
from mnemory.api.user_events import (
    USER_EVENT_PATH,
    _claims_match_body,
    canonical_request_hash,
)
from mnemory.auth import CognisJWTValidator
from mnemory.config import MemoryConfig
from mnemory.memory import MemoryService
from mnemory.revisions import (
    EvidenceConflictError,
    EvidenceLeaseLostError,
    RevisionOperationStore,
    RevisionService,
)
from mnemory.storage.vector import VectorStore

FIXTURE = Path(__file__).parent / "contract/fixtures/evidence_remember_v1.json"
os.environ.setdefault("LLM_API_KEY", "test-key")


def _keypair(tmp_path: Path) -> tuple[object, str]:
    private = ec.generate_private_key(ec.SECP256R1())
    public = private.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    path = tmp_path / "cognis-public.pem"
    path.write_bytes(public)
    return private, str(path)


def _user_event_token(
    private: object, body: dict, overrides: dict | None = None
) -> str:
    now = int(time.time())
    claims = {
        "iss": "cognis",
        "aud": "mnemory",
        "typ": "user_event",
        "scope": "mnemory:remember:user",
        "evop": "remember",
        "ver": 1,
        "sub": body["actor"]["user_id"],
        "aow": body["actor"]["owner_id"],
        "evt": body["event"]["id"],
        "event_hash": body["event"]["event_hash"],
        "request_hash": canonical_request_hash(body),
        "evidence_root": derive_evidence_root(body),
        "cognis_session_id": body["event"]["cognis_session_id"],
        "conversation_id": body["event"]["conversation_id"],
        "turn_id": body["event"]["turn_id"],
        "jti": "user-event-jti",
        "iat": now,
        "nbf": now,
        "exp": now + 60,
    }
    if overrides:
        claims.update(overrides)
    return jwt.encode(
        claims,
        private,
        algorithm="ES256",
    )


def _middleware_client(
    tmp_path: Path, monkeypatch
) -> tuple[TestClient, dict, object, str]:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    private, public_path = _keypair(tmp_path)
    from mnemory import server

    async def handler(request: Request) -> JSONResponse:
        return JSONResponse({"claims": request.state.user_event_claims})

    app = Starlette(
        routes=[
            Route(USER_EVENT_PATH, handler, methods=["POST"]),
            Route("/api/evidence/remember/v1", handler, methods=["POST"]),
            Route("/other", handler, methods=["GET"]),
        ],
        middleware=[Middleware(server.EvidenceAuthMiddleware)],
    )
    monkeypatch.setattr(
        server,
        "_get_config",
        lambda: SimpleNamespace(
            server=SimpleNamespace(jwt_public_key=public_path, jwks_url="")
        ),
    )
    return TestClient(app), fixture["body"], private, public_path


def test_user_event_jwt_is_scope_and_route_confined(
    tmp_path: Path, monkeypatch
) -> None:
    client, body, private, _ = _middleware_client(tmp_path, monkeypatch)
    token = _user_event_token(private, body)
    with client:
        assert (
            client.post(
                USER_EVENT_PATH,
                headers={"Authorization": f"Bearer {token}"},
            ).status_code
            == 200
        )
        assert (
            client.get(
                "/other",
                headers={
                    "Authorization": f"Bearer {token}",
                    "X-API-Key": "configured-api-key",
                },
            ).status_code
            == 403
        )


def test_user_event_jwt_accepts_only_matching_scope_and_rejects_agent_headers(
    tmp_path: Path, monkeypatch
) -> None:
    client, body, private, public_path = _middleware_client(tmp_path, monkeypatch)
    token = _user_event_token(private, body)
    with client:
        response = client.post(
            USER_EVENT_PATH,
            headers={
                "Authorization": f"Bearer {token}",
                "X-Agent-Id": "agent-a",
            },
        )
    assert response.status_code == 401

    validator = CognisJWTValidator(public_key_path=public_path)
    evidence_token = jwt.encode(
        {
            "iss": "cognis",
            "aud": "mnemory",
            "typ": "user_event",
            "scope": "mnemory:evidence",
            "evop": "remember",
            "ver": 1,
            "sub": body["actor"]["user_id"],
            "aow": body["actor"]["owner_id"],
            "evt": body["event"]["id"],
            "event_hash": body["event"]["event_hash"],
            "request_hash": canonical_request_hash(body),
            "evidence_root": derive_evidence_root(body),
            "cognis_session_id": body["event"]["cognis_session_id"],
            "conversation_id": body["event"]["conversation_id"],
            "turn_id": body["event"]["turn_id"],
            "jti": "evidence-jti",
            "iat": int(time.time()),
            "nbf": int(time.time()),
            "exp": int(time.time()) + 60,
        },
        private,
        algorithm="ES256",
    )
    with pytest.raises(InvalidTokenError):
        validator.validate_user_event(evidence_token)


@pytest.mark.parametrize(
    ("claim", "value"),
    [
        ("iss", "other"),
        ("aud", "other"),
        ("typ", "other"),
        ("evop", "other"),
        ("ver", 2),
        ("exp", 1),
    ],
)
def test_user_event_jwt_rejects_security_claim_substitutions(
    tmp_path: Path, monkeypatch, claim: str, value: object
) -> None:
    _client, body, private, public_path = _middleware_client(tmp_path, monkeypatch)
    token = _user_event_token(private, body, {claim: value})
    with pytest.raises(InvalidTokenError):
        CognisJWTValidator(public_key_path=public_path).validate_user_event(token)


@pytest.mark.parametrize(
    "claim",
    [
        "sub",
        "aow",
        "evt",
        "event_hash",
        "request_hash",
        "evidence_root",
        "cognis_session_id",
        "conversation_id",
        "turn_id",
    ],
)
def test_user_event_jwt_rejects_body_binding_substitutions(
    tmp_path: Path, monkeypatch, claim: str
) -> None:
    _client, body, private, _ = _middleware_client(tmp_path, monkeypatch)
    value = "other-user" if claim in {"sub", "aow"} else "other"
    token = _user_event_token(private, body, {claim: value})
    claims = jwt.decode(token, options={"verify_signature": False})
    parsed = UserEventRememberRequest.model_validate(body)
    assert not _claims_match_body(
        claims,
        parsed,
        canonical_request_hash(parsed.model_dump(mode="json")),
    )


def test_user_event_jwt_rejects_both_agent_headers_and_cross_route_scopes(
    tmp_path: Path, monkeypatch
) -> None:
    client, body, private, _ = _middleware_client(tmp_path, monkeypatch)
    user_token = _user_event_token(private, body)
    evidence_token = _user_event_token(private, body, {"scope": "mnemory:evidence"})
    with client:
        assert (
            client.post(
                USER_EVENT_PATH,
                headers={
                    "Authorization": f"Bearer {user_token}",
                    "X-Agent-Id": "agent-a",
                    "X-Agent-Owner": "owner-a",
                },
            ).status_code
            == 401
        )
        assert (
            client.post(
                USER_EVENT_PATH,
                headers={"Authorization": f"Bearer {evidence_token}"},
            ).status_code
            == 401
        )
        assert (
            client.post(
                "/api/evidence/remember/v1",
                headers={"Authorization": f"Bearer {user_token}"},
            ).status_code
            == 401
        )


def test_user_event_body_and_route_hash_are_strict() -> None:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    body = EvidenceRememberRequest.model_validate(fixture["body"])
    with pytest.raises(ValueError):
        EvidenceRememberRequest.model_validate(
            {**fixture["body"], "context": "forbidden"}
        )
    assert (
        canonical_request_hash(body.model_dump(mode="json")) != fixture["request_hash"]
    )
    assert _claims_match_body(
        {
            "sub": body.actor.user_id,
            "aow": body.actor.owner_id,
            "evt": body.event.id,
            "event_hash": body.event.event_hash,
            "request_hash": canonical_request_hash(body.model_dump(mode="json")),
            "evidence_root": derive_evidence_root(body.model_dump(mode="json")),
            "cognis_session_id": body.event.cognis_session_id,
            "conversation_id": body.event.conversation_id,
            "turn_id": body.event.turn_id,
        },
        body,
        canonical_request_hash(body.model_dump(mode="json")),
    )


def test_ingestion_sets_shared_raw_provenance_without_validation() -> None:
    class Operations:
        def __init__(self) -> None:
            self.status = "prepared"
            self.completed: dict | None = None
            self.claim_status = "claimed"

        @staticmethod
        def user_event_operation_id(**kwargs):
            return "operation-id"

        @staticmethod
        def user_event_content_fingerprint(**kwargs):
            return "content-fingerprint"

        @staticmethod
        def user_event_content_claim_id(_fingerprint):
            return "content-claim-id"

        def prepare_user_event_ingestion(self, **kwargs):
            if self.status == "committed":
                return {
                    "status": "committed",
                    "memory_id": "content-claim-id",
                    "result": self.completed,
                }
            return {"status": self.status}

        def claim_user_event_content(self, **kwargs):
            return {
                "claim_owner": "operation-id",
                "claim_epoch": 1,
                "claim_nonce": "nonce-1",
                "status": self.claim_status,
            }

        def renew_user_event_content_claim(self, *args, **kwargs):
            return {
                "claim_owner": "operation-id",
                "claim_epoch": 1,
                "claim_nonce": "nonce-1",
                "status": "claimed",
            }

        def complete_user_event_content_claim(self, *args, **kwargs):
            self.claim_status = "committed"
            return {"status": "committed"}

        def complete_user_event_ingestion(self, operation_id, **kwargs):
            self.status = "committed"
            self.completed = kwargs["result"]
            return {"status": "committed", "result": self.completed}

    operations = Operations()
    service = MemoryService.__new__(MemoryService)
    service.revisions = SimpleNamespace(operations=operations)
    service._user_locks = {}
    service._locks_lock = threading.Lock()
    service._max_user_locks = 10
    service._core_cache = SimpleNamespace(invalidate_prefix=lambda _: None)
    service._category_cache = SimpleNamespace(invalidate=lambda _: None)
    inserted = False

    def strict_memory(_memory_id):
        if not inserted:
            return None
        return {
            "id": "content-claim-id",
            "memory": "I live in Prague.",
            "user_id": "user-1",
            "owner_id": "owner-1",
            "metadata": {
                "source_kind": "raw_user_message",
                "memory_layer": "raw",
            },
        }

    service.vector = SimpleNamespace(
        get_by_id=lambda _: None,
        get_by_id_strict=strict_memory,
        embedding=SimpleNamespace(embed=lambda _: [1.0]),
        search_similar=lambda *args, **kwargs: [],
    )
    captured: dict = {}
    add_count = 0

    def add_direct(*args, **kwargs):
        nonlocal add_count, inserted
        add_count += 1
        inserted = True
        captured.update(kwargs)
        return {"results": [{"id": "operation-id", "memory": args[0], "event": "ADD"}]}

    service._add_direct = add_direct
    result = service.ingest_trusted_user_event(
        content="I live in Prague.",
        user_id="user-1",
        owner_id="owner-1",
        evidence_root_id="root-1",
        request_hash="request-1",
        source_event={"event_id": "event-1"},
    )
    assert result["status"] == "accepted"
    assert captured["agent_id"] is None
    assert captured["role"] == "user"
    assert captured["memory_layer"] == "raw"
    assert captured["source_kind"] == "raw_user_message"
    assert captured["validation_eligible"] is True
    assert captured["evidence_root_id"] == "root-1"
    replay = service.ingest_trusted_user_event(
        content="I live in Prague.",
        user_id="user-1",
        owner_id="owner-1",
        evidence_root_id="root-1",
        request_hash="request-1",
        source_event={"event_id": "event-1"},
    )
    assert replay["status"] == "replayed"
    assert add_count == 1


def test_user_event_operation_is_durable_and_root_conflicts_are_rejected() -> None:
    client = QdrantClient(location=":memory:")
    store = RevisionOperationStore(client, is_remote=False)
    first = store.prepare_user_event_ingestion(
        protocol="mnemory.trusted-evidence.v1",
        user_id="user-1",
        owner_id="owner-1",
        evidence_root_id="root-1",
        request_fingerprint="request-1",
        memory_id="memory-1",
        source_event={"event_id": "event-1"},
    )
    assert first["status"] == "prepared"
    replay = store.prepare_user_event_ingestion(
        protocol="mnemory.trusted-evidence.v1",
        user_id="user-1",
        owner_id="owner-1",
        evidence_root_id="root-1",
        request_fingerprint="request-1",
        memory_id="memory-1",
        source_event={"event_id": "event-1"},
    )
    assert replay["operation_id"] == first["operation_id"]
    with pytest.raises(EvidenceConflictError):
        store.prepare_user_event_ingestion(
            protocol="mnemory.trusted-evidence.v1",
            user_id="user-1",
            owner_id="owner-1",
            evidence_root_id="root-1",
            request_fingerprint="request-2",
            memory_id="memory-1",
            source_event={"event_id": "event-1"},
        )


def test_two_events_skip_add_then_confirm_once_without_cross_scope() -> None:
    client = QdrantClient(location=":memory:")
    client.create_collection(
        collection_name="memories",
        vectors_config=VectorParams(size=2, distance=Distance.COSINE),
    )
    vector = VectorStore.__new__(VectorStore)
    vector._client = client
    vector._config = SimpleNamespace(
        vector=SimpleNamespace(is_remote=False, collection_name="memories")
    )
    vector._write_lock = None
    vector._embedding = SimpleNamespace(
        embed=lambda _text: [1.0, 0.0],
        embed_batch=lambda texts: [[1.0, 0.0] for _ in texts],
    )
    service = MemoryService.__new__(MemoryService)
    service._config = SimpleNamespace(
        memory=MemoryConfig(
            auto_classify=False,
            validation_enabled=True,
            validation_ttl_multiplier=1.0,
            validation_max_score_roots=3,
        ),
    )
    service.vector = vector
    service._sparse = None
    service.revisions = RevisionService(vector)
    service._remember_extract = lambda content, **_kwargs: (
        [{"text": content}],
        None,
        None,
    )
    service._get_available_categories = lambda _user_id: []
    service._evidence_semantic_equivalence = lambda source, target, _hint: (
        source == target
    )
    service._user_locks = {}
    service._locks_lock = threading.Lock()
    service._max_user_locks = 10
    service._core_cache = SimpleNamespace(invalidate_prefix=lambda _prefix: None)
    service._category_cache = SimpleNamespace(invalidate=lambda _user_id: None)

    user_id = "user-1"
    owner_id = "owner-1"
    content = "Qdrant is Mnemory's only database"

    def apply_evidence(root: str, request_hash: str) -> str:
        plan = service.plan_evidence(
            [],
            user_id=user_id,
            owner_id=owner_id,
            evidence_root_id=root,
            content=content,
        )
        sealed = service.seal_evidence_plan(plan, request_fingerprint=request_hash)
        if sealed.get("status") == "committed":
            return "replayed"
        epoch = int(sealed.get("claim_epoch", 0)) + 1
        nonce = uuid.uuid4().hex
        claimed = service.revisions.operations.claim_evidence_plan(
            sealed["operation_id"],
            request_fingerprint=request_hash,
            epoch=epoch,
            nonce=nonce,
        )
        result = service.apply_evidence_plan(
            claimed["operation_id"],
            request_fingerprint=request_hash,
            epoch=epoch,
            nonce=nonce,
            user_id=user_id,
            owner_id=owner_id,
        )
        return (
            "skipped"
            if result.get("result", {}).get("status") == "skipped"
            else "accepted"
        )

    root_a = "a" * 64
    root_b = "b" * 64
    assert apply_evidence(root_a, "evidence-a") == "skipped"
    ingested_a = service.ingest_trusted_user_event(
        content=content,
        user_id=user_id,
        owner_id=owner_id,
        evidence_root_id=root_a,
        request_hash="ingest-a",
        source_event={"event_id": "event-a"},
    )
    assert ingested_a["status"] == "accepted"
    assert ingested_a["result"]["results"][0]["event"] == "ADD"

    memories = client.scroll(
        collection_name="memories",
        limit=10,
        with_payload=True,
        with_vectors=False,
    )[0]
    assert len(memories) == 1
    target_id = memories[0].id
    target = memories[0].payload or {}
    assert "agent_id" not in target
    assert target["validation_eligible"] is True
    assert target["validation_count"] == 0
    assert target["validation_state"] == "unverified"

    assert apply_evidence(root_b, "evidence-b") == "accepted"
    confirmed = (
        client.retrieve(
            collection_name="memories",
            ids=[target_id],
            with_payload=True,
            with_vectors=False,
        )[0].payload
        or {}
    )
    assert confirmed["validation_count"] == 1
    assert confirmed["validation_state"] == "confirmed"
    assert root_b in confirmed["evidence_root_ids"]
    assert (
        len(
            client.scroll(
                collection_name="memories",
                limit=10,
                with_payload=True,
                with_vectors=False,
            )[0]
        )
        == 1
    )

    ingested_b = service.ingest_trusted_user_event(
        content=content,
        user_id=user_id,
        owner_id=owner_id,
        evidence_root_id=root_b,
        request_hash="ingest-b",
        source_event={"event_id": "event-b"},
    )
    assert ingested_b["status"] == "accepted"
    assert ingested_b["result"]["results"][0]["event"] == "SKIP"
    assert apply_evidence(root_b, "evidence-b") == "replayed"
    replayed = (
        client.retrieve(
            collection_name="memories",
            ids=[target_id],
            with_payload=True,
            with_vectors=False,
        )[0].payload
        or {}
    )
    assert replayed["validation_count"] == 1
    assert replayed["evidence_root_ids"].count(root_b) == 1

    cross_owner = service.plan_evidence(
        [],
        user_id=user_id,
        owner_id="owner-2",
        evidence_root_id="c" * 64,
        content=content,
    )
    assert all(item["action"] == "SKIP" for item in cross_owner["targets"])

    agent_text = "Agent-only assertion"
    agent_memory_id = "22222222-2222-4222-8222-222222222222"
    client.upsert(
        collection_name="memories",
        points=[
            PointStruct(
                id=agent_memory_id,
                vector=[1.0, 0.0],
                payload={
                    "data": agent_text,
                    "hash": hashlib.sha256(agent_text.encode()).hexdigest(),
                    "fact_hash": hashlib.sha256(
                        agent_text.casefold().encode()
                    ).hexdigest(),
                    "user_id": user_id,
                    "owner_id": owner_id,
                    "agent_id": "agent-1",
                    "role": "user",
                    "memory_layer": "raw",
                    "validation_eligible": True,
                    "lineage_id": agent_memory_id,
                    "revision": 1,
                    "revision_state": "active",
                },
            )
        ],
        wait=True,
    )
    agent_scoped = service.plan_evidence(
        [],
        user_id=user_id,
        owner_id=owner_id,
        evidence_root_id="d" * 64,
        content=agent_text,
    )
    assert all(item["action"] == "SKIP" for item in agent_scoped["targets"])


def _make_shared_ingest_service(client: QdrantClient) -> MemoryService:
    vector = VectorStore.__new__(VectorStore)
    vector._client = client
    vector._config = SimpleNamespace(
        vector=SimpleNamespace(is_remote=False, collection_name="memories")
    )
    vector._write_lock = None
    vector._embedding = SimpleNamespace(
        embed=lambda _text: [1.0, 0.0],
        embed_batch=lambda texts: [[1.0, 0.0] for _ in texts],
    )
    service = MemoryService.__new__(MemoryService)
    service._config = SimpleNamespace(
        memory=MemoryConfig(
            auto_classify=False,
            validation_enabled=True,
            validation_ttl_multiplier=1.0,
            validation_max_score_roots=3,
        ),
    )
    service.vector = vector
    service._sparse = None
    service.revisions = RevisionService(vector)
    service._user_locks = {}
    service._locks_lock = threading.Lock()
    service._max_user_locks = 10
    service._core_cache = SimpleNamespace(invalidate_prefix=lambda _prefix: None)
    service._category_cache = SimpleNamespace(invalidate=lambda _user_id: None)
    return service


def test_two_service_instances_claim_one_shared_content() -> None:
    client = QdrantClient(location=":memory:")
    client.create_collection(
        collection_name="memories",
        vectors_config=VectorParams(size=2, distance=Distance.COSINE),
    )
    services = [_make_shared_ingest_service(client) for _ in range(2)]
    barrier = threading.Barrier(2)

    def ingest(service: MemoryService, suffix: str) -> dict:
        barrier.wait()
        return service.ingest_trusted_user_event(
            content="One shared fact",
            user_id="user-1",
            owner_id="owner-1",
            evidence_root_id=suffix * 64,
            request_hash=f"request-{suffix}",
            source_event={"event_id": f"event-{suffix}"},
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda item: ingest(*item),
                [(services[0], "a"), (services[1], "b")],
            )
        )

    assert sorted(item["result"]["results"][0]["event"] for item in results) == [
        "ADD",
        "SKIP",
    ]
    points = client.scroll(
        collection_name="memories",
        limit=10,
        with_payload=True,
        with_vectors=False,
    )[0]
    assert len(points) == 1
    assert (points[0].payload or {})["validation_count"] == 0


def test_stale_shared_content_claim_has_bounded_takeover() -> None:
    client = QdrantClient(location=":memory:")
    store = RevisionOperationStore(client, is_remote=False)
    first = store.claim_user_event_content(
        content_fingerprint="content-fingerprint",
        operation_id="operation-a",
        user_id="user-1",
        owner_id="owner-1",
        memory_id="memory-1",
        lease_seconds=1,
    )
    assert first["claim_owner"] == "operation-a"
    time.sleep(1.05)
    second = store.claim_user_event_content(
        content_fingerprint="content-fingerprint",
        operation_id="operation-b",
        user_id="user-1",
        owner_id="owner-1",
        memory_id="memory-1",
        lease_seconds=1,
    )
    assert second["claim_owner"] == "operation-b"
    assert second["claim_epoch"] == 2


def test_active_old_worker_is_fenced_after_takeover() -> None:
    client = QdrantClient(location=":memory:")
    store = RevisionOperationStore(client, is_remote=False)
    old_claim = store.claim_user_event_content(
        content_fingerprint="content-fingerprint",
        operation_id="operation-a",
        user_id="user-1",
        owner_id="owner-1",
        memory_id="memory-1",
        lease_seconds=1,
    )
    time.sleep(1.05)
    new_claim = store.claim_user_event_content(
        content_fingerprint="content-fingerprint",
        operation_id="operation-b",
        user_id="user-1",
        owner_id="owner-1",
        memory_id="memory-1",
        lease_seconds=30,
    )
    assert new_claim["claim_owner"] == "operation-b"

    service = MemoryService.__new__(MemoryService)
    service.vector = SimpleNamespace(get_by_id_strict=lambda _: None)
    writes: list[str] = []
    service._add_direct = lambda *args, **kwargs: writes.append("write")
    with pytest.raises(EvidenceLeaseLostError):
        service._ingest_owned_user_event(
            claim=old_claim,
            content="One shared fact",
            user_id="user-1",
            owner_id="owner-1",
            evidence_root_id="root",
            source_event={"event_id": "event"},
            operations=store,
            content_fingerprint="content-fingerprint",
            operation_id="operation-a",
            memory_id="memory-1",
        )
    assert writes == []


def test_user_event_strict_read_failure_does_not_overwrite_memory() -> None:
    class Operations:
        @staticmethod
        def user_event_content_fingerprint(**kwargs):
            return "content-fingerprint"

        @staticmethod
        def user_event_content_claim_id(_fingerprint):
            return "claim-id"

        @staticmethod
        def user_event_operation_id(**kwargs):
            return "operation-id"

        @staticmethod
        def prepare_user_event_ingestion(**kwargs):
            return {"status": "prepared"}

        @staticmethod
        def claim_user_event_content(**kwargs):
            return {
                "claim_owner": "operation-id",
                "claim_epoch": 1,
                "claim_nonce": "nonce-1",
                "status": "claimed",
            }

        @staticmethod
        def renew_user_event_content_claim(*args, **kwargs):
            return {
                "claim_owner": "operation-id",
                "claim_epoch": 1,
                "claim_nonce": "nonce-1",
                "status": "claimed",
            }

    service = MemoryService.__new__(MemoryService)
    service.revisions = SimpleNamespace(operations=Operations())
    service._user_locks = {}
    service._locks_lock = threading.Lock()
    service._max_user_locks = 10
    service.vector = SimpleNamespace(
        get_by_id_strict=lambda _memory_id: (_ for _ in ()).throw(
            ConnectionError("qdrant unavailable")
        )
    )

    with pytest.raises(ConnectionError, match="qdrant unavailable"):
        service.ingest_trusted_user_event(
            content="Already validated",
            user_id="user-1",
            owner_id="owner-1",
            evidence_root_id="a" * 64,
            request_hash="request",
            source_event={"event_id": "event"},
        )


def test_user_event_maximum_is_rejected_before_durable_work() -> None:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    body = fixture["body"]
    body["messages"][0]["content"] = "x" * 1_001
    UserEventRememberRequest.model_validate(body)
    body["messages"][0]["content"] = "x" * 400_001
    with pytest.raises(ValueError):
        UserEventRememberRequest.model_validate(body)

    writes: list[str] = []
    service = MemoryService.__new__(MemoryService)
    service._user_locks = {}
    service._locks_lock = threading.Lock()
    service._max_user_locks = 10
    service.revisions = SimpleNamespace(
        operations=SimpleNamespace(
            user_event_content_fingerprint=lambda **_: writes.append("fingerprint"),
            user_event_operation_id=lambda **_: writes.append("operation"),
            prepare_user_event_ingestion=lambda **_: writes.append("prepare"),
        )
    )
    with pytest.raises(ValueError):
        service.ingest_trusted_user_event(
            content="x" * 1_001,
            user_id="user-1",
            owner_id="owner-1",
            evidence_root_id="root",
            request_hash="request",
            source_event={"event_id": "event"},
        )
    assert writes == []


def test_committed_replay_requires_existing_exact_memory() -> None:
    service = MemoryService.__new__(MemoryService)
    service._user_locks = {}
    service._locks_lock = threading.Lock()
    service._max_user_locks = 10
    operations = SimpleNamespace(
        user_event_content_fingerprint=lambda **_: "content-fingerprint",
        user_event_content_claim_id=lambda _: "claim-id",
        user_event_operation_id=lambda **_: "operation-id",
        prepare_user_event_ingestion=lambda **_: {
            "status": "committed",
            "memory_id": "memory-id",
            "result": {"results": []},
        },
    )
    service.revisions = SimpleNamespace(operations=operations)
    service.vector = SimpleNamespace(get_by_id_strict=lambda _: None)
    with pytest.raises(RuntimeError, match="not visible"):
        service.ingest_trusted_user_event(
            content="A durable fact",
            user_id="user-1",
            owner_id="owner-1",
            evidence_root_id="root",
            request_hash="request",
            source_event={"event_id": "event"},
        )

    service.vector = SimpleNamespace(
        get_by_id_strict=lambda _: {
            "memory": "Different fact",
            "user_id": "user-1",
            "owner_id": "owner-1",
            "metadata": {
                "source_kind": "raw_user_message",
                "memory_layer": "raw",
            },
        }
    )
    with pytest.raises(EvidenceConflictError):
        service.ingest_trusted_user_event(
            content="A durable fact",
            user_id="user-1",
            owner_id="owner-1",
            evidence_root_id="root",
            request_hash="request",
            source_event={"event_id": "event"},
        )


def test_memory_insert_requires_strong_acknowledgement() -> None:
    class DelayedClient:
        def __init__(self, failure: Exception | None = None) -> None:
            self.failure = failure
            self.kwargs: dict | None = None

        def upsert(self, **kwargs):
            time.sleep(0.02)
            self.kwargs = kwargs
            if self.failure is not None:
                raise self.failure

    client = DelayedClient()
    vector = VectorStore.__new__(VectorStore)
    vector._client = client
    vector._config = SimpleNamespace(
        vector=SimpleNamespace(is_remote=True, collection_name="memories")
    )
    vector._write_lock = None
    started = time.monotonic()
    vector.insert(
        text="acknowledged",
        vector=[1.0, 0.0],
        user_id="user-1",
        owner_id="owner-1",
        metadata={},
    )
    assert time.monotonic() - started >= 0.02
    assert client.kwargs is not None
    assert client.kwargs["wait"] is True
    assert client.kwargs["ordering"].value == "strong"

    failed = VectorStore.__new__(VectorStore)
    failed._client = DelayedClient(ConnectionError("remote write failed"))
    failed._config = vector._config
    failed._write_lock = None
    with pytest.raises(ConnectionError, match="remote write failed"):
        failed.insert(
            text="not acknowledged",
            vector=[1.0, 0.0],
            user_id="user-1",
            owner_id="owner-1",
            metadata={},
        )


def test_mounted_fastapi_route_dispatches_jwt_and_persists_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    from mnemory import server

    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    body = fixture["body"]
    private, public_path = _keypair(tmp_path)
    config = SimpleNamespace(
        server=SimpleNamespace(
            jwt_public_key=public_path,
            jwks_url="",
            api_keys={},
            api_key="",
            enable_metrics=False,
            enable_delete_all=False,
            thread_pool_size=2,
            mgmt_port=None,
            has_mgmt_port=False,
            port=8080,
        ),
        memory=SimpleNamespace(
            session_backend="memory",
            session_path=str(tmp_path / "sessions.db"),
            redis_url="",
            memory_session_ttl=86400,
            memory_session_sweep_interval=300,
        ),
    )
    client_store = QdrantClient(location=":memory:")
    client_store.create_collection(
        collection_name="memories",
        vectors_config=VectorParams(size=2, distance=Distance.COSINE),
    )
    service = _make_shared_ingest_service(client_store)
    from tests.test_trusted_semantic import llm_responses

    llm_responses(service, "User lives in Prague.")
    monkeypatch.setattr(server, "_get_config", lambda: config)
    monkeypatch.setattr(server, "_get_service", lambda: service)

    app = server.create_app()
    token = _user_event_token(private, body)
    http = TestClient(app)
    response = http.post(
        "/api/user-events/remember/v1",
        json=body,
        headers={"Authorization": f"Bearer {token}"},
    )
    http.close()

    assert response.status_code == 200
    assert response.json()["status"] == "accepted"
    points = client_store.scroll(
        collection_name="memories",
        limit=10,
        with_payload=True,
        with_vectors=False,
    )[0]
    assert len(points) == 1
    payload = points[0].payload or {}
    assert payload["source_kind"] == "raw_user_message"
    assert payload["validation_eligible"] is True
    assert payload["validation_count"] == 0
    assert payload["source_event"]["event_id"] == body["event"]["id"]
