"""Contract tests for the confined trusted-evidence transport."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
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

from mnemory.api.evidence import (
    _claims_match_body,
    canonical_request_hash,
    derive_evidence_root,
    remember_evidence,
)
from mnemory.api.schemas import EvidenceRememberRequest
from mnemory.auth import CognisJWTValidator
from mnemory.memory import MemoryService
from mnemory.revisions import RevisionOperationStore, RevisionService
from mnemory.storage.vector import VectorStore

os.environ.setdefault("LLM_API_KEY", "test-key")

FIXTURE = Path(__file__).parent / "contract/fixtures/evidence_remember_v1.json"


def _keypair(tmp_path: Path) -> tuple[object, str]:
    private = ec.generate_private_key(ec.SECP256R1())
    public = private.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    path = tmp_path / "cognis-public.pem"
    path.write_bytes(public)
    return private, str(path)


def _token(
    private: object, body: dict, *, remove: tuple[str, ...] = (), **changes: object
) -> str:
    now = int(time.time())
    claims = {
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
        "jti": "jti-1",
        "iat": now,
        "nbf": now,
        "exp": now + 60,
    }
    for name in remove:
        claims.pop(name, None)
    claims.update(changes)
    return jwt.encode(claims, private, algorithm="ES256")


def _middleware_client(
    tmp_path: Path, body: dict, monkeypatch
) -> tuple[TestClient, object]:
    private, public_path = _keypair(tmp_path)
    from mnemory import server

    async def handler(request: Request) -> JSONResponse:
        return JSONResponse({"ok": True})

    app = Starlette(
        routes=[
            Route("/api/evidence/remember/v1", handler, methods=["POST"]),
            Route("/other", handler, methods=["GET"]),
            Route("/mcp", handler, methods=["POST"]),
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
    return TestClient(app), private


def test_fixture_hash_is_consumable() -> None:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    body = EvidenceRememberRequest.model_validate(fixture["body"])
    assert (
        canonical_request_hash(body.model_dump(mode="json")) == fixture["request_hash"]
    )
    assert fixture["canonical_request_bytes_utf8"].startswith('{"body":')
    assert (
        derive_evidence_root(body.model_dump(mode="json")) == fixture["evidence_root"]
    )
    assert (
        RevisionOperationStore.evidence_operation_id(
            protocol=fixture["protocol"],
            user_id=body.actor.user_id,
            owner_id=body.actor.owner_id,
            evidence_root_id=fixture["evidence_root"],
        )
        == fixture["expected_operation_id"]
    )


def test_evidence_auth_is_route_and_method_confined(
    tmp_path: Path, monkeypatch
) -> None:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    client, private = _middleware_client(tmp_path, fixture["body"], monkeypatch)
    token = _token(private, fixture["body"])
    with client:
        assert (
            client.post(
                "/api/evidence/remember/v1",
                headers={"Authorization": f"Bearer {token}"},
            ).status_code
            == 200
        )
        assert (
            client.get(
                "/other", headers={"Authorization": f"Bearer {token}"}
            ).status_code
            == 403
        )
        assert (
            client.post(
                "/mcp", headers={"Authorization": f"Bearer {token}"}
            ).status_code
            == 403
        )
        assert (
            client.post(
                "/api/evidence/remember/v1",
                headers={
                    "Authorization": f"Bearer {token}",
                    "X-Agent-Id": "forbidden",
                },
            ).status_code
            == 401
        )


def test_invalid_evidence_token_does_not_fall_back(tmp_path: Path, monkeypatch) -> None:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    client, _ = _middleware_client(tmp_path, fixture["body"], monkeypatch)
    with client:
        response = client.post(
            "/api/evidence/remember/v1",
            headers={
                "Authorization": "Bearer not-a-valid-evidence-token",
                "X-API-Key": "configured-api-key",
            },
        )
    assert response.status_code == 401


def test_malformed_but_signed_evidence_intent_cannot_fall_back(
    tmp_path: Path, monkeypatch
) -> None:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    client, private = _middleware_client(tmp_path, fixture["body"], monkeypatch)
    token = _token(private, fixture["body"], remove=("scope",))
    with client:
        response = client.get(
            "/other",
            headers={
                "Authorization": f"Bearer {token}",
                "X-API-Key": "configured-api-key",
            },
        )
    assert response.status_code == 403


@pytest.mark.parametrize(
    "changes",
    [
        {"scope": "mnemory:evidence other"},
        {"typ": "service"},
        {"evop": "forget"},
        {"ver": 2},
        {"agent_id": "agent-1"},
        {"exp": 61},
    ],
)
def test_evidence_jwt_contract_rejects_substitutions(
    tmp_path: Path, changes: dict[str, object]
) -> None:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    private, public_path = _keypair(tmp_path)
    token = _token(private, fixture["body"], **changes)
    validator = CognisJWTValidator(public_key_path=public_path)
    with pytest.raises(InvalidTokenError):
        validator.validate_evidence(token)


@pytest.mark.parametrize(
    "claim",
    [
        "typ",
        "scope",
        "evop",
        "ver",
        "sub",
        "aow",
        "evt",
        "event_hash",
        "request_hash",
        "evidence_root",
        "cognis_session_id",
        "conversation_id",
        "turn_id",
        "jti",
        "iat",
        "nbf",
        "exp",
    ],
)
def test_evidence_jwt_contract_rejects_missing_stable_claim(
    tmp_path: Path, claim: str
) -> None:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    private, public_path = _keypair(tmp_path)
    token = _token(private, fixture["body"], remove=(claim,))
    validator = CognisJWTValidator(public_key_path=public_path)
    with pytest.raises(InvalidTokenError):
        validator.validate_evidence(token)


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
def test_claim_substitution_is_rejected(tmp_path: Path, claim: str) -> None:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    body = EvidenceRememberRequest.model_validate(fixture["body"])
    claims = dict(fixture["claims"])
    claims[claim] = "substituted"
    if claim == "request_hash":
        assert claims[claim] != canonical_request_hash(body.model_dump(mode="json"))
    else:
        assert not _claims_match_body(claims, body)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda body: {**body, "context": "not allowed"},
        lambda body: {**body, "messages": [{"role": "assistant", "content": "x"}]},
        lambda body: {**body, "messages": [{"role": "user", "content": " "}]},
        lambda body: {**body, "messages": [None]},
        lambda body: {**body, "actor": {**body["actor"], "owner_id": None}},
    ],
)
def test_strict_body_rejection(tmp_path: Path, mutator) -> None:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    body = mutator(fixture["body"])
    with pytest.raises(ValueError):
        EvidenceRememberRequest.model_validate(body)


def test_endpoint_runs_plan_seal_claim_apply_without_logging_content(
    monkeypatch,
) -> None:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    body = EvidenceRememberRequest.model_validate(fixture["body"])
    request = Request(
        {"type": "http", "method": "POST", "path": "/api/evidence/remember/v1"}
    )
    request.state.evidence_claims = fixture["claims"]

    operation_id = fixture["expected_operation_id"]
    operations = SimpleNamespace(
        claim_evidence_plan=lambda *args, **kwargs: {
            "operation_id": operation_id,
            "status": "claimed",
            "claim_epoch": 1,
            "claim_nonce": "jti-1",
        }
    )
    service = SimpleNamespace(
        vector=SimpleNamespace(get_all=lambda **kwargs: {"results": []}),
        revisions=SimpleNamespace(operations=operations),
        plan_evidence=lambda *args, **kwargs: {
            "user_id": body.actor.user_id,
            "owner_id": body.actor.owner_id,
            "evidence_root_id": body.event.event_hash,
            "targets": [],
        },
        seal_evidence_plan=lambda *args, **kwargs: {
            "operation_id": operation_id,
            "status": "planned",
            "claim_epoch": 0,
            "checkpoints": [],
        },
        apply_evidence_plan=lambda *args, **kwargs: {
            "operation_id": operation_id,
            "result": {"status": "skipped"},
        },
    )
    monkeypatch.setattr("mnemory.server._get_service", lambda: service)
    result = asyncio.run(remember_evidence(request, body))
    assert result == {
        "status": "skipped",
        "operation_id": operation_id,
        "result": {"status": "skipped"},
    }


def _real_evidence_service(
    *,
    target_id: str,
    target_text: str,
    layer: str,
    include_fact_hash: bool,
) -> tuple[MemoryService, QdrantClient]:
    client = QdrantClient(location=":memory:")
    client.create_collection(
        collection_name="memories",
        vectors_config=VectorParams(size=2, distance=Distance.COSINE),
    )
    payload = {
        "data": target_text,
        "hash": hashlib.sha256(target_text.encode()).hexdigest(),
        "user_id": "user-1",
        "owner_id": "owner-1",
        "role": "user",
        "memory_layer": layer,
        "lineage_id": target_id,
        "revision": 1,
        "revision_state": "active",
        "derived_from": ["raw-source"] if layer == "consolidated" else [],
    }
    if include_fact_hash:
        payload["fact_hash"] = _normalized_fact_hash_for_test(target_text)
    client.upsert(
        collection_name="memories",
        points=[PointStruct(id=target_id, vector=[1.0, 0.0], payload=payload)],
        wait=True,
    )
    vector = VectorStore.__new__(VectorStore)
    vector._client = client
    vector._config = SimpleNamespace(
        vector=SimpleNamespace(is_remote=False, collection_name="memories")
    )
    vector._embedding = SimpleNamespace(
        embed_batch=lambda texts: [[1.0, 0.0] for _ in texts]
    )
    service = MemoryService.__new__(MemoryService)
    service._config = SimpleNamespace(
        memory=SimpleNamespace(
            max_memory_length=1000,
            validation_enabled=True,
            validation_ttl_multiplier=1.0,
            validation_max_score_roots=3,
        )
    )
    service.vector = vector
    service.revisions = RevisionService(vector)
    service._remember_extract = lambda *args, **kwargs: (
        [{"text": "User lives in Prague"}],
        None,
        None,
    )
    service._get_available_categories = lambda _user_id: []
    service._evidence_semantic_equivalence = lambda *args: True
    return service, client


def _normalized_fact_hash_for_test(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).casefold().encode()).hexdigest()


@pytest.mark.parametrize("layer", ["raw", "consolidated"])
@pytest.mark.parametrize("include_fact_hash", [True, False])
def test_signed_endpoint_plans_real_search_targets_once(
    monkeypatch,
    tmp_path: Path,
    layer: str,
    include_fact_hash: bool,
) -> None:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    body = json.loads(json.dumps(fixture["body"]))
    body["event"]["id"] = (
        f"event-{layer}-{'current' if include_fact_hash else 'legacy'}"
    )
    target_id = f"{'1' if layer == 'raw' else '2'}" * 8 + "-1111-4111-8111-111111111111"
    service, client = _real_evidence_service(
        target_id=target_id,
        target_text="User lives in Prague",
        layer=layer,
        include_fact_hash=include_fact_hash,
    )
    private, public_path = _keypair(tmp_path)
    from mnemory import server

    monkeypatch.setattr(
        server,
        "_get_config",
        lambda: SimpleNamespace(
            server=SimpleNamespace(jwt_public_key=public_path, jwks_url="")
        ),
    )
    monkeypatch.setattr(server, "_get_service", lambda: service)

    async def endpoint(request: Request) -> JSONResponse:
        request_body = EvidenceRememberRequest.model_validate(await request.json())
        return JSONResponse(await remember_evidence(request, request_body))

    app = Starlette(
        routes=[
            Route("/api/evidence/remember/v1", endpoint, methods=["POST"]),
        ],
        middleware=[Middleware(server.EvidenceAuthMiddleware)],
    )
    token = _token(private, body)
    with TestClient(app) as client_app:
        response = client_app.post(
            "/api/evidence/remember/v1",
            json=body,
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "accepted"
        replay = client_app.post(
            "/api/evidence/remember/v1",
            json=body,
            headers={"Authorization": f"Bearer {token}"},
        )
        assert replay.status_code == 200
        assert replay.json()["status"] == "replayed"
    target = client.retrieve(
        collection_name="memories",
        ids=[target_id],
        with_payload=True,
        with_vectors=False,
    )[0]
    assert target.payload["validation_count"] == 1
    assert "transition_token" not in target.payload
    operation_id = response.json()["operation_id"]
    operation = service.revisions.operations.get_evidence_plan(operation_id)
    assert operation is not None
    assert (
        operation["targets"][0]["source_hash"]
        == hashlib.sha256(b"User lives in Prague").hexdigest()
    )


def test_signed_endpoint_skips_partial_consolidated_search_target(
    monkeypatch,
    tmp_path: Path,
) -> None:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    body = json.loads(json.dumps(fixture["body"]))
    body["event"]["id"] = "event-partial-consolidated"
    target_id = "33333333-3333-4333-8333-333333333333"
    service, client = _real_evidence_service(
        target_id=target_id,
        target_text="User lives in Prague and uses Rust",
        layer="consolidated",
        include_fact_hash=False,
    )
    private, public_path = _keypair(tmp_path)
    from mnemory import server

    monkeypatch.setattr(
        server,
        "_get_config",
        lambda: SimpleNamespace(
            server=SimpleNamespace(jwt_public_key=public_path, jwks_url="")
        ),
    )
    monkeypatch.setattr(server, "_get_service", lambda: service)

    async def endpoint(request: Request) -> JSONResponse:
        request_body = EvidenceRememberRequest.model_validate(await request.json())
        return JSONResponse(await remember_evidence(request, request_body))

    app = Starlette(
        routes=[
            Route("/api/evidence/remember/v1", endpoint, methods=["POST"]),
        ],
        middleware=[Middleware(server.EvidenceAuthMiddleware)],
    )
    token = _token(private, body)
    with TestClient(app) as client_app:
        response = client_app.post(
            "/api/evidence/remember/v1",
            json=body,
            headers={"Authorization": f"Bearer {token}"},
        )
    assert response.status_code == 200
    assert response.json()["status"] == "skipped"
    target = client.retrieve(
        collection_name="memories",
        ids=[target_id],
        with_payload=True,
        with_vectors=False,
    )[0]
    assert target.payload.get("validation_count", 0) == 0
    assert "transition_token" not in target.payload


def test_evidence_metrics_bound_outcome_and_reason_labels() -> None:
    from unittest.mock import MagicMock

    from mnemory.metrics import MetricsCollector

    collector = object.__new__(MetricsCollector)
    collector._evidence_requests = MagicMock()
    MetricsCollector.record_evidence(collector, "accepted", "private-event-id")
    collector._evidence_requests.labels.assert_called_once_with(
        outcome="accepted", reason="server_error"
    )


def test_trusted_extraction_does_not_log_model_response(caplog) -> None:
    from unittest.mock import MagicMock

    private = "PRIVATE_MODEL_RESPONSE_9f2d"
    service = MemoryService.__new__(MemoryService)
    service._llm = MagicMock(
        generate=MagicMock(
            return_value=json.dumps(
                {
                    "facts": [
                        {
                            "text": private,
                            "memory_type": "fact",
                            "categories": [],
                            "importance": "normal",
                            "pinned": False,
                        }
                    ],
                    "summary": "",
                    "store_artifact": False,
                }
            )
        )
    )
    service._config = SimpleNamespace(memory=SimpleNamespace(max_memory_length=1000))
    service._get_available_categories = lambda _user_id: []
    with caplog.at_level("DEBUG", logger="mnemory"):
        service._extract_evidence_claims(
            private,
            user_id="user-1",
            owner_id="owner-1",
        )
    assert private not in caplog.text
