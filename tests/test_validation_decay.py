"""Focused tests for evidence confirmation and bounded ranking decay."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from mnemory.auth import CognisJWTValidator
from mnemory.revisions import RevisionConflictError, RevisionService
from mnemory.ttl import apply_validation_and_decay_score

MEMORY_ID = "22222222-2222-4222-8222-222222222222"


def _revision_service() -> RevisionService:
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
    )
    service = RevisionService(vector)
    payload = {
        "data": "User lives in Prague",
        "user_id": "user-1",
        "owner_id": "owner-1",
        "role": "user",
        "ttl_days": 7,
        "expires_at": "2020-01-01T00:00:00+00:00",
        "evidence_root_ids": ["origin"],
        "validation_count": 0,
        **RevisionService.initial_metadata(MEMORY_ID),
    }
    client.upsert(
        collection_name="memories",
        points=[PointStruct(id=MEMORY_ID, vector=[1.0, 0.0], payload=payload)],
        wait=True,
    )
    return service


def _score_config(**overrides):
    defaults = {
        "validation_enabled": True,
        "validation_max_score_roots": 3,
        "validation_max_score_boost": 0.10,
        "slow_decay_enabled": False,
        "slow_decay_half_life_days": 30.0,
        "slow_decay_score_floor": 0.25,
        "slow_decay_validation_half_life_multiplier": 2.0,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_confirmation_is_idempotent_and_extends_ttl() -> None:
    service = _revision_service()
    kwargs = {
        "user_id": "user-1",
        "owner_id": "owner-1",
        "session_agent_id": None,
        "evidence_root_id": "root-2",
        "source_kind": "raw_user_message",
        "source_fingerprint": "source-2",
        "ttl_multiplier": 2.0,
        "max_score_roots": 3,
    }

    first = service.confirm(MEMORY_ID, **kwargs)
    second = service.confirm(MEMORY_ID, **kwargs)
    payload = service._read_point(MEMORY_ID).payload

    assert first["status"] == "confirmed"
    assert second["replayed"] is True
    assert payload["evidence_root_ids"] == ["origin", "root-2"]
    assert payload["validation_count"] == 1
    assert payload["validation_strength"] == pytest.approx(1 / 3)
    assert payload["expires_at"] > datetime.now(timezone.utc).isoformat()


def test_confirmation_keeps_permanent_memory_permanent() -> None:
    service = _revision_service()
    service._client.delete_payload(
        collection_name="memories",
        keys=["ttl_days", "expires_at"],
        points=[MEMORY_ID],
        wait=True,
    )
    service.confirm(
        MEMORY_ID,
        user_id="user-1",
        owner_id="owner-1",
        session_agent_id=None,
        evidence_root_id="root-2",
        source_kind="raw_user_message",
        source_fingerprint="source-2",
        ttl_multiplier=2.0,
        max_score_roots=3,
    )
    assert service._read_point(MEMORY_ID).payload.get("expires_at") is None


def test_confirmation_does_not_follow_a_stale_candidate_to_successor() -> None:
    service = _revision_service()
    successor_id = "33333333-3333-4333-8333-333333333333"
    original = service._read_point(MEMORY_ID).payload
    service._client.set_payload(
        collection_name="memories",
        payload={"revision_state": "superseded"},
        points=[MEMORY_ID],
        wait=True,
    )
    service._client.upsert(
        collection_name="memories",
        points=[
            PointStruct(
                id=successor_id,
                vector=[1.0, 0.0],
                payload={
                    **original,
                    "data": "User lives in Berlin",
                    "revision": 2,
                    "revision_state": "active",
                    "supersedes": MEMORY_ID,
                },
            )
        ],
        wait=True,
    )

    with pytest.raises(RevisionConflictError):
        service.confirm(
            MEMORY_ID,
            user_id="user-1",
            owner_id="owner-1",
            session_agent_id=None,
            evidence_root_id="root-2",
            source_kind="raw_user_message",
            source_fingerprint="source-2",
            ttl_multiplier=2.0,
            max_score_roots=3,
        )


def test_validation_boost_is_bounded() -> None:
    memory = {"score": 0.5, "metadata": {"validation_count": 99}}
    assert apply_validation_and_decay_score(memory, _score_config()) == pytest.approx(
        0.55
    )


def test_disabled_slow_decay_preserves_score() -> None:
    memory = {
        "score": 0.5,
        "metadata": {"revision_created_at_utc": "2020-01-01T00:00:00+00:00"},
    }
    assert (
        apply_validation_and_decay_score(
            memory,
            _score_config(validation_enabled=False, slow_decay_enabled=False),
        )
        == 0.5
    )


def test_slow_decay_uses_half_life() -> None:
    memory = {
        "score": 1.0,
        "metadata": {
            "revision_created_at_utc": (
                datetime.now(timezone.utc) - timedelta(days=30)
            ).isoformat()
        },
    }
    score = apply_validation_and_decay_score(
        memory,
        _score_config(validation_enabled=False, slow_decay_enabled=True),
    )
    assert score == pytest.approx(0.5, rel=0.01)


def test_user_event_jwt_exposes_evidence_scope() -> None:
    validator = CognisJWTValidator()
    validator._decode = MagicMock(  # type: ignore[method-assign]
        return_value={
            "sub": "user-1",
            "typ": "user_event",
            "scope": "mnemory:evidence other",
        }
    )
    context = validator.validate("token")
    assert context.agent_id is None
    assert context.token_type == "user_event"
    assert "mnemory:evidence" in context.scopes


def test_agent_jwt_cannot_be_a_trusted_user_event() -> None:
    validator = CognisJWTValidator()
    validator._decode = MagicMock(  # type: ignore[method-assign]
        return_value={
            "sub": "user-1",
            "agent_id": "agent-1",
            "typ": "user_event",
            "scope": ["mnemory:evidence"],
        }
    )
    context = validator.validate("token")
    trusted = (
        context.agent_id is None
        and context.token_type == "user_event"
        and "mnemory:evidence" in context.scopes
    )
    assert trusted is False
