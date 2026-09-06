"""Synchronous Cognis trusted-evidence transport."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import unicodedata
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from mnemory.api.schemas import EvidenceRememberRequest
from mnemory.revisions import (
    EvidenceClaimActiveError,
    EvidenceConflictError,
    EvidenceCorruptError,
    EvidenceLeaseLostError,
)

logger = logging.getLogger("mnemory")
router = APIRouter()

EVIDENCE_PATH = "/api/evidence/remember/v1"
EVIDENCE_WALL_BUDGET_SECONDS = 90
EVIDENCE_PROTOCOL = "mnemory.trusted-evidence.v1"


def _normalize(value: Any) -> Any:
    """Normalize strings to NFC recursively before hashing."""
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, dict):
        return {key: _normalize(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    return value


def canonical_request_bytes(body: dict[str, Any]) -> bytes:
    """Return canonical method, route, and body bytes for request hashing."""
    envelope = {
        "body": _normalize(body),
        "method": "POST",
        "route": EVIDENCE_PATH,
    }
    return json.dumps(
        envelope,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_request_hash(body: dict[str, Any]) -> str:
    """Return the SHA-256 hash of the canonical evidence request."""
    return hashlib.sha256(canonical_request_bytes(body)).hexdigest()


def derive_evidence_root(body: dict[str, Any]) -> str:
    """Derive the stable root from protocol, actor, and event identity."""
    actor = body["actor"]
    event = body["event"]
    identity = [
        EVIDENCE_PROTOCOL,
        actor["user_id"],
        actor["owner_id"],
        event["id"],
        event["event_hash"],
        event["cognis_session_id"],
        event["conversation_id"],
        event["turn_id"],
    ]
    return hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _collector() -> Any:
    from mnemory.metrics import get_collector

    return get_collector()


def _record(outcome: str, reason: str = "none") -> None:
    collector = _collector()
    if collector is not None:
        collector.record_evidence(outcome, reason)


def _claims_match_body(claims: dict[str, Any], body: EvidenceRememberRequest) -> bool:
    """Check every required Cognis claim against its body binding."""
    event = body.event
    body_dict = body.model_dump(mode="json")
    pairs = {
        "sub": body.actor.user_id,
        "aow": body.actor.owner_id,
        "evt": event.id,
        "event_hash": event.event_hash,
        "request_hash": canonical_request_hash(body_dict),
        "evidence_root": derive_evidence_root(body_dict),
        "cognis_session_id": event.cognis_session_id,
        "conversation_id": event.conversation_id,
        "turn_id": event.turn_id,
    }
    return all(claims.get(name) == value for name, value in pairs.items())


@router.post("/evidence/remember/v1", response_model=dict)
async def remember_evidence(
    request: Request,
    body: EvidenceRememberRequest,
) -> dict[str, Any]:
    """Plan, seal, claim, and apply one Cognis user-event synchronously."""
    claims = getattr(request.state, "evidence_claims", None)
    if not isinstance(claims, dict):
        _record("rejected", "auth")
        raise HTTPException(status_code=401, detail="Evidence authentication required")

    body_dict = body.model_dump(mode="json")
    request_hash = canonical_request_hash(body_dict)
    if not _claims_match_body(claims, body):
        _record("rejected", "schema")
        raise HTTPException(status_code=400, detail="Evidence claims do not match body")
    if claims.get("request_hash") != request_hash:
        _record("rejected", "schema")
        raise HTTPException(status_code=400, detail="Evidence request hash mismatch")
    evidence_root = claims.get("evidence_root")
    if evidence_root != derive_evidence_root(body_dict):
        _record("rejected", "schema")
        raise HTTPException(status_code=400, detail="Evidence root mismatch")

    from mnemory.server import _get_service

    try:
        async with asyncio.timeout(EVIDENCE_WALL_BUDGET_SECONDS):
            service = _get_service()
            plan = await asyncio.to_thread(
                service.plan_evidence,
                [],
                user_id=body.actor.user_id,
                owner_id=body.actor.owner_id,
                evidence_root_id=evidence_root,
                content=body.messages[0].content,
            )
            sealed = await asyncio.to_thread(
                service.seal_evidence_plan,
                plan,
                request_fingerprint=request_hash,
            )
            if sealed.get("status") == "committed":
                _record("replayed")
                return {
                    "status": "replayed",
                    "operation_id": sealed["operation_id"],
                    "result": sealed.get("result"),
                }

            prior_checkpoints = bool(sealed.get("checkpoints"))
            epoch = int(sealed.get("claim_epoch", 0)) + 1
            nonce = uuid.uuid4().hex
            claimed = await asyncio.to_thread(
                service.revisions.operations.claim_evidence_plan,
                sealed["operation_id"],
                request_fingerprint=request_hash,
                epoch=epoch,
                nonce=nonce,
            )
            result = await asyncio.to_thread(
                service.apply_evidence_plan,
                claimed["operation_id"],
                request_fingerprint=request_hash,
                epoch=epoch,
                nonce=nonce,
                user_id=body.actor.user_id,
                owner_id=body.actor.owner_id,
            )
            outcome = (
                "skipped"
                if result.get("result", {}).get("status") == "skipped"
                else ("recovered" if prior_checkpoints else "accepted")
            )
            checkpoints = result.get("result", {}).get("checkpoints", [])
            reason = (
                "stale_target"
                if any(item.get("reason") == "stale_target" for item in checkpoints)
                else ("recovery" if prior_checkpoints else "none")
            )
            _record(outcome, reason)
            return {
                "status": outcome,
                "operation_id": result["operation_id"],
                "result": result.get("result"),
            }
    except EvidenceClaimActiveError as exc:
        _record("rejected", "claim_active")
        raise HTTPException(
            status_code=425,
            detail="Evidence operation is actively claimed",
            headers={"Retry-After": "90"},
        ) from exc
    except EvidenceCorruptError as exc:
        _record("rejected", "server_error")
        raise HTTPException(
            status_code=503, detail="Evidence journal is corrupt; retry"
        ) from exc
    except EvidenceLeaseLostError as exc:
        _record("rejected", "lease_lost")
        raise HTTPException(
            status_code=503, detail="Evidence lease lost; retry"
        ) from exc
    except EvidenceConflictError as exc:
        _record("rejected", "conflict")
        raise HTTPException(
            status_code=409, detail="Evidence request conflict"
        ) from exc
    except asyncio.TimeoutError as exc:
        _record("rejected", "timeout")
        raise HTTPException(
            status_code=503, detail="Evidence operation timed out; retry"
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("Trusted evidence operation failed")
        _record("rejected", "server_error")
        raise HTTPException(
            status_code=503, detail="Evidence operation failed; retry"
        ) from exc
