"""Synchronous Cognis trusted user-event ingestion transport."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from mnemory.api.evidence import (
    _canonical_request_hash_for_path,
    derive_evidence_root,
    dispatch_trusted_event,
)
from mnemory.api.schemas import UserEventRememberRequest
from mnemory.revisions import EvidenceConflictError

logger = logging.getLogger("mnemory")
router = APIRouter()

USER_EVENT_PATH = "/api/user-events/remember/v1"
USER_EVENT_PROTOCOL = "mnemory.trusted-evidence.v1"
USER_EVENT_WALL_BUDGET_SECONDS = 90


def canonical_request_hash(body: dict[str, Any]) -> str:
    """Return the hash of the strict request at the user-event route."""
    return _canonical_request_hash_for_path(body, USER_EVENT_PATH)


def _claims_match_body(
    claims: dict[str, Any], body: UserEventRememberRequest, request_hash: str
) -> bool:
    """Check every signed identity and request binding."""
    body_dict = body.model_dump(mode="json")
    event = body.event
    expected = {
        "sub": body.actor.user_id,
        "aow": body.actor.owner_id,
        "evt": event.id,
        "event_hash": event.event_hash,
        "request_hash": request_hash,
        "evidence_root": derive_evidence_root(body_dict),
        "cognis_session_id": event.cognis_session_id,
        "conversation_id": event.conversation_id,
        "turn_id": event.turn_id,
    }
    return all(claims.get(name) == value for name, value in expected.items())


@router.post("/user-events/remember/v1", response_model=dict)
async def remember_user_event(
    request: Request,
    body: UserEventRememberRequest,
) -> dict[str, Any]:
    """Persist one authenticated raw user message and commit its operation."""
    claims = getattr(request.state, "user_event_claims", None)
    if not isinstance(claims, dict):
        raise HTTPException(
            status_code=401, detail="User-event authentication required"
        )

    body_dict = body.model_dump(mode="json")
    request_hash = canonical_request_hash(body_dict)
    evidence_root = derive_evidence_root(body_dict)
    if not _claims_match_body(claims, body, request_hash):
        raise HTTPException(
            status_code=400, detail="User-event claims do not match body"
        )

    from mnemory.server import _get_service

    event = body.event
    try:
        async with asyncio.timeout(USER_EVENT_WALL_BUDGET_SECONDS):
            service = _get_service()
            semantic = await dispatch_trusted_event(service, body, route="ingest")
            if semantic is not None:
                return semantic
            return await asyncio.to_thread(
                service.ingest_trusted_user_event,
                content=body.messages[0].content,
                user_id=body.actor.user_id,
                owner_id=body.actor.owner_id,
                evidence_root_id=evidence_root,
                request_hash=request_hash,
                source_event={
                    "protocol": USER_EVENT_PROTOCOL,
                    "event_id": event.id,
                    "event_hash": event.event_hash,
                    "cognis_session_id": event.cognis_session_id,
                    "conversation_id": event.conversation_id,
                    "turn_id": event.turn_id,
                    "evidence_root": evidence_root,
                    "request_hash": request_hash,
                },
            )
    except EvidenceConflictError as exc:
        raise HTTPException(
            status_code=409, detail="User-event request conflict"
        ) from exc
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=503, detail="User-event operation timed out; retry"
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("Trusted user-event operation failed", exc_info=True)
        raise HTTPException(
            status_code=503, detail="User-event operation failed; retry"
        ) from exc
