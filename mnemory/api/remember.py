"""POST /api/remember — Store memories from conversation turns.

Fire-and-forget endpoint for plugins. Accepts conversation messages,
formats them as text, and runs through the extraction pipeline in
a background task. Returns immediately with {"accepted": true}.

Rate-limited per user to prevent abuse.
"""

from __future__ import annotations

import logging
import threading
import time

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from mnemory.api.deps import SessionContext, get_session_context
from mnemory.api.schemas import MessageParam, RememberRequest, RememberResponse

logger = logging.getLogger("mnemory")

router = APIRouter()

# Simple in-memory rate limiter: user_id -> list of timestamps
_rate_limits: dict[str, list[float]] = {}
_rate_lock = threading.Lock()


def _get_service():
    from mnemory.server import _get_service

    return _get_service()


def _get_session_store():
    from mnemory.api import _session_store

    return _session_store


def _check_rate_limit(user_id: str) -> bool:
    """Check if user is within rate limit. Returns True if allowed.

    Uses a sliding window of 60 seconds. Cleans up old entries
    and removes empty user keys to prevent unbounded growth.
    """
    from mnemory.server import _get_config

    limit = _get_config().memory.remember_rate_limit
    if limit <= 0:
        return True  # No limit

    now = time.monotonic()
    window = 60.0  # 1 minute

    with _rate_lock:
        timestamps = _rate_limits.get(user_id, [])
        # Remove entries outside the window
        timestamps = [t for t in timestamps if now - t < window]
        if len(timestamps) >= limit:
            _rate_limits[user_id] = timestamps
            return False
        timestamps.append(now)
        _rate_limits[user_id] = timestamps

        # Periodic cleanup: remove empty user entries to prevent memory leak.
        # Only run when the dict grows large enough to matter.
        if len(_rate_limits) > 100:
            empty_keys = [k for k, v in _rate_limits.items() if not v]
            for k in empty_keys:
                del _rate_limits[k]

        return True


# Valid message roles for the remember endpoint.
# Only user and assistant are accepted — system and tool messages contain
# raw injected content (recalled memories, tool outputs) that should never
# be re-stored as new memories.
_VALID_MESSAGE_ROLES = {"user", "assistant"}


def _format_messages(messages: list[MessageParam]) -> str:
    """Format OpenAI-style messages into text for extraction.

    Only user and assistant messages are included. System and tool messages
    are silently skipped — system messages may contain already-recalled
    memories (causing duplicates) and tool messages contain raw data rather
    than conversational content.

    Example output:
        User: I just moved to Berlin
        Assistant: That's exciting! How do you like it?
    """
    parts = []
    for msg in messages:
        content = msg.content
        if not content:
            continue
        role = msg.role.strip().lower()
        if role not in _VALID_MESSAGE_ROLES:
            logger.debug("Skipping message with unknown role: %r", msg.role)
            continue
        label = role.capitalize()
        parts.append(f"{label}: {content}")
    return "\n".join(parts)


def _process_remember(
    content: str,
    user_id: str,
    owner_id: str | None,
    agent_id: str | None,
    role: str,
    session_id: str | None,
    timezone: str | None,
    context: str | None = None,
    labels: dict | None = None,
    trusted_evidence: bool = False,
    evidence_content: str | None = None,
) -> None:
    """Background task: extract and store memories from conversation content.

    All exceptions are caught and logged — never propagated.
    """
    try:
        service = _get_service()
        result = service.remember(
            content=content,
            user_id=user_id,
            owner_id=owner_id,
            agent_id=agent_id,
            role=role,
            session_id=session_id,
            session_timezone=timezone,
            context=context,
            labels=labels,
            _trusted_evidence=trusted_evidence,
            _evidence_content=evidence_content,
        )

        # Update session: prolong TTL and track stored memory IDs
        if session_id:
            try:
                session_store = _get_session_store()
                # Touch session to reset idle timeout (auto-prolong)
                session_store.touch(session_id)
                # Track stored IDs to prevent echo on next recall
                if result.get("results"):
                    stored_ids = {r["id"] for r in result["results"] if r.get("id")}
                    if stored_ids:
                        session_store.add_known_ids(session_id, stored_ids)
            except Exception:
                logger.warning(
                    "Failed to update session %s after remember",
                    session_id,
                )

        stored_count = sum(
            item.get("event") in {"ADD", "UPDATE"} for item in result.get("results", [])
        )
        if stored_count > 0:
            logger.info(
                "Remember: stored %d memories for user=%s",
                stored_count,
                user_id,
            )

    except Exception:
        logger.warning(
            "Remember background task failed for user=%s",
            user_id,
            exc_info=True,
        )


@router.post("/remember", response_model=RememberResponse)
def remember(
    req: RememberRequest,
    background_tasks: BackgroundTasks,
    ctx: SessionContext = Depends(get_session_context),
):
    """Store memories from a conversation exchange (fire-and-forget).

    Accepts OpenAI-format messages (typically last 2: user + assistant),
    formats them as text, and processes asynchronously. Returns immediately.
    """
    from mnemory.metrics import get_collector

    collector = get_collector()
    if collector:
        collector.record_operation("remember", ctx.user_id, ctx.agent_id)

    # Rate limit check
    if not _check_rate_limit(ctx.user_id):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Try again later.",
        )

    # Validate role context — Pydantic enforces valid values (user/assistant/null),
    # but agent_id presence for role="assistant" requires runtime context.
    if req.role == "assistant" and not ctx.agent_id:
        raise HTTPException(
            status_code=422,
            detail="role='assistant' requires agent_id (set X-Agent-Id header)",
        )

    # Format messages to text
    content = _format_messages(req.messages)
    if not content.strip():
        return RememberResponse(accepted=True)  # Nothing to process
    user_evidence = "\n".join(
        msg.content.strip()
        for msg in req.messages
        if msg.role == "user" and msg.content and msg.content.strip()
    )

    # Fire-and-forget background processing
    background_tasks.add_task(
        _process_remember,
        content=content,
        user_id=ctx.user_id,
        owner_id=ctx.owner_id,
        agent_id=ctx.agent_id,
        role=req.role,
        session_id=req.session_id,
        timezone=ctx.timezone,
        context=req.context,
        labels=req.labels,
        trusted_evidence=ctx.trusted_evidence and bool(user_evidence),
        evidence_content=user_evidence or None,
    )

    return RememberResponse(accepted=True)
