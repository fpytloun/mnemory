"""REST API endpoints for session summaries.

Provides endpoints to list and view persistent session summaries
stored in the _mnemory_sessions Qdrant collection.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query

from mnemory.api.deps import SessionContext, get_session_context
from mnemory.api.schemas import FailedSessionRetryRequest
from mnemory.consolidation import RetryInputAssessment, assess_retry_inputs

logger = logging.getLogger("mnemory")

router = APIRouter(tags=["sessions"])


def _get_service():
    from mnemory.server import _get_service

    return _get_service()


async def _run_with_lease(
    operation,
    *args,
    operations,
    operation_token: str,
    claimant: str,
    lease_seconds: int,
    deadline: float | None = None,
    session_store=None,
    lease_session_record: dict | None = None,
    mutation_guard_supported: bool = False,
    **kwargs,
):
    """Run one blocking operation while renewing its journal lease."""
    done = asyncio.Event()
    lease_lost = False
    interval = max(0.1, min(float(lease_seconds) / 3.0, 30.0))
    loop = asyncio.get_running_loop()
    deadline = deadline if deadline is not None else loop.time() + lease_seconds

    def renew_ownership() -> bool:
        if loop.time() >= deadline:
            return False
        if not operations.renew_claim(
            operation_token,
            claimant,
            lease_seconds=lease_seconds,
        ):
            return False
        if session_store is None or lease_session_record is None:
            return True
        if session_store.renew_retry_claim(
            lease_session_record["session_id"],
            point_id=lease_session_record.get("_point_id"),
            retry_token=operation_token,
            retry_claimant=claimant,
            retry_lease_seconds=lease_seconds,
        ):
            return True
        current = session_store.get(
            lease_session_record["session_id"],
            point_id=lease_session_record.get("_point_id"),
        )
        return bool(
            current
            and current.get("consolidation_state") == "consolidated"
            and current.get("retry_operation_token") is None
            and current.get("retry_claimant") is None
        )

    def require_ownership() -> None:
        if not renew_ownership():
            raise RuntimeError("Retry batch lease was lost before mutation")

    if not await asyncio.to_thread(renew_ownership):
        raise RuntimeError("Retry batch lease was lost before mutation")

    async def renew_lease() -> None:
        nonlocal lease_lost
        while not done.is_set():
            try:
                await asyncio.wait_for(done.wait(), timeout=interval)
                return
            except TimeoutError:
                renewed = await asyncio.to_thread(renew_ownership)
                if not renewed:
                    lease_lost = True
                    return

    operation_kwargs = dict(kwargs)
    if mutation_guard_supported:
        operation_kwargs["mutation_guard"] = require_ownership
    worker = asyncio.create_task(
        asyncio.to_thread(operation, *args, **operation_kwargs)
    )
    heartbeat = asyncio.create_task(renew_lease())

    def consume_worker_result(completed: asyncio.Task) -> None:
        try:
            completed.result()
        except BaseException:
            logger.debug("Timed-out retry worker stopped", exc_info=True)

    try:
        result = await asyncio.wait_for(
            asyncio.shield(worker),
            timeout=max(0.1, deadline - loop.time()),
        )
    except TimeoutError as exc:
        done.set()
        await heartbeat
        worker.add_done_callback(consume_worker_result)
        raise TimeoutError("Retry batch operation timed out") from exc
    except asyncio.CancelledError:
        done.set()
        await heartbeat
        worker.add_done_callback(consume_worker_result)
        raise
    else:
        done.set()
        await heartbeat
    if lease_lost:
        raise RuntimeError("Retry batch lease was lost during consolidation")
    return result


@router.get("/sessions")
async def list_sessions(
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    consolidation_state: str | None = Query(None),
    q: str | None = Query(None),
    sort_by: Literal["updated_at", "created_at"] = Query("updated_at"),
    sort_dir: Literal["asc", "desc"] = Query("desc"),
    ctx: SessionContext = Depends(get_session_context),
) -> dict:
    """List persistent session summaries for the current user.

    Returns session summaries ordered by most recently updated first.
    These are conversation summaries persisted by the remember endpoint,
    used by the consolidation service to synthesize durable knowledge.
    """
    service = _get_service()

    try:
        result = service._session_summary_store.list_for_user(
            ctx.user_id,
            owner_id=ctx.owner_id,
            session_agent_id=ctx.agent_id,
            offset=offset,
            limit=limit,
            consolidation_state=consolidation_state,
            q=q,
            sort_by=sort_by,
            sort_dir=sort_dir,
            include_metadata=True,
        )
    except RuntimeError as exc:
        logger.exception("Failed to list sessions for user %s", ctx.user_id)
        raise HTTPException(status_code=503, detail="Failed to list sessions") from exc

    return result


def _failure_class(session: dict) -> str:
    if session.get("failure_class"):
        return str(session["failure_class"])
    if session.get("last_error_code"):
        return str(session["last_error_code"])
    return "legacy_unknown"


def _failed_session_eligibility(
    session: dict,
    max_memories: int,
    input_assessment: RetryInputAssessment | None = None,
) -> list[str]:
    reasons = []
    if session.get("consolidation_state") != "failed":
        reasons.append("not_failed")
    if session.get("legacy_failure") is not True:
        reasons.append("not_legacy_failure")
    if session.get("consolidation_token"):
        reasons.append("active_token")
    if session.get("consolidation_plan"):
        reasons.append("recovery_plan")
    memory_count = len(session.get("memory_ids") or [])
    if memory_count == 0:
        reasons.append("no_raw_memories")
    if memory_count > max_memories:
        reasons.append("too_many_raw_memories")
    if session.get("session_revision") is None:
        reasons.append("missing_session_revision")
    if input_assessment is not None:
        reasons.extend(input_assessment.reasons)
    return reasons


def _assess_retry_inputs(
    service,
    session: dict,
    ctx: SessionContext,
) -> RetryInputAssessment:
    """Assess linked revisions under the exact session identity."""
    return assess_retry_inputs(
        service.vector,
        session,
        user_id=ctx.user_id,
        owner_id=ctx.owner_id,
        agent_id=ctx.agent_id,
    )


def _retry_report(
    session: dict,
    reasons: list[str],
    assessment: RetryInputAssessment,
) -> dict:
    """Return one transport-safe retry eligibility report."""
    return {
        "session_id": session.get("session_id"),
        "session_revision": session.get("session_revision"),
        "raw_memory_count": len(session.get("memory_ids") or []),
        "attempt_count": session.get("attempt_count"),
        "failure_class": _failure_class(session),
        "legacy_failure": bool(session.get("legacy_failure")),
        "eligible": not reasons,
        "ineligibility_reasons": reasons,
        "input_fingerprint": assessment.input_fingerprint,
    }


def _retry_lease_is_live(session: dict) -> bool:
    value = session.get("retry_lease_expires_at")
    try:
        expires_at = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return False
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    return expires_at > datetime.now(timezone.utc)


def _list_failed_sessions(service, ctx: SessionContext) -> list[dict]:
    sessions = []
    offset = 0
    while True:
        page = service._session_summary_store.list_for_user(
            ctx.user_id,
            owner_id=ctx.owner_id,
            session_agent_id=ctx.agent_id,
            offset=offset,
            limit=200,
            consolidation_state="failed",
            include_metadata=True,
        )
        current = page.get("sessions") or []
        sessions.extend(
            item
            for item in current
            if ctx.agent_id is None
            or item.get("agent_id") == ctx.agent_id
            or str(item.get("agent_id") or "").startswith(ctx.agent_id + ":")
        )
        if len(current) < 200:
            break
        offset += len(current)
    return sessions


@router.get("/sessions/failed/diagnostics")
async def failed_session_diagnostics(
    ctx: SessionContext = Depends(get_session_context),
) -> dict:
    """Return bounded metadata-only diagnostics for failed sessions."""
    service = _get_service()
    sessions = _list_failed_sessions(service, ctx)
    now = datetime.now(timezone.utc)
    maximum = service._config.memory.legacy_failed_retry_max_raw_memories
    reason_counts: Counter[str] = Counter()
    eligible = 0
    for session in sessions:
        assessment = _assess_retry_inputs(service, session, ctx)
        reasons = _failed_session_eligibility(session, maximum, assessment)
        reason_counts.update(reasons)
        eligible += not reasons

    def age_bucket(session: dict) -> str:
        value = session.get("last_error_at") or session.get("updated_at")
        try:
            age = (now - datetime.fromisoformat(str(value))).days
        except (TypeError, ValueError):
            return "unknown"
        if age < 7:
            return "lt_7d"
        if age < 30:
            return "7d_30d"
        if age < 90:
            return "30d_90d"
        return "gt_90d"

    def size_bucket(session: dict) -> str:
        count = len(session.get("memory_ids") or [])
        if count <= 5:
            return "1_5"
        if count <= 20:
            return "6_20"
        if count <= 100:
            return "21_100"
        return "gt_100"

    return {
        "total": len(sessions),
        "legacy": sum(bool(item.get("legacy_failure")) for item in sessions),
        "failure_classes": dict(Counter(_failure_class(item) for item in sessions)),
        "age_buckets": dict(Counter(age_bucket(item) for item in sessions)),
        "size_buckets": dict(Counter(size_bucket(item) for item in sessions)),
        "eligible": eligible,
        "ineligibility_reasons": dict(reason_counts),
    }


@router.get("/sessions/failed/retry-eligibility")
async def failed_session_retry_eligibility(
    limit: int = Query(50, ge=1, le=200),
    ctx: SessionContext = Depends(get_session_context),
) -> dict:
    """Return a mutation-free failed-session retry report."""
    service = _get_service()
    maximum = service._config.memory.legacy_failed_retry_max_raw_memories
    sessions = _list_failed_sessions(service, ctx)
    sessions.sort(
        key=lambda item: (
            len(item.get("memory_ids") or []),
            item.get("updated_at") or "",
            item.get("session_id") or "",
        )
    )
    candidates = []
    for session in sessions[:limit]:
        assessment = _assess_retry_inputs(service, session, ctx)
        reasons = _failed_session_eligibility(session, maximum, assessment)
        candidates.append(_retry_report(session, reasons, assessment))
    return {"total": len(sessions), "candidates": candidates}


@router.post("/sessions/failed/retry")
async def retry_failed_sessions(
    req: FailedSessionRetryRequest,
    ctx: SessionContext = Depends(get_session_context),
) -> dict:
    """Dry-run or execute a bounded sequential failed-session retry."""
    service = _get_service()
    config = service._config.memory
    if len(req.session_ids) > config.legacy_failed_retry_max_batch:
        raise HTTPException(
            status_code=400, detail="Retry batch exceeds configured limit"
        )

    sessions = []
    for session_id in req.session_ids:
        session = service._session_summary_store.get(
            session_id,
            user_id=ctx.user_id,
            owner_id=ctx.owner_id,
            agent_id=ctx.agent_id,
        )
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        assessment = _assess_retry_inputs(service, session, ctx)
        reasons = _failed_session_eligibility(
            session,
            config.legacy_failed_retry_max_raw_memories,
            assessment,
        )
        sessions.append((session, reasons, assessment))
    report = [
        _retry_report(session, reasons, assessment)
        for session, reasons, assessment in sessions
    ]
    if req.dry_run:
        return {"dry_run": True, "results": report}
    if not config.legacy_failed_retry_enabled:
        raise HTTPException(status_code=403, detail="Legacy failed retry is disabled")
    if not req.idempotency_key:
        raise HTTPException(status_code=400, detail="idempotency_key is required")
    from mnemory.revisions import canonical_fingerprint
    from mnemory.server import _maintenance_service

    if _maintenance_service is None or _maintenance_service._consolidation is None:
        raise HTTPException(status_code=503, detail="Consolidation service unavailable")
    operations = service.revisions.operations
    token = canonical_fingerprint(
        [
            ctx.user_id,
            ctx.owner_id,
            ctx.agent_id,
            "session_retry_batch",
            sorted(req.session_ids),
            req.idempotency_key,
        ]
    )
    request_fingerprint = canonical_fingerprint(
        {
            "session_ids": sorted(req.session_ids),
            "stop_on_failure": req.stop_on_failure,
        }
    )
    existing = operations.get(token)
    if existing and (
        existing.get("user_id") != ctx.user_id
        or existing.get("owner_id") != ctx.owner_id
        or existing.get("agent_id") != ctx.agent_id
    ):
        raise HTTPException(status_code=404, detail="Retry batch not found")
    if existing and existing.get("request_fingerprint") not in {
        None,
        request_fingerprint,
    }:
        raise HTTPException(status_code=409, detail="Retry batch request changed")
    if existing and existing.get("status") == "committed":
        return {**existing["result"], "replayed": True}
    if (
        existing
        and existing.get("status") == "applying"
        and any(
            session.get("retry_operation_token") == token
            and session.get("retry_claimant")
            and session.get("consolidation_state") in {"idle", "consolidating"}
            and _retry_lease_is_live(session)
            for session, _, _ in sessions
        )
    ):
        raise HTTPException(
            status_code=409,
            detail="Previous retry claimant still owns a session",
        )
    if existing is None and any(reasons for _, reasons, _ in sessions):
        raise HTTPException(
            status_code=409, detail="One or more sessions are ineligible"
        )
    if existing is None:
        refreshed = []
        for original, _, original_assessment in sessions:
            current = service._session_summary_store.get(
                original["session_id"],
                user_id=ctx.user_id,
                owner_id=ctx.owner_id,
                agent_id=ctx.agent_id,
            )
            if current is None:
                raise HTTPException(status_code=409, detail="Retry preflight changed")
            current_assessment = _assess_retry_inputs(service, current, ctx)
            current_reasons = _failed_session_eligibility(
                current,
                config.legacy_failed_retry_max_raw_memories,
                current_assessment,
            )
            if (
                current_reasons
                or current.get("session_revision") != original.get("session_revision")
                or current_assessment.input_fingerprint
                != original_assessment.input_fingerprint
            ):
                raise HTTPException(status_code=409, detail="Retry preflight changed")
            refreshed.append((current, current_reasons, current_assessment))
        sessions = refreshed
    checkpoints = list((existing or {}).get("session_checkpoints") or []) or [
        {
            "session_id": item.get("session_id"),
            "expected_revision": item.get("session_revision"),
            "input_fingerprint": assessment.input_fingerprint,
            "status": "pending",
        }
        for item, _, assessment in sessions
    ]
    if existing is not None:
        for checkpoint in checkpoints:
            if checkpoint.get("status") == "committed" or checkpoint.get(
                "input_fingerprint"
            ):
                continue
            current = service._session_summary_store.get(
                checkpoint["session_id"],
                user_id=ctx.user_id,
                owner_id=ctx.owner_id,
                agent_id=ctx.agent_id,
            )
            retry_owned = bool(
                current
                and current.get("retry_operation_token") == token
                and current.get("retry_claimant")
                and current.get("consolidation_state") in {"idle", "consolidating"}
            )
            valid_revision = bool(
                current
                and (
                    current.get("session_revision")
                    == checkpoint.get("expected_revision")
                    or (
                        retry_owned
                        and int(current.get("session_revision", 0))
                        >= int(checkpoint.get("expected_revision", 0))
                    )
                )
            )
            if current is None or not valid_revision:
                raise HTTPException(status_code=409, detail="Retry preflight changed")
            assessment = _assess_retry_inputs(service, current, ctx)
            reasons = list(assessment.reasons)
            if not retry_owned:
                reasons.extend(
                    _failed_session_eligibility(
                        current,
                        config.legacy_failed_retry_max_raw_memories,
                    )
                )
            if reasons:
                raise HTTPException(status_code=409, detail="Retry preflight changed")
            checkpoint["input_fingerprint"] = assessment.input_fingerprint
    if existing is None:
        operations.write(
            token,
            {
                "status": "planned",
                "operation_kind": "session_retry_batch",
                "actor_kind": "manual",
                "user_id": ctx.user_id,
                "owner_id": ctx.owner_id,
                "agent_id": ctx.agent_id,
                "session_checkpoints": checkpoints,
                "request_fingerprint": request_fingerprint,
            },
        )
    claimant = str(uuid.uuid4())
    if not operations.claim(
        token,
        claimant=claimant,
        lease_seconds=config.legacy_failed_retry_timeout_seconds,
        allowed_statuses=("planned", "failed", "applying"),
    ):
        raise HTTPException(status_code=409, detail="Retry batch is already claimed")
    if not operations.write_claimed(
        token,
        claimant,
        {
            "status": "applying",
            "operation_kind": "session_retry_batch",
            "session_checkpoints": checkpoints,
        },
    ):
        raise HTTPException(status_code=409, detail="Retry batch lease was lost")
    started = time.monotonic()
    batch_deadline = started + config.legacy_failed_retry_timeout_seconds

    def require_batch_deadline() -> None:
        if time.monotonic() >= batch_deadline:
            raise TimeoutError("Retry batch deadline expired before mutation")

    succeeded = failed = 0
    stop_reason = None
    try:
        for index, checkpoint in enumerate(checkpoints):
            if checkpoint.get("status") == "committed":
                succeeded += 1
                continue
            if time.monotonic() - started > config.legacy_failed_retry_timeout_seconds:
                stop_reason = "timeout"
                break
            current = service._session_summary_store.get(
                checkpoint["session_id"],
                user_id=ctx.user_id,
                owner_id=ctx.owner_id,
                agent_id=ctx.agent_id,
            )
            expected_revision = int(checkpoint["expected_revision"])
            if (
                current
                and current.get("consolidation_state") == "consolidated"
                and int(current.get("session_revision", 0)) > expected_revision
            ):
                checkpoint["status"] = "committed"
                succeeded += 1
                continue
            if (
                current
                and current.get("consolidation_state") == "failed"
                and int(current.get("session_revision", 0)) > expected_revision
            ):
                checkpoint["status"] = "failed"
                failed += 1
                stop_reason = "first_failure"
                break
            current_assessment = (
                _assess_retry_inputs(service, current, ctx) if current else None
            )
            if (
                current_assessment is None
                or current_assessment.reasons
                or current_assessment.input_fingerprint
                != checkpoint.get("input_fingerprint")
            ):
                checkpoint["status"] = "skipped"
                stop_reason = "input_changed"
                break
            owned_retry = bool(
                current
                and current.get("retry_operation_token") == token
                and current.get("retry_claimant") == claimant
                and current.get("consolidation_state") in {"idle", "consolidating"}
            )
            if (
                current
                and current.get("retry_operation_token") == token
                and current.get("retry_claimant") not in {None, claimant}
                and current.get("consolidation_state") in {"idle", "consolidating"}
            ):
                require_batch_deadline()
                current = service._session_summary_store.transfer_retry_claim(
                    current["session_id"],
                    point_id=current.get("_point_id"),
                    retry_token=token,
                    previous_claimant=current["retry_claimant"],
                    retry_claimant=claimant,
                    retry_lease_seconds=config.legacy_failed_retry_timeout_seconds,
                )
                if current is None:
                    raise RuntimeError(
                        "A previous retry claimant still owns the session"
                    )
                owned_retry = True
            if owned_retry and current.get("consolidation_state") == "consolidating":
                recovered = await _run_with_lease(
                    _maintenance_service._consolidation.recover_session,
                    current,
                    operations=operations,
                    operation_token=token,
                    claimant=claimant,
                    lease_seconds=config.legacy_failed_retry_timeout_seconds,
                    deadline=batch_deadline,
                    session_store=service._session_summary_store,
                    lease_session_record=current,
                    mutation_guard_supported=True,
                )
                current = service._session_summary_store.get(
                    checkpoint["session_id"],
                    user_id=ctx.user_id,
                    owner_id=ctx.owner_id,
                    agent_id=ctx.agent_id,
                )
                if (
                    recovered
                    and current
                    and current.get("consolidation_state") == "consolidated"
                ):
                    checkpoint["status"] = "committed"
                    succeeded += 1
                    continue
                owned_retry = bool(
                    current
                    and current.get("consolidation_state") == "idle"
                    and current.get("retry_operation_token") == token
                    and current.get("retry_claimant") == claimant
                )
            if current is None or (
                not owned_retry
                and (
                    current.get("session_revision") != expected_revision
                    or _failed_session_eligibility(
                        current,
                        config.legacy_failed_retry_max_raw_memories,
                        current_assessment,
                    )
                )
            ):
                checkpoint["status"] = "skipped"
                stop_reason = "revision_changed"
                break
            if owned_retry:
                reset = current
            else:
                require_batch_deadline()
                if not operations.renew_claim(
                    token,
                    claimant,
                    lease_seconds=config.legacy_failed_retry_timeout_seconds,
                ):
                    raise RuntimeError("Retry batch lease was lost")
                require_batch_deadline()
                reset = service._session_summary_store.reset_failed_for_retry(
                    current["session_id"],
                    point_id=current.get("_point_id"),
                    expected_revision=expected_revision,
                    retry_token=token,
                    retry_claimant=claimant,
                    retry_lease_seconds=config.legacy_failed_retry_timeout_seconds,
                )
            if reset is None:
                checkpoint["status"] = "skipped"
                stop_reason = "revision_changed"
                break
            result = await _run_with_lease(
                _maintenance_service._consolidation.consolidate_session,
                current["session_id"],
                session_record=reset,
                operations=operations,
                operation_token=token,
                claimant=claimant,
                lease_seconds=config.legacy_failed_retry_timeout_seconds,
                deadline=batch_deadline,
                session_store=service._session_summary_store,
                lease_session_record=reset,
                mutation_guard_supported=True,
            )
            if result.state == "consolidated":
                succeeded += 1
                checkpoints[index]["status"] = "committed"
            else:
                failed += 1
                checkpoints[index]["status"] = "failed"
                if req.stop_on_failure:
                    stop_reason = "first_failure"
                    break
            attempts = succeeded + failed
            if (
                attempts >= config.legacy_failed_retry_stop_min_attempts
                and failed / attempts > config.legacy_failed_retry_stop_failure_ratio
            ):
                stop_reason = "failure_ratio"
                break
        result_payload = {
            "dry_run": False,
            "succeeded": succeeded,
            "failed": failed,
            "stopped": stop_reason is not None,
            "stop_reason": stop_reason,
            "replayed": False,
        }
        pending = any(checkpoint["status"] == "pending" for checkpoint in checkpoints)
        terminal_status = (
            "failed" if failed or stop_reason is not None or pending else "committed"
        )
        if not operations.write_claimed(
            token,
            claimant,
            {
                "status": terminal_status,
                "session_checkpoints": checkpoints,
                "result": result_payload,
                "terminal_at": datetime.now(timezone.utc).isoformat(),
            },
        ):
            raise RuntimeError("Retry batch lease was lost")
        from mnemory.metrics import get_collector

        collector = get_collector()
        if collector:
            collector.record_session_retry(
                batch_outcome=terminal_status,
                succeeded=succeeded,
                failed=failed,
                stop_reason=stop_reason,
            )
        return result_payload
    except BaseException as exc:
        terminalized = operations.write_claimed(
            token,
            claimant,
            {
                "status": "failed",
                "error_class": type(exc).__name__,
                "error_at": datetime.now(timezone.utc).isoformat(),
                "session_checkpoints": checkpoints,
                "terminal_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        if not terminalized:
            operations.terminalize_unclaimed(
                token,
                status="failed",
                allowed_statuses=("planned", "failed", "applying"),
                payload={
                    "error_class": type(exc).__name__,
                    "error_at": datetime.now(timezone.utc).isoformat(),
                    "session_checkpoints": checkpoints,
                    "terminal_reason": "retry_exception_after_lease_expiry",
                },
            )
        raise


@router.get("/sessions/{session_id}")
async def get_session(
    session_id: str,
    ctx: SessionContext = Depends(get_session_context),
) -> dict:
    """Get a single session summary by ID.

    Returns the full session summary including linked memory IDs
    and consolidation state.
    """
    service = _get_service()

    session = service._session_summary_store.get(
        session_id,
        user_id=ctx.user_id,
        owner_id=ctx.owner_id,
        agent_id=ctx.agent_id,
    )
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Verify user owns this session
    if session.get("user_id") != ctx.user_id:
        raise HTTPException(status_code=404, detail="Session not found")

    return session


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    delete_memories: bool = Query(False),
    ctx: SessionContext = Depends(get_session_context),
) -> dict:
    """Delete a session summary and optionally its linked raw memories.

    Args:
        session_id: Session to delete.
        delete_memories: If true, retract linked raw memories
            (not consolidated). Retraction preserves artifacts and revision
            history. Use explicit privacy erasure to remove retained content
            and artifact blobs.
    """
    service = _get_service()

    # Verify session exists and user owns it
    session = service._session_summary_store.get(
        session_id,
        user_id=ctx.user_id,
        owner_id=ctx.owner_id,
        agent_id=ctx.agent_id,
    )
    if session is None or session.get("user_id") != ctx.user_id:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.get("retry_operation_token"):
        raise HTTPException(
            status_code=409,
            detail="Session is owned by a failed-session retry batch",
        )

    # Prevent deletion during active consolidation
    if session.get("consolidation_state") == "consolidating":
        raise HTTPException(
            status_code=409,
            detail="Cannot delete session while consolidation is in progress",
        )

    deleted_memories = 0

    # Optionally delete linked raw memories
    if delete_memories:
        memory_ids = session.get("memory_ids") or []
        failed_memory_ids: list[str] = []
        for mid in memory_ids:
            try:
                # Check if memory exists and is raw before deleting
                mem = service.vector.get_by_id_strict(mid)
                if mem is None:
                    continue
                meta = mem.get("metadata") or {}
                revision_state = meta.get("revision_state", "active")
                if revision_state in {
                    "retracted",
                    "source",
                    "superseded",
                    "aborted",
                }:
                    continue
                if revision_state != "active":
                    raise RuntimeError(
                        f"Linked memory {mid} is not in a deletable state"
                    )
                layer = meta.get("memory_layer", "raw")
                if layer != "raw":
                    continue
                # delete_memory handles artifact cleanup
                service.delete_memory(
                    mid,
                    user_id=ctx.user_id,
                    owner_id=ctx.owner_id,
                    session_agent_id=ctx.agent_id,
                )
                deleted_memories += 1
            except Exception:
                logger.warning(
                    "Failed to look up or delete linked memory %s", mid, exc_info=True
                )
                failed_memory_ids.append(mid)
        if failed_memory_ids:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "Linked memory cleanup was incomplete",
                    "deleted_memories": deleted_memories,
                    "failed_memory_ids": failed_memory_ids,
                    "session_retained": True,
                },
            )

    # Delete the session record
    service._session_summary_store.delete(
        session_id,
        point_id=session.get("_point_id"),
    )

    return {
        "deleted_session": True,
        "deleted_memories": deleted_memories,
    }


@router.post("/sessions/{session_id}/consolidate")
async def consolidate_session_endpoint(
    session_id: str,
    ctx: SessionContext = Depends(get_session_context),
) -> dict:
    """Trigger consolidation for a specific session.

    Ignores the idle threshold. Runs consolidation in a background task
    and returns immediately with status 202. The client should poll
    GET /api/sessions/{id} to check when consolidation completes.
    """
    service = _get_service()

    # Verify session exists and user owns it
    session = service._session_summary_store.get(
        session_id,
        user_id=ctx.user_id,
        owner_id=ctx.owner_id,
        agent_id=ctx.agent_id,
    )
    if session is None or session.get("user_id") != ctx.user_id:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.get("retry_operation_token"):
        raise HTTPException(
            status_code=409,
            detail="Session is owned by a failed-session retry batch",
        )

    # Check if already consolidating (race condition guard)
    if session.get("consolidation_state") == "consolidating":
        raise HTTPException(status_code=409, detail="Consolidation already in progress")

    # Reset failed sessions to idle so consolidate_session() can proceed
    if session.get("consolidation_state") == "failed":
        service._session_summary_store.update_consolidation_state(
            session_id,
            "idle",
            point_id=session.get("_point_id"),
        )

    # Reuse the ConsolidationService from MaintenanceService
    from mnemory.server import _maintenance_service

    if _maintenance_service is None or _maintenance_service._consolidation is None:
        raise HTTPException(
            status_code=503, detail="Consolidation service not available"
        )

    # Fire and forget — run in background to avoid HTTP timeout
    async def _run_consolidation() -> None:
        try:
            await asyncio.to_thread(
                _maintenance_service._consolidation.consolidate_session,
                session_id,
                session_record=session,
            )
        except Exception:
            logger.exception("Manual consolidation failed for session %s", session_id)

    asyncio.create_task(_run_consolidation())

    return {"status": "consolidating", "session_id": session_id}
