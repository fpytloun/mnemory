"""Memory consistency check (fsck) REST API endpoints.

POST /api/fsck           — Start a memory check (background task)
GET  /api/fsck/{id}      — Poll check status and results
POST /api/fsck/{id}/apply — Apply selected fixes from a completed check
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from mnemory.api.deps import SessionContext, get_session_context
from mnemory.api.schemas import (
    FsckAction as FsckActionSchema,
)
from mnemory.api.schemas import (
    FsckAffectedMemory as FsckAffectedMemorySchema,
)
from mnemory.api.schemas import (
    FsckApplyDetail,
    FsckApplyRequest,
    FsckApplyResponse,
    FsckAuditRequest,
    FsckReEvaluateRequest,
    FsckReEvaluateResponse,
    FsckRequest,
    FsckStartResponse,
    FsckStatusResponse,
)
from mnemory.api.schemas import (
    FsckIssue as FsckIssueSchema,
)
from mnemory.api.schemas import (
    FsckProgress as FsckProgressSchema,
)
from mnemory.api.schemas import (
    FsckSummary as FsckSummarySchema,
)

logger = logging.getLogger("mnemory")

router = APIRouter()


def _get_fsck_service():
    """Lazy import to avoid circular dependencies."""
    from mnemory.server import _get_fsck_service

    return _get_fsck_service()


def _record(operation: str, ctx: SessionContext) -> None:
    """Record a metrics operation."""
    from mnemory.metrics import get_collector

    collector = get_collector()
    if collector:
        collector.record_operation(operation, ctx.user_id, ctx.agent_id)


def _check_to_response(check) -> FsckStatusResponse:
    """Convert an FsckCheck dataclass to the API response model."""

    issues = None
    if check.status == "completed" and check.issues is not None:
        issues = []
        for issue in check.issues:
            affected = [
                FsckAffectedMemorySchema(
                    id=am.id,
                    content=am.content,
                    metadata=am.metadata,
                    agent_id=am.agent_id,
                )
                for am in issue.affected_memories
            ]
            actions = [
                FsckActionSchema(
                    action=a.action,
                    memory_id=a.memory_id,
                    new_content=a.new_content,
                    new_metadata=a.new_metadata,
                )
                for a in issue.actions
            ]
            issues.append(
                FsckIssueSchema(
                    issue_id=issue.issue_id,
                    type=issue.type,
                    severity=issue.severity,
                    confidence=issue.confidence,
                    reasoning=issue.reasoning,
                    affected_memories=affected,
                    actions=actions,
                    applied=issue.issue_id in check.applied_issue_ids,
                )
            )

    summary = None
    if check.summary:
        summary = FsckSummarySchema(
            duplicate=check.summary.duplicate,
            quality=check.summary.quality,
            split=check.summary.split,
            contradiction=check.summary.contradiction,
            reclassify=check.summary.reclassify,
            security=check.summary.security,
            total=check.summary.total,
        )

    progress = FsckProgressSchema(
        phase=check.progress.phase,
        total_memories=check.progress.total_memories,
        processed=check.progress.processed,
        percent=check.progress.percent,
        issues_found=check.progress.issues_found,
    )

    return FsckStatusResponse(
        check_id=check.check_id,
        status=check.status,
        progress=progress,
        summary=summary,
        issues=issues,
        error=check.error,
        created_at=check.created_at_utc,
        expires_at=check.expires_at_utc,
        mode=check.mode,
        target_state=check.target_state,
    )


@router.post("", response_model=FsckStartResponse)
def start_fsck(
    req: FsckRequest,
    background_tasks: BackgroundTasks,
    ctx: SessionContext = Depends(get_session_context),
):
    """Start a memory consistency check.

    Runs in the background. Returns a check_id to poll for results.
    """
    _record("fsck_check", ctx)

    fsck = _get_fsck_service()

    # Validate and resolve agent_id — enforce session agent boundary.
    # If the request specifies an agent_id, it must be the session agent
    # itself or a valid sub-agent (colon-prefixed). This mirrors the
    # protection in server.py::_resolve_agent_id().
    if req.agent_id is not None:
        if ctx.agent_id is not None:
            is_same = req.agent_id == ctx.agent_id
            is_sub = req.agent_id.startswith(ctx.agent_id + ":")
            if not is_same and not is_sub:
                raise HTTPException(
                    status_code=403,
                    detail=(
                        f"Cannot run fsck for agent '{req.agent_id}' — "
                        f"session is bound to agent '{ctx.agent_id}'"
                    ),
                )
        agent_id = req.agent_id
    else:
        agent_id = ctx.agent_id

    check = fsck.start_check(
        user_id=ctx.user_id,
        owner_id=ctx.owner_id,
        agent_id=agent_id,
    )

    # Run the check in background.
    # Manual UI checks are always full scan (incremental=False).
    background_tasks.add_task(
        fsck.run_check,
        check.check_id,
        categories=req.categories,
        memory_type=req.memory_type,
        include_raw=req.include_raw,
        incremental=req.incremental,
    )

    return FsckStartResponse(check_id=check.check_id, status="running")


@router.post("/audit", response_model=FsckStartResponse)
def start_fsck_audit(
    req: FsckAuditRequest,
    background_tasks: BackgroundTasks,
    ctx: SessionContext = Depends(get_session_context),
):
    """Start a bounded exact-revision audit that cannot mutate memories."""
    _record("fsck_audit", ctx)
    fsck = _get_fsck_service()
    try:
        check = fsck.start_exact_audit(
            basis_operation_id=req.basis_operation_id,
            targets=[target.model_dump() for target in req.targets],
            user_id=ctx.user_id,
            owner_id=ctx.owner_id,
            session_agent_id=ctx.agent_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    background_tasks.add_task(fsck.run_exact_audit, check.check_id)
    return FsckStartResponse(check_id=check.check_id, status="running")


def _get_maintenance_service():
    """Lazy import to avoid circular dependencies."""
    from mnemory.server import _get_maintenance_service

    return _get_maintenance_service()


@router.post("/auto-run", response_model=FsckStartResponse)
def auto_run_fsck(
    background_tasks: BackgroundTasks,
    ctx: SessionContext = Depends(get_session_context),
):
    """Trigger an immediate auto-fsck run for the current user.

    Runs the same pipeline as the scheduled auto-fsck: check all memories,
    then auto-apply fixes that meet the configured confidence and severity
    thresholds. Returns the check_id to poll for progress.

    The check + auto-apply runs in the background. Poll
    ``GET /api/fsck/{check_id}`` for progress and results.
    """
    _record("fsck_auto_run", ctx)

    maintenance = _get_maintenance_service()
    if maintenance is None:
        raise HTTPException(
            status_code=400,
            detail="Auto-fsck is not available (maintenance service not initialized)",
        )

    try:
        check_id = maintenance.start_run_now(user_id=ctx.user_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Run check + auto-apply in background
    background_tasks.add_task(maintenance.finish_run_now, check_id, ctx.user_id)

    return FsckStartResponse(check_id=check_id, status="running")


@router.get("/{check_id}", response_model=FsckStatusResponse)
def get_fsck_status(
    check_id: str,
    ctx: SessionContext = Depends(get_session_context),
):
    """Get the status and results of a memory check."""
    fsck = _get_fsck_service()
    check = fsck.get_check(
        check_id,
        user_id=ctx.user_id,
        owner_id=ctx.owner_id,
        session_agent_id=ctx.agent_id,
    )

    if check is None:
        raise HTTPException(
            status_code=404,
            detail="Check not found or expired. Please start a new check.",
        )

    # Verify ownership
    if check.user_id != ctx.user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    return _check_to_response(check)


@router.post("/{check_id}/apply", response_model=FsckApplyResponse)
def apply_fsck(
    check_id: str,
    req: FsckApplyRequest,
    ctx: SessionContext = Depends(get_session_context),
):
    """Apply selected fixes from a completed memory check."""
    _record("fsck_apply", ctx)

    fsck = _get_fsck_service()
    check = fsck.get_check(
        check_id,
        user_id=ctx.user_id,
        owner_id=ctx.owner_id,
        session_agent_id=ctx.agent_id,
    )

    if check is None:
        raise HTTPException(
            status_code=410,
            detail="Check not found or expired. Please re-run the check.",
        )

    # Verify ownership
    if check.user_id != ctx.user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    if check.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Check is not completed (status: {check.status})",
        )

    result = fsck.apply_check(
        check_id,
        issue_ids=req.issue_ids,
        user_id=ctx.user_id,
        owner_id=ctx.owner_id,
        session_agent_id=ctx.agent_id,
    )

    if result.get("error"):
        raise HTTPException(
            status_code=400,
            detail=result.get("message", "Apply failed"),
        )

    details = [
        FsckApplyDetail(
            issue_id=d["issue_id"],
            status=d["status"],
            actions_executed=d.get("actions_executed", 0),
            actions_skipped=d.get("actions_skipped", 0),
            error=d.get("error"),
        )
        for d in result.get("details", [])
    ]

    return FsckApplyResponse(
        applied=result.get("applied", 0),
        skipped=result.get("skipped", 0),
        failed=result.get("failed", 0),
        details=details,
    )


@router.post(
    "/operations/{operation_id}/re-evaluate",
    response_model=FsckReEvaluateResponse,
)
def re_evaluate_fsck_operation(
    operation_id: str,
    req: FsckReEvaluateRequest,
    ctx: SessionContext = Depends(get_session_context),
):
    """Compare an old operation with a fresh check without applying actions."""
    fsck = _get_fsck_service()
    try:
        result = fsck.re_evaluate_operation(
            operation_id,
            fresh_check_id=req.fresh_check_id,
            user_id=ctx.user_id,
            owner_id=ctx.owner_id,
            session_agent_id=ctx.agent_id,
            terminalize=req.terminalize,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return FsckReEvaluateResponse(**result)
