"""Memory CRUD endpoints for the REST API.

Thin wrappers around MemoryService methods. FastAPI handles sync→async
automatically for sync route functions.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from fastapi.responses import Response

from mnemory.api.deps import SessionContext, get_session_context
from mnemory.api.schemas import (
    AddMemoriesBatchRequest,
    AddMemoryRequest,
    AddMemoryResponse,
    AskMemoriesRequest,
    AskMemoriesResponse,
    CoreMemoriesResponse,
    DeleteMemoriesBatchRequest,
    DownloadTokenRequest,
    DownloadTokenResponse,
    FindMemoriesRequest,
    GetMemoriesByIdsRequest,
    GetMemoriesByIdsResponse,
    ListMemoriesResponse,
    MemoryHistoryResponse,
    MemoryLinksResponse,
    RecentMemoriesResponse,
    SaveArtifactRequest,
    SearchMemoriesRequest,
    SearchMemoriesResponse,
    UpdateMemoryRequest,
    format_memory_item,
)
from mnemory.api.ui_projections import (
    decode_cursor,
    encode_cursor,
    project_history,
    project_links,
    project_memory_item,
)
from mnemory.revisions import RevisionConflictError

logger = logging.getLogger("mnemory")

router = APIRouter()


def _get_service():
    """Get the MemoryService singleton (lazy import to avoid circular deps)."""
    from mnemory.server import _get_service

    return _get_service()


def _record(operation: str, ctx: SessionContext) -> None:
    """Record a metrics operation if metrics are enabled."""
    from mnemory.metrics import get_collector

    collector = get_collector()
    if collector:
        collector.record_operation(operation, ctx.user_id, ctx.agent_id)


def _resolve_revision_precondition(
    *, if_match: str | None, legacy_revision: int | None
) -> int | None:
    """Resolve one native integer revision from REST precondition inputs."""
    header_revision: int | None = None
    if if_match is not None:
        token = if_match.strip()
        if token.startswith('"') and token.endswith('"') and len(token) >= 2:
            token = token[1:-1]
        if (
            not token
            or not token.isascii()
            or not token.isdigit()
            or token.startswith("0")
        ):
            raise HTTPException(
                status_code=400,
                detail='If-Match must be a positive integer revision, for example "3"',
            )
        header_revision = int(token)
    if (
        header_revision is not None
        and legacy_revision is not None
        and header_revision != legacy_revision
    ):
        raise HTTPException(
            status_code=409,
            detail="If-Match and expected_revision must identify the same revision",
        )
    return header_revision if header_revision is not None else legacy_revision


# ── Memory CRUD ───────────────────────────────────────────────────────


@router.post("", response_model=AddMemoryResponse)
def add_memory(
    req: AddMemoryRequest,
    ctx: SessionContext = Depends(get_session_context),
):
    """Add a single memory with optional LLM-based fact extraction."""
    _record("add_memory", ctx)
    service = _get_service()

    # Server-side infer control
    from mnemory.server import _get_config

    effective_infer = req.infer
    if not req.infer and not _get_config().memory.allow_client_infer:
        logger.debug("Overriding infer=False to True (ALLOW_CLIENT_INFER=False)")
        effective_infer = True

    try:
        result = service.add_memory(
            content=req.content,
            user_id=ctx.user_id,
            owner_id=ctx.owner_id,
            agent_id=ctx.agent_id,
            memory_type=req.memory_type,
            categories=req.categories,
            importance=req.importance,
            pinned=req.pinned,
            infer=effective_infer,
            role=req.role,
            ttl_days=req.ttl_days,
            event_date=req.event_date,
            session_timezone=ctx.timezone,
            labels=req.labels,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if result.get("error"):
        raise HTTPException(
            status_code=400, detail=result.get("message", "Unknown error")
        )

    return result


@router.post("/batch", response_model=list[AddMemoryResponse])
def add_memories_batch(
    req: AddMemoriesBatchRequest,
    ctx: SessionContext = Depends(get_session_context),
):
    """Batch-add multiple memories."""
    _record("add_memories", ctx)
    service = _get_service()

    # Server-side infer control
    from mnemory.server import _get_config

    allow_infer = _get_config().memory.allow_client_infer

    results = []
    for item in req.memories:
        try:
            effective_infer = item.infer
            if not item.infer and not allow_infer:
                effective_infer = True

            result = service.add_memory(
                content=item.content,
                user_id=ctx.user_id,
                owner_id=ctx.owner_id,
                agent_id=ctx.agent_id,
                memory_type=item.memory_type,
                categories=item.categories,
                importance=item.importance,
                pinned=item.pinned,
                infer=effective_infer,
                role=item.role,
                ttl_days=item.ttl_days,
                event_date=item.event_date,
                session_timezone=ctx.timezone,
                labels=item.labels,
            )
            results.append(result)
        except Exception:
            logger.exception("Failed to add memory in batch")
            results.append(
                {"results": [], "error": True, "message": "Failed to add memory"}
            )
    return results


@router.post("/batch/delete")
def delete_memories_batch(
    req: DeleteMemoriesBatchRequest,
    ctx: SessionContext = Depends(get_session_context),
):
    """Batch-delete multiple memories."""
    _record("delete_memories", ctx)
    service = _get_service()
    results = []
    errors = []

    for mid in req.memory_ids:
        try:
            result = service.delete_memory(
                memory_id=mid,
                user_id=ctx.user_id,
                owner_id=ctx.owner_id,
                session_agent_id=ctx.agent_id,
            )
            results.append(result)
        except ValueError as e:
            errors.append({"memory_id": mid, "error": True, "message": str(e)})
        except Exception:
            logger.exception("Failed to delete memory %s in batch", mid)
            errors.append(
                {"memory_id": mid, "error": True, "message": "Failed to delete memory"}
            )

    return {
        "results": results,
        "errors": errors,
        "total": len(req.memory_ids),
        "succeeded": len(results),
        "failed": len(errors),
    }


@router.post("/search", response_model=SearchMemoriesResponse)
def search_memories(
    req: SearchMemoriesRequest,
    ctx: SessionContext = Depends(get_session_context),
):
    """Semantic search across memories."""
    _record("search_memories", ctx)
    service = _get_service()
    try:
        # When session has agent_id, use dual-scope to return both
        # agent-specific and shared user memories (mirrors MCP behavior).
        if ctx.agent_id:
            results = service.search_memories_dual_scope(
                query=req.query,
                user_id=ctx.user_id,
                owner_id=ctx.owner_id,
                session_agent_id=ctx.agent_id,
                memory_type=req.memory_type,
                categories=req.categories,
                role=req.role,
                limit=req.limit,
                include_decayed=req.include_decayed,
                date_start=req.date_start,
                date_end=req.date_end,
                labels=req.labels,
            )
        else:
            results = service.search_memories(
                query=req.query,
                user_id=ctx.user_id,
                owner_id=ctx.owner_id,
                agent_id=None,
                memory_type=req.memory_type,
                categories=req.categories,
                role=req.role,
                limit=req.limit,
                include_decayed=req.include_decayed,
                date_start=req.date_start,
                date_end=req.date_end,
                labels=req.labels,
            )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    items = [format_memory_item(r) for r in results]
    return SearchMemoriesResponse(results=items)


@router.post("/find", response_model=SearchMemoriesResponse)
def find_memories(
    req: FindMemoriesRequest,
    ctx: SessionContext = Depends(get_session_context),
):
    """AI-powered multi-query search with LLM reranking."""
    _record("find_memories", ctx)
    service = _get_service()
    try:
        result = service.find_memories(
            question=req.question,
            user_id=ctx.user_id,
            owner_id=ctx.owner_id,
            session_agent_id=ctx.agent_id,
            agent_id=ctx.agent_id,
            memory_type=req.memory_type,
            categories=req.categories,
            role=req.role,
            limit=req.limit,
            include_decayed=req.include_decayed,
            session_timezone=ctx.timezone,
            context=req.context,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    items = [format_memory_item(r) for r in result.get("results", [])]
    return SearchMemoriesResponse(results=items)


@router.post("/ask", response_model=AskMemoriesResponse)
def ask_memories(
    req: AskMemoriesRequest,
    ctx: SessionContext = Depends(get_session_context),
):
    """Ask a question and get a human-readable answer based on memories."""
    _record("ask_memories", ctx)
    service = _get_service()
    try:
        result = service.answer_question(
            question=req.question,
            user_id=ctx.user_id,
            owner_id=ctx.owner_id,
            session_agent_id=ctx.agent_id,
            agent_id=ctx.agent_id,
            memory_type=req.memory_type,
            categories=req.categories,
            role=req.role,
            limit=req.limit,
            include_decayed=req.include_decayed,
            session_timezone=ctx.timezone,
            context=req.context,
            labels=req.labels,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    memories = result.get("results", [])
    if req.include_memories:
        items = [format_memory_item(r) for r in memories]
    else:
        items = []

    return AskMemoriesResponse(
        answer=result.get("answer", ""),
        results=items,
        count=len(memories),
        queries=result.get("queries", []),
        stats=result.get("stats", {}),
    )


@router.get(
    "/core", response_model=CoreMemoriesResponse, response_model_exclude_none=True
)
def get_core_memories(
    recent_days: int = Query(7, description="Days of recent context"),
    include_stats: bool = Query(
        False, description="Include structured core-memory stats"
    ),
    ctx: SessionContext = Depends(get_session_context),
):
    """Load pinned + recent context memories."""
    _record("get_core_memories", ctx)
    service = _get_service()
    core_result = service.get_core_memories(
        user_id=ctx.user_id,
        owner_id=ctx.owner_id,
        agent_id=ctx.agent_id,
        recent_days=recent_days,
    )
    return CoreMemoriesResponse(
        text=core_result.text,
        stats=asdict(core_result.stats)
        if include_stats and core_result.stats
        else None,
    )


@router.get("/recent", response_model=RecentMemoriesResponse)
def get_recent_memories(
    days: int = Query(7, description="Days back to look"),
    scope: str = Query("all", description="Scope: all, user, agent"),
    limit: int = Query(25, ge=1, le=100, description="Max results per scope"),
    include_decayed: bool = Query(False, description="Include expired memories"),
    ctx: SessionContext = Depends(get_session_context),
):
    """Get recent memories from the last N days."""
    _record("get_recent_memories", ctx)
    service = _get_service()
    try:
        result = service.get_recent_memories(
            user_id=ctx.user_id,
            owner_id=ctx.owner_id,
            agent_id=ctx.agent_id,
            days=days,
            scope=scope,
            limit=limit,
            include_decayed=include_decayed,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return RecentMemoriesResponse(text=result)


@router.get("", response_model=ListMemoriesResponse)
def list_memories(
    memory_type: str | None = Query(None, description="Filter by type"),
    categories: str | None = Query(None, description="Comma-separated categories"),
    role: str | None = Query(None, description="Filter by role"),
    limit: int = Query(50, ge=1, le=5000, description="Max results"),
    include_decayed: bool = Query(False, description="Include expired memories"),
    sort: str | None = Query(
        None, description="Sort order: newest (desc by created_at), oldest (asc)"
    ),
    labels: str | None = Query(
        None,
        description='Filter by labels as JSON object (AND logic), e.g. \'{"source":"web"}\'',
    ),
    memory_layer: str | None = Query(
        None,
        description='Filter by memory layer: "raw" or "consolidated"',
    ),
    ctx: SessionContext = Depends(get_session_context),
):
    """List all or filtered memories with optional sorting."""
    if memory_layer and memory_layer not in ("raw", "consolidated"):
        raise HTTPException(
            status_code=422,
            detail="memory_layer must be 'raw' or 'consolidated'",
        )

    _record("list_memories", ctx)
    service = _get_service()
    cat_list = [c.strip() for c in categories.split(",")] if categories else None
    labels_dict: dict | None = None
    if labels:
        try:
            labels_dict = json.loads(labels)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=422, detail="Invalid JSON in labels parameter"
            )
    try:
        # When session has agent_id, use dual-scope to return both
        # agent-specific and shared user memories (mirrors MCP behavior).
        if ctx.agent_id:
            results = service.list_memories_dual_scope(
                user_id=ctx.user_id,
                owner_id=ctx.owner_id,
                session_agent_id=ctx.agent_id,
                memory_type=memory_type,
                categories=cat_list,
                role=role,
                limit=limit,
                include_decayed=include_decayed,
                sort=sort,
                labels=labels_dict,
            )
        else:
            results = service.list_memories(
                user_id=ctx.user_id,
                owner_id=ctx.owner_id,
                agent_id=None,
                memory_type=memory_type,
                categories=cat_list,
                role=role,
                limit=limit,
                include_decayed=include_decayed,
                sort=sort,
                labels=labels_dict,
            )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Filter by memory_layer if specified (post-retrieval filtering)
    if memory_layer:
        results = [
            m
            for m in results
            if (m.get("metadata") or {}).get("memory_layer", "consolidated")
            == memory_layer
        ]

    items = [format_memory_item(r) for r in results]
    return ListMemoriesResponse(results=items)


@router.get("/browse", response_model=ListMemoriesResponse)
def browse_memories(
    cursor: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    memory_type: str | None = Query(None),
    categories: str | None = Query(None),
    role: str | None = Query(None),
    labels: str | None = Query(None),
    memory_layer: str | None = Query(None),
    agent_id: str | None = Query(None),
    has_artifacts: bool = Query(False),
    decayed_only: bool = Query(False),
    include_decayed: bool = Query(False),
    ctx: SessionContext = Depends(get_session_context),
):
    """Browse every authorized active memory with a stable opaque cursor."""
    if memory_layer and memory_layer not in ("raw", "consolidated"):
        raise HTTPException(status_code=422, detail="Invalid memory layer")
    cat_list = (
        [value.strip() for value in categories.split(",")] if categories else None
    )
    try:
        labels_dict = json.loads(labels) if labels else None
    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail="Invalid labels filter") from None
    if labels_dict is not None and not isinstance(labels_dict, dict):
        raise HTTPException(status_code=422, detail="Invalid labels filter")
    binding = {
        "user_id": ctx.user_id,
        "owner_id": ctx.owner_id,
        "session_agent_id": ctx.agent_id,
        "memory_type": memory_type,
        "categories": cat_list,
        "role": role,
        "labels": labels_dict,
        "memory_layer": memory_layer,
        "agent_id": agent_id,
        "has_artifacts": has_artifacts,
        "decayed_only": decayed_only,
        "include_decayed": include_decayed,
    }
    try:
        offset = decode_cursor(cursor, binding) if cursor else None
        if memory_type:
            from mnemory.memory import validate_memory_type

            memory_type = validate_memory_type(memory_type)
        if role and role not in ("user", "assistant"):
            raise ValueError("Invalid role")
        result = _get_service().vector.browse_active(
            user_id=ctx.user_id,
            owner_id=ctx.owner_id,
            session_agent_id=ctx.agent_id,
            cursor=offset,
            limit=limit,
            memory_type=memory_type,
            categories=cat_list,
            role=role,
            labels_filter=labels_dict,
            memory_layer=memory_layer,
            agent_id=agent_id,
            has_artifacts=has_artifacts,
            decayed_only=decayed_only,
            include_decayed=include_decayed,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    next_offset = result.get("next_offset")
    next_cursor = encode_cursor(next_offset, binding) if next_offset else None
    items = [project_memory_item(item) for item in result["results"]]
    return ListMemoriesResponse(
        results=items,
        next_cursor=next_cursor,
        has_more=next_cursor is not None,
    )


@router.post("/by-ids", response_model=GetMemoriesByIdsResponse)
def get_memories_by_ids(
    req: GetMemoriesByIdsRequest,
    ctx: SessionContext = Depends(get_session_context),
):
    """Fetch multiple memories by their IDs.

    Returns memories matching the provided IDs that belong to the
    current user.  Missing or unauthorized IDs are silently skipped.
    Used by the management UI Sessions tab to efficiently load linked
    memories without fetching the entire memory store.
    """
    _record("get_memories_by_ids", ctx)
    service = _get_service()
    results = service.get_memories_by_ids(
        req.ids,
        user_id=ctx.user_id,
        owner_id=ctx.owner_id,
        session_agent_id=ctx.agent_id,
    )
    items = [format_memory_item(r) for r in results]
    return GetMemoriesByIdsResponse(results=items)


@router.put("/{memory_id}")
def update_memory(
    memory_id: str,
    req: UpdateMemoryRequest,
    if_match: str | None = Header(None, alias="If-Match"),
    ctx: SessionContext = Depends(get_session_context),
):
    """Update an existing memory's content or metadata."""
    _record("update_memory", ctx)
    service = _get_service()
    expected_revision = _resolve_revision_precondition(
        if_match=if_match, legacy_revision=req.expected_revision
    )
    try:
        kwargs: dict = {
            "memory_id": memory_id,
            "user_id": ctx.user_id,
            "owner_id": ctx.owner_id,
            "session_agent_id": ctx.agent_id,
            "content": req.content,
            "memory_type": req.memory_type,
            "categories": req.categories,
            "importance": req.importance,
            "pinned": req.pinned,
            "expected_revision": expected_revision,
            "idempotency_key": req.idempotency_key,
        }
        if "ttl_days" in req.model_fields_set:
            kwargs["ttl_days"] = req.ttl_days
        if "event_date" in req.model_fields_set:
            kwargs["event_date"] = req.event_date
        if req.labels is not None:
            kwargs["labels"] = req.labels
        # agent_id: non-None means caller wants to change it
        if req.agent_id is not None:
            if req.agent_id == "":
                # Clear agent_id
                kwargs["agent_id"] = None
            else:
                # Security: session with agent_id can only set own agent or
                # a valid sub-agent (colon-prefixed). Mirrors the protection
                # in server.py::_resolve_agent_id().
                if ctx.agent_id:
                    is_same = req.agent_id == ctx.agent_id
                    is_sub = req.agent_id.startswith(ctx.agent_id + ":")
                    if not is_same and not is_sub:
                        raise HTTPException(
                            status_code=403,
                            detail="Cannot set agent_id to a different agent",
                        )
                kwargs["agent_id"] = req.agent_id
        result = service.update_memory(**kwargs)
    except RevisionConflictError as e:
        raise HTTPException(status_code=409, detail=e.to_dict())
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if result.get("error"):
        raise HTTPException(
            status_code=400, detail=result.get("message", "Unknown error")
        )
    return result


@router.delete("/{memory_id}")
def delete_memory(
    memory_id: str,
    expected_revision: int | None = Query(None, ge=1),
    idempotency_key: str | None = Query(None, min_length=1, max_length=256),
    if_match: str | None = Header(None, alias="If-Match"),
    ctx: SessionContext = Depends(get_session_context),
):
    """Retract a memory while preserving its revision history and artifacts."""
    _record("delete_memory", ctx)
    service = _get_service()
    resolved_revision = _resolve_revision_precondition(
        if_match=if_match, legacy_revision=expected_revision
    )
    try:
        result = service.delete_memory(
            memory_id=memory_id,
            user_id=ctx.user_id,
            owner_id=ctx.owner_id,
            session_agent_id=ctx.agent_id,
            expected_revision=resolved_revision,
            idempotency_key=idempotency_key,
        )
    except RevisionConflictError as e:
        raise HTTPException(status_code=409, detail=e.to_dict())
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if result.get("error"):
        raise HTTPException(
            status_code=400, detail=result.get("message", "Unknown error")
        )
    return result


@router.delete("/{memory_id}/privacy")
def privacy_erase_memory(
    memory_id: str,
    ctx: SessionContext = Depends(get_session_context),
):
    """Physically erase one memory lineage and its unreferenced artifacts."""
    _record("privacy_erase_memory", ctx)
    service = _get_service()
    try:
        return service.privacy_erase_memory(
            memory_id,
            user_id=ctx.user_id,
            owner_id=ctx.owner_id,
            session_agent_id=ctx.agent_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.get("/{memory_id}/history", response_model=MemoryHistoryResponse)
def get_memory_history(
    memory_id: str,
    revision_cursor: str | None = Query(None),
    operation_cursor: str | None = Query(None),
    limit: int = Query(50, ge=1, le=100),
    ctx: SessionContext = Depends(get_session_context),
):
    """Return immutable revisions and audited operations for one lineage."""
    _record("get_memory_history", ctx)
    service = _get_service()
    try:
        binding = {
            "user_id": ctx.user_id,
            "owner_id": ctx.owner_id,
            "session_agent_id": ctx.agent_id,
            "memory_id": memory_id,
        }
        revision_before = (
            int(decode_cursor(revision_cursor, {**binding, "kind": "revision"}))
            if revision_cursor
            else None
        )
        operation_before = (
            decode_cursor(operation_cursor, {**binding, "kind": "operation"})
            if operation_cursor
            else None
        )
        raw = service.get_memory_history_page(
            memory_id,
            user_id=ctx.user_id,
            owner_id=ctx.owner_id,
            session_agent_id=ctx.agent_id,
            revision_before=revision_before,
            operation_before=operation_before,
            limit=limit,
        )
        projected = project_history(raw, limit=limit)
        if raw.get("next_revision_before") is not None:
            projected.next_revision_cursor = encode_cursor(
                str(raw["next_revision_before"]),
                {**binding, "kind": "revision"},
            )
        if raw.get("next_operation_before"):
            projected.next_operation_cursor = encode_cursor(
                raw["next_operation_before"],
                {**binding, "kind": "operation"},
            )
        return projected
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.get("/{memory_id}/links", response_model=MemoryLinksResponse)
def get_memory_links(
    memory_id: str,
    ctx: SessionContext = Depends(get_session_context),
):
    """Return exact supersession and derivation links for one revision."""
    _record("get_memory_links", ctx)
    service = _get_service()
    try:
        links = service.get_memory_links(
            memory_id,
            user_id=ctx.user_id,
            owner_id=ctx.owner_id,
            session_agent_id=ctx.agent_id,
        )
        for field in ("supersedes", "successor"):
            link_id = links.get(field)
            links[field] = service.vector.get_by_id(link_id) if link_id else None
        for field in ("derived_from", "derived_outputs"):
            links[field] = [
                item
                for link_id in links.get(field, [])
                if (item := service.vector.get_by_id(link_id)) is not None
            ]
        return project_links(links)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


# ── Artifacts ─────────────────────────────────────────────────────────


@router.post("/{memory_id}/artifacts")
def save_artifact(
    memory_id: str,
    req: SaveArtifactRequest,
    if_match: str | None = Header(None, alias="If-Match"),
    ctx: SessionContext = Depends(get_session_context),
):
    """Attach an artifact to a memory."""
    _record("save_artifact", ctx)
    service = _get_service()
    expected_revision = _resolve_revision_precondition(
        if_match=if_match, legacy_revision=None
    )
    try:
        result = service.save_artifact(
            memory_id=memory_id,
            user_id=ctx.user_id,
            content=req.content,
            filename=req.filename,
            content_type=req.content_type,
            owner_id=ctx.owner_id,
            session_agent_id=ctx.agent_id,
            expected_revision=expected_revision,
        )
    except RevisionConflictError as e:
        raise HTTPException(status_code=409, detail=e.to_dict())
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if result.get("error"):
        raise HTTPException(
            status_code=400, detail=result.get("message", "Unknown error")
        )
    return result


@router.get("/{memory_id}/artifacts")
def list_artifacts(
    memory_id: str,
    ctx: SessionContext = Depends(get_session_context),
):
    """List artifacts attached to a memory."""
    _record("list_artifacts", ctx)
    service = _get_service()
    try:
        result = service.list_artifacts(
            memory_id=memory_id,
            user_id=ctx.user_id,
            owner_id=ctx.owner_id,
            session_agent_id=ctx.agent_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return result


@router.get("/{memory_id}/artifacts/{artifact_id}")
def get_artifact(
    memory_id: str,
    artifact_id: str,
    offset: int = Query(0, description="Character/byte offset"),
    limit: int = Query(5000, description="Max characters/bytes to return"),
    ctx: SessionContext = Depends(get_session_context),
):
    """Retrieve artifact content with pagination."""
    _record("get_artifact", ctx)
    service = _get_service()
    try:
        result = service.get_artifact(
            memory_id=memory_id,
            artifact_id=artifact_id,
            user_id=ctx.user_id,
            owner_id=ctx.owner_id,
            session_agent_id=ctx.agent_id,
            offset=offset,
            limit=limit,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Artifact not found")

    if result.get("error"):
        raise HTTPException(
            status_code=400, detail=result.get("message", "Unknown error")
        )
    return result


@router.post(
    "/{memory_id}/artifacts/{artifact_id}/download-token",
    response_model=DownloadTokenResponse,
)
def create_download_token(
    memory_id: str,
    artifact_id: str,
    req: DownloadTokenRequest | None = None,
    ctx: SessionContext = Depends(get_session_context),
):
    """Generate a short-lived signed download token for an artifact.

    Returns a token and URL that can be used for direct browser access
    (``<img src="...?token=...">``, download links) without requiring
    API key headers. Tokens are HMAC-signed, scoped to the specific
    artifact, and expire after ``ttl`` seconds.

    The returned URL includes the token as a query parameter and can
    be opened directly in a browser.
    """
    from mnemory.server import _get_config, _get_signing_key
    from mnemory.tokens import generate_download_token

    _record("create_download_token", ctx)
    service = _get_service()

    # Verify artifact exists and user has access.
    artifacts = service.list_artifacts(
        memory_id,
        user_id=ctx.user_id,
        owner_id=ctx.owner_id,
        session_agent_id=ctx.agent_id,
    )
    if not any(a["id"] == artifact_id for a in artifacts):
        raise HTTPException(status_code=404, detail="Artifact not found")

    cfg = _get_config().server
    ttl_seconds = min(
        (req.ttl if req and req.ttl else cfg.download_token_ttl),
        cfg.download_token_max_ttl,
    )

    token = generate_download_token(
        signing_key=_get_signing_key(),
        user_id=ctx.user_id,
        memory_id=memory_id,
        artifact_id=artifact_id,
        ttl_seconds=ttl_seconds,
        owner_id=ctx.owner_id,
        agent_id=ctx.agent_id,
    )

    path = f"/api/memories/{memory_id}/artifacts/{artifact_id}/raw?token={token}"
    base_url = cfg.base_url.rstrip("/") if cfg.base_url else ""
    url = f"{base_url}{path}" if base_url else path

    meta = next(a for a in artifacts if a["id"] == artifact_id)
    return DownloadTokenResponse(
        token=token,
        url=url,
        expires_in=ttl_seconds,
        content_type=meta.get("content_type", "application/octet-stream"),
        filename=meta.get("filename", "artifact"),
        size=meta.get("size", 0),
    )


@router.get("/{memory_id}/artifacts/{artifact_id}/raw")
def get_artifact_raw(
    memory_id: str,
    artifact_id: str,
    ctx: SessionContext = Depends(get_session_context),
):
    """Download raw artifact content with proper Content-Type header.

    Returns the raw bytes directly — suitable for rendering images in
    browsers, downloading files, or embedding via ``<img src="...">``.

    Authentication: Use a short-lived download token as a ``token``
    query parameter. Generate tokens via the ``download-token`` endpoint
    or the ``get_artifact_url`` MCP tool. Standard API key headers also
    work for programmatic access.
    """
    _record("get_artifact", ctx)
    service = _get_service()
    try:
        raw_bytes, content_type, filename = service.get_artifact_raw(
            memory_id=memory_id,
            artifact_id=artifact_id,
            user_id=ctx.user_id,
            owner_id=ctx.owner_id,
            session_agent_id=ctx.agent_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Artifact not found")

    # Sanitize filename for Content-Disposition header to prevent
    # header injection via quotes, newlines, or control characters.
    safe_name = (
        filename.replace('"', "").replace("\r", "").replace("\n", "").replace("\\", "")
    )
    if not safe_name:
        safe_name = "artifact"

    return Response(
        content=raw_bytes,
        media_type=content_type,
        headers={
            "Content-Disposition": f'inline; filename="{safe_name}"',
            "Content-Length": str(len(raw_bytes)),
            # Prevent caching of authenticated content and limit
            # API key leakage via Referer headers.
            "Cache-Control": "private, no-store",
            "Referrer-Policy": "no-referrer",
        },
    )


@router.delete("/{memory_id}/artifacts/{artifact_id}")
def delete_artifact(
    memory_id: str,
    artifact_id: str,
    if_match: str | None = Header(None, alias="If-Match"),
    ctx: SessionContext = Depends(get_session_context),
):
    """Delete an artifact from a memory."""
    _record("delete_artifact", ctx)
    service = _get_service()
    expected_revision = _resolve_revision_precondition(
        if_match=if_match, legacy_revision=None
    )
    try:
        result = service.delete_artifact(
            memory_id=memory_id,
            artifact_id=artifact_id,
            user_id=ctx.user_id,
            owner_id=ctx.owner_id,
            session_agent_id=ctx.agent_id,
            expected_revision=expected_revision,
        )
    except RevisionConflictError as e:
        raise HTTPException(status_code=409, detail=e.to_dict())
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if result.get("error"):
        raise HTTPException(
            status_code=400, detail=result.get("message", "Unknown error")
        )
    return result


# ── Categories ────────────────────────────────────────────────────────

categories_router = APIRouter()


@categories_router.get("", tags=["categories"])
def list_categories(
    ctx: SessionContext = Depends(get_session_context),
):
    """List all available memory categories with descriptions and counts."""
    _record("list_categories", ctx)
    service = _get_service()
    result = service.list_categories(user_id=ctx.user_id)
    return result
