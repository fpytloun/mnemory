"""MCP server for mnemory.

Exposes 19 tools over Streamable HTTP for memory management:
- initialize_memory (session init with instructions + core memories)
- add_memory, add_memories, search_memories, find_memories, ask_memories
- get_core_memories, get_recent_memories
- list_memories, update_memory, delete_memory, delete_memories, delete_all_memories
- list_categories
- save_artifact, get_artifact, get_artifact_url, list_artifacts, delete_artifact

Includes /health and /metrics endpoints, optional API key authentication,
session-level identity resolution (user_id from API key mapping,
agent_id from X-Agent-Id header), and an optional management port for
unauthenticated health/metrics access.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import hmac
import json
import logging
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from jwt import InvalidTokenError
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse, Response
from starlette.routing import Mount, Route

from mnemory import __version__
from mnemory.auth import get_jwt_validator, looks_like_jwt
from mnemory.config import load_config
from mnemory.instructions import VALID_MODES, build_instructions
from mnemory.memory import MemoryService
from mnemory.metrics import get_collector
from mnemory.revisions import RevisionConflictError
from mnemory.sanitize import escape_memory_headers

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("mnemory")

# ── Session Identity Context ──────────────────────────────────────────
# Set by the auth middleware per-request, read by tool handlers via
# _resolve_user_id / _resolve_agent_id helpers.

_session_user_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "session_user_id", default=None
)
_session_agent_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "session_agent_id", default=None
)
_session_owner_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "session_owner_id", default=None
)
_session_timezone: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "session_timezone", default=None
)
_session_user_bound: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "session_user_bound", default=False
)
_session_auth_kind: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "session_auth_kind", default=None
)
_session_trusted_evidence: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "session_trusted_evidence", default=False
)

# ── Configuration ─────────────────────────────────────────────────────
# Lazy initialization: config and service are created on first access
# to avoid import-time side effects (env var requirements, backend connections).

_config = None
_service = None
_service_lock = threading.Lock()


def _get_config():
    global _config
    if _config is None:
        _config = load_config()
    return _config


def _get_service():
    global _service
    if _service is not None:
        return _service
    # Thread-safe lazy initialization — prevents double-init when
    # multiple asyncio.to_thread() calls race during startup.
    with _service_lock:
        if _service is None:
            from mnemory.api import _session_store

            _service = MemoryService(_get_config(), session_store=_session_store)

            # Run data migrations after service init (collection exists).
            # Migrations run synchronously before the server accepts requests.
            _run_migrations(_service)
    return _service


_fsck_service = None
_maintenance_service = None
_signing_key: bytes | None = None

# Max binary artifact size (bytes) that can be returned inline via the
# MCP get_artifact tool. Larger binaries must use get_artifact_url.
_MCP_BINARY_INLINE_LIMIT = 1_048_576  # 1 MB


def _get_signing_key() -> bytes:
    """Get or derive the HMAC signing key for download tokens.

    Cached after first call. Derived from API key material so it
    survives restarts (unless no auth is configured, in which case
    an ephemeral random key is used).
    """
    global _signing_key
    if _signing_key is None:
        from mnemory.tokens import derive_signing_key

        cfg = _get_config().server
        if (
            not cfg.api_key
            and not cfg.api_keys
            and (cfg.jwt_public_key or cfg.jwks_url)
        ):
            logger.warning(
                "JWT auth is enabled without API keys; artifact download tokens "
                "remain ephemeral across restarts in this stage-0 implementation"
            )
        _signing_key = derive_signing_key(
            api_key=cfg.api_key,
            api_keys=cfg.api_keys,
        )
    return _signing_key


# Migration state flag — health endpoint returns 503 while True
_migration_running = False


def _run_migrations(service: MemoryService) -> None:
    """Run pending data migrations on startup.

    Called once during service initialization, before the server
    accepts requests. The health endpoint returns 503 while
    migrations are running (Kubernetes readiness probe integration).
    """
    global _migration_running
    from mnemory.migration import MigrationRunner, get_migrations

    sparse = getattr(service, "_sparse", None)
    runner = MigrationRunner(
        client=service.vector._client,
        migrations=get_migrations(
            collection_name=service.vector.collection_name,
            sparse_client=sparse,
        ),
    )

    _migration_running = True
    try:
        runner.run_pending()
    finally:
        _migration_running = False

    # Recreate payload indexes after migrations. If a migration recreated
    # the collection (e.g. to add sparse vector config), the original
    # payload indexes were lost. _ensure_indexes() is idempotent.
    try:
        service.vector._ensure_indexes(
            labels_indexes=service._config.memory.labels_indexes or None,
        )
    except Exception:
        logger.warning(
            "Failed to ensure payload indexes after migration",
            exc_info=True,
        )


def _get_maintenance_service():
    """Get the MaintenanceService singleton (may be None if not yet started)."""
    return _maintenance_service


def _get_fsck_service():
    """Get or create the FsckService singleton."""
    global _fsck_service
    if _fsck_service is None:
        from mnemory.fsck import FsckService, FsckStore

        cfg = _get_config()
        service = _get_service()

        # Use a separate LLM client for fsck if FSCK_LLM_MODEL is set,
        # otherwise reuse the main LLM client.
        if cfg.memory.fsck_model:
            from mnemory.config import LLMConfig
            from mnemory.llm import LLMClient

            fsck_llm_config = LLMConfig(
                model=cfg.memory.fsck_model,
                base_url=cfg.llm.base_url,
                api_key=cfg.llm.api_key,
                temperature=cfg.llm.temperature,
                reasoning_effort=cfg.llm.reasoning_effort,
            )
            fsck_llm = LLMClient(fsck_llm_config)
        else:
            fsck_llm = service._llm

        _fsck_service = FsckService(
            config=cfg,
            vector=service.vector,
            llm=fsck_llm,
            store=FsckStore(default_ttl=cfg.memory.fsck_cache_ttl),
            memory_service=service,
        )
        # Start the cleanup task immediately if an event loop is already
        # running (i.e., the service is first used during a request, after
        # lifespan startup). The lifespan guard is kept as a belt-and-suspenders
        # fallback for the rare case where the service is created before the
        # first request.
        import asyncio as _asyncio

        try:
            loop = _asyncio.get_running_loop()
            if loop.is_running():
                _fsck_service._store.start_cleanup_task()
        except RuntimeError:
            pass  # No event loop yet — lifespan will start it
    return _fsck_service


# ── Identity Resolution ───────────────────────────────────────────────


def _resolve_user_id(param_user_id: str | None = None) -> str:
    """Resolve user_id from session context or tool parameter.

    Priority:
    1. API key mapping (set by middleware from MCP_API_KEYS, non-wildcard)
    2. X-User-Id header (set by middleware)
    3. Tool parameter (backward compat)
    4. Error if none available
    """
    session_uid = _session_user_id.get()
    if session_uid:
        return session_uid
    if param_user_id:
        return param_user_id
    raise ValueError(
        "user_id is required — configure via API key mapping (MCP_API_KEYS), "
        "X-User-Id header, or pass as tool parameter"
    )


def _is_sub_agent(agent_id: str, session_agent_id: str) -> bool:
    """Check if agent_id is a sub-agent of the session agent.

    Sub-agents use a colon-separated prefix: if the session agent is
    "openwebui", then "openwebui:bob" is a valid sub-agent. This allows
    creating multiple independent agent identities under a single session.
    """
    return agent_id.startswith(session_agent_id + ":")


def _resolve_agent_id(param_agent_id: str | None = None) -> str | None:
    """Resolve agent_id with session-level protection.

    Supports "self" as a sentinel value — resolves to the session's
    agent_id from X-Agent-Id header. Raises ValueError if "self" is
    used without a session agent.

    When X-Agent-Id header is set (session has an agent):
    - param="self"  → returns session agent_id
    - param=None  → returns None (shared memory, no agent scope)
    - param=session value → returns session agent_id
    - param="session:sub" → returns param (sub-agent allowed)
    - param=different value → raises ValueError (cross-agent blocked)

    When X-Agent-Id is NOT set:
    - param="self" → raises ValueError (no session agent to resolve)
    - returns param as-is (no protection, backward compatible)
    """
    session_aid = _session_agent_id.get()
    if param_agent_id == "self":
        if session_aid is None:
            raise ValueError("agent_id='self' requires X-Agent-Id header to be set")
        return session_aid
    if session_aid is None:
        # No session agent — pass through whatever the LLM sent
        return param_agent_id
    # Session agent is set — enforce protection
    if param_agent_id is None:
        return None  # LLM wants shared/no agent scope
    if param_agent_id == session_aid:
        return session_aid
    if _is_sub_agent(param_agent_id, session_aid):
        return param_agent_id
    raise ValueError(
        f"Cannot use agent_id '{param_agent_id}' — "
        f"session is bound to agent '{session_aid}'"
    )


def _resolve_agent_id_for_core(
    param_agent_id: str | None = None,
) -> str | None:
    """Resolve agent_id for get_core_memories (auto-injects session agent).

    Unlike _resolve_agent_id, this auto-injects the session agent_id
    when the LLM doesn't pass one, because get_core_memories always
    needs the agent_id to load agent identity and knowledge.

    Supports "self" sentinel and sub-agents — same as _resolve_agent_id.
    """
    session_aid = _session_agent_id.get()
    if param_agent_id == "self":
        if session_aid is None:
            raise ValueError("agent_id='self' requires X-Agent-Id header to be set")
        return session_aid
    if session_aid is None:
        return param_agent_id
    if param_agent_id is None:
        return session_aid  # Auto-inject
    if param_agent_id == session_aid:
        return session_aid
    if _is_sub_agent(param_agent_id, session_aid):
        return param_agent_id
    raise ValueError(
        f"Cannot use agent_id '{param_agent_id}' — "
        f"session is bound to agent '{session_aid}'"
    )


def _get_session_agent_id() -> str | None:
    """Get the session-level agent_id (from X-Agent-Id header), or None."""
    return _session_agent_id.get()


def _get_session_owner_id() -> str | None:
    """Get the session-level agent owner id, or None."""
    return _session_owner_id.get()


def _get_session_timezone() -> str | None:
    """Get the session-level timezone (from X-Timezone header), or None."""
    return _session_timezone.get()


# ── MCP Server ────────────────────────────────────────────────────────

mcp = FastMCP(
    "mnemory",
    instructions=build_instructions(os.environ.get("INSTRUCTION_MODE", "proactive")),
    stateless_http=True,
    json_response=True,
    # Disable DNS rebinding protection — mnemory runs behind a reverse
    # proxy / ingress that handles host validation and TLS termination.
    # Without this, the MCP SDK rejects requests with external Host
    # headers (e.g., mem.fpy.cz) with HTTP 421.
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=False,
    ),
)


# ── Tool 0: initialize_memory ────────────────────────────────────────


@mcp.tool()
async def initialize_memory(
    user_id: str | None = None,
    agent_id: str | None = None,
    recent_days: int = 7,
    mode: str | None = None,
    include_instructions: bool = True,
) -> str:
    """ALWAYS call this at the start of every conversation to initialize memory.

    Returns core memories (pinned facts, identity, recent context).
    Also returns behavioral instructions unless include_instructions=False.
    This MUST be the first tool called in every session.

    Args:
        include_instructions: Return instructions in response (default true).
            Set to false if your client already injects MCP server
            instructions into the system prompt.
        recent_days: Days back for recent context (default 7).
        mode: Instruction mode override: passive, proactive, or personality.
    """
    # Validate mode if provided
    if mode is not None and mode not in VALID_MODES:
        return json.dumps(
            {
                "error": True,
                "message": f"Invalid mode: '{mode}'. Must be one of: {', '.join(VALID_MODES)}",
            }
        )

    # Determine effective mode
    effective_mode = mode if mode is not None else _get_config().server.instruction_mode

    # Build instructions (skip if client already has them via MCP instructions field)
    instructions = None
    if include_instructions:
        instructions = build_instructions(effective_mode)

    # Get core memories (fail gracefully).
    # ContextVars are resolved eagerly for clarity; asyncio.to_thread()
    # copies context automatically via copy_context().
    core_memories_text = None
    core_memories_error = None
    try:
        uid = _resolve_user_id(user_id)
        owner_id = _get_session_owner_id()
        aid = _resolve_agent_id_for_core(agent_id)
        collector = get_collector()
        if collector:
            collector.record_operation("initialize_memory", uid, aid)
        service = _get_service()
        core_result = await asyncio.to_thread(
            service.get_core_memories,
            user_id=uid,
            owner_id=owner_id,
            agent_id=aid,
            recent_days=recent_days,
        )
        core_memories_text = core_result.text
    except ValueError as e:
        # Sanitize error message to prevent user-influenced content
        # from being injected into the LLM context via error text
        err_msg = str(e)[:200].replace("\n", " ")
        core_memories_error = err_msg
    except Exception:
        logger.exception("Error getting core memories in initialize_memory")
        core_memories_error = "Internal error loading core memories"

    # Build response
    parts = []
    if instructions:
        parts.extend(["## MEMORY INSTRUCTIONS", "", instructions])

    if core_memories_text is not None:
        parts.extend(["", "---", "", "## CORE MEMORIES", "", core_memories_text])
    elif core_memories_error:
        parts.extend(
            [
                "",
                "---",
                "",
                "## CORE MEMORIES",
                "",
                f"(Error loading core memories: {core_memories_error})",
            ]
        )

    return "\n".join(parts)


# ── Tool 1: add_memory ───────────────────────────────────────────────


@mcp.tool()
async def add_memory(
    content: str,
    user_id: str | None = None,
    memory_type: str | None = None,
    categories: list[str] | None = None,
    importance: str | None = None,
    pinned: bool | None = None,
    agent_id: str | None = None,
    infer: bool = True,
    role: str = "user",
    ttl_days: int | None = None,
    event_date: str | None = None,
    labels: dict[str, Any] | None = None,
) -> str:
    """Store a memory about the user or agent.

    Call this whenever the user shares personal information, preferences,
    facts, decisions, project context, or anything worth remembering.

    Content must be concise (max 1000 chars). For detailed content, store
    a summary here and attach the full content with save_artifact.

    All metadata fields are OPTIONAL — if omitted, the server auto-classifies
    them using an LLM. You can provide any combination of fields; only the
    missing ones will be auto-classified.

    Args:
        content: The memory to store (max 1000 chars).
        memory_type: preference, fact, episodic, procedural, or context.
        categories: Predefined tags. Call list_categories to see options.
        importance: low, normal, high, or critical.
        pinned: Load at every conversation start.
        agent_id: Scope to agent. Use "self" for yours.
        infer: Run fact extraction and deduplication (default true). Leave
            true unless performing maintenance or bulk import.
        role: "user" (default) or "assistant" (requires agent_id).
        ttl_days: Time-to-live in days. Omit for type defaults.
        event_date: ISO 8601 datetime of when the event occurred.
        labels: Key-value metadata for filtering.
    """
    try:
        uid = _resolve_user_id(user_id)
        aid = _resolve_agent_id(agent_id)
        owner_id = _get_session_owner_id()
        tz = _get_session_timezone()
        collector = get_collector()
        if collector:
            collector.record_operation("add_memory", uid, aid)

        # Server-side infer control: when ALLOW_CLIENT_INFER=False,
        # always use infer=True regardless of client request.
        effective_infer = infer
        if not infer and not _get_config().memory.allow_client_infer:
            logger.debug("Overriding infer=False to True (ALLOW_CLIENT_INFER=False)")
            effective_infer = True

        service = _get_service()
        result = await asyncio.to_thread(
            service.add_memory,
            content,
            user_id=uid,
            owner_id=owner_id,
            agent_id=aid,
            memory_type=memory_type,
            categories=categories,
            importance=importance,
            pinned=pinned,
            infer=effective_infer,
            role=role,
            ttl_days=ttl_days,
            event_date=event_date,
            labels=labels,
            session_timezone=tz,
        )
        return json.dumps(result, default=str)
    except ValueError as e:
        return json.dumps({"error": True, "message": str(e)})
    except Exception:
        logger.exception("Error in add_memory")
        return json.dumps(
            {"error": True, "message": "Internal error processing add_memory"}
        )


# ── Tool 1b: add_memories (batch) ─────────────────────────────────────

MAX_BATCH_SIZE = 20


@mcp.tool()
async def add_memories(
    memories: list[dict],
    user_id: str | None = None,
    agent_id: str | None = None,
    infer: bool = True,
    role: str = "user",
) -> str:
    """Store multiple memories in a single call (batch operation).

    Each memory is processed independently — failures on individual
    items do not block the rest.

    Args:
        memories: List of memory objects, each with "content" (str) and
                  optional: memory_type, categories, importance, pinned,
                  role, ttl_days, event_date, labels.
        agent_id: Scope to agent. Use "self" for yours.
        infer: Run fact extraction and deduplication (default true). Leave
              true unless performing maintenance or bulk import.
        role: Default role for all items ("user" or "assistant").
    """
    try:
        uid = _resolve_user_id(user_id)
        aid = _resolve_agent_id(agent_id)
        owner_id = _get_session_owner_id()
    except ValueError as e:
        return json.dumps({"error": True, "message": str(e)})

    tz = _get_session_timezone()
    collector = get_collector()
    if collector:
        collector.record_operation("add_memories", uid, aid)

    if not memories:
        return json.dumps({"error": True, "message": "memories list is empty"})

    if len(memories) > MAX_BATCH_SIZE:
        return json.dumps(
            {
                "error": True,
                "message": f"Too many memories: {len(memories)} (max {MAX_BATCH_SIZE})",
            }
        )

    # Server-side infer control
    effective_infer = infer
    if not infer and not _get_config().memory.allow_client_infer:
        logger.debug("Overriding infer=False to True (ALLOW_CLIENT_INFER=False)")
        effective_infer = True

    service = _get_service()

    # Run the entire batch loop in a thread to avoid blocking the event
    # loop during sequential LLM + embedding + Qdrant calls per item.
    def _process_batch():
        results = []
        errors = []

        for i, mem in enumerate(memories):
            if not isinstance(mem, dict) or "content" not in mem:
                errors.append(
                    {
                        "index": i,
                        "error": True,
                        "message": "Each memory must be an object with a 'content' field",
                    }
                )
                continue

            try:
                item_role = mem.get("role", role)
                item_labels = mem.get("labels")
                result = service.add_memory(
                    mem["content"],
                    user_id=uid,
                    owner_id=owner_id,
                    agent_id=aid,
                    memory_type=mem.get("memory_type"),
                    categories=mem.get("categories"),
                    importance=mem.get("importance"),
                    pinned=mem.get("pinned"),
                    infer=effective_infer,
                    role=item_role,
                    ttl_days=mem.get("ttl_days"),
                    event_date=mem.get("event_date"),
                    labels=item_labels,
                    session_timezone=tz,
                )
                # Check for error returned by add_memory (e.g., content too long)
                if result.get("error"):
                    errors.append({"index": i, **result})
                else:
                    results.append({"index": i, **result})
            except ValueError as e:
                errors.append({"index": i, "error": True, "message": str(e)})
            except Exception:
                logger.exception("Error in add_memories item %d", i)
                errors.append(
                    {
                        "index": i,
                        "error": True,
                        "message": f"Internal error processing item {i}",
                    }
                )

        return results, errors

    results, errors = await asyncio.to_thread(_process_batch)

    return json.dumps(
        {
            "results": results,
            "errors": errors,
            "total": len(memories),
            "succeeded": len(results),
            "failed": len(errors),
        },
        default=str,
    )


# ── Tool 2: search_memories ──────────────────────────────────────────


@mcp.tool()
async def search_memories(
    query: str,
    user_id: str | None = None,
    memory_type: str | None = None,
    categories: list[str] | None = None,
    role: str | None = None,
    limit: int = 10,
    agent_id: str | None = None,
    include_decayed: bool = False,
    date_start: str | None = None,
    date_end: str | None = None,
    labels: dict[str, Any] | None = None,
) -> str:
    """Search memories by semantic similarity with filtering and importance reranking.

    Results are ranked by relevance and importance. Memories with artifacts
    show has_artifacts: true — use get_artifact to fetch details.

    Args:
        query: What to search for (natural language).
        memory_type: Filter: preference, fact, episodic, procedural, context.
        categories: Filter by categories. "project" matches all project:*.
        role: Filter: "user" or "assistant". Omit for all.
        limit: Max results (default 10).
        include_decayed: Include expired memories (default false).
        date_start: Filter start date (YYYY-MM-DD).
        date_end: Filter end date (YYYY-MM-DD).
        labels: Filter by label key-value pairs (AND logic).
    """
    try:
        uid = _resolve_user_id(user_id)
        session_aid = _get_session_agent_id()
        owner_id = _get_session_owner_id()
        collector = get_collector()
        if collector:
            collector.record_operation("search_memories", uid, session_aid)

        service = _get_service()

        # When session has agent_id, always use dual-scope to return
        # both agent-specific and shared user memories.
        # Honor explicit agent_id from tool param (e.g. sub-agent scope).
        if session_aid:
            effective_aid = (
                (_resolve_agent_id(agent_id) or session_aid)
                if agent_id
                else session_aid
            )
            results = await asyncio.to_thread(
                service.search_memories_dual_scope,
                query,
                user_id=uid,
                owner_id=owner_id,
                session_agent_id=effective_aid,
                memory_type=memory_type,
                categories=categories,
                role=role,
                limit=limit,
                include_decayed=include_decayed,
                date_start=date_start,
                date_end=date_end,
                labels=labels,
            )
        else:
            # No session agent — use param for backward compat
            aid = _resolve_agent_id(agent_id)
            results = await asyncio.to_thread(
                service.search_memories,
                query,
                user_id=uid,
                owner_id=owner_id,
                agent_id=aid,
                memory_type=memory_type,
                categories=categories,
                role=role,
                limit=limit,
                include_decayed=include_decayed,
                date_start=date_start,
                date_end=date_end,
                labels=labels,
            )
        return _format_memories(results)
    except ValueError as e:
        return json.dumps({"error": True, "message": str(e)})
    except Exception:
        logger.exception("Error in search_memories")
        return json.dumps(
            {"error": True, "message": "Internal error processing search_memories"}
        )


# ── Tool: find_memories ───────────────────────────────────────────────


@mcp.tool()
async def find_memories(
    question: str,
    user_id: str | None = None,
    memory_type: str | None = None,
    categories: list[str] | None = None,
    role: str | None = None,
    limit: int = 10,
    agent_id: str | None = None,
    include_decayed: bool = False,
    context: str | None = None,
    labels: dict[str, Any] | None = None,
) -> str:
    """Find memories relevant to a complex question using AI-powered search.

    Generates multiple targeted searches covering different angles and
    associations, then reranks results by relevance. Temporal-aware.
    Slower than search_memories (2 extra LLM calls) but higher quality
    for complex, multi-faceted questions.

    Args:
        question: The question in natural language.
        memory_type: Filter: preference, fact, episodic, procedural, context.
        categories: Filter by categories. "project" matches all project:*.
        role: Filter: "user" or "assistant". Omit for all.
        limit: Max results (default 10).
        include_decayed: Include expired memories (default false).
        context: Optional context hint for query generation.
        labels: Filter by label key-value pairs (AND logic).
    """
    try:
        uid = _resolve_user_id(user_id)
        session_aid = _get_session_agent_id()
        owner_id = _get_session_owner_id()
        collector = get_collector()
        if collector:
            collector.record_operation("find_memories", uid, session_aid)

        session_tz = _get_session_timezone()
        service = _get_service()

        if session_aid:
            effective_aid = (
                (_resolve_agent_id(agent_id) or session_aid)
                if agent_id
                else session_aid
            )
            result = await asyncio.to_thread(
                service.find_memories,
                question,
                user_id=uid,
                owner_id=owner_id,
                session_agent_id=effective_aid,
                memory_type=memory_type,
                categories=categories,
                role=role,
                limit=limit,
                include_decayed=include_decayed,
                session_timezone=session_tz,
                context=context,
                labels=labels,
            )
        else:
            aid = _resolve_agent_id(agent_id)
            result = await asyncio.to_thread(
                service.find_memories,
                question,
                user_id=uid,
                owner_id=owner_id,
                agent_id=aid,
                memory_type=memory_type,
                categories=categories,
                role=role,
                limit=limit,
                include_decayed=include_decayed,
                session_timezone=session_tz,
                context=context,
                labels=labels,
            )
        memories = result.get("results", [])
        queries = result.get("queries", [])
        stats = result.get("stats", {})
        formatted = _format_memories(memories)
        # Inject queries and stats into the response JSON
        response = json.loads(formatted)
        response["queries"] = queries
        response["stats"] = stats
        return json.dumps(response, default=str)
    except ValueError as e:
        return json.dumps({"error": True, "message": str(e)})
    except Exception:
        logger.exception("Error in find_memories")
        return json.dumps(
            {"error": True, "message": "Internal error processing find_memories"}
        )


# ── Tool 2b: ask_memories ────────────────────────────────────────────


@mcp.tool()
async def ask_memories(
    question: str,
    user_id: str | None = None,
    memory_type: str | None = None,
    categories: list[str] | None = None,
    role: str | None = None,
    limit: int = 10,
    agent_id: str | None = None,
    include_decayed: bool = False,
    context: str | None = None,
    include_memories: bool = False,
    labels: dict[str, Any] | None = None,
) -> str:
    """Ask a question and get a human-readable answer based on stored memories.

    Uses find_memories internally, then generates a natural language answer.
    Most expensive operation (3 LLM calls). Use when you need a synthesized
    answer rather than raw memory results.

    Args:
        question: The question in natural language.
        memory_type: Filter: preference, fact, episodic, procedural, context.
        categories: Filter by categories. "project" matches all project:*.
        role: Filter: "user" or "assistant". Omit for all.
        limit: Max supporting memories (default 10).
        include_decayed: Include expired memories (default false).
        context: Optional context hint for query generation.
        include_memories: Include supporting memories in response (default false).
        labels: Filter by label key-value pairs (AND logic).
    """
    try:
        uid = _resolve_user_id(user_id)
        session_aid = _get_session_agent_id()
        owner_id = _get_session_owner_id()
        collector = get_collector()
        if collector:
            collector.record_operation("ask_memories", uid, session_aid)

        session_tz = _get_session_timezone()
        service = _get_service()

        if session_aid:
            effective_aid = (
                (_resolve_agent_id(agent_id) or session_aid)
                if agent_id
                else session_aid
            )
            result = await asyncio.to_thread(
                service.answer_question,
                question,
                user_id=uid,
                owner_id=owner_id,
                session_agent_id=effective_aid,
                memory_type=memory_type,
                categories=categories,
                role=role,
                limit=limit,
                include_decayed=include_decayed,
                session_timezone=session_tz,
                context=context,
                labels=labels,
            )
        else:
            aid = _resolve_agent_id(agent_id)
            result = await asyncio.to_thread(
                service.answer_question,
                question,
                user_id=uid,
                owner_id=owner_id,
                agent_id=aid,
                memory_type=memory_type,
                categories=categories,
                role=role,
                limit=limit,
                include_decayed=include_decayed,
                session_timezone=session_tz,
                context=context,
                labels=labels,
            )

        answer = result.get("answer", "")
        queries = result.get("queries", [])
        stats = result.get("stats", {})

        if include_memories:
            memories = result.get("results", [])
            formatted = _format_memories(memories)
            response = json.loads(formatted)
            response["count"] = len(memories)
        else:
            response = {"results": [], "count": 0}

        response["answer"] = answer
        response["queries"] = queries
        response["stats"] = stats
        return json.dumps(response, default=str)
    except ValueError as e:
        return json.dumps({"error": True, "message": str(e)})
    except Exception:
        logger.exception("Error in ask_memories")
        return json.dumps(
            {"error": True, "message": "Internal error processing ask_memories"}
        )


# ── Tool 3: get_core_memories ────────────────────────────────────────


@mcp.tool()
async def get_core_memories(
    user_id: str | None = None,
    agent_id: str | None = None,
    recent_days: int = 7,
) -> str:
    """Load pinned memories and recent context. Equivalent to
    initialize_memory(include_instructions=False).

    Prefer calling initialize_memory directly.

    Args:
        recent_days: Days back for recent context (default 7).
    """
    try:
        uid = _resolve_user_id(user_id)
        aid = _resolve_agent_id_for_core(agent_id)
        owner_id = _get_session_owner_id()
        collector = get_collector()
        if collector:
            collector.record_operation("get_core_memories", uid, aid)
        service = _get_service()
        core_result = await asyncio.to_thread(
            service.get_core_memories,
            user_id=uid,
            owner_id=owner_id,
            agent_id=aid,
            recent_days=recent_days,
        )
        return core_result.text
    except ValueError as e:
        return json.dumps({"error": True, "message": str(e)})
    except Exception:
        logger.exception("Error in get_core_memories")
        return json.dumps(
            {"error": True, "message": "Internal error processing get_core_memories"}
        )


# ── Tool 3b: get_recent_memories ─────────────────────────────────────


@mcp.tool()
async def get_recent_memories(
    user_id: str | None = None,
    agent_id: str | None = None,
    days: int = 7,
    scope: str = "all",
    limit: int = 25,
    include_decayed: bool = False,
) -> str:
    """Get recent memories from the last N days.

    Returns memories of all types ordered by most recent first.

    Args:
        days: How many days back to look (default 7).
        scope: "all" (default), "user", or "agent".
        limit: Max results per scope (default 25).
        include_decayed: Include expired memories (default false).
    """
    try:
        uid = _resolve_user_id(user_id)
        aid = _resolve_agent_id(agent_id)
        owner_id = _get_session_owner_id()
        # Auto-resolve agent_id from session for scopes that include
        # agent memories, so the LLM doesn't have to pass it explicitly.
        if aid is None and scope in ("all", "agent"):
            aid = _session_agent_id.get()
        collector = get_collector()
        if collector:
            collector.record_operation("get_recent_memories", uid, aid)
        service = _get_service()
        return await asyncio.to_thread(
            service.get_recent_memories,
            user_id=uid,
            owner_id=owner_id,
            agent_id=aid,
            days=days,
            scope=scope,
            limit=limit,
            include_decayed=include_decayed,
        )
    except ValueError as e:
        return json.dumps({"error": True, "message": str(e)})
    except Exception:
        logger.exception("Error in get_recent_memories")
        return json.dumps(
            {"error": True, "message": "Internal error processing get_recent_memories"}
        )


# ── Tool 4: list_memories ────────────────────────────────────────────


@mcp.tool()
async def list_memories(
    user_id: str | None = None,
    memory_type: str | None = None,
    categories: list[str] | None = None,
    role: str | None = None,
    limit: int = 50,
    agent_id: str | None = None,
    include_decayed: bool = False,
    labels: dict[str, Any] | None = None,
    memory_layer: str | None = None,
) -> str:
    """List all stored memories for a user, optionally filtered.

    Args:
        memory_type: Filter: preference, fact, episodic, procedural, context.
        categories: Filter by categories.
        role: Filter: "user" or "assistant". Omit for all.
        limit: Max results (default 50).
        include_decayed: Include expired memories (default false).
        labels: Filter by label key-value pairs (AND logic).
        memory_layer: Filter by memory layer: "raw" or "consolidated". Omit for all.
    """
    try:
        if memory_layer and memory_layer not in ("raw", "consolidated"):
            return json.dumps(
                {
                    "error": True,
                    "message": "memory_layer must be 'raw' or 'consolidated'",
                }
            )

        uid = _resolve_user_id(user_id)
        session_aid = _get_session_agent_id()
        owner_id = _get_session_owner_id()
        collector = get_collector()
        if collector:
            collector.record_operation("list_memories", uid, session_aid)

        service = _get_service()

        # When session has agent_id, always use dual-scope to return
        # both agent-specific and shared user memories.
        # Honor explicit agent_id from tool param (e.g. sub-agent scope).
        if session_aid:
            effective_aid = (
                (_resolve_agent_id(agent_id) or session_aid)
                if agent_id
                else session_aid
            )
            results = await asyncio.to_thread(
                service.list_memories_dual_scope,
                user_id=uid,
                owner_id=owner_id,
                session_agent_id=effective_aid,
                memory_type=memory_type,
                categories=categories,
                role=role,
                limit=limit,
                include_decayed=include_decayed,
                labels=labels,
            )
        else:
            # No session agent — use param for backward compat
            aid = _resolve_agent_id(agent_id)
            results = await asyncio.to_thread(
                service.list_memories,
                user_id=uid,
                owner_id=owner_id,
                agent_id=aid,
                memory_type=memory_type,
                categories=categories,
                role=role,
                limit=limit,
                include_decayed=include_decayed,
                labels=labels,
            )

        # Filter by memory_layer if specified (post-retrieval filtering)
        if memory_layer:
            results = [
                m
                for m in results
                if (m.get("metadata") or {}).get("memory_layer", "consolidated")
                == memory_layer
            ]

        return _format_memories(results)
    except ValueError as e:
        return json.dumps({"error": True, "message": str(e)})
    except Exception:
        logger.exception("Error in list_memories")
        return json.dumps(
            {"error": True, "message": "Internal error processing list_memories"}
        )


# ── Tool 5: update_memory ────────────────────────────────────────────


@mcp.tool()
async def update_memory(
    memory_id: str,
    content: str | None = None,
    memory_type: str | None = None,
    categories: list[str] | None = None,
    importance: str | None = None,
    pinned: bool | None = None,
    ttl_days: int | None = None,
    event_date: str | None = None,
    labels: dict[str, Any] | None = None,
    expected_revision: int | None = None,
    idempotency_key: str | None = None,
) -> str:
    """Update an existing memory's content or metadata.

    Args:
        memory_id: ID of the memory to update.
        content: New content text (max 1000 chars).
        memory_type: New type: preference, fact, episodic, procedural, context.
        categories: New categories (replaces existing).
        importance: New importance: low, normal, high, critical.
        pinned: New pinned state.
        ttl_days: New TTL in days. Restores decayed memories. Null = permanent.
        event_date: New event date (ISO 8601). Null to clear.
        labels: New labels (merged with existing). Empty dict to clear.
        expected_revision: Expected active revision for stale-write detection.
        idempotency_key: Retry key scoped to this lineage update.
    """
    try:
        # Verify ownership: must be own agent's memory or shared
        session_aid = _get_session_agent_id()
        session_uid = _session_user_id.get()
        owner_id = _get_session_owner_id()
        collector = get_collector()
        if collector:
            collector.record_operation("update_memory", session_uid or "", session_aid)
        service = _get_service()

        # Build kwargs, using sentinel for ttl_days to distinguish
        # "not provided" from "explicitly set to None"
        kwargs: dict = {
            "user_id": session_uid,
            "owner_id": owner_id,
            "session_agent_id": session_aid,
            "content": content,
            "memory_type": memory_type,
            "categories": categories,
            "importance": importance,
            "pinned": pinned,
            "expected_revision": expected_revision,
            "idempotency_key": idempotency_key,
        }
        if ttl_days is not None:
            kwargs["ttl_days"] = ttl_days
        if event_date is not None:
            kwargs["event_date"] = event_date
        if labels is not None:
            kwargs["labels"] = labels

        result = await asyncio.to_thread(service.update_memory, memory_id, **kwargs)
        return json.dumps(result, default=str)
    except RevisionConflictError as e:
        return json.dumps(e.to_dict())
    except ValueError as e:
        return json.dumps({"error": True, "message": str(e)})
    except Exception:
        logger.exception("Error in update_memory")
        return json.dumps(
            {"error": True, "message": "Internal error processing update_memory"}
        )


# ── Tool 6: delete_memory ────────────────────────────────────────────


@mcp.tool()
async def delete_memory(
    memory_id: str,
    user_id: str | None = None,
    expected_revision: int | None = None,
    idempotency_key: str | None = None,
    privacy_erase: bool = False,
) -> str:
    """Retract a memory, or explicitly erase its lineage for privacy.

    Args:
        memory_id: ID of the memory to delete.
        expected_revision: Expected active revision for stale-write detection.
        idempotency_key: Retry key scoped to this lineage retraction.
        privacy_erase: Physically erase all revisions and artifacts.
    """
    try:
        uid = _resolve_user_id(user_id)
        # Verify ownership: must be own agent's memory or shared
        session_aid = _get_session_agent_id()
        owner_id = _get_session_owner_id()
        collector = get_collector()
        if collector:
            collector.record_operation("delete_memory", uid, session_aid)
        service = _get_service()

        if privacy_erase:
            result = await asyncio.to_thread(
                service.privacy_erase_memory,
                memory_id,
                user_id=uid,
                owner_id=owner_id,
                session_agent_id=session_aid,
            )
        else:
            result = await asyncio.to_thread(
                service.delete_memory,
                memory_id,
                user_id=uid,
                owner_id=owner_id,
                session_agent_id=session_aid,
                expected_revision=expected_revision,
                idempotency_key=idempotency_key,
            )
        return json.dumps(result)
    except RevisionConflictError as e:
        return json.dumps(e.to_dict())
    except ValueError as e:
        return json.dumps({"error": True, "message": str(e)})
    except Exception:
        logger.exception("Error in delete_memory")
        return json.dumps(
            {"error": True, "message": "Internal error processing delete_memory"}
        )


# ── Tool 6b: delete_memories (batch) ────────────────────────────────


@mcp.tool()
async def delete_memories(
    memory_ids: list[str],
    user_id: str | None = None,
    idempotency_key: str | None = None,
) -> str:
    """Delete multiple memories in a single call (batch operation).

    Args:
        memory_ids: List of memory IDs to delete.
    """
    try:
        uid = _resolve_user_id(user_id)
    except ValueError as e:
        return json.dumps({"error": True, "message": str(e)})

    session_aid = _get_session_agent_id()
    owner_id = _get_session_owner_id()
    collector = get_collector()
    if collector:
        collector.record_operation("delete_memories", uid, session_aid)

    if not memory_ids:
        return json.dumps({"error": True, "message": "memory_ids list is empty"})

    if len(memory_ids) > MAX_BATCH_SIZE:
        return json.dumps(
            {
                "error": True,
                "message": (
                    f"Too many memory IDs: {len(memory_ids)} (max {MAX_BATCH_SIZE})"
                ),
            }
        )

    service = _get_service()

    def _process_batch():
        results = []
        errors = []

        for i, mid in enumerate(memory_ids):
            try:
                result = service.delete_memory(
                    mid,
                    user_id=uid,
                    owner_id=owner_id,
                    session_agent_id=session_aid,
                    idempotency_key=(
                        f"{idempotency_key}:{i}" if idempotency_key else None
                    ),
                )
                results.append({"index": i, **result})
            except RevisionConflictError as e:
                errors.append({"index": i, **e.to_dict()})
            except ValueError as e:
                errors.append({"index": i, "error": True, "message": str(e)})
            except Exception:
                logger.exception("Error in delete_memories item %d", i)
                errors.append(
                    {
                        "index": i,
                        "error": True,
                        "message": f"Internal error processing item {i}",
                    }
                )

        return results, errors

    results, errors = await asyncio.to_thread(_process_batch)

    return json.dumps(
        {
            "results": results,
            "errors": errors,
            "total": len(memory_ids),
            "succeeded": len(results),
            "failed": len(errors),
        },
        default=str,
    )


# ── Tool 7: delete_all_memories ──────────────────────────────────────


@mcp.tool()
async def delete_all_memories(
    user_id: str | None = None, agent_id: str | None = None
) -> str:
    """Delete ALL memories for a user. Destructive, cannot be undone.

    Disabled by default. Set ENABLE_DELETE_ALL=true to enable.

    Args:
        agent_id: If set, only delete memories for this agent scope.
    """
    try:
        uid = _resolve_user_id(user_id)
        aid = _resolve_agent_id(agent_id)
        collector = get_collector()
        if collector:
            collector.record_operation("delete_all", uid, aid)
        service = _get_service()
        result = await asyncio.to_thread(
            service.delete_all_memories, user_id=uid, agent_id=aid
        )
        return json.dumps(result)
    except ValueError as e:
        return json.dumps({"error": True, "message": str(e)})
    except Exception:
        logger.exception("Error in delete_all_memories")
        return json.dumps(
            {"error": True, "message": "Internal error processing delete_all_memories"}
        )


# ── Tool 8: list_categories ──────────────────────────────────────────


@mcp.tool()
async def list_categories(user_id: str | None = None) -> str:
    """List all available memory categories with descriptions and counts.

    Shows predefined categories and any dynamic project:<name> categories.
    Categories are PREDEFINED — do not invent new ones.
    """
    try:
        uid = _resolve_user_id(user_id)
        collector = get_collector()
        if collector:
            collector.record_operation("list_categories", uid)
        service = _get_service()
        result = await asyncio.to_thread(service.list_categories, user_id=uid)
        lines = []
        for cat in result["categories"]:
            count = cat["count"]
            marker = f" ({count})" if count > 0 else ""
            lines.append(f"  {cat['name']}{marker} — {cat['description']}")
        header = f"Categories ({result['total_memories']} total memories):"
        return header + "\n" + "\n".join(lines)
    except ValueError as e:
        return json.dumps({"error": True, "message": str(e)})
    except Exception:
        logger.exception("Error in list_categories")
        return json.dumps(
            {"error": True, "message": "Internal error processing list_categories"}
        )


# ── Tool 9: save_artifact ────────────────────────────────────────────


@mcp.tool()
async def save_artifact(
    memory_id: str,
    content: str,
    user_id: str | None = None,
    filename: str = "note.md",
    content_type: str = "text/markdown",
) -> str:
    """Attach an artifact to a memory (slow memory tier).

    Use for detailed content too long for fast memory — research reports,
    analysis, logs, notes, code, data, images, PDFs (max 10MB).

    Args:
        memory_id: ID of the parent fast memory.
        content: Text content or base64-encoded binary content.
        filename: Name for the artifact (default: note.md).
        content_type: MIME type (default: text/markdown).
    """
    try:
        uid = _resolve_user_id(user_id)
        owner_id = _get_session_owner_id()
        session_aid = _get_session_agent_id()
        collector = get_collector()
        if collector:
            collector.record_operation("save_artifact", uid)
        service = _get_service()
        result = await asyncio.to_thread(
            service.save_artifact,
            memory_id,
            user_id=uid,
            content=content,
            filename=filename,
            content_type=content_type,
            owner_id=owner_id,
            session_agent_id=session_aid,
        )
        return json.dumps(result, default=str)
    except ValueError as e:
        return json.dumps({"error": True, "message": str(e)})
    except Exception:
        logger.exception("Error in save_artifact")
        return json.dumps(
            {"error": True, "message": "Internal error processing save_artifact"}
        )


# ── Tool 10: get_artifact ────────────────────────────────────────────


@mcp.tool()
async def get_artifact(
    memory_id: str,
    artifact_id: str,
    user_id: str | None = None,
    offset: int = 0,
    limit: int = 5000,
) -> str:
    """Retrieve artifact content.

    Text artifacts support pagination. Binary artifacts >1MB require
    get_artifact_url instead.

    Args:
        memory_id: ID of the parent memory.
        artifact_id: ID of the artifact to retrieve.
        offset: Character offset for text (default 0).
        limit: Max characters for text (default 5000).
    """
    try:
        uid = _resolve_user_id(user_id)
        owner_id = _get_session_owner_id()
        session_aid = _get_session_agent_id()
        collector = get_collector()
        if collector:
            collector.record_operation("get_artifact", uid)
        service = _get_service()
        result = await asyncio.to_thread(
            service.get_artifact,
            memory_id,
            artifact_id,
            user_id=uid,
            owner_id=owner_id,
            session_agent_id=session_aid,
            offset=offset,
            limit=limit,
        )

        # Binary artifacts over 1 MB are too large for MCP inline transport.
        # Return a guidance message instead of the content.
        if (
            not result.get("is_text", True)
            and result.get("total_size", 0) > _MCP_BINARY_INLINE_LIMIT
        ):
            return json.dumps(
                {
                    "error": True,
                    "message": (
                        f"Binary artifact is too large for inline retrieval "
                        f"({result['total_size']} bytes, limit is "
                        f"{_MCP_BINARY_INLINE_LIMIT} bytes). "
                        f"Use get_artifact_url to generate a signed download URL."
                    ),
                    "total_size": result["total_size"],
                    "content_type": result.get(
                        "content_type", "application/octet-stream"
                    ),
                    "use_tool": "get_artifact_url",
                }
            )

        return json.dumps(result, default=str)
    except ValueError as e:
        return json.dumps({"error": True, "message": str(e)})
    except Exception:
        logger.exception("Error in get_artifact")
        return json.dumps(
            {"error": True, "message": "Internal error processing get_artifact"}
        )


# ── Tool 11: list_artifacts ──────────────────────────────────────────


@mcp.tool()
async def list_artifacts(memory_id: str, user_id: str | None = None) -> str:
    """List all artifacts attached to a memory.

    Args:
        memory_id: ID of the parent fast memory.
    """
    try:
        uid = _resolve_user_id(user_id)
        owner_id = _get_session_owner_id()
        session_aid = _get_session_agent_id()
        collector = get_collector()
        if collector:
            collector.record_operation("list_artifacts", uid)
        service = _get_service()
        artifacts = await asyncio.to_thread(
            service.list_artifacts,
            memory_id,
            user_id=uid,
            owner_id=owner_id,
            session_agent_id=session_aid,
        )
        if not artifacts:
            return json.dumps({"artifacts": [], "message": "No artifacts found."})
        return json.dumps({"artifacts": artifacts})
    except ValueError as e:
        return json.dumps({"error": True, "message": str(e)})
    except Exception:
        logger.exception("Error in list_artifacts")
        return json.dumps(
            {"error": True, "message": "Internal error processing list_artifacts"}
        )


# ── Tool 12: delete_artifact ─────────────────────────────────────────


@mcp.tool()
async def delete_artifact(
    memory_id: str, artifact_id: str, user_id: str | None = None
) -> str:
    """Delete an artifact from a memory.

    Args:
        memory_id: ID of the parent fast memory.
        artifact_id: ID of the artifact to delete.
    """
    try:
        uid = _resolve_user_id(user_id)
        owner_id = _get_session_owner_id()
        session_aid = _get_session_agent_id()
        owner_id = _get_session_owner_id()
        session_aid = _get_session_agent_id()
        collector = get_collector()
        if collector:
            collector.record_operation("delete_artifact", uid)
        service = _get_service()
        result = await asyncio.to_thread(
            service.delete_artifact,
            memory_id,
            artifact_id,
            user_id=uid,
            owner_id=owner_id,
            session_agent_id=session_aid,
        )
        return json.dumps(result)
    except ValueError as e:
        return json.dumps({"error": True, "message": str(e)})
    except Exception:
        logger.exception("Error in delete_artifact")
        return json.dumps(
            {"error": True, "message": "Internal error processing delete_artifact"}
        )


# ── Tool 13: get_artifact_url ─────────────────────────────────────────


@mcp.tool()
async def get_artifact_url(
    memory_id: str,
    artifact_id: str,
    user_id: str | None = None,
    ttl: int | None = None,
) -> str:
    """Generate a short-lived signed URL for direct artifact download.

    Use instead of get_artifact for binary artifacts (images, PDFs),
    large artifacts (>1MB), or when a direct browser URL is needed.

    Args:
        memory_id: ID of the parent memory.
        artifact_id: ID of the artifact.
        ttl: URL lifetime in seconds (default ~3600, max ~86400).
    """
    try:
        from mnemory.tokens import generate_download_token

        uid = _resolve_user_id(user_id)
        owner_id = _get_session_owner_id()
        session_aid = _get_session_agent_id()
        collector = get_collector()
        if collector:
            collector.record_operation("get_artifact_url", uid)

        cfg = _get_config().server
        if ttl is not None and ttl < 1:
            return json.dumps(
                {"error": True, "message": "ttl must be at least 1 second"}
            )
        ttl_seconds = min(ttl or cfg.download_token_ttl, cfg.download_token_max_ttl)

        # Verify the artifact exists and user has access before generating a token.
        service = _get_service()
        artifacts = await asyncio.to_thread(
            service.list_artifacts,
            memory_id,
            user_id=uid,
            owner_id=owner_id,
            session_agent_id=session_aid,
        )
        if not any(a["id"] == artifact_id for a in artifacts):
            return json.dumps(
                {"error": True, "message": "Artifact not found or access denied"}
            )

        token = generate_download_token(
            signing_key=_get_signing_key(),
            user_id=uid,
            memory_id=memory_id,
            artifact_id=artifact_id,
            ttl_seconds=ttl_seconds,
            owner_id=owner_id,
            agent_id=session_aid,
        )

        # Build the URL path.
        path = f"/api/memories/{memory_id}/artifacts/{artifact_id}/raw?token={token}"
        base_url = cfg.base_url.rstrip("/") if cfg.base_url else ""
        url = f"{base_url}{path}" if base_url else path

        # Include artifact metadata for convenience.
        meta = next(a for a in artifacts if a["id"] == artifact_id)
        return json.dumps(
            {
                "url": url,
                "expires_in": ttl_seconds,
                "content_type": meta.get("content_type", "application/octet-stream"),
                "filename": meta.get("filename", "artifact"),
                "size": meta.get("size", 0),
            }
        )
    except ValueError as e:
        return json.dumps({"error": True, "message": str(e)})
    except Exception:
        logger.exception("Error in get_artifact_url")
        return json.dumps(
            {"error": True, "message": "Internal error processing get_artifact_url"}
        )


# ── Helpers ───────────────────────────────────────────────────────────


def _format_memories(memories: list[dict]) -> str:
    """Format memory results for LLM consumption.

    Memory text is escaped to prevent markdown header forgery
    when results are included in LLM context.
    """
    if not memories:
        return json.dumps({"results": [], "count": 0})

    formatted = []
    for mem in memories:
        entry: dict = {
            "id": mem.get("id"),
            "memory": escape_memory_headers(mem.get("memory", "")),
            "created_at": mem.get("created_at"),
        }
        # Include score if present (from search)
        if "score" in mem:
            entry["score"] = round(mem["score"], 4)

        metadata = mem.get("metadata") or {}
        if metadata.get("memory_type"):
            entry["type"] = metadata["memory_type"]
        if metadata.get("categories"):
            entry["categories"] = metadata["categories"]
        if metadata.get("importance") and metadata["importance"] != "normal":
            entry["importance"] = metadata["importance"]
        if metadata.get("pinned"):
            entry["pinned"] = True
        if metadata.get("role") and metadata["role"] != "user":
            entry["role"] = metadata["role"]
        if metadata.get("artifacts"):
            entry["has_artifacts"] = True
            entry["artifact_count"] = len(metadata["artifacts"])

        # Event date (when the event occurred)
        if metadata.get("event_date"):
            entry["event_date"] = metadata["event_date"]

        # Labels
        if metadata.get("labels"):
            entry["labels"] = metadata["labels"]

        # TTL state
        if metadata.get("expires_at"):
            entry["expires_at"] = metadata["expires_at"]
        if metadata.get("decayed_at"):
            entry["is_decayed"] = True
            entry["decayed_at"] = metadata["decayed_at"]

        if mem.get("agent_id"):
            entry["agent_id"] = mem["agent_id"]

        formatted.append(entry)

    return json.dumps({"results": formatted, "count": len(formatted)}, default=str)


# ── HTTP Application ──────────────────────────────────────────────────


class APIKeyMiddleware(BaseHTTPMiddleware):
    """JWT/API-key authentication and session identity middleware.

    Authentication:
    1. If Cognis JWT validation is configured, try Bearer JWT validation first
    2. Check token against MCP_API_KEYS (JSON dict: key -> user_id)
    3. Check token against MCP_API_KEY (single key, backward compat)
    4. If neither JWT nor API key auth is configured, auth is disabled
    5. If configured but no match, return 401

    Identity resolution (set as contextvars for tool handlers):
    - user_id: from JWT `sub`, API key mapping, or identity headers
    - agent_id: from JWT `agent_id` or X-Agent-Id header fallback
    """

    async def dispatch(self, request: Request, call_next):
        cfg = _get_config().server
        validator = get_jwt_validator(cfg.jwt_public_key, cfg.jwks_url)
        has_api_key_auth = bool(cfg.api_keys or cfg.api_key)
        has_auth = has_api_key_auth or validator is not None

        # Skip auth for UI static files, root redirect, public OpenAPI
        # metadata, and the exchange token endpoint (it validates the token
        # itself).  /health and /metrics intentionally go through auth on the
        # main port so that Cognis health checks reflect the real auth state.
        # Use MGMT_PORT for unauthenticated health probes (k8s liveness/
        # readiness).
        if (
            request.url.path == "/"
            or request.url.path.startswith("/ui")
            or request.url.path in {"/api/openapi.json", "/api/docs", "/api/redoc"}
            or request.url.path == "/api/auth/exchange"
        ):
            self._set_identity_from_headers(request)
            try:
                return await call_next(request)
            finally:
                _session_user_id.set(None)
                _session_agent_id.set(None)
                _session_timezone.set(None)
                _session_user_bound.set(False)

        if not has_auth:
            # No auth configured — still check identity headers.
            # Download tokens still work (they carry user_id in payload).
            if request.url.path.endswith("/raw"):
                dl_token = request.query_params.get("token", "")
                if dl_token:
                    dl_claims = self._validate_download_token(request, dl_token)
                    if dl_claims:
                        _session_user_id.set(dl_claims["u"])
                        _session_owner_id.set(dl_claims["o"])
                        _session_agent_id.set(dl_claims["g"])
                        _session_user_bound.set(True)
                    # If token is invalid, fall through to no-auth path
                    # (no auth configured, so access is allowed anyway)
            self._set_identity_from_headers(request)
        else:
            # For /raw endpoints, check for download token first.
            # Download tokens bypass API key auth — they are self-contained
            # HMAC-signed tokens scoped to a specific artifact.
            # NOTE: When a ?token= parameter is present, it takes precedence
            # over API key headers. An invalid token returns 401 even if
            # valid API key headers are also present. This is intentional —
            # the presence of a token signals the client's auth intent.
            if request.url.path.endswith("/raw"):
                dl_token = request.query_params.get("token", "")
                if dl_token:
                    dl_claims = self._validate_download_token(request, dl_token)
                    if dl_claims:
                        _session_user_id.set(dl_claims["u"])
                        _session_owner_id.set(dl_claims["o"])
                        _session_agent_id.set(dl_claims["g"])
                        _session_user_bound.set(True)
                        try:
                            return await call_next(request)
                        finally:
                            _session_user_id.set(None)
                            _session_agent_id.set(None)
                            _session_owner_id.set(None)
                            _session_timezone.set(None)
                            _session_user_bound.set(False)
                    else:
                        return JSONResponse(
                            {"error": "Invalid or expired download token"},
                            status_code=401,
                        )

            # Auth precedence (after download tokens):
            # 1. mnemory_exchange_session cookie (exchange SSO — server-side session)
            # 2. Authorization: Bearer JWT (Cognis JWT validation)
            # 3. Authorization: Bearer / X-API-Key (API key auth)
            # 4. cognis_session cookie (direct SSO — JWT in cookie, via _extract_token)
            from mnemory.api.auth import get_exchange_session

            exchange_cookie = request.cookies.get("mnemory_exchange_session", "")
            if exchange_cookie:
                exchange_session = get_exchange_session(exchange_cookie)
                if exchange_session:
                    logger.info(
                        "Auth resolved via exchange session for user=%s",
                        exchange_session.user_id,
                    )
                    _session_user_id.set(exchange_session.user_id)
                    _session_owner_id.set(exchange_session.user_id)
                    _session_user_bound.set(True)
                    if exchange_session.agent_id:
                        _session_agent_id.set(exchange_session.agent_id)
                    header_tz = request.headers.get("x-timezone", "").strip() or None
                    if header_tz:
                        _session_timezone.set(header_tz)
                    try:
                        return await call_next(request)
                    finally:
                        _session_user_id.set(None)
                        _session_agent_id.set(None)
                        _session_owner_id.set(None)
                        _session_timezone.set(None)
                        _session_user_bound.set(False)

            # Extract token from Authorization: Bearer, X-API-Key, or cookie
            token = self._extract_token(request)
            if not token:
                return JSONResponse({"error": "Unauthorized"}, status_code=401)

            header_agent_id = request.headers.get("x-agent-id", "").strip() or None
            header_owner_id = request.headers.get("x-agent-owner", "").strip() or None
            header_tz = request.headers.get("x-timezone", "").strip() or None

            if validator is not None and looks_like_jwt(token):
                try:
                    auth_ctx = validator.validate(
                        token,
                        header_agent_id=header_agent_id,
                        header_owner_id=header_owner_id,
                    )
                except InvalidTokenError:
                    logger.info("JWT auth rejected; falling back to API key auth")
                else:
                    logger.info("Auth resolved via JWT for user=%s", auth_ctx.user_id)
                    _session_auth_kind.set("jwt")
                    _session_trusted_evidence.set(
                        auth_ctx.agent_id is None
                        and auth_ctx.token_type == "user_event"
                        and "mnemory:evidence" in auth_ctx.scopes
                    )
                    _session_user_id.set(auth_ctx.user_id)
                    _session_owner_id.set(auth_ctx.owner_id or auth_ctx.user_id)
                    _session_user_bound.set(True)
                    if auth_ctx.agent_id:
                        _session_agent_id.set(auth_ctx.agent_id)
                    if header_tz:
                        _session_timezone.set(header_tz)
                    try:
                        return await call_next(request)
                    finally:
                        _session_user_id.set(None)
                        _session_agent_id.set(None)
                        _session_owner_id.set(None)
                        _session_timezone.set(None)
                        _session_user_bound.set(False)
                        _session_auth_kind.set(None)
                        _session_trusted_evidence.set(False)

            # Try MCP_API_KEYS first (key -> user_id mapping)
            mapped_user_id = None
            if cfg.api_keys:
                for key, uid in cfg.api_keys.items():
                    if hmac.compare_digest(token, key):
                        # "*" means wildcard — auth OK but no user binding
                        mapped_user_id = uid if uid != "*" else None
                        break
                else:
                    # Token not in api_keys — try legacy single key
                    if cfg.api_key and hmac.compare_digest(token, cfg.api_key):
                        logger.info("Auth resolved via legacy API key")
                        pass  # Auth OK, no user mapping
                    else:
                        return JSONResponse({"error": "Unauthorized"}, status_code=401)
            elif cfg.api_key:
                # Only legacy single key configured
                if not hmac.compare_digest(token, cfg.api_key):
                    return JSONResponse({"error": "Unauthorized"}, status_code=401)
                logger.info("Auth resolved via single API key")
            else:
                return JSONResponse({"error": "Unauthorized"}, status_code=401)

            # Set session identity
            if mapped_user_id:
                logger.info(
                    "Auth resolved via mapped API key for user=%s", mapped_user_id
                )
                _session_user_id.set(mapped_user_id)
                _session_user_bound.set(True)
            else:
                # Fall back to identity headers (X-User-Id, then
                # X-OpenWebUI-User-Email for Open WebUI integration).
                header_uid = (
                    request.headers.get("x-user-id", "").strip()
                    or request.headers.get("x-openwebui-user-email", "").strip()
                )
                if header_uid:
                    _session_user_id.set(header_uid)
                    _session_owner_id.set(header_owner_id or header_uid)

            # Agent ID from header
            header_aid = request.headers.get("x-agent-id", "").strip()
            if header_aid:
                _session_agent_id.set(header_aid)
            elif _session_owner_id.get() is None and _session_user_id.get():
                _session_owner_id.set(_session_user_id.get())
            if header_owner_id:
                _session_owner_id.set(header_owner_id)

            # Timezone from header (overrides DEFAULT_TIMEZONE for this session)
            header_tz = request.headers.get("x-timezone", "").strip()
            if header_tz:
                _session_timezone.set(header_tz)

        try:
            return await call_next(request)
        finally:
            # Reset contextvars after request — ensures no identity leakage
            # between requests, regardless of auth path taken.
            _session_user_id.set(None)
            _session_agent_id.set(None)
            _session_owner_id.set(None)
            _session_timezone.set(None)
            _session_user_bound.set(False)

    def _extract_token(self, request: Request) -> str:
        """Extract API token from request headers or cognis_session cookie.

        Checks in order: Authorization: Bearer header, X-API-Key header,
        cognis_session cookie (cross-service SSO).
        No query parameter fallback — browser-embedded requests (e.g.,
        ``<img src="...">``) must use short-lived download tokens instead.
        """
        auth_header = request.headers.get("authorization", "")
        if auth_header.lower().startswith("bearer "):
            return auth_header[7:]
        header_key = request.headers.get("x-api-key", "")
        if header_key:
            return header_key
        # Fallback to cognis_session cookie (cross-service SSO)
        cookie = request.cookies.get("cognis_session", "")
        if cookie:
            return cookie
        return ""

    def _set_identity_from_headers(self, request: Request) -> None:
        """Set session identity from HTTP headers (no-auth mode).

        User is never hard-bound when resolved from headers — switching
        is always allowed.
        """
        header_uid = (
            request.headers.get("x-user-id", "").strip()
            or request.headers.get("x-openwebui-user-email", "").strip()
        )
        if header_uid:
            _session_user_id.set(header_uid)
            _session_owner_id.set(
                request.headers.get("x-agent-owner", "").strip() or header_uid
            )
        _session_user_bound.set(False)
        header_aid = request.headers.get("x-agent-id", "").strip()
        if header_aid:
            _session_agent_id.set(header_aid)
        elif _session_user_id.get() and _session_owner_id.get() is None:
            _session_owner_id.set(_session_user_id.get())
        header_tz = request.headers.get("x-timezone", "").strip()
        if header_tz:
            _session_timezone.set(header_tz)

    # Regex for extracting memory_id and artifact_id from /raw URLs.
    # Using a compiled regex avoids fragile index-based parsing that could
    # match wrong segments if an ID happened to equal "memories"/"artifacts".
    _RAW_PATH_RE = re.compile(r"/memories/([^/]+)/artifacts/([^/]+)/raw$")

    @staticmethod
    def _validate_download_token(
        request: Request, token: str
    ) -> dict[str, str | None] | None:
        """Validate a download token from a /raw URL query parameter.

        Extracts memory_id and artifact_id from the URL path and
        validates the token's HMAC signature, expiry, and scope.

        Returns:
            The user_id from the token if valid, None otherwise.
        """
        from mnemory.tokens import validate_download_token_claims

        # Extract memory_id and artifact_id from URL path.
        # Expected: /api/memories/{mid}/artifacts/{aid}/raw
        match = APIKeyMiddleware._RAW_PATH_RE.search(request.url.path)
        if not match:
            logger.debug("Cannot extract IDs from path: %s", request.url.path)
            return None
        memory_id, artifact_id = match.group(1), match.group(2)

        return validate_download_token_claims(
            signing_key=_get_signing_key(),
            token=token,
            memory_id=memory_id,
            artifact_id=artifact_id,
        )


class EvidenceAuthMiddleware(BaseHTTPMiddleware):
    """Confine Cognis user-event JWTs to the synchronous evidence route."""

    _PATH = "/api/evidence/remember/v1"

    async def dispatch(self, request: Request, call_next):
        is_target = request.method == "POST" and request.url.path == self._PATH
        authorization = request.headers.get("authorization", "")
        token = (
            authorization[7:].strip()
            if authorization.lower().startswith("bearer ")
            else ""
        )
        cookie_token = request.cookies.get("cognis_session", "").strip()
        if not is_target and not token:
            token = cookie_token
        validator = None
        if token:
            cfg = _get_config().server
            validator = get_jwt_validator(cfg.jwt_public_key, cfg.jwks_url)

        if is_target:
            if any(name.lower().startswith("x-agent-") for name in request.headers):
                return JSONResponse(
                    {"error": "Evidence requests cannot set agent headers"},
                    status_code=401,
                )
            if not token or validator is None:
                return JSONResponse({"error": "Evidence JWT required"}, status_code=401)
            try:
                claims = validator.validate_evidence(token)
            except (InvalidTokenError, ValueError, TypeError):
                return JSONResponse({"error": "Invalid evidence JWT"}, status_code=401)
            request.state.evidence_claims = claims
            request.state.evidence_token = True
            return await call_next(request)

        if token and validator is not None:
            try:
                evidence_intent = validator.is_evidence_intent(token)
            except (InvalidTokenError, ValueError, TypeError):
                evidence_intent = False
            if evidence_intent:
                return JSONResponse(
                    {"error": "Evidence JWT is only valid for the evidence route"},
                    status_code=403,
                )
        return await call_next(request)


async def health_check(request: Request) -> JSONResponse:
    """Health check endpoint for Kubernetes probes.

    Returns 503 during data migrations so that Kubernetes readiness
    probes wait for migration to complete before routing traffic.
    """
    if _migration_running:
        return JSONResponse(
            {
                "status": "migrating",
                "message": "Data migration in progress — server not ready",
            },
            status_code=503,
        )
    cfg = _get_config()
    return JSONResponse(
        {
            "status": "healthy",
            "service": "mnemory",
            "version": __version__,
            "vector_backend": "qdrant" if cfg.vector.is_remote else "qdrant-local",
            "artifact_backend": cfg.artifact.backend,
        }
    )


async def metrics_endpoint(request: Request) -> Response:
    """Prometheus metrics endpoint.

    Collects gauge values from Qdrant (cached) and returns all metrics
    in Prometheus text exposition format.
    """
    from mnemory.metrics import get_collector

    collector = get_collector()
    if collector is None:
        return JSONResponse({"error": "Metrics disabled"}, status_code=404)

    collector.collect_gauges()
    return Response(
        content=collector.generate_metrics(),
        media_type=collector.content_type,
    )


@contextlib.asynccontextmanager
async def lifespan(app):
    """Application lifespan manager."""
    cfg = _get_config()
    auth_mode = (
        "api_keys"
        if cfg.server.api_keys
        else ("api_key" if cfg.server.api_key else "disabled")
    )

    # Remove delete_all_memories tool if disabled (default)
    if not cfg.server.enable_delete_all:
        try:
            mcp.remove_tool("delete_all_memories")
            logger.info(
                "delete_all_memories tool disabled "
                "(set ENABLE_DELETE_ALL=true to enable)"
            )
        except Exception:
            pass  # Tool may not exist if already removed

    # Set up a bounded thread pool for MCP tool execution.
    # MCP tools are async but offload blocking I/O (LLM calls, embeddings,
    # Qdrant queries) to threads via asyncio.to_thread(). This pool limits
    # concurrent blocking operations to prevent thread exhaustion.
    pool_size = cfg.server.thread_pool_size
    executor = ThreadPoolExecutor(max_workers=pool_size, thread_name_prefix="mcp-tool")
    loop = asyncio.get_running_loop()
    loop.set_default_executor(executor)
    logger.info("Thread pool: max_workers=%d", pool_size)

    # Eagerly initialize the service so startup failures are caught early
    service = _get_service()

    # Initialize metrics collector
    from mnemory.api import _session_store
    from mnemory.metrics import init_collector

    init_collector(
        vector_store=service.vector,
        session_store=_session_store,
        config=cfg,
    )

    mgmt_info = (
        f"mgmt_port={cfg.server.mgmt_port}"
        if cfg.server.has_mgmt_port
        else "mgmt_port=disabled"
    )
    logger.info(
        "mnemory starting (vector=%s, artifact=%s, port=%d, auth=%s, "
        "auto_classify=%s, metrics=%s, %s)",
        "qdrant-remote" if cfg.vector.is_remote else "qdrant-local",
        cfg.artifact.backend,
        cfg.server.port,
        auth_mode,
        cfg.memory.auto_classify,
        cfg.server.enable_metrics,
        mgmt_info,
    )

    # Start session cleanup task for REST API
    _session_store.start_cleanup_task()

    # Start fsck check cleanup task (lazy init — only if fsck was used)
    if _fsck_service is not None:
        _fsck_service._store.start_cleanup_task()

    # Start periodic maintenance (auto-fsck) if enabled
    global _maintenance_service
    from mnemory.consolidation import ConsolidationService
    from mnemory.maintenance import MaintenanceService

    # Use a separate LLM client for consolidation if configured.
    # Priority: CONSOLIDATION_LLM_MODEL > FSCK_LLM_MODEL > main LLM_MODEL
    consolidation_model = cfg.memory.consolidation_model or cfg.memory.fsck_model
    consolidation_reasoning = cfg.memory.consolidation_reasoning_effort
    if consolidation_model:
        from mnemory.config import LLMConfig
        from mnemory.llm import LLMClient

        consolidation_llm_config = LLMConfig(
            model=consolidation_model,
            base_url=cfg.llm.base_url,
            api_key=cfg.llm.api_key,
            temperature=cfg.llm.temperature,
            reasoning_effort=consolidation_reasoning,
        )
        consolidation_llm = LLMClient(consolidation_llm_config)
        logger.info(
            "Consolidation using model=%s reasoning=%s",
            consolidation_model,
            consolidation_reasoning or "default",
        )
    else:
        consolidation_llm = service._llm

    consolidation = ConsolidationService(
        config=cfg,
        vector=service.vector,
        llm=consolidation_llm,
        embedding=service.vector.embedding,
        memory_service=service,
        session_summary_store=service._session_summary_store,
        collector=get_collector(),
    )

    maintenance = MaintenanceService(
        config=cfg,
        fsck=_get_fsck_service(),
        collector=get_collector(),
        consolidation=consolidation,
    )
    _maintenance_service = maintenance

    # Wire maintenance service and thread pool to metrics collector
    collector = get_collector()
    if collector is not None:
        collector.set_maintenance_service(maintenance)
        collector.set_thread_pool(executor)

    await maintenance.start()

    async with mcp.session_manager.run():
        yield

    await maintenance.stop()
    _maintenance_service = None
    await _session_store.stop_cleanup_task()
    _session_store.close()
    if _fsck_service is not None:
        await _fsck_service._store.stop_cleanup_task()
    # wait=False: Kubernetes provides a grace period via SIGTERM→SIGKILL.
    # Waiting here could stall pod termination for 30+ seconds if an LLM
    # call is in flight.
    executor.shutdown(wait=False)
    logger.info("mnemory shutting down")


def _build_mgmt_routes() -> list[Route]:
    """Build the list of management routes (/health, optionally /metrics)."""
    routes: list[Route] = [Route("/health", health_check)]
    cfg = _get_config()
    if cfg.server.enable_metrics:
        routes.append(Route("/metrics", metrics_endpoint))
    return routes


def create_app() -> Starlette:
    """Create the main ASGI application.

    /health and /metrics are always served on the main port and go through
    standard API key authentication.
    When MGMT_PORT is set, the same management routes are also served on a
    separate management port without auth.

    The management UI is mounted at /ui when the static directory exists.
    UI static files are exempt from API key auth (handled in middleware).
    """
    from pathlib import Path

    from starlette.staticfiles import StaticFiles

    from mnemory.api import create_api_app

    middleware = [
        Middleware(EvidenceAuthMiddleware),
        Middleware(APIKeyMiddleware),
    ]

    routes: list[Route | Mount] = []

    # Always expose authenticated management routes on the main port.
    routes.extend(_build_mgmt_routes())

    # Mount management UI if static directory exists (graceful degradation)
    ui_static_dir = Path(__file__).parent / "ui" / "static"
    if ui_static_dir.is_dir():
        # Redirect / and /ui (no trailing slash) → /ui/ so index.html is served.
        # Uses 302 (not 301) to preserve query strings for exchange token flow.
        async def _redirect_to_ui(request: Request) -> RedirectResponse:
            qs = str(request.query_params)
            target = f"/ui/?{qs}" if qs else "/ui/"
            return RedirectResponse(target, status_code=302)

        routes.append(Route("/", _redirect_to_ui))
        routes.append(Route("/ui", _redirect_to_ui))
        routes.append(Mount("/ui", app=StaticFiles(directory=ui_static_dir, html=True)))

    routes.extend(
        [
            Mount("/api", app=create_api_app()),
            Mount("/", app=mcp.streamable_http_app()),
        ]
    )

    return Starlette(
        routes=routes,
        middleware=middleware,
        lifespan=lifespan,
    )


def create_mgmt_app() -> Starlette:
    """Create the management ASGI application.

    Lightweight app serving /health and /metrics without authentication.
    Only used when MGMT_PORT is configured. Shares the same service and
    session store singletons as the main app.
    """
    return Starlette(routes=_build_mgmt_routes())


app = create_app()


def main():
    """Run the server with uvicorn.

    When MGMT_PORT is configured, runs two servers in parallel:
    the main MCP/API server and a lightweight management server.
    """
    import asyncio

    import uvicorn

    cfg = _get_config()
    log_level = os.environ.get("LOG_LEVEL", "info").lower()

    if cfg.server.has_mgmt_port:
        # Run main app + management app on separate ports
        async def _serve_dual():
            main_config = uvicorn.Config(
                app,
                host=cfg.server.host,
                port=cfg.server.port,
                log_level=log_level,
            )
            mgmt_config = uvicorn.Config(
                create_mgmt_app(),
                host=cfg.server.mgmt_host,
                port=cfg.server.mgmt_port,
                log_level=log_level,
            )
            main_server = uvicorn.Server(main_config)
            mgmt_server = uvicorn.Server(mgmt_config)
            await asyncio.gather(
                main_server.serve(),
                mgmt_server.serve(),
            )

        asyncio.run(_serve_dual())
    else:
        # Single server — management routes are on the main port (with auth)
        uvicorn.run(
            "mnemory.server:app",
            host=cfg.server.host,
            port=cfg.server.port,
            log_level=log_level,
        )


if __name__ == "__main__":
    main()
