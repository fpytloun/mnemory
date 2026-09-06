"""REST API sub-application for mnemory.

Provides OpenAPI-documented endpoints for memory operations, mounted
at /api/ alongside the MCP server at /mcp. Auto-generates OpenAPI spec
at /api/openapi.json and Swagger UI at /api/docs.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI

from mnemory import __version__
from mnemory.session import SessionStore

# Module-level session store — shared across all API endpoints.
# Initialized with defaults; reconfigured in create_api_app() from config.
_session_store = SessionStore()


def _sanitize_schema_for_openai(schema: dict[str, Any]) -> None:
    """Recursively sanitize an OpenAPI schema for client compatibility.

    OpenAI's function-calling JSON Schema validation rejects:
    - ``"default": null`` — null is not valid under any of the given schemas
    - ``"anyOf": [{"type": "X"}, {"type": "null"}]`` — not supported

    Some OpenAPI tool importers also embed JSON schema snippets into Python
    code and choke on JSON boolean defaults (``true``/``false`` instead of
    Python's ``True``/``False``). Defaults are metadata only; server-side
    defaults remain enforced by FastAPI/Pydantic.

    This mutates *schema* in place: removes null/boolean defaults and
    collapses ``anyOf`` nullable patterns into a plain type (the parameter
    stays optional by not being listed in ``required``).
    """
    if not isinstance(schema, dict):
        return

    # Remove defaults that are known to confuse OpenAPI tool importers.
    if "default" in schema and (
        schema["default"] is None or isinstance(schema["default"], bool)
    ):
        del schema["default"]

    # Collapse anyOf nullable: [{"type": "X"}, {"type": "null"}] -> {"type": "X"}
    # Also handles anyOf with $ref: [{"$ref": "..."}, {"type": "null"}]
    if "anyOf" in schema:
        non_null = [s for s in schema["anyOf"] if s != {"type": "null"}]
        if len(non_null) < len(schema["anyOf"]):
            # Had a null variant — collapse
            del schema["anyOf"]
            if len(non_null) == 1:
                schema.update(non_null[0])
            elif non_null:
                schema["anyOf"] = non_null

    # Recurse through the full nested structure because operation objects can
    # contain schema dictionaries inside lists (parameters, responses, etc.).
    for value in schema.values():
        if isinstance(value, dict):
            _sanitize_schema_for_openai(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    _sanitize_schema_for_openai(item)


def _sanitize_openapi_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Sanitize an entire OpenAPI spec for tool/client compatibility.

    Walks component schemas and path operations, removing problematic
    defaults and collapsing nullable ``anyOf`` patterns.
    """
    # Sanitize component schemas (request/response models)
    for component in spec.get("components", {}).get("schemas", {}).values():
        _sanitize_schema_for_openai(component)

    # Sanitize path-level parameter, request, and response schemas.
    for methods in spec.get("paths", {}).values():
        for operation in methods.values():
            if not isinstance(operation, dict):
                continue
            _sanitize_schema_for_openai(operation)

    return spec


def _add_openapi_security(spec: dict[str, Any]) -> dict[str, Any]:
    """Document accepted API key auth mechanisms in the OpenAPI spec."""
    components = spec.setdefault("components", {})
    security_schemes = components.setdefault("securitySchemes", {})
    security_schemes.setdefault(
        "BearerAuth",
        {
            "type": "http",
            "scheme": "bearer",
            "description": "mnemory API key or Cognis-issued JWT",
        },
    )
    security_schemes.setdefault(
        "ApiKeyAuth",
        {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "mnemory API key",
        },
    )
    security_schemes.setdefault(
        "CognisEvidence",
        {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "Cognis ES256 user-event JWT; evidence route only",
        },
    )

    public_operations = {("/auth/exchange", "post")}
    for path, methods in spec.get("paths", {}).items():
        if not isinstance(methods, dict):
            continue
        for method, operation in methods.items():
            if not isinstance(operation, dict):
                continue
            if (path, method.lower()) in public_operations:
                operation.setdefault("security", [])
            elif path == "/evidence/remember/v1":
                operation["security"] = [{"CognisEvidence": []}]
            else:
                operation.setdefault(
                    "security",
                    [{"BearerAuth": []}, {"ApiKeyAuth": []}],
                )

    return spec


def create_api_app() -> FastAPI:
    """Create the FastAPI sub-application for REST API.

    Routers are imported here (not at module level) to avoid circular
    imports — route handlers reference server.py globals.
    """
    global _session_store

    # Reconfigure session store from config with persistent backend
    from mnemory.server import _get_config
    from mnemory.storage.session import create_session_backend

    cfg = _get_config()
    backend = create_session_backend(
        backend_type=cfg.memory.session_backend,
        session_path=cfg.memory.session_path,
        redis_url=cfg.memory.redis_url,
    )
    _session_store = SessionStore(
        default_ttl=cfg.memory.memory_session_ttl,
        sweep_interval=cfg.memory.memory_session_sweep_interval,
        backend=backend,
    )

    app = FastAPI(
        title="mnemory",
        description="Persistent memory for AI agents — REST API",
        version=__version__,
        docs_url="/docs",
        openapi_url="/openapi.json",
    )

    from mnemory.api.auth import router as auth_router
    from mnemory.api.evidence import router as evidence_router
    from mnemory.api.fsck import router as fsck_router
    from mnemory.api.memories import categories_router
    from mnemory.api.memories import router as memories_router
    from mnemory.api.recall import router as recall_router
    from mnemory.api.remember import router as remember_router
    from mnemory.api.sessions import router as sessions_router
    from mnemory.api.ui import router as ui_router

    app.include_router(auth_router, tags=["auth"])
    app.include_router(evidence_router, tags=["evidence"])
    app.include_router(memories_router, prefix="/memories", tags=["memories"])
    app.include_router(categories_router, prefix="/categories", tags=["categories"])
    app.include_router(recall_router, tags=["intelligence"])
    app.include_router(remember_router, tags=["intelligence"])
    app.include_router(sessions_router, tags=["sessions"])
    app.include_router(fsck_router, prefix="/fsck", tags=["fsck"])
    app.include_router(ui_router, tags=["ui"])

    # Override OpenAPI schema generation to sanitize for OpenAI compatibility.
    # Pydantic v2 / OpenAPI 3.1 emits "anyOf": [{"type": "X"}, {"type": "null"}]
    # with "default": null for Optional fields. OpenAI's function-calling
    # schema validation rejects these patterns.
    _original_openapi = app.openapi

    def _openai_compatible_openapi() -> dict[str, Any]:
        schema = _original_openapi()
        _add_openapi_security(schema)
        return _sanitize_openapi_spec(schema)

    app.openapi = _openai_compatible_openapi  # type: ignore[method-assign]

    return app
