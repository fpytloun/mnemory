"""Explicit, UI-safe projections for revision management endpoints."""

from __future__ import annotations

import base64
import hashlib
import json
from typing import Any

from mnemory.api.schemas import (
    HistoryOperationItem,
    HistoryRevisionItem,
    MemoryHistoryResponse,
    MemoryItem,
    MemoryLinkItem,
    MemoryLinksResponse,
)

_SAFE_METADATA = {
    "memory_type",
    "categories",
    "importance",
    "pinned",
    "role",
    "agent_id",
    "event_date",
    "created_at_utc",
    "ttl_days",
    "expires_at",
    "decayed_at",
    "labels",
    "last_accessed_at",
    "access_count",
    "checked_at",
    "validation_state",
    "validation_count",
    "last_validated_at",
    "memory_layer",
    "revision",
    "revision_state",
    "state_reason",
    "provenance_quality",
}


def safe_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Return only metadata fields approved for the management UI."""
    source = metadata or {}
    return {key: source[key] for key in _SAFE_METADATA if key in source}


def project_memory_item(item: dict[str, Any]) -> MemoryItem:
    """Project one active memory for the management UI browse endpoint."""
    source_metadata = item.get("metadata") or {}
    metadata = safe_metadata(source_metadata)
    if item.get("agent_id"):
        metadata["agent_id"] = item["agent_id"]
    return MemoryItem(
        id=item["id"],
        memory=item.get("memory", ""),
        metadata=metadata,
        has_artifacts=bool(source_metadata.get("artifacts")),
    )


def encode_cursor(offset: str, binding: dict[str, Any]) -> str:
    """Encode an opaque cursor bound to the current identity and filters."""
    canonical = json.dumps(binding, sort_keys=True, separators=(",", ":"))
    payload = {
        "v": 1,
        "o": offset,
        "b": hashlib.sha256(canonical.encode()).hexdigest(),
    }
    raw = json.dumps(payload, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def decode_cursor(cursor: str, binding: dict[str, Any]) -> str:
    """Decode a cursor and reject reuse with another scope or filter set."""
    try:
        padding = "=" * (-len(cursor) % 4)
        payload = json.loads(base64.urlsafe_b64decode(cursor + padding))
        canonical = json.dumps(binding, sort_keys=True, separators=(",", ":"))
        expected = hashlib.sha256(canonical.encode()).hexdigest()
        if payload.get("v") != 1 or payload.get("b") != expected:
            raise ValueError
        offset = payload["o"]
        if not isinstance(offset, str) or not offset:
            raise ValueError
        return offset
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        raise ValueError("The cursor is invalid for this request") from None


def project_history(
    raw: dict[str, Any],
    *,
    revision_end: int | None = None,
    operation_end: int | None = None,
    limit: int = 50,
) -> MemoryHistoryResponse:
    """Project an already-authorized history result."""
    all_revisions = []
    current_id = raw.get("current_revision_id")
    authorized_ids = {item["id"] for item in raw.get("revisions", [])}
    for item in raw.get("revisions", []):
        raw_metadata = item.get("metadata") or {}
        metadata = safe_metadata(raw_metadata)
        if metadata.get("revision_state", "active") == "active":
            current_id = item["id"]
        all_revisions.append(
            HistoryRevisionItem(
                id=item["id"],
                memory=item.get("memory", ""),
                metadata=metadata,
                has_artifacts=bool(raw_metadata.get("artifacts")),
                predecessor_id=(
                    raw_metadata.get("supersedes")
                    if raw_metadata.get("supersedes") in authorized_ids
                    else None
                ),
                successor_id=(
                    raw_metadata.get("revision_successor_id")
                    if raw_metadata.get("revision_successor_id") in authorized_ids
                    else None
                ),
            )
        )
    all_operations = [
        HistoryOperationItem(
            operation_id=item.get("operation_id"),
            operation_kind=item.get("operation_kind", "unknown"),
            status=item.get("status", "unknown"),
            actor_kind=item.get("actor_kind"),
            source_kind=item.get("source_kind"),
            reason=item.get("reason"),
            created_at_utc=item.get("created_at_utc"),
            completed_at_utc=item.get("completed_at_utc"),
        )
        for item in raw.get("operations", [])
    ]
    revision_end = len(all_revisions) if revision_end is None else revision_end
    operation_end = len(all_operations) if operation_end is None else operation_end
    revisions = all_revisions[max(0, revision_end - limit) : revision_end]
    operations = all_operations[max(0, operation_end - limit) : operation_end]
    return MemoryHistoryResponse(
        lineage_id=raw["lineage_id"],
        current_revision_id=current_id,
        revisions=revisions,
        operations=operations,
    )


def project_link_item(item: dict[str, Any] | None) -> MemoryLinkItem | None:
    """Project one authorized related memory."""
    if item is None:
        return None
    metadata = item.get("metadata") or {}
    return MemoryLinkItem(
        id=item["id"],
        memory=item.get("memory", ""),
        revision=metadata.get("revision", 1),
        revision_state=metadata.get("revision_state", "active"),
        memory_type=metadata.get("memory_type"),
        memory_layer=metadata.get("memory_layer", "consolidated"),
        has_artifacts=bool(metadata.get("artifacts")),
    )


def project_links(raw: dict[str, Any]) -> MemoryLinksResponse:
    """Project already-authorized relationship results."""
    return MemoryLinksResponse(
        revision_id=raw["revision_id"],
        lineage_id=raw["lineage_id"],
        supersedes=project_link_item(raw.get("supersedes")),
        successor=project_link_item(raw.get("successor")),
        derived_from=[
            item
            for value in raw.get("derived_from", [])
            if (item := project_link_item(value))
        ],
        derived_outputs=[
            item
            for value in raw.get("derived_outputs", [])
            if (item := project_link_item(value))
        ],
        provenance_quality=raw.get("provenance_quality", "exact"),
    )
