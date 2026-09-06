"""REST and service tests for native revision preconditions."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from mnemory.api.deps import SessionContext
from mnemory.api.memories import (
    _resolve_revision_precondition,
    delete_artifact,
    delete_memory,
    save_artifact,
    update_memory,
)
from mnemory.api.schemas import SaveArtifactRequest, UpdateMemoryRequest
from mnemory.revisions import RevisionConflictError


def _ctx() -> SessionContext:
    return SessionContext(
        user_id="user-1",
        owner_id="owner-1",
        agent_id="agent-1",
    )


@pytest.mark.parametrize("token", ["2", '"2"'])
def test_native_if_match_token(token: str) -> None:
    assert _resolve_revision_precondition(if_match=token, legacy_revision=None) == 2


@pytest.mark.parametrize(
    "token",
    ["", "0", "-1", "*", 'W/"2"', "2, 3", "revision:2", "deadbeef"],
)
def test_malformed_if_match_fails_closed(token: str) -> None:
    with pytest.raises(HTTPException) as exc_info:
        _resolve_revision_precondition(if_match=token, legacy_revision=None)
    assert exc_info.value.status_code == 400


def test_conflicting_if_match_and_legacy_revision_fails_closed() -> None:
    with pytest.raises(HTTPException) as exc_info:
        _resolve_revision_precondition(if_match='"2"', legacy_revision=3)
    assert exc_info.value.status_code == 409


def test_update_conflicting_preconditions_do_not_mutate() -> None:
    service = MagicMock()
    request = UpdateMemoryRequest(content="updated", expected_revision=2)
    with patch("mnemory.api.memories._get_service", return_value=service):
        with pytest.raises(HTTPException) as exc_info:
            update_memory("memory-1", request, if_match="3", ctx=_ctx())
    assert exc_info.value.status_code == 409
    service.update_memory.assert_not_called()


@pytest.mark.parametrize(
    "operation", ["update", "delete", "artifact_save", "artifact_delete"]
)
def test_malformed_endpoint_if_match_does_not_mutate(operation: str) -> None:
    service = MagicMock()
    with patch("mnemory.api.memories._get_service", return_value=service):
        with pytest.raises(HTTPException) as exc_info:
            if operation == "update":
                update_memory(
                    "memory-1",
                    UpdateMemoryRequest(content="updated"),
                    if_match="snapshot-deadbeef",
                    ctx=_ctx(),
                )
            elif operation == "delete":
                delete_memory(
                    "memory-1",
                    expected_revision=None,
                    idempotency_key=None,
                    if_match="snapshot-deadbeef",
                    ctx=_ctx(),
                )
            elif operation == "artifact_save":
                save_artifact(
                    "memory-1",
                    SaveArtifactRequest(content="body"),
                    if_match="snapshot-deadbeef",
                    ctx=_ctx(),
                )
            else:
                delete_artifact(
                    "memory-1",
                    "artifact-1",
                    if_match="snapshot-deadbeef",
                    ctx=_ctx(),
                )
    assert exc_info.value.status_code == 400
    service.update_memory.assert_not_called()
    service.delete_memory.assert_not_called()
    service.save_artifact.assert_not_called()
    service.delete_artifact.assert_not_called()


def test_update_if_match_and_legacy_revision_are_canonical() -> None:
    service = MagicMock()
    service.update_memory.return_value = {
        "status": "updated",
        "revision": 3,
        "lineage_id": "lineage-1",
    }
    request = UpdateMemoryRequest(content="updated", expected_revision=2)
    with patch("mnemory.api.memories._get_service", return_value=service):
        result = update_memory("memory-1", request, if_match='"2"', ctx=_ctx())
    assert result["revision"] == 3
    assert service.update_memory.call_args.kwargs["expected_revision"] == 2
    assert service.update_memory.call_args.kwargs["owner_id"] == "owner-1"
    assert service.update_memory.call_args.kwargs["session_agent_id"] == "agent-1"


def test_delete_if_match_and_legacy_revision_are_canonical() -> None:
    service = MagicMock()
    service.delete_memory.return_value = {
        "status": "retracted",
        "revision": 2,
        "lineage_id": "lineage-1",
    }
    with patch("mnemory.api.memories._get_service", return_value=service):
        result = delete_memory(
            "memory-1",
            expected_revision=2,
            idempotency_key="retry-1",
            if_match="2",
            ctx=_ctx(),
        )
    assert result["revision"] == 2
    assert service.delete_memory.call_args.kwargs["expected_revision"] == 2
    assert service.delete_memory.call_args.kwargs["owner_id"] == "owner-1"
    assert service.delete_memory.call_args.kwargs["session_agent_id"] == "agent-1"


def test_artifact_mutations_forward_if_match_and_scope() -> None:
    service = MagicMock()
    service.save_artifact.return_value = {
        "status": "saved",
        "revision": 2,
        "artifact_revision": 1,
        "lineage_id": "lineage-1",
    }
    service.delete_artifact.return_value = {
        "status": "deleted",
        "revision": 2,
        "artifact_revision": 2,
        "lineage_id": "lineage-1",
    }
    with patch("mnemory.api.memories._get_service", return_value=service):
        saved = save_artifact(
            "memory-1",
            SaveArtifactRequest(content="body"),
            if_match='"2"',
            ctx=_ctx(),
        )
        deleted = delete_artifact(
            "memory-1",
            "artifact-1",
            if_match="2",
            ctx=_ctx(),
        )
    assert saved["artifact_revision"] == 1
    assert deleted["artifact_revision"] == 2
    for call in (
        service.save_artifact.call_args,
        service.delete_artifact.call_args,
    ):
        assert call.kwargs["expected_revision"] == 2
        assert call.kwargs["owner_id"] == "owner-1"
        assert call.kwargs["session_agent_id"] == "agent-1"


@pytest.mark.parametrize(
    "operation", ["update", "delete", "artifact_save", "artifact_delete"]
)
def test_stale_if_match_returns_conflict(operation: str) -> None:
    service = MagicMock()
    error = RevisionConflictError(
        "stale revision",
        lineage_id="lineage-1",
        current_revision_id="revision-2",
        current_revision=2,
    )
    getattr(
        service,
        {
            "update": "update_memory",
            "delete": "delete_memory",
            "artifact_save": "save_artifact",
            "artifact_delete": "delete_artifact",
        }[operation],
    ).side_effect = error
    with patch("mnemory.api.memories._get_service", return_value=service):
        with pytest.raises(HTTPException) as exc_info:
            if operation == "update":
                update_memory(
                    "memory-1",
                    UpdateMemoryRequest(content="updated"),
                    if_match="1",
                    ctx=_ctx(),
                )
            elif operation == "delete":
                delete_memory(
                    "memory-1",
                    expected_revision=None,
                    idempotency_key=None,
                    if_match="1",
                    ctx=_ctx(),
                )
            elif operation == "artifact_save":
                save_artifact(
                    "memory-1",
                    SaveArtifactRequest(content="body"),
                    if_match="1",
                    ctx=_ctx(),
                )
            else:
                delete_artifact(
                    "memory-1",
                    "artifact-1",
                    if_match="1",
                    ctx=_ctx(),
                )
    assert exc_info.value.status_code == 409
