"""Security and contract tests for the revision management UI API."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mnemory.api.ui_projections import (
    decode_cursor,
    encode_cursor,
    project_history,
    project_links,
    project_memory_item,
    safe_metadata,
)
from mnemory.revisions import RevisionService
from mnemory.storage.vector import VectorStore


def test_cursor_is_bound_to_authorization_and_filters():
    binding = {"user_id": "alice", "owner_id": "alice", "memory_type": "fact"}
    cursor = encode_cursor("3d22831a-0b75-4f6e-a6ae-19b144d0653c", binding)

    assert decode_cursor(cursor, binding) == "3d22831a-0b75-4f6e-a6ae-19b144d0653c"
    with pytest.raises(ValueError, match="invalid"):
        decode_cursor(cursor, {**binding, "user_id": "mallory"})
    with pytest.raises(ValueError, match="invalid"):
        decode_cursor(cursor, {**binding, "memory_type": "preference"})


def test_history_projection_allowlists_metadata_and_operation_fields():
    attack = '<img src=x onerror="alert(1)">'
    projected = project_history(
        {
            "lineage_id": "lineage-1",
            "revisions": [
                {
                    "id": "revision-1",
                    "memory": attack,
                    "metadata": {
                        "revision": 1,
                        "revision_state": "active",
                        "labels": {"text": attack},
                        "user_id": "secret-user",
                        "owner_id": "secret-owner",
                        "transition_token": "secret-token",
                        "artifacts": [{"path": "/secret/path"}],
                    },
                }
            ],
            "operations": [
                {
                    "operation_id": "operation-1",
                    "operation_kind": "update",
                    "status": "committed",
                    "actor_kind": "api",
                    "reason": attack,
                    "created_at_utc": "2026-09-05T00:00:00Z",
                    "operation_token": "secret-token",
                    "idempotency_key": "secret-key",
                    "request_fingerprint": "secret-hash",
                    "target_revision_ids": ["inaccessible-id"],
                }
            ],
        }
    )

    payload = projected.model_dump()
    revision = payload["revisions"][0]
    operation = payload["operations"][0]
    assert revision["memory"] == attack
    assert revision["metadata"]["labels"]["text"] == attack
    assert "user_id" not in revision["metadata"]
    assert "owner_id" not in revision["metadata"]
    assert "transition_token" not in revision["metadata"]
    assert "artifacts" not in revision["metadata"]
    assert operation["reason"] == attack
    assert "operation_token" not in operation
    assert "idempotency_key" not in operation
    assert "request_fingerprint" not in operation
    assert "target_revision_ids" not in operation


def test_history_page_bounds_revision_and_operation_reads() -> None:
    service = RevisionService.__new__(RevisionService)
    service._client = MagicMock()
    service._vector = MagicMock()
    service._vector.collection_name = "memories"
    service._vector._point_to_memory.side_effect = lambda point: {
        "id": str(point.id),
        "memory": point.payload["memory"],
        "metadata": point.payload,
    }
    requested = MagicMock()
    requested.id = "revision-3"
    requested.payload = {
        "lineage_id": "lineage-1",
        "user_id": "user-1",
        "owner_id": "owner-1",
        "agent_id": "agent-1",
        "revision": 3,
        "revision_state": "active",
    }
    revisions = []
    for revision in (3, 2, 1):
        point = MagicMock()
        point.id = f"revision-{revision}"
        point.payload = {
            "memory": f"revision {revision}",
            "lineage_id": "lineage-1",
            "user_id": "user-1",
            "owner_id": "owner-1",
            "agent_id": "agent-1",
            "revision": revision,
            "revision_state": "active" if revision == 3 else "superseded",
        }
        revisions.append(point)
    service._read_point = MagicMock(return_value=requested)
    service._materialize_legacy = MagicMock(return_value=requested.payload)
    service._client.scroll.side_effect = [
        (revisions, None),
        ([requested], None),
    ]
    service.operations = MagicMock()
    service.operations.list_for_lineage_page.return_value = {
        "operations": [{"operation_id": "operation-2"}],
        "next_before": "2026-09-04T00:00:00+00:00",
    }

    page = service.history_page(
        "revision-3",
        user_id="user-1",
        owner_id="owner-1",
        session_agent_id="agent-1",
        revision_before=None,
        operation_before=None,
        limit=2,
    )

    first_scroll = service._client.scroll.call_args_list[0].kwargs
    assert first_scroll["limit"] == 3
    assert first_scroll["with_vectors"] is False
    assert [item["id"] for item in page["revisions"]] == [
        "revision-2",
        "revision-3",
    ]
    assert page["next_revision_before"] == 2
    assert page["next_operation_before"] == "2026-09-04T00:00:00+00:00"


def test_safe_metadata_does_not_mutate_input():
    metadata = {"revision": 2, "user_id": "secret"}
    assert safe_metadata(metadata) == {"revision": 2}
    assert metadata == {"revision": 2, "user_id": "secret"}


def test_browse_projection_omits_storage_and_artifact_metadata():
    projected = project_memory_item(
        {
            "id": "revision-1",
            "memory": "<svg onload=alert(1)>",
            "owner_id": "secret-owner",
            "agent_id": "agent",
            "metadata": {
                "revision": 1,
                "revision_state": "active",
                "artifacts": [{"signed_url": "secret-url"}],
                "transition_token": "secret-token",
            },
        }
    ).model_dump()

    assert projected["memory"] == "<svg onload=alert(1)>"
    assert projected["metadata"]["agent_id"] == "agent"
    assert projected["has_artifacts"] is True
    assert "secret-owner" not in str(projected)
    assert "secret-url" not in str(projected)
    assert "secret-token" not in str(projected)


def test_links_projection_contains_only_safe_related_summaries():
    related = {
        "id": "source-1",
        "memory": "<script>alert(1)</script>",
        "metadata": {
            "revision": 4,
            "revision_state": "source",
            "memory_type": "fact",
            "memory_layer": "raw",
            "artifacts": [{"id": "artifact-1", "signed_url": "secret"}],
            "owner_id": "secret-owner",
        },
    }
    projected = project_links(
        {
            "revision_id": "revision-2",
            "lineage_id": "lineage-2",
            "supersedes": None,
            "successor": None,
            "derived_from": [related],
            "derived_outputs": [],
            "provenance_quality": "legacy_batch",
        }
    ).model_dump()

    assert projected["derived_from"] == [
        {
            "id": "source-1",
            "memory": "<script>alert(1)</script>",
            "revision": 4,
            "revision_state": "source",
            "memory_type": "fact",
            "memory_layer": "raw",
            "has_artifacts": True,
        }
    ]
    assert "signed_url" not in str(projected)
    assert "secret-owner" not in str(projected)


def test_browse_active_advances_cursor_without_skipping_authorized_rows():
    store = VectorStore.__new__(VectorStore)
    store._config = MagicMock()
    store._config.vector.collection_name = "memories"
    store._client = MagicMock()

    def point(point_id: str, agent_id: str | None = None):
        value = MagicMock()
        value.id = point_id
        value.payload = {
            "data": point_id,
            "user_id": "alice",
            "owner_id": "alice",
            "agent_id": agent_id,
            "revision_state": "active",
            "revision": 1,
        }
        return value

    store._client.scroll.side_effect = [
        ([point("revision-1", "other-agent")], "revision-1"),
        ([point("revision-2", "agent")], "revision-2"),
        ([point("revision-3", None)], None),
    ]

    page = store.browse_active(
        user_id="alice",
        owner_id="alice",
        session_agent_id="agent",
        limit=2,
    )

    assert [item["id"] for item in page["results"]] == [
        "revision-2",
        "revision-3",
    ]
    assert page["next_offset"] is None
    assert store._client.scroll.call_args_list[1].kwargs["offset"] == "revision-1"


def test_history_projection_paginates_after_redaction():
    raw = {
        "lineage_id": "lineage-1",
        "revisions": [
            {
                "id": f"revision-{index}",
                "memory": f"text-{index}",
                "metadata": {
                    "revision": index,
                    "revision_state": "superseded",
                    "owner_id": f"secret-{index}",
                },
            }
            for index in range(1, 5)
        ],
        "operations": [],
    }

    page = project_history(raw, revision_end=3, limit=1).model_dump()

    assert [item["id"] for item in page["revisions"]] == ["revision-3"]
    assert "secret-3" not in str(page)


def test_management_ui_uses_text_bindings_and_revision_preconditions():
    root = Path(__file__).parents[1]
    html = (root / "mnemory/ui/static/index.html").read_text()
    api_js = (root / "mnemory/ui/static/js/api.js").read_text()
    memories_js = (root / "mnemory/ui/static/js/memories.js").read_text()

    assert "x-html=" not in html
    assert "formatDate(result.metadata?.last_accessed_at)" in html
    assert 'role="tabpanel"' in html
    assert "aria-controls" in html
    assert "@keydown.arrow-right.prevent" in html
    assert "'If-Match': `\"${revision}\"`" in api_js
    assert "history.revisions.unshift(...page.revisions)" in memories_js
    assert "generation === this.loadGeneration" in memories_js
    assert "detail.memory?.metadata?.revision_state" in html
