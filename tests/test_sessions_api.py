"""Tests for session summary listing APIs and store helpers."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from mnemory.api.deps import SessionContext
from mnemory.api.schemas import FailedSessionRetryRequest
from mnemory.api.sessions import (
    _run_with_lease,
    consolidate_session_endpoint,
    delete_session,
    failed_session_retry_eligibility,
    list_sessions,
    retry_failed_sessions,
)
from mnemory.consolidation import RetryInputAssessment, assess_retry_inputs
from mnemory.storage.vector import SessionSummaryStore


def _point(payload: dict | None) -> MagicMock:
    point = MagicMock()
    point.payload = payload
    point.id = (payload or {}).get("session_id", "point")
    return point


def test_blocking_retry_work_renews_its_operation_lease() -> None:
    operations = MagicMock()
    operations.renew_claim.return_value = True

    result = asyncio.run(
        _run_with_lease(
            lambda: time.sleep(1.1) or "complete",
            operations=operations,
            operation_token="operation-token",
            claimant="claimant",
            lease_seconds=2,
        )
    )

    assert result == "complete"
    operations.renew_claim.assert_called()


def test_blocking_retry_work_stops_mutation_after_timeout() -> None:
    operations = MagicMock()
    operations.renew_claim.return_value = True
    write = MagicMock()

    def operation(*, mutation_guard):
        time.sleep(0.15)
        mutation_guard()
        write()

    with pytest.raises(TimeoutError, match="timed out"):
        asyncio.run(
            _run_with_lease(
                operation,
                operations=operations,
                operation_token="operation-token",
                claimant="claimant",
                lease_seconds=0.05,
                mutation_guard_supported=True,
            )
        )
    time.sleep(0.15)

    write.assert_not_called()


def test_blocking_retry_work_uses_one_batch_deadline() -> None:
    operations = MagicMock()
    operations.renew_claim.return_value = True

    async def run_batch() -> None:
        deadline = asyncio.get_running_loop().time() + 0.15
        await _run_with_lease(
            lambda: time.sleep(0.1),
            operations=operations,
            operation_token="operation-token",
            claimant="claimant",
            lease_seconds=1,
            deadline=deadline,
        )
        with pytest.raises(TimeoutError, match="timed out"):
            await _run_with_lease(
                lambda: time.sleep(0.1),
                operations=operations,
                operation_token="operation-token",
                claimant="claimant",
                lease_seconds=1,
                deadline=deadline,
            )

    asyncio.run(run_batch())


def test_blocking_retry_work_does_not_start_without_lease() -> None:
    operations = MagicMock()
    operations.renew_claim.return_value = False
    operation = MagicMock()

    with pytest.raises(RuntimeError, match="before mutation"):
        asyncio.run(
            _run_with_lease(
                operation,
                operations=operations,
                operation_token="operation-token",
                claimant="claimant",
                lease_seconds=1,
            )
        )

    operation.assert_not_called()


def test_terminal_session_allows_operation_journal_completion() -> None:
    operations = MagicMock()
    operations.renew_claim.return_value = True
    session_store = MagicMock()
    session_store.renew_retry_claim.side_effect = [True, True, False]
    session_store.get.return_value = {
        "session_id": "session-1",
        "consolidation_state": "consolidated",
        "retry_operation_token": None,
        "retry_claimant": None,
    }

    def operation(*, mutation_guard):
        mutation_guard()
        mutation_guard()
        return "complete"

    result = asyncio.run(
        _run_with_lease(
            operation,
            operations=operations,
            operation_token="operation-token",
            claimant="claimant",
            lease_seconds=60,
            session_store=session_store,
            lease_session_record={
                "session_id": "session-1",
                "_point_id": "point-1",
            },
            mutation_guard_supported=True,
        )
    )

    assert result == "complete"
    session_store.get.assert_called_once()


class TestListSessionsEndpoint:
    """Tests for GET /api/sessions endpoint behavior."""

    def test_list_sessions_passes_query_params_and_returns_metadata(self):
        """Endpoint should pass through paging/search/sort params."""
        mock_store = MagicMock()
        mock_store.list_for_user.return_value = {
            "sessions": [{"session_id": "ses-1", "summary": "hello"}],
            "total": 1,
            "offset": 20,
            "limit": 10,
            "has_more": False,
            "total_truncated": False,
        }
        mock_service = MagicMock(_session_summary_store=mock_store)
        ctx = SessionContext(user_id="user-1", agent_id=None, timezone=None)

        with patch("mnemory.api.sessions._get_service", return_value=mock_service):
            result = asyncio.run(
                list_sessions(
                    offset=20,
                    limit=10,
                    consolidation_state="idle",
                    q="hello",
                    sort_by="created_at",
                    sort_dir="asc",
                    ctx=ctx,
                )
            )

        assert result["total"] == 1
        assert result["offset"] == 20
        assert result["limit"] == 10
        mock_store.list_for_user.assert_called_once_with(
            "user-1",
            owner_id=None,
            session_agent_id=None,
            offset=20,
            limit=10,
            consolidation_state="idle",
            q="hello",
            sort_by="created_at",
            sort_dir="asc",
            include_metadata=True,
        )

    def test_list_sessions_returns_clean_503_on_store_error(self):
        """Endpoint should wrap store failures in a clean HTTP 503."""
        mock_store = MagicMock()
        mock_store.list_for_user.side_effect = RuntimeError("boom")
        mock_service = MagicMock(_session_summary_store=mock_store)
        ctx = SessionContext(user_id="user-1", agent_id=None, timezone=None)

        with patch("mnemory.api.sessions._get_service", return_value=mock_service):
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(list_sessions(ctx=ctx))

        assert exc_info.value.status_code == 503
        assert exc_info.value.detail == "Failed to list sessions"


class TestDeleteSessionEndpoint:
    def test_delete_memories_passes_authenticated_scope(self):
        store = MagicMock()
        store.get.return_value = {
            "session_id": "ses-1",
            "user_id": "user-1",
            "memory_ids": ["mem-1"],
        }
        service = MagicMock(_session_summary_store=store)
        service.vector.get_by_id_strict.return_value = {
            "metadata": {"memory_layer": "raw"}
        }
        ctx = SessionContext(
            user_id="user-1",
            owner_id="owner-1",
            agent_id="agent-1",
        )

        with patch("mnemory.api.sessions._get_service", return_value=service):
            result = asyncio.run(delete_session("ses-1", delete_memories=True, ctx=ctx))

        assert result == {"deleted_session": True, "deleted_memories": 1}
        service.delete_memory.assert_called_once_with(
            "mem-1",
            user_id="user-1",
            owner_id="owner-1",
            session_agent_id="agent-1",
        )
        store.delete.assert_called_once()

    def test_cleanup_failure_retains_session_for_retry(self):
        store = MagicMock()
        store.get.return_value = {
            "session_id": "ses-1",
            "user_id": "user-1",
            "memory_ids": ["mem-1", "mem-2"],
        }
        service = MagicMock(_session_summary_store=store)
        service.vector.get_by_id_strict.return_value = {
            "metadata": {"memory_layer": "raw"}
        }
        service.delete_memory.side_effect = [None, RuntimeError("failed")]
        ctx = SessionContext(user_id="user-1", owner_id="owner-1")

        with patch("mnemory.api.sessions._get_service", return_value=service):
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(delete_session("ses-1", delete_memories=True, ctx=ctx))

        assert exc_info.value.status_code == 409
        assert exc_info.value.detail["deleted_memories"] == 1
        assert exc_info.value.detail["failed_memory_ids"] == ["mem-2"]
        assert exc_info.value.detail["session_retained"] is True
        store.delete.assert_not_called()

    def test_retry_owned_session_cannot_be_deleted(self):
        store = MagicMock()
        store.get.return_value = {
            "session_id": "ses-1",
            "user_id": "user-1",
            "consolidation_state": "idle",
            "retry_operation_token": "retry-token",
        }
        service = MagicMock(_session_summary_store=store)
        ctx = SessionContext(user_id="user-1", owner_id="owner-1")

        with (
            patch("mnemory.api.sessions._get_service", return_value=service),
            pytest.raises(HTTPException) as exc_info,
        ):
            asyncio.run(delete_session("ses-1", delete_memories=False, ctx=ctx))

        assert exc_info.value.status_code == 409
        store.delete.assert_not_called()


def test_retry_owned_session_cannot_start_manual_consolidation() -> None:
    store = MagicMock()
    store.get.return_value = {
        "session_id": "ses-1",
        "user_id": "user-1",
        "consolidation_state": "idle",
        "retry_operation_token": "retry-token",
    }
    service = MagicMock(_session_summary_store=store)
    ctx = SessionContext(user_id="user-1", owner_id="owner-1")

    with (
        patch("mnemory.api.sessions._get_service", return_value=service),
        pytest.raises(HTTPException) as exc_info,
    ):
        asyncio.run(consolidate_session_endpoint("ses-1", ctx=ctx))

    assert exc_info.value.status_code == 409
    store.update_consolidation_state.assert_not_called()


class TestFailedSessionRetryControl:
    def test_dry_run_is_mutation_free(self):
        store = MagicMock()
        store.get.return_value = {
            "session_id": "session-1",
            "user_id": "user-1",
            "owner_id": "owner-1",
            "consolidation_state": "failed",
            "session_revision": 2,
            "memory_ids": ["memory-1"],
        }
        service = MagicMock(_session_summary_store=store)
        service._config.memory.legacy_failed_retry_max_batch = 10
        service._config.memory.legacy_failed_retry_max_raw_memories = 100
        ctx = SessionContext(user_id="user-1", owner_id="owner-1")

        with patch("mnemory.api.sessions._get_service", return_value=service):
            result = asyncio.run(
                retry_failed_sessions(
                    FailedSessionRetryRequest(session_ids=["session-1"], dry_run=True),
                    ctx,
                )
            )

        assert result["dry_run"] is True
        service.revisions.operations.write.assert_not_called()
        store.update_consolidation_state.assert_not_called()

    def test_execute_is_disabled_by_default(self):
        store = MagicMock()
        store.get.return_value = {
            "session_id": "session-1",
            "consolidation_state": "failed",
            "session_revision": 2,
            "memory_ids": ["memory-1"],
        }
        service = MagicMock(_session_summary_store=store)
        service._config.memory.legacy_failed_retry_max_batch = 10
        service._config.memory.legacy_failed_retry_max_raw_memories = 100
        service._config.memory.legacy_failed_retry_enabled = False
        ctx = SessionContext(user_id="user-1", owner_id="owner-1")

        with patch("mnemory.api.sessions._get_service", return_value=service):
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(
                    retry_failed_sessions(
                        FailedSessionRetryRequest(
                            session_ids=["session-1"],
                            dry_run=False,
                            idempotency_key="retry-1",
                        ),
                        ctx,
                    )
                )

        assert exc_info.value.status_code == 403
        service.revisions.operations.write.assert_not_called()

    def test_eligibility_orders_smallest_sessions_first(self):
        store = MagicMock()
        store.list_for_user.side_effect = [
            {
                "sessions": [
                    {
                        "session_id": "large",
                        "consolidation_state": "failed",
                        "session_revision": 1,
                        "memory_ids": ["1", "2", "3"],
                    },
                    {
                        "session_id": "small",
                        "consolidation_state": "failed",
                        "session_revision": 1,
                        "memory_ids": ["1"],
                    },
                ]
            }
        ]
        service = MagicMock(_session_summary_store=store)
        service._config.memory.legacy_failed_retry_max_raw_memories = 100
        ctx = SessionContext(user_id="user-1", owner_id="owner-1")

        with patch("mnemory.api.sessions._get_service", return_value=service):
            result = asyncio.run(failed_session_retry_eligibility(50, ctx))

        assert [item["session_id"] for item in result["candidates"]] == [
            "small",
            "large",
        ]
        service.revisions.operations.write.assert_not_called()

    @pytest.mark.parametrize(
        ("memory", "reason"),
        [
            (None, "linked_memory_missing"),
            (
                {
                    "id": "memory-1",
                    "user_id": "other-user",
                    "owner_id": "owner-1",
                    "agent_id": "agent-1",
                    "metadata": {"memory_layer": "raw", "revision_state": "active"},
                },
                "linked_memory_user_mismatch",
            ),
            (
                {
                    "id": "memory-1",
                    "user_id": "user-1",
                    "owner_id": "other-owner",
                    "agent_id": "agent-1",
                    "metadata": {"memory_layer": "raw", "revision_state": "active"},
                },
                "linked_memory_owner_mismatch",
            ),
            (
                {
                    "id": "memory-1",
                    "user_id": "user-1",
                    "owner_id": "owner-1",
                    "agent_id": "other-agent",
                    "metadata": {"memory_layer": "raw", "revision_state": "active"},
                },
                "linked_memory_agent_mismatch",
            ),
            (
                {
                    "id": "memory-1",
                    "user_id": "user-1",
                    "owner_id": "owner-1",
                    "agent_id": "agent-1",
                    "metadata": {
                        "memory_layer": "consolidated",
                        "revision_state": "active",
                    },
                },
                "linked_memory_not_raw",
            ),
            (
                {
                    "id": "memory-1",
                    "user_id": "user-1",
                    "owner_id": "owner-1",
                    "agent_id": "agent-1",
                    "metadata": {
                        "memory_layer": "raw",
                        "revision_state": "superseded",
                    },
                },
                "linked_memory_not_active",
            ),
            (
                {
                    "id": "memory-1",
                    "user_id": "user-1",
                    "owner_id": "owner-1",
                    "agent_id": "agent-1",
                    "metadata": {
                        "memory_layer": "raw",
                        "revision_state": "active",
                        "superseded_by": "memory-2",
                    },
                },
                "linked_memory_superseded",
            ),
        ],
    )
    def test_retry_input_assessment_has_fixed_reason(
        self,
        memory: dict | None,
        reason: str,
    ) -> None:
        vector = MagicMock()
        vector.get_by_ids_strict.return_value = [] if memory is None else [memory]

        result = assess_retry_inputs(
            vector,
            {"memory_ids": ["memory-1"]},
            user_id="user-1",
            owner_id="owner-1",
            agent_id="agent-1",
        )

        assert reason in result.reasons

    def test_invalid_linked_input_creates_no_retry_operation(self) -> None:
        session = {
            "session_id": "session-1",
            "user_id": "user-1",
            "owner_id": "owner-1",
            "agent_id": None,
            "consolidation_state": "failed",
            "legacy_failure": True,
            "session_revision": 2,
            "memory_ids": ["missing-memory"],
        }
        store = MagicMock()
        store.get.return_value = session
        service = MagicMock(_session_summary_store=store)
        service.vector.get_by_ids_strict.return_value = []
        service._config.memory.legacy_failed_retry_max_batch = 10
        service._config.memory.legacy_failed_retry_max_raw_memories = 100
        service._config.memory.legacy_failed_retry_enabled = True
        service._config.memory.legacy_failed_retry_timeout_seconds = 300
        operations = service.revisions.operations
        operations.get.return_value = None
        ctx = SessionContext(user_id="user-1", owner_id="owner-1")

        with (
            patch("mnemory.api.sessions._get_service", return_value=service),
            patch("mnemory.server._maintenance_service", MagicMock()),
            pytest.raises(HTTPException) as exc_info,
        ):
            asyncio.run(
                retry_failed_sessions(
                    FailedSessionRetryRequest(
                        session_ids=["session-1"],
                        dry_run=False,
                        idempotency_key="retry-1",
                    ),
                    ctx,
                )
            )

        assert exc_info.value.status_code == 409
        operations.write.assert_not_called()
        operations.claim.assert_not_called()
        store.reset_failed_for_retry.assert_not_called()

    def test_race_before_journal_creates_no_operation_or_claim(self) -> None:
        session = {
            "session_id": "session-1",
            "user_id": "user-1",
            "owner_id": "owner-1",
            "agent_id": None,
            "consolidation_state": "failed",
            "legacy_failure": True,
            "session_revision": 2,
            "memory_ids": ["memory-1"],
        }
        store = MagicMock()
        store.get.side_effect = [session, session]
        service = MagicMock(_session_summary_store=store)
        service._config.memory.legacy_failed_retry_max_batch = 10
        service._config.memory.legacy_failed_retry_max_raw_memories = 100
        service._config.memory.legacy_failed_retry_enabled = True
        service._config.memory.legacy_failed_retry_timeout_seconds = 300
        operations = service.revisions.operations
        operations.get.return_value = None
        valid = RetryInputAssessment("input-1", (), 1)
        changed = RetryInputAssessment("input-2", ("linked_memory_missing",), 1)
        ctx = SessionContext(user_id="user-1", owner_id="owner-1")

        with (
            patch("mnemory.api.sessions._get_service", return_value=service),
            patch("mnemory.server._maintenance_service", MagicMock()),
            patch(
                "mnemory.api.sessions._assess_retry_inputs",
                side_effect=[valid, changed],
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            asyncio.run(
                retry_failed_sessions(
                    FailedSessionRetryRequest(
                        session_ids=["session-1"],
                        dry_run=False,
                        idempotency_key="retry-1",
                    ),
                    ctx,
                )
            )

        assert exc_info.value.status_code == 409
        assert exc_info.value.detail == "Retry preflight changed"
        operations.write.assert_not_called()
        operations.claim.assert_not_called()
        store.reset_failed_for_retry.assert_not_called()

    def test_committed_base_format_retry_token_replays_after_upgrade(self) -> None:
        from mnemory.revisions import canonical_fingerprint

        session = {
            "session_id": "session-1",
            "consolidation_state": "failed",
            "legacy_failure": True,
            "session_revision": 2,
            "memory_ids": ["memory-1"],
        }
        store = MagicMock()
        store.get.return_value = session
        service = MagicMock(_session_summary_store=store)
        service._config.memory.legacy_failed_retry_max_batch = 10
        service._config.memory.legacy_failed_retry_max_raw_memories = 100
        service._config.memory.legacy_failed_retry_enabled = True
        service._config.memory.legacy_failed_retry_timeout_seconds = 300
        token = canonical_fingerprint(
            [
                "user-1",
                "owner-1",
                None,
                "session_retry_batch",
                ["session-1"],
                "retry-1",
            ]
        )
        operations = service.revisions.operations
        operations.get.return_value = {
            "operation_id": token,
            "status": "committed",
            "user_id": "user-1",
            "owner_id": "owner-1",
            "agent_id": None,
            "result": {
                "dry_run": False,
                "succeeded": 1,
                "failed": 0,
                "stopped": False,
                "stop_reason": None,
                "replayed": False,
            },
        }
        context = SessionContext(user_id="user-1", owner_id="owner-1")

        with (
            patch("mnemory.api.sessions._get_service", return_value=service),
            patch("mnemory.server._maintenance_service", MagicMock()),
            patch(
                "mnemory.api.sessions._assess_retry_inputs",
                return_value=RetryInputAssessment("input-1", (), 1),
            ),
        ):
            result = asyncio.run(
                retry_failed_sessions(
                    FailedSessionRetryRequest(
                        session_ids=["session-1"],
                        dry_run=False,
                        idempotency_key="retry-1",
                    ),
                    context,
                )
            )

        assert result["replayed"] is True
        operations.get.assert_called_once_with(token)
        operations.write.assert_not_called()
        store.reset_failed_for_retry.assert_not_called()

    def test_expired_batch_does_not_take_over_active_previous_claimant(self):
        store = MagicMock()
        consolidating = {
            "session_id": "session-1",
            "user_id": "user-1",
            "owner_id": "owner-1",
            "agent_id": None,
            "consolidation_state": "consolidating",
            "session_revision": 3,
            "retry_operation_token": "stored-token",
            "retry_claimant": "previous-worker",
            "retry_lease_expires_at": (
                datetime.now(timezone.utc) + timedelta(minutes=5)
            ).isoformat(),
            "memory_ids": ["memory-1"],
        }
        store.get.return_value = consolidating
        operations = MagicMock()
        operations.get.return_value = {
            "status": "applying",
            "user_id": "user-1",
            "owner_id": "owner-1",
            "agent_id": None,
            "session_checkpoints": [
                {
                    "session_id": "session-1",
                    "expected_revision": 2,
                    "status": "pending",
                }
            ],
        }
        operations.claim.return_value = True
        service = MagicMock(_session_summary_store=store)
        service.revisions.operations = operations
        service._config.memory.legacy_failed_retry_max_batch = 10
        service._config.memory.legacy_failed_retry_max_raw_memories = 100
        service._config.memory.legacy_failed_retry_enabled = True
        service._config.memory.legacy_failed_retry_timeout_seconds = 300
        service._config.memory.legacy_failed_retry_stop_min_attempts = 5
        service._config.memory.legacy_failed_retry_stop_failure_ratio = 0.2
        consolidation = MagicMock()
        consolidation.recover_session.return_value = True
        maintenance = MagicMock(_consolidation=consolidation)
        ctx = SessionContext(user_id="user-1", owner_id="owner-1")

        with (
            patch("mnemory.api.sessions._get_service", return_value=service),
            patch("mnemory.server._maintenance_service", maintenance),
            patch(
                "mnemory.revisions.canonical_fingerprint",
                return_value="stored-token",
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(
                    retry_failed_sessions(
                        FailedSessionRetryRequest(
                            session_ids=["session-1"],
                            dry_run=False,
                            idempotency_key="retry-1",
                        ),
                        ctx,
                    )
                )

        assert exc_info.value.status_code == 409
        operations.claim.assert_not_called()
        consolidation.recover_session.assert_not_called()
        consolidation.consolidate_session.assert_not_called()

    def test_expired_batch_transfers_and_recovers_orphaned_session(self):
        store = MagicMock()
        expired = {
            "session_id": "session-1",
            "user_id": "user-1",
            "owner_id": "owner-1",
            "agent_id": None,
            "_point_id": "point-1",
            "consolidation_state": "consolidating",
            "session_revision": 3,
            "retry_operation_token": "stored-token",
            "retry_claimant": "previous-worker",
            "retry_lease_expires_at": (
                datetime.now(timezone.utc) - timedelta(seconds=1)
            ).isoformat(),
            "memory_ids": ["memory-1"],
        }
        transferred = {
            **expired,
            "retry_claimant": "new-worker",
            "retry_lease_expires_at": (
                datetime.now(timezone.utc) + timedelta(minutes=5)
            ).isoformat(),
        }
        consolidated = {
            **transferred,
            "consolidation_state": "consolidated",
            "session_revision": 4,
            "retry_operation_token": None,
            "retry_claimant": None,
        }
        store.get.side_effect = [expired, expired, expired, consolidated]
        store.transfer_retry_claim.return_value = transferred
        store.renew_retry_claim.return_value = True
        operations = MagicMock()
        operations.get.return_value = {
            "status": "applying",
            "user_id": "user-1",
            "owner_id": "owner-1",
            "agent_id": None,
            "session_checkpoints": [
                {
                    "session_id": "session-1",
                    "expected_revision": 2,
                    "status": "pending",
                }
            ],
        }
        operations.claim.return_value = True
        operations.write_claimed.return_value = True
        operations.renew_claim.return_value = True
        service = MagicMock(_session_summary_store=store)
        service.revisions.operations = operations
        service._config.memory.legacy_failed_retry_max_batch = 10
        service._config.memory.legacy_failed_retry_max_raw_memories = 100
        service._config.memory.legacy_failed_retry_enabled = True
        service._config.memory.legacy_failed_retry_timeout_seconds = 300
        service._config.memory.legacy_failed_retry_stop_min_attempts = 5
        service._config.memory.legacy_failed_retry_stop_failure_ratio = 0.2
        consolidation = MagicMock()
        consolidation.recover_session.return_value = True
        maintenance = MagicMock(_consolidation=consolidation)
        ctx = SessionContext(user_id="user-1", owner_id="owner-1")

        with (
            patch("mnemory.api.sessions._get_service", return_value=service),
            patch("mnemory.server._maintenance_service", maintenance),
            patch(
                "mnemory.revisions.canonical_fingerprint",
                return_value="stored-token",
            ),
            patch(
                "mnemory.api.sessions._assess_retry_inputs",
                return_value=RetryInputAssessment("input-1", (), 1),
            ),
            patch("mnemory.api.sessions.uuid.uuid4", return_value="new-worker"),
        ):
            result = asyncio.run(
                retry_failed_sessions(
                    FailedSessionRetryRequest(
                        session_ids=["session-1"],
                        dry_run=False,
                        idempotency_key="retry-1",
                    ),
                    ctx,
                )
            )

        assert result["succeeded"] == 1
        store.transfer_retry_claim.assert_called_once()
        assert consolidation.recover_session.call_args.args == (transferred,)
        assert consolidation.recover_session.call_args.kwargs["mutation_guard"]

    def test_partial_cleanup_retry_skips_already_retracted_memory(self):
        store = MagicMock()
        store.get.return_value = {
            "session_id": "ses-1",
            "user_id": "user-1",
            "memory_ids": ["mem-1", "mem-2"],
        }
        service = MagicMock(_session_summary_store=store)
        service.vector.get_by_id_strict.side_effect = [
            {"metadata": {"memory_layer": "raw", "revision_state": "active"}},
            {"metadata": {"memory_layer": "raw", "revision_state": "active"}},
            {"metadata": {"memory_layer": "raw", "revision_state": "retracted"}},
            {"metadata": {"memory_layer": "raw", "revision_state": "active"}},
        ]
        service.delete_memory.side_effect = [None, RuntimeError("failed"), None]
        ctx = SessionContext(user_id="user-1", owner_id="owner-1")

        with patch("mnemory.api.sessions._get_service", return_value=service):
            with pytest.raises(HTTPException):
                asyncio.run(delete_session("ses-1", delete_memories=True, ctx=ctx))
            result = asyncio.run(delete_session("ses-1", delete_memories=True, ctx=ctx))

        assert result == {"deleted_session": True, "deleted_memories": 1}
        assert service.delete_memory.call_count == 3
        store.delete.assert_called_once()

    def test_transient_lookup_failure_retains_session(self):
        store = MagicMock()
        store.get.return_value = {
            "session_id": "ses-1",
            "user_id": "user-1",
            "memory_ids": ["mem-1"],
        }
        service = MagicMock(_session_summary_store=store)
        service.vector.get_by_id_strict.side_effect = TimeoutError("qdrant timeout")
        ctx = SessionContext(
            user_id="user-1",
            owner_id="owner-1",
            agent_id="agent-1",
        )

        with patch("mnemory.api.sessions._get_service", return_value=service):
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(delete_session("ses-1", delete_memories=True, ctx=ctx))

        assert exc_info.value.status_code == 409
        assert exc_info.value.detail["failed_memory_ids"] == ["mem-1"]
        assert exc_info.value.detail["session_retained"] is True
        service.delete_memory.assert_not_called()
        store.delete.assert_not_called()


class TestSessionSummaryStoreListForUser:
    """Tests for SessionSummaryStore.list_for_user pagination helpers."""

    def test_list_for_user_filters_sorts_and_paginates(self):
        """Store should dedupe, filter, sort, and paginate after full scroll."""
        store = SessionSummaryStore.__new__(SessionSummaryStore)
        store._client = MagicMock()
        store._client.scroll.side_effect = [
            (
                [
                    _point(
                        {
                            "session_id": "ses-3",
                            "user_id": "user-1",
                            "summary": "Zebra project summary",
                            "created_at": "2026-03-03T00:00:00+00:00",
                            "updated_at": "2026-03-04T00:00:00+00:00",
                        }
                    ),
                    _point(
                        {
                            "session_id": "ses-1",
                            "user_id": "user-1",
                            "summary": "Alpha project summary",
                            "created_at": "2026-03-01T00:00:00+00:00",
                            "updated_at": "2026-03-02T00:00:00+00:00",
                        }
                    ),
                    _point({"session_id": "broken", "user_id": "user-1"}),
                ],
                "offset-2",
            ),
            (
                [
                    _point(
                        {
                            "session_id": "ses-2",
                            "user_id": "user-1",
                            "summary": "Beta project summary",
                            "created_at": "2026-03-02T00:00:00+00:00",
                            "updated_at": "2026-03-03T00:00:00+00:00",
                        }
                    ),
                    _point(
                        {
                            "session_id": "ses-4",
                            "user_id": "user-1",
                            "summary": "Other topic",
                            "created_at": "2026-03-05T00:00:00+00:00",
                            "updated_at": "2026-03-05T00:00:00+00:00",
                        }
                    ),
                    _point(
                        {
                            "session_id": "ses-1",
                            "user_id": "user-1",
                            "summary": "Alpha project summary duplicate",
                            "created_at": "2026-03-01T00:00:00+00:00",
                            "updated_at": "2026-03-02T00:00:00+00:00",
                        }
                    ),
                ],
                None,
            ),
        ]

        result = store.list_for_user(
            "user-1",
            q="project",
            sort_by="created_at",
            sort_dir="asc",
            offset=1,
            limit=2,
            include_metadata=True,
        )

        assert result["total"] == 3
        assert result["offset"] == 1
        assert result["limit"] == 2
        assert result["has_more"] is False
        assert result["total_truncated"] is False
        assert [s["session_id"] for s in result["sessions"]] == ["ses-2", "ses-3"]

    def test_list_for_user_scroll_failure_raises_runtime_error(self):
        """Store should raise a clean RuntimeError on scroll failure."""
        store = SessionSummaryStore.__new__(SessionSummaryStore)
        store._client = MagicMock()
        store._client.scroll.side_effect = [
            (
                [
                    _point(
                        {
                            "session_id": "ses-1",
                            "user_id": "user-1",
                            "summary": "Alpha project summary",
                            "created_at": "2026-03-01T00:00:00+00:00",
                            "updated_at": "2026-03-02T00:00:00+00:00",
                        }
                    )
                ],
                "offset-2",
            ),
            Exception("qdrant unavailable"),
        ]

        with pytest.raises(RuntimeError, match="Failed to list session summaries"):
            store.list_for_user("user-1", include_metadata=True)
