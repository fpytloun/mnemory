"""Focused operation-journal tests for fsck apply and re-evaluation."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from qdrant_client import QdrantClient

from mnemory.fsck import (
    FsckAction,
    FsckAffectedMemory,
    FsckCheck,
    FsckIssue,
    FsckService,
    FsckStore,
)
from mnemory.revisions import RevisionOperationStore, canonical_fingerprint


def _service() -> tuple[FsckService, MagicMock]:
    config = MagicMock()
    config.memory.fsck_recovery_lease_seconds = 300
    config.memory.fsck_recovery_max_attempts = 3
    vector = MagicMock()
    vector.get_by_id.return_value = {
        "id": "memory-1",
        "user_id": "user-1",
        "owner_id": "owner-1",
        "metadata": {
            "lineage_id": "lineage-1",
            "revision": 1,
            "revision_state": "active",
        },
    }
    vector.get_by_id_strict.return_value = vector.get_by_id.return_value
    store = FsckStore()
    service = FsckService(config=config, vector=vector, llm=MagicMock(), store=store)
    operations = MagicMock()
    operations.get.return_value = None
    operations.claim.return_value = True
    memory_service = MagicMock()
    memory_service.revisions.operations = operations
    service._memory_service = memory_service
    return service, operations


def _completed_check(service: FsckService, check_id: str = "check-1") -> FsckCheck:
    check = service._store.create(user_id="user-1", owner_id="owner-1", agent_id=None)
    check.check_id = check_id
    issue = FsckIssue(
        issue_id="issue-1",
        type="duplicate",
        severity="medium",
        reasoning="duplicate",
        affected_memories=[
            FsckAffectedMemory(
                id="memory-1",
                content="",
                metadata={"lineage_id": "lineage-1"},
            )
        ],
        actions=[
            FsckAction(
                action="update",
                memory_id="memory-1",
                new_content="replacement",
            )
        ],
        confidence=1.0,
    )
    check.status = "completed"
    check.issues = [issue]
    service._store._checks[check_id] = check
    return check


def test_apply_exception_terminalizes_operation_as_failed() -> None:
    service, operations = _service()
    check = _completed_check(service)
    service._apply_issue = MagicMock(side_effect=RuntimeError("sensitive detail"))

    result = service.apply_check(check.check_id)

    assert result["failed"] == 1
    terminal = operations.write_claimed.call_args_list[-1].args[2]
    assert terminal["status"] == "failed"
    assert terminal["error_class"] == "RuntimeError"
    assert "sensitive detail" not in str(terminal)


def test_fsck_journal_uses_check_owner_when_source_is_missing() -> None:
    service, operations = _service()
    check = _completed_check(service)
    service._vector.get_by_id.return_value = None
    service._vector.get_by_id_strict.return_value = None
    service._apply_issue = MagicMock(return_value=(0, 1))

    service.apply_check(check.check_id)

    planned = operations.write.call_args_list[0].args[1]
    assert planned["owner_id"] == "owner-1"


def test_apply_success_persists_action_checkpoint() -> None:
    service, operations = _service()
    check = _completed_check(service)

    def apply(*args, before_action, on_action, **kwargs):
        before_action(0)
        on_action(0, "committed", "revision-operation-1")
        return 1, 0

    service._apply_issue = MagicMock(side_effect=apply)

    result = service.apply_check(check.check_id)

    assert result["applied"] == 1
    terminal = operations.write_claimed.call_args_list[-1].args[2]
    assert terminal["status"] == "committed"
    assert terminal["action_checkpoints"][0]["status"] == "committed"
    assert terminal["action_checkpoints"][0]["expected_revision"] == 1


def test_retry_runs_only_pending_actions() -> None:
    service, operations = _service()
    check = _completed_check(service)
    check.issues[0].affected_memories.append(
        FsckAffectedMemory(
            id="memory-2",
            content="",
            metadata={
                "lineage_id": "lineage-2",
                "revision": 1,
                "revision_state": "active",
            },
        )
    )
    check.issues[0].actions.append(
        FsckAction(
            action="update",
            memory_id="memory-2",
            new_content="second replacement",
        )
    )

    def get_memory(memory_id):
        return {
            "id": memory_id,
            "user_id": "user-1",
            "owner_id": "owner-1",
            "metadata": {
                "lineage_id": memory_id,
                "revision": 1,
                "revision_state": "active",
            },
        }

    service._vector.get_by_id.side_effect = get_memory
    service._vector.get_by_id_strict.side_effect = get_memory
    operations.get.return_value = {
        "status": "failed",
        "recovery_attempt_count": 1,
        "action_checkpoints": [
            {
                "action_id": "action-1",
                "action_index": 0,
                "action_kind": "update",
                "expected_revision_id": "memory-1",
                "expected_revision": 1,
                "status": "committed",
            },
            {
                "action_id": "action-2",
                "action_index": 1,
                "action_kind": "update",
                "expected_revision_id": "memory-2",
                "expected_revision": 1,
                "status": "pending",
            },
        ],
    }

    def apply(*args, action_indexes, before_action, on_action, **kwargs):
        assert action_indexes == {1}
        before_action(1)
        on_action(1, "committed", "revision-operation-2")
        return 1, 0

    service._apply_issue = MagicMock(side_effect=apply)

    result = service.apply_check(check.check_id)

    assert result["applied"] == 1
    assert operations.claim.call_args.kwargs["allowed_statuses"] == (
        "planned",
        "failed",
        "applying",
    )
    terminal = operations.write_claimed.call_args_list[-1].args[2]
    assert terminal["actions_executed"] == 2
    assert [checkpoint["status"] for checkpoint in terminal["action_checkpoints"]] == [
        "committed",
        "committed",
    ]


def test_checkpoint_does_not_claim_foreign_revision_transition() -> None:
    service, operations = _service()
    service._vector.get_by_id_strict.return_value = {
        "id": "memory-1",
        "metadata": {
            "revision": 1,
            "revision_state": "superseded",
            "revision_operation_id": "foreign-operation",
        },
    }
    operations.get_by_id.return_value = {
        "status": "committed",
        "fsck_check_id": "other-check",
        "fsck_issue_id": "other-issue",
        "previous_revision_id": "memory-1",
    }

    resolved = service._resolve_action_checkpoints(
        [
            {
                "action": "update",
                "memory_id": "memory-1",
                "expected_revision_id": "memory-1",
                "expected_revision": 1,
            }
        ],
        [
            {
                "action_id": "action-1",
                "action_index": 0,
                "action_kind": "update",
                "status": "applying",
            }
        ],
        complete=False,
        check_id="check-1",
        issue_id="issue-1",
    )

    assert resolved[0]["status"] == "applying"


def test_action_does_not_start_after_lease_loss() -> None:
    operations = MagicMock()
    operations.renew_claim.return_value = False
    checkpoints = [{"status": "pending"}]

    with pytest.raises(RuntimeError, match="lease was lost"):
        FsckService._start_action_checkpoint(
            operations,
            "operation-token",
            "claimant",
            checkpoints,
            300,
            0,
        )

    operations.write_claimed.assert_not_called()


def test_split_source_transition_uses_planned_revision() -> None:
    service, _ = _service()
    issue = FsckIssue(
        issue_id="split-1",
        type="split",
        severity="medium",
        reasoning="split",
        affected_memories=[],
        actions=[FsckAction(action="delete", memory_id="memory-1")],
    )

    executed, skipped = service._apply_issue(
        issue,
        "user-1",
        check_id="check-1",
        expected_revisions={"memory-1": 4},
    )

    assert (executed, skipped) == (1, 0)
    service._memory_service.revisions.mark_source.assert_called_once_with(
        ["memory-1"],
        operation_id="fsck:check-1:split-1",
        user_id="user-1",
        owner_id="owner-1",
        session_agent_id=None,
        reason="split",
        expected_revisions={"memory-1": 4},
    )


def _operation() -> dict:
    return {
        "operation_id": "operation-1",
        "operation_token": "token-1",
        "operation_kind": "fsck",
        "status": "planned",
        "user_id": "user-1",
        "owner_id": "owner-1",
        "agent_id": None,
        "created_at_utc": "2026-01-01T00:00:00+00:00",
        "fsck_check_id": "historical-check",
        "fsck_issue_id": "historical-issue",
        "issue": {
            "type": "duplicate",
            "affected_memories": [
                {
                    "id": "memory-1",
                    "agent_id": None,
                    "metadata": {"revision": 1},
                }
            ],
        },
        "plan": [
            {
                "action": "update",
                "memory_id": "memory-1",
                "new_content": "replacement",
                "new_metadata": None,
                "expected_revision": 1,
            }
        ],
    }


def _audit_record(
    service: FsckService,
    operation: dict,
    *,
    issue: bool,
    target_state: str = "complete",
) -> dict:
    targets = service._operation_target_revisions(operation)
    normalized = service._normalized_action_plan(operation["plan"])
    return {
        "operation_kind": "fsck_audit",
        "mode": "exact_audit",
        "status": "committed",
        "audit_check_id": "fresh-check",
        "user_id": "user-1",
        "owner_id": "owner-1",
        "agent_id": None,
        "basis_operation_id": "operation-1",
        "basis_operation_fingerprint": service._basis_operation_fingerprint(operation),
        "target_revisions": targets,
        "target_state": target_state,
        "target_snapshot_fingerprint": "snapshot-1",
        "created_at_utc": "2026-01-02T00:00:00+00:00",
        "issue_signatures": (
            [
                {
                    "issue_id": "fresh-issue",
                    "type": "duplicate",
                    "action_target_ids": ["memory-1"],
                    "action_count": 1,
                    "plan_fingerprint": canonical_fingerprint(normalized),
                }
            ]
            if issue
            else []
        ),
    }


def test_re_evaluate_rejects_historical_in_memory_check() -> None:
    service, operations = _service()
    check = _completed_check(service, "fresh-check")
    operations.get_by_id.return_value = _operation()
    operations.get_fsck_audit.return_value = None
    operations.reset_mock()

    with pytest.raises(ValueError, match="Fresh exact fsck audit"):
        service.re_evaluate_operation(
            "operation-1",
            fresh_check_id=check.check_id,
            user_id="user-1",
            owner_id="owner-1",
            session_agent_id=None,
            terminalize=False,
        )
    operations.terminalize_unclaimed.assert_not_called()


def test_re_evaluate_terminalizes_without_applying_actions() -> None:
    service, operations = _service()
    service._apply_issue = MagicMock()
    operation = _operation()
    operations.get_by_id.return_value = operation
    operations.get_fsck_audit.return_value = _audit_record(
        service, operation, issue=False
    )
    service._load_exact_audit_targets = MagicMock(
        return_value=(
            "complete",
            [service._vector.get_by_id.return_value],
            "snapshot-1",
        )
    )

    result = service.re_evaluate_operation(
        "operation-1",
        fresh_check_id="fresh-check",
        user_id="user-1",
        owner_id="owner-1",
        session_agent_id=None,
        terminalize=True,
    )

    assert result["outcome"] == "absent"
    assert operations.terminalize_unclaimed.call_args.kwargs["status"] == "superseded"
    service._apply_issue.assert_not_called()


def test_re_evaluate_uses_bound_exact_audit_without_writing() -> None:
    service, operations = _service()
    operation = _operation()
    operations.get_by_id.return_value = operation
    operations.get_fsck_audit.return_value = _audit_record(
        service, operation, issue=True
    )
    service._load_exact_audit_targets = MagicMock(
        return_value=(
            "complete",
            [service._vector.get_by_id.return_value],
            "snapshot-1",
        )
    )

    result = service.re_evaluate_operation(
        "operation-1",
        fresh_check_id="fresh-check",
        user_id="user-1",
        owner_id="owner-1",
        session_agent_id=None,
        terminalize=False,
    )

    assert result["outcome"] == "still_valid"
    assert result["old_action_count"] == 1
    assert result["fresh_action_count"] == 1
    assert result["terminalized"] is False
    operations.terminalize_unclaimed.assert_not_called()


@pytest.mark.parametrize("state", ["absent", "stale"])
def test_re_evaluate_returns_exact_audit_target_state(state: str) -> None:
    service, operations = _service()
    operation = _operation()
    operations.get_by_id.return_value = operation
    operations.get_fsck_audit.return_value = _audit_record(
        service,
        operation,
        issue=False,
        target_state=state,
    )
    memories = [] if state == "absent" else [service._vector.get_by_id.return_value]
    service._load_exact_audit_targets = MagicMock(
        return_value=(state, memories, "snapshot-1")
    )

    result = service.re_evaluate_operation(
        "operation-1",
        fresh_check_id="fresh-check",
        user_id="user-1",
        owner_id="owner-1",
        session_agent_id=None,
        terminalize=False,
    )

    assert result["outcome"] == state
    assert result["terminalized"] is False
    operations.terminalize_unclaimed.assert_not_called()


def test_re_evaluate_reports_changed_exact_audit_plan() -> None:
    service, operations = _service()
    operation = _operation()
    audit = _audit_record(service, operation, issue=True)
    audit["issue_signatures"][0]["plan_fingerprint"] = "changed-plan"
    operations.get_by_id.return_value = operation
    operations.get_fsck_audit.return_value = audit
    service._load_exact_audit_targets = MagicMock(
        return_value=(
            "complete",
            [service._vector.get_by_id.return_value],
            "snapshot-1",
        )
    )

    result = service.re_evaluate_operation(
        "operation-1",
        fresh_check_id="fresh-check",
        user_id="user-1",
        owner_id="owner-1",
        session_agent_id=None,
        terminalize=False,
    )

    assert result["outcome"] == "changed"
    operations.terminalize_unclaimed.assert_not_called()


def test_re_evaluate_prefers_matching_signature_over_changed_signature() -> None:
    service, operations = _service()
    operation = _operation()
    audit = _audit_record(service, operation, issue=True)
    matching = dict(audit["issue_signatures"][0])
    audit["issue_signatures"] = [
        {**matching, "issue_id": "changed", "plan_fingerprint": "changed-plan"},
        matching,
    ]
    operations.get_by_id.return_value = operation
    operations.get_fsck_audit.return_value = audit
    service._load_exact_audit_targets = MagicMock(
        return_value=(
            "complete",
            [service._vector.get_by_id.return_value],
            "snapshot-1",
        )
    )

    result = service.re_evaluate_operation(
        "operation-1",
        fresh_check_id="fresh-check",
        user_id="user-1",
        owner_id="owner-1",
        session_agent_id=None,
        terminalize=False,
    )

    assert result["outcome"] == "still_valid"
    assert result["fresh_issue_id"] == "fresh-issue"


def test_exact_audit_is_bounded_and_does_not_mutate_memory() -> None:
    service, operations = _service()
    operation = _operation()
    operations.get_by_id.return_value = operation
    memory = service._vector.get_by_id.return_value
    service._vector.get_by_ids_strict.return_value = [memory]
    service._evaluate_duplicate_cluster = MagicMock(
        return_value=_completed_check(service).issues
    )

    check = service.start_exact_audit(
        basis_operation_id="operation-1",
        targets=[{"memory_id": "memory-1", "revision": 1}],
        user_id="user-1",
        owner_id="owner-1",
        session_agent_id=None,
    )
    service.run_exact_audit(check.check_id)

    assert check.status == "completed"
    assert check.mode == "exact_audit"
    assert check.target_state == "complete"
    assert service._vector.get_by_ids_strict.call_count == 2
    service._vector.scroll_with_vectors.assert_not_called()
    service._vector.update_metadata.assert_not_called()
    payload = operations.create_fsck_audit.call_args.args[0]
    assert payload["target_revision_ids"] == ["memory-1"]
    assert "replacement" not in str(payload)


def test_exact_audit_includes_affected_memory_without_an_action() -> None:
    service, operations = _service()
    operation = _operation()
    operation["issue"]["affected_memories"].append(
        {
            "id": "memory-2",
            "agent_id": None,
            "metadata": {"revision": 4},
        }
    )
    operations.get_by_id.return_value = operation
    service._vector.get_by_ids_strict.return_value = [
        {
            "id": "memory-1",
            "user_id": "user-1",
            "owner_id": "owner-1",
            "agent_id": None,
            "hash": "hash-1",
            "metadata": {"revision": 1, "revision_state": "active"},
        },
        {
            "id": "memory-2",
            "user_id": "user-1",
            "owner_id": "owner-1",
            "agent_id": None,
            "hash": "hash-2",
            "metadata": {"revision": 4, "revision_state": "active"},
        },
    ]
    service._evaluate_duplicate_cluster = MagicMock(return_value=[])

    check = service.start_exact_audit(
        basis_operation_id="operation-1",
        targets=[
            {"memory_id": "memory-2", "revision": 4},
            {"memory_id": "memory-1", "revision": 1},
        ],
        user_id="user-1",
        owner_id="owner-1",
        session_agent_id=None,
    )
    service.run_exact_audit(check.check_id)

    assert [item["memory_id"] for item in check.target_revisions] == [
        "memory-1",
        "memory-2",
    ]
    evaluated = service._evaluate_duplicate_cluster.call_args.args[0]
    assert [item["id"] for item in evaluated] == ["memory-1", "memory-2"]
    service._vector.scroll_with_vectors.assert_not_called()
    service._vector.update_metadata.assert_not_called()


@pytest.mark.parametrize(
    ("memories", "expected_state"),
    [
        ([], "absent"),
        (
            [
                {
                    "id": "memory-1",
                    "user_id": "user-1",
                    "owner_id": "owner-1",
                    "metadata": {"revision": 2, "revision_state": "active"},
                }
            ],
            "stale",
        ),
    ],
)
def test_exact_audit_records_missing_or_stale_target(
    memories: list[dict],
    expected_state: str,
) -> None:
    service, operations = _service()
    operations.get_by_id.return_value = _operation()
    service._vector.get_by_ids_strict.return_value = memories
    service._evaluate_duplicate_cluster = MagicMock()

    check = service.start_exact_audit(
        basis_operation_id="operation-1",
        targets=[{"memory_id": "memory-1", "revision": 1}],
        user_id="user-1",
        owner_id="owner-1",
        session_agent_id=None,
    )
    service.run_exact_audit(check.check_id)

    assert check.target_state == expected_state
    service._evaluate_duplicate_cluster.assert_not_called()
    assert operations.create_fsck_audit.call_args.args[0]["target_state"] == (
        expected_state
    )
    service._vector.update_metadata.assert_not_called()


def test_exact_audit_rejects_wrong_scope_and_target_revision() -> None:
    service, operations = _service()
    operation = _operation()
    operation["agent_id"] = "agent-1"
    operation["issue"]["affected_memories"][0]["agent_id"] = "agent-1"
    operations.get_by_id.return_value = operation

    with pytest.raises(ValueError, match="not found"):
        service.start_exact_audit(
            basis_operation_id="operation-1",
            targets=[{"memory_id": "memory-1", "revision": 1}],
            user_id="user-1",
            owner_id="owner-1",
            session_agent_id="other-agent",
        )
    with pytest.raises(ValueError, match="not found"):
        service.start_exact_audit(
            basis_operation_id="operation-1",
            targets=[{"memory_id": "memory-1", "revision": 2}],
            user_id="user-1",
            owner_id="owner-1",
            session_agent_id="agent-1",
        )


def test_fsck_audit_store_is_immutable_and_content_free() -> None:
    client = QdrantClient(location=":memory:")
    store = RevisionOperationStore(client, is_remote=False)
    payload = {
        "audit_check_id": "audit-1",
        "user_id": "user-1",
        "owner_id": "owner-1",
        "agent_id": None,
        "basis_operation_id": "operation-1",
        "basis_operation_fingerprint": "basis-fingerprint",
        "target_revisions": [
            {"memory_id": "memory-1", "revision": 1, "agent_id": None}
        ],
        "target_revision_ids": ["memory-1"],
        "target_state": "complete",
        "target_snapshot_fingerprint": "snapshot-1",
        "issue_signatures": [],
        "summary": {"total": 0},
        "created_at_utc": "2026-01-02T00:00:00+00:00",
        "completed_at_utc": "2026-01-02T00:00:01+00:00",
    }

    first = store.create_fsck_audit(payload)
    second = store.create_fsck_audit(payload)

    assert first == second
    assert store.get_fsck_audit("audit-1") == first
    with pytest.raises(ValueError, match="already bound"):
        store.create_fsck_audit(
            {**payload, "basis_operation_fingerprint": "other-basis"}
        )


def test_operation_claim_has_one_concurrent_winner() -> None:
    client = QdrantClient(location=":memory:")
    store = RevisionOperationStore(client, is_remote=False)
    store.write(
        "claim-token",
        {
            "status": "planned",
            "operation_kind": "fsck",
            "lineage_id": "lineage-1",
        },
    )
    barrier = __import__("threading").Barrier(2)
    outcomes: list[bool] = []

    def claim(name: str) -> None:
        barrier.wait(timeout=2)
        outcomes.append(store.claim("claim-token", claimant=name, lease_seconds=60))

    threads = [
        __import__("threading").Thread(target=claim, args=(name,))
        for name in ("worker-1", "worker-2")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert sorted(outcomes) == [False, True]


def test_expired_applying_operation_can_be_reclaimed() -> None:
    from datetime import datetime, timedelta, timezone

    client = QdrantClient(location=":memory:")
    store = RevisionOperationStore(client, is_remote=False)
    store.write(
        "expired-token",
        {
            "status": "applying",
            "operation_kind": "session_retry_batch",
            "lineage_id": "session-1",
            "recovery_token": "old-worker",
            "lease_expires_at": (
                datetime.now(timezone.utc) - timedelta(seconds=1)
            ).isoformat(),
        },
    )

    assert store.claim(
        "expired-token",
        claimant="new-worker",
        lease_seconds=60,
        allowed_statuses=("planned", "failed", "applying"),
    )


def test_only_current_lease_owner_can_renew_or_write() -> None:
    client = QdrantClient(location=":memory:")
    store = RevisionOperationStore(client, is_remote=False)
    store.write(
        "owned-token",
        {
            "status": "planned",
            "operation_kind": "fsck",
            "lineage_id": "lineage-1",
        },
    )
    assert store.claim("owned-token", claimant="worker-1", lease_seconds=60)

    assert store.renew_claim(
        "owned-token",
        "worker-1",
        lease_seconds=60,
    )
    assert not store.renew_claim(
        "owned-token",
        "worker-2",
        lease_seconds=60,
    )
    assert not store.write_claimed(
        "owned-token",
        "worker-2",
        {"status": "failed"},
    )
    assert store.get("owned-token")["status"] == "applying"
