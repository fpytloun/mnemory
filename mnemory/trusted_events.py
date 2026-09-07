"""Trusted semantic event ownership in the existing operation journal.

Both authenticated routes enter here only after their route-specific signature
checks. Ordinary remembers do not acquire this trusted shared-scope lease.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from mnemory.categories import PREDEFINED_CATEGORIES
from mnemory.revisions import (
    EvidenceClaimActiveError,
    EvidenceConflictError,
    EvidenceCorruptError,
    EvidenceLeaseLostError,
    TrustedBudgetRejection,
    canonical_fingerprint,
)
from mnemory.ttl import build_expiry_metadata

PROTOCOL = "mnemory.trusted-semantic.v1"
SOURCE_PROTOCOL = "mnemory.trusted-evidence.v1"


def legacy_result(
    operations: Any,
    *,
    user_id: str,
    owner_id: str,
    root: str,
    route: str,
    request_hashes: dict[str, str],
) -> tuple[bool, dict[str, Any] | None]:
    """Return a historical result, or preserve the original recovery route."""
    rows = {}
    for name, make_id in (
        ("evidence", operations.evidence_operation_id),
        ("ingest", operations.user_event_operation_id),
    ):
        row = operations._user_event_readback(
            make_id(
                protocol=SOURCE_PROTOCOL,
                user_id=user_id,
                owner_id=owner_id,
                evidence_root_id=root,
            )
        )
        if row is None:
            continue
        if (
            row.get("user_id") != user_id
            or row.get("owner_id") != owner_id
            or row.get("evidence_root_id") != root
            or row.get("request_fingerprint") != request_hashes.get(name)
        ):
            raise EvidenceConflictError("Historical event request binding differs")
        rows[name] = row
    # No historical terminal event is reinterpreted by the semantic pipeline.
    for name in (route, "ingest" if route == "evidence" else "evidence"):
        row = rows.get(name)
        if row is not None and row.get("status") == "committed":
            return True, {
                "status": "replayed",
                "operation_id": row["operation_id"],
                "result": row.get("result"),
            }
    if route in rows:
        return True, None
    if rows:
        raise EvidenceClaimActiveError("Historical event requires its original route")
    return False, None


class TrustedEventLease:
    """Bounded renewal and cancellation for one trusted shared scope."""

    def __init__(
        self, operations: Any, claim: dict[str, Any], cancel: threading.Event
    ) -> None:
        self.operations = operations
        self.claim = claim
        self.cancel = cancel
        self.deadline = time.monotonic() + 85
        self.stopped = threading.Event()
        self.lost = threading.Event()
        self.renew_lock = threading.Lock()
        self.thread = threading.Thread(target=self._renew, daemon=True)

    def check(self) -> None:
        if (
            self.cancel.is_set()
            or self.lost.is_set()
            or time.monotonic() >= self.deadline
        ):
            raise EvidenceLeaseLostError("Trusted event was cancelled or fenced")
        with self.renew_lock:
            if self.stopped.is_set():
                raise EvidenceLeaseLostError("Trusted scope renewal stopped")
            self.operations.renew_user_event_content_claim(
                self.claim["operation_id"],
                operation_id=self.claim["claim_owner"],
                claim_epoch=self.claim["claim_epoch"],
                claim_nonce=self.claim["claim_nonce"],
                lease_seconds=15,
                require_live=True,
            )

    def _renew(self) -> None:
        while not self.stopped.wait(3):
            try:
                self.check()
            except Exception:
                self.lost.set()
                return

    def __enter__(self) -> TrustedEventLease:
        self.check()
        self.thread.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stopped.set()
        self.thread.join(timeout=2)

    def release(self, pending_operation_id: str | None = None) -> None:
        """Fence renewal before publishing an expired scope lease."""
        self.stopped.set()
        with self.renew_lock:
            self.operations.release_trusted_scope_claim(
                self.claim, pending_operation_id=pending_operation_id
            )


def process(
    service: Any,
    *,
    content: str,
    user_id: str,
    owner_id: str,
    evidence_root_id: str,
    source_event: dict[str, Any],
    request_hashes: dict[str, str],
    route: str,
    cancel: threading.Event,
) -> dict[str, Any] | None:
    """Execute one immutable event plan, or defer to historical recovery."""
    operations = service.revisions.operations
    fingerprint = canonical_fingerprint(
        {
            "protocol": SOURCE_PROTOCOL,
            "user_id": user_id,
            "owner_id": owner_id,
            "root": evidence_root_id,
            "content": content,
            "event": {
                key: value
                for key, value in source_event.items()
                if key != "request_hash"
            },
        }
    )
    operation_id = operations.evidence_operation_id(
        protocol=PROTOCOL,
        user_id=user_id,
        owner_id=owner_id,
        evidence_root_id=evidence_root_id,
    )
    scope_key = canonical_fingerprint(
        ["trusted_semantic_scope", user_id, owner_id, None, "user"]
    )
    scope_id = operations.user_event_content_claim_id(scope_key)
    worker_id = str(uuid.uuid4())
    wait_until = time.monotonic() + 5
    while True:
        if cancel.is_set():
            raise EvidenceLeaseLostError("Trusted event was cancelled")
        previous = operations._user_event_readback(scope_id)
        pending_id = (previous or {}).get("memory_id") or operation_id
        claim = operations.claim_user_event_content(
            content_fingerprint=scope_key,
            operation_id=worker_id,
            user_id=user_id,
            owner_id=owner_id,
            memory_id=pending_id,
            lease_seconds=15,
        )
        if claim.get("claim_owner") == worker_id:
            break
        if time.monotonic() >= wait_until:
            raise EvidenceClaimActiveError("Trusted semantic scope is busy; retry")
        cancel.wait(0.05)

    with TrustedEventLease(operations, claim, cancel) as lease:
        # A sealed predecessor must finish before another event makes a semantic
        # ADD decision. Otherwise a crash between sealing and ADD could duplicate
        # a later paraphrase. No extraction is repeated during this recovery.
        pending = operations.get_evidence_plan(claim["memory_id"])
        if pending is not None and pending.get("status") != "committed":
            apply_plan(service, pending, lease.check)
        historical, result = legacy_result(
            operations,
            user_id=user_id,
            owner_id=owner_id,
            root=evidence_root_id,
            route=route,
            request_hashes=request_hashes,
        )
        if historical:
            lease.release()
            return result
        record = operations.get_evidence_plan(operation_id)
        if record is not None:
            if record["request_fingerprint"] != fingerprint:
                raise EvidenceConflictError("Trusted event identity differs")
            if record.get("status") == "committed":
                lease.release()
                if (record.get("result") or {}).get(
                    "outcome"
                ) == "rejected_before_write":
                    return {
                        **record["result"],
                        "operation_id": operation_id,
                    }
                return {
                    "status": "replayed",
                    "operation_id": operation_id,
                    "result": record.get("result"),
                }
        if record is None:
            try:
                targets = plan(
                    service,
                    content,
                    user_id=user_id,
                    owner_id=owner_id,
                    evidence_root_id=evidence_root_id,
                    operation_id=operation_id,
                    source_event=source_event,
                    guard=lease.check,
                )
                lease.check()
                # Bind scope recovery before sealing. An absent parent has no effects.
                operations.bind_trusted_scope_plan(claim, operation_id)
                record = operations.seal_evidence_plan(
                    protocol=PROTOCOL,
                    user_id=user_id,
                    owner_id=owner_id,
                    evidence_root_id=evidence_root_id,
                    request_fingerprint=fingerprint,
                    targets=targets,
                )
            except TrustedBudgetRejection as exc:
                # Only typed pre-insert budget failures qualify. Transport
                # errors, malformed LLM responses and application failures never
                # enter this path, even if they occur before the first write.
                lease.check()
                record = operations.seal_evidence_plan(
                    protocol=PROTOCOL,
                    user_id=user_id,
                    owner_id=owner_id,
                    evidence_root_id=evidence_root_id,
                    request_fingerprint=fingerprint,
                    targets=[],
                    terminal_rejection=exc.reason,
                )
                if (
                    record.get("status") != "committed"
                    or record.get("targets") != []
                    or record.get("checkpoints") != []
                    or (record.get("result") or {}).get("outcome")
                    != "rejected_before_write"
                ):
                    raise EvidenceConflictError("Existing semantic plan must resume")
                lease.release(operation_id)
                return {**record["result"], "operation_id": operation_id}
        lease.check()
        result = apply_plan(service, record, lease.check)
        lease.release(operation_id)
        return {
            "status": "accepted",
            "operation_id": operation_id,
            "result": result,
        }


def plan(
    service: Any,
    content: str,
    *,
    user_id: str,
    owner_id: str,
    evidence_root_id: str,
    operation_id: str,
    source_event: dict[str, Any],
    guard: Callable[[], None],
) -> list[dict[str, Any]]:
    """Reuse remember extraction and LLM dedup without executing their actions."""
    maximum = service._config.memory.max_input_length
    if len(content) > maximum:
        # Never drop a prefix of the signed message or store it as one fact.
        raise TrustedBudgetRejection("input_budget_exceeded")
    guard()
    facts, _, _ = service._remember_extract(
        content,
        role="user",
        session_context=None,
        available_categories=list(PREDEFINED_CATEGORIES),
        max_memory_length=service._config.memory.max_memory_length,
        session_timezone=None,
        context=None,
        fail_closed=True,
    )
    if len(facts) > 32:
        raise TrustedBudgetRejection("action_limit_exceeded")
    if any(
        isinstance(fact.get("text"), str)
        and len(fact["text"]) > service._config.memory.max_memory_length
        for fact in facts
    ):
        raise TrustedBudgetRejection("extraction_fact_budget_exceeded")
    if any(
        not isinstance(fact.get("text"), str) or not fact["text"].strip()
        for fact in facts
    ):
        raise ValueError("Trusted extraction contains an invalid fact")
    targets = []
    pending = []
    touched = set()
    for ordinal, fact in enumerate(facts):
        guard()
        vector = service.vector.embedding.embed(fact["text"])
        candidates = service.vector.search_similar(
            vector,
            user_id=user_id,
            owner_id=owner_id,
            subject_user_id=user_id,
            agent_id=None,
            shared_only=True,
            limit=10,
        )
        resolved = []
        for candidate in candidates:
            if (
                candidate.get("score", 0)
                < service._config.memory.dedup_similarity_threshold
            ):
                continue
            current = service.vector.get_by_id_strict(candidate["id"])
            if (
                current
                and current.get("user_id") == user_id
                and current.get("owner_id") == owner_id
                and current.get("agent_id") is None
                and (current.get("metadata") or {}).get("role") == "user"
                and (current.get("metadata") or {}).get("revision_state") == "active"
            ):
                resolved.append({**current, "text": current["memory"]})
        candidates = resolved
        candidates.sort(
            key=lambda item: (item.get("metadata") or {}).get("memory_layer") != "raw"
        )
        # Include earlier sealed-in-this-plan facts so one event cannot add its
        # own paraphrase and then count it as an independent confirmation.
        choices = candidates + pending
        action = service._dedup_with_llm(
            [{"index": 0, "text": fact["text"], "candidates": choices}],
            [fact],
            trusted=True,
        )[0]
        action = {**action, "ordinal": ordinal, "source_event": source_event}
        action["expiry"] = build_expiry_metadata(
            None, action["memory_type"], service._config.memory
        )
        if action["memory_type"] == "episodic" and not action.get("event_date"):
            action["event_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        target_id = action.get("target_id")
        candidate = next((item for item in choices if item["id"] == target_id), None)
        if action["action"] in {"CONFIRM", "UPDATE"}:
            if candidate is None or candidate in pending or target_id in touched:
                action.update(action="SKIP", reason="same_event_or_missing_target")
            else:
                metadata = candidate.get("metadata") or {}
                action["snapshot"] = {
                    "revision_id": candidate["id"],
                    "revision": metadata.get("revision"),
                    "lineage_id": metadata.get("lineage_id"),
                    **service._evidence_candidate_hashes(candidate),
                }
                if not isinstance(metadata.get("revision"), int):
                    action.update(action="SKIP", reason="unversioned_target")
                elif action["action"] == "CONFIRM":
                    if not service._evidence_semantic_equivalence(
                        content, candidate["memory"], fact["text"]
                    ):
                        action.update(action="SKIP", reason="not_complete_equivalence")
                elif metadata.get("memory_layer") != "raw":
                    # A raw assertion cannot rewrite a multi-fact consolidation.
                    action.update(action="SKIP", reason="consolidated_update")
            if action["action"] in {"CONFIRM", "UPDATE"}:
                touched.add(target_id)
        if action["action"] == "ADD":
            memory_id = str(uuid.uuid5(uuid.UUID(operation_id), f"action:{ordinal}"))
            action["memory_id"] = memory_id
            pending.append(
                {
                    "id": memory_id,
                    "memory": action["text"],
                    "text": action["text"],
                    "metadata": {"role": "user", "memory_layer": "raw"},
                }
            )
        targets.append(action)
    return targets


def apply_plan(
    service: Any, record: dict[str, Any], guard: Callable[[], None]
) -> dict[str, Any]:
    """Resume persisted actions without repeating extraction or semantic decisions."""
    if record.get("protocol") != PROTOCOL:
        raise EvidenceCorruptError("Scope recovery points to a non-semantic journal")
    operations = service.revisions.operations
    guard()
    record = operations.claim_evidence_plan(
        record["operation_id"],
        request_fingerprint=record["request_fingerprint"],
        epoch=int(record["claim_epoch"]) + 1,
        nonce=uuid.uuid4().hex,
    )
    identity = {
        "request_fingerprint": record["request_fingerprint"],
        "epoch": record["claim_epoch"],
        "nonce": record["claim_nonce"],
    }

    def fence() -> None:
        guard()
        operations.verify_evidence_claim(record["operation_id"], **identity)

    checkpoints = list(record["checkpoints"])
    finished = {item["ordinal"] for item in checkpoints}
    for action in record["targets"]:
        if action["ordinal"] in finished:
            continue
        fence()
        result = service._execute_action(
            action,
            user_id=record["user_id"],
            owner_id=record["owner_id"],
            agent_id=None,
            role="user",
            ttl_days=None,
            explicit_fields={},
            vector_map={},
            memory_layer="raw",
            source_kind="raw_user_message",
            source_fingerprint=record["request_fingerprint"],
            evidence_root_id=record["evidence_root_id"],
            validation_eligible=True,
            trusted_parent=record,
            mutation_guard=fence,
        )
        if result is None:
            raise EvidenceCorruptError("Sealed trusted action returned no result")
        status = {
            "ADD": "added",
            "UPDATE": "updated",
            "CONFIRM": "confirmed",
            "SKIP": "skipped",
        }[result["event"]]
        checkpoints.append(
            {
                "ordinal": action["ordinal"],
                "action": action["action"],
                "target_id": action.get("target_id"),
                "status": status,
                "result": result,
            }
        )
        fence()
        operations.checkpoint_evidence_plan(
            record["operation_id"], checkpoints=checkpoints, **identity
        )
    result = {
        "status": "committed",
        "checkpoints": checkpoints,
        "results": [item["result"] for item in checkpoints],
    }
    fence()
    operations.commit_evidence_plan(record["operation_id"], result=result, **identity)
    return result
