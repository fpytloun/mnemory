"""Immutable memory revision transitions backed only by Qdrant."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from qdrant_client.models import (
    DatetimeRange,
    Direction,
    Distance,
    FieldCondition,
    Filter,
    HasIdCondition,
    IsEmptyCondition,
    MatchAny,
    MatchValue,
    OrderBy,
    PayloadField,
    PointIdsList,
    PointStruct,
    Range,
    VectorParams,
    WriteOrdering,
)

logger = logging.getLogger(__name__)

OPERATIONS_COLLECTION = "_mnemory_operations"
_OPERATION_CLAIM_LOCK = threading.RLock()
ACTIVE_REVISION_STATE = "active"
EVIDENCE_OPERATION_KIND = "evidence_plan"
EVIDENCE_PLAN_PROTOCOL = "mnemory.trusted-evidence.v1"
EVIDENCE_MAX_TARGETS = 32
EVIDENCE_MAX_PLAN_BYTES = 64 * 1024
EVIDENCE_CLAIM_DEADLINE_SECONDS = 90
FSCK_AUDIT_OPERATION_KIND = "fsck_audit"
FSCK_AUDIT_MODE = "exact_audit"
FSCK_AUDIT_MAX_TARGETS = 20
TERMINAL_REVISION_STATES = {
    "superseded",
    "source",
    "retracted",
    "aborted",
}
_TRANSITION_FIELDS = {
    "transition_token",
    "transition_started_at_utc",
    "transition_kind",
    "transition_successor_id",
    "transition_operation_token",
    "evidence_child_fence",
}


def canonical_fingerprint(value: Any) -> str:
    """Return a stable SHA-256 fingerprint for a normalized request."""
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _normalized_fact_hash(text: str) -> str:
    """Hash a whitespace- and case-normalized assertion."""
    normalized = " ".join(text.split()).casefold()
    return hashlib.sha256(normalized.encode()).hexdigest()


def _operation_point_id(token: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"mnemory:operation:{token}"))


def _successor_point_id(token: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"mnemory:revision:{token}"))


class RevisionConflictError(ValueError):
    """Raised when a revision operation targets a stale or claimed head."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "revision_conflict",
        lineage_id: str | None = None,
        current_revision_id: str | None = None,
        current_revision: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.lineage_id = lineage_id
        self.current_revision_id = current_revision_id
        self.current_revision = current_revision

    def to_dict(self) -> dict[str, Any]:
        """Return a transport-safe conflict payload."""
        return {
            "error": True,
            "code": self.code,
            "message": str(self),
            "lineage_id": self.lineage_id,
            "current_revision_id": self.current_revision_id,
            "current_revision": self.current_revision,
        }


class EvidenceConflictError(RevisionConflictError):
    """Raised when an evidence operation conflicts with a durable winner."""


class EvidenceCorruptError(RevisionConflictError):
    """Raised when a persisted evidence operation is malformed."""


class EvidenceClaimActiveError(RevisionConflictError):
    """Raised when another worker owns an unexpired evidence claim."""


class EvidenceLeaseLostError(RevisionConflictError):
    """Raised when a worker no longer owns an evidence lease."""


class RevisionOperationStore:
    """Durable revision operation journal and audit store."""

    def __init__(
        self,
        client: Any,
        *,
        is_remote: bool,
        memory_collection: str = "memories",
    ) -> None:
        self._client = client
        self._is_remote = is_remote
        self._memory_collection = memory_collection
        self.ensure_collection()

    def ensure_collection(self) -> None:
        """Create the operation collection and its payload indexes."""
        collections = {item.name for item in self._client.get_collections().collections}
        verify_peer_schema = False
        if OPERATIONS_COLLECTION not in collections:
            try:
                self._client.create_collection(
                    collection_name=OPERATIONS_COLLECTION,
                    vectors_config=VectorParams(size=1, distance=Distance.COSINE),
                )
            except Exception:
                # Another replica can create the collection after our initial
                # list. Treat that race as success only after schema checks.
                collections = {
                    item.name for item in self._client.get_collections().collections
                }
                if OPERATIONS_COLLECTION not in collections:
                    raise
                verify_peer_schema = True
        if verify_peer_schema:
            info = self._client.get_collection(OPERATIONS_COLLECTION)
            vectors = info.config.params.vectors
            if (
                not isinstance(vectors, VectorParams)
                or vectors.size != 1
                or vectors.distance != Distance.COSINE
            ):
                raise RuntimeError(
                    f"{OPERATIONS_COLLECTION} has an incompatible vector schema"
                )
        if not self._is_remote:
            return
        schemas = {
            "user_id": "keyword",
            "owner_id": "keyword",
            "agent_id": "keyword",
            "lineage_id": "keyword",
            "operation_kind": "keyword",
            "status": "keyword",
            "idempotency_key_hash": "keyword",
            "fsck_check_id": "keyword",
            "fsck_issue_id": "keyword",
            "session_id": "keyword",
            "target_revision_ids": "keyword",
            "affected_lineage_ids": "keyword",
            "target_revision_id": "keyword",
            "source_kind": "keyword",
            "recovery_token": "keyword",
            "created_at_utc": "datetime",
            "updated_at_utc": "datetime",
            "lease_expires_at": "datetime",
            "protocol": "keyword",
            "evidence_root_id": "keyword",
            "request_fingerprint": "keyword",
            "target_ids": "keyword",
            "claim_epoch": "integer",
            "claim_nonce": "keyword",
            "audit_check_id": "keyword",
            "basis_operation_id": "keyword",
            "basis_operation_fingerprint": "keyword",
            "target_snapshot_fingerprint": "keyword",
            "mode": "keyword",
        }
        for field, schema in schemas.items():
            try:
                self._client.create_payload_index(
                    collection_name=OPERATIONS_COLLECTION,
                    field_name=field,
                    field_schema=schema,
                )
            except Exception:
                logger.debug("Operation index %s already exists", field)

    def write(self, token: str, payload: dict[str, Any]) -> str:
        """Upsert one operation checkpoint and return its stable ID."""
        if str(payload.get("operation_kind", "")).startswith("evidence"):
            raise ValueError("Evidence operations must use evidence-specific methods")
        operation_id = _operation_point_id(token)
        now = datetime.now(timezone.utc).isoformat()
        existing = self.get(token)
        body = dict(existing or {})
        body.update(payload)
        body.setdefault("created_at_utc", now)
        body["updated_at_utc"] = now
        body["operation_id"] = operation_id
        body["operation_token"] = token
        self._client.upsert(
            collection_name=OPERATIONS_COLLECTION,
            points=[PointStruct(id=operation_id, vector=[0.0], payload=body)],
            wait=True,
        )
        return operation_id

    @staticmethod
    def fsck_audit_operation_id(check_id: str) -> str:
        """Return the deterministic point ID for one exact fsck audit."""
        return _operation_point_id(f"fsck-audit:{check_id}")

    def get_fsck_audit(self, check_id: str) -> dict[str, Any] | None:
        """Read one completed exact fsck audit with all-replica consistency."""
        result = self._client.retrieve(
            collection_name=OPERATIONS_COLLECTION,
            ids=[self.fsck_audit_operation_id(check_id)],
            with_payload=True,
            with_vectors=False,
            consistency="all",
        )
        if not result:
            return None
        record = dict(result[0].payload or {})
        if (
            record.get("operation_kind") != FSCK_AUDIT_OPERATION_KIND
            or record.get("mode") != FSCK_AUDIT_MODE
            or record.get("audit_check_id") != check_id
        ):
            return None
        return record

    def create_fsck_audit(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Insert one immutable completed exact-target fsck audit."""
        check_id = payload.get("audit_check_id")
        targets = payload.get("target_revisions")
        if not isinstance(check_id, str) or not check_id:
            raise ValueError("Fsck audit check ID is required")
        if (
            not isinstance(targets, list)
            or not targets
            or len(targets) > FSCK_AUDIT_MAX_TARGETS
        ):
            raise ValueError("Fsck audit targets are invalid")
        operation_id = self.fsck_audit_operation_id(check_id)
        core = {
            **payload,
            "operation_id": operation_id,
            "operation_kind": FSCK_AUDIT_OPERATION_KIND,
            "mode": FSCK_AUDIT_MODE,
            "status": "committed",
        }
        body = {
            **core,
            "audit_fingerprint": canonical_fingerprint(core),
            "updated_at_utc": payload["completed_at_utc"],
        }
        self._client.upsert(
            collection_name=OPERATIONS_COLLECTION,
            points=[PointStruct(id=operation_id, vector=[0.0], payload=body)],
            update_filter=Filter(must_not=[HasIdCondition(has_id=[operation_id])]),
            ordering=WriteOrdering.STRONG,
            wait=True,
        )
        winner = self.get_fsck_audit(check_id)
        if winner is None:
            raise RuntimeError("Fsck audit insert was not readable")
        if winner.get("audit_fingerprint") != body["audit_fingerprint"]:
            raise ValueError("Fsck audit check ID is already bound")
        return winner

    def claim_evidence_child(
        self,
        token: str,
        *,
        parent_operation_id: str,
        ordinal: int,
        epoch: int,
        nonce: str,
    ) -> dict[str, Any]:
        """Claim the deterministic child fence before mutating its target."""
        operation_id = _operation_point_id(token)
        record = self.get(token)
        if record and record.get("status") == "committed":
            return record
        if record and (
            record.get("parent_operation_id") != parent_operation_id
            or record.get("evidence_child_key") != f"{parent_operation_id}:{ordinal}"
        ):
            raise EvidenceConflictError("Evidence child binding conflict")
        current_epoch = int((record or {}).get("child_claim_epoch", 0))
        if record and record.get("status") == "claimed":
            deadline = record.get("child_claim_deadline_utc")
            try:
                active = datetime.fromisoformat(deadline) > datetime.now(timezone.utc)
            except (TypeError, ValueError):
                active = False
            if active and (
                record.get("child_claim_epoch") != epoch
                or record.get("child_claim_nonce") != nonce
            ):
                raise EvidenceClaimActiveError("Evidence child is actively claimed")
            if (
                record.get("child_claim_epoch") == epoch
                and record.get("child_claim_nonce") == nonce
            ):
                self._repair_child_target_fence(record)
                return record
        if epoch <= current_epoch:
            raise EvidenceConflictError("Evidence child epoch is stale")
        deadline = (
            datetime.now(timezone.utc)
            + timedelta(seconds=EVIDENCE_CLAIM_DEADLINE_SECONDS)
        ).isoformat()
        child_transition_token = canonical_fingerprint([token, epoch, nonce])
        self._client.set_payload(
            collection_name=OPERATIONS_COLLECTION,
            payload={
                "status": "claimed",
                "child_claim_epoch": epoch,
                "child_claim_nonce": nonce,
                "child_claim_deadline_utc": deadline,
                "child_transition_token": child_transition_token,
            },
            points=Filter(
                must=[
                    HasIdCondition(has_id=[operation_id]),
                    Filter(
                        should=[
                            FieldCondition(
                                key="status", match=MatchValue(value="prepared")
                            ),
                            FieldCondition(
                                key="status", match=MatchValue(value="claimed")
                            ),
                        ]
                    ),
                    FieldCondition(
                        key="parent_operation_id",
                        match=MatchValue(value=parent_operation_id),
                    ),
                    Filter(
                        should=[
                            FieldCondition(
                                key="child_claim_epoch",
                                match=MatchValue(value=current_epoch),
                            ),
                            IsEmptyCondition(
                                is_empty=PayloadField(key="child_claim_epoch")
                            ),
                        ]
                    ),
                ]
            ),
            ordering=WriteOrdering.STRONG,
            wait=True,
        )
        result = self.get(token)
        if result is None or result.get("status") not in {"claimed", "committed"}:
            raise EvidenceLeaseLostError("Evidence child fence was not claimed")
        self._repair_child_target_fence(result)
        return result

    def _repair_child_target_fence(self, record: dict[str, Any]) -> None:
        """Conditionally repair a target fence after a child claim."""
        if record and record.get("target_revision_id"):
            child_transition_token = record.get("child_transition_token")
            if not isinstance(child_transition_token, str):
                raise EvidenceLeaseLostError("Evidence child fence is missing")
            operation_token = record.get("operation_token")
            if not isinstance(operation_token, str):
                raise EvidenceLeaseLostError("Evidence child operation is missing")
            target_id = record["target_revision_id"]
            target_points = self._client.retrieve(
                collection_name=self._memory_collection,
                ids=[target_id],
                with_payload=True,
                with_vectors=False,
                consistency="all",
            )
            if not target_points:
                raise EvidenceLeaseLostError("Evidence child target is missing")
            target_payload = dict(target_points[0].payload or {})
            target_conditions: list[Any] = [HasIdCondition(has_id=[target_id])]
            for record_key, target_key in (
                ("lineage_id", "lineage_id"),
                ("target_revision", "revision"),
                ("target_content_hash", "hash"),
                ("target_fact_hash", "fact_hash"),
            ):
                expected = record.get(record_key)
                if expected is not None:
                    actual = target_payload.get(target_key)
                    if target_key == "fact_hash":
                        actual = actual or self._normalized_fact_hash(
                            str(target_payload.get("data", ""))
                        )
                    if actual != expected:
                        raise EvidenceLeaseLostError(
                            "Evidence child target identity is stale"
                        )
                    if (
                        target_key != "fact_hash"
                        or target_payload.get(target_key) is not None
                    ):
                        target_conditions.append(
                            FieldCondition(
                                key=target_key,
                                match=MatchValue(value=expected),
                            )
                        )

            target_operation_token = target_payload.get("transition_operation_token")
            target_transition_token = target_payload.get("transition_token")
            if (
                target_operation_token is not None
                and target_operation_token != operation_token
            ):
                raise EvidenceLeaseLostError(
                    "Evidence child target is claimed elsewhere"
                )
            if target_operation_token == operation_token:
                if (
                    not isinstance(target_transition_token, str)
                    or target_payload.get("transition_kind") != "confirm"
                ):
                    raise EvidenceLeaseLostError(
                        "Evidence child target transition is incomplete"
                    )
                target_conditions.extend(
                    [
                        FieldCondition(
                            key="transition_operation_token",
                            match=MatchValue(value=operation_token),
                        ),
                        FieldCondition(
                            key="transition_kind",
                            match=MatchValue(value="confirm"),
                        ),
                        FieldCondition(
                            key="transition_token",
                            match=MatchValue(value=target_transition_token),
                        ),
                    ]
                )
            else:
                if (
                    target_transition_token is not None
                    or target_payload.get("transition_kind") is not None
                ):
                    raise EvidenceLeaseLostError(
                        "Evidence child target transition is claimed elsewhere"
                    )
                target_conditions.extend(
                    [
                        IsEmptyCondition(
                            is_empty=PayloadField(key="transition_operation_token")
                        ),
                        IsEmptyCondition(is_empty=PayloadField(key="transition_token")),
                        IsEmptyCondition(is_empty=PayloadField(key="transition_kind")),
                    ]
                )
            self._client.set_payload(
                collection_name=self._memory_collection,
                payload={"evidence_child_fence": child_transition_token},
                points=Filter(must=target_conditions),
                ordering=WriteOrdering.STRONG,
                wait=True,
            )
            target = self._client.retrieve(
                collection_name=self._memory_collection,
                ids=[target_id],
                with_payload=True,
                with_vectors=False,
                consistency="all",
            )
            if (
                not target
                or (target[0].payload or {}).get("evidence_child_fence")
                != child_transition_token
                or (target[0].payload or {}).get("transition_operation_token")
                != target_operation_token
                or (
                    target_operation_token == operation_token
                    and (target[0].payload or {}).get("transition_token")
                    != target_transition_token
                )
            ):
                raise EvidenceLeaseLostError("Evidence target child fence was lost")

    def verify_evidence_child(
        self,
        token: str,
        *,
        epoch: int,
        nonce: str,
    ) -> dict[str, Any]:
        """Verify the child lease immediately before target mutation."""
        record = self.get(token)
        if record is None or record.get("status") != "claimed":
            raise EvidenceLeaseLostError("Evidence child is not claimed")
        if (
            record.get("child_claim_epoch") != epoch
            or record.get("child_claim_nonce") != nonce
        ):
            raise EvidenceLeaseLostError("Evidence child lease is no longer owned")
        try:
            active = datetime.fromisoformat(
                record["child_claim_deadline_utc"]
            ) > datetime.now(timezone.utc)
        except (KeyError, TypeError, ValueError):
            active = False
        if not active:
            raise EvidenceLeaseLostError("Evidence child lease has expired")
        return record

    def complete_evidence_child(
        self,
        token: str,
        *,
        epoch: int,
        nonce: str,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """Commit the fixed child operation under its own lease."""
        self.verify_evidence_child(token, epoch=epoch, nonce=nonce)
        operation_id = _operation_point_id(token)
        self._client.set_payload(
            collection_name=OPERATIONS_COLLECTION,
            payload={"status": "committed", "result": result},
            points=Filter(
                must=[
                    HasIdCondition(has_id=[operation_id]),
                    FieldCondition(
                        key="child_claim_epoch", match=MatchValue(value=epoch)
                    ),
                    FieldCondition(
                        key="child_claim_nonce", match=MatchValue(value=nonce)
                    ),
                    FieldCondition(key="status", match=MatchValue(value="claimed")),
                ]
            ),
            ordering=WriteOrdering.STRONG,
            wait=True,
        )
        updated = self.get(token)
        if updated is None or updated.get("status") != "committed":
            raise EvidenceLeaseLostError("Evidence child completion lost")
        return updated

    @staticmethod
    def evidence_operation_id(
        *,
        protocol: str,
        user_id: str,
        owner_id: str,
        evidence_root_id: str,
    ) -> str:
        """Return the deterministic parent ID for an evidence request."""
        token = canonical_fingerprint(
            [
                protocol,
                user_id,
                owner_id,
                evidence_root_id,
            ]
        )
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"mnemory:evidence:{token}"))

    @staticmethod
    def _evidence_record_valid(record: dict[str, Any]) -> bool:
        """Validate the immutable fields required by the evidence journal."""
        targets = record.get("targets")
        immutable_strings = (
            record.get("protocol"),
            record.get("user_id"),
            record.get("owner_id"),
            record.get("evidence_root_id"),
            record.get("request_fingerprint"),
        )
        return (
            record.get("operation_kind") == EVIDENCE_OPERATION_KIND
            and all(isinstance(value, str) and value for value in immutable_strings)
            and isinstance(targets, list)
            and len(targets) <= EVIDENCE_MAX_TARGETS
            and all(isinstance(target, dict) for target in targets)
            and all(
                target.get("action") in {"CONFIRM", "SKIP"}
                and isinstance(target.get("ordinal"), int)
                and target.get("ordinal") == index
                for index, target in enumerate(targets)
            )
            and isinstance(record.get("claim_epoch"), int)
            and isinstance(record.get("checkpoints"), list)
            and record.get("status") in {"planned", "claimed", "committed", "corrupt"}
            and (
                record.get("status") != "claimed"
                or (
                    isinstance(record.get("claim_nonce"), str)
                    and bool(record.get("claim_nonce"))
                    and isinstance(record.get("claim_deadline_utc"), str)
                )
            )
        )

    def _evidence_readback(self, operation_id: str) -> dict[str, Any] | None:
        result = self._client.retrieve(
            collection_name=OPERATIONS_COLLECTION,
            ids=[operation_id],
            with_payload=True,
            with_vectors=False,
            consistency="all",
        )
        return dict(result[0].payload or {}) if result else None

    def seal_evidence_plan(
        self,
        *,
        protocol: str = EVIDENCE_PLAN_PROTOCOL,
        user_id: str,
        owner_id: str,
        evidence_root_id: str,
        request_fingerprint: str,
        targets: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Atomically persist or resume one complete, immutable evidence plan."""
        if len(targets) > EVIDENCE_MAX_TARGETS:
            raise ValueError("Evidence plans may contain at most 32 targets")
        operation_id = self.evidence_operation_id(
            protocol=protocol,
            user_id=user_id,
            owner_id=owner_id,
            evidence_root_id=evidence_root_id,
        )
        body: dict[str, Any] = {
            "operation_id": operation_id,
            "operation_kind": EVIDENCE_OPERATION_KIND,
            "protocol": protocol,
            "user_id": user_id,
            "owner_id": owner_id,
            "evidence_root_id": evidence_root_id,
            "request_fingerprint": request_fingerprint,
            "targets": targets,
            "target_ids": [
                target["target_id"]
                for target in targets
                if isinstance(target.get("target_id"), str)
            ],
            "status": "planned",
            "claim_epoch": 0,
            "claim_nonce": None,
            "checkpoints": [],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        encoded_size = len(
            json.dumps(
                body, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode()
        )
        if encoded_size > EVIDENCE_MAX_PLAN_BYTES:
            raise ValueError("Evidence plan exceeds 64 KiB")
        self._client.upsert(
            collection_name=OPERATIONS_COLLECTION,
            points=[
                PointStruct(
                    id=operation_id,
                    vector=[0.0],
                    payload=body,
                )
            ],
            update_filter=Filter(must_not=[HasIdCondition(has_id=[operation_id])]),
            ordering=WriteOrdering.STRONG,
            wait=True,
        )
        winner = self._evidence_readback(operation_id)
        if winner is None:
            raise EvidenceCorruptError("Evidence plan insert was not readable")
        if not self._evidence_record_valid(winner):
            self._mark_evidence_corrupt(operation_id, winner)
            raise EvidenceCorruptError("Persisted evidence plan is malformed")
        if (
            winner.get("protocol") != protocol
            or winner.get("user_id") != user_id
            or winner.get("owner_id") != owner_id
            or winner.get("evidence_root_id") != evidence_root_id
            or winner.get("request_fingerprint") != request_fingerprint
        ):
            raise EvidenceConflictError(
                "Evidence operation ID is already bound to another request",
                code="idempotency_conflict",
            )
        return winner

    def _mark_evidence_corrupt(self, operation_id: str, record: dict[str, Any]) -> None:
        """Move a malformed winner to the terminal corrupt state."""
        self._client.set_payload(
            collection_name=OPERATIONS_COLLECTION,
            payload={"status": "corrupt"},
            points=Filter(must=[HasIdCondition(has_id=[operation_id])]),
            ordering=WriteOrdering.STRONG,
            wait=True,
        )

    def get_evidence_plan(self, operation_id: str) -> dict[str, Any] | None:
        """Read an evidence plan with strong consistency and validate it."""
        record = self._evidence_readback(operation_id)
        if record is not None and not self._evidence_record_valid(record):
            self._mark_evidence_corrupt(operation_id, record)
            raise EvidenceCorruptError("Persisted evidence plan is malformed")
        return record

    def _conditional_evidence_update(
        self,
        *,
        operation_id: str,
        request_fingerprint: str,
        status: str,
        payload: dict[str, Any],
        epoch: int | None = None,
        nonce: str | None = None,
    ) -> dict[str, Any]:
        """Apply a guarded evidence journal update and verify the winner."""
        must: list[Any] = [
            HasIdCondition(has_id=[operation_id]),
            FieldCondition(
                key="request_fingerprint", match=MatchValue(value=request_fingerprint)
            ),
            FieldCondition(key="status", match=MatchValue(value=status)),
        ]
        if epoch is not None:
            must.append(
                FieldCondition(key="claim_epoch", match=MatchValue(value=epoch))
            )
        if nonce is not None:
            must.append(
                FieldCondition(key="claim_nonce", match=MatchValue(value=nonce))
            )
        else:
            must.append(IsEmptyCondition(is_empty=PayloadField(key="claim_nonce")))
        self._client.set_payload(
            collection_name=OPERATIONS_COLLECTION,
            payload=payload,
            points=Filter(must=must),
            ordering=WriteOrdering.STRONG,
            wait=True,
        )
        record = self._evidence_readback(operation_id)
        if record is None or not self._evidence_record_valid(record):
            raise EvidenceCorruptError("Evidence plan disappeared or became malformed")
        for key, value in payload.items():
            if record.get(key) != value:
                raise EvidenceConflictError(
                    "Evidence operation claim or checkpoint is stale",
                    code="evidence_claim_stale",
                )
        return record

    def claim_evidence_plan(
        self,
        operation_id: str,
        *,
        request_fingerprint: str,
        epoch: int,
        nonce: str,
    ) -> dict[str, Any]:
        """Claim a planned or expired evidence operation for 90 seconds."""
        record = self.get_evidence_plan(operation_id)
        if record is None:
            raise ValueError(f"Evidence plan not found: {operation_id}")
        if record["request_fingerprint"] != request_fingerprint:
            raise EvidenceConflictError("Evidence request fingerprint mismatch")
        current_epoch = int(record.get("claim_epoch", 0))
        if (
            record.get("status") == "claimed"
            and current_epoch == epoch
            and record.get("claim_nonce") == nonce
        ):
            return record
        if record["status"] == "claimed":
            deadline = record.get("claim_deadline_utc")
            try:
                active = deadline and datetime.fromisoformat(deadline) > datetime.now(
                    timezone.utc
                )
            except (TypeError, ValueError):
                active = False
            if active:
                raise EvidenceClaimActiveError(
                    "Evidence plan is actively claimed", code="evidence_claim_active"
                )
        if record["status"] not in {"planned", "claimed"} or epoch <= current_epoch:
            raise EvidenceConflictError("Evidence claim epoch is stale")
        deadline = (
            datetime.now(timezone.utc)
            + timedelta(seconds=EVIDENCE_CLAIM_DEADLINE_SECONDS)
        ).isoformat()
        previous_nonce = record.get("claim_nonce")
        return self._conditional_evidence_update(
            operation_id=operation_id,
            request_fingerprint=request_fingerprint,
            status=record["status"],
            epoch=current_epoch,
            nonce=previous_nonce,
            payload={
                "status": "claimed",
                "claim_epoch": epoch,
                "claim_nonce": nonce,
                "claim_deadline_utc": deadline,
            },
        )

    def verify_evidence_claim(
        self,
        operation_id: str,
        *,
        request_fingerprint: str,
        epoch: int,
        nonce: str,
    ) -> dict[str, Any]:
        """Require an exact, unexpired claim before any journal or target write."""
        record = self.get_evidence_plan(operation_id)
        if record is None:
            raise EvidenceLeaseLostError("Evidence operation is missing")
        if (
            record.get("request_fingerprint") != request_fingerprint
            or record.get("status") != "claimed"
            or record.get("claim_epoch") != epoch
            or record.get("claim_nonce") != nonce
        ):
            raise EvidenceLeaseLostError("Evidence lease is no longer owned")
        deadline = record.get("claim_deadline_utc")
        try:
            active = datetime.fromisoformat(deadline) > datetime.now(timezone.utc)
        except (TypeError, ValueError):
            active = False
        if not active:
            raise EvidenceLeaseLostError("Evidence lease has expired")
        return record

    def checkpoint_evidence_plan(
        self,
        operation_id: str,
        *,
        request_fingerprint: str,
        epoch: int,
        nonce: str,
        checkpoints: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Persist embedded per-target checkpoints under the active claim."""
        if len(checkpoints) > EVIDENCE_MAX_TARGETS:
            raise ValueError("Evidence checkpoints exceed the target limit")
        record = self.verify_evidence_claim(
            operation_id,
            request_fingerprint=request_fingerprint,
            epoch=epoch,
            nonce=nonce,
        )
        targets_by_ordinal = {
            int(target["ordinal"]): target for target in record["targets"]
        }
        if any(
            not isinstance(item, dict)
            or not isinstance(item.get("ordinal"), int)
            or item["ordinal"] not in targets_by_ordinal
            for item in checkpoints
        ):
            raise EvidenceConflictError("Checkpoint is outside the sealed plan")
        normalized_checkpoints: list[dict[str, Any]] = []
        for item in checkpoints:
            if isinstance(item, dict) and isinstance(item.get("ordinal"), int):
                target = targets_by_ordinal.get(item["ordinal"])
                if target is not None:
                    normalized_checkpoints.append(
                        {
                            "target_id": target.get("target_id"),
                            "action": target.get("action"),
                            **item,
                        }
                    )
        checkpoints = normalized_checkpoints
        if any(
            item.get("target_id")
            != targets_by_ordinal[item["ordinal"]].get("target_id")
            or item.get("action", targets_by_ordinal[item["ordinal"]].get("action"))
            != targets_by_ordinal[item["ordinal"]].get("action")
            for item in checkpoints
        ):
            raise EvidenceConflictError("Checkpoint is outside the sealed plan")
        return self._conditional_evidence_update(
            operation_id=operation_id,
            request_fingerprint=request_fingerprint,
            status="claimed",
            epoch=epoch,
            nonce=nonce,
            payload={"checkpoints": checkpoints},
        )

    def commit_evidence_plan(
        self,
        operation_id: str,
        *,
        request_fingerprint: str,
        epoch: int,
        nonce: str,
        result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Commit an evidence plan, including an empty all-skipped plan."""
        current = self.get_evidence_plan(operation_id)
        if current is None:
            raise ValueError(f"Evidence plan not found: {operation_id}")
        if current.get("status") == "committed":
            return current
        record = self.verify_evidence_claim(
            operation_id,
            request_fingerprint=request_fingerprint,
            epoch=epoch,
            nonce=nonce,
        )
        checkpoints = (result or {}).get("checkpoints", record.get("checkpoints", []))
        self._validate_terminal_checkpoints(record, checkpoints)
        return self._conditional_evidence_update(
            operation_id=operation_id,
            request_fingerprint=request_fingerprint,
            status="claimed",
            epoch=epoch,
            nonce=nonce,
            payload={
                "status": "committed",
                "result": result
                or {
                    "status": (
                        "skipped"
                        if all(
                            target.get("action") == "SKIP"
                            for target in record["targets"]
                        )
                        else "committed"
                    )
                },
            },
        )

    @staticmethod
    def _validate_terminal_checkpoints(
        record: dict[str, Any], checkpoints: Any
    ) -> None:
        """Require exactly one terminal checkpoint for every sealed target."""
        if not isinstance(checkpoints, list):
            raise EvidenceConflictError("Evidence checkpoints are malformed")
        targets = {
            int(item["ordinal"]): item
            for item in record.get("targets", [])
            if isinstance(item, dict) and isinstance(item.get("ordinal"), int)
        }
        if len(targets) != len(record.get("targets", [])) or len(checkpoints) != len(
            targets
        ):
            raise EvidenceConflictError("Evidence checkpoints are incomplete")
        seen: set[int] = set()
        for item in checkpoints:
            if not isinstance(item, dict) or item.get("status") not in {
                "confirmed",
                "skipped",
            }:
                raise EvidenceConflictError("Evidence checkpoint is not terminal")
            ordinal = item.get("ordinal")
            if (
                not isinstance(ordinal, int)
                or ordinal in seen
                or ordinal not in targets
                or item.get("target_id") != targets[ordinal].get("target_id")
                or item.get("action") != targets[ordinal].get("action")
            ):
                raise EvidenceConflictError("Evidence checkpoint is not bound")
            seen.add(ordinal)

    @staticmethod
    def _normalized_fact_hash(text: str) -> str:
        """Hash a whitespace- and case-normalized assertion."""
        return _normalized_fact_hash(text)

    def plan_evidence(
        self,
        claims: list[dict[str, Any]],
        *,
        user_id: str,
        owner_id: str,
        evidence_root_id: str,
        semantic_equivalence: Callable[[str, str, str | None], bool] | None = None,
    ) -> list[dict[str, Any]]:
        """Build a read-only, fail-closed CONFIRM/SKIP evidence plan.

        Candidate IDs are resolved directly from Qdrant.  This method never
        calls legacy materialization and never writes either memory or audit
        records.
        """
        if len(claims) > EVIDENCE_MAX_TARGETS:
            raise ValueError("Evidence plans may contain at most 32 targets")
        planned: list[dict[str, Any]] = []

        def evaluate(claim: dict[str, Any], target_id: Any) -> dict[str, Any]:
            target: dict[str, Any] = {
                "target_id": target_id,
                "action": "SKIP",
                "reason": "candidate_not_equivalent",
            }
            if not isinstance(target_id, str) or not target_id:
                return target
            result = self._client.retrieve(
                collection_name=self._memory_collection,
                ids=[target_id],
                with_payload=True,
                with_vectors=False,
                consistency="all",
            )
            point = result[0] if result else None
            if point is None:
                return target
            payload = dict(point.payload or {})
            stored_fact_hash = payload.get("fact_hash") or _normalized_fact_hash(
                str(payload.get("data", ""))
            )
            target.update(
                {
                    "lineage_id": payload.get("lineage_id"),
                    "revision_id": target_id,
                    "revision": payload.get("revision"),
                    "content_hash": payload.get("hash"),
                    "fact_hash": stored_fact_hash,
                }
            )
            layer = payload.get("memory_layer")
            derived_from = payload.get("derived_from")
            claim_text = (
                claim.get("assertion_text") or claim.get("text") or claim.get("content")
            )
            source_text = claim.get("source_text")
            expected_hashes = (claim.get("candidate_hashes") or {}).get(target_id, {})
            claim_fact_hash = claim.get("fact_hash")
            if claim_fact_hash is None and isinstance(claim_text, str):
                claim_fact_hash = _normalized_fact_hash(claim_text)
            claim_content_hash = claim.get("content_hash") or expected_hashes.get(
                "content_hash"
            )
            if claim_content_hash is None:
                claim_content_hash = payload.get("hash")
            equivalent = (
                payload.get("user_id") == user_id
                and payload.get("owner_id") == owner_id
                and payload.get("agent_id") is None
                and payload.get("role") == "user"
                and layer in {"raw", "consolidated"}
                and payload.get("revision_state") == ACTIVE_REVISION_STATE
                and isinstance(payload.get("lineage_id"), str)
                and isinstance(payload.get("revision"), int)
                and claim_content_hash == payload.get("hash")
                and claim_fact_hash == stored_fact_hash
            )
            if semantic_equivalence is not None and isinstance(source_text, str):
                equivalent = equivalent and semantic_equivalence(
                    source_text,
                    str(payload.get("data", "")),
                    claim.get("assertion_text"),
                )
            for field in ("lineage_id", "revision", "revision_id"):
                if field in claim and claim[field] != target.get(field):
                    equivalent = False
            if layer == "consolidated":
                if not isinstance(derived_from, list) or not derived_from:
                    equivalent = False
                if len(derived_from or []) > 1 and semantic_equivalence is None:
                    equivalent = False
                if len(claim.get("claims", [])) > 1 or claim.get("claim_count", 1) != 1:
                    equivalent = False
            if equivalent:
                target.update(
                    {
                        "action": "CONFIRM",
                        "reason": "exact_equivalence",
                        "evidence_root_id": evidence_root_id,
                        "_derived_from": derived_from or [],
                        "_layer": layer,
                        "support_ids": list(derived_from or []),
                        "source_hash": (
                            hashlib.sha256(
                                claim["assertion_text"].encode("utf-8")
                            ).hexdigest()
                            if isinstance(claim.get("assertion_text"), str)
                            else None
                        ),
                    }
                )
            return target

        for claim in claims:
            claim = dict(claim)
            candidate_ids = claim.get("candidate_ids") or claim.get("candidates")
            if not candidate_ids:
                candidate_ids = [
                    claim.get("candidate_id")
                    or claim.get("target_id")
                    or claim.get("memory_id")
                ]
            if isinstance(candidate_ids, dict):
                candidate_ids = [candidate_ids]
            evaluated = [
                evaluate(
                    claim,
                    (
                        item.get("candidate_id") or item.get("id")
                        if isinstance(item, dict)
                        else item
                    ),
                )
                for item in candidate_ids
            ]
            eligible = [item for item in evaluated if item["action"] == "CONFIRM"]
            consolidated = [
                item for item in eligible if item["_layer"] == "consolidated"
            ]
            if consolidated:
                covered = {
                    source for item in consolidated for source in item["_derived_from"]
                }
                eligible = [
                    item
                    for item in eligible
                    if item["_layer"] == "consolidated"
                    or item["target_id"] not in covered
                ]
            eligible.sort(
                key=lambda item: (
                    0 if item["_layer"] == "consolidated" else 1,
                    str(item.get("lineage_id") or ""),
                    -int(item.get("revision") or 0),
                    str(item.get("target_id") or ""),
                )
            )
            target = eligible[0] if eligible else evaluated[0]
            planned.append(target)
        # Never emit two confirmations for one lineage, even when extraction
        # produced duplicate assertions.
        seen_lineages: set[str] = set()
        covered_targets = {
            source
            for target in planned
            if target.get("action") == "CONFIRM"
            and target.get("_layer") == "consolidated"
            for source in target.get("_derived_from", [])
        }
        for target in planned:
            lineage = target.get("lineage_id")
            if (
                target.get("action") == "CONFIRM"
                and target.get("_layer") == "raw"
                and target.get("target_id") in covered_targets
            ):
                target["action"] = "SKIP"
                target["reason"] = "covered_by_consolidated"
            elif target.get("action") == "CONFIRM" and lineage in seen_lineages:
                target["action"] = "SKIP"
                target["reason"] = "duplicate_lineage"
            elif target.get("action") == "CONFIRM" and isinstance(lineage, str):
                seen_lineages.add(lineage)
        families: dict[tuple[str, ...], list[dict[str, Any]]] = {}
        for target in planned:
            if target.get("action") != "CONFIRM":
                continue
            support = target.get("_derived_from") or [target.get("target_id")]
            family = tuple(sorted(str(item) for item in support if item))
            families.setdefault(family, []).append(target)
        for family_targets in families.values():
            winner = min(
                family_targets,
                key=lambda item: (
                    0 if item.get("_layer") == "consolidated" else 1,
                    str(item.get("lineage_id") or ""),
                    -int(item.get("revision") or 0),
                    str(item.get("target_id") or ""),
                ),
            )
            for target in family_targets:
                if target is not winner:
                    target["action"] = "SKIP"
                    target["reason"] = "duplicate_provenance_family"
        for target in planned:
            target.pop("_derived_from", None)
            target.pop("_layer", None)
        planned.sort(
            key=lambda item: (
                str(item.get("lineage_id") or ""),
                int(item.get("revision") or 0),
                str(item.get("target_id") or ""),
            )
        )
        for ordinal, target in enumerate(planned):
            target["ordinal"] = ordinal
        encoded_size = len(
            json.dumps(planned, sort_keys=True, separators=(",", ":")).encode()
        )
        if encoded_size > EVIDENCE_MAX_PLAN_BYTES:
            raise ValueError("Evidence plan exceeds 64 KiB")
        return planned

    def get(self, token: str) -> dict[str, Any] | None:
        """Get an operation by its transition token."""
        result = self._client.retrieve(
            collection_name=OPERATIONS_COLLECTION,
            ids=[_operation_point_id(token)],
            with_payload=True,
            with_vectors=False,
        )
        if not result:
            return None
        return dict(result[0].payload or {})

    def get_by_id(self, operation_id: str) -> dict[str, Any] | None:
        """Get an operation by its persisted Qdrant point ID."""
        result = self._client.retrieve(
            collection_name=OPERATIONS_COLLECTION,
            ids=[operation_id],
            with_payload=True,
            with_vectors=False,
        )
        return dict(result[0].payload or {}) if result else None

    def claim(
        self,
        token: str,
        *,
        claimant: str,
        lease_seconds: int,
        allowed_statuses: tuple[str, ...] = ("planned", "applying"),
    ) -> bool:
        """Claim a recoverable operation with a bounded lease."""
        with _OPERATION_CLAIM_LOCK:
            return self._claim_locked(
                token,
                claimant=claimant,
                lease_seconds=lease_seconds,
                allowed_statuses=allowed_statuses,
            )

    def _claim_locked(
        self,
        token: str,
        *,
        claimant: str,
        lease_seconds: int,
        allowed_statuses: tuple[str, ...],
    ) -> bool:
        """Claim an operation while serializing in-process contenders."""
        operation_id = _operation_point_id(token)
        now = datetime.now(timezone.utc)
        existing = self.get(token)
        if existing is None or existing.get("status") not in allowed_statuses:
            return False
        self._client.set_payload(
            collection_name=OPERATIONS_COLLECTION,
            payload={
                "status": "applying",
                "recovery_token": claimant,
                "recovery_started_at": now.isoformat(),
                "lease_expires_at": (
                    now + timedelta(seconds=max(lease_seconds, 1))
                ).isoformat(),
                "recovery_attempt_count": int(existing.get("recovery_attempt_count", 0))
                + 1,
            },
            points=Filter(
                must=[
                    HasIdCondition(has_id=[operation_id]),
                    FieldCondition(
                        key="status",
                        match=MatchAny(any=list(allowed_statuses)),
                    ),
                ],
                should=[
                    IsEmptyCondition(is_empty=PayloadField(key="recovery_token")),
                    FieldCondition(
                        key="recovery_token", match=MatchValue(value=claimant)
                    ),
                    FieldCondition(
                        key="lease_expires_at",
                        range=DatetimeRange(lte=now),
                    ),
                ],
            ),
            wait=True,
        )
        claimed = self.get(token)
        return bool(
            claimed
            and claimed.get("status") == "applying"
            and claimed.get("recovery_token") == claimant
        )

    def terminalize_unclaimed(
        self,
        token: str,
        *,
        status: str,
        payload: dict[str, Any],
        allowed_statuses: tuple[str, ...] = ("planned", "failed"),
    ) -> bool:
        """Terminalize an operation only when no worker owns a live lease."""
        operation_id = _operation_point_id(token)
        now = datetime.now(timezone.utc)
        self._client.set_payload(
            collection_name=OPERATIONS_COLLECTION,
            payload={
                **payload,
                "status": status,
                "terminal_at": now.isoformat(),
                "updated_at_utc": now.isoformat(),
            },
            points=Filter(
                must=[
                    HasIdCondition(has_id=[operation_id]),
                    FieldCondition(
                        key="status", match=MatchAny(any=list(allowed_statuses))
                    ),
                ],
                should=[
                    IsEmptyCondition(is_empty=PayloadField(key="recovery_token")),
                    FieldCondition(
                        key="lease_expires_at",
                        range=DatetimeRange(lte=now),
                    ),
                ],
            ),
            wait=True,
        )
        current = self.get(token)
        return bool(current and current.get("status") == status)

    def write_claimed(self, token: str, claimant: str, payload: dict[str, Any]) -> bool:
        """Update an applying operation only while the caller owns its lease."""
        operation_id = _operation_point_id(token)
        now = datetime.now(timezone.utc)
        timestamp = now.isoformat()
        self._client.set_payload(
            collection_name=OPERATIONS_COLLECTION,
            payload={**payload, "updated_at_utc": timestamp},
            points=Filter(
                must=[
                    HasIdCondition(has_id=[operation_id]),
                    FieldCondition(key="status", match=MatchValue(value="applying")),
                    FieldCondition(
                        key="recovery_token", match=MatchValue(value=claimant)
                    ),
                    FieldCondition(
                        key="lease_expires_at", range=DatetimeRange(gte=now)
                    ),
                ]
            ),
            wait=True,
        )
        current = self.get(token)
        return bool(
            current
            and current.get("recovery_token") == claimant
            and current.get("updated_at_utc") == timestamp
        )

    def renew_claim(
        self,
        token: str,
        claimant: str,
        *,
        lease_seconds: int,
    ) -> bool:
        """Renew an applying operation lease only for its current owner."""
        operation_id = _operation_point_id(token)
        now = datetime.now(timezone.utc)
        lease_expires_at = now + timedelta(seconds=max(lease_seconds, 1))
        timestamp = now.isoformat()
        lease_timestamp = lease_expires_at.isoformat()
        self._client.set_payload(
            collection_name=OPERATIONS_COLLECTION,
            payload={
                "lease_expires_at": lease_timestamp,
                "updated_at_utc": timestamp,
            },
            points=Filter(
                must=[
                    HasIdCondition(has_id=[operation_id]),
                    FieldCondition(key="status", match=MatchValue(value="applying")),
                    FieldCondition(
                        key="recovery_token", match=MatchValue(value=claimant)
                    ),
                    FieldCondition(
                        key="lease_expires_at",
                        range=DatetimeRange(gte=now),
                    ),
                ]
            ),
            wait=True,
        )
        current = self.get(token)
        return bool(
            current
            and current.get("status") == "applying"
            and current.get("recovery_token") == claimant
            and current.get("lease_expires_at") == lease_timestamp
            and current.get("updated_at_utc") == timestamp
        )

    def list_for_lineage(
        self,
        *,
        lineage_id: str,
        user_id: str,
        owner_id: str,
    ) -> list[dict[str, Any]]:
        """Return ordered audit records for one authorized lineage."""
        points = []
        offset = None
        while True:
            page, next_offset = self._client.scroll(
                collection_name=OPERATIONS_COLLECTION,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="lineage_id", match=MatchValue(value=lineage_id)
                        ),
                        FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                        FieldCondition(
                            key="owner_id", match=MatchValue(value=owner_id)
                        ),
                    ]
                ),
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            points.extend(page)
            if next_offset is None:
                break
            offset = next_offset
        records = [dict(point.payload or {}) for point in points]
        records.sort(key=lambda item: item.get("created_at_utc", ""))
        return records

    def list_for_lineage_page(
        self,
        *,
        lineage_id: str,
        user_id: str,
        owner_id: str,
        agent_id: str | None,
        before_created_at: str | None,
        limit: int,
    ) -> dict[str, Any]:
        """Return one bounded newest-first page of lineage operations."""
        agent_condition: Any
        if agent_id:
            agent_condition = FieldCondition(
                key="agent_id", match=MatchValue(value=agent_id)
            )
        else:
            agent_condition = IsEmptyCondition(is_empty=PayloadField(key="agent_id"))
        must: list[Any] = [
            FieldCondition(key="lineage_id", match=MatchValue(value=lineage_id)),
            FieldCondition(key="user_id", match=MatchValue(value=user_id)),
            FieldCondition(key="owner_id", match=MatchValue(value=owner_id)),
            agent_condition,
        ]
        if before_created_at:
            must.append(
                FieldCondition(
                    key="created_at_utc",
                    range=DatetimeRange(lt=before_created_at),
                )
            )
        points, _ = self._client.scroll(
            collection_name=OPERATIONS_COLLECTION,
            scroll_filter=Filter(must=must),
            limit=limit + 1,
            order_by=OrderBy(
                key="created_at_utc",
                direction=Direction.DESC,
            ),
            with_payload=True,
            with_vectors=False,
        )
        records = [dict(point.payload or {}) for point in points]
        has_more = len(records) > limit
        records = records[:limit]
        next_before = (
            records[-1].get("created_at_utc") if has_more and records else None
        )
        records.reverse()
        return {"operations": records, "next_before": next_before}

    def delete_lineage(
        self,
        *,
        lineage_id: str,
        user_id: str,
        owner_id: str,
    ) -> None:
        """Physically erase audit records for a privacy-deleted lineage."""
        self._client.delete(
            collection_name=OPERATIONS_COLLECTION,
            points_selector=Filter(
                must=[
                    FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                    FieldCondition(key="owner_id", match=MatchValue(value=owner_id)),
                ],
                should=[
                    FieldCondition(
                        key="lineage_id", match=MatchValue(value=lineage_id)
                    ),
                    FieldCondition(
                        key="affected_lineage_ids",
                        match=MatchValue(value=lineage_id),
                    ),
                ],
            ),
            wait=True,
        )

    def find_privacy_erase(self, memory_id: str) -> dict[str, Any] | None:
        """Find a resumable privacy operation by any planned revision ID."""
        points, _ = self._client.scroll(
            collection_name=OPERATIONS_COLLECTION,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="operation_kind",
                        match=MatchValue(value="privacy_erase"),
                    ),
                    FieldCondition(
                        key="target_revision_ids",
                        match=MatchValue(value=memory_id),
                    ),
                ]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        return dict(points[0].payload or {}) if points else None

    def list_fsck(
        self,
        check_id: str,
        *,
        user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return durable issue plans for one fsck check."""
        must: list[Any] = [
            FieldCondition(
                key="fsck_check_id",
                match=MatchValue(value=check_id),
            )
        ]
        if user_id is not None:
            must.append(FieldCondition(key="user_id", match=MatchValue(value=user_id)))
        records: list[dict[str, Any]] = []
        offset = None
        while True:
            points, next_offset = self._client.scroll(
                collection_name=OPERATIONS_COLLECTION,
                scroll_filter=Filter(must=must),
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            records.extend(dict(point.payload or {}) for point in points)
            if next_offset is None:
                break
            offset = next_offset
        records.sort(key=lambda item: item.get("fsck_issue_id", ""))
        return records


class RevisionService:
    """Own immutable memory revisions and guarded lifecycle transitions."""

    def __init__(
        self,
        vector: Any,
        *,
        sparse_embed: Callable[[str], Any] | None = None,
    ) -> None:
        self._vector = vector
        self._client = vector._client
        self._is_remote = vector._config.vector.is_remote
        self._sparse_embed = sparse_embed
        self.operations = RevisionOperationStore(
            self._client,
            is_remote=self._is_remote,
            memory_collection=self._vector.collection_name,
        )

    @staticmethod
    def _normalized_fact_hash(text: str) -> str:
        """Hash a whitespace- and case-normalized assertion."""
        return _normalized_fact_hash(text)

    def plan_evidence(
        self,
        claims: list[dict[str, Any]],
        *,
        user_id: str,
        owner_id: str,
        evidence_root_id: str,
        semantic_equivalence: Callable[[str, str, str | None], bool] | None = None,
    ) -> list[dict[str, Any]]:
        """Build a read-only evidence plan through the lineage service."""
        return self.operations.plan_evidence(
            claims,
            user_id=user_id,
            owner_id=owner_id,
            evidence_root_id=evidence_root_id,
            semantic_equivalence=semantic_equivalence,
        )

    def verify_evidence_target(
        self,
        target: dict[str, Any],
        *,
        user_id: str,
        owner_id: str,
    ) -> bool:
        """Verify a sealed target without rerunning semantic planning."""
        target_id = target.get("target_id")
        if not isinstance(target_id, str):
            return False
        point = self._read_point(target_id, with_vectors=False)
        if point is None:
            return False
        payload = self._payload(point)
        if (
            payload.get("user_id") != user_id
            or payload.get("owner_id") != owner_id
            or payload.get("agent_id") is not None
            or payload.get("role") != "user"
            or payload.get("memory_layer") not in {"raw", "consolidated"}
            or payload.get("revision_state") != ACTIVE_REVISION_STATE
            or str(point.id) != target.get("revision_id")
            or payload.get("lineage_id") != target.get("lineage_id")
            or payload.get("revision") != target.get("revision")
            or payload.get("hash") != target.get("content_hash")
        ):
            return False
        fact_hash = payload.get("fact_hash") or _normalized_fact_hash(
            str(payload.get("data", ""))
        )
        if fact_hash != target.get("fact_hash"):
            return False
        if payload.get("memory_layer") == "consolidated":
            derived_from = payload.get("derived_from")
            if (
                not isinstance(derived_from, list)
                or not derived_from
                or target.get("support_ids") != derived_from
            ):
                return False
        return True

    @staticmethod
    def initial_metadata(
        memory_id: str,
        *,
        operation_id: str | None = None,
        derived_from: list[str] | None = None,
        source_session_id: str | None = None,
        provenance_quality: str = "exact",
    ) -> dict[str, Any]:
        """Build revision metadata for a new lineage."""
        metadata: dict[str, Any] = {
            "lineage_id": memory_id,
            "revision": 1,
            "revision_state": ACTIVE_REVISION_STATE,
            "revision_created_at_utc": datetime.now(timezone.utc).isoformat(),
            "validation_state": "unverified",
            "provenance_quality": provenance_quality,
        }
        if operation_id:
            metadata["revision_operation_id"] = operation_id
        if derived_from:
            metadata["derived_from"] = list(dict.fromkeys(derived_from))
        if source_session_id:
            metadata["source_session_id"] = source_session_id
        return metadata

    def _read_point(
        self,
        memory_id: str,
        *,
        with_vectors: bool = True,
    ) -> Any | None:
        kwargs: dict[str, Any] = {
            "collection_name": self._vector.collection_name,
            "ids": [memory_id],
            "with_payload": True,
            "with_vectors": with_vectors,
        }
        if self._is_remote:
            kwargs["consistency"] = "all"
        result = self._client.retrieve(**kwargs)
        return result[0] if result else None

    @staticmethod
    def _payload(point: Any) -> dict[str, Any]:
        return dict(point.payload or {})

    def _materialize_legacy(self, point: Any) -> dict[str, Any]:
        payload = self._payload(point)
        additions: dict[str, Any] = {}
        point_id = str(point.id)
        if not payload.get("lineage_id"):
            additions["lineage_id"] = point_id
        if not isinstance(payload.get("revision"), int):
            additions["revision"] = 1
        if not payload.get("revision_state"):
            additions["revision_state"] = ACTIVE_REVISION_STATE
        if not payload.get("revision_created_at_utc"):
            additions["revision_created_at_utc"] = (
                payload.get("created_at_utc")
                or payload.get("created_at")
                or datetime.now(timezone.utc).isoformat()
            )
        if not payload.get("validation_state"):
            additions["validation_state"] = "unverified"
        if not payload.get("provenance_quality"):
            additions["provenance_quality"] = (
                "legacy_batch" if payload.get("derived_from") else "exact"
            )
        if additions:
            self._client.set_payload(
                collection_name=self._vector.collection_name,
                payload=additions,
                points=[point_id],
                wait=True,
            )
            payload.update(additions)
        return payload

    def _lineage_points(self, lineage_id: str) -> list[Any]:
        points = []
        offset = None
        while True:
            page, next_offset = self._client.scroll(
                collection_name=self._vector.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="lineage_id", match=MatchValue(value=lineage_id)
                        )
                    ]
                ),
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            points.extend(page)
            if next_offset is None:
                break
            offset = next_offset
        return points

    def _recover_claimed(self, point: Any, payload: dict[str, Any]) -> None:
        """Complete a claimed transition from durable Qdrant state."""
        token = payload.get("transition_token")
        kind = payload.get("transition_kind")
        if not token:
            return
        point_id = str(point.id)
        if kind == "privacy_erase":
            raise RevisionConflictError(
                "The memory lineage is being erased",
                lineage_id=payload.get("lineage_id"),
                current_revision_id=point_id,
                current_revision=int(payload.get("revision", 1)),
            )
        if kind == "retract":
            operation = self.operations.get(token) or {}
            operation_id = operation.get("operation_id") or _operation_point_id(token)
            key_hash = operation.get("idempotency_key_hash")
            fingerprint = operation.get("request_fingerprint")
            self._client.set_payload(
                collection_name=self._vector.collection_name,
                payload={
                    "revision_state": "retracted",
                    "state_reason": operation.get("reason", "deleted"),
                    "revision_operation_id": operation_id,
                    "revision_operation_key_hash": key_hash,
                    "revision_operation_fingerprint": fingerprint,
                },
                points=[point_id],
                wait=True,
            )
            result = self._result(
                operation_id=operation_id,
                lineage_id=payload["lineage_id"],
                previous_revision_id=point_id,
                revision_id=None,
                revision=int(payload.get("revision", 1)),
                status="retracted",
                replayed=True,
            )
            self.operations.write(
                token,
                {**operation, "status": "committed", "result": result},
            )
            return
        if kind in {"artifact_save", "artifact_delete"}:
            operation = self.operations.get(token) or {}
            operation_id = operation.get("operation_id") or _operation_point_id(token)
            artifacts = operation.get("artifacts")
            if not isinstance(artifacts, list):
                raise RevisionConflictError(
                    "The memory has an incomplete artifact transition",
                    lineage_id=payload.get("lineage_id"),
                    current_revision_id=point_id,
                    current_revision=int(payload.get("revision", 1)),
                )
            self._client.set_payload(
                collection_name=self._vector.collection_name,
                payload={
                    "artifacts": artifacts,
                    "artifact_revision": int(
                        operation.get(
                            "expected_artifact_revision",
                            payload.get("artifact_revision", 0),
                        )
                    )
                    + 1,
                },
                points=[point_id],
                wait=True,
            )
            result = {
                "status": kind.removeprefix("artifact_"),
                "memory_id": point_id,
                "operation_id": operation_id,
                "lineage_id": payload["lineage_id"],
                "revision": int(payload.get("revision", 1)),
                "artifact_revision": int(
                    operation.get(
                        "expected_artifact_revision",
                        payload.get("artifact_revision", 0),
                    )
                )
                + 1,
            }
            self.operations.write(
                token,
                {
                    **operation,
                    "status": "committed",
                    "result": result,
                },
            )
            self._release_claim(point_id, token)
            return
        if kind == "confirm":
            operation_token = payload.get("transition_operation_token") or token
            operation = self.operations.get(operation_token) or {}
            child_epoch = operation.get("child_claim_epoch")
            child_nonce = operation.get("child_claim_nonce")
            child_fence = operation.get("child_transition_token")
            if operation.get("parent_operation_id") is not None:
                if not isinstance(child_epoch, int) or not isinstance(child_nonce, str):
                    raise EvidenceLeaseLostError(
                        "Evidence child recovery lease is missing"
                    )
                self.operations.verify_evidence_child(
                    operation_token,
                    epoch=child_epoch,
                    nonce=child_nonce,
                )
                if payload.get("evidence_child_fence") != child_fence:
                    raise EvidenceLeaseLostError(
                        "Evidence child recovery fence is stale"
                    )
            projection = operation.get("validation_projection")
            if not isinstance(projection, dict):
                raise RevisionConflictError(
                    "The memory has an incomplete confirmation transition",
                    lineage_id=payload.get("lineage_id"),
                    current_revision_id=point_id,
                    current_revision=int(payload.get("revision", 1)),
                )
            recovery_filter = [
                HasIdCondition(has_id=[point_id]),
                FieldCondition(
                    key="transition_token",
                    match=MatchValue(value=payload.get("transition_token") or token),
                ),
            ]
            if operation.get("parent_operation_id") is not None:
                recovery_filter.append(
                    FieldCondition(
                        key="evidence_child_fence",
                        match=MatchValue(value=child_fence),
                    )
                )
            recovery_kwargs: dict[str, Any] = {
                "collection_name": self._vector.collection_name,
                "payload": projection,
                "points": Filter(must=recovery_filter),
                "wait": True,
            }
            if self._is_remote:
                recovery_kwargs["ordering"] = WriteOrdering.STRONG
            self._client.set_payload(**recovery_kwargs)
            recovered = self._read_point(point_id, with_vectors=False)
            if recovered is None or any(
                self._payload(recovered).get(key) != value
                for key, value in projection.items()
            ):
                raise EvidenceLeaseLostError(
                    "Evidence recovery projection lost its fence"
                )
            result = operation.get("result") or {
                "operation_id": operation.get("operation_id")
                or _operation_point_id(token),
                "lineage_id": payload["lineage_id"],
                "revision_id": point_id,
                "revision": int(payload.get("revision", 1)),
                "status": "confirmed",
                "replayed": True,
            }
            if operation.get("parent_operation_id") is not None:
                self.operations.complete_evidence_child(
                    operation_token,
                    epoch=int(child_epoch),
                    nonce=str(child_nonce),
                    result=result,
                )
            else:
                self.operations.write(
                    operation_token,
                    {**operation, "status": "committed", "result": result},
                )
            self._release_claim(point_id, token)
            return
        successor_id = payload.get("transition_successor_id")
        successor = self._read_point(successor_id) if successor_id else None
        if successor is None:
            raise RevisionConflictError(
                "The memory has an incomplete revision transition",
                lineage_id=payload.get("lineage_id"),
                current_revision_id=point_id,
                current_revision=int(payload.get("revision", 1)),
            )
        if self._payload(successor).get("revision_state") == "pending":
            self._client.set_payload(
                collection_name=self._vector.collection_name,
                payload={"revision_state": ACTIVE_REVISION_STATE},
                points=[successor_id],
                wait=True,
            )
        self._client.set_payload(
            collection_name=self._vector.collection_name,
            payload={
                "revision_state": "superseded",
                "revision_successor_id": successor_id,
            },
            points=[point_id],
            wait=True,
        )

    def _release_claim(self, point_id: str, token: str) -> None:
        """Release a matching lifecycle claim without changing revision state."""
        point = self._read_point(point_id, with_vectors=False)
        if point is None or self._payload(point).get("transition_token") != token:
            return
        self._client.delete_payload(
            collection_name=self._vector.collection_name,
            keys=list(_TRANSITION_FIELDS),
            points=[point_id],
            wait=True,
        )

    def current(
        self,
        memory_id: str,
        *,
        user_id: str,
        owner_id: str,
        session_agent_id: str | None,
        expected_revision: int | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Return the authorized active revision and its payload."""
        current, payload, _ = self._resolve(memory_id)
        self._authorize(
            payload,
            user_id=user_id,
            owner_id=owner_id,
            session_agent_id=session_agent_id,
        )
        if payload.get("transition_token"):
            self._recover_claimed(current, payload)
            current, payload, _ = self._resolve(memory_id)
        if payload.get("revision_state") != ACTIVE_REVISION_STATE:
            self._raise_stale(payload, str(current.id))
        if (
            expected_revision is not None
            and int(payload.get("revision", 1)) != expected_revision
        ):
            self._raise_stale(payload, str(current.id))
        return str(current.id), payload

    def artifact_source(
        self,
        memory_id: str,
        *,
        user_id: str,
        owner_id: str,
        session_agent_id: str | None,
    ) -> dict[str, Any]:
        """Return an authorized exact revision for artifact reads.

        Active revisions and consolidation source revisions can retain artifact
        references. Other historical states remain inaccessible.
        """
        point = self._read_point(memory_id, with_vectors=False)
        if point is None:
            raise ValueError(f"Memory not found: {memory_id}")
        payload = self._materialize_legacy(point)
        self._authorize(
            payload,
            user_id=user_id,
            owner_id=owner_id,
            session_agent_id=session_agent_id,
        )
        if payload.get("revision_state") not in {ACTIVE_REVISION_STATE, "source"}:
            self._raise_stale(payload, str(point.id))
        return payload

    def set_artifacts(
        self,
        memory_id: str,
        *,
        user_id: str,
        owner_id: str,
        session_agent_id: str | None,
        artifact: dict[str, Any] | None = None,
        remove_artifact_id: str | None = None,
        operation_kind: str,
        idempotency_key: str,
        expected_revision: int | None = None,
        _attempt: int = 0,
    ) -> dict[str, Any]:
        """Apply an artifact annotation under the revision transition guard."""
        current, payload, _ = self._resolve(memory_id)
        self._authorize(
            payload,
            user_id=user_id,
            owner_id=owner_id,
            session_agent_id=session_agent_id,
        )
        if payload.get("transition_token"):
            self._recover_claimed(current, payload)
            current, payload, _ = self._resolve(memory_id)
        current_id = str(current.id)
        current_revision = int(payload.get("revision", 1))
        if expected_revision is not None and expected_revision != current_revision:
            self._raise_stale(payload, current_id)
        persisted_artifacts = list(payload.get("artifacts") or [])
        artifacts = list(persisted_artifacts)
        if artifact is not None and not any(
            item.get("id") == artifact.get("id") for item in artifacts
        ):
            artifacts.append(artifact)
        if remove_artifact_id is not None:
            artifacts = [
                item for item in artifacts if item.get("id") != remove_artifact_id
            ]
        fingerprint = canonical_fingerprint(
            [
                operation_kind,
                current_id,
                artifact,
                remove_artifact_id,
                expected_revision,
            ]
        )
        key_hash = canonical_fingerprint(
            [user_id, owner_id, payload["lineage_id"], operation_kind, idempotency_key]
        )
        token = canonical_fingerprint([key_hash, fingerprint])
        existing_operation = self.operations.get(token)
        if existing_operation and existing_operation.get("status") == "committed":
            return {**existing_operation["result"], "replayed": True}
        desired_present = bool(
            artifact
            and any(
                item.get("id") == artifact.get("id") for item in persisted_artifacts
            )
        )
        desired_absent = bool(
            remove_artifact_id
            and not any(
                item.get("id") == remove_artifact_id for item in persisted_artifacts
            )
        )
        if existing_operation and (desired_present or desired_absent):
            result = {
                "status": operation_kind.removeprefix("artifact_"),
                "memory_id": current_id,
                "operation_id": existing_operation["operation_id"],
                "lineage_id": payload["lineage_id"],
                "revision": current_revision,
                "artifact_revision": int(payload.get("artifact_revision", 0)),
                "replayed": True,
            }
            self.operations.write(
                token,
                {
                    **existing_operation,
                    "status": "committed",
                    "result": result,
                },
            )
            return result
        artifact_revision = int(payload.get("artifact_revision", 0))
        operation = {
            "status": "prepared",
            "operation_kind": operation_kind,
            "actor_kind": "artifact",
            "user_id": user_id,
            "owner_id": owner_id,
            "agent_id": payload.get("agent_id"),
            "lineage_id": payload["lineage_id"],
            "idempotency_key_hash": key_hash,
            "request_fingerprint": fingerprint,
            "previous_revision_id": current_id,
            "artifacts": artifacts,
            "artifact_id": (artifact.get("id") if artifact else remove_artifact_id),
            "expected_artifact_revision": artifact_revision,
            "expected_revision": expected_revision,
        }
        operation_id = self.operations.write(token, operation)
        try:
            self._claim(
                point_id=current_id,
                payload=payload,
                token=token,
                kind=operation_kind,
                successor_id=None,
                expected_artifact_revision=artifact_revision,
            )
        except RevisionConflictError:
            if _attempt >= 7:
                raise
            return self.set_artifacts(
                memory_id,
                user_id=user_id,
                owner_id=owner_id,
                session_agent_id=session_agent_id,
                artifact=artifact,
                remove_artifact_id=remove_artifact_id,
                operation_kind=operation_kind,
                idempotency_key=idempotency_key,
                expected_revision=expected_revision,
                _attempt=_attempt + 1,
            )
        self.operations.write(token, {**operation, "status": "claimed"})
        self._client.set_payload(
            collection_name=self._vector.collection_name,
            payload={
                "artifacts": artifacts,
                "artifact_revision": artifact_revision + 1,
            },
            points=Filter(
                must=[
                    HasIdCondition(has_id=[current_id]),
                    FieldCondition(
                        key="transition_token", match=MatchValue(value=token)
                    ),
                ]
            ),
            wait=True,
        )
        result = {
            "status": operation_kind.removeprefix("artifact_"),
            "memory_id": current_id,
            "operation_id": operation_id,
            "lineage_id": payload["lineage_id"],
            "revision": current_revision,
            "artifact_revision": artifact_revision + 1,
        }
        self.operations.write(
            token,
            {**operation, "status": "committed", "result": result},
        )
        self._release_claim(current_id, token)
        return result

    def remove_artifact_from_lineage(
        self,
        memory_id: str,
        artifact_id: str,
        *,
        user_id: str,
        owner_id: str,
        session_agent_id: str | None,
    ) -> list[str]:
        """Remove one artifact reference from every authorized lineage revision."""
        _, payload, points = self._resolve(memory_id)
        revision_ids = []
        for point in points:
            point_payload = self._payload(point)
            if not any(
                item.get("id") == artifact_id
                for item in point_payload.get("artifacts", [])
            ):
                revision_ids.append(str(point.id))
                continue
            self._authorize(
                point_payload,
                user_id=user_id,
                owner_id=owner_id,
                session_agent_id=session_agent_id,
            )
            artifacts = [
                item
                for item in point_payload.get("artifacts", [])
                if item.get("id") != artifact_id
            ]
            self._client.set_payload(
                collection_name=self._vector.collection_name,
                payload={"artifacts": artifacts},
                points=[str(point.id)],
                wait=True,
            )
            revision_ids.append(str(point.id))
        if not revision_ids:
            revision_ids.append(payload["lineage_id"])
        return revision_ids

    def _resolve(self, memory_id: str) -> tuple[Any, dict[str, Any], list[Any]]:
        point = self._read_point(memory_id)
        if point is None:
            raise ValueError(f"Memory {memory_id} not found")
        payload = self._materialize_legacy(point)
        lineage_id = payload["lineage_id"]
        points = self._lineage_points(lineage_id)
        if not points:
            points = [self._read_point(memory_id)]
        active = [
            item
            for item in points
            if self._payload(item).get("revision_state", ACTIVE_REVISION_STATE)
            == ACTIVE_REVISION_STATE
        ]
        if active:
            point = max(
                active,
                key=lambda item: int(self._payload(item).get("revision", 1)),
            )
            payload = self._materialize_legacy(point)
        return point, payload, [item for item in points if item is not None]

    @staticmethod
    def _authorize(
        payload: dict[str, Any],
        *,
        user_id: str,
        owner_id: str,
        session_agent_id: str | None,
    ) -> None:
        if payload.get("user_id") != user_id or payload.get("owner_id") != owner_id:
            raise ValueError("Cannot access memory")
        memory_agent_id = payload.get("agent_id")
        if (
            session_agent_id
            and memory_agent_id
            and memory_agent_id != session_agent_id
            and not memory_agent_id.startswith(session_agent_id + ":")
        ):
            raise ValueError("Cannot access memory")

    @classmethod
    def _is_authorized(
        cls,
        payload: dict[str, Any],
        *,
        user_id: str,
        owner_id: str,
        session_agent_id: str | None,
    ) -> bool:
        try:
            cls._authorize(
                payload,
                user_id=user_id,
                owner_id=owner_id,
                session_agent_id=session_agent_id,
            )
        except ValueError:
            return False
        return True

    @staticmethod
    def _result(
        *,
        operation_id: str,
        lineage_id: str,
        previous_revision_id: str,
        revision_id: str | None,
        revision: int,
        status: str,
        replayed: bool,
    ) -> dict[str, Any]:
        return {
            "operation_id": operation_id,
            "lineage_id": lineage_id,
            "previous_revision_id": previous_revision_id,
            "revision_id": revision_id,
            "revision": revision,
            "status": status,
            "replayed": replayed,
        }

    def _raise_stale(self, payload: dict[str, Any], point_id: str) -> None:
        raise RevisionConflictError(
            "The memory revision is stale",
            lineage_id=payload.get("lineage_id"),
            current_revision_id=point_id,
            current_revision=int(payload.get("revision", 1)),
        )

    def _claim(
        self,
        *,
        point_id: str,
        payload: dict[str, Any],
        token: str,
        kind: str,
        successor_id: str | None,
        expected_artifact_revision: int | None = None,
        expected_child_token: str | None = None,
        transition_operation_token: str | None = None,
    ) -> dict[str, Any]:
        conditions: list[Any] = [
            HasIdCondition(has_id=[point_id]),
            FieldCondition(key="user_id", match=MatchValue(value=payload["user_id"])),
            FieldCondition(key="owner_id", match=MatchValue(value=payload["owner_id"])),
            FieldCondition(
                key="revision_state", match=MatchValue(value=ACTIVE_REVISION_STATE)
            ),
            FieldCondition(
                key="revision", match=MatchValue(value=int(payload["revision"]))
            ),
            IsEmptyCondition(is_empty=PayloadField(key="transition_token")),
        ]
        if payload.get("agent_id"):
            conditions.append(
                FieldCondition(
                    key="agent_id", match=MatchValue(value=payload["agent_id"])
                )
            )
        else:
            conditions.append(IsEmptyCondition(is_empty=PayloadField(key="agent_id")))
        if expected_artifact_revision is not None:
            revision_condition = FieldCondition(
                key="artifact_revision",
                match=MatchValue(value=expected_artifact_revision),
            )
            if expected_artifact_revision == 0:
                conditions.append(
                    Filter(
                        should=[
                            revision_condition,
                            IsEmptyCondition(
                                is_empty=PayloadField(key="artifact_revision")
                            ),
                        ]
                    )
                )
            else:
                conditions.append(revision_condition)
        if expected_child_token is not None:
            conditions.append(
                FieldCondition(
                    key="evidence_child_fence",
                    match=MatchValue(value=expected_child_token),
                )
            )
        update: dict[str, Any] = {
            "transition_token": token,
            "transition_kind": kind,
            "transition_operation_token": transition_operation_token or token,
            "transition_started_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        if successor_id:
            update["transition_successor_id"] = successor_id
        kwargs: dict[str, Any] = {
            "collection_name": self._vector.collection_name,
            "payload": update,
            "points": Filter(must=conditions),
            "wait": True,
        }
        if self._is_remote:
            kwargs["ordering"] = WriteOrdering.STRONG
        self._client.set_payload(**kwargs)
        claimed = self._read_point(point_id, with_vectors=False)
        if claimed is None:
            raise ValueError(f"Memory {point_id} not found")
        claimed_payload = self._payload(claimed)
        if claimed_payload.get("transition_token") != token or claimed_payload.get(
            "transition_operation_token"
        ) != (transition_operation_token or token):
            self._raise_stale(claimed_payload, point_id)
        return claimed_payload

    def _check_replay(
        self,
        *,
        points: list[Any],
        key_hash: str,
        fingerprint: str,
        token: str,
    ) -> dict[str, Any] | None:
        for point in points:
            payload = self._payload(point)
            if payload.get("revision_operation_key_hash") != key_hash:
                continue
            if payload.get("revision_operation_fingerprint") != fingerprint:
                raise RevisionConflictError(
                    "The idempotency key was already used for another request",
                    code="idempotency_conflict",
                    lineage_id=payload.get("lineage_id"),
                )
            operation = self.operations.get(token)
            if operation and operation.get("result"):
                result = dict(operation["result"])
                result["replayed"] = True
                return result
            state = payload.get("revision_state")
            if state == ACTIVE_REVISION_STATE and payload.get("supersedes"):
                previous_id = payload["supersedes"]
                previous = self._read_point(previous_id, with_vectors=False)
                if (
                    previous is not None
                    and self._payload(previous).get("transition_token") == token
                ):
                    self._client.set_payload(
                        collection_name=self._vector.collection_name,
                        payload={
                            "revision_state": "superseded",
                            "revision_successor_id": str(point.id),
                        },
                        points=[previous_id],
                        wait=True,
                    )
                result = self._result(
                    operation_id=_operation_point_id(token),
                    lineage_id=payload["lineage_id"],
                    previous_revision_id=previous_id,
                    revision_id=str(point.id),
                    revision=int(payload["revision"]),
                    status="updated",
                    replayed=True,
                )
                self.operations.write(
                    token,
                    {
                        "status": "committed",
                        "lineage_id": payload["lineage_id"],
                        "result": result,
                    },
                )
                return result
            if state == "retracted":
                result = self._result(
                    operation_id=_operation_point_id(token),
                    lineage_id=payload["lineage_id"],
                    previous_revision_id=str(point.id),
                    revision_id=None,
                    revision=int(payload["revision"]),
                    status="retracted",
                    replayed=True,
                )
                self.operations.write(
                    token,
                    {
                        "status": "committed",
                        "lineage_id": payload["lineage_id"],
                        "result": result,
                    },
                )
                return result
        return None

    def confirm(
        self,
        memory_id: str,
        *,
        user_id: str,
        owner_id: str,
        session_agent_id: str | None,
        evidence_root_id: str,
        source_kind: str,
        source_fingerprint: str,
        ttl_multiplier: float,
        max_score_roots: int,
        idempotency_key: str | None = None,
        expected_revision_id: str | None = None,
        expected_lineage_id: str | None = None,
        expected_content_hash: str | None = None,
        expected_fact_hash: str | None = None,
        parent_operation_id: str | None = None,
        parent_epoch: int | None = None,
        parent_nonce: str | None = None,
        _attempt: int = 0,
    ) -> dict[str, Any]:
        """Confirm an active revision with one independent evidence root."""
        child_record: dict[str, Any] | None = None
        child_token: str | None = None
        if parent_operation_id is not None:
            if parent_epoch is None or parent_nonce is None:
                raise EvidenceLeaseLostError("Evidence parent lease is incomplete")
            self.operations.verify_evidence_claim(
                parent_operation_id,
                request_fingerprint=source_fingerprint,
                epoch=parent_epoch,
                nonce=parent_nonce,
            )
        current, payload, _ = self._resolve(memory_id)
        self._authorize(
            payload,
            user_id=user_id,
            owner_id=owner_id,
            session_agent_id=session_agent_id,
        )
        # A crashed worker may leave a target claim while its child lease
        # expires. Reclaim the fixed child first. This updates the target
        # fence to the replacement child identity before recovery can write.
        if (
            parent_operation_id is not None
            and payload.get("transition_token")
            and payload.get("transition_operation_token")
        ):
            recovery_token = str(payload["transition_operation_token"])
            recovery_operation = self.operations.get(recovery_token) or {}
            child_key = recovery_operation.get("evidence_child_key")
            if not isinstance(child_key, str) or not child_key.startswith(
                f"{parent_operation_id}:"
            ):
                raise EvidenceLeaseLostError("Evidence child recovery binding is stale")
            recovery_ordinal = int(child_key.rsplit(":", 1)[-1])
            self.operations.claim_evidence_child(
                recovery_token,
                parent_operation_id=parent_operation_id,
                ordinal=recovery_ordinal,
                epoch=parent_epoch,
                nonce=parent_nonce,
            )
            current, payload, _ = self._resolve(memory_id)
        if payload.get("transition_token"):
            self._recover_claimed(current, payload)
            current, payload, _ = self._resolve(memory_id)
        current_id = str(current.id)
        if current_id != memory_id:
            self._raise_stale(payload, current_id)
        if payload.get("revision_state") != ACTIVE_REVISION_STATE:
            self._raise_stale(payload, current_id)
        if expected_revision_id is not None and expected_revision_id != current_id:
            self._raise_stale(payload, current_id)
        if expected_lineage_id is not None and expected_lineage_id != payload.get(
            "lineage_id"
        ):
            self._raise_stale(payload, current_id)
        if expected_content_hash is not None and expected_content_hash != payload.get(
            "hash"
        ):
            self._raise_stale(payload, current_id)
        if expected_fact_hash is not None and expected_fact_hash != (
            payload.get("fact_hash")
            or self._normalized_fact_hash(str(payload.get("data", "")))
        ):
            self._raise_stale(payload, current_id)

        lineage_id = payload["lineage_id"]
        fingerprint = canonical_fingerprint(
            {
                "kind": "confirm",
                "lineage_id": lineage_id,
                "revision_id": current_id,
                "evidence_root_id": evidence_root_id,
                "source_kind": source_kind,
                "source_fingerprint": source_fingerprint,
            }
        )
        key_hash = canonical_fingerprint(
            [
                user_id,
                owner_id,
                lineage_id,
                "confirm",
                idempotency_key or evidence_root_id,
            ]
        )
        token = canonical_fingerprint([key_hash, fingerprint])
        existing = self.operations.get(token)
        if existing and existing.get("status") == "committed":
            result = dict(existing["result"])
            result["replayed"] = True
            return result

        lineage_points = self._lineage_points(lineage_id)
        historical_roots: list[str] = []
        consumed_roots: list[str] = []
        for lineage_point in lineage_points:
            lineage_payload = self._payload(lineage_point)
            historical_roots.extend(lineage_payload.get("evidence_root_ids") or [])
            consumed_roots.extend(
                lineage_payload.get("consumed_evidence_root_ids") or []
            )
        roots = list(dict.fromkeys(payload.get("evidence_root_ids") or []))
        consumed = list(dict.fromkeys([*consumed_roots, *historical_roots]))
        if evidence_root_id in consumed:
            return {
                "operation_id": None,
                "lineage_id": lineage_id,
                "revision_id": current_id,
                "revision": int(payload.get("revision", 1)),
                "status": "skipped",
                "replayed": True,
            }
        roots.append(evidence_root_id)
        consumed.append(evidence_root_id)
        now = datetime.now(timezone.utc)
        confirmation_count = max(len(roots) - 1, 1)
        bounded_count = min(confirmation_count, max(max_score_roots, 1))
        projection: dict[str, Any] = {
            "evidence_root_ids": roots,
            "consumed_evidence_root_ids": consumed,
            "validation_count": bounded_count,
            "validation_strength": bounded_count / max(max_score_roots, 1),
            "validation_state": "confirmed",
            "last_validated_at": now.isoformat(),
            "validation_projection_hash": canonical_fingerprint(roots),
        }
        ttl_days = payload.get("ttl_days")
        if ttl_days is not None:
            candidate_expiry = now + timedelta(
                days=float(ttl_days) * max(float(ttl_multiplier), 1.0)
            )
            current_expiry = payload.get("expires_at")
            try:
                parsed_expiry = datetime.fromisoformat(current_expiry)
                if parsed_expiry.tzinfo is None:
                    parsed_expiry = parsed_expiry.replace(tzinfo=timezone.utc)
                candidate_expiry = max(candidate_expiry, parsed_expiry)
            except (TypeError, ValueError):
                pass
            projection["expires_at"] = candidate_expiry.isoformat()
            projection["decayed_at"] = None

        result = {
            "operation_id": _operation_point_id(token),
            "lineage_id": lineage_id,
            "revision_id": current_id,
            "revision": int(payload.get("revision", 1)),
            "status": "confirmed",
            "replayed": False,
        }
        operation_payload = {
            "status": "prepared",
            "operation_kind": "confirm",
            "actor_kind": "remember",
            "user_id": user_id,
            "owner_id": owner_id,
            "agent_id": payload.get("agent_id"),
            "lineage_id": lineage_id,
            "target_revision_id": current_id,
            "evidence_root_id": evidence_root_id,
            "source_kind": source_kind,
            "source_fingerprint": source_fingerprint,
            "parent_operation_id": parent_operation_id,
            "parent_claim_epoch": parent_epoch,
            "parent_claim_nonce": parent_nonce,
            "evidence_child_key": idempotency_key,
            "idempotency_key_hash": key_hash,
            "request_fingerprint": fingerprint,
            "transition_operation_token": token,
            "target_revision": int(payload.get("revision", 1)),
            "target_content_hash": payload.get("hash"),
            "target_fact_hash": payload.get("fact_hash")
            or self._normalized_fact_hash(str(payload.get("data", ""))),
            "validation_projection": projection,
            "result": result,
        }
        if existing is None:
            self.operations.write(token, operation_payload)
        elif parent_operation_id is not None and (
            existing.get("parent_operation_id") != parent_operation_id
            or existing.get("evidence_child_key") != idempotency_key
        ):
            raise EvidenceConflictError("Evidence child binding conflict")
        if parent_operation_id is not None:
            child = self.operations.claim_evidence_child(
                token,
                parent_operation_id=parent_operation_id,
                ordinal=int(str(idempotency_key).rsplit(":", 1)[-1]),
                epoch=parent_epoch,
                nonce=parent_nonce,
            )
            if child.get("status") == "committed" and child.get("result"):
                replay = dict(child["result"])
                replay["replayed"] = True
                return replay
            child_record = child
            child_token = child.get("child_transition_token")
            if not isinstance(child_token, str):
                raise EvidenceLeaseLostError("Evidence child token is missing")
        try:
            if parent_operation_id is not None:
                self.operations.verify_evidence_child(
                    token,
                    epoch=parent_epoch,
                    nonce=parent_nonce,
                )
            self._claim(
                point_id=current_id,
                payload=payload,
                token=child_token or token,
                kind="confirm",
                successor_id=None,
                expected_child_token=child_token,
                transition_operation_token=token,
            )
        except RevisionConflictError:
            if _attempt >= 2:
                raise
            latest = self._read_point(current_id)
            latest_payload = self._payload(latest) if latest is not None else {}
            if latest_payload.get("transition_token"):
                self._recover_claimed(latest, latest_payload)
            return self.confirm(
                memory_id,
                user_id=user_id,
                owner_id=owner_id,
                session_agent_id=session_agent_id,
                evidence_root_id=evidence_root_id,
                source_kind=source_kind,
                source_fingerprint=source_fingerprint,
                ttl_multiplier=ttl_multiplier,
                max_score_roots=max_score_roots,
                idempotency_key=idempotency_key,
                expected_revision_id=expected_revision_id,
                expected_lineage_id=expected_lineage_id,
                expected_content_hash=expected_content_hash,
                expected_fact_hash=expected_fact_hash,
                parent_operation_id=parent_operation_id,
                parent_epoch=parent_epoch,
                parent_nonce=parent_nonce,
                _attempt=_attempt + 1,
            )
        self.operations.write(
            token,
            {
                **operation_payload,
                "status": "claimed",
                "transition_operation_token": token,
            },
        )
        self._client.set_payload(
            collection_name=self._vector.collection_name,
            payload=projection,
            points=Filter(
                must=[
                    HasIdCondition(has_id=[current_id]),
                    FieldCondition(
                        key="transition_token",
                        match=MatchValue(value=child_token or token),
                    ),
                    *(
                        [
                            FieldCondition(
                                key="evidence_child_fence",
                                match=MatchValue(value=child_token),
                            )
                        ]
                        if child_token is not None
                        else []
                    ),
                ]
            ),
            wait=True,
        )
        projected = self._read_point(current_id, with_vectors=False)
        if (
            projected is None
            or self._payload(projected).get("transition_token")
            != (child_token or token)
            or (
                child_token is not None
                and self._payload(projected).get("evidence_child_fence") != child_token
            )
        ):
            raise EvidenceLeaseLostError("Evidence projection lost its target fence")
        if child_record is not None:
            self.operations.complete_evidence_child(
                token,
                epoch=parent_epoch,
                nonce=parent_nonce,
                result=result,
            )
        else:
            self.operations.write(
                token,
                {**operation_payload, "status": "committed", "result": result},
            )
        self._release_claim(current_id, child_token or token)
        return result

    def revise(
        self,
        memory_id: str,
        *,
        user_id: str,
        owner_id: str,
        session_agent_id: str | None,
        changes: dict[str, Any],
        expected_revision: int | None = None,
        idempotency_key: str | None = None,
        operation_kind: str = "update",
        actor_kind: str = "api",
        reason: str | None = None,
        derived_from: list[str] | None = None,
        audit: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create and activate an immutable successor revision."""
        current, current_payload, points = self._resolve(memory_id)
        self._authorize(
            current_payload,
            user_id=user_id,
            owner_id=owner_id,
            session_agent_id=session_agent_id,
        )
        lineage_id = current_payload["lineage_id"]
        fingerprint = canonical_fingerprint(
            {
                "kind": operation_kind,
                "lineage_id": lineage_id,
                "expected_revision": expected_revision,
                "changes": changes,
                "derived_from": derived_from,
            }
        )
        if current_payload.get("transition_token"):
            recovery_token = current_payload["transition_token"]
            recovery_operation = self.operations.get(recovery_token) or {}
            self._recover_claimed(current, current_payload)
            current, current_payload, points = self._resolve(memory_id)
            if recovery_operation.get("request_fingerprint") == fingerprint:
                replay = self._check_replay(
                    points=points,
                    key_hash=recovery_operation["idempotency_key_hash"],
                    fingerprint=fingerprint,
                    token=recovery_token,
                )
                if replay:
                    return replay
        else:
            for candidate in points:
                candidate_payload = self._payload(candidate)
                if candidate_payload.get(
                    "revision_operation_fingerprint"
                ) != fingerprint or not candidate_payload.get("supersedes"):
                    continue
                predecessor = self._read_point(
                    candidate_payload["supersedes"],
                    with_vectors=False,
                )
                if predecessor is None:
                    continue
                predecessor_payload = self._payload(predecessor)
                recovery_token = predecessor_payload.get("transition_token")
                key_hash = candidate_payload.get("revision_operation_key_hash")
                if not recovery_token or not key_hash:
                    continue
                self._recover_claimed(predecessor, predecessor_payload)
                current, current_payload, points = self._resolve(memory_id)
                replay = self._check_replay(
                    points=points,
                    key_hash=key_hash,
                    fingerprint=fingerprint,
                    token=recovery_token,
                )
                if replay:
                    return replay
        current_id = str(current.id)
        current_revision = int(current_payload.get("revision", 1))
        key = idempotency_key or (
            f"implicit:{current_id}:{current_revision}:{fingerprint}"
        )
        key_hash = canonical_fingerprint(
            [user_id, owner_id, lineage_id, operation_kind, key]
        )
        token = canonical_fingerprint([key_hash, fingerprint])
        replay = self._check_replay(
            points=points,
            key_hash=key_hash,
            fingerprint=fingerprint,
            token=token,
        )
        if replay:
            return replay

        if current_payload.get("revision_state") != ACTIVE_REVISION_STATE:
            self._raise_stale(current_payload, current_id)
        if expected_revision is not None and expected_revision != current_revision:
            self._raise_stale(current_payload, current_id)

        successor_id = _successor_point_id(token)
        successor_payload = dict(current_payload)
        for field in _TRANSITION_FIELDS:
            successor_payload.pop(field, None)
        successor_payload.update(changes)
        text = successor_payload.get("data", "")
        if "data" in changes:
            successor_payload["hash"] = hashlib.sha256(text.encode()).hexdigest()
            successor_payload["fact_hash"] = self._normalized_fact_hash(text)
        historical_consumed = list(
            dict.fromkeys(
                root
                for item in points
                for root in (
                    self._payload(item).get("consumed_evidence_root_ids") or []
                )
                + (self._payload(item).get("evidence_root_ids") or [])
            )
        )
        if historical_consumed:
            successor_payload["consumed_evidence_root_ids"] = historical_consumed
        semantic_transformation = "data" in changes or "derived_from" in changes
        if semantic_transformation:
            successor_payload.update(
                {
                    "validation_count": 0,
                    "validation_strength": 0.0,
                    "validation_state": "unverified",
                    "last_validated_at": None,
                    "evidence_root_ids": [],
                }
            )
        now = datetime.now(timezone.utc).isoformat()
        successor_payload.update(
            {
                "lineage_id": lineage_id,
                "revision": current_revision + 1,
                "revision_state": "pending",
                "supersedes": current_id,
                "revision_created_at_utc": now,
                "revision_operation_id": _operation_point_id(token),
                "revision_operation_key_hash": key_hash,
                "revision_operation_fingerprint": fingerprint,
                "updated_at": now,
                "updated_at_utc": now,
            }
        )
        if derived_from is not None:
            successor_payload["derived_from"] = list(dict.fromkeys(derived_from))
        existing_successor = self._read_point(successor_id)
        if existing_successor is None:
            vector = current.vector
            if "data" in changes:
                dense = self._vector.embedding.embed(text)
                sparse = self._sparse_embed(text) if self._sparse_embed else None
                vector = {"": dense, "bm25": sparse} if sparse is not None else dense
            self._client.upsert(
                collection_name=self._vector.collection_name,
                points=[
                    PointStruct(
                        id=successor_id,
                        vector=vector,
                        payload=successor_payload,
                    )
                ],
                wait=True,
            )
        else:
            existing_payload = self._payload(existing_successor)
            if existing_payload.get("revision_operation_fingerprint") != fingerprint:
                raise RevisionConflictError(
                    "The deterministic successor ID contains another operation",
                    code="idempotency_conflict",
                    lineage_id=lineage_id,
                )

        operation_payload = {
            "status": "pending",
            "operation_kind": operation_kind,
            "actor_kind": actor_kind,
            "user_id": user_id,
            "owner_id": owner_id,
            "agent_id": current_payload.get("agent_id"),
            "lineage_id": lineage_id,
            "idempotency_key_hash": key_hash,
            "request_fingerprint": fingerprint,
            "expected_revision": expected_revision,
            "previous_revision_id": current_id,
            "successor_revision_id": successor_id,
            "reason": reason,
        }
        if audit:
            operation_payload.update(audit)
        self.operations.write(token, operation_payload)

        try:
            claimed_payload = self._claim(
                point_id=current_id,
                payload=current_payload,
                token=token,
                kind=operation_kind,
                successor_id=successor_id,
            )
        except RevisionConflictError:
            self._client.set_payload(
                collection_name=self._vector.collection_name,
                payload={"revision_state": "aborted"},
                points=[successor_id],
                wait=True,
            )
            self.operations.write(token, {**operation_payload, "status": "conflict"})
            raise
        if claimed_payload.get("transition_token") != token:
            self._client.set_payload(
                collection_name=self._vector.collection_name,
                payload={"revision_state": "aborted"},
                points=[successor_id],
                wait=True,
            )
            self._raise_stale(claimed_payload, current_id)

        self._client.set_payload(
            collection_name=self._vector.collection_name,
            payload={"revision_state": ACTIVE_REVISION_STATE},
            points=Filter(
                must=[
                    HasIdCondition(has_id=[successor_id]),
                    FieldCondition(
                        key="revision_state", match=MatchValue(value="pending")
                    ),
                    FieldCondition(
                        key="revision_operation_fingerprint",
                        match=MatchValue(value=fingerprint),
                    ),
                ]
            ),
            wait=True,
        )
        self.operations.write(token, {**operation_payload, "status": "activated"})
        self._client.set_payload(
            collection_name=self._vector.collection_name,
            payload={
                "revision_state": "superseded",
                "revision_successor_id": successor_id,
            },
            points=Filter(
                must=[
                    HasIdCondition(has_id=[current_id]),
                    FieldCondition(
                        key="transition_token", match=MatchValue(value=token)
                    ),
                ]
            ),
            wait=True,
        )
        result = self._result(
            operation_id=_operation_point_id(token),
            lineage_id=lineage_id,
            previous_revision_id=current_id,
            revision_id=successor_id,
            revision=current_revision + 1,
            status="updated",
            replayed=False,
        )
        self.operations.write(
            token,
            {**operation_payload, "status": "committed", "result": result},
        )
        return result

    def retract(
        self,
        memory_id: str,
        *,
        user_id: str,
        owner_id: str,
        session_agent_id: str | None,
        expected_revision: int | None = None,
        idempotency_key: str | None = None,
        actor_kind: str = "api",
        reason: str = "deleted",
        audit: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Retract an active revision without erasing its content."""
        current, payload, points = self._resolve(memory_id)
        self._authorize(
            payload,
            user_id=user_id,
            owner_id=owner_id,
            session_agent_id=session_agent_id,
        )
        lineage_id = payload["lineage_id"]
        fingerprint = canonical_fingerprint(
            {
                "kind": "retract",
                "lineage_id": lineage_id,
                "expected_revision": expected_revision,
                "reason": reason,
            }
        )
        if payload.get("transition_token"):
            recovery_token = payload["transition_token"]
            recovery_operation = self.operations.get(recovery_token) or {}
            self._recover_claimed(current, payload)
            current, payload, points = self._resolve(memory_id)
            if recovery_operation.get("request_fingerprint") == fingerprint:
                replay = self._check_replay(
                    points=points,
                    key_hash=recovery_operation["idempotency_key_hash"],
                    fingerprint=fingerprint,
                    token=recovery_token,
                )
                if replay:
                    return replay
        current_id = str(current.id)
        revision = int(payload.get("revision", 1))
        key = idempotency_key or f"implicit:{current_id}:{revision}:{fingerprint}"
        key_hash = canonical_fingerprint(
            [user_id, owner_id, lineage_id, "retract", key]
        )
        token = canonical_fingerprint([key_hash, fingerprint])
        replay = self._check_replay(
            points=points,
            key_hash=key_hash,
            fingerprint=fingerprint,
            token=token,
        )
        if replay:
            return replay
        if payload.get("revision_state") != ACTIVE_REVISION_STATE:
            self._raise_stale(payload, current_id)
        if expected_revision is not None and expected_revision != revision:
            self._raise_stale(payload, current_id)
        operation_payload = {
            "status": "prepared",
            "operation_kind": "retract",
            "actor_kind": actor_kind,
            "user_id": user_id,
            "owner_id": owner_id,
            "agent_id": payload.get("agent_id"),
            "lineage_id": lineage_id,
            "idempotency_key_hash": key_hash,
            "request_fingerprint": fingerprint,
            "expected_revision": expected_revision,
            "previous_revision_id": current_id,
            "reason": reason,
        }
        if audit:
            operation_payload.update(audit)
        self.operations.write(token, operation_payload)
        self._claim(
            point_id=current_id,
            payload=payload,
            token=token,
            kind="retract",
            successor_id=None,
        )
        self.operations.write(token, {**operation_payload, "status": "claimed"})
        self._client.set_payload(
            collection_name=self._vector.collection_name,
            payload={
                "revision_state": "retracted",
                "state_reason": reason,
                "revision_operation_id": _operation_point_id(token),
                "revision_operation_key_hash": key_hash,
                "revision_operation_fingerprint": fingerprint,
            },
            points=Filter(
                must=[
                    HasIdCondition(has_id=[current_id]),
                    FieldCondition(
                        key="transition_token", match=MatchValue(value=token)
                    ),
                ]
            ),
            wait=True,
        )
        result = self._result(
            operation_id=_operation_point_id(token),
            lineage_id=lineage_id,
            previous_revision_id=current_id,
            revision_id=None,
            revision=revision,
            status="retracted",
            replayed=False,
        )
        self.operations.write(
            token,
            {**operation_payload, "status": "committed", "result": result},
        )
        return result

    def mark_source(
        self,
        revision_ids: list[str],
        *,
        operation_id: str,
        user_id: str,
        owner_id: str,
        session_agent_id: str | None = None,
        reason: str = "consolidated",
        expected_revisions: dict[str, int] | None = None,
        mutation_guard: Callable[[], None] | None = None,
    ) -> None:
        """Mark exact raw evidence revisions as retained derivation sources."""
        for revision_id in dict.fromkeys(revision_ids):
            point = self._read_point(revision_id, with_vectors=False)
            if point is None:
                raise ValueError(f"Source revision {revision_id} not found")
            source_payload = self._payload(point)
            self._authorize(
                source_payload,
                user_id=user_id,
                owner_id=owner_id,
                session_agent_id=session_agent_id,
            )
            agent_condition: Any
            if source_payload.get("agent_id"):
                agent_condition = FieldCondition(
                    key="agent_id",
                    match=MatchValue(value=source_payload["agent_id"]),
                )
            else:
                agent_condition = IsEmptyCondition(
                    is_empty=PayloadField(key="agent_id")
                )
            revision_conditions: list[Any] = []
            expected_revision = (expected_revisions or {}).get(revision_id)
            if expected_revision is not None:
                revision_conditions.append(
                    FieldCondition(
                        key="revision",
                        match=MatchValue(value=expected_revision),
                    )
                )
            if mutation_guard is not None:
                mutation_guard()
            self._client.set_payload(
                collection_name=self._vector.collection_name,
                payload={
                    "revision_state": "source",
                    "state_reason": reason,
                    "revision_operation_id": operation_id,
                },
                points=Filter(
                    must=[
                        HasIdCondition(has_id=[revision_id]),
                        FieldCondition(
                            key="revision_state",
                            match=MatchValue(value=ACTIVE_REVISION_STATE),
                        ),
                        FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                        FieldCondition(
                            key="owner_id", match=MatchValue(value=owner_id)
                        ),
                        agent_condition,
                        *revision_conditions,
                    ]
                ),
                wait=True,
            )
            updated = self._read_point(revision_id, with_vectors=False)
            updated_payload = self._payload(updated) if updated is not None else {}
            if (
                updated_payload.get("revision_state") != "source"
                or updated_payload.get("revision_operation_id") != operation_id
            ):
                raise RevisionConflictError(
                    f"Source revision {revision_id} changed before transition"
                )

    def history(
        self,
        memory_id: str,
        *,
        user_id: str,
        owner_id: str,
        session_agent_id: str | None,
    ) -> dict[str, Any]:
        """Return all authorized revisions and audit records for a lineage."""
        _, payload, points = self._resolve(memory_id)
        self._authorize(
            payload,
            user_id=user_id,
            owner_id=owner_id,
            session_agent_id=session_agent_id,
        )
        revisions = []
        operation_ids = set()
        for point in points:
            point_payload = self._payload(point)
            if not self._is_authorized(
                point_payload,
                user_id=user_id,
                owner_id=owner_id,
                session_agent_id=session_agent_id,
            ):
                continue
            revisions.append(self._vector._point_to_memory(point))
            if point_payload.get("revision_operation_id"):
                operation_ids.add(point_payload["revision_operation_id"])
        revisions.sort(
            key=lambda item: int((item.get("metadata") or {}).get("revision", 1))
        )
        operations = self.operations.list_for_lineage(
            lineage_id=payload["lineage_id"],
            user_id=user_id,
            owner_id=owner_id,
        )
        operations = [
            operation
            for operation in operations
            if self._is_authorized(
                operation,
                user_id=user_id,
                owner_id=owner_id,
                session_agent_id=session_agent_id,
            )
        ]
        known_operation_ids = {
            operation.get("operation_id") for operation in operations
        }
        for operation_id in operation_ids - known_operation_ids:
            operation = self.operations.get_by_id(operation_id)
            if operation and self._is_authorized(
                operation,
                user_id=user_id,
                owner_id=owner_id,
                session_agent_id=session_agent_id,
            ):
                operations.append(operation)
        operations.sort(key=lambda item: item.get("created_at_utc", ""))
        return {
            "lineage_id": payload["lineage_id"],
            "revisions": revisions,
            "operations": operations,
        }

    def history_page(
        self,
        memory_id: str,
        *,
        user_id: str,
        owner_id: str,
        session_agent_id: str | None,
        revision_before: int | None,
        operation_before: str | None,
        limit: int,
    ) -> dict[str, Any]:
        """Return bounded revision and operation pages for the management UI."""
        requested = self._read_point(memory_id, with_vectors=False)
        if requested is None:
            raise ValueError("Memory not found")
        payload = self._materialize_legacy(requested)
        self._authorize(
            payload,
            user_id=user_id,
            owner_id=owner_id,
            session_agent_id=session_agent_id,
        )
        lineage_id = payload["lineage_id"]
        agent_id = payload.get("agent_id")
        agent_condition: Any
        if agent_id:
            agent_condition = FieldCondition(
                key="agent_id", match=MatchValue(value=agent_id)
            )
        else:
            agent_condition = IsEmptyCondition(is_empty=PayloadField(key="agent_id"))
        must: list[Any] = [
            FieldCondition(key="lineage_id", match=MatchValue(value=lineage_id)),
            FieldCondition(key="user_id", match=MatchValue(value=user_id)),
            FieldCondition(key="owner_id", match=MatchValue(value=owner_id)),
            agent_condition,
        ]
        if revision_before is not None:
            must.append(FieldCondition(key="revision", range=Range(lt=revision_before)))
        points, _ = self._client.scroll(
            collection_name=self._vector.collection_name,
            scroll_filter=Filter(must=must),
            limit=limit + 1,
            order_by=OrderBy(key="revision", direction=Direction.DESC),
            with_payload=True,
            with_vectors=False,
        )
        has_more = len(points) > limit
        points = points[:limit]
        next_revision_before = None
        if has_more and points:
            next_revision_before = int(self._payload(points[-1]).get("revision", 1))
        revisions = [self._vector._point_to_memory(point) for point in points]
        revisions.reverse()
        active_points, _ = self._client.scroll(
            collection_name=self._vector.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="lineage_id", match=MatchValue(value=lineage_id)
                    ),
                    FieldCondition(
                        key="revision_state",
                        match=MatchValue(value=ACTIVE_REVISION_STATE),
                    ),
                    FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                    FieldCondition(key="owner_id", match=MatchValue(value=owner_id)),
                    agent_condition,
                ]
            ),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        operations_page = self.operations.list_for_lineage_page(
            lineage_id=lineage_id,
            user_id=user_id,
            owner_id=owner_id,
            agent_id=agent_id,
            before_created_at=operation_before,
            limit=limit,
        )
        return {
            "lineage_id": lineage_id,
            "current_revision_id": (
                str(active_points[0].id) if active_points else None
            ),
            "revisions": revisions,
            "operations": operations_page["operations"],
            "next_revision_before": next_revision_before,
            "next_operation_before": operations_page["next_before"],
        }

    def links(
        self,
        memory_id: str,
        *,
        user_id: str,
        owner_id: str,
        session_agent_id: str | None,
    ) -> dict[str, Any]:
        """Return authorized supersession and derivation links."""
        point, payload, _ = self._resolve(memory_id)
        self._authorize(
            payload,
            user_id=user_id,
            owner_id=owner_id,
            session_agent_id=session_agent_id,
        )
        requested = self._read_point(memory_id, with_vectors=False) or point
        requested_payload = self._payload(requested)
        self._authorize(
            requested_payload,
            user_id=user_id,
            owner_id=owner_id,
            session_agent_id=session_agent_id,
        )
        source_ids = list(requested_payload.get("derived_from") or [])
        sources = []
        for source_id in source_ids:
            source = self._read_point(source_id, with_vectors=False)
            if source is None:
                continue
            source_payload = self._payload(source)
            try:
                self._authorize(
                    source_payload,
                    user_id=user_id,
                    owner_id=owner_id,
                    session_agent_id=session_agent_id,
                )
            except ValueError:
                continue
            sources.append(source_id)
        derived_points, _ = self._client.scroll(
            collection_name=self._vector.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="derived_from",
                        match=MatchValue(value=str(requested.id)),
                    ),
                    FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                    FieldCondition(key="owner_id", match=MatchValue(value=owner_id)),
                ]
            ),
            limit=1000,
            with_payload=True,
            with_vectors=False,
        )
        derived = []
        for item in derived_points:
            try:
                self._authorize(
                    self._payload(item),
                    user_id=user_id,
                    owner_id=owner_id,
                    session_agent_id=session_agent_id,
                )
            except ValueError:
                continue
            derived.append(str(item.id))
        supersedes = requested_payload.get("supersedes")
        successor = requested_payload.get("revision_successor_id")
        for target_id, field in ((supersedes, "supersedes"), (successor, "successor")):
            if not target_id:
                continue
            target = self._read_point(target_id, with_vectors=False)
            try:
                if target is None:
                    raise ValueError
                self._authorize(
                    self._payload(target),
                    user_id=user_id,
                    owner_id=owner_id,
                    session_agent_id=session_agent_id,
                )
            except ValueError:
                if field == "supersedes":
                    supersedes = None
                else:
                    successor = None
        return {
            "revision_id": str(requested.id),
            "lineage_id": requested_payload.get("lineage_id", str(requested.id)),
            "supersedes": supersedes,
            "successor": successor,
            "derived_from": sources,
            "derived_outputs": derived,
            "provenance_quality": requested_payload.get("provenance_quality", "exact"),
        }

    def prepare_privacy_erase(
        self,
        memory_id: str,
        *,
        user_id: str,
        owner_id: str,
        session_agent_id: str | None,
    ) -> dict[str, Any]:
        """Claim a lineage and persist a resumable privacy-erasure plan."""
        try:
            current, payload, points = self._resolve(memory_id)
        except ValueError:
            operation = self.operations.find_privacy_erase(memory_id)
            if operation and self._is_authorized(
                operation,
                user_id=user_id,
                owner_id=owner_id,
                session_agent_id=session_agent_id,
            ):
                return operation
            raise
        self._authorize(
            payload,
            user_id=user_id,
            owner_id=owner_id,
            session_agent_id=session_agent_id,
        )
        if payload.get("transition_token"):
            if payload.get("transition_kind") == "privacy_erase":
                operation = self.operations.get(payload["transition_token"])
                if operation:
                    return operation
            self._recover_claimed(current, payload)
            current, payload, points = self._resolve(memory_id)
        revision_ids = [str(point.id) for point in points]
        artifact_ids = list(
            dict.fromkeys(
                artifact["id"]
                for point in points
                for artifact in (self._payload(point).get("artifacts") or [])
                if artifact.get("id")
            )
        )
        token = canonical_fingerprint(
            ["privacy_erase", user_id, owner_id, payload["lineage_id"]]
        )
        operation = {
            "status": "prepared",
            "operation_kind": "privacy_erase",
            "actor_kind": "privacy",
            "user_id": user_id,
            "owner_id": owner_id,
            "agent_id": payload.get("agent_id"),
            "lineage_id": payload["lineage_id"],
            "target_revision_ids": revision_ids,
            "artifact_ids": artifact_ids,
        }
        operation_id = self.operations.write(token, operation)
        try:
            self._claim(
                point_id=str(current.id),
                payload=payload,
                token=token,
                kind="privacy_erase",
                successor_id=None,
            )
        except Exception:
            self.operations.write(token, {**operation, "status": "conflict"})
            raise
        self.operations.write(token, {**operation, "status": "claimed"})
        operation.update({"operation_id": operation_id, "operation_token": token})
        return operation

    def finalize_privacy_erase(
        self,
        plan: dict[str, Any],
    ) -> dict[str, Any]:
        """Erase Qdrant lineage and audit data after artifacts are handled."""
        lineage_id = plan["lineage_id"]
        deleted_revision_ids: list[str] = []
        while True:
            points = self._lineage_points(lineage_id)
            revision_ids = [str(point.id) for point in points]
            if not revision_ids:
                break
            deleted_revision_ids.extend(revision_ids)
            self._client.delete(
                collection_name=self._vector.collection_name,
                points_selector=PointIdsList(points=revision_ids),
                wait=True,
            )
        self.operations.delete_lineage(
            lineage_id=lineage_id,
            user_id=plan["user_id"],
            owner_id=plan["owner_id"],
        )
        return {
            "status": "erased",
            "lineage_id": lineage_id,
            "revision_ids": plan.get("target_revision_ids", deleted_revision_ids),
        }
