"""Memory consistency check (fsck) — detect and fix quality issues.

Three-phase pipeline:
  Phase 0 — Security scan: regex-based prompt injection detection (no LLM)
  Phase 1 — Duplicate detection: vector similarity clustering + LLM evaluation
  Phase 2 — Quality check: LLM-based batch evaluation for spelling, sense,
             split candidates, metadata misclassification, and subtle injection

Results are cached in-memory with configurable TTL so the user can review
issues in the UI and then apply selected fixes without re-running the check.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import partial
from typing import TYPE_CHECKING, Any

from qdrant_client.models import PointStruct

from mnemory.categories import (
    PREDEFINED_CATEGORIES,
    validate_categories,
    validate_importance,
    validate_memory_type,
)
from mnemory.config import Config
from mnemory.embeddings import EmbeddingClient
from mnemory.llm import LLMClient, parse_json_response
from mnemory.prompts import (
    build_fsck_content_quality_prompt,
    build_fsck_duplicate_prompt,
    build_fsck_metadata_normalization_prompt,
    build_fsck_security_reeval_prompt,
)
from mnemory.revisions import (
    FSCK_AUDIT_MAX_TARGETS,
    FSCK_AUDIT_MODE,
    canonical_fingerprint,
)
from mnemory.sanitize import detect_injection_patterns
from mnemory.storage.vector import VectorStore

if TYPE_CHECKING:
    from mnemory.memory import MemoryService

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────

# Similarity threshold for duplicate detection (higher than dedup_similarity
# during ingestion because we want to flag clear duplicates, not just related).
_DUPLICATE_SIMILARITY_THRESHOLD = 0.75

# Maximum memories per LLM quality-check batch.
_QUALITY_BATCH_SIZE = 20

# Maximum memories per duplicate cluster sent to LLM.
_MAX_CLUSTER_SIZE = 15

# Maximum similar neighbors to check per memory during duplicate detection.
_DUPLICATE_NEIGHBORS = 5


# ── Data structures ──────────────────────────────────────────────────


@dataclass
class FsckAction:
    """A single action to fix an issue."""

    action: str  # "update", "delete", "add"
    memory_id: str | None = None
    new_content: str | None = None
    new_metadata: dict | None = None


@dataclass
class FsckAffectedMemory:
    """A memory affected by an issue."""

    id: str
    content: str
    metadata: dict | None = None
    agent_id: str | None = None


@dataclass
class FsckIssue:
    """A single issue found during memory check."""

    issue_id: str
    type: str  # duplicate, quality, split, contradiction, reclassify, security
    severity: str  # low, medium, high
    reasoning: str
    affected_memories: list[FsckAffectedMemory]
    actions: list[FsckAction]
    confidence: float | None = None  # 0.0-1.0, LLM-reported confidence


@dataclass
class FsckProgress:
    """Progress of a running memory check."""

    phase: str = "starting"
    total_memories: int = 0
    processed: int = 0
    percent: int = 0
    issues_found: int = 0
    truncated: bool = False


@dataclass
class FsckSummary:
    """Summary of issues found."""

    duplicate: int = 0
    quality: int = 0
    split: int = 0
    contradiction: int = 0
    reclassify: int = 0
    security: int = 0
    total: int = 0


@dataclass
class FsckCheck:
    """State of a single memory check run."""

    check_id: str
    user_id: str
    owner_id: str | None = None
    agent_id: str | None = None
    status: str = "running"  # running, applying, completed, failed
    progress: FsckProgress = field(default_factory=FsckProgress)
    issues: list[FsckIssue] = field(default_factory=list)
    summary: FsckSummary | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.monotonic)
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    ttl_seconds: int = 1800
    budget_exhausted: bool = False
    llm_calls: int = 0
    # Track which issues have been applied to prevent double-apply.
    applied_issue_ids: set[str] = field(default_factory=set)
    mode: str = "scan"
    basis_operation_id: str | None = None
    basis_operation_fingerprint: str | None = None
    audit_issue_type: str | None = None
    target_revisions: list[dict[str, Any]] = field(default_factory=list)
    target_state: str | None = None
    target_snapshot_fingerprint: str | None = None

    @property
    def is_expired(self) -> bool:
        return time.monotonic() - self.created_at > self.ttl_seconds

    @property
    def expires_at_utc(self) -> str:
        """Compute expiration time as ISO 8601 UTC string."""
        created = datetime.fromisoformat(self.created_at_utc)
        expires = created + timedelta(seconds=self.ttl_seconds)
        return expires.isoformat()


# ── FsckStore — in-memory state for check runs ──────────────────────


class FsckStore:
    """Thread-safe in-memory store for fsck check results.

    Similar to SessionStore but for fsck check runs. Checks are stored
    with a TTL and cleaned up periodically via a background sweep task.
    """

    def __init__(self, default_ttl: int = 1800, sweep_interval: int = 300):
        self._checks: dict[str, FsckCheck] = {}
        self._lock = threading.Lock()
        self._default_ttl = default_ttl
        self._sweep_interval = sweep_interval
        self._sweep_task: asyncio.Task | None = None

    def create(
        self,
        user_id: str,
        owner_id: str | None = None,
        agent_id: str | None = None,
    ) -> FsckCheck:
        """Create a new check and return it."""
        check = FsckCheck(
            check_id=str(uuid.uuid4()),
            user_id=user_id,
            owner_id=owner_id or user_id,
            agent_id=agent_id,
            ttl_seconds=self._default_ttl,
        )
        with self._lock:
            self._checks[check.check_id] = check
        return check

    def get(self, check_id: str) -> FsckCheck | None:
        """Get a check by ID, or None if not found/expired."""
        with self._lock:
            check = self._checks.get(check_id)
            if check is None:
                return None
            if check.is_expired:
                del self._checks[check_id]
                return None
            return check

    def put(self, check: FsckCheck) -> None:
        """Store a reconstructed durable check."""
        with self._lock:
            self._checks[check.check_id] = check

    def sweep(self) -> int:
        """Remove expired checks. Returns count removed."""
        with self._lock:
            expired = [cid for cid, c in self._checks.items() if c.is_expired]
            for cid in expired:
                del self._checks[cid]
            return len(expired)

    def start_cleanup_task(self) -> None:
        """Start periodic background sweep for expired checks.

        Safe to call multiple times — only starts one task.
        Must be called from within an async context (event loop running).
        """
        if self._sweep_task is not None:
            return
        self._sweep_task = asyncio.create_task(self._sweep_loop())
        logger.info("Fsck cleanup task started (interval=%ds)", self._sweep_interval)

    async def stop_cleanup_task(self) -> None:
        """Stop the background sweep task."""
        if self._sweep_task is not None:
            self._sweep_task.cancel()
            try:
                await self._sweep_task
            except asyncio.CancelledError:
                pass
            self._sweep_task = None
            logger.info("Fsck cleanup task stopped")

    async def _sweep_loop(self) -> None:
        """Periodically remove expired checks."""
        while True:
            await asyncio.sleep(self._sweep_interval)
            removed = self.sweep()
            if removed > 0:
                logger.info("Fsck sweep: removed %d expired checks", removed)


# ── Union-Find for clustering ────────────────────────────────────────


class _UnionFind:
    """Simple union-find (disjoint set) for clustering similar memories."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}
        self._rank: dict[str, int] = {}

    def find(self, x: str) -> str:
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]  # path compression
            x = self._parent[x]
        return x

    def union(self, x: str, y: str) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1

    def clusters(self) -> dict[str, list[str]]:
        """Return clusters as {root_id: [member_ids]}."""
        groups: dict[str, list[str]] = {}
        for x in self._parent:
            root = self.find(x)
            groups.setdefault(root, []).append(x)
        return groups


# ── FsckService — the check pipeline ────────────────────────────────


class FsckService:
    """Memory consistency check service.

    Runs a three-phase pipeline to detect and suggest fixes for memory
    quality issues. Results are stored in FsckStore for later application.
    """

    def __init__(
        self,
        config: Config,
        vector: VectorStore,
        llm: LLMClient,
        store: FsckStore,
        memory_service: MemoryService | None = None,
    ):
        self._config = config
        self._vector = vector
        self._llm = llm
        self._store = store
        self._memory_service = memory_service
        self._reasoning_effort = config.memory.fsck_reasoning_effort

    # ── Budget helpers ──────────────────────────────────────────────

    def _check_budget(self, check: FsckCheck) -> bool:
        """Return True if LLM budget is exhausted."""
        max_calls = self._config.memory.fsck_max_llm_calls
        return max_calls > 0 and check.llm_calls >= max_calls

    # ── Public API ───────────────────────────────────────────────────

    def start_check(
        self,
        user_id: str,
        owner_id: str | None = None,
        agent_id: str | None = None,
    ) -> FsckCheck:
        """Create a new check and return it (status=running).

        The caller is responsible for running run_check() in a background
        task after this returns, passing any filter parameters (categories,
        memory_type) directly to run_check().
        """
        check = self._store.create(user_id, owner_id, agent_id)
        return check

    @staticmethod
    def _normalized_action_plan(plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return the behavior-bearing action fields in stable order."""
        normalized = [
            {
                "action": item.get("action"),
                "memory_id": item.get("memory_id"),
                "new_content": item.get("new_content"),
                "new_metadata": item.get("new_metadata"),
            }
            for item in plan
        ]
        normalized.sort(
            key=lambda item: (
                str(item.get("memory_id") or ""),
                str(item.get("action") or ""),
                canonical_fingerprint(item),
            )
        )
        return normalized

    @classmethod
    def _basis_operation_fingerprint(cls, operation: dict[str, Any]) -> str:
        """Bind an audit to one immutable fsck operation definition."""
        return canonical_fingerprint(
            {
                "operation_id": operation.get("operation_id"),
                "fsck_check_id": operation.get("fsck_check_id"),
                "fsck_issue_id": operation.get("fsck_issue_id"),
                "issue_type": (operation.get("issue") or {}).get("type"),
                "plan": cls._normalized_action_plan(operation.get("plan") or []),
            }
        )

    @staticmethod
    def _operation_target_revisions(
        operation: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Return exact revision and agent bindings from an fsck operation."""
        issue = operation.get("issue") or {}
        affected = {
            str(item.get("id")): item
            for item in issue.get("affected_memories") or []
            if isinstance(item, dict) and item.get("id")
        }
        targets: dict[str, dict[str, Any]] = {}
        expected_revisions = {
            str(item["memory_id"]): item.get("expected_revision")
            for item in operation.get("plan") or []
            if isinstance(item, dict) and item.get("memory_id")
        }
        for memory_id, source in affected.items():
            metadata = source.get("metadata") or {}
            revision = expected_revisions.get(memory_id) or metadata.get("revision")
            if revision is None:
                raise ValueError("Fsck operation target revision is unavailable")
            targets[memory_id] = {
                "memory_id": memory_id,
                "revision": int(revision),
                "agent_id": source.get("agent_id"),
            }
        for item in operation.get("plan") or []:
            if not isinstance(item, dict) or not item.get("memory_id"):
                continue
            memory_id = str(item["memory_id"])
            source = affected.get(memory_id) or {}
            metadata = source.get("metadata") or {}
            revision = item.get("expected_revision", metadata.get("revision"))
            if revision is None:
                raise ValueError("Fsck operation target revision is unavailable")
            target = {
                "memory_id": memory_id,
                "revision": int(revision),
                "agent_id": source.get("agent_id"),
            }
            previous = targets.get(memory_id)
            if previous is not None and previous != target:
                raise ValueError("Fsck operation target binding is inconsistent")
            targets[memory_id] = target
        result = sorted(targets.values(), key=lambda item: item["memory_id"])
        if not result or len(result) > FSCK_AUDIT_MAX_TARGETS:
            raise ValueError("Fsck operation target set is not auditable")
        return result

    def _get_authorized_fsck_operation(
        self,
        operation_id: str,
        *,
        user_id: str,
        owner_id: str,
        session_agent_id: str | None,
    ) -> dict[str, Any]:
        """Return one authorized fsck operation without leaking its scope."""
        if self._memory_service is None:
            raise RuntimeError("Fsck operation journal is not available")
        operation = self._memory_service.revisions.operations.get_by_id(operation_id)
        if operation is None or operation.get("operation_kind") != "fsck":
            raise ValueError("Fsck operation not found")
        if operation.get("user_id") != user_id or operation.get("owner_id") != owner_id:
            raise ValueError("Fsck operation not found")
        if not self._agent_scope_allows(operation.get("agent_id"), session_agent_id):
            raise ValueError("Fsck operation not found")
        return operation

    def start_exact_audit(
        self,
        *,
        basis_operation_id: str,
        targets: list[dict[str, Any]],
        user_id: str,
        owner_id: str,
        session_agent_id: str | None,
    ) -> FsckCheck:
        """Create a bounded audit bound to one operation and exact revisions."""
        operation = self._get_authorized_fsck_operation(
            basis_operation_id,
            user_id=user_id,
            owner_id=owner_id,
            session_agent_id=session_agent_id,
        )
        expected = self._operation_target_revisions(operation)
        requested = sorted(
            (
                {"memory_id": str(item["memory_id"]), "revision": int(item["revision"])}
                for item in targets
            ),
            key=lambda item: item["memory_id"],
        )
        expected_request = [
            {"memory_id": item["memory_id"], "revision": item["revision"]}
            for item in expected
        ]
        if requested != expected_request:
            raise ValueError("Fsck operation or target set not found")
        check = self.start_check(
            user_id=user_id,
            owner_id=owner_id,
            agent_id=operation.get("agent_id"),
        )
        check.mode = FSCK_AUDIT_MODE
        check.basis_operation_id = basis_operation_id
        check.basis_operation_fingerprint = self._basis_operation_fingerprint(operation)
        check.audit_issue_type = (operation.get("issue") or {}).get("type")
        check.target_revisions = expected
        return check

    def _load_exact_audit_targets(
        self,
        targets: list[dict[str, Any]],
        *,
        user_id: str,
        owner_id: str,
    ) -> tuple[str, list[dict[str, Any]], str]:
        """Strictly load exact revisions and return state plus snapshot hash."""
        ids = [item["memory_id"] for item in targets]
        memories = self._vector.get_by_ids_strict(ids)
        by_id = {str(memory["id"]): memory for memory in memories}
        state = "complete"
        snapshots: list[dict[str, Any]] = []
        ordered_memories: list[dict[str, Any]] = []
        for target in targets:
            memory_id = target["memory_id"]
            memory = by_id.get(memory_id)
            if memory is None:
                state = "absent"
                snapshots.append({"memory_id": memory_id, "state": "absent"})
                continue
            metadata = memory.get("metadata") or {}
            stale = (
                memory.get("user_id") != user_id
                or (memory.get("owner_id") or memory.get("user_id")) != owner_id
                or memory.get("agent_id") != target.get("agent_id")
                or int(metadata.get("revision", 1)) != int(target["revision"])
                or metadata.get("revision_state", "active") != "active"
                or bool(metadata.get("superseded_by"))
            )
            if stale and state != "absent":
                state = "stale"
            snapshots.append(
                {
                    "memory_id": memory_id,
                    "revision": metadata.get("revision", 1),
                    "lineage_id": metadata.get("lineage_id", memory_id),
                    "revision_state": metadata.get("revision_state", "active"),
                    "superseded_by": metadata.get("superseded_by"),
                    "user_id": memory.get("user_id"),
                    "owner_id": memory.get("owner_id") or memory.get("user_id"),
                    "agent_id": memory.get("agent_id"),
                    "content_hash": memory.get("hash"),
                }
            )
            ordered_memories.append(memory)
        return state, ordered_memories, canonical_fingerprint(snapshots)

    def run_exact_audit(self, check_id: str) -> None:
        """Analyze only bound revisions and persist a content-free audit."""
        check = self._store.get(check_id)
        if check is None or check.mode != FSCK_AUDIT_MODE:
            return
        try:
            state, memories, before_fingerprint = self._load_exact_audit_targets(
                check.target_revisions,
                user_id=check.user_id,
                owner_id=check.owner_id or check.user_id,
            )
            issues: list[FsckIssue] = []
            if state == "complete":
                if check.audit_issue_type in {"duplicate", "contradiction"}:
                    issues = self._evaluate_duplicate_cluster(memories)
                elif check.audit_issue_type == "security":
                    issues = self._phase_security_reeval(
                        self._phase_security_scan_regex(memories)
                    )
                else:
                    issues = self._phase_quality_check(
                        memories,
                        check,
                        available_categories=sorted(
                            set(PREDEFINED_CATEGORIES)
                            | {
                                str(category)
                                for memory in memories
                                for category in (
                                    (memory.get("metadata") or {}).get("categories", [])
                                )
                            }
                        ),
                    )
                after_state, _, after_fingerprint = self._load_exact_audit_targets(
                    check.target_revisions,
                    user_id=check.user_id,
                    owner_id=check.owner_id or check.user_id,
                )
                if after_state != "complete" or after_fingerprint != before_fingerprint:
                    state = "stale"
                    issues = []
            else:
                after_fingerprint = before_fingerprint
            issues.sort(
                key=lambda issue: (
                    issue.type,
                    sorted(action.memory_id or "" for action in issue.actions),
                    issue.issue_id,
                )
            )
            check.issues = issues
            check.target_state = state
            check.target_snapshot_fingerprint = after_fingerprint
            check.summary = self._build_summary(issues)
            check.progress = FsckProgress(
                phase="done",
                total_memories=len(check.target_revisions),
                processed=len(memories),
                percent=100,
                issues_found=len(issues),
            )
            check.status = "completed"
            completed_at = datetime.now(timezone.utc).isoformat()
            signatures = []
            for issue in issues:
                plan = [
                    {
                        "action": action.action,
                        "memory_id": action.memory_id,
                        "new_content": action.new_content,
                        "new_metadata": action.new_metadata,
                    }
                    for action in issue.actions
                ]
                signatures.append(
                    {
                        "issue_id": issue.issue_id,
                        "type": issue.type,
                        "action_target_ids": sorted(
                            action.memory_id
                            for action in issue.actions
                            if action.memory_id
                        ),
                        "action_count": len(plan),
                        "plan_fingerprint": canonical_fingerprint(
                            self._normalized_action_plan(plan)
                        ),
                    }
                )
            self._memory_service.revisions.operations.create_fsck_audit(
                {
                    "audit_check_id": check.check_id,
                    "user_id": check.user_id,
                    "owner_id": check.owner_id or check.user_id,
                    "agent_id": check.agent_id,
                    "basis_operation_id": check.basis_operation_id,
                    "basis_operation_fingerprint": (check.basis_operation_fingerprint),
                    "target_revisions": check.target_revisions,
                    "target_revision_ids": [
                        item["memory_id"] for item in check.target_revisions
                    ],
                    "target_state": state,
                    "target_snapshot_fingerprint": after_fingerprint,
                    "issue_signatures": signatures,
                    "summary": vars(check.summary),
                    "created_at_utc": check.created_at_utc,
                    "completed_at_utc": completed_at,
                }
            )
        except Exception as exc:
            logger.exception("Exact fsck audit %s failed", check_id)
            check.status = "failed"
            check.error = type(exc).__name__

    def run_check(
        self,
        check_id: str,
        *,
        categories: list[str] | None = None,
        memory_type: str | None = None,
        include_raw: bool = False,
        incremental: bool = False,
    ) -> None:
        """Execute the full check pipeline. Called in a background task.

        Updates the FsckCheck in-place with progress, issues, and status.

        Phase weight allocation for percent:
          Phase 0a (security_scan):    0 –  3%  (instant, regex only)
          Phase 0b (security_reeval):  3 –  8%  (LLM call per flagged memory)
          Phase 1a (duplicate_search): 8 – 30%  (vector search per memory)
          Phase 1b (duplicate_eval):  30 – 55%  (LLM call per cluster)
          Phase 2 (quality_check):    55 – 100% (LLM call per batch)
        """
        check = self._store.get(check_id)
        if check is None:
            logger.warning("Fsck check %s not found or expired", check_id)
            return

        try:
            # Fetch all memories
            filters: dict[str, Any] = {}
            if memory_type:
                filters["memory_type"] = memory_type

            # When agent_id is set, perform dual-scope scroll: fetch both
            # agent-scoped AND shared (no agent_id) memories. This allows
            # fsck to detect cross-scope duplicates (e.g., same fact stored
            # as both agent-scoped and shared).
            memories = self._vector.scroll_with_vectors(
                user_id=check.user_id,
                owner_id=check.owner_id or check.user_id,
                agent_id=check.agent_id,
                filters=filters,
                exclude_layers=None if include_raw else ["raw"],
                exclude_expired=True,
            )
            if check.agent_id:
                shared_memories = self._vector.scroll_with_vectors(
                    user_id=check.user_id,
                    owner_id=check.owner_id or check.user_id,
                    agent_id=None,
                    shared_only=True,
                    filters=filters,
                    exclude_layers=None if include_raw else ["raw"],
                    exclude_expired=True,
                )
                # Merge and deduplicate by memory ID
                seen_ids = {m["id"] for m in memories}
                for m in shared_memories:
                    if m["id"] not in seen_ids:
                        seen_ids.add(m["id"])
                        memories.append(m)

            # Filter by categories if specified (client-side since Qdrant
            # MatchAny on array fields needs special handling)
            if categories:
                cat_set = set(categories)
                memories = [
                    m
                    for m in memories
                    if cat_set.intersection(
                        (m.get("metadata") or {}).get("categories", [])
                    )
                ]

            # Incremental mode: only check memories changed since last maintenance.
            # A memory is "changed" if it has no checked_at or its updated_at_utc
            # is more recent than checked_at. checked_at is stamped all-or-nothing
            # after a memory passes through ALL enabled phases.
            if incremental:
                full_corpus = list(
                    memories
                )  # Keep full corpus for duplicate neighbor search
                memories = [
                    m
                    for m in memories
                    if not (m.get("metadata") or {}).get("checked_at")
                    or (m.get("metadata") or {}).get("updated_at_utc", "")
                    > (m.get("metadata") or {}).get("checked_at", "")
                ]
                logger.info(
                    "Fsck check %s: incremental mode — %d changed out of %d total",
                    check_id,
                    len(memories),
                    len(full_corpus),
                )
            else:
                full_corpus = list(memories)

            # Cap the working set after incremental selection. Keep the full
            # eligible corpus for nearest-neighbor resolution.
            max_memories = self._config.memory.fsck_max_memories
            if max_memories > 0 and len(memories) > max_memories:
                import random

                logger.info(
                    "Fsck check %s: capping %d candidates to %d (random sample)",
                    check_id,
                    len(memories),
                    max_memories,
                )
                memories = random.sample(memories, max_memories)
                check.progress.truncated = True

            total = len(memories)
            check.progress.total_memories = total
            logger.info("Fsck check %s: %d memories to check", check_id, total)

            if total == 0:
                check.status = "completed"
                check.summary = FsckSummary()
                check.progress.phase = "done"
                check.progress.percent = 100
                return

            # mem_by_id is the working set for phases that only need changed
            # memories.  full_corpus_by_id includes ALL non-raw memories and
            # is used by duplicate detection so that a changed memory can be
            # matched against an unchanged clean memory.
            mem_by_id: dict[str, dict] = {m["id"]: m for m in memories}
            full_corpus_by_id: dict[str, dict] = (
                {m["id"]: m for m in full_corpus}
                if full_corpus is not memories
                else mem_by_id
            )
            check.issues.extend(self._phase_validation_projection(memories))

            # Phase 0a: Security scan (regex, no LLM) — 0-3%
            check.progress.phase = "security_scan"
            check.progress.processed = 0
            logger.info(
                "Fsck check %s phase 0a: scanning %d memories for injection patterns",
                check_id,
                total,
            )
            regex_flagged = self._phase_security_scan_regex(memories)
            check.progress.processed = total
            check.progress.percent = 3

            # Phase 0b: Security re-evaluation (LLM) — 3-8%
            # Re-evaluate regex hits to drop false positives before adding issues.
            logger.info(
                "Fsck check %s phase 0b: re-evaluating %d security flags with LLM",
                check_id,
                len(regex_flagged),
            )
            security_issues = self._phase_security_reeval(regex_flagged)
            check.issues.extend(security_issues)
            check.progress.percent = 8
            check.progress.issues_found = len(check.issues)

            # Estimate LLM calls for budget tracking
            check.llm_calls += len(regex_flagged)  # 1 call per flagged memory

            # Phases execute in order; under budget pressure, earlier phases
            # take precedence. Later phases catch up over subsequent
            # incremental runs.
            if self._check_budget(check):
                logger.info(
                    "Fsck check %s: LLM budget exhausted (%d calls), stopping after security phase",
                    check_id,
                    check.llm_calls,
                )
                check.budget_exhausted = True
                self._stamp_clean_memories(check, memories, check_id)
                check.progress.phase = "done"
                check.summary = self._build_summary(check.issues)
                check.status = "completed"
                return

            # Phase 1: Duplicate detection (vector similarity + LLM) — 8-55%
            logger.info(
                "Fsck check %s phase 1: duplicate detection on %d memories",
                check_id,
                total,
            )
            dup_issues, clustered_ids = self._phase_duplicate_detection(
                memories, mem_by_id, check, full_corpus_by_id=full_corpus_by_id
            )
            check.issues.extend(dup_issues)
            check.progress.issues_found = len(check.issues)

            # Estimate LLM calls for duplicate evaluation (1 per cluster).
            # clustered_ids is a flat set; average cluster size ~2-3 members,
            # so number of clusters ≈ len(clustered_ids) / 2.
            estimated_dup_clusters = (
                max(len(clustered_ids) // 2, 1) if clustered_ids else 0
            )
            check.llm_calls += estimated_dup_clusters

            if self._check_budget(check):
                logger.info(
                    "Fsck check %s: LLM budget exhausted (%d calls), stopping after duplicate phase",
                    check_id,
                    check.llm_calls,
                )
                check.budget_exhausted = True
                self._stamp_clean_memories(check, memories, check_id)
                check.progress.phase = "done"
                check.summary = self._build_summary(check.issues)
                check.status = "completed"
                return

            # Phase 2: Quality check (LLM batches) — 55-100%
            # Only exclude memories that were part of confirmed duplicate/contradiction
            # issues. Memories in clusters that the LLM declared clean should still
            # reach quality review.
            confirmed_issue_mem_ids: set[str] = set()
            for issue in dup_issues:
                for am in issue.affected_memories:
                    confirmed_issue_mem_ids.add(am.id)
            quality_memories = [
                m for m in memories if m["id"] not in confirmed_issue_mem_ids
            ]
            logger.info(
                "Fsck check %s phase 2: quality check on %d memories (%d skipped, in confirmed issues)",
                check_id,
                len(quality_memories),
                len(memories) - len(quality_memories),
            )
            # Load available categories for this user so the LLM only proposes valid ones
            available_categories = self._get_available_categories(check.user_id)
            quality_issues = self._phase_quality_check(
                quality_memories, check, available_categories=available_categories
            )
            check.issues.extend(quality_issues)
            check.progress.issues_found = len(check.issues)
            check.progress.percent = 100

            # Estimate LLM calls for quality check (2 passes per batch:
            # content quality + metadata normalization).
            quality_batches = (
                (len(quality_memories) + _QUALITY_BATCH_SIZE - 1) // _QUALITY_BATCH_SIZE
                if quality_memories
                else 0
            )
            check.llm_calls += (
                quality_batches * 2
            )  # 2 LLM calls per batch (Pass A + Pass B)

            self._stamp_clean_memories(check, memories, check_id)

            # Build summary
            check.progress.phase = "done"
            check.summary = self._build_summary(check.issues)
            check.status = "completed"

            logger.info(
                "Fsck check %s completed: %d memories, %d issues found",
                check_id,
                total,
                len(check.issues),
            )

        except Exception as e:
            logger.exception("Fsck check %s failed", check_id)
            check.status = "failed"
            check.error = str(e)

    def get_check(
        self,
        check_id: str,
        *,
        user_id: str | None = None,
        owner_id: str | None = None,
        session_agent_id: str | None = None,
    ) -> FsckCheck | None:
        """Get a check by ID."""
        check = self._store.get(check_id)
        if check is not None:
            if user_id is not None and check.user_id != user_id:
                return None
            if owner_id is not None and (check.owner_id or check.user_id) != owner_id:
                return None
            if not self._agent_scope_allows(check.agent_id, session_agent_id):
                return None
            return check
        if self._memory_service is None:
            return None
        records = self._memory_service.revisions.operations.list_fsck(
            check_id,
            user_id=user_id,
        )
        if not records:
            audit = self._memory_service.revisions.operations.get_fsck_audit(check_id)
            if audit is None:
                return None
            if user_id is not None and audit.get("user_id") != user_id:
                return None
            if (
                owner_id is not None
                and audit.get("owner_id") != owner_id
                or not self._agent_scope_allows(audit.get("agent_id"), session_agent_id)
            ):
                return None
            summary_payload = audit.get("summary") or {}
            check = FsckCheck(
                check_id=check_id,
                user_id=audit["user_id"],
                owner_id=audit["owner_id"],
                agent_id=audit.get("agent_id"),
                status="completed",
                summary=FsckSummary(**summary_payload),
                mode=FSCK_AUDIT_MODE,
                basis_operation_id=audit.get("basis_operation_id"),
                basis_operation_fingerprint=audit.get("basis_operation_fingerprint"),
                target_revisions=list(audit.get("target_revisions") or []),
                target_state=audit.get("target_state"),
                target_snapshot_fingerprint=audit.get("target_snapshot_fingerprint"),
            )
            check.progress = FsckProgress(
                phase="done",
                total_memories=len(check.target_revisions),
                processed=len(check.target_revisions),
                percent=100,
                issues_found=int(summary_payload.get("total", 0)),
            )
            self._store.put(check)
            return check
        if owner_id is not None:
            records = [
                record
                for record in records
                if record.get("owner_id") == owner_id
                and self._agent_scope_allows(
                    record.get("agent_id"),
                    session_agent_id,
                )
            ]
        if not records:
            return None
        issues = []
        applied_ids = set()
        for record in records:
            issue_payload = record.get("issue")
            if not isinstance(issue_payload, dict):
                continue
            issues.append(
                FsckIssue(
                    issue_id=issue_payload["issue_id"],
                    type=issue_payload["type"],
                    severity=issue_payload["severity"],
                    reasoning=issue_payload["reasoning"],
                    affected_memories=[
                        FsckAffectedMemory(**memory)
                        for memory in issue_payload.get("affected_memories", [])
                    ],
                    actions=[
                        FsckAction(**action)
                        for action in issue_payload.get("actions", [])
                    ],
                    confidence=issue_payload.get("confidence"),
                )
            )
            if record.get("status") == "committed":
                applied_ids.add(issue_payload["issue_id"])
        if not issues:
            return None
        check = FsckCheck(
            check_id=check_id,
            user_id=records[0]["user_id"],
            owner_id=records[0].get("owner_id") or records[0]["user_id"],
            agent_id=records[0].get("agent_id"),
            status="completed",
            issues=issues,
            applied_issue_ids=applied_ids,
        )
        self._store.put(check)
        return check

    @staticmethod
    def _agent_scope_allows(
        check_agent_id: str | None,
        session_agent_id: str | None,
    ) -> bool:
        """Return whether a session agent can access a scoped fsck check."""
        if check_agent_id is None:
            return True
        if session_agent_id is None:
            return False
        return check_agent_id == session_agent_id or check_agent_id.startswith(
            session_agent_id + ":"
        )

    def apply_check(
        self,
        check_id: str,
        issue_ids: list[str] | None = None,
        *,
        user_id: str | None = None,
        owner_id: str | None = None,
        session_agent_id: str | None = None,
    ) -> dict[str, Any]:
        """Apply fixes from a completed check.

        Idempotent: issues that have already been applied are skipped
        (tracked in check.applied_issue_ids). Safe to call multiple times
        with different issue_ids to apply fixes incrementally.

        Args:
            check_id: The check to apply.
            issue_ids: Specific issue IDs to apply. None/empty = apply all.

        Returns:
            Dict with applied/skipped/failed counts and details.
        """
        check = self.get_check(
            check_id,
            user_id=user_id,
            owner_id=owner_id,
            session_agent_id=session_agent_id,
        )
        if check is None:
            return {
                "error": True,
                "message": "Check not found or expired. Please re-run the check.",
            }
        if check.mode == FSCK_AUDIT_MODE:
            return {
                "error": True,
                "message": "Audit-only fsck checks cannot apply actions",
            }
        if check.status not in ("completed", "applying"):
            return {
                "error": True,
                "message": f"Check is not completed (status: {check.status})",
            }

        # Select issues to apply
        if issue_ids:
            id_set = set(issue_ids)
            issues = [i for i in check.issues if i.issue_id in id_set]
        else:
            issues = list(check.issues)

        applied = 0
        skipped = 0
        failed = 0
        details: list[dict] = []

        for issue in issues:
            # Idempotency: skip issues that were already applied.
            if issue.issue_id in check.applied_issue_ids:
                skipped += 1
                details.append(
                    {
                        "issue_id": issue.issue_id,
                        "status": "skipped",
                        "actions_executed": 0,
                        "actions_skipped": 0,
                    }
                )
                continue

            operation_token = None
            operation_claimed = False
            checkpoints: list[dict[str, Any]] = []
            plan: list[dict[str, Any]] = []
            try:
                if self._memory_service is not None:
                    from mnemory.revisions import canonical_fingerprint

                    sources = [
                        self._vector.get_by_id(memory.id)
                        for memory in issue.affected_memories
                    ]
                    check_sources = {
                        memory.id: memory.metadata or {}
                        for memory in issue.affected_memories
                    }
                    for index, action in enumerate(issue.actions):
                        source = (
                            self._vector.get_by_id(action.memory_id)
                            if action.memory_id
                            else None
                        )
                        metadata = check_sources.get(action.memory_id or "", {})
                        result_revision_id = (
                            str(
                                uuid.uuid5(
                                    uuid.NAMESPACE_URL,
                                    (
                                        f"mnemory:fsck:{check.check_id}:"
                                        f"{issue.issue_id}:{action.new_content}"
                                    ),
                                )
                            )
                            if action.action == "add"
                            else None
                        )
                        plan.append(
                            {
                                "action": action.action,
                                "memory_id": action.memory_id,
                                "new_content": action.new_content,
                                "new_metadata": action.new_metadata,
                                "expected_revision": int(metadata.get("revision", 1))
                                if source
                                else None,
                                "expected_revision_id": (
                                    action.memory_id if source else None
                                ),
                                "result_revision_id": result_revision_id,
                                "action_id": canonical_fingerprint(
                                    [
                                        check.check_id,
                                        issue.issue_id,
                                        index,
                                        action.action,
                                        action.memory_id,
                                        action.new_content,
                                        action.new_metadata,
                                    ]
                                ),
                            }
                        )
                    operation_token = canonical_fingerprint(
                        ["fsck", check.check_id, issue.issue_id, plan]
                    )
                    operations = self._memory_service.revisions.operations
                    existing_operation = operations.get(operation_token)
                    if existing_operation and existing_operation.get("status") in {
                        "committed",
                        "skipped",
                        "superseded",
                    }:
                        skipped += 1
                        details.append(
                            {
                                "issue_id": issue.issue_id,
                                "status": "skipped",
                                "actions_executed": 0,
                                "actions_skipped": len(plan),
                            }
                        )
                        continue
                    if check.owner_id is None:
                        raise RuntimeError("Fsck check owner scope is missing")
                    audit_owner_id = check.owner_id
                    affected_lineage_ids = list(
                        dict.fromkeys(
                            (
                                (source_memory.get("metadata") or {}).get(
                                    "lineage_id", memory.id
                                )
                                if source_memory
                                else memory.id
                            )
                            for memory, source_memory in zip(
                                issue.affected_memories, sources, strict=True
                            )
                        )
                    )
                    checkpoints = list(
                        (existing_operation or {}).get("action_checkpoints") or []
                    ) or [
                        {
                            "action_id": item["action_id"],
                            "action_index": index,
                            "action_kind": item["action"],
                            "expected_revision_id": item["expected_revision_id"],
                            "expected_revision": item["expected_revision"],
                            "status": "pending",
                        }
                        for index, item in enumerate(plan)
                    ]
                    if existing_operation is None:
                        operations.write(
                            operation_token,
                            {
                                "status": "planned",
                                "operation_kind": "fsck",
                                "actor_kind": "fsck",
                                "user_id": check.user_id,
                                "owner_id": audit_owner_id,
                                "agent_id": check.agent_id,
                                "lineage_id": (
                                    (issue.affected_memories[0].metadata or {}).get(
                                        "lineage_id", issue.affected_memories[0].id
                                    )
                                    if issue.affected_memories
                                    else f"fsck:{check.check_id}"
                                ),
                                "fsck_check_id": check.check_id,
                                "fsck_issue_id": issue.issue_id,
                                "issue": {
                                    "issue_id": issue.issue_id,
                                    "type": issue.type,
                                    "severity": issue.severity,
                                    "reasoning": issue.reasoning,
                                    "confidence": issue.confidence,
                                    "affected_memories": [
                                        {
                                            "id": memory.id,
                                            "content": "",
                                            "metadata": memory.metadata,
                                            "agent_id": memory.agent_id,
                                        }
                                        for memory in issue.affected_memories
                                    ],
                                    "actions": [
                                        {
                                            "action": action.action,
                                            "memory_id": action.memory_id,
                                            "new_content": action.new_content,
                                            "new_metadata": action.new_metadata,
                                        }
                                        for action in issue.actions
                                    ],
                                },
                                "issue_fingerprint": canonical_fingerprint(
                                    [
                                        issue.type,
                                        sorted(affected_lineage_ids),
                                        plan,
                                    ]
                                ),
                                "plan_hash": canonical_fingerprint(plan),
                                "affected_lineage_ids": affected_lineage_ids,
                                "plan": plan,
                                "action_checkpoints": checkpoints,
                            },
                        )
                    claimant = str(uuid.uuid4())
                    lease_seconds = (
                        self._config.memory.fsck_recovery_lease_seconds
                        if isinstance(
                            getattr(
                                self._config.memory,
                                "fsck_recovery_lease_seconds",
                                None,
                            ),
                            int,
                        )
                        else 300
                    )
                    max_attempts = getattr(
                        self._config.memory,
                        "fsck_recovery_max_attempts",
                        3,
                    )
                    if not isinstance(max_attempts, int):
                        max_attempts = 3
                    if (
                        int((existing_operation or {}).get("recovery_attempt_count", 0))
                        >= max_attempts
                    ):
                        exhausted = operations.terminalize_unclaimed(
                            operation_token,
                            status="failed",
                            payload={
                                "error_class": "RecoveryAttemptsExhausted",
                                "error_at": datetime.now(timezone.utc).isoformat(),
                                "terminal_reason": "recovery_attempts_exhausted",
                            },
                        )
                        if not exhausted:
                            skipped += 1
                            details.append(
                                {
                                    "issue_id": issue.issue_id,
                                    "status": "skipped",
                                    "actions_executed": 0,
                                    "actions_skipped": len(plan),
                                    "error": "Operation is already claimed",
                                }
                            )
                            continue
                        failed += 1
                        details.append(
                            {
                                "issue_id": issue.issue_id,
                                "status": "failed",
                                "actions_executed": 0,
                                "actions_skipped": 0,
                                "error": "Recovery attempts exhausted",
                            }
                        )
                        continue
                    if not operations.claim(
                        operation_token,
                        claimant=claimant,
                        lease_seconds=lease_seconds,
                        allowed_statuses=("planned", "failed", "applying"),
                    ):
                        skipped += 1
                        details.append(
                            {
                                "issue_id": issue.issue_id,
                                "status": "skipped",
                                "actions_executed": 0,
                                "actions_skipped": len(plan),
                            }
                        )
                        continue
                    operation_claimed = True
                    checkpoints = self._resolve_action_checkpoints(
                        plan,
                        checkpoints,
                        complete=False,
                        check_id=check.check_id,
                        issue_id=issue.issue_id,
                    )
                    if not operations.write_claimed(
                        operation_token,
                        claimant,
                        {"action_checkpoints": checkpoints},
                    ):
                        raise RuntimeError("Fsck operation lease was lost")
                    pending_indexes = {
                        index
                        for index, checkpoint in enumerate(checkpoints)
                        if checkpoint.get("status") not in {"committed", "skipped"}
                    }
                    stale = False
                    for index, item in enumerate(plan):
                        if index not in pending_indexes:
                            continue
                        target = item.get("memory_id")
                        if target is None or item.get("expected_revision") is None:
                            continue
                        current = self._vector.get_by_id_strict(target)
                        if not isinstance(current, dict):
                            current = self._vector.get_by_id(target)
                        metadata = (current or {}).get("metadata") or {}
                        if (
                            current is None
                            or current.get("id") != item.get("expected_revision_id")
                            or metadata.get("revision_state", "active") != "active"
                            or int(metadata.get("revision", 1))
                            != int(item["expected_revision"])
                        ):
                            stale = True
                            break
                    if stale:
                        if not operations.write_claimed(
                            operation_token,
                            claimant,
                            {
                                "status": "superseded",
                                "terminal_reason": "target_revision_changed",
                                "terminal_at": datetime.now(timezone.utc).isoformat(),
                            },
                        ):
                            raise RuntimeError("Fsck operation lease was lost")
                        skipped += 1
                        details.append(
                            {
                                "issue_id": issue.issue_id,
                                "status": "skipped",
                                "actions_executed": 0,
                                "actions_skipped": len(plan),
                            }
                        )
                        continue

                    before_action = partial(
                        self._start_action_checkpoint,
                        operations,
                        operation_token,
                        claimant,
                        checkpoints,
                        lease_seconds,
                    )
                    on_action = partial(
                        self._finish_action_checkpoint,
                        operations,
                        operation_token,
                        claimant,
                        checkpoints,
                    )

                actions_executed, actions_skipped = self._apply_issue(
                    issue,
                    check.user_id,
                    check_id=check.check_id,
                    expected_revisions={
                        item["memory_id"]: item["expected_revision"]
                        for item in plan
                        if item.get("memory_id")
                        and item.get("expected_revision") is not None
                    },
                    action_indexes=(
                        pending_indexes if operation_token is not None else None
                    ),
                    before_action=before_action
                    if operation_token is not None
                    else None,
                    on_action=on_action if operation_token is not None else None,
                )
                if operation_token and self._memory_service is not None:
                    checkpoints = self._resolve_action_checkpoints(
                        plan,
                        checkpoints,
                        complete=True,
                        check_id=check.check_id,
                        issue_id=issue.issue_id,
                    )
                    committed_count = sum(
                        checkpoint.get("status") == "committed"
                        for checkpoint in checkpoints
                    )
                    skipped_count = sum(
                        checkpoint.get("status") == "skipped"
                        for checkpoint in checkpoints
                    )
                    pending_count = len(checkpoints) - committed_count - skipped_count
                    if pending_count:
                        raise RuntimeError("Fsck action journal is incomplete")
                    outcome = "committed" if committed_count else "skipped"
                    if not self._memory_service.revisions.operations.write_claimed(
                        operation_token,
                        claimant,
                        {
                            "status": outcome,
                            "actions_executed": committed_count,
                            "actions_skipped": skipped_count,
                            "action_checkpoints": checkpoints,
                            "terminal_at": datetime.now(timezone.utc).isoformat(),
                        },
                    ):
                        raise RuntimeError("Fsck operation lease was lost")
                    actions_executed = committed_count
                    actions_skipped = skipped_count
                    from mnemory.metrics import get_collector

                    collector = get_collector()
                    if collector:
                        collector.record_fsck_operation(outcome)
                        if existing_operation is not None:
                            collector.record_fsck_recovery(outcome)
                # Only mark as applied (and prevent retry) when at least one
                # action actually executed. If all actions were skipped (e.g.,
                # memory not found), leave the issue available for future apply
                # attempts so transient misses don't permanently block fixes.
                if actions_executed > 0:
                    check.applied_issue_ids.add(issue.issue_id)
                    applied += 1
                    status = "applied"
                else:
                    skipped += 1
                    status = "skipped"
                details.append(
                    {
                        "issue_id": issue.issue_id,
                        "status": status,
                        "actions_executed": actions_executed,
                        "actions_skipped": actions_skipped,
                    }
                )
            except BaseException as e:
                if operation_token and self._memory_service is not None:
                    checkpoints = self._resolve_action_checkpoints(
                        plan,
                        checkpoints,
                        complete=False,
                        check_id=check.check_id,
                        issue_id=issue.issue_id,
                    )
                    if operation_claimed:
                        terminalized = (
                            self._memory_service.revisions.operations.write_claimed(
                                operation_token,
                                claimant,
                                {
                                    "status": "failed",
                                    "error_class": type(e).__name__,
                                    "error_at": datetime.now(timezone.utc).isoformat(),
                                    "terminal_at": datetime.now(
                                        timezone.utc
                                    ).isoformat(),
                                    "terminal_reason": "apply_exception",
                                    "action_checkpoints": checkpoints,
                                },
                            )
                        )
                    else:
                        terminalized = self._memory_service.revisions.operations.terminalize_unclaimed(
                            operation_token,
                            status="failed",
                            payload={
                                "error_class": type(e).__name__,
                                "error_at": datetime.now(timezone.utc).isoformat(),
                                "terminal_reason": "apply_exception",
                                "action_checkpoints": checkpoints,
                            },
                        )
                    from mnemory.metrics import get_collector

                    collector = get_collector()
                    if collector and terminalized:
                        collector.record_fsck_operation("failed")
                        if existing_operation is not None:
                            collector.record_fsck_recovery("failed")
                logger.warning(
                    "Failed to apply fsck issue %s: %s",
                    issue.issue_id,
                    e,
                )
                failed += 1
                details.append(
                    {
                        "issue_id": issue.issue_id,
                        "status": "failed",
                        "actions_executed": 0,
                        "actions_skipped": 0,
                        "error": str(e),
                    }
                )
                if not isinstance(e, Exception):
                    raise

        # Invalidate caches so mutations are reflected immediately.
        if self._memory_service is not None and (applied > 0 or failed > 0):
            try:
                self._memory_service._core_cache.invalidate_prefix(check.user_id)
                self._memory_service._category_cache.invalidate(check.user_id)
            except Exception:
                logger.warning(
                    "Fsck apply: failed to invalidate caches for user %s",
                    check.user_id,
                    exc_info=True,
                )

        return {
            "applied": applied,
            "skipped": skipped,
            "failed": failed,
            "details": details,
        }

    def _resolve_action_checkpoints(
        self,
        plan: list[dict[str, Any]],
        checkpoints: list[dict[str, Any]],
        *,
        complete: bool,
        check_id: str,
        issue_id: str,
    ) -> list[dict[str, Any]]:
        """Resolve checkpoints only from exact operation-owned side effects."""
        del complete
        now = datetime.now(timezone.utc).isoformat()
        resolved = []
        for item, checkpoint in zip(plan, checkpoints, strict=True):
            status = checkpoint.get("status", "pending")
            if status in {"committed", "skipped"}:
                resolved.append(checkpoint)
                continue
            action = item.get("action")
            target = (
                item.get("result_revision_id")
                if action == "add"
                else (item.get("expected_revision_id") or item.get("memory_id"))
            )
            point = self._vector.get_by_id_strict(target) if target else None
            if point is not None and not isinstance(point, dict):
                point = self._vector.get_by_id(target)
            metadata = (point or {}).get("metadata") or {}
            subordinate_operation_id = checkpoint.get("subordinate_operation_id")
            if subordinate_operation_id is None and action in {"update", "delete"}:
                subordinate_operation_id = metadata.get("revision_operation_id")
            applied = False
            if action == "add":
                applied = bool(
                    point
                    and point.get("id") == item.get("result_revision_id")
                    and metadata.get("revision_state", "active") == "active"
                )
            elif action == "delete" and metadata.get("revision_operation_id") == (
                f"fsck:{check_id}:{issue_id}"
            ):
                applied = metadata.get("revision_state") == "source"
                subordinate_operation_id = f"fsck:{check_id}:{issue_id}"
            elif subordinate_operation_id and self._memory_service is not None:
                operation = (
                    self._memory_service.revisions.operations.get_by_id(
                        subordinate_operation_id
                    )
                    or {}
                )
                applied = bool(
                    operation.get("status") == "committed"
                    and operation.get("fsck_check_id") == check_id
                    and operation.get("fsck_issue_id") == issue_id
                    and operation.get("previous_revision_id")
                    == item.get("expected_revision_id")
                )
            if applied:
                status = "committed"
            resolved.append(
                {
                    **checkpoint,
                    "status": status,
                    "subordinate_operation_id": subordinate_operation_id,
                    **(
                        {"completed_at": now}
                        if status in {"committed", "skipped"}
                        else {}
                    ),
                }
            )
        return resolved

    @staticmethod
    def _start_action_checkpoint(
        operations: Any,
        operation_token: str,
        claimant: str,
        checkpoints: list[dict[str, Any]],
        lease_seconds: int,
        action_index: int,
    ) -> None:
        """Fence one action with a lease-owned applying checkpoint."""
        if not operations.renew_claim(
            operation_token,
            claimant,
            lease_seconds=lease_seconds,
        ):
            raise RuntimeError("Fsck operation lease was lost")
        checkpoints[action_index] = {
            **checkpoints[action_index],
            "status": "applying",
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        if not operations.write_claimed(
            operation_token,
            claimant,
            {"action_checkpoints": checkpoints},
        ):
            raise RuntimeError("Fsck operation lease was lost")

    @staticmethod
    def _finish_action_checkpoint(
        operations: Any,
        operation_token: str,
        claimant: str,
        checkpoints: list[dict[str, Any]],
        action_index: int,
        status: str,
        subordinate_operation_id: str | None,
    ) -> None:
        """Persist one action outcome while the caller owns the lease."""
        checkpoints[action_index] = {
            **checkpoints[action_index],
            "status": status,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "subordinate_operation_id": subordinate_operation_id,
        }
        if not operations.write_claimed(
            operation_token,
            claimant,
            {"action_checkpoints": checkpoints},
        ):
            raise RuntimeError("Fsck operation lease was lost")

    def re_evaluate_operation(
        self,
        operation_id: str,
        *,
        fresh_check_id: str,
        user_id: str,
        owner_id: str,
        session_agent_id: str | None,
        terminalize: bool = False,
    ) -> dict[str, Any]:
        """Compare an old fsck operation with a separately generated check."""
        operation = self._get_authorized_fsck_operation(
            operation_id,
            user_id=user_id,
            owner_id=owner_id,
            session_agent_id=session_agent_id,
        )
        operations = self._memory_service.revisions.operations
        audit = operations.get_fsck_audit(fresh_check_id)
        expected_targets = self._operation_target_revisions(operation)
        basis_fingerprint = self._basis_operation_fingerprint(operation)
        if (
            audit is None
            or audit.get("status") != "committed"
            or audit.get("user_id") != user_id
            or audit.get("owner_id") != owner_id
            or audit.get("agent_id") != operation.get("agent_id")
            or audit.get("basis_operation_id") != operation_id
            or audit.get("basis_operation_fingerprint") != basis_fingerprint
            or audit.get("target_revisions") != expected_targets
            or str(audit.get("created_at_utc") or "")
            <= str(operation.get("created_at_utc") or "")
        ):
            raise ValueError("Fresh exact fsck audit is not available")

        current_state, current_memories, current_fingerprint = (
            self._load_exact_audit_targets(
                expected_targets,
                user_id=user_id,
                owner_id=owner_id,
            )
        )
        old_plan = operation.get("plan") or []
        old_targets = sorted(
            item.get("memory_id")
            for item in old_plan
            if isinstance(item, dict) and item.get("memory_id")
        )
        old_type = (operation.get("issue") or {}).get("type")
        candidates = [
            signature
            for signature in audit.get("issue_signatures") or []
            if signature.get("type") == old_type
            and signature.get("action_target_ids") == old_targets
        ]
        audit_state = audit.get("target_state")
        if audit_state == "absent" or current_state == "absent":
            outcome = "absent"
        elif (
            audit_state != "complete"
            or current_state != "complete"
            or current_fingerprint != audit.get("target_snapshot_fingerprint")
        ):
            outcome = "stale"
        else:
            outcome = "absent"
        fresh_issue_id = None
        fresh_action_count = 0
        if outcome == "absent" and audit_state == "complete" and candidates:
            normalized_old_fingerprint = canonical_fingerprint(
                self._normalized_action_plan(old_plan)
            )
            matching = next(
                (
                    candidate
                    for candidate in candidates
                    if candidate.get("plan_fingerprint") == normalized_old_fingerprint
                ),
                None,
            )
            fresh_issue = matching or candidates[0]
            fresh_issue_id = fresh_issue.get("issue_id")
            fresh_action_count = int(fresh_issue.get("action_count", 0))
            outcome = "still_valid" if matching is not None else "changed"
        if current_state == "complete":
            stale_targets = 0
        elif current_state == "absent":
            stale_targets = max(len(expected_targets) - len(current_memories), 1)
        else:
            stale_targets = 1

        if terminalize:
            if operation.get("status") in {"committed", "skipped", "superseded"}:
                raise ValueError("Fsck operation is already terminal")
            terminalized = operations.terminalize_unclaimed(
                operation["operation_token"],
                status="superseded",
                payload={
                    "terminal_reason": f"re_evaluated_{outcome}",
                    "fresh_check_id": fresh_check_id,
                    "fresh_issue_id": fresh_issue_id,
                },
            )
            if not terminalized:
                raise RuntimeError("Fsck operation is currently leased")

        return {
            "operation_id": operation_id,
            "outcome": outcome,
            "terminalized": terminalize,
            "old_action_count": len(old_plan),
            "fresh_action_count": fresh_action_count,
            "stale_target_count": stale_targets,
            "fresh_check_id": fresh_check_id,
            "fresh_issue_id": fresh_issue_id,
        }

    # ── Phase 0: deterministic projection and security checks ───────

    def _phase_validation_projection(self, memories: list[dict]) -> list[FsckIssue]:
        """Detect rebuildable validation projection mismatches."""
        from mnemory.revisions import canonical_fingerprint

        configured_roots = getattr(self._config.memory, "validation_max_score_roots", 3)
        max_roots = max(configured_roots if isinstance(configured_roots, int) else 3, 1)
        issues: list[FsckIssue] = []
        for memory in memories:
            metadata = memory.get("metadata") or {}
            if not (
                metadata.get("evidence_root_ids")
                or metadata.get("validation_count")
                or metadata.get("validation_state") == "confirmed"
            ):
                continue
            can_carry_validation = (
                metadata.get("validation_eligible")
                or metadata.get("validation_state") == "confirmed"
            )
            roots = (
                list(dict.fromkeys(metadata.get("evidence_root_ids") or []))
                if can_carry_validation
                else []
            )
            expected_count = min(max(len(roots) - 1, 0), max_roots)
            expected = {
                "evidence_root_ids": roots,
                "validation_count": expected_count,
                "validation_strength": expected_count / max_roots,
                "validation_projection_hash": canonical_fingerprint(roots),
                "validation_state": ("confirmed" if expected_count else "unverified"),
            }
            if all(metadata.get(key) == value for key, value in expected.items()):
                continue
            issues.append(
                FsckIssue(
                    issue_id=str(uuid.uuid4()),
                    type="quality",
                    severity="low",
                    reasoning="Validation projection metadata is inconsistent",
                    affected_memories=[
                        FsckAffectedMemory(
                            id=memory["id"],
                            content=memory.get("memory", ""),
                            metadata=metadata,
                            agent_id=memory.get("agent_id"),
                        )
                    ],
                    # Projection drift is report-only. A confirmation can
                    # change roots between scan and apply, so an fsck proposal
                    # must not write an older projection without a claim.
                    actions=[],
                    confidence=1.0,
                )
            )
        return issues

    def _phase_security_scan_regex(
        self,
        memories: list[dict],
    ) -> list[tuple[dict, list[str]]]:
        """Scan all memories for prompt injection patterns using regex.

        Fast, no LLM cost. Uses the existing detect_injection_patterns()
        from sanitize.py.

        Returns a list of (memory, matched_patterns) tuples for flagged memories.
        """
        flagged: list[tuple[dict, list[str]]] = []

        for mem in memories:
            text = mem.get("memory", "")
            patterns = detect_injection_patterns(text)
            if patterns:
                flagged.append((mem, patterns))

        if flagged:
            logger.info(
                "Fsck security scan: %d memories flagged by regex (pending LLM re-eval)",
                len(flagged),
            )

        return flagged

    def _phase_security_reeval(
        self,
        flagged: list[tuple[dict, list[str]]],
    ) -> list[FsckIssue]:
        """Re-evaluate regex-flagged memories with LLM to drop false positives.

        Runs in parallel using the configured concurrency. Only confirmed
        threats become FsckIssue objects — false positives are silently dropped.
        """
        if not flagged:
            return []

        concurrency = max(1, self._config.memory.fsck_llm_concurrency)
        issues: list[FsckIssue] = []
        lock = threading.Lock()

        def _reeval(mem: dict, patterns: list[str]) -> FsckIssue | None:
            return self._evaluate_security_flag(mem, patterns)

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(_reeval, mem, patterns): (mem, patterns)
                for mem, patterns in flagged
            }
            for future in as_completed(futures):
                mem, patterns = futures[future]
                try:
                    issue = future.result()
                    if issue is not None:
                        with lock:
                            issues.append(issue)
                except Exception:
                    logger.warning(
                        "Failed to re-evaluate security flag for memory %s",
                        mem.get("id"),
                        exc_info=True,
                    )
                    # On LLM failure, include the issue conservatively
                    with lock:
                        issues.append(
                            FsckIssue(
                                issue_id=str(uuid.uuid4()),
                                type="security",
                                severity="high",
                                reasoning=(
                                    f"Prompt injection patterns detected: {', '.join(patterns)}. "
                                    "LLM re-evaluation failed — flagged conservatively."
                                ),
                                affected_memories=[self._make_affected_memory(mem)],
                                actions=[
                                    FsckAction(action="delete", memory_id=mem["id"])
                                ],
                            )
                        )

        confirmed = len(issues)
        dropped = len(flagged) - confirmed
        logger.info(
            "Fsck security re-eval: %d confirmed threats, %d false positives dropped",
            confirmed,
            dropped,
        )
        return issues

    def _evaluate_security_flag(
        self,
        mem: dict,
        patterns: list[str],
    ) -> FsckIssue | None:
        """Ask the LLM whether a regex-flagged memory is a real threat.

        Returns an FsckIssue if confirmed threat, None if false positive.
        """
        # Security re-evaluation handles one memory at a time, so aliasing the
        # memory ID is unnecessary here unlike duplicate/quality batch prompts.
        messages, schema = build_fsck_security_reeval_prompt(mem, patterns)

        response = self._llm.generate(
            messages,
            json_schema=schema,
            temperature=0.1,
            reasoning_effort=self._reasoning_effort,
            operation="fsck_security",
        )

        try:
            parsed = parse_json_response(response)
        except Exception:
            parsed = None

        if not parsed:
            # Parsing failed — include conservatively
            logger.warning(
                "Security re-eval: failed to parse LLM response for memory %s",
                mem.get("id"),
            )
            verdict = "threat"
            reasoning = "LLM response could not be parsed — flagged conservatively."
        else:
            verdict = parsed.get("verdict", "threat")
            reasoning = parsed.get("reasoning", "")

        if verdict == "false_positive":
            logger.debug(
                "Security re-eval: memory %s is a false positive: %s",
                mem.get("id"),
                reasoning,
            )
            return None

        # Confirmed threat
        return FsckIssue(
            issue_id=str(uuid.uuid4()),
            type="security",
            severity="high",
            reasoning=(
                f"Prompt injection patterns detected: {', '.join(patterns)}. "
                f"LLM confirmed: {reasoning}"
            ),
            affected_memories=[self._make_affected_memory(mem)],
            actions=[FsckAction(action="delete", memory_id=mem["id"])],
        )

    # ── checked_at stamping ─────────────────────────────────────────

    def _stamp_clean_memories(
        self,
        check: FsckCheck,
        memories: list[dict],
        check_id: str,
    ) -> None:
        """Stamp ``checked_at`` on memories that have no unresolved issues.

        Memories with issues are intentionally left unstamped so the next
        incremental run re-examines them.  Called both on normal completion
        and on budget-exhausted early exits so that already-processed clean
        memories are not re-checked unnecessarily.
        """
        now_utc = datetime.now(timezone.utc).isoformat()
        issue_memory_ids: set[str] = set()
        for issue in check.issues:
            for am in issue.affected_memories:
                issue_memory_ids.add(am.id)
        clean_ids = [m["id"] for m in memories if m["id"] not in issue_memory_ids]
        if clean_ids:
            for mid in clean_ids:
                try:
                    self._vector.update_metadata(mid, {"checked_at": now_utc})
                except Exception:
                    logger.warning(
                        "Failed to stamp checked_at on memory %s",
                        mid,
                        exc_info=True,
                    )
            logger.info(
                "Fsck check %s: stamped checked_at on %d/%d memories "
                "(%d with issues left unstamped for re-check)",
                check_id,
                len(clean_ids),
                len(memories),
                len(issue_memory_ids),
            )

    # ── Phase 1: Duplicate detection ─────────────────────────────────

    def _phase_duplicate_detection(
        self,
        memories: list[dict],
        mem_by_id: dict[str, dict],
        check: FsckCheck,
        *,
        full_corpus_by_id: dict[str, dict] | None = None,
    ) -> tuple[list[FsckIssue], set[str]]:
        """Find duplicate clusters via vector similarity, then evaluate with LLM.

        Sub-phase 1a (duplicate_search): Build similarity graph — 5-30%
        Sub-phase 1b (duplicate_eval): Evaluate clusters with LLM — 30-55%

        Args:
            memories: Working set of memories to iterate over.
            mem_by_id: Lookup dict for the working set.
            check: The running FsckCheck instance.
            full_corpus_by_id: When in incremental mode, includes ALL
                non-raw memories (not just the changed working set).
                Used for neighbor filtering and cluster member lookup
                so that a changed memory can be matched against an
                unchanged clean memory.

        Returns (issues, set of memory IDs that were part of clusters).
        """
        # For neighbor filtering and cluster lookup, use the full corpus
        # when available so incremental mode can detect duplicates against
        # unchanged memories.
        corpus_by_id = full_corpus_by_id if full_corpus_by_id is not None else mem_by_id
        total = len(memories)
        neighbor_limit = max(_DUPLICATE_NEIGHBORS, len(corpus_by_id))

        # ── Sub-phase 1a: Build similarity graph ────────────────────
        check.progress.phase = "duplicate_search"
        check.progress.processed = 0
        uf = _UnionFind()

        for idx, mem in enumerate(memories):
            vector = mem.get("vector")
            if vector is not None:
                # Search for similar memories using stored vector.
                # When agent_id is set, also search shared memories
                # to detect cross-scope duplicates.
                similar = self._vector.search_by_vector_ids(
                    vector,
                    user_id=check.user_id,
                    agent_id=check.agent_id,
                    limit=neighbor_limit,
                    exclude_ids=[mem["id"]],
                )
                if check.agent_id:
                    shared_similar = self._vector.search_by_vector_ids(
                        vector,
                        user_id=check.user_id,
                        agent_id=None,
                        shared_only=True,
                        limit=neighbor_limit,
                        exclude_ids=[mem["id"]],
                    )
                    # Merge, deduplicate by ID, keep highest score
                    seen = {s["id"] for s in similar}
                    for s in shared_similar:
                        if s["id"] not in seen:
                            seen.add(s["id"])
                            similar.append(s)

                # Only neighbors that survived the main run_check() filters
                # may influence duplicate clustering. This prevents excluded
                # raw memories from acting as graph bridges between durable
                # memories during union-find.  Uses corpus_by_id (full corpus
                # in incremental mode) so unchanged clean memories can still
                # be detected as duplicates of changed memories.
                similar = [s for s in similar if s.get("id") in corpus_by_id]

                for sim in similar:
                    score = sim.get("score", 0)
                    if score >= _DUPLICATE_SIMILARITY_THRESHOLD:
                        uf.union(mem["id"], sim["id"])

            check.progress.processed = idx + 1
            # 8-30% range
            check.progress.percent = 8 + int(22 * (idx + 1) / total) if total else 30

        # Extract clusters with 2+ members
        clusters = {
            root: members
            for root, members in uf.clusters().items()
            if len(members) >= 2
        }

        logger.info(
            "Fsck duplicate search: found %d clusters from %d memories",
            len(clusters),
            total,
        )

        if not clusters:
            check.progress.percent = 55
            return [], set()

        # ── Sub-phase 1b: Evaluate clusters with LLM (parallel) ────
        check.progress.phase = "duplicate_eval"
        check.progress.processed = 0
        issues: list[FsckIssue] = []
        clustered_ids: set[str] = set()
        cluster_list = list(clusters.items())
        total_clusters = len(cluster_list)

        # Prepare cluster data and collect clustered IDs
        cluster_inputs: list[tuple[str, list[dict]]] = []
        for _root, member_ids in cluster_list:
            member_ids = member_ids[:_MAX_CLUSTER_SIZE]
            clustered_ids.update(member_ids)
            cluster_mems = [
                corpus_by_id[mid] for mid in member_ids if mid in corpus_by_id
            ]
            if len(cluster_mems) >= 2:
                cluster_inputs.append((_root, cluster_mems))

        concurrency = max(1, self._config.memory.fsck_llm_concurrency)
        progress_lock = threading.Lock()
        completed_count = 0

        def _eval_cluster(root: str, mems: list[dict]) -> list[FsckIssue]:
            return self._evaluate_duplicate_cluster(mems)

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(_eval_cluster, root, mems): root
                for root, mems in cluster_inputs
            }

            for future in as_completed(futures):
                root = futures[future]
                try:
                    cluster_issues = future.result()
                    with progress_lock:
                        issues.extend(cluster_issues)
                        completed_count += 1
                        check.progress.processed = completed_count
                        check.progress.issues_found = len(check.issues) + len(issues)
                        check.progress.percent = (
                            30 + int(25 * completed_count / total_clusters)
                            if total_clusters
                            else 55
                        )
                except Exception:
                    logger.warning(
                        "Failed to evaluate duplicate cluster (root=%s)",
                        root,
                        exc_info=True,
                    )
                    with progress_lock:
                        completed_count += 1
                        check.progress.processed = completed_count
                        check.progress.percent = (
                            30 + int(25 * completed_count / total_clusters)
                            if total_clusters
                            else 55
                        )

        logger.info(
            "Fsck duplicate detection: %d clusters (%d evaluated), %d issues",
            len(clusters),
            len(cluster_inputs),
            len(issues),
        )

        return issues, clustered_ids

    def _evaluate_duplicate_cluster(
        self,
        cluster: list[dict],
    ) -> list[FsckIssue]:
        """Send a cluster of similar memories to LLM for duplicate evaluation."""
        messages, schema, id_mapping = build_fsck_duplicate_prompt(
            cluster,
            max_memory_length=self._config.memory.max_memory_length,
        )

        response = self._llm.generate(
            messages,
            json_schema=schema,
            temperature=0.1,
            reasoning_effort=self._reasoning_effort,
            operation="fsck_dedup",
        )

        parsed = parse_json_response(response)
        if not parsed or "issues" not in parsed:
            return []

        # Build lookup for affected memory content
        mem_lookup = {m["id"]: m for m in cluster}

        issues: list[FsckIssue] = []
        for raw_issue in parsed["issues"]:
            actions, action_target_ids = self._parse_issue_actions(
                raw_issue, id_mapping, mem_lookup
            )
            affected_mems = self._resolve_affected_memories(
                raw_issue, id_mapping, mem_lookup, action_target_ids
            )
            if not affected_mems:
                logger.warning(
                    "Fsck duplicate check: dropping issue with no resolvable affected memories (type=%s, raw_ids=%s)",
                    raw_issue.get("type", "duplicate"),
                    raw_issue.get("affected_memory_ids", []),
                )
                continue

            issue_type = raw_issue.get("type", "duplicate")
            if issue_type not in ("duplicate", "contradiction"):
                issue_type = "duplicate"

            severity = raw_issue.get("severity", "medium")
            if severity not in ("low", "medium", "high"):
                severity = "medium"

            raw_confidence = raw_issue.get("confidence")
            confidence: float | None = None
            if raw_confidence is not None:
                try:
                    confidence = max(0.0, min(1.0, float(raw_confidence)))
                except (TypeError, ValueError):
                    confidence = None

            issues.append(
                FsckIssue(
                    issue_id=str(uuid.uuid4()),
                    type=issue_type,
                    severity=severity,
                    reasoning=raw_issue.get("reasoning", ""),
                    affected_memories=affected_mems,
                    actions=actions,
                    confidence=confidence,
                )
            )

        return issues

    # ── Phase 2: Quality check ───────────────────────────────────────

    def _get_available_categories(self, user_id: str) -> list[str]:
        """Return valid category names for this user.

        Combines predefined categories with any dynamic project:* categories
        found in the user's memories. Falls back to predefined list on error.
        """
        try:
            if self._memory_service is not None:
                cats = self._memory_service.list_categories(user_id=user_id)
                return [c["name"] for c in cats.get("categories", []) if c.get("name")]
        except Exception:
            logger.warning("Failed to load categories for fsck, using predefined list")
        return list(PREDEFINED_CATEGORIES.keys())

    def _phase_quality_check(
        self,
        memories: list[dict],
        check: FsckCheck,
        *,
        available_categories: list[str] | None = None,
    ) -> list[FsckIssue]:
        """Batch memories and send to LLM for quality evaluation.

        Checks for spelling, sense/completeness, split candidates,
        metadata misclassification, and subtle injection patterns.

        Progress: 55-100% range.
        """
        check.progress.phase = "quality_check"
        check.progress.processed = 0
        issues: list[FsckIssue] = []
        total = len(memories)
        total_batches = (total + _QUALITY_BATCH_SIZE - 1) // _QUALITY_BATCH_SIZE

        # Pre-split into batches
        batches: list[tuple[int, list[dict]]] = []
        for i in range(0, total, _QUALITY_BATCH_SIZE):
            batches.append((i, memories[i : i + _QUALITY_BATCH_SIZE]))

        concurrency = max(1, self._config.memory.fsck_llm_concurrency)
        progress_lock = threading.Lock()
        processed_mems = 0

        def _eval_batch(batch: list[dict]) -> list[FsckIssue]:
            return self._evaluate_quality_batch(
                batch, available_categories=available_categories
            )

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(_eval_batch, batch): (offset, batch)
                for offset, batch in batches
            }

            for future in as_completed(futures):
                offset, batch = futures[future]
                try:
                    batch_issues = future.result()
                    with progress_lock:
                        issues.extend(batch_issues)
                        processed_mems += len(batch)
                        check.progress.processed = processed_mems
                        check.progress.issues_found = len(check.issues) + len(issues)
                        check.progress.percent = (
                            55 + int(45 * processed_mems / total) if total else 100
                        )
                except Exception:
                    logger.warning(
                        "Failed to evaluate quality batch (offset=%d, size=%d)",
                        offset,
                        len(batch),
                        exc_info=True,
                    )
                    with progress_lock:
                        processed_mems += len(batch)
                        check.progress.processed = processed_mems
                        check.progress.percent = (
                            55 + int(45 * processed_mems / total) if total else 100
                        )

        logger.info(
            "Fsck quality check: %d memories in %d batches, %d issues",
            total,
            total_batches,
            len(issues),
        )

        return issues

    def _evaluate_quality_batch(
        self,
        batch: list[dict],
        *,
        available_categories: list[str] | None = None,
    ) -> list[FsckIssue]:
        """Send a batch of memories through both quality passes.

        Runs Pass A (content quality) and Pass B (metadata normalization)
        sequentially and merges the results.
        """
        issues: list[FsckIssue] = []
        issues.extend(self._evaluate_content_quality_batch(batch))
        issues.extend(
            self._evaluate_metadata_normalization_batch(
                batch, available_categories=available_categories
            )
        )
        return issues

    def _evaluate_content_quality_batch(
        self,
        batch: list[dict],
    ) -> list[FsckIssue]:
        """Pass A: check a batch of memories for content quality issues."""
        messages, schema, id_mapping = build_fsck_content_quality_prompt(batch)

        response = self._llm.generate(
            messages,
            json_schema=schema,
            temperature=0.1,
            reasoning_effort=self._reasoning_effort,
            operation="fsck_content_quality",
        )

        return self._parse_quality_response(
            response, id_mapping, batch, default_type="quality"
        )

    def _evaluate_metadata_normalization_batch(
        self,
        batch: list[dict],
        *,
        available_categories: list[str] | None = None,
    ) -> list[FsckIssue]:
        """Pass B: check a batch of memories for metadata issues."""
        messages, schema, id_mapping = build_fsck_metadata_normalization_prompt(
            batch, available_categories=available_categories
        )

        response = self._llm.generate(
            messages,
            json_schema=schema,
            temperature=0.1,
            reasoning_effort=self._reasoning_effort,
            operation="fsck_metadata_normalization",
        )

        return self._parse_quality_response(
            response, id_mapping, batch, default_type="reclassify"
        )

    def _parse_quality_response(
        self,
        response: str,
        id_mapping: dict[str, str],
        batch: list[dict],
        *,
        default_type: str = "quality",
    ) -> list[FsckIssue]:
        """Parse an LLM quality/metadata response into ``FsckIssue`` objects."""
        parsed = parse_json_response(response)
        if not parsed or "issues" not in parsed:
            return []

        # Build lookup for affected memory content
        mem_lookup = {m["id"]: m for m in batch}

        issues: list[FsckIssue] = []
        for raw_issue in parsed["issues"]:
            actions, action_target_ids = self._parse_issue_actions(
                raw_issue, id_mapping, mem_lookup
            )
            affected_mems = self._resolve_affected_memories(
                raw_issue, id_mapping, mem_lookup, action_target_ids
            )
            if not affected_mems:
                logger.warning(
                    "Fsck quality check: dropping issue with no resolvable affected memories (type=%s, raw_ids=%s)",
                    raw_issue.get("type", default_type),
                    raw_issue.get("affected_memory_ids", []),
                )
                continue

            issue_type = raw_issue.get("type", default_type)
            if issue_type not in (
                "quality",
                "reclassify",
            ):
                issue_type = default_type

            severity = raw_issue.get("severity", "medium")
            if severity not in ("low", "medium", "high"):
                severity = "medium"

            raw_confidence = raw_issue.get("confidence")
            confidence: float | None = None
            if raw_confidence is not None:
                try:
                    confidence = max(0.0, min(1.0, float(raw_confidence)))
                except (TypeError, ValueError):
                    confidence = None

            issues.append(
                FsckIssue(
                    issue_id=str(uuid.uuid4()),
                    type=issue_type,
                    severity=severity,
                    reasoning=raw_issue.get("reasoning", ""),
                    affected_memories=affected_mems,
                    actions=actions,
                    confidence=confidence,
                )
            )

        return issues

    @staticmethod
    def _get_memory_agent_id(mem: dict[str, Any]) -> str | None:
        """Return the memory agent ID from the normalized memory shape."""
        agent_id = mem.get("agent_id")
        if agent_id is not None:
            return str(agent_id)
        metadata = mem.get("metadata") or {}
        fallback = metadata.get("agent_id")
        return str(fallback) if fallback is not None else None

    def _make_affected_memory(self, mem: dict[str, Any]) -> FsckAffectedMemory:
        """Convert a memory dict into an affected-memory payload."""
        return FsckAffectedMemory(
            id=str(mem.get("id", "")),
            content=mem.get("memory", ""),
            metadata=mem.get("metadata"),
            agent_id=self._get_memory_agent_id(mem),
        )

    @staticmethod
    def _resolve_llm_memory_id(
        raw_memory_id: Any,
        id_mapping: dict[str, str],
        mem_lookup: dict[str, dict[str, Any]],
    ) -> str | None:
        """Resolve an LLM-returned alias or direct ID to a real memory ID."""
        if raw_memory_id is None:
            return None
        candidate = str(raw_memory_id).strip()
        if not candidate:
            return None
        resolved = id_mapping.get(candidate, candidate)
        return resolved if resolved in mem_lookup else None

    def _parse_issue_actions(
        self,
        raw_issue: dict[str, Any],
        id_mapping: dict[str, str],
        mem_lookup: dict[str, dict[str, Any]],
    ) -> tuple[list[FsckAction], list[str]]:
        """Parse and validate fsck issue actions from LLM output."""
        actions: list[FsckAction] = []
        action_target_ids: list[str] = []

        for raw_action in raw_issue.get("actions", []):
            if not isinstance(raw_action, dict):
                continue
            action_type = raw_action.get("action", "")
            resolved_memory_id = self._resolve_llm_memory_id(
                raw_action.get("memory_id"), id_mapping, mem_lookup
            )
            if action_type != "add" and resolved_memory_id is None:
                continue
            actions.append(
                FsckAction(
                    action=action_type,
                    memory_id=resolved_memory_id,
                    new_content=raw_action.get("new_content"),
                    new_metadata=raw_action.get("new_metadata"),
                )
            )
            if resolved_memory_id and resolved_memory_id not in action_target_ids:
                action_target_ids.append(resolved_memory_id)

        return actions, action_target_ids

    def _resolve_affected_memories(
        self,
        raw_issue: dict[str, Any],
        id_mapping: dict[str, str],
        mem_lookup: dict[str, dict[str, Any]],
        action_target_ids: list[str],
    ) -> list[FsckAffectedMemory]:
        """Resolve issue affected memories from aliases and validated actions."""
        resolved_ids: list[str] = []
        for raw_memory_id in raw_issue.get("affected_memory_ids", []):
            resolved = self._resolve_llm_memory_id(
                raw_memory_id, id_mapping, mem_lookup
            )
            if resolved and resolved not in resolved_ids:
                resolved_ids.append(resolved)

        for memory_id in action_target_ids:
            if memory_id not in resolved_ids:
                resolved_ids.append(memory_id)

        return [self._make_affected_memory(mem_lookup[mid]) for mid in resolved_ids]

    # ── Apply logic ──────────────────────────────────────────────────

    def _apply_issue(
        self,
        issue: FsckIssue,
        user_id: str,
        *,
        check_id: str = "manual",
        expected_revisions: dict[str, int] | None = None,
        action_indexes: set[int] | None = None,
        before_action: Callable[[int], None] | None = None,
        on_action: Callable[[int, str, str | None], None] | None = None,
    ) -> tuple[int, int]:
        """Apply all actions for a single issue.

        Returns:
            Tuple of (actions_executed, actions_skipped).
        """
        executed = 0
        skipped = 0
        split_source_ids: list[str] = []
        split_owner_id: str | None = None

        def start_action(index: int) -> None:
            if before_action is not None:
                before_action(index)

        def finish_action(
            index: int,
            status: str,
            subordinate_operation_id: str | None = None,
        ) -> None:
            if on_action is not None:
                on_action(index, status, subordinate_operation_id)

        # Guard: skip split actions on memories with artifacts.
        # Splitting would delete the original (destroying artifacts) and
        # create new memories without them. The prompt should prevent this,
        # but this is a hard safety net.
        if issue.type == "split":
            for index, action in enumerate(issue.actions):
                if action_indexes is not None and index not in action_indexes:
                    continue
                if action.action == "delete" and action.memory_id:
                    mem = self._vector.get_by_id(action.memory_id)
                    if mem and (mem.get("metadata") or {}).get("artifacts"):
                        logger.warning(
                            "Fsck apply: skipping split of memory %s — "
                            "has %d artifact(s) that would be destroyed",
                            action.memory_id,
                            len(mem["metadata"]["artifacts"]),
                        )
                        for skipped_index in action_indexes or range(
                            len(issue.actions)
                        ):
                            finish_action(skipped_index, "skipped")
                        return 0, len(issue.actions)

        if issue.type == "split" and self._memory_service is not None:
            for index, action in enumerate(issue.actions):
                if action_indexes is not None and index not in action_indexes:
                    continue
                if action.action != "delete" or not action.memory_id:
                    continue
                existing = self._vector.get_by_id(action.memory_id)
                if existing is None or existing.get("user_id") != user_id:
                    skipped += 1
                    finish_action(index, "skipped")
                    continue
                split_source_ids.append(action.memory_id)
                split_owner_id = existing.get("owner_id") or user_id
                start_action(index)
                subordinate_operation_id = f"fsck:{check_id}:{issue.issue_id}"
                self._memory_service.revisions.mark_source(
                    [action.memory_id],
                    operation_id=subordinate_operation_id,
                    user_id=user_id,
                    owner_id=split_owner_id,
                    session_agent_id=None,
                    reason="split",
                    expected_revisions=expected_revisions,
                )
                executed += 1
                finish_action(index, "committed", subordinate_operation_id)

        for index, action in enumerate(issue.actions):
            if action_indexes is not None and index not in action_indexes:
                continue
            if action.action == "delete" and action.memory_id:
                # Verify ownership before deleting
                existing = self._vector.get_by_id(action.memory_id)
                if existing is None:
                    logger.warning(
                        "Fsck apply: memory %s not found, skipping delete",
                        action.memory_id,
                    )
                    skipped += 1
                    finish_action(index, "skipped")
                    continue
                if existing.get("user_id") != user_id:
                    logger.warning(
                        "Fsck apply: memory %s belongs to a different user, skipping delete",
                        action.memory_id,
                    )
                    skipped += 1
                    finish_action(index, "skipped")
                    continue
                # Warn when deleting a memory with artifacts — the prompt
                # should prevent this in most cases, but log for visibility.
                existing_meta = existing.get("metadata") or {}
                if issue.type == "split" and self._memory_service is not None:
                    continue
                if existing_meta.get("artifacts"):
                    logger.warning(
                        "Fsck apply: deleting memory %s which has %d artifact(s)",
                        action.memory_id,
                        len(existing_meta["artifacts"]),
                    )
                # Route through MemoryService to clean up artifacts and
                # invalidate caches. Fall back to direct delete in tests.
                if self._memory_service is not None:
                    existing_meta = existing.get("metadata") or {}
                    start_action(index)
                    mutation = self._memory_service.delete_memory(
                        memory_id=action.memory_id,
                        user_id=user_id,
                        owner_id=existing.get("owner_id") or user_id,
                        expected_revision=(expected_revisions or {}).get(
                            action.memory_id,
                            int(existing_meta.get("revision", 1)),
                        ),
                        idempotency_key=(
                            f"fsck:{check_id}:{issue.issue_id}:"
                            f"delete:{action.memory_id}"
                        ),
                        actor_kind="fsck",
                        reason=issue.type,
                        audit={
                            "fsck_check_id": check_id,
                            "fsck_issue_id": issue.issue_id,
                        },
                    )
                    finish_action(index, "committed", mutation.get("operation_id"))
                else:
                    start_action(index)
                    self._vector.delete(action.memory_id)
                    finish_action(index, "committed")
                executed += 1

            elif action.action == "update" and action.memory_id:
                # Verify ownership before updating
                existing = self._vector.get_by_id(action.memory_id)
                if existing is None:
                    logger.warning(
                        "Fsck apply: memory %s not found, skipping update",
                        action.memory_id,
                    )
                    skipped += 1
                    finish_action(index, "skipped")
                    continue
                if existing.get("user_id") != user_id:
                    logger.warning(
                        "Fsck apply: memory %s belongs to a different user, skipping update",
                        action.memory_id,
                    )
                    skipped += 1
                    finish_action(index, "skipped")
                    continue
                if self._memory_service is not None:
                    meta = {
                        key: value
                        for key, value in (action.new_metadata or {}).items()
                        if key
                        in {
                            "memory_type",
                            "categories",
                            "importance",
                            "pinned",
                            "role",
                            "labels",
                        }
                        and value is not None
                    }
                    if meta.get("memory_type") in ("fact", "preference"):
                        from mnemory.prompts import _correct_memory_type

                        source_text = action.new_content or existing.get("memory", "")
                        if _correct_memory_type("fact", source_text) != "fact":
                            meta.pop("memory_type", None)
                    if meta.get("role") == "assistant" and not existing.get("agent_id"):
                        meta.pop("role", None)
                    kwargs: dict[str, Any] = {
                        "user_id": user_id,
                        "owner_id": existing.get("owner_id") or user_id,
                        "content": action.new_content,
                        "expected_revision": (expected_revisions or {}).get(
                            action.memory_id,
                            int((existing.get("metadata") or {}).get("revision", 1)),
                        ),
                        "idempotency_key": (
                            f"fsck:{check_id}:{issue.issue_id}:"
                            f"update:{action.memory_id}"
                        ),
                        "operation_kind": "fsck_update",
                        "actor_kind": "fsck",
                        "operation_reason": issue.type,
                        "audit": {
                            "fsck_check_id": check_id,
                            "fsck_issue_id": issue.issue_id,
                        },
                    }
                    kwargs.update(meta)
                    if action.new_content is None:
                        kwargs.pop("content")
                    start_action(index)
                    mutation = self._memory_service.update_memory(
                        action.memory_id, **kwargs
                    )
                    executed += 1
                    finish_action(index, "committed", mutation.get("operation_id"))
                    continue
                if action.new_content:
                    # Generate sparse vector for hybrid search if available
                    sparse_vector = None
                    if self._memory_service is not None:
                        sparse_vector = self._memory_service._get_sparse_vector(
                            action.new_content
                        )
                    start_action(index)
                    self._vector.update_content(
                        action.memory_id,
                        action.new_content,
                        sparse_vector=sparse_vector,
                    )
                    # Stamp updated_at_utc so incremental fsck picks up the change.
                    self._vector.update_metadata(
                        action.memory_id,
                        {"updated_at_utc": datetime.now(timezone.utc).isoformat()},
                    )
                    executed += 1
                    finish_action(index, "committed")
                if action.new_metadata:
                    # Filter to allowed metadata fields, dropping None values
                    # (None means "unchanged" in LLM output — don't overwrite).
                    allowed = {
                        "memory_type",
                        "categories",
                        "importance",
                        "pinned",
                        "role",
                        "labels",
                        "evidence_root_ids",
                        "validation_count",
                        "validation_strength",
                        "validation_projection_hash",
                    }
                    clean_meta = {
                        k: v
                        for k, v in action.new_metadata.items()
                        if k in allowed and v is not None
                    }
                    if clean_meta:
                        # Validate memory_type — strip LLM-hallucinated values
                        # and reject promotions that contradict heuristic rules.
                        if "memory_type" in clean_meta:
                            try:
                                clean_meta["memory_type"] = validate_memory_type(
                                    clean_meta["memory_type"]
                                )
                            except ValueError:
                                logger.warning(
                                    "Fsck apply: stripping invalid memory_type '%s' "
                                    "from reclassify action for memory %s",
                                    clean_meta["memory_type"],
                                    action.memory_id,
                                )
                                del clean_meta["memory_type"]
                        # Heuristic safety net: if the fsck model suggests
                        # promoting to "fact" or "preference", validate the
                        # memory text against our post-LLM heuristic patterns.
                        # We test as if the type were "fact" — if the heuristic
                        # demotes it (→ episodic/context), the text clearly
                        # shouldn't be permanent, so promoting to fact OR
                        # preference is wrong.
                        if "memory_type" in clean_meta and clean_meta[
                            "memory_type"
                        ] in ("fact", "preference"):
                            from mnemory.prompts import _correct_memory_type

                            mem_text = existing.get("memory", "")
                            corrected = _correct_memory_type("fact", mem_text)
                            if corrected != "fact":
                                logger.info(
                                    "Fsck apply: rejecting promotion to '%s' "
                                    "for memory %s — heuristic says '%s' "
                                    "(text: %.80s)",
                                    clean_meta["memory_type"],
                                    action.memory_id,
                                    corrected,
                                    mem_text,
                                )
                                del clean_meta["memory_type"]
                        # Validate importance — strip LLM-hallucinated values.
                        if "importance" in clean_meta:
                            try:
                                clean_meta["importance"] = validate_importance(
                                    clean_meta["importance"]
                                )
                            except ValueError:
                                logger.warning(
                                    "Fsck apply: stripping invalid importance '%s' "
                                    "from reclassify action for memory %s",
                                    clean_meta["importance"],
                                    action.memory_id,
                                )
                                del clean_meta["importance"]
                        # Validate categories — strip any LLM-hallucinated
                        # categories that don't exist in the predefined set,
                        # keeping valid ones rather than dropping the whole field.
                        if (
                            "categories" in clean_meta
                            and clean_meta["categories"] is not None
                        ):
                            valid_cats = []
                            for cat in clean_meta["categories"]:
                                try:
                                    valid_cats.extend(validate_categories([cat]))
                                except ValueError as cat_err:
                                    logger.warning(
                                        "Fsck apply: stripping invalid category '%s' "
                                        "from reclassify action for memory %s: %s",
                                        cat,
                                        action.memory_id,
                                        cat_err,
                                    )
                            if valid_cats:
                                clean_meta["categories"] = valid_cats
                            else:
                                del clean_meta["categories"]
                        # Validate role — must be "user" or "assistant",
                        # and "assistant" requires the memory to have an
                        # agent_id (invariant from add_memory validation).
                        if "role" in clean_meta:
                            if clean_meta["role"] not in ("user", "assistant"):
                                logger.warning(
                                    "Fsck apply: stripping invalid role '%s' "
                                    "from reclassify action for memory %s",
                                    clean_meta["role"],
                                    action.memory_id,
                                )
                                del clean_meta["role"]
                            elif clean_meta["role"] == "assistant" and not existing.get(
                                "agent_id"
                            ):
                                logger.warning(
                                    "Fsck apply: cannot set role='assistant' "
                                    "on shared memory %s (no agent_id), "
                                    "stripping role change",
                                    action.memory_id,
                                )
                                del clean_meta["role"]
                        if clean_meta:
                            start_action(index)
                            self._vector.update_metadata(action.memory_id, clean_meta)
                            executed += 1
                            finish_action(index, "committed")

            elif action.action == "add" and action.new_content:
                # Derive metadata from source memory when available.
                # For split issues, the source is the memory being deleted.
                source_meta: dict[str, Any] = {}
                source_agent_id: str | None = None
                source_owner_id = user_id
                for other_action in issue.actions:
                    if other_action.action == "delete" and other_action.memory_id:
                        source = self._vector.get_by_id(other_action.memory_id)
                        if source:
                            source_meta = source.get("metadata") or {}
                            source_agent_id = source.get("agent_id")
                            source_owner_id = source.get("owner_id") or user_id
                        break

                if self._memory_service is not None:
                    meta = action.new_metadata or {}
                    source_id = split_source_ids[0] if split_source_ids else None
                    if source_id is None:
                        for other_action in issue.actions:
                            if (
                                other_action.action == "delete"
                                and other_action.memory_id
                            ):
                                source_id = other_action.memory_id
                                break
                    memory_id = str(
                        uuid.uuid5(
                            uuid.NAMESPACE_URL,
                            (
                                f"mnemory:fsck:{check_id}:{issue.issue_id}:"
                                f"{action.new_content}"
                            ),
                        )
                    )
                    from mnemory.revisions import RevisionService

                    start_action(index)
                    self._memory_service.add_memory(
                        content=action.new_content,
                        user_id=user_id,
                        owner_id=source_owner_id,
                        agent_id=source_agent_id or meta.get("agent_id"),
                        role=source_meta.get("role") or meta.get("role") or "user",
                        memory_type=meta.get("memory_type")
                        or source_meta.get("memory_type"),
                        categories=meta.get("categories")
                        or source_meta.get("categories"),
                        importance=meta.get("importance")
                        or source_meta.get("importance"),
                        pinned=meta.get("pinned", source_meta.get("pinned")),
                        labels=meta.get("labels") or source_meta.get("labels"),
                        event_date=source_meta.get("event_date"),
                        ttl_days=source_meta.get("ttl_days"),
                        infer=False,
                        _trusted=True,
                        _memory_id=memory_id,
                        _revision_metadata=RevisionService.initial_metadata(
                            memory_id,
                            derived_from=[source_id] if source_id else None,
                            source_session_id=None,
                        ),
                    )
                else:
                    # Fallback: direct vector store insert when MemoryService
                    # is not available (e.g., in tests).
                    embed = EmbeddingClient(self._config.embed)
                    vector = embed.embed(action.new_content)

                    now = datetime.now(timezone.utc)
                    payload: dict[str, Any] = {
                        "data": action.new_content,
                        "hash": hashlib.sha256(action.new_content.encode()).hexdigest(),
                        "user_id": user_id,
                        "created_at": now.isoformat(),
                        "updated_at": now.isoformat(),
                    }
                    if action.new_metadata:
                        for k, v in action.new_metadata.items():
                            if k in (
                                "memory_type",
                                "categories",
                                "importance",
                                "pinned",
                            ):
                                payload[k] = v

                    payload.setdefault("memory_type", "fact")
                    payload.setdefault("importance", "normal")
                    payload.setdefault("pinned", False)

                    point_id = str(uuid.uuid4())
                    start_action(index)
                    self._vector._client.upsert(
                        collection_name=self._vector.collection_name,
                        points=[
                            PointStruct(
                                id=point_id,
                                vector=vector,
                                payload=payload,
                            )
                        ],
                    )
                executed += 1
                finish_action(index, "committed")

        return executed, skipped

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _build_summary(issues: list[FsckIssue]) -> FsckSummary:
        """Build a summary from a list of issues."""
        summary = FsckSummary()
        for issue in issues:
            if issue.type == "duplicate":
                summary.duplicate += 1
            elif issue.type == "quality":
                summary.quality += 1
            elif issue.type == "split":
                summary.split += 1
            elif issue.type == "contradiction":
                summary.contradiction += 1
            elif issue.type == "reclassify":
                summary.reclassify += 1
            elif issue.type == "security":
                summary.security += 1
            summary.total += 1
        return summary
