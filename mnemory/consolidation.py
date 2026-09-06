"""Memory consolidation service — synthesize durable knowledge from raw memories.

Within-session consolidation reads a session summary and its linked raw
memories, then uses an LLM to synthesize canonical durable memories
(decisions, preferences, facts, actions). Raw memories are marked as
superseded after successful consolidation.

The service is scheduled by MaintenanceService to run periodically,
checking for idle sessions that have raw memories awaiting consolidation.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from mnemory.revisions import RevisionService, canonical_fingerprint

if TYPE_CHECKING:
    from mnemory.config import Config
    from mnemory.embeddings import EmbeddingClient
    from mnemory.llm import LLMClient
    from mnemory.memory import MemoryService
    from mnemory.metrics import MetricsCollector
    from mnemory.storage.vector import SessionSummaryStore, VectorStore

logger = logging.getLogger(__name__)

# Minimum consolidated memories per raw memories (warning threshold)
_MIN_CONSOLIDATED_RATIO = 0.05

# Maximum similarity between consolidated outputs (near-duplicate detection)
_MAX_OUTPUT_SIMILARITY = 0.90

# Minimum content length for a consolidated memory
_MIN_CONTENT_LENGTH = 20

_CONSOLIDATION_TYPE_ALIASES = {
    "goal": "episodic",
    "decision": "episodic",
    "action": "episodic",
    "workflow": "procedural",
}
_VALID_MEMORY_TYPES = {"preference", "fact", "episodic", "procedural", "context"}


def _normalize_consolidation_memory_type(value: Any) -> str:
    """Normalize descriptive provider labels to supported storage types."""
    if not isinstance(value, str):
        return "episodic"
    normalized = value.strip().lower()
    if normalized in _VALID_MEMORY_TYPES:
        return normalized
    result = _CONSOLIDATION_TYPE_ALIASES.get(normalized, "episodic")
    logger.info("Normalized consolidation memory_type %r to %r", value, result)
    return result


@dataclass
class ConsolidationResult:
    """Result of a consolidation run."""

    session_id: str
    memories_produced: int = 0
    memories_superseded: int = 0
    consolidated_memory_ids: list[str] | None = None
    state: str = "idle"  # idle, consolidating, consolidated, failed
    error: str | None = None
    duration_seconds: float = 0.0


RETRY_INPUT_REASON_CODES = (
    "linked_memory_lookup_failed",
    "linked_memory_missing",
    "linked_memory_user_mismatch",
    "linked_memory_owner_mismatch",
    "linked_memory_agent_mismatch",
    "linked_memory_not_raw",
    "linked_memory_not_active",
    "linked_memory_superseded",
)


@dataclass(frozen=True)
class RetryInputAssessment:
    """Immutable assessment of one session's linked raw-memory inputs."""

    input_fingerprint: str
    reasons: tuple[str, ...]
    memory_count: int


def assess_retry_inputs(
    vector: VectorStore,
    session: dict[str, Any],
    *,
    user_id: str,
    owner_id: str,
    agent_id: str | None,
) -> RetryInputAssessment:
    """Validate every linked memory revision for an explicit retry."""
    memory_ids = list(
        dict.fromkeys(str(item) for item in session.get("memory_ids") or [])
    )
    try:
        memories = vector.get_by_ids_strict(memory_ids)
    except Exception:
        return RetryInputAssessment(
            input_fingerprint=canonical_fingerprint(
                {"memory_ids": memory_ids, "state": "lookup_failed"}
            ),
            reasons=("linked_memory_lookup_failed",),
            memory_count=len(memory_ids),
        )

    by_id = {str(memory["id"]): memory for memory in memories}
    reasons: list[str] = []
    snapshots: list[dict[str, Any]] = []
    for memory_id in memory_ids:
        memory = by_id.get(memory_id)
        if memory is None:
            reasons.append("linked_memory_missing")
            snapshots.append({"memory_id": memory_id, "state": "missing"})
            continue
        metadata = memory.get("metadata") or {}
        if memory.get("user_id") != user_id:
            reasons.append("linked_memory_user_mismatch")
        if (memory.get("owner_id") or memory.get("user_id")) != owner_id:
            reasons.append("linked_memory_owner_mismatch")
        if memory.get("agent_id") != agent_id:
            reasons.append("linked_memory_agent_mismatch")
        if metadata.get("memory_layer", "raw") != "raw":
            reasons.append("linked_memory_not_raw")
        if metadata.get("revision_state", "active") != "active":
            reasons.append("linked_memory_not_active")
        if metadata.get("superseded_by"):
            reasons.append("linked_memory_superseded")
        snapshots.append(
            {
                "memory_id": memory_id,
                "user_id": memory.get("user_id"),
                "owner_id": memory.get("owner_id") or memory.get("user_id"),
                "agent_id": memory.get("agent_id"),
                "memory_layer": metadata.get("memory_layer", "raw"),
                "lineage_id": metadata.get("lineage_id", memory_id),
                "revision": metadata.get("revision", 1),
                "revision_state": metadata.get("revision_state", "active"),
                "superseded_by": metadata.get("superseded_by"),
                "content_hash": memory.get("hash"),
            }
        )
    ordered_reasons = tuple(
        reason for reason in RETRY_INPUT_REASON_CODES if reason in set(reasons)
    )
    return RetryInputAssessment(
        input_fingerprint=canonical_fingerprint(snapshots),
        reasons=ordered_reasons,
        memory_count=len(memory_ids),
    )


class ConsolidationService:
    """Synthesize durable knowledge from raw session memories.

    Within-session consolidation:
    1. Find idle sessions with raw memories (consolidation_state=idle)
    2. Read session summary + linked raw memories
    3. LLM synthesizes consolidated memories
    4. Validate output quality
    5. Store consolidated memories (memory_layer=consolidated, derived_from=[...])
    6. Mark raw memories as superseded (except artifact-bearing ones)
    7. Update session consolidation state

    The service acquires a per-user lock to prevent race conditions
    with concurrent remember calls.
    """

    def __init__(
        self,
        config: Config,
        vector: VectorStore,
        llm: LLMClient,
        embedding: EmbeddingClient,
        memory_service: MemoryService,
        session_summary_store: SessionSummaryStore,
        collector: MetricsCollector | None = None,
    ) -> None:
        self._config = config
        self._vector = vector
        self._llm = llm
        self._embedding = embedding
        self._memory = memory_service
        self._sessions = session_summary_store
        self._collector = collector

    def find_pending_sessions(self, user_id: str) -> list[dict]:
        """Find sessions awaiting consolidation for a user."""
        return self._sessions.find_pending(
            user_id,
            idle_threshold_seconds=self._config.memory.consolidation_idle_threshold,
        )

    def consolidate_session(
        self,
        session_id: str,
        *,
        session_record: dict[str, Any] | None = None,
        mutation_guard: Callable[[], None] | None = None,
    ) -> ConsolidationResult:
        """Consolidate one session's raw memories into durable knowledge.

        State machine: idle -> consolidating -> consolidated (or failed).
        Crash recovery: if state is 'consolidating' on entry, checks for
        orphaned consolidated memories and resumes or resets.
        """
        result = ConsolidationResult(session_id=session_id)
        t0 = time.monotonic()

        def guard_mutation() -> None:
            if mutation_guard is not None:
                mutation_guard()

        try:
            # 1. Read session summary
            session = session_record or self._sessions.get(session_id)
            if session is None:
                result.error = "Session not found"
                result.state = "failed"
                return result

            user_id = session.get("user_id", "")
            owner_id = session.get("owner_id") or user_id
            agent_id = session.get("agent_id")
            session_point_id = session.get("_point_id")
            memory_ids = session.get("memory_ids", [])
            summary = session.get("summary", "")

            logger.info(
                "Consolidation session %s: %d linked memories, summary_len=%d",
                session_id,
                len(memory_ids),
                len(summary),
            )

            # 2. Claim the exact session generation before model work.
            session_revision = int(session.get("session_revision", 1))
            if "session_revision" not in session:
                guard_mutation()
                self._sessions._client.set_payload(
                    collection_name=self._sessions.COLLECTION,
                    payload={"session_revision": session_revision},
                    points=[
                        session_point_id or self._sessions.get_point_id(session_id)
                    ],
                    wait=True,
                )
            consolidation_token = str(uuid.uuid4())
            guard_mutation()
            if not self._sessions.claim_consolidation(
                session_id,
                expected_revision=session_revision,
                token=consolidation_token,
                attempt_count=int(session.get("attempt_count", 0)) + 1,
                point_id=session_point_id,
            ):
                result.error = "Session revision changed or is already claimed"
                result.state = "failed"
                return result
            result.state = "consolidating"

            # 3. Fetch linked raw memories
            raw_memories = self._fetch_raw_memories(
                memory_ids,
                user_id,
                owner_id,
                agent_id,
            )
            if not raw_memories:
                logger.info(
                    "Session %s: no raw memories found (may already be consolidated)",
                    session_id,
                )
                guard_mutation()
                result.state = self._sessions.finalize_consolidation(
                    session_id,
                    expected_revision=session_revision,
                    token=consolidation_token,
                    consolidated_memory_ids=session.get("consolidated_memory_ids", []),
                    point_id=session_point_id,
                )
                return result

            # 3b. Fetch previously consolidated memories (for re-consolidation)
            previous_consolidated = []
            prev_consolidated_ids = session.get("consolidated_memory_ids") or []
            if prev_consolidated_ids:
                for prev_id in prev_consolidated_ids:
                    try:
                        mem = self._vector.get_by_id(prev_id)
                        if (
                            mem
                            and mem.get("memory")
                            and self._in_session_scope(
                                mem,
                                user_id=user_id,
                                owner_id=owner_id,
                                agent_id=agent_id,
                            )
                        ):
                            previous_consolidated.append(mem)
                    except Exception:
                        pass
                if not previous_consolidated and prev_consolidated_ids:
                    logger.debug(
                        "Could not fetch previous consolidated memories for %s",
                        session_id,
                    )
                if previous_consolidated:
                    logger.info(
                        "Consolidation session %s: %d previous consolidated memories for context",
                        session_id,
                        len(previous_consolidated),
                    )

            # 4. Identify artifact-bearing memories (protected from superseding)
            artifact_ids = {
                m["id"]
                for m in raw_memories
                if (m.get("metadata") or {}).get("artifacts")
            }

            logger.info(
                "Consolidation session %s: %d raw memories fetched (%d with artifacts)",
                session_id,
                len(raw_memories),
                len(artifact_ids),
            )

            # 5. Split raw memories by role and consolidate each scope
            #
            # User and assistant memories are consolidated independently
            # with separate LLM calls. This ensures assistant actions are
            # not drowned out by user-focused synthesis.
            #
            # Both passes ALWAYS run (even with 0 raw memories for a role)
            # because the session summary may contain facts for a role that
            # the remember endpoint didn't extract as raw memories. The LLM
            # can extract durable knowledge from the summary alone.
            #
            # User consolidation runs first. Its output is passed as
            # read-only context to the assistant pass to prevent cross-role
            # duplication (e.g., the same commit attributed to both roles).
            user_raw = [
                m
                for m in raw_memories
                if (m.get("metadata") or {}).get("role", "user") == "user"
            ]
            assistant_raw = [
                m
                for m in raw_memories
                if (m.get("metadata") or {}).get("role") == "assistant"
            ]

            # Split previous consolidated by role too
            prev_user = [
                m
                for m in previous_consolidated
                if (m.get("metadata") or {}).get("role", "user") == "user"
            ]
            prev_assistant = [
                m
                for m in previous_consolidated
                if (m.get("metadata") or {}).get("role") == "assistant"
            ]

            # Only run assistant pass if there's an agent_id (required for
            # assistant-role memories) AND either raw memories or a
            # substantive summary to extract from.
            # NOTE: explicit bool() on assistant_raw to avoid Python's
            # short-circuit returning the list itself (which would leak
            # memory contents into log lines).
            run_assistant = bool(agent_id) and (
                bool(assistant_raw) or len(summary) > 50
            )

            logger.info(
                "Consolidation session %s: %d user raw, %d assistant raw, "
                "%d prev user consolidated, %d prev assistant consolidated, "
                "run_assistant=%s",
                session_id,
                len(user_raw),
                len(assistant_raw),
                len(prev_user),
                len(prev_assistant),
                run_assistant,
            )

            all_consolidated_facts: list[dict] = []
            all_raw_ids_map: list[tuple[list[dict], list[str]]] = []

            # Derive session date for event_date context
            session_date = (session.get("created_at") or "")[:10] or None

            # Consolidate user memories (always runs — summary may contain
            # user decisions even when no user-role raw memories exist)
            user_facts, user_ids_map = self._consolidate_role(
                session_id=session_id,
                raw_memories=user_raw,
                role="user",
                summary=summary,
                artifact_ids=artifact_ids,
                previous_consolidated=prev_user,
                session_date=session_date,
            )
            all_consolidated_facts.extend(user_facts)
            all_raw_ids_map.extend(user_ids_map)

            # Consolidate assistant memories (runs when agent_id is set
            # and there's content to consolidate). User consolidated output
            # is passed as cross-role context to prevent duplication.
            if run_assistant:
                asst_facts, asst_ids_map = self._consolidate_role(
                    session_id=session_id,
                    raw_memories=assistant_raw,
                    role="assistant",
                    summary=summary,
                    artifact_ids=artifact_ids,
                    previous_consolidated=prev_assistant,
                    session_date=session_date,
                    other_role_consolidated=user_facts,
                )
                all_consolidated_facts.extend(asst_facts)
                all_raw_ids_map.extend(asst_ids_map)

            if not all_consolidated_facts:
                logger.info(
                    "Consolidation session %s: no facts produced",
                    session_id,
                )
                guard_mutation()
                result.state = self._sessions.finalize_consolidation(
                    session_id,
                    expected_revision=session_revision,
                    token=consolidation_token,
                    consolidated_memory_ids=session.get("consolidated_memory_ids", []),
                    point_id=session_point_id,
                )
                return result

            # 6. Validate output quality (warn but proceed — don't block)
            validation_warning = self._validate_output(
                all_consolidated_facts, len(raw_memories)
            )
            if validation_warning:
                logger.warning(
                    "Session %s: consolidation quality warning: %s",
                    session_id,
                    validation_warning,
                )

            # 7. Persist the deterministic plan before the first output write.
            guard_mutation()
            operation_id = self._memory.revisions.operations.write(
                consolidation_token,
                {
                    "status": "planned",
                    "operation_kind": "consolidation",
                    "actor_kind": "consolidation",
                    "user_id": user_id,
                    "owner_id": owner_id,
                    "agent_id": agent_id,
                    "session_id": session_id,
                    "session_revision": session_revision,
                    "lineage_id": f"session:{session_id}",
                },
            )
            output_index = 0
            plan: list[dict[str, Any]] = []
            for batch_facts, _ in all_raw_ids_map:
                for fact in batch_facts:
                    fact = dict(fact)
                    fact["memory_id"] = str(
                        uuid.uuid5(
                            uuid.NAMESPACE_URL,
                            (
                                f"mnemory:consolidation:{consolidation_token}:"
                                f"{output_index}"
                            ),
                        )
                    )
                    fact["operation_id"] = operation_id
                    fact["source_session_id"] = session_id
                    plan.append(fact)
                    output_index += 1
            guard_mutation()
            self._sessions.update_consolidation_state(
                session_id,
                "consolidating",
                consolidation_token=consolidation_token,
                consolidation_plan=plan,
                point_id=session_point_id,
            )
            guard_mutation()
            self._memory.revisions.operations.write(
                consolidation_token,
                {"status": "planned", "plan": plan},
            )

            # 8. Store every output with its exact sources.
            all_stored_ids: list[str] = []
            all_stored_ids.extend(
                self._store_consolidated(
                    plan,
                    user_id=user_id,
                    owner_id=owner_id,
                    agent_id=agent_id,
                    raw_memories=raw_memories,
                    mutation_guard=mutation_guard,
                )
            )

            result.memories_produced = len(all_stored_ids)
            result.consolidated_memory_ids = all_stored_ids

            logger.info(
                "Consolidation session %s: stored %d consolidated memories",
                session_id,
                len(all_stored_ids),
            )

            # 9. Write consolidated_memory_ids to session (append to existing)
            # Append-only: previous consolidated memories are kept intact.
            # New IDs are added alongside old ones.
            merged_consolidated_ids = list(prev_consolidated_ids) + all_stored_ids
            guard_mutation()
            self._sessions.update_consolidation_state(
                session_id,
                "consolidating",  # still consolidating until supersede done
                consolidated_memory_ids=merged_consolidated_ids,
                point_id=session_point_id,
            )

            # 10. Retain exact referenced raw revisions as derivation sources.
            source_ids = list(
                dict.fromkeys(
                    source_id
                    for fact in plan
                    for source_id in fact.get("derived_from", [])
                )
            )
            guard_mutation()
            self._memory.revisions.mark_source(
                source_ids,
                operation_id=operation_id,
                user_id=user_id,
                owner_id=owner_id,
                session_agent_id=agent_id,
                mutation_guard=mutation_guard,
            )
            result.memories_superseded = len(source_ids)
            guard_mutation()
            self._memory.revisions.operations.write(
                consolidation_token,
                {
                    "status": "sources_marked",
                    "result_revision_ids": all_stored_ids,
                    "source_revision_ids": source_ids,
                },
            )

            # 11. Set final state (with merged IDs)
            guard_mutation()
            final_state = self._sessions.finalize_consolidation(
                session_id,
                expected_revision=session_revision,
                token=consolidation_token,
                consolidated_memory_ids=merged_consolidated_ids,
                point_id=session_point_id,
            )
            guard_mutation()
            self._memory.revisions.operations.write(
                consolidation_token,
                {
                    "status": "committed",
                    "result_revision_ids": all_stored_ids,
                    "source_revision_ids": source_ids,
                },
            )
            result.state = final_state

            reconsolidation = bool(prev_consolidated_ids)
            logger.info(
                "Session %s %s: %d raw -> %d consolidated, %d superseded",
                session_id,
                "re-consolidated" if reconsolidation else "consolidated",
                len(raw_memories),
                len(all_stored_ids),
                len(source_ids),
            )

        except Exception as exc:
            logger.exception("Consolidation failed for session %s", session_id)
            result.error = "Unexpected error during consolidation"
            result.state = "failed"
            try:
                persisted = (
                    self._sessions.get(
                        session_id,
                        point_id=(
                            session.get("_point_id")
                            if "session" in locals() and session
                            else None
                        ),
                    )
                    or {}
                )
                if persisted.get("consolidation_plan"):
                    result.state = "consolidating"
                else:
                    guard_mutation()
                    self._sessions.update_consolidation_state(
                        session_id,
                        "failed",
                        error_code=type(exc).__name__,
                        consolidation_token=None,
                        point_id=persisted.get("_point_id"),
                    )
            except Exception:
                pass

        result.duration_seconds = time.monotonic() - t0

        # Record metrics
        if self._collector is not None and result.state in (
            "consolidated",
            "failed",
        ):
            self._collector.record_consolidation_run(
                user_id=session.get("user_id", "") if session else "",
                run_type="session",
                memories_produced=result.memories_produced,
                memories_superseded=result.memories_superseded,
                duration_seconds=result.duration_seconds,
                validation_failed=bool(result.error),
            )

        return result

    def recover_incomplete(self, user_id: str) -> int:
        """Check for and recover orphaned 'consolidating' sessions.

        If consolidated_memory_ids exist, resume from supersede step.
        Otherwise, reset to 'idle' for retry.

        Returns number of sessions recovered.
        """
        recovered = 0
        try:
            sessions = self._sessions.list_for_user(
                user_id, consolidation_state="consolidating"
            )
            for session in sessions:
                self.recover_session(session)
                recovered += 1
        except Exception:
            logger.exception("Recovery failed for user %s", user_id)
        return recovered

    def recover_session(
        self,
        session: dict[str, Any],
        *,
        mutation_guard: Callable[[], None] | None = None,
    ) -> bool:
        """Recover one consolidating session.

        Returns True when existing consolidation output completed recovery.
        Returns False when the session was reset to idle for retry.
        """
        sid = session.get("session_id", "")
        user_id = session.get("user_id", "")
        plan = session.get("consolidation_plan")
        token = session.get("consolidation_token")

        def guard_mutation() -> None:
            if mutation_guard is not None:
                mutation_guard()

        if isinstance(plan, list) and token:
            logger.info("Recovering session %s from durable consolidation plan", sid)
            owner_id = session.get("owner_id") or user_id
            raw_memories = self._fetch_raw_memories(
                session.get("memory_ids", []),
                user_id,
                owner_id,
                session.get("agent_id"),
            )
            stored_ids = self._store_consolidated(
                plan,
                user_id=user_id,
                owner_id=owner_id,
                agent_id=session.get("agent_id"),
                raw_memories=raw_memories,
                mutation_guard=mutation_guard,
            )
            source_ids = list(
                dict.fromkeys(
                    source_id
                    for fact in plan
                    for source_id in fact.get("derived_from", [])
                )
            )
            operation = self._memory.revisions.operations.get(token)
            operation_id = (
                operation.get("operation_id")
                if operation
                else self._memory.revisions.operations.write(
                    token,
                    {
                        "status": "recovering",
                        "operation_kind": "consolidation",
                        "actor_kind": "consolidation",
                        "user_id": user_id,
                        "owner_id": owner_id,
                        "session_id": sid,
                        "lineage_id": f"session:{sid}",
                        "plan": plan,
                    },
                )
            )
            guard_mutation()
            self._memory.revisions.mark_source(
                source_ids,
                operation_id=operation_id,
                user_id=user_id,
                owner_id=owner_id,
                session_agent_id=session.get("agent_id"),
                mutation_guard=mutation_guard,
            )
            previous_ids = session.get("consolidated_memory_ids") or []
            merged_ids = list(dict.fromkeys([*previous_ids, *stored_ids]))
            guard_mutation()
            self._sessions.finalize_consolidation(
                sid,
                expected_revision=int(session.get("session_revision", 1)),
                token=token,
                consolidated_memory_ids=merged_ids,
                point_id=session.get("_point_id"),
            )
            guard_mutation()
            self._memory.revisions.operations.write(
                token,
                {
                    "status": "committed",
                    "result_revision_ids": stored_ids,
                    "source_revision_ids": source_ids,
                },
            )
            return True

        consolidated_ids = session.get("consolidated_memory_ids")
        if consolidated_ids:
            logger.info("Recovering legacy session %s without exact source plan", sid)
            guard_mutation()
            self._sessions.update_consolidation_state(
                sid,
                "consolidated",
                consolidated_memory_ids=consolidated_ids,
                point_id=session.get("_point_id"),
            )
            return True

        logger.info("Recovering session %s: resetting to idle", sid)
        guard_mutation()
        self._sessions.update_consolidation_state(
            sid,
            "idle",
            consolidation_token=None,
            consolidation_plan=None,
            point_id=session.get("_point_id"),
        )
        return False

    def _fetch_raw_memories(
        self,
        memory_ids: list[str],
        user_id: str,
        owner_id: str | None = None,
        agent_id: str | None = None,
    ) -> list[dict]:
        """Fetch raw memories by IDs, filtering to only unsuperseded raw."""
        # Deduplicate IDs (preserve order) — sessions may accumulate
        # duplicate IDs from the remember endpoint's dedup-update path.
        memory_ids = list(dict.fromkeys(memory_ids))

        memories = []
        not_found = 0
        skipped_layer = 0
        skipped_superseded = 0
        fetch_errors = 0
        for mid in memory_ids:
            try:
                result = self._vector.get_by_id(mid)
                if result is None:
                    not_found += 1
                    continue
                if not self._in_session_scope(
                    result,
                    user_id=user_id,
                    owner_id=owner_id or user_id,
                    agent_id=agent_id,
                ):
                    fetch_errors += 1
                    logger.warning("Memory %s is outside the session owner", mid)
                    continue
                meta = result.get("metadata") or {}
                # Only include raw, unsuperseded memories
                layer = meta.get("memory_layer", "raw")
                if layer != "raw":
                    skipped_layer += 1
                    continue
                if meta.get("superseded_by"):
                    skipped_superseded += 1
                    continue
                if meta.get("revision_state", "active") != "active":
                    skipped_superseded += 1
                    continue
                memories.append(result)
            except Exception:
                fetch_errors += 1
                logger.warning("Could not fetch memory %s", mid, exc_info=True)
        # Always log summary at INFO so production can diagnose issues
        skipped = not_found + skipped_layer + skipped_superseded + fetch_errors
        if skipped > 0:
            logger.info(
                "Fetch raw memories: %d eligible, %d skipped "
                "(not_found=%d, wrong_layer=%d, superseded=%d, errors=%d) "
                "from %d total",
                len(memories),
                skipped,
                not_found,
                skipped_layer,
                skipped_superseded,
                fetch_errors,
                len(memory_ids),
            )
        return memories

    @staticmethod
    def _in_session_scope(
        memory: dict[str, Any],
        *,
        user_id: str,
        owner_id: str,
        agent_id: str | None,
    ) -> bool:
        """Check tenant and agent scope for session-linked memory IDs."""
        if memory.get("user_id") != user_id or memory.get("owner_id") not in (
            None,
            owner_id,
        ):
            return False
        memory_agent_id = memory.get("agent_id")
        if agent_id is None:
            return memory_agent_id is None
        return (
            memory_agent_id is None
            or memory_agent_id == agent_id
            or memory_agent_id.startswith(agent_id + ":")
        )

    def _consolidate_role(
        self,
        *,
        session_id: str,
        raw_memories: list[dict],
        role: str,
        summary: str,
        artifact_ids: set[str],
        previous_consolidated: list[dict],
        session_date: str | None = None,
        other_role_consolidated: list[dict] | None = None,
    ) -> tuple[list[dict], list[tuple[list[dict], list[str]]]]:
        """Consolidate raw memories for a single role (user or assistant).

        Handles batching when there are more memories than batch_size.
        Each batch gets accumulated context from prior batches.

        Args:
            other_role_consolidated: Consolidated facts from the other role's
                pass (e.g., user facts when consolidating assistant). Passed
                as read-only context to prevent cross-role duplication.

        Returns:
            Tuple of (all_facts, raw_ids_map) where:
            - all_facts: list of consolidated fact dicts (with 'role' set)
            - raw_ids_map: list of (batch_facts, batch_raw_ids) tuples
        """
        from mnemory.llm import parse_json_response
        from mnemory.prompts import build_consolidation_prompt

        batch_size = self._config.memory.consolidation_batch_size
        all_facts: list[dict] = []
        raw_ids_map: list[tuple[list[dict], list[str]]] = []

        # Split into batches if needed
        if len(raw_memories) <= batch_size:
            batches = [raw_memories]
        else:
            batches = [
                raw_memories[i : i + batch_size]
                for i in range(0, len(raw_memories), batch_size)
            ]
            logger.info(
                "Consolidation session %s [%s]: splitting %d memories into %d batches",
                session_id,
                role,
                len(raw_memories),
                len(batches),
            )

        # Accumulated context from prior batches (prevents cross-batch duplication)
        accumulated: list[dict] = []

        for batch_idx, batch in enumerate(batches):
            if len(batches) > 1:
                logger.info(
                    "Consolidation session %s [%s]: batch %d/%d (%d memories)",
                    session_id,
                    role,
                    batch_idx + 1,
                    len(batches),
                    len(batch),
                )

            # Combine previous consolidated (from prior runs) with
            # accumulated context (from prior batches in this run)
            full_context = list(previous_consolidated)
            full_context.extend(accumulated)

            batch_artifact_ids = artifact_ids & {m["id"] for m in batch}

            messages, json_schema = build_consolidation_prompt(
                summary=summary,
                raw_memories=batch,
                role=role,
                artifact_memory_ids=batch_artifact_ids,
                previous_consolidated=full_context if full_context else None,
                session_date=session_date,
                other_role_consolidated=other_role_consolidated,
            )

            response_text = self._llm.generate(
                messages,
                json_schema=json_schema,
                temperature=0.3,
                operation="consolidation",
            )

            parsed = parse_json_response(response_text)

            if not parsed or not isinstance(parsed.get("memories"), list):
                logger.warning(
                    "Consolidation session %s [%s] batch %d: invalid LLM output",
                    session_id,
                    role,
                    batch_idx + 1,
                )
                continue

            batch_facts = parsed["memories"]
            batch_raw_ids = [m["id"] for m in batch]
            source_aliases = {
                f"S{index}": memory_id for index, memory_id in enumerate(batch_raw_ids)
            }

            # Resolve exact source aliases before any durable write.
            valid_facts: list[dict[str, Any]] = []
            for f in batch_facts:
                aliases = f.get("source_ids")
                if not batch and aliases is None:
                    aliases = []
                if not isinstance(aliases, list):
                    logger.warning(
                        "Consolidation session %s [%s] batch %d: "
                        "output omitted source_ids",
                        session_id,
                        role,
                        batch_idx + 1,
                    )
                    continue
                if batch and not aliases:
                    logger.warning(
                        "Consolidation session %s [%s] batch %d: "
                        "output has no raw source",
                        session_id,
                        role,
                        batch_idx + 1,
                    )
                    continue
                if any(alias not in source_aliases for alias in aliases):
                    logger.warning(
                        "Consolidation session %s [%s] batch %d: "
                        "output referenced an unknown source alias",
                        session_id,
                        role,
                        batch_idx + 1,
                    )
                    continue
                f["derived_from"] = list(
                    dict.fromkeys(source_aliases[alias] for alias in aliases)
                )
                f["role"] = role
                f["memory_type"] = _normalize_consolidation_memory_type(
                    f.get("memory_type")
                )
                valid_facts.append(f)
            batch_facts = valid_facts

            logger.info(
                "Consolidation session %s [%s] batch %d: produced %d facts",
                session_id,
                role,
                batch_idx + 1,
                len(batch_facts),
            )

            all_facts.extend(batch_facts)
            raw_ids_map.append((batch_facts, batch_raw_ids))

            # Add to accumulated context for next batch
            for f in batch_facts:
                accumulated.append(
                    {
                        "memory": f.get("text", ""),
                        "metadata": {
                            "memory_type": f.get("memory_type", "episodic"),
                            "role": role,
                            "importance": f.get("importance", "normal"),
                        },
                    }
                )

        return all_facts, raw_ids_map

    def _validate_output(self, facts: list[dict], raw_count: int) -> str | None:
        """Validate consolidation output quality.

        Returns error message if validation fails, None if OK.
        """
        if not facts:
            return "No consolidated memories produced"

        # Check minimum content length
        for f in facts:
            text = f.get("text", "")
            if len(text) < _MIN_CONTENT_LENGTH:
                return f"Consolidated memory too short ({len(text)} chars)"

        # Check minimum ratio (at least 1 per 5 raw)
        min_expected = max(1, int(raw_count * _MIN_CONSOLIDATED_RATIO))
        if len(facts) < min_expected:
            return (
                f"Too few consolidated memories: {len(facts)} "
                f"(expected at least {min_expected} from {raw_count} raw)"
            )

        return None

    def _store_consolidated(
        self,
        facts: list[dict],
        *,
        user_id: str,
        owner_id: str | None = None,
        agent_id: str | None,
        raw_memories: list[dict] | None = None,
        mutation_guard: Callable[[], None] | None = None,
    ) -> list[str]:
        """Store consolidated memories via add_memory(infer=False).

        Returns list of stored memory IDs.
        """
        # Collect fallback categories from raw memories
        fallback_categories: list[str] = []
        if raw_memories:
            cats_set: set[str] = set()
            for mem in raw_memories:
                meta = mem.get("metadata") or {}
                for cat in meta.get("categories", []):
                    if cat:
                        cats_set.add(cat)
            fallback_categories = sorted(cats_set)

        stored_ids: list[str] = []
        for fact in facts:
            try:
                text = fact.get("text", "")
                if not text:
                    continue

                # Determine effective agent_id: only set for assistant role
                effective_agent_id = (
                    agent_id if fact.get("role") == "assistant" else None
                )

                # Use LLM-assigned categories, fall back to raw memory
                # categories if the LLM returned an empty list
                categories = fact.get("categories") or fallback_categories

                memory_id = fact["memory_id"]
                revision_metadata = RevisionService.initial_metadata(
                    memory_id,
                    operation_id=fact["operation_id"],
                    derived_from=fact.get("derived_from"),
                    source_session_id=fact.get("source_session_id"),
                )
                consumed_roots: list[str] = []
                for source_id in fact.get("derived_from") or []:
                    source = self._memory.vector.get_by_id(source_id)
                    source_meta = (source or {}).get("metadata") or {}
                    consumed_roots.extend(source_meta.get("evidence_root_ids") or [])
                    consumed_roots.extend(
                        source_meta.get("consumed_evidence_root_ids") or []
                    )
                unique_consumed = list(dict.fromkeys(consumed_roots))
                revision_metadata.update(
                    {
                        "source_kind": "consolidation",
                        "validation_eligible": False,
                        "evidence_root_ids": [],
                        "consumed_evidence_root_ids": unique_consumed,
                        "validation_count": 0,
                        "validation_strength": 0.0,
                        "validation_state": "unverified",
                    }
                )
                if mutation_guard is not None:
                    mutation_guard()
                result = self._memory.add_memory(
                    text,
                    user_id=user_id,
                    owner_id=owner_id or user_id,
                    agent_id=effective_agent_id,
                    infer=False,
                    _trusted=True,
                    memory_type=fact.get("memory_type", "episodic"),
                    categories=categories,
                    importance=fact.get("importance", "normal"),
                    pinned=fact.get("pinned", False),
                    role=fact.get("role", "user"),
                    event_date=fact.get("event_date"),
                    _memory_id=memory_id,
                    _revision_metadata=revision_metadata,
                    _mutation_guard=mutation_guard,
                )

                results = result.get("results", [])
                if results:
                    mem_id = results[0].get("id", "")
                    if mem_id:
                        stored_ids.append(mem_id)
            except Exception:
                logger.warning(
                    "Failed to store consolidated memory (type=%s, role=%s)",
                    fact.get("memory_type", "unknown"),
                    fact.get("role", "user"),
                    exc_info=True,
                )
                raise
        return stored_ids
