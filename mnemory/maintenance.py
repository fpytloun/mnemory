"""Periodic background maintenance service for mnemory.

Runs automatic fsck (memory consistency checks) for all users at a
configurable interval and auto-applies fixes that meet confidence and
severity thresholds.

Usage::

    service = MaintenanceService(config=cfg, fsck=fsck_service, collector=collector)
    await service.start()
    # ... server runs ...
    await service.stop()
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mnemory.config import Config
    from mnemory.fsck import FsckService
    from mnemory.metrics import MetricsCollector
    from mnemory.storage.vector import VectorStore

logger = logging.getLogger(__name__)

# Severity ordering for threshold comparison.
_SEVERITY_ORDER: dict[str, int] = {"low": 0, "medium": 1, "high": 2}


def gc_superseded_raw(
    vector: VectorStore,
    user_id: str,
    *,
    retention_days: int = 30,
) -> dict[str, int]:
    """Retain raw evidence. Kept as a compatibility no-op for old callers."""
    del vector, user_id, retention_days
    return {"deleted": 0}


class MaintenanceService:
    """Periodic background maintenance: auto-fsck for all users.

    When ``fsck_auto_interval`` is 0 (default), the service is disabled
    and ``start()`` / ``stop()`` are no-ops.

    The loop is sleep-first: it waits the full interval before running the
    first check. This avoids hammering the LLM immediately on startup.
    """

    def __init__(
        self,
        config: Config,
        fsck: FsckService,
        collector: MetricsCollector | None = None,
        consolidation: Any | None = None,
    ) -> None:
        self._config = config
        self._fsck = fsck
        self._collector = collector
        self._task: asyncio.Task | None = None
        self._consolidation = consolidation
        self._consolidation_task: asyncio.Task | None = None

        # Scheduling state — exposed via properties for the stats endpoint.
        self._next_run_at: float | None = None  # Unix timestamp
        self._running: bool = False
        self._last_check_ids: dict[str, str] = {}  # user_id → check_id

    # ── Public properties ────────────────────────────────────────────

    @property
    def next_run_at(self) -> float | None:
        """Unix timestamp of the next scheduled auto-fsck run.

        Returns None if auto-fsck is disabled or a run is in progress.
        """
        return self._next_run_at

    @property
    def is_running(self) -> bool:
        """Whether an auto-fsck run is currently in progress."""
        return self._running

    @property
    def last_check_ids(self) -> dict[str, str]:
        """Mapping of user_id → most recent auto-fsck check_id."""
        return dict(self._last_check_ids)

    # ── Lifecycle ────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the background maintenance loop (no-op if disabled)."""
        interval = self._config.memory.fsck_auto_interval
        if interval <= 0:
            logger.debug("Auto-fsck disabled (FSCK_AUTO_INTERVAL=0)")
            return
        logger.info(
            "Auto-fsck enabled: interval=%dh, min_confidence=%.2f, min_severity=%s",
            interval,
            self._config.memory.fsck_auto_min_confidence,
            self._config.memory.fsck_auto_min_severity,
        )
        self._task = asyncio.create_task(self._loop(), name="maintenance-autofsck")

        # Start consolidation loop if service is available
        if self._consolidation is not None:
            consolidation_interval = self._config.memory.consolidation_check_interval
            if consolidation_interval > 0:
                logger.info(
                    "Consolidation enabled: checking every %ds for idle sessions",
                    consolidation_interval,
                )
                self._consolidation_task = asyncio.create_task(
                    self._consolidation_loop(),
                    name="maintenance-consolidation",
                )

    async def stop(self) -> None:
        """Stop the background maintenance loop."""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
            self._next_run_at = None
            logger.debug("Auto-fsck maintenance loop stopped")

        if self._consolidation_task is not None:
            self._consolidation_task.cancel()
            try:
                await self._consolidation_task
            except asyncio.CancelledError:
                pass
            self._consolidation_task = None
            logger.debug("Consolidation maintenance loop stopped")

    # ── Run Now ──────────────────────────────────────────────────────

    def start_run_now(self, user_id: str) -> str:
        """Start an immediate auto-fsck check for a single user.

        Creates the check synchronously and returns the check_id. The
        caller is responsible for running the check in the background
        (via :meth:`finish_run_now`).

        Args:
            user_id: The user to run auto-fsck for.

        Returns:
            The check_id of the created check.

        Raises:
            RuntimeError: If auto-fsck is disabled.
        """
        if self._config.memory.fsck_auto_interval <= 0:
            raise RuntimeError("Auto-fsck is disabled (FSCK_AUTO_INTERVAL=0)")

        check = self._fsck.start_check(user_id=user_id)
        self._last_check_ids[user_id] = check.check_id
        logger.info(
            "Auto-fsck: manual run started for user %s (check %s)",
            user_id,
            check.check_id,
        )
        return check.check_id

    def finish_run_now(self, check_id: str, user_id: str) -> None:
        """Run the check and auto-apply qualifying fixes (blocking).

        Intended to be called in a background task after
        :meth:`start_run_now`. This is synchronous and may take minutes.
        """
        min_confidence = self._config.memory.fsck_auto_min_confidence
        min_severity = self._config.memory.fsck_auto_min_severity

        # Auto-fsck is durable-only by design. Raw memories are handled by the
        # separate session/consolidation lifecycle.
        # Manual run-now is always a full scan (incremental=False).
        self._fsck.run_check(check_id, include_raw=False, incremental=False)

        check = self._fsck.get_check(check_id)
        if check is None or check.status != "completed":
            status = check.status if check else "not found"
            logger.warning(
                "Auto-fsck (manual): check %s ended with status=%s",
                check_id,
                status,
            )
            return

        # Filter issues that meet the thresholds
        qualifying_ids = [
            issue.issue_id
            for issue in check.issues
            if self._meets_thresholds(issue, min_confidence, min_severity)
        ]

        fixes_applied = 0
        fixes_failed = 0

        if qualifying_ids:
            # Set status to "applying" so the UI keeps polling until
            # auto-apply is done (prevents race where UI sees "completed"
            # before applied_issue_ids is populated).
            check.status = "applying"

            logger.info(
                "Auto-fsck (manual): applying %d/%d qualifying fixes for user %s",
                len(qualifying_ids),
                len(check.issues),
                user_id,
            )
            result = self._fsck.apply_check(check_id, qualifying_ids)
            fixes_applied = result.get("applied", 0)
            fixes_failed = result.get("failed", 0)
            logger.info(
                "Auto-fsck (manual): user %s — applied=%d, failed=%d",
                user_id,
                fixes_applied,
                fixes_failed,
            )
        else:
            logger.debug(
                "Auto-fsck (manual): no qualifying fixes for user %s "
                "(%d issues below threshold)",
                user_id,
                len(check.issues),
            )

        # Mark completed after auto-apply is done so the UI sees the
        # final state with applied_issue_ids populated.
        check.status = "completed"

        if self._collector is not None:
            self._collector.record_autofsck_run(
                user_id=user_id,
                issues_found=len(qualifying_ids),
                fixes_applied=fixes_applied,
                fixes_failed=fixes_failed,
            )

    # ── Loop ─────────────────────────────────────────────────────────

    async def _loop(self) -> None:
        """Main maintenance loop: sleep first, then run."""
        interval_seconds = self._config.memory.fsck_auto_interval * 3600
        while True:
            self._next_run_at = time.time() + interval_seconds
            try:
                await asyncio.sleep(interval_seconds)
            except asyncio.CancelledError:
                self._next_run_at = None
                return
            self._next_run_at = None
            try:
                self._running = True
                await self._run_auto_fsck()
            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception("Auto-fsck run failed unexpectedly")
            finally:
                self._running = False

    # ── Consolidation loop ─────────────────────────────────────────

    async def _consolidation_loop(self) -> None:
        """Consolidation maintenance loop: check for idle sessions periodically."""
        interval = self._config.memory.consolidation_check_interval
        if interval <= 0:
            return
        while True:
            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                return
            try:
                await self._run_consolidation()
            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception("Consolidation run failed unexpectedly")

    async def _run_consolidation(self) -> None:
        """Run within-session consolidation for all users with idle sessions."""
        candidates: list[dict[str, Any]] = await asyncio.to_thread(
            self._consolidation._sessions.scan_consolidation_candidates
        )
        if not candidates:
            return
        by_user: dict[str, list[dict[str, Any]]] = {}
        for session in candidates:
            by_user.setdefault(session["user_id"], []).append(session)

        total_processed = 0
        total_succeeded = 0
        total_failed = 0
        cutoff = datetime.now(timezone.utc) - timedelta(
            seconds=self._config.memory.consolidation_idle_threshold
        )

        for user_id, sessions in by_user.items():
            try:
                pending: list[dict[str, Any]] = []
                for session in sessions:
                    if session.get("consolidation_state") == "consolidating":
                        try:
                            completed = await asyncio.to_thread(
                                self._consolidation.recover_session,
                                session,
                            )
                        except Exception:
                            logger.exception(
                                "Recovery failed for session %s, continuing",
                                session.get("session_id", ""),
                            )
                            continue
                        if completed:
                            continue

                    updated_at_str = session.get("updated_at", "")
                    if not updated_at_str or not session.get("memory_ids"):
                        continue
                    try:
                        updated_at = datetime.fromisoformat(updated_at_str)
                    except (TypeError, ValueError):
                        continue
                    if updated_at.tzinfo is None:
                        updated_at = updated_at.replace(tzinfo=timezone.utc)
                    if updated_at < cutoff:
                        pending.append(session)
                if not pending:
                    continue

                logger.info(
                    "Consolidation: %d pending session(s) for user %s",
                    len(pending),
                    user_id,
                )

                for session in pending:
                    session_id = session.get("session_id", "")
                    if not session_id:
                        continue
                    try:
                        result = await asyncio.to_thread(
                            self._consolidation.consolidate_session,
                            session_id,
                            session_record=session,
                        )
                        total_processed += 1
                        if result.state == "consolidated":
                            total_succeeded += 1
                        elif result.state == "failed":
                            total_failed += 1
                            logger.warning(
                                "Consolidation failed for session %s: %s",
                                session_id,
                                result.error,
                            )
                    except Exception:
                        total_processed += 1
                        total_failed += 1
                        logger.exception(
                            "Consolidation error for session %s", session_id
                        )
            except Exception:
                logger.exception("Consolidation error for user %s, continuing", user_id)

        if total_processed > 0:
            logger.info(
                "Consolidation run complete: %d processed, %d succeeded, %d failed",
                total_processed,
                total_succeeded,
                total_failed,
            )

    # ── Core logic ───────────────────────────────────────────────────

    async def _run_auto_fsck(self) -> None:
        """Run fsck for every user and auto-apply qualifying fixes."""
        # list_user_ids() is a synchronous Qdrant scroll — run in thread pool
        # to avoid blocking the event loop.
        user_ids: list[str] = await asyncio.to_thread(self._fsck._vector.list_user_ids)

        if not user_ids:
            logger.debug("Auto-fsck: no users found, skipping run")
            return

        logger.info("Auto-fsck: starting run for %d user(s)", len(user_ids))
        t0 = time.monotonic()

        for user_id in user_ids:
            try:
                await self._run_for_user(user_id)
            except Exception:
                logger.exception(
                    "Auto-fsck: error processing user %s, continuing", user_id
                )

        elapsed = time.monotonic() - t0
        logger.info("Auto-fsck: run complete in %.1fs", elapsed)

    async def _run_for_user(self, user_id: str) -> None:
        """Run fsck for a single user and apply qualifying fixes."""
        min_confidence = self._config.memory.fsck_auto_min_confidence
        min_severity = self._config.memory.fsck_auto_min_severity

        # start_check() is synchronous (just creates a FsckCheck object)
        check = self._fsck.start_check(user_id=user_id)
        check_id = check.check_id

        # Track the check_id for this user (for UI retrieval).
        self._last_check_ids[user_id] = check_id

        logger.debug("Auto-fsck: running check %s for user %s", check_id, user_id)

        # run_check() is synchronous and CPU/IO-bound (LLM calls via requests).
        # Run in a thread pool to avoid blocking the event loop.
        # Auto-fsck is durable-only by design. Raw memories are handled by the
        # separate session/consolidation lifecycle.
        await asyncio.to_thread(
            self._fsck.run_check, check_id, include_raw=False, incremental=True
        )

        check = self._fsck.get_check(check_id)
        if check is None or check.status != "completed":
            status = check.status if check else "not found"
            logger.warning(
                "Auto-fsck: check %s for user %s ended with status=%s",
                check_id,
                user_id,
                status,
            )
        else:
            # Filter issues that meet the thresholds
            qualifying_ids = [
                issue.issue_id
                for issue in check.issues
                if self._meets_thresholds(issue, min_confidence, min_severity)
            ]

            issues_found = len(qualifying_ids)
            fixes_applied = 0
            fixes_failed = 0

            if qualifying_ids:
                # Set status to "applying" so the UI keeps polling until
                # auto-apply is done (prevents race where UI sees "completed"
                # before applied_issue_ids is populated).
                check.status = "applying"

                logger.info(
                    "Auto-fsck: applying %d/%d qualifying fixes for user %s",
                    issues_found,
                    len(check.issues),
                    user_id,
                )
                # apply_check() is synchronous (direct vector store mutations)
                result: dict[str, Any] = await asyncio.to_thread(
                    self._fsck.apply_check, check_id, qualifying_ids
                )
                fixes_applied = result.get("applied", 0)
                fixes_failed = result.get("failed", 0)
                logger.info(
                    "Auto-fsck: user %s — applied=%d, failed=%d",
                    user_id,
                    fixes_applied,
                    fixes_failed,
                )
            else:
                logger.debug(
                    "Auto-fsck: no qualifying fixes for user %s "
                    "(%d issues below threshold)",
                    user_id,
                    len(check.issues),
                )

            # Mark completed after auto-apply is done so the UI sees the
            # final state with applied_issue_ids populated.
            check.status = "completed"

            if self._collector is not None:
                self._collector.record_autofsck_run(
                    user_id=user_id,
                    issues_found=issues_found,
                    fixes_applied=fixes_applied,
                    fixes_failed=fixes_failed,
                )

    # ── Helpers ──────────────────────────────────────────────────────

    @classmethod
    def _meets_thresholds(
        cls,
        issue: Any,
        min_confidence: float,
        min_severity: str,
    ) -> bool:
        """Return True if the issue meets both confidence and severity thresholds.

        Args:
            issue: FsckIssue object with ``confidence`` (float) and
                   ``severity`` (str) attributes.
            min_confidence: Minimum confidence score (0.0-1.0).
            min_severity: Minimum severity level ("low", "medium", "high").
        """
        confidence = getattr(issue, "confidence", 0.0) or 0.0
        severity = getattr(issue, "severity", "low") or "low"

        if confidence < min_confidence:
            return False

        issue_sev_order = _SEVERITY_ORDER.get(severity.lower(), 0)
        min_sev_order = _SEVERITY_ORDER.get(min_severity.lower(), 0)
        return issue_sev_order >= min_sev_order
