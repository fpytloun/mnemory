"""Data migration framework for mnemory.

Migrations run automatically on startup before the server accepts
requests. State is tracked in a separate ``_mnemory_meta`` Qdrant
collection (not the filesystem) to support stateless Kubernetes
deployments where no persistent local storage is available.

Each migration must be **idempotent** — safe to re-run if interrupted
mid-way or if multiple instances start simultaneously. Progress
checkpoints are saved after each batch for resumable migrations.

Usage::

    runner = MigrationRunner(client, get_migrations(collection, sparse))
    runner.run_pending()
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

logger = logging.getLogger(__name__)

# Qdrant collection for migration state. Separate from the main memory
# collection to avoid polluting search results with metadata points.
META_COLLECTION = "_mnemory_meta"

# Well-known point ID for the migration state document.
META_POINT_ID = "00000000-0000-0000-0000-000000000000"


class Migration(ABC):
    """Base class for data migrations.

    Subclasses must define ``id`` (unique, e.g. ``001_add_sparse_vectors``)
    and ``description`` (human-readable), and implement ``run()``.

    Migrations receive the Qdrant client, optional progress checkpoint
    (for resuming), and a callback to persist state between batches.
    """

    id: str
    description: str

    @abstractmethod
    def run(
        self,
        client: QdrantClient,
        *,
        progress: dict[str, Any] | None,
        state_callback: Callable[[dict[str, Any]], None],
        state: dict[str, Any],
    ) -> None:
        """Execute the migration.

        Args:
            client: Qdrant client for all operations.
            progress: Checkpoint from a previous interrupted run, or None.
            state_callback: Callable(state) to persist intermediate state.
            state: The full migration state dict (for checkpoint writes).
        """
        ...


class MigrationRunner:
    """Runs pending data migrations on startup.

    State is stored in a ``_mnemory_meta`` Qdrant collection as a single
    point with a well-known UUID. The payload tracks which migrations
    have been applied and optional per-migration progress checkpoints.
    """

    def __init__(
        self,
        client: QdrantClient,
        migrations: list[Migration],
    ):
        self._client = client
        self._migrations = migrations

    def _ensure_meta_collection(self) -> None:
        """Create the ``_mnemory_meta`` collection if it doesn't exist.

        Uses a 1-dimensional dummy vector config — the collection only
        stores migration metadata in payloads, not real vectors.
        """
        existing = [c.name for c in self._client.get_collections().collections]
        if META_COLLECTION not in existing:
            logger.debug("Creating migration metadata collection '%s'", META_COLLECTION)
            self._client.create_collection(
                collection_name=META_COLLECTION,
                vectors_config=VectorParams(size=1, distance=Distance.COSINE),
            )

    def _load_state(self) -> dict[str, Any]:
        """Load migration state from ``_mnemory_meta``."""
        try:
            result = self._client.retrieve(
                collection_name=META_COLLECTION,
                ids=[META_POINT_ID],
                with_payload=True,
            )
            if result:
                return dict(result[0].payload or {})
        except Exception:
            logger.debug("No existing migration state found")
        return {}

    def _save_state(self, state: dict[str, Any]) -> None:
        """Persist migration state to ``_mnemory_meta``."""
        self._client.upsert(
            collection_name=META_COLLECTION,
            points=[
                PointStruct(
                    id=META_POINT_ID,
                    vector=[0.0],  # dummy vector
                    payload=state,
                )
            ],
        )

    def run_pending(self) -> None:
        """Run all pending migrations in order.

        For each migration that hasn't been applied yet:
        1. Mark it as "running" (optimistic lock for multi-pod safety)
        2. Execute the migration
        3. Mark it as complete

        If a migration was previously interrupted (``running`` marker
        present), it is re-run since all migrations are idempotent.
        """
        self._ensure_meta_collection()
        state = self._load_state()
        applied = list(state.get("applied", []))

        for migration in self._migrations:
            if migration.id in applied:
                logger.debug("Migration %s already applied, skipping", migration.id)
                continue

            # Check for interrupted previous run
            running_key = f"{migration.id}:running"
            if running_key in applied:
                logger.warning(
                    "Migration %s was previously interrupted or is running "
                    "on another instance — re-running (idempotent)",
                    migration.id,
                )

            # Optimistic lock: mark as running before starting
            if running_key not in applied:
                applied.append(running_key)
            state["applied"] = applied
            self._save_state(state)

            logger.info(
                "Running migration: %s — %s",
                migration.id,
                migration.description,
            )

            try:
                progress = state.get(f"{migration.id}_progress")
                migration.run(
                    self._client,
                    progress=progress,
                    state_callback=self._save_state,
                    state=state,
                )

                # Mark complete: replace running marker with final ID
                applied = [m for m in state.get("applied", []) if m != running_key]
                applied.append(migration.id)
                state["applied"] = applied
                state.pop(f"{migration.id}_progress", None)
                state["last_run"] = datetime.now(timezone.utc).isoformat()
                self._save_state(state)
                logger.info("Migration %s completed successfully", migration.id)

            except Exception:
                logger.exception("Migration %s failed", migration.id)
                # Remove running marker on failure so it can be retried
                applied = [m for m in state.get("applied", []) if m != running_key]
                state["applied"] = applied
                self._save_state(state)
                raise


# Temp collection used during collection recreation when the Qdrant
# server doesn't support adding new named vectors to existing collections.
_TEMP_COLLECTION = "_mnemory_migration_temp"


# ── Migration 001: Add BM25 sparse vectors ──────────────────────────


class AddSparseVectorsMigration(Migration):
    """Add BM25 sparse vectors to existing memories for hybrid search.

    Two paths depending on Qdrant server capabilities:

    **Fast path** (QdrantLocal / future Qdrant versions):
    1. ``update_collection()`` adds sparse vector config to the collection
    2. Scroll existing points in batches, generate BM25 sparse vectors
    3. ``update_vectors()`` adds sparse vectors without touching dense vectors

    **Recreation path** (remote Qdrant ≤1.16 — cannot add new named vectors):
    1. ``update_collection()`` fails (server rejects adding new vector names)
    2. Create a temp collection with both dense + sparse config
    3. Copy all points from original → temp (with dense vectors + new sparse)
    4. Delete original, recreate with sparse config, copy back from temp
    5. Delete temp collection

    Both paths are idempotent and resumable via checkpoints. The recreation
    path uses a temp collection as a safety buffer — data is never deleted
    until it's fully copied.

    Performance: BM25 via FastEmbed runs at ~1000 texts/sec on CPU.
    Fast path: 10K ≈ 10s, 100K ≈ 2-3 min.
    Recreation path: ~2x slower (two full copies + sparse generation).
    """

    id = "001_add_sparse_vectors"
    description = "Add BM25 sparse vectors for hybrid search"

    def __init__(
        self,
        collection_name: str,
        sparse_client: Any,
    ):
        self._collection_name = collection_name
        self._sparse = sparse_client

    def run(
        self,
        client: QdrantClient,
        *,
        progress: dict[str, Any] | None,
        state_callback: Callable[[dict[str, Any]], None],
        state: dict[str, Any],
    ) -> None:
        from qdrant_client.models import (
            Modifier,
            SparseVectorParams,
        )

        if self._sparse is None or not self._sparse.available:
            logger.warning(
                "Skipping migration %s: fastembed not installed — "
                "hybrid search will use dense-only mode",
                self.id,
            )
            return

        # Check if we're resuming a recreation that was already in progress
        phase = (progress or {}).get("phase")
        if phase in ("copy_to_temp", "copy_back"):
            logger.info(
                "Resuming interrupted collection recreation (phase: %s)",
                phase,
            )
            self._run_recreation(
                client, progress=progress, state_callback=state_callback, state=state
            )
            return

        # Step 1: Try to add sparse vector config via update_collection.
        # Works on QdrantLocal (embedded) and may work on future Qdrant
        # server versions. Remote Qdrant ≤1.16 rejects this with 400.
        try:
            client.update_collection(
                collection_name=self._collection_name,
                sparse_vectors_config={
                    "bm25": SparseVectorParams(modifier=Modifier.IDF),
                },
            )
            logger.info(
                "Added sparse vector config 'bm25' to collection '%s'",
                self._collection_name,
            )
        except Exception:
            logger.debug(
                "update_collection with sparse config failed on '%s' "
                "(expected on remote Qdrant ≤1.16)",
                self._collection_name,
                exc_info=True,
            )

        # Step 2: Verify sparse config actually exists.
        has_sparse = self._has_sparse_config(client)

        if has_sparse:
            # Fast path: sparse config exists, just add sparse vectors
            # to existing points via update_vectors().
            self._run_fast_path(
                client, progress=progress, state_callback=state_callback, state=state
            )
        else:
            # Recreation path: server can't add new vector names to an
            # existing collection. Recreate the collection with sparse
            # config included from the start.
            logger.warning(
                "Cannot add sparse vectors to existing collection '%s' — "
                "Qdrant server does not support adding new named vectors. "
                "Recreating collection with sparse config (data is preserved "
                "via temp collection '%s').",
                self._collection_name,
                _TEMP_COLLECTION,
            )
            self._run_recreation(
                client, progress=progress, state_callback=state_callback, state=state
            )

    def _has_sparse_config(self, client: QdrantClient) -> bool:
        """Check if the collection has 'bm25' sparse vector config."""
        try:
            info = client.get_collection(self._collection_name)
            sparse = info.config.params.sparse_vectors
            if sparse and "bm25" in sparse:
                return True
        except Exception:
            logger.debug(
                "Could not check sparse config on '%s'",
                self._collection_name,
                exc_info=True,
            )
        return False

    # ── Fast path: update_vectors on existing collection ─────────────

    def _run_fast_path(
        self,
        client: QdrantClient,
        *,
        progress: dict[str, Any] | None,
        state_callback: Callable[[dict[str, Any]], None],
        state: dict[str, Any],
    ) -> None:
        """Add sparse vectors to existing points via update_vectors().

        Used when the collection already has 'bm25' sparse vector config
        (either added by update_collection or created with it).
        """
        from qdrant_client.models import PointVectors

        collection_info = client.get_collection(self._collection_name)
        total_points = collection_info.points_count or 0
        if total_points == 0:
            logger.info(
                "Collection '%s' is empty, nothing to migrate",
                self._collection_name,
            )
            return

        estimated_seconds = max(1, total_points / 1000)
        logger.info(
            "Fast path: migrating %d points — generating BM25 sparse "
            "vectors (estimated %.0f seconds)",
            total_points,
            estimated_seconds,
        )

        resume_offset: str | None = None
        processed = 0
        if progress:
            resume_offset = progress.get("offset")
            processed = progress.get("processed", 0)
            if resume_offset:
                logger.info(
                    "Resuming from checkpoint: %d points already processed",
                    processed,
                )

        BATCH_SIZE = 256
        offset = resume_offset
        start_time = time.monotonic()

        while True:
            points, next_offset = client.scroll(
                collection_name=self._collection_name,
                limit=BATCH_SIZE,
                offset=offset,
                with_payload=["data"],
                with_vectors=False,
            )

            if not points:
                break

            texts = [p.payload.get("data", "") if p.payload else "" for p in points]
            sparse_vectors = self._sparse.embed_batch(texts)

            if sparse_vectors:
                point_vectors = [
                    PointVectors(
                        id=p.id,
                        vector={"bm25": sv},
                    )
                    for p, sv in zip(points, sparse_vectors)
                ]
                client.update_vectors(
                    collection_name=self._collection_name,
                    points=point_vectors,
                )

            processed += len(points)
            elapsed = time.monotonic() - start_time
            total_batches = max(1, (total_points + BATCH_SIZE - 1) // BATCH_SIZE)
            batch_num = min(total_batches, (processed + BATCH_SIZE - 1) // BATCH_SIZE)

            logger.info(
                "Migration %s: batch %d/%d (%d/%d points, %.1fs elapsed)",
                self.id,
                batch_num,
                total_batches,
                processed,
                total_points,
                elapsed,
            )

            state[f"{self.id}_progress"] = {
                "offset": str(next_offset) if next_offset else None,
                "processed": processed,
            }
            state_callback(state)

            if next_offset is None:
                break
            offset = next_offset

        elapsed = time.monotonic() - start_time
        logger.info(
            "Migration %s complete: %d points updated in %.1fs",
            self.id,
            processed,
            elapsed,
        )

    # ── Recreation path: copy via temp collection ────────────────────

    def _run_recreation(
        self,
        client: QdrantClient,
        *,
        progress: dict[str, Any] | None,
        state_callback: Callable[[dict[str, Any]], None],
        state: dict[str, Any],
    ) -> None:
        """Recreate the collection with sparse vector config.

        Uses a temp collection as a safety buffer:
        1. Copy original → temp (with sparse vectors)
        2. Delete original, recreate with sparse config
        3. Copy temp → original
        4. Delete temp

        Resumable: checkpoints track phase and offset.
        """
        from qdrant_client.models import (
            Modifier,
            SparseVectorParams,
        )

        phase = (progress or {}).get("phase", "copy_to_temp")
        processed = (progress or {}).get("processed", 0)
        resume_offset = (progress or {}).get("offset")

        # ── Phase 1: Copy original → temp ────────────────────────────

        if phase == "copy_to_temp":
            # Get original collection config for dense vectors
            original_info = client.get_collection(self._collection_name)
            total_points = original_info.points_count or 0

            if total_points == 0:
                logger.info(
                    "Collection '%s' is empty, nothing to recreate",
                    self._collection_name,
                )
                return

            # Read dense vector config from the original collection
            dense_config = self._get_dense_config(original_info)

            # Create temp collection with dense + sparse config
            existing = [c.name for c in client.get_collections().collections]
            if _TEMP_COLLECTION not in existing:
                client.create_collection(
                    collection_name=_TEMP_COLLECTION,
                    vectors_config=dense_config,
                    sparse_vectors_config={
                        "bm25": SparseVectorParams(modifier=Modifier.IDF),
                    },
                )
                logger.info(
                    "Created temp collection '%s' for recreation",
                    _TEMP_COLLECTION,
                )
            else:
                logger.info(
                    "Temp collection '%s' already exists (resuming)",
                    _TEMP_COLLECTION,
                )

            # Copy points from original → temp with sparse vectors
            logger.info(
                "Recreation phase 1: copying %d points from '%s' → '%s'",
                total_points,
                self._collection_name,
                _TEMP_COLLECTION,
            )

            self._copy_with_sparse(
                client,
                source=self._collection_name,
                target=_TEMP_COLLECTION,
                total_points=total_points,
                phase="copy_to_temp",
                resume_offset=resume_offset,
                processed=processed,
                state_callback=state_callback,
                state=state,
            )

            # Verify counts match
            temp_info = client.get_collection(_TEMP_COLLECTION)
            temp_count = temp_info.points_count or 0
            if temp_count < total_points:
                raise RuntimeError(
                    f"Recreation failed: temp collection has {temp_count} "
                    f"points but original has {total_points}. Aborting to "
                    f"prevent data loss."
                )

            logger.info(
                "Phase 1 complete: %d points copied to temp collection",
                temp_count,
            )

            # Transition to phase 2
            processed = 0
            resume_offset = None
            phase = "copy_back"
            state[f"{self.id}_progress"] = {
                "phase": "copy_back",
                "offset": None,
                "processed": 0,
            }
            state_callback(state)

        # ── Phase 2: Delete original, recreate, copy back ────────────

        if phase == "copy_back":
            # Read config from temp collection (it has the correct config)
            temp_info = client.get_collection(_TEMP_COLLECTION)
            total_points = temp_info.points_count or 0
            dense_config = self._get_dense_config(temp_info)

            # Delete original collection
            existing = [c.name for c in client.get_collections().collections]
            if self._collection_name in existing:
                logger.info(
                    "Deleting original collection '%s'",
                    self._collection_name,
                )
                client.delete_collection(self._collection_name)

            # Recreate with sparse config
            logger.info(
                "Recreating collection '%s' with sparse vector config",
                self._collection_name,
            )
            client.create_collection(
                collection_name=self._collection_name,
                vectors_config=dense_config,
                sparse_vectors_config={
                    "bm25": SparseVectorParams(modifier=Modifier.IDF),
                },
            )

            # Copy points from temp → original (already have sparse vectors)
            logger.info(
                "Recreation phase 2: copying %d points from '%s' → '%s'",
                total_points,
                _TEMP_COLLECTION,
                self._collection_name,
            )

            self._copy_points(
                client,
                source=_TEMP_COLLECTION,
                target=self._collection_name,
                total_points=total_points,
                phase="copy_back",
                resume_offset=resume_offset,
                processed=processed,
                state_callback=state_callback,
                state=state,
            )

            # Verify counts match
            final_info = client.get_collection(self._collection_name)
            final_count = final_info.points_count or 0
            if final_count < total_points:
                raise RuntimeError(
                    f"Recreation failed: new collection has {final_count} "
                    f"points but temp has {total_points}. Temp collection "
                    f"'{_TEMP_COLLECTION}' preserved for recovery."
                )

            # Clean up temp collection
            logger.info("Deleting temp collection '%s'", _TEMP_COLLECTION)
            client.delete_collection(_TEMP_COLLECTION)

            logger.info(
                "Migration %s complete: collection '%s' recreated with "
                "sparse vector config (%d points preserved)",
                self.id,
                self._collection_name,
                final_count,
            )

    @staticmethod
    def _get_dense_config(
        collection_info: Any,
    ) -> VectorParams | dict[str, VectorParams]:
        """Extract dense vector config from a collection info object.

        Handles both single (unnamed) vector config and named vector
        configs. Returns the appropriate format for ``create_collection``.
        """
        params = collection_info.config.params
        vectors = params.vectors

        # Single unnamed vector (most common for mnemory)
        if isinstance(vectors, VectorParams):
            return vectors

        # Named vectors — return as dict
        if isinstance(vectors, dict):
            return {
                name: VectorParams(
                    size=cfg.size,
                    distance=cfg.distance,
                )
                for name, cfg in vectors.items()
            }

        # Fallback: try to read size/distance from the object
        try:
            return VectorParams(
                size=vectors.size,
                distance=vectors.distance,
            )
        except AttributeError:
            raise RuntimeError(
                f"Cannot extract dense vector config from collection: {type(vectors)}"
            )

    def _copy_with_sparse(
        self,
        client: QdrantClient,
        *,
        source: str,
        target: str,
        total_points: int,
        phase: str,
        resume_offset: str | None,
        processed: int,
        state_callback: Callable[[dict[str, Any]], None],
        state: dict[str, Any],
    ) -> None:
        """Copy points from source → target, adding sparse vectors.

        Reads dense vectors + payload from source, generates BM25 sparse
        vectors, and upserts into target with both vector types.
        """
        BATCH_SIZE = 256
        offset = resume_offset
        start_time = time.monotonic()

        while True:
            points, next_offset = client.scroll(
                collection_name=source,
                limit=BATCH_SIZE,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )

            if not points:
                break

            # Generate sparse vectors from memory text
            texts = [p.payload.get("data", "") if p.payload else "" for p in points]
            sparse_vectors = self._sparse.embed_batch(texts)

            # Build points with both dense + sparse vectors
            new_points = []
            for i, p in enumerate(points):
                # Get the dense vector (unnamed or named)
                dense_vec = p.vector
                if isinstance(dense_vec, dict):
                    # Named vectors — keep as-is, add sparse
                    vectors = dict(dense_vec)
                else:
                    # Unnamed vector — use empty string key (Qdrant default)
                    vectors = {"": dense_vec}

                if sparse_vectors and i < len(sparse_vectors):
                    vectors["bm25"] = sparse_vectors[i]

                new_points.append(
                    PointStruct(
                        id=p.id,
                        vector=vectors,
                        payload=p.payload or {},
                    )
                )

            if new_points:
                client.upsert(
                    collection_name=target,
                    points=new_points,
                )

            processed += len(points)
            elapsed = time.monotonic() - start_time
            total_batches = max(1, (total_points + BATCH_SIZE - 1) // BATCH_SIZE)
            batch_num = min(total_batches, (processed + BATCH_SIZE - 1) // BATCH_SIZE)

            logger.info(
                "Migration %s [%s]: batch %d/%d (%d/%d points, %.1fs)",
                self.id,
                phase,
                batch_num,
                total_batches,
                processed,
                total_points,
                elapsed,
            )

            state[f"{self.id}_progress"] = {
                "phase": phase,
                "offset": str(next_offset) if next_offset else None,
                "processed": processed,
            }
            state_callback(state)

            if next_offset is None:
                break
            offset = next_offset

    def _copy_points(
        self,
        client: QdrantClient,
        *,
        source: str,
        target: str,
        total_points: int,
        phase: str,
        resume_offset: str | None,
        processed: int,
        state_callback: Callable[[dict[str, Any]], None],
        state: dict[str, Any],
    ) -> None:
        """Copy points from source → target (no sparse generation).

        Used for the copy-back phase where points already have sparse
        vectors from the copy-to-temp phase.
        """
        BATCH_SIZE = 256
        offset = resume_offset
        start_time = time.monotonic()

        while True:
            points, next_offset = client.scroll(
                collection_name=source,
                limit=BATCH_SIZE,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )

            if not points:
                break

            new_points = []
            for p in points:
                dense_vec = p.vector
                if isinstance(dense_vec, dict):
                    vectors = dict(dense_vec)
                else:
                    vectors = {"": dense_vec}

                new_points.append(
                    PointStruct(
                        id=p.id,
                        vector=vectors,
                        payload=p.payload or {},
                    )
                )

            if new_points:
                client.upsert(
                    collection_name=target,
                    points=new_points,
                )

            processed += len(points)
            elapsed = time.monotonic() - start_time
            total_batches = max(1, (total_points + BATCH_SIZE - 1) // BATCH_SIZE)
            batch_num = min(total_batches, (processed + BATCH_SIZE - 1) // BATCH_SIZE)

            logger.info(
                "Migration %s [%s]: batch %d/%d (%d/%d points, %.1fs)",
                self.id,
                phase,
                batch_num,
                total_batches,
                processed,
                total_points,
                elapsed,
            )

            state[f"{self.id}_progress"] = {
                "phase": phase,
                "offset": str(next_offset) if next_offset else None,
                "processed": processed,
            }
            state_callback(state)

            if next_offset is None:
                break
            offset = next_offset


# ── Migration 002: Create _mnemory_sessions collection ───────────────

SESSIONS_COLLECTION = "_mnemory_sessions"


class CreateSessionsCollectionMigration(Migration):
    """Create the _mnemory_sessions collection for persistent session summaries.

    Used by the two-layer memory system to persist conversation summaries
    beyond session expiry. The consolidation service uses these summaries
    alongside raw memories to synthesize durable knowledge.

    Uses a 1-dim dummy vector (like _mnemory_meta) since this collection
    stores metadata, not searchable vectors. A real summary embedding
    may be added later by the consolidation service.
    """

    id = "002_create_sessions_collection"
    description = "Create _mnemory_sessions collection for persistent session summaries"

    def run(
        self,
        client: QdrantClient,
        *,
        progress: dict[str, Any] | None,
        state_callback: Callable[[dict[str, Any]], None],
        state: dict[str, Any],
    ) -> None:
        existing = [c.name for c in client.get_collections().collections]
        if SESSIONS_COLLECTION in existing:
            logger.info("Collection '%s' already exists, skipping", SESSIONS_COLLECTION)
            return

        client.create_collection(
            collection_name=SESSIONS_COLLECTION,
            vectors_config=VectorParams(size=1, distance=Distance.COSINE),
        )
        logger.info("Created collection '%s'", SESSIONS_COLLECTION)

        # Create payload indexes for efficient filtering
        # user_id: required for all queries (isolation)
        # consolidation_state: for finding pending sessions
        for field in ("user_id", "consolidation_state"):
            try:
                client.create_payload_index(
                    collection_name=SESSIONS_COLLECTION,
                    field_name=field,
                    field_schema="keyword",
                )
            except Exception:
                pass  # Index may already exist


# ── Migration 003: Add memory_layer index ────────────────────────────


class AddMemoryLayerIndexMigration(Migration):
    """Add memory_layer keyword index to the main memory collection.

    Supports filtering by memory layer (raw vs consolidated) in search
    and dedup operations. The index is created idempotently.
    """

    id = "003_add_memory_layer_index"
    description = "Add memory_layer keyword index for two-layer memory system"

    def __init__(self, collection_name: str):
        self._collection_name = collection_name

    def run(
        self,
        client: QdrantClient,
        *,
        progress: dict[str, Any] | None,
        state_callback: Callable[[dict[str, Any]], None],
        state: dict[str, Any],
    ) -> None:
        # Create keyword index for memory_layer field
        try:
            client.create_payload_index(
                collection_name=self._collection_name,
                field_name="memory_layer",
                field_schema="keyword",
            )
            logger.info("Created memory_layer index on '%s'", self._collection_name)
        except Exception:
            logger.debug(
                "memory_layer index may already exist on '%s'", self._collection_name
            )


# ── Migration 004: Backfill owner_id for legacy memories ───────────────────


class BackfillOwnerIdMigration(Migration):
    """Backfill `owner_id = user_id` for legacy memories missing owner_id."""

    id = "004_backfill_owner_id"
    description = (
        "Backfill owner_id on memories created before shared-agent owner scope"
    )

    def __init__(self, collection_name: str):
        self._collection_name = collection_name

    def run(
        self,
        client: QdrantClient,
        *,
        progress: dict[str, Any] | None,
        state_callback: Callable[[dict[str, Any]], None],
        state: dict[str, Any],
    ) -> None:
        from qdrant_client.models import Filter, IsEmptyCondition, PayloadField

        batch_size = 256
        offset = (progress or {}).get("offset")
        processed = int((progress or {}).get("processed", 0) or 0)
        updated = int((progress or {}).get("updated", 0) or 0)
        start_time = time.monotonic()

        while True:
            points, next_offset = client.scroll(
                collection_name=self._collection_name,
                scroll_filter=Filter(
                    must=[IsEmptyCondition(is_empty=PayloadField(key="owner_id"))]
                ),
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if not points:
                break

            for point in points:
                payload = dict(point.payload or {})
                user_id = payload.get("user_id")
                if not isinstance(user_id, str) or not user_id.strip():
                    continue
                client.set_payload(
                    collection_name=self._collection_name,
                    payload={"owner_id": user_id},
                    points=[point.id],
                )
                updated += 1

            processed += len(points)
            elapsed = time.monotonic() - start_time
            logger.info(
                "Migration %s: processed=%d updated=%d elapsed=%.1fs",
                self.id,
                processed,
                updated,
                elapsed,
            )
            state[f"{self.id}_progress"] = {
                "offset": str(next_offset) if next_offset else None,
                "processed": processed,
                "updated": updated,
            }
            state_callback(state)
            if next_offset is None:
                break
            offset = next_offset


# ── Migration 005: Immutable memory revisions and operation journal ──────────


class AddRevisionFoundationMigration(Migration):
    """Create revision indexes and backfill legacy revision metadata."""

    id = "005_add_revision_foundation"
    description = (
        "Create revision operation journal, indexes, and legacy revision metadata"
    )

    def __init__(self, collection_name: str):
        self._collection_name = collection_name

    def run(
        self,
        client: QdrantClient,
        *,
        progress: dict[str, Any] | None,
        state_callback: Callable[[dict[str, Any]], None],
        state: dict[str, Any],
    ) -> None:
        from qdrant_client.models import (
            Distance,
            Filter,
            IsEmptyCondition,
            PayloadField,
            VectorParams,
        )

        from mnemory.revisions import OPERATIONS_COLLECTION

        existing = {
            collection.name for collection in client.get_collections().collections
        }
        if OPERATIONS_COLLECTION not in existing:
            client.create_collection(
                collection_name=OPERATIONS_COLLECTION,
                vectors_config=VectorParams(size=1, distance=Distance.COSINE),
            )

        main_indexes = {
            "lineage_id": "keyword",
            "revision": "integer",
            "revision_state": "keyword",
            "supersedes": "keyword",
            "derived_from": "keyword",
            "source_session_id": "keyword",
            "transition_token": "keyword",
            "revision_operation_key_hash": "keyword",
        }
        operation_indexes = {
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
            "created_at_utc": "datetime",
        }
        session_indexes = {
            "owner_id": "keyword",
            "session_revision": "integer",
            "consolidation_token": "keyword",
        }
        for collection_name, indexes in (
            (self._collection_name, main_indexes),
            (OPERATIONS_COLLECTION, operation_indexes),
            (SESSIONS_COLLECTION, session_indexes),
        ):
            if (
                collection_name not in existing
                and collection_name != OPERATIONS_COLLECTION
            ):
                continue
            for field_name, field_schema in indexes.items():
                try:
                    client.create_payload_index(
                        collection_name=collection_name,
                        field_name=field_name,
                        field_schema=field_schema,
                    )
                except Exception:
                    logger.debug(
                        "Revision index %s already exists on %s",
                        field_name,
                        collection_name,
                    )

        if SESSIONS_COLLECTION in existing:
            session_offset = None
            while True:
                sessions, next_session_offset = client.scroll(
                    collection_name=SESSIONS_COLLECTION,
                    scroll_filter=Filter(
                        must=[IsEmptyCondition(is_empty=PayloadField(key="owner_id"))]
                    ),
                    limit=256,
                    offset=session_offset,
                    with_payload=["user_id", "session_revision"],
                    with_vectors=False,
                )
                for session in sessions:
                    session_payload = dict(session.payload or {})
                    session_user_id = session_payload.get("user_id")
                    if not session_user_id:
                        continue
                    client.set_payload(
                        collection_name=SESSIONS_COLLECTION,
                        payload={
                            "owner_id": session_user_id,
                            "session_revision": int(
                                session_payload.get("session_revision", 1)
                            ),
                        },
                        points=[session.id],
                        wait=True,
                    )
                if next_session_offset is None:
                    break
                session_offset = next_session_offset

        offset = (progress or {}).get("offset")
        processed = int((progress or {}).get("processed", 0) or 0)
        updated = int((progress or {}).get("updated", 0) or 0)
        warnings = int((progress or {}).get("warnings", 0) or 0)
        start_time = time.monotonic()

        while True:
            points, next_offset = client.scroll(
                collection_name=self._collection_name,
                scroll_filter=Filter(
                    must=[IsEmptyCondition(is_empty=PayloadField(key="revision_state"))]
                ),
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if not points:
                break

            for point in points:
                payload = dict(point.payload or {})
                point_id = str(point.id)
                revision_state = "active"
                provenance_quality = "exact"
                provenance_warning = None

                legacy_target_id = payload.get("superseded_by")
                if (
                    legacy_target_id
                    and payload.get("memory_layer", "consolidated") == "raw"
                ):
                    target_result = client.retrieve(
                        collection_name=self._collection_name,
                        ids=[legacy_target_id],
                        with_payload=True,
                        with_vectors=False,
                    )
                    target_payload = (
                        dict(target_result[0].payload or {}) if target_result else {}
                    )
                    same_owner = target_payload.get("owner_id") == payload.get(
                        "owner_id"
                    ) and target_payload.get("user_id") == payload.get("user_id")
                    has_reverse_link = point_id in (
                        target_payload.get("derived_from") or []
                    )
                    if same_owner and has_reverse_link:
                        revision_state = "source"
                        provenance_quality = "legacy_batch"
                    else:
                        provenance_quality = "legacy_batch"
                        provenance_warning = "invalid_legacy_superseded_by"
                        warnings += 1

                revision_metadata: dict[str, Any] = {
                    "lineage_id": point_id,
                    "revision": 1,
                    "revision_state": revision_state,
                    "revision_created_at_utc": (
                        payload.get("created_at_utc")
                        or payload.get("created_at")
                        or datetime.now(timezone.utc).isoformat()
                    ),
                    "validation_state": "unverified",
                    "provenance_quality": provenance_quality,
                }
                if provenance_warning:
                    revision_metadata["provenance_warning"] = provenance_warning
                client.set_payload(
                    collection_name=self._collection_name,
                    payload=revision_metadata,
                    points=[point.id],
                    wait=True,
                )
                updated += 1

            processed += len(points)
            logger.info(
                "Migration %s: processed=%d updated=%d warnings=%d elapsed=%.1fs",
                self.id,
                processed,
                updated,
                warnings,
                time.monotonic() - start_time,
            )
            state[f"{self.id}_progress"] = {
                "offset": str(next_offset) if next_offset else None,
                "processed": processed,
                "updated": updated,
                "warnings": warnings,
            }
            state_callback(state)
            if next_offset is None:
                break
            offset = next_offset


class AddValidationProjectionMigration(Migration):
    """Add validation indexes and neutral projections for legacy memories."""

    id = "006_add_validation_projection"
    description = "Add evidence validation projection fields and indexes"

    def __init__(self, collection_name: str):
        self._collection_name = collection_name

    def run(
        self,
        client: QdrantClient,
        *,
        progress: dict[str, Any] | None,
        state_callback: Callable[[dict[str, Any]], None],
        state: dict[str, Any],
    ) -> None:
        from qdrant_client.models import Filter, IsEmptyCondition, PayloadField

        schemas = {
            "source_kind": "keyword",
            "validation_eligible": "bool",
            "evidence_root_ids": "keyword",
            "expires_at": "datetime",
        }
        for field, schema in schemas.items():
            try:
                client.create_payload_index(
                    collection_name=self._collection_name,
                    field_name=field,
                    field_schema=schema,
                )
            except Exception:
                logger.debug("Validation index %s already exists", field)

        offset = (progress or {}).get("offset")
        processed = int((progress or {}).get("processed", 0) or 0)
        while True:
            points, next_offset = client.scroll(
                collection_name=self._collection_name,
                scroll_filter=Filter(
                    must=[
                        IsEmptyCondition(is_empty=PayloadField(key="validation_count"))
                    ]
                ),
                limit=256,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            if not points:
                break
            client.set_payload(
                collection_name=self._collection_name,
                payload={
                    "source_kind": "legacy_unknown",
                    "validation_eligible": False,
                    "evidence_root_ids": [],
                    "validation_count": 0,
                    "validation_strength": 0.0,
                },
                points=[point.id for point in points],
                wait=True,
            )
            processed += len(points)
            state[f"{self.id}_progress"] = {
                "offset": str(next_offset) if next_offset else None,
                "processed": processed,
            }
            state_callback(state)
            if next_offset is None:
                break
            offset = next_offset


class ClassifyLegacyFailedSessionsMigration(Migration):
    """Classify legacy failed sessions without retrying or changing state."""

    id = "007_classify_legacy_failed_sessions"
    description = "Classify failed sessions that predate failure diagnostics"

    def run(
        self,
        client: QdrantClient,
        *,
        progress: dict[str, Any] | None,
        state_callback: Callable[[dict[str, Any]], None],
        state: dict[str, Any],
    ) -> None:
        from qdrant_client.models import (
            FieldCondition,
            Filter,
            IsEmptyCondition,
            MatchValue,
            PayloadField,
        )

        collection = "_mnemory_sessions"
        for field, schema in {
            "legacy_failure": "bool",
            "failure_class": "keyword",
            "last_error_at": "datetime",
            "retry_operation_token": "keyword",
            "retry_claimant": "keyword",
            "retry_lease_expires_at": "datetime",
        }.items():
            try:
                client.create_payload_index(
                    collection_name=collection,
                    field_name=field,
                    field_schema=schema,
                )
            except Exception:
                logger.debug("Session failure index %s already exists", field)

        offset = (progress or {}).get("offset")
        processed = int((progress or {}).get("processed", 0) or 0)
        while True:
            points, next_offset = client.scroll(
                collection_name=collection,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="consolidation_state",
                            match=MatchValue(value="failed"),
                        ),
                        IsEmptyCondition(
                            is_empty=PayloadField(key="failure_schema_version")
                        ),
                    ]
                ),
                limit=256,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            if not points:
                break
            client.set_payload(
                collection_name=collection,
                payload={
                    "failure_schema_version": 1,
                    "legacy_failure": True,
                    "failure_class": "legacy_unknown",
                },
                points=[point.id for point in points],
                wait=True,
            )
            processed += len(points)
            state[f"{self.id}_progress"] = {
                "offset": str(next_offset) if next_offset else None,
                "processed": processed,
            }
            state_callback(state)
            if next_offset is None:
                break
            offset = next_offset


# ── Migration registry ───────────────────────────────────────────────


def get_migrations(
    collection_name: str,
    sparse_client: Any,
) -> list[Migration]:
    """Return the ordered list of all migrations.

    New migrations should be appended to this list. Never reorder or
    remove existing migrations — they are identified by their ``id``.
    """
    return [
        AddSparseVectorsMigration(collection_name, sparse_client),
        CreateSessionsCollectionMigration(),
        AddMemoryLayerIndexMigration(collection_name),
        BackfillOwnerIdMigration(collection_name),
        AddRevisionFoundationMigration(collection_name),
        AddValidationProjectionMigration(collection_name),
        ClassifyLegacyFailedSessionsMigration(),
    ]
