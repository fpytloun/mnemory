"""Vector store using Qdrant directly.

Handles all vector operations: insert, search, update, delete, scroll.
Supports both remote Qdrant (production) and local/embedded Qdrant
(development) via qdrant-client's path mode.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    IsEmptyCondition,
    MatchAny,
    MatchValue,
    OrderBy,
    PayloadField,
    PointIdsList,
    PointStruct,
    VectorParams,
)

from mnemory.config import Config
from mnemory.embeddings import EmbeddingClient

logger = logging.getLogger(__name__)

# Use SHA-256 for content hashing (dedup, not security-critical,
# but avoids MD5 deprecation warnings from security scanners).
_hash = hashlib.sha256


def _build_labels_conditions(labels_filter: dict) -> list[FieldCondition]:
    """Build Qdrant filter conditions from a labels dict.

    Each key-value pair becomes a must condition on ``labels.<key>``.
    List values use MatchAny (any-of), scalar values use MatchValue (exact).

    Args:
        labels_filter: Dict of label key-value pairs to filter by.

    Returns:
        List of FieldCondition objects to append to a must clause.
    """
    conditions: list[FieldCondition] = []
    for key, value in labels_filter.items():
        if isinstance(value, list):
            conditions.append(
                FieldCondition(key=f"labels.{key}", match=MatchAny(any=value))
            )
        else:
            conditions.append(
                FieldCondition(key=f"labels.{key}", match=MatchValue(value=value))
            )
    return conditions


def _build_owner_scope_condition(
    user_id: str, owner_id: str | None
) -> FieldCondition | Filter:
    """Build a filter condition for owner-aware backward-compatible queries.

    When `owner_id` is set, legacy memories that predate the owner dimension
    may still have no `owner_id` payload. Those records should still be visible
    when their `user_id` matches the requested owner.
    """

    if not owner_id:
        return FieldCondition(key="user_id", match=MatchValue(value=user_id))
    return Filter(
        should=[
            FieldCondition(key="owner_id", match=MatchValue(value=owner_id)),
            Filter(
                must=[
                    IsEmptyCondition(is_empty=PayloadField(key="owner_id")),
                    FieldCondition(key="user_id", match=MatchValue(value=owner_id)),
                ]
            ),
        ]
    )


@contextmanager
def _qdrant_timer(operation: str):
    """Context manager that logs and records Qdrant operation timing."""
    t0 = time.monotonic()
    yield
    duration = time.monotonic() - t0
    logger.debug(
        "Qdrant: operation=%s duration_ms=%d",
        operation,
        int(duration * 1000),
    )
    from mnemory.metrics import get_collector

    collector = get_collector()
    if collector:
        collector.observe_qdrant_duration(operation, duration)


class VectorStore:
    """Direct Qdrant vector store for memory operations.

    Supports two modes:
    - Remote: connects to a Qdrant server (QDRANT_HOST set)
    - Local: embedded Qdrant via qdrant-client path mode (no server needed)
    """

    def __init__(self, config: Config):
        self._config = config
        self._embedding = EmbeddingClient(config.embed)
        self._client = self._create_client(config)
        self._ensure_collection()

        # Local embedded Qdrant uses SQLite which isn't thread-safe for
        # concurrent writes. A lock serializes write operations in local
        # mode. Remote Qdrant handles concurrency server-side, so no
        # lock is needed.
        self._write_lock: threading.Lock | None = (
            threading.Lock() if not config.vector.is_remote else None
        )

    @contextmanager
    def _write_guard(self):
        """Acquire write lock for local embedded mode (no-op for remote)."""
        if self._write_lock is not None:
            with self._write_lock:
                yield
        else:
            yield

    @staticmethod
    def _create_client(config: Config) -> QdrantClient:
        """Create a Qdrant client based on configuration."""
        vc = config.vector
        if vc.is_remote:
            kwargs: dict[str, Any] = {
                "host": vc.qdrant_host,
                "port": vc.qdrant_port,
            }
            if vc.qdrant_api_key:
                kwargs["api_key"] = vc.qdrant_api_key
            logger.info(
                "Connecting to remote Qdrant at %s:%d", vc.qdrant_host, vc.qdrant_port
            )
            return QdrantClient(**kwargs)
        else:
            logger.info("Using local Qdrant at %s", vc.qdrant_path)
            return QdrantClient(path=vc.qdrant_path)

    def _ensure_collection(self) -> None:
        """Create the collection if it doesn't exist, then ensure indexes.

        New collections are created with both dense vector config and
        sparse vector config (BM25) for hybrid search. Existing
        collections get sparse config added via the migration framework.
        """
        name = self.collection_name
        existing = [c.name for c in self._client.get_collections().collections]
        if name not in existing:
            from qdrant_client.models import (
                Modifier,
                SparseVectorParams,
            )

            logger.info(
                "Creating collection '%s' (dims=%d, distance=COSINE)",
                name,
                self._embedding.dims,
            )
            self._client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=self._embedding.dims,
                    distance=Distance.COSINE,
                ),
                sparse_vectors_config={
                    "bm25": SparseVectorParams(modifier=Modifier.IDF),
                },
            )
        else:
            logger.debug("Collection '%s' already exists", name)

        # Ensure payload indexes exist (idempotent — safe to run every startup)
        self._ensure_indexes(
            labels_indexes=self._config.memory.labels_indexes or None,
        )

    def _ensure_indexes(self, labels_indexes: list[str] | None = None) -> None:
        """Create payload indexes for efficient filtering and ordering.

        Runs on every startup (not just collection creation) to ensure
        indexes exist for existing collections that were created before
        new indexes were added. Qdrant's create_payload_index is
        idempotent — it no-ops if the index already exists.

        Only applies to remote Qdrant; local embedded mode doesn't
        need explicit indexes.

        Args:
            labels_indexes: Optional list of label field names to create
                keyword indexes for. Each name gets indexed as
                ``labels.<field_name>`` for efficient label filtering.
        """
        if not self._config.vector.is_remote:
            return

        name = self.collection_name
        # Keyword/bool indexes for filtering
        for field in ("user_id", "owner_id", "agent_id", "pinned", "memory_type"):
            try:
                self._client.create_payload_index(
                    collection_name=name,
                    field_name=field,
                    field_schema="keyword" if field != "pinned" else "bool",
                )
            except Exception:
                pass  # Index may already exist
        # Datetime indexes for DatetimeRange filters and order_by
        for field in ("created_at", "expires_at"):
            try:
                self._client.create_payload_index(
                    collection_name=name,
                    field_name=field,
                    field_schema="datetime",
                )
            except Exception:
                pass  # Index may already exist
        # Keyword index for event_date (stored as YYYY-MM-DD string,
        # supports lexicographic Range filtering for date queries)
        try:
            self._client.create_payload_index(
                collection_name=name,
                field_name="event_date",
                field_schema="keyword",
            )
        except Exception:
            pass  # Index may already exist
        # Keyword indexes for configured label fields
        for field_name in labels_indexes or []:
            try:
                self._client.create_payload_index(
                    collection_name=name,
                    field_name=f"labels.{field_name}",
                    field_schema="keyword",
                )
            except Exception:
                pass  # Index may already exist

    @property
    def collection_name(self) -> str:
        return self._config.vector.collection_name

    @property
    def embedding(self) -> EmbeddingClient:
        """Access the embedding client for external use."""
        return self._embedding

    # ── Core operations ──────────────────────────────────────────────

    def insert(
        self,
        *,
        text: str,
        vector: list[float],
        user_id: str,
        owner_id: str | None = None,
        agent_id: str | None = None,
        metadata: dict[str, Any],
        role: str = "user",
        sparse_vector: Any = None,
    ) -> str:
        """Insert a new memory point.

        Args:
            text: The memory content text.
            vector: Pre-computed dense embedding vector.
            user_id: User scope.
            agent_id: Optional agent scope.
            metadata: Custom metadata fields (memory_type, categories, etc.).
            role: "user" or "assistant".
            sparse_vector: Optional BM25 sparse vector for hybrid search.
                When provided, stored alongside the dense vector as a
                named vector ``{"": dense, "bm25": sparse}``.

        Returns:
            The generated memory ID (UUID string).
        """
        memory_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        payload: dict[str, Any] = {
            "data": text,
            "hash": _hash(text.encode()).hexdigest(),
            "created_at": now.isoformat(),
            "user_id": user_id,
            "owner_id": owner_id or user_id,
            "role": role,
        }
        if agent_id:
            payload["agent_id"] = agent_id

        # Merge custom metadata (memory_type, categories, importance, etc.)
        payload.update(metadata)

        # Use named vectors when sparse is available, unnamed otherwise
        point_vector: Any = (
            {"": vector, "bm25": sparse_vector} if sparse_vector is not None else vector
        )

        with self._write_guard():
            with _qdrant_timer("insert"):
                self._client.upsert(
                    collection_name=self.collection_name,
                    points=[
                        PointStruct(id=memory_id, vector=point_vector, payload=payload)
                    ],
                )
        return memory_id

    def insert_batch(
        self,
        points: list[dict[str, Any]],
    ) -> list[str]:
        """Insert multiple memory points in a single call.

        Each point dict should have: text, vector, user_id, owner_id (optional), agent_id (optional),
        metadata, role, sparse_vector (optional).

        Returns list of generated memory IDs.
        """
        if not points:
            return []

        now = datetime.now(timezone.utc)
        structs = []
        ids = []

        for p in points:
            memory_id = str(uuid.uuid4())
            ids.append(memory_id)

            payload: dict[str, Any] = {
                "data": p["text"],
                "hash": _hash(p["text"].encode()).hexdigest(),
                "created_at": now.isoformat(),
                "user_id": p["user_id"],
                "owner_id": p.get("owner_id") or p["user_id"],
                "role": p.get("role", "user"),
            }
            if p.get("agent_id"):
                payload["agent_id"] = p["agent_id"]
            payload.update(p.get("metadata", {}))

            # Use named vectors when sparse is available
            sparse_vec = p.get("sparse_vector")
            point_vector: Any = (
                {"": p["vector"], "bm25": sparse_vec}
                if sparse_vec is not None
                else p["vector"]
            )

            structs.append(
                PointStruct(id=memory_id, vector=point_vector, payload=payload)
            )

        with self._write_guard():
            with _qdrant_timer("insert_batch"):
                self._client.upsert(
                    collection_name=self.collection_name,
                    points=structs,
                )
        return ids

    def search(
        self,
        query: str,
        *,
        user_id: str,
        owner_id: str | None = None,
        subject_user_id: str | None = None,
        agent_id: str | None = None,
        shared_only: bool = False,
        filters: dict | None = None,
        categories: list[str] | None = None,
        labels_filter: dict | None = None,
        limit: int = 100,
        exclude_expired: bool = False,
        include_decayed: bool = False,
        similarity_weight: float = 0.9,
        query_vector: list[float] | None = None,
        query_sparse_vector: Any = None,
        date_start: str | None = None,
        date_end: str | None = None,
    ) -> dict:
        """Semantic search with optional hybrid (dense + sparse) retrieval.

        When ``query_sparse_vector`` is provided, uses Qdrant's native
        hybrid search: two prefetch branches (dense + BM25 sparse) fused
        via Reciprocal Rank Fusion (RRF). This surfaces results that
        match by keyword even if they're not semantically similar.

        When ``query_sparse_vector`` is None, falls back to dense-only
        search with FormulaQuery importance reranking (original behavior).

        Note: In hybrid mode, importance reranking is NOT applied
        server-side (FormulaQuery cannot be reliably nested inside
        Prefetch — see https://github.com/qdrant/qdrant/issues/6836).
        The caller (memory.py) applies importance boost in Python
        post-processing after RRF fusion.

        Args:
            query: Search query text.
            user_id: Required user scope.
            agent_id: Optional agent scope.
            shared_only: If True and agent_id is None, restrict to memories
                without any agent_id (shared user memories only). Used by
                dual-scope search to avoid leaking sub-agent memories.
            filters: Simple key-value metadata filters (memory_type, role).
            categories: Category filter (OR logic via MatchAny).
            limit: Maximum results.
            exclude_expired: If True, filter out expired/decayed memories.
            include_decayed: If True, include decayed memories.
            similarity_weight: Weight for cosine similarity in the combined
                score formula (0.0-1.0). Importance gets 1 - similarity_weight.
                Default 0.9 (90% similarity, 10% importance).
                Only used in dense-only mode (FormulaQuery).
            query_vector: Pre-computed dense embedding vector. If provided,
                skips the embedding API call.
            query_sparse_vector: Pre-computed BM25 sparse vector for hybrid
                search. When provided, enables RRF fusion of dense + sparse.
            date_start: Optional start date (YYYY-MM-DD) for temporal
                filtering.
            date_end: Optional end date (YYYY-MM-DD) for temporal filtering.

        Returns:
            Dict with "results" key containing list of memory dicts.
            Each result has a "score" field. In hybrid mode, scores are
            RRF-fused (range depends on the server's k constant — with
            Qdrant's default k=1, scores are ~0.1-1.0; with k=60, scores
            are ~0.01-0.03). In dense-only mode, scores are cosine
            similarity weighted by importance (range 0-1).
        """
        from qdrant_client.models import DatetimeRange, Fusion, FusionQuery, Prefetch

        from mnemory.categories import IMPORTANCE_WEIGHTS

        # 1. Embed the query (skip if pre-computed vector provided)
        embeddings = (
            query_vector if query_vector is not None else self._embedding.embed(query)
        )

        # 2. Build filter conditions
        must_conditions: list = [_build_owner_scope_condition(user_id, owner_id)]
        if subject_user_id:
            must_conditions.append(
                FieldCondition(key="user_id", match=MatchValue(value=subject_user_id))
            )
        if agent_id:
            must_conditions.append(
                FieldCondition(key="agent_id", match=MatchValue(value=agent_id))
            )
        elif shared_only:
            must_conditions.append(
                IsEmptyCondition(is_empty=PayloadField(key="agent_id"))
            )
        if filters:
            for key, value in filters.items():
                must_conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
        if categories:
            must_conditions.append(
                FieldCondition(key="categories", match=MatchAny(any=categories))
            )
        if labels_filter:
            must_conditions.extend(_build_labels_conditions(labels_filter))
        if exclude_expired and not include_decayed:
            now = datetime.now(timezone.utc)
            ttl_filter = Filter(
                should=[
                    IsEmptyCondition(is_empty=PayloadField(key="expires_at")),
                    FieldCondition(key="expires_at", range=DatetimeRange(gt=now)),
                    FieldCondition(key="pinned", match=MatchValue(value=True)),
                ]
            )
            must_conditions.append(ttl_filter)

        if date_start or date_end:
            date_conditions = self._build_date_range_filter(date_start, date_end)
            if date_conditions:
                must_conditions.append(date_conditions)

        query_filter = Filter(must=must_conditions)

        # 3. Execute search — hybrid or dense-only
        t0 = time.monotonic()
        result = None
        used_hybrid = False
        search_mode = "dense-simple"  # track granular mode for diagnostics

        if query_sparse_vector is not None:
            # Hybrid search: dense + sparse → RRF fusion.
            # Importance reranking is handled in Python post-processing
            # by the caller (memory.py), not here, because FormulaQuery
            # cannot be reliably nested inside Prefetch.
            try:
                result = self._client.query_points(
                    collection_name=self.collection_name,
                    prefetch=[
                        Prefetch(
                            query=embeddings,
                            filter=query_filter,
                            limit=limit,
                        ),
                        Prefetch(
                            query=query_sparse_vector,
                            using="bm25",
                            filter=query_filter,
                            limit=limit,
                        ),
                    ],
                    query=FusionQuery(fusion=Fusion.RRF),
                    limit=limit,
                    with_payload=True,
                )
                used_hybrid = True
                search_mode = "hybrid"
            except Exception:
                logger.warning(
                    "Hybrid search failed (sparse vector was available), "
                    "falling back to dense-only search",
                    exc_info=True,
                )
                # Fall through to dense-only path
        else:
            logger.warning(
                "Sparse vector not available for search — using dense-only mode. "
                "Check that fastembed is installed and the BM25 model is loaded."
            )

        if result is None:
            # Dense-only path with FormulaQuery importance reranking
            importance_weight = 1.0 - similarity_weight
            try:
                from qdrant_client.models import (
                    FormulaQuery,
                    SumExpression,
                )

                formula_terms: list = [
                    {"mult": [similarity_weight, "$score"]},
                ]
                for imp_level, imp_weight in IMPORTANCE_WEIGHTS.items():
                    boost = importance_weight * imp_weight
                    if boost > 0:
                        formula_terms.append(
                            {
                                "mult": [
                                    boost,
                                    {
                                        "key": "importance",
                                        "match": {"value": imp_level},
                                    },
                                ]
                            }
                        )

                result = self._client.query_points(
                    collection_name=self.collection_name,
                    prefetch=Prefetch(
                        query=embeddings,
                        filter=query_filter,
                        limit=limit,
                    ),
                    query=FormulaQuery(formula=SumExpression(sum=formula_terms)),
                    limit=limit,
                    with_payload=True,
                )
                search_mode = "dense-formula"
            except Exception:
                logger.warning(
                    "Formula-based reranking failed, falling back to simple search"
                )
                result = self._client.query_points(
                    collection_name=self.collection_name,
                    query=embeddings,
                    query_filter=query_filter,
                    limit=limit,
                    with_payload=True,
                )
                search_mode = "dense-simple"

        # 4. Record search timing
        search_duration = time.monotonic() - t0
        logger.info(
            "Search: mode=%s duration_ms=%d results=%d",
            search_mode,
            int(search_duration * 1000),
            len(result.points),
        )
        from mnemory.metrics import get_collector

        collector = get_collector()
        if collector:
            collector.observe_qdrant_duration("search", search_duration)

        # 5. Convert to memory dicts
        memories = []
        for point in result.points:
            mem = self._point_to_memory(point)
            mem["score"] = point.score
            memories.append(mem)

        return {
            "results": memories,
            "used_hybrid": used_hybrid,
            "search_mode": search_mode,
        }

    def search_similar(
        self,
        vector: list[float],
        *,
        user_id: str,
        owner_id: str | None = None,
        subject_user_id: str | None = None,
        agent_id: str | None = None,
        shared_only: bool = False,
        limit: int = 5,
        exclude_layers: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar existing memories using a pre-computed vector.

        Used during the add pipeline to find candidates for deduplication.
        Excludes expired/decayed memories so the LLM doesn't deduplicate
        against memories the user can no longer see.

        Args:
            vector: Pre-computed embedding vector.
            user_id: Required user scope.
            agent_id: Optional agent scope.
            shared_only: If True and agent_id is None, restrict to memories
                without any agent_id (shared user memories only). Used by
                dual-scope dedup to search shared memories separately.
            limit: Maximum results.
            exclude_layers: Optional list of memory_layer values to exclude
                from results (e.g., ["consolidated"] to exclude consolidated
                memories during remember dedup).

        Returns simple dicts with "id" and "text" keys.
        """
        from qdrant_client.models import DatetimeRange

        must_conditions: list = [_build_owner_scope_condition(user_id, owner_id)]
        if subject_user_id:
            must_conditions.append(
                FieldCondition(key="user_id", match=MatchValue(value=subject_user_id))
            )
        if agent_id:
            must_conditions.append(
                FieldCondition(key="agent_id", match=MatchValue(value=agent_id))
            )
        elif shared_only:
            must_conditions.append(
                IsEmptyCondition(is_empty=PayloadField(key="agent_id"))
            )

        # Exclude expired/decayed memories from dedup candidates.
        # Use IsEmptyCondition for the same reason as in search().
        now = datetime.now(timezone.utc)
        ttl_filter = Filter(
            should=[
                IsEmptyCondition(is_empty=PayloadField(key="expires_at")),
                FieldCondition(key="expires_at", range=DatetimeRange(gt=now)),
                FieldCondition(key="pinned", match=MatchValue(value=True)),
            ]
        )
        must_conditions.append(ttl_filter)

        # Exclude memories by layer (e.g., exclude consolidated from remember dedup)
        must_not_conditions: list = []
        if exclude_layers:
            for layer in exclude_layers:
                must_not_conditions.append(
                    FieldCondition(key="memory_layer", match=MatchValue(value=layer))
                )
            # Backward compat: memories without memory_layer field are treated
            # as "consolidated". Qdrant must_not with MatchValue only matches
            # points where the field EXISTS and equals the value — field-absent
            # points pass through. We need to also exclude those.
            if "consolidated" in exclude_layers:
                must_not_conditions.append(
                    IsEmptyCondition(is_empty=PayloadField(key="memory_layer"))
                )

        query_filter = Filter(
            must=must_conditions,
            must_not=must_not_conditions if must_not_conditions else None,
        )

        result = self._client.query_points(
            collection_name=self.collection_name,
            query=vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )

        return [
            {
                "id": str(p.id),
                "text": p.payload.get("data", ""),
                "score": p.score,
                "type": p.payload.get("memory_type", ""),
                "categories": p.payload.get("categories", []),
            }
            for p in result.points
        ]

    def get_by_id(self, memory_id: str) -> dict | None:
        """Get a single memory by ID.

        Returns the memory dict (in standard format), or None if not found.
        """
        try:
            with _qdrant_timer("retrieve"):
                result = self._client.retrieve(
                    collection_name=self.collection_name,
                    ids=[memory_id],
                    with_payload=True,
                )
            if result:
                return self._point_to_memory(result[0])
            return None
        except Exception:
            logger.debug("Memory %s not found", memory_id)
            return None

    def get_by_ids(self, memory_ids: list[str]) -> list[dict]:
        """Get multiple memories by their IDs.

        Uses Qdrant's batch retrieve for efficient multi-ID lookup.
        Returns list of memory dicts (in standard format) for IDs that
        exist.  Missing IDs are silently skipped.
        """
        if not memory_ids:
            return []
        logger.debug("Retrieving %d memories by IDs", len(memory_ids))
        try:
            with _qdrant_timer("retrieve"):
                result = self._client.retrieve(
                    collection_name=self.collection_name,
                    ids=memory_ids,
                    with_payload=True,
                    with_vectors=False,
                )
            return [self._point_to_memory(p) for p in result]
        except Exception:
            logger.warning("Failed to retrieve memories by IDs", exc_info=True)
            return []

    def get_all(
        self,
        *,
        user_id: str,
        owner_id: str | None = None,
        subject_user_id: str | None = None,
        agent_id: str | None = None,
        shared_only: bool = False,
        filters: dict | None = None,
        labels_filter: dict | None = None,
        exclude_layers: list[str] | None = None,
        limit: int = 100,
        order_by: OrderBy | None = None,
    ) -> dict:
        """List memories with optional filters and ordering.

        Args:
            user_id: Required user scope.
            agent_id: Optional agent scope.
            shared_only: If True and agent_id is None, restrict to memories
                without any agent_id (shared user memories only). Used by
                dual-scope list to avoid leaking sub-agent memories.
            filters: Metadata filters. Scalar values use exact match,
                list values use any-of match (MatchAny).
            exclude_layers: Memory layers to exclude from results.
            limit: Maximum results.
            order_by: Optional Qdrant OrderBy for server-side sorting.
                When set, results are returned in the specified order.
                Note: points missing the ordered field are excluded.

        Returns dict with "results" key containing list of memory dicts.
        """
        must_conditions: list = [_build_owner_scope_condition(user_id, owner_id)]
        if subject_user_id:
            must_conditions.append(
                FieldCondition(key="user_id", match=MatchValue(value=subject_user_id))
            )
        if agent_id:
            must_conditions.append(
                FieldCondition(key="agent_id", match=MatchValue(value=agent_id))
            )
        elif shared_only:
            must_conditions.append(
                IsEmptyCondition(is_empty=PayloadField(key="agent_id"))
            )
        if filters:
            for key, value in filters.items():
                if isinstance(value, list):
                    must_conditions.append(
                        FieldCondition(key=key, match=MatchAny(any=value))
                    )
                else:
                    must_conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )
        if labels_filter:
            must_conditions.extend(_build_labels_conditions(labels_filter))

        must_not_conditions: list = []
        if exclude_layers:
            for layer in exclude_layers:
                must_not_conditions.append(
                    FieldCondition(key="memory_layer", match=MatchValue(value=layer))
                )
            if "consolidated" in exclude_layers:
                must_not_conditions.append(
                    IsEmptyCondition(is_empty=PayloadField(key="memory_layer"))
                )

        scroll_kwargs: dict[str, Any] = {
            "collection_name": self.collection_name,
            "scroll_filter": Filter(
                must=must_conditions,
                must_not=must_not_conditions if must_not_conditions else None,
            ),
            "limit": limit,
            "with_payload": True,
            "with_vectors": False,
        }
        if order_by is not None:
            scroll_kwargs["order_by"] = order_by

        try:
            with _qdrant_timer("scroll"):
                points, _ = self._client.scroll(**scroll_kwargs)
        except Exception:
            if order_by is not None:
                # Fallback: order_by failed (e.g., missing payload index).
                # Retry without order_by and sort in Python instead.
                logger.warning(
                    "order_by scroll failed (missing index?), "
                    "falling back to client-side sort"
                )
                scroll_kwargs.pop("order_by", None)
                with _qdrant_timer("scroll"):
                    points, _ = self._client.scroll(**scroll_kwargs)
                results = [self._point_to_memory(p) for p in points]
                # Replicate the requested sort order in Python
                reverse = (
                    order_by.direction is None
                    or str(order_by.direction).lower() != "asc"
                )
                results.sort(
                    key=lambda m: m.get(order_by.key, ""),  # type: ignore[union-attr]
                    reverse=reverse,
                )
                return {"results": results}
            raise

        return {"results": [self._point_to_memory(p) for p in points]}

    def update_content(
        self,
        memory_id: str,
        text: str,
        vector: list[float] | None = None,
        sparse_vector: Any = None,
    ) -> None:
        """Update a memory's content text and re-embed.

        Preserves all existing metadata. Only changes: data, hash,
        updated_at, and the embedding vector(s).

        Args:
            memory_id: The memory to update.
            text: New content text.
            vector: Pre-computed dense embedding vector. If None, the text
                    is re-embedded automatically.
            sparse_vector: Optional BM25 sparse vector. When provided,
                    stored alongside the dense vector.
        """
        # Fetch existing payload to preserve metadata
        with _qdrant_timer("retrieve"):
            existing = self._client.retrieve(
                collection_name=self.collection_name,
                ids=[memory_id],
                with_payload=True,
                with_vectors=False,
            )
        if not existing:
            raise ValueError(f"Memory {memory_id} not found")

        payload = dict(existing[0].payload or {})
        payload["data"] = text
        payload["hash"] = _hash(text.encode()).hexdigest()
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()

        # Use pre-computed vector or re-embed
        if vector is None:
            vector = self._embedding.embed(text)

        # Use named vectors when sparse is available
        point_vector: Any = (
            {"": vector, "bm25": sparse_vector} if sparse_vector is not None else vector
        )

        with self._write_guard():
            with _qdrant_timer("update"):
                self._client.upsert(
                    collection_name=self.collection_name,
                    points=[
                        PointStruct(id=memory_id, vector=point_vector, payload=payload)
                    ],
                )

    def update_metadata(self, memory_id: str, metadata: dict) -> None:
        """Update metadata fields on a memory without changing content.

        Uses Qdrant's set_payload for efficient partial updates.
        """
        with self._write_guard():
            with _qdrant_timer("set_payload"):
                self._client.set_payload(
                    collection_name=self.collection_name,
                    payload=metadata,
                    points=[memory_id],
                )

    def batch_update_metadata(self, updates: list[tuple[str, dict]]) -> None:
        """Batch update metadata on multiple memories.

        Args:
            updates: List of (memory_id, metadata_dict) tuples.
        """
        with self._write_guard():
            for memory_id, metadata in updates:
                try:
                    self._client.set_payload(
                        collection_name=self.collection_name,
                        payload=metadata,
                        points=[memory_id],
                    )
                except Exception:
                    logger.warning("Failed to update metadata for memory %s", memory_id)

    def delete(self, memory_id: str) -> None:
        """Delete a single memory by ID."""
        with self._write_guard():
            with _qdrant_timer("delete"):
                self._client.delete(
                    collection_name=self.collection_name,
                    points_selector=PointIdsList(points=[memory_id]),
                )

    def delete_all(self, *, user_id: str, agent_id: str | None = None) -> None:
        """Delete all memories for a user/agent scope.

        Uses filter-based deletion — does NOT destroy the collection.
        """
        must_conditions: list = [
            FieldCondition(key="user_id", match=MatchValue(value=user_id)),
        ]
        if agent_id:
            must_conditions.append(
                FieldCondition(key="agent_id", match=MatchValue(value=agent_id))
            )

        with self._write_guard():
            with _qdrant_timer("delete_all"):
                self._client.delete(
                    collection_name=self.collection_name,
                    points_selector=Filter(must=must_conditions),
                )

    def artifact_has_references(
        self,
        *,
        artifact_id: str,
        exclude_memory_id: str,
    ) -> bool:
        """Check if any memory (other than exclude_memory_id) references an artifact.

        Uses Qdrant nested field filtering on artifacts[].id to find
        memories that reference the given artifact_id, excluding the
        specified memory.

        Returns True if at least one other memory references the artifact.
        """
        from qdrant_client.models import HasIdCondition

        scroll_filter = Filter(
            must=[
                FieldCondition(
                    key="artifacts[].id",
                    match=MatchValue(value=artifact_id),
                ),
            ],
            must_not=[
                HasIdCondition(has_id=[exclude_memory_id]),
            ],
        )

        results, _ = self._client.scroll(
            collection_name=self.collection_name,
            scroll_filter=scroll_filter,
            limit=1,
        )

        return len(results) > 0

    # ── Fsck helpers ───────────────────────────────────────────────────

    def scroll_with_vectors(
        self,
        *,
        user_id: str,
        agent_id: str | None = None,
        shared_only: bool = False,
        filters: dict | None = None,
        exclude_layers: list[str] | None = None,
        exclude_expired: bool = False,
    ) -> list[dict[str, Any]]:
        """Scroll ALL memories for a user, including stored embedding vectors.

        Used by the fsck (memory check) feature to retrieve stored vectors
        for similarity comparison without re-embedding.

        Paginates through all Qdrant points using the scroll cursor so that
        users with large memory sets are fully covered — no hard limit.

        Args:
            user_id: Required user scope.
            agent_id: Optional agent scope.
            shared_only: If True and agent_id is None, restrict to memories
                without any agent_id (shared user memories only). Used by
                dual-scope fsck to scroll shared memories separately.
            filters: Additional metadata filters.
            exclude_layers: Memory layers to exclude (e.g. ["raw"]).
            exclude_expired: If True, exclude memories whose expires_at is
                in the past (non-pinned). Best-effort Qdrant-side filter.

        Returns list of memory dicts with an extra "vector" key containing
        the stored embedding.
        """
        from qdrant_client.models import IsEmptyCondition, MatchAny

        must_conditions: list = [
            FieldCondition(key="user_id", match=MatchValue(value=user_id)),
        ]
        must_not_conditions: list = []

        if agent_id:
            must_conditions.append(
                FieldCondition(key="agent_id", match=MatchValue(value=agent_id))
            )
        elif shared_only:
            must_conditions.append(
                IsEmptyCondition(is_empty=PayloadField(key="agent_id"))
            )
        if filters:
            for key, value in filters.items():
                if isinstance(value, list):
                    must_conditions.append(
                        FieldCondition(key=key, match=MatchAny(any=value))
                    )
                else:
                    must_conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )

        if exclude_layers:
            from qdrant_client.models import MatchValue as MV

            for layer in exclude_layers:
                must_not_conditions.append(
                    FieldCondition(key="memory_layer", match=MV(value=layer))
                )

        if exclude_expired:
            from qdrant_client.models import DatetimeRange

            now_iso = datetime.now(timezone.utc).isoformat()
            # Exclude expired non-pinned memories. We filter out memories
            # whose expires_at is in the past. Pinned memories are exempt
            # (they have no expires_at or it's ignored).
            # Note: this is a best-effort filter — Python-side filtering
            # in fsck also catches edge cases.
            must_not_conditions.append(
                FieldCondition(
                    key="expires_at",
                    range=DatetimeRange(lt=now_iso),
                )
            )

        scroll_filter = Filter(
            must=must_conditions,
            must_not=must_not_conditions or None,
        )
        # Use a page size that balances memory and round-trips.
        # with_vectors=True makes payloads larger, so keep pages moderate.
        _PAGE_SIZE = 256

        results = []
        offset = None  # None = start from beginning

        while True:
            points, next_offset = self._client.scroll(
                collection_name=self.collection_name,
                scroll_filter=scroll_filter,
                limit=_PAGE_SIZE,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            for p in points:
                mem = self._point_to_memory(p)
                # Normalize vector format: extract dense vector from named
                # vectors dict (hybrid mode) or use as-is (legacy unnamed).
                raw_vector = p.vector
                if isinstance(raw_vector, dict):
                    mem["vector"] = raw_vector.get("", raw_vector.get(None))
                else:
                    mem["vector"] = raw_vector
                results.append(mem)

            if next_offset is None:
                break
            offset = next_offset

        return results

    def scroll_gc_metadata(self, *, user_id: str) -> list[dict[str, Any]]:
        """Scroll lightweight metadata used by superseded raw-memory GC."""
        scroll_filter = Filter(
            must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
        )
        results: list[dict[str, Any]] = []
        offset = None

        while True:
            points, next_offset = self._client.scroll(
                collection_name=self.collection_name,
                scroll_filter=scroll_filter,
                limit=2048,
                offset=offset,
                with_payload=[
                    "memory_layer",
                    "superseded_by",
                    "artifacts",
                    "created_at_utc",
                ],
                with_vectors=False,
            )
            for point in points:
                results.append(
                    {
                        "id": str(point.id),
                        "metadata": dict(point.payload or {}),
                    }
                )
            if next_offset is None:
                break
            offset = next_offset

        return results

    def search_by_vector(
        self,
        vector: list[float],
        *,
        user_id: str,
        agent_id: str | None = None,
        shared_only: bool = False,
        limit: int = 5,
        exclude_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar memories using a pre-computed vector.

        Returns full memory dicts (not just id/text like search_similar).
        Used by fsck for duplicate detection with stored vectors.

        Args:
            vector: Pre-computed embedding vector.
            user_id: Required user scope.
            agent_id: Optional agent scope.
            shared_only: If True and agent_id is None, restrict to memories
                without any agent_id (shared user memories only). Used by
                dual-scope fsck to search shared memories separately.
            limit: Maximum results.
            exclude_ids: Point IDs to exclude from results.
        """
        from qdrant_client.models import HasIdCondition, IsEmptyCondition

        must_conditions: list = [
            FieldCondition(key="user_id", match=MatchValue(value=user_id)),
        ]
        if agent_id:
            must_conditions.append(
                FieldCondition(key="agent_id", match=MatchValue(value=agent_id))
            )
        elif shared_only:
            must_conditions.append(
                IsEmptyCondition(is_empty=PayloadField(key="agent_id"))
            )

        must_not: list = []
        if exclude_ids:
            must_not.append(HasIdCondition(has_id=exclude_ids))

        result = self._client.query_points(
            collection_name=self.collection_name,
            query=vector,
            query_filter=Filter(must=must_conditions, must_not=must_not or None),
            limit=limit,
            with_payload=True,
        )

        memories = []
        for point in result.points:
            mem = self._point_to_memory(point)
            mem["score"] = point.score
            memories.append(mem)
        return memories

    def search_by_vector_ids(
        self,
        vector: list[float],
        *,
        user_id: str,
        agent_id: str | None = None,
        shared_only: bool = False,
        limit: int = 5,
        exclude_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar memories by vector, returning only IDs and scores.

        Lightweight variant of search_by_vector that skips payload transfer.
        Used by fsck duplicate detection where only the similarity graph
        structure matters, not the memory content.

        Args:
            vector: Pre-computed embedding vector.
            user_id: Required user scope.
            agent_id: Optional agent scope.
            shared_only: If True and agent_id is None, restrict to memories
                without any agent_id (shared user memories only).
            limit: Maximum results.
            exclude_ids: Point IDs to exclude from results.
        """
        from qdrant_client.models import HasIdCondition, IsEmptyCondition

        must_conditions: list = [
            FieldCondition(key="user_id", match=MatchValue(value=user_id)),
        ]
        if agent_id:
            must_conditions.append(
                FieldCondition(key="agent_id", match=MatchValue(value=agent_id))
            )
        elif shared_only:
            must_conditions.append(
                IsEmptyCondition(is_empty=PayloadField(key="agent_id"))
            )

        must_not: list = []
        if exclude_ids:
            must_not.append(HasIdCondition(has_id=exclude_ids))

        result = self._client.query_points(
            collection_name=self.collection_name,
            query=vector,
            query_filter=Filter(must=must_conditions, must_not=must_not or None),
            limit=limit,
            with_payload=False,
        )

        return [{"id": str(point.id), "score": point.score} for point in result.points]

    # ── Specialized queries ──────────────────────────────────────────

    def get_recent_memories(
        self,
        *,
        user_id: str,
        owner_id: str | None = None,
        subject_user_id: str | None = None,
        agent_id: str | None = None,
        since: datetime,
        limit: int = 50,
        memory_types: list[str] | None = None,
        importance_levels: list[str] | None = None,
        exclude_layers: list[str] | None = None,
    ) -> list[dict]:
        """Get memories created after a given timestamp.

        Uses Qdrant's DatetimeRange filtering for efficient date queries.

        Args:
            user_id: Required user scope.
            agent_id: If set, filter to only memories with this agent_id.
                      If None, returns memories without agent_id (shared).
            since: Only return memories created after this timestamp.
            limit: Maximum number of results.
            memory_types: If set, filter to only these memory types.
            importance_levels: If set, filter to only these importance levels
                (e.g., ["normal", "high", "critical"]).
            exclude_layers: Memory layers to exclude from results.

        Returns memories ordered by created_at descending (most recent first).
        """
        from qdrant_client.models import DatetimeRange, MatchAny

        must_conditions: list = [
            _build_owner_scope_condition(user_id, owner_id),
            FieldCondition(
                key="created_at",
                range=DatetimeRange(gte=since),
            ),
        ]
        if subject_user_id:
            must_conditions.append(
                FieldCondition(key="user_id", match=MatchValue(value=subject_user_id))
            )

        # Filter by agent_id: if provided, match exactly; if None, match only
        # memories without agent_id (shared user memories).
        # Use IsEmptyCondition (not IsNullCondition) because agent_id is
        # omitted from the payload for shared memories — IsNullCondition
        # only matches keys explicitly set to null, not absent keys.
        if agent_id:
            must_conditions.append(
                FieldCondition(key="agent_id", match=MatchValue(value=agent_id))
            )
        else:
            must_conditions.append(
                IsEmptyCondition(is_empty=PayloadField(key="agent_id"))
            )

        # Filter by memory types if specified
        if memory_types:
            must_conditions.append(
                FieldCondition(key="memory_type", match=MatchAny(any=memory_types))
            )

        # Filter by importance levels if specified
        if importance_levels:
            must_conditions.append(
                FieldCondition(key="importance", match=MatchAny(any=importance_levels))
            )

        must_not_conditions: list = []
        if exclude_layers:
            for layer in exclude_layers:
                must_not_conditions.append(
                    FieldCondition(key="memory_layer", match=MatchValue(value=layer))
                )
            if "consolidated" in exclude_layers:
                must_not_conditions.append(
                    IsEmptyCondition(is_empty=PayloadField(key="memory_layer"))
                )

        points, _ = self._client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=must_conditions,
                must_not=must_not_conditions if must_not_conditions else None,
            ),
            limit=limit,
            with_payload=True,
        )

        results = [self._point_to_memory(p) for p in points]

        # Sort by created_at descending (most recent first)
        results.sort(key=lambda m: m.get("created_at", ""), reverse=True)

        return results

    def get_pinned_memories(
        self,
        *,
        user_id: str,
        owner_id: str | None = None,
        subject_user_id: str | None = None,
        agent_id: str | None = None,
        exclude_agent: bool = False,
        exclude_layers: list[str] | None = None,
    ) -> list[dict]:
        """Get pinned memories, optionally filtering by agent scope.

        Args:
            user_id: Required user scope.
            agent_id: If set, also include agent-specific pinned memories.
            exclude_agent: If True, only return memories WITHOUT agent_id
                          (user-scoped memories shared across agents).
            exclude_layers: Memory layers to exclude from results.
        """
        # Fetch all pinned memories for the user
        all_pinned = self.get_all(
            user_id=user_id,
            owner_id=owner_id,
            subject_user_id=subject_user_id,
            filters={"pinned": True},
            exclude_layers=exclude_layers,
            limit=500,
        )
        results = all_pinned.get("results", [])

        filtered = []
        for mem in results:
            mem_agent_id = mem.get("agent_id")
            if exclude_agent and mem_agent_id:
                continue
            if (
                not exclude_agent
                and agent_id
                and mem_agent_id
                and mem_agent_id != agent_id
            ):
                continue
            filtered.append(mem)

        return filtered

    # ── Date range filtering ────────────────────────────────────────

    @staticmethod
    def _build_date_range_filter(
        date_start: str | None,
        date_end: str | None,
    ) -> Filter | None:
        """Build a Qdrant filter for date range queries.

        Matches memories where:
        - event_date is in the range, OR
        - event_date is missing AND created_at is in the range

        This ensures memories without event_date metadata (legacy or
        timeless facts) are still found based on their creation date.

        Args:
            date_start: Start date (YYYY-MM-DD), inclusive.
            date_end: End date (YYYY-MM-DD), inclusive. The end date
                is extended to end-of-day (23:59:59) for created_at
                comparison.

        Returns:
            A Qdrant Filter with OR logic, or None if no dates provided.
        """
        from qdrant_client.models import (
            DatetimeRange,
            IsEmptyCondition,
        )

        if not date_start and not date_end:
            return None

        # Build range for event_date (YYYY-MM-DD string comparison)
        event_date_range: dict[str, Any] = {}
        created_at_range: dict[str, Any] = {}

        if date_start:
            # event_date is stored as YYYY-MM-DD string
            event_date_range["gte"] = date_start
            # created_at is stored as ISO 8601 datetime
            created_at_range["gte"] = datetime.fromisoformat(
                f"{date_start}T00:00:00+00:00"
            )
        if date_end:
            event_date_range["lte"] = date_end
            # End of day for created_at
            created_at_range["lte"] = datetime.fromisoformat(
                f"{date_end}T23:59:59+00:00"
            )

        # Branch 1: event_date exists and is in range
        event_date_conditions: list = []
        from qdrant_client.models import Range

        event_date_conditions.append(
            FieldCondition(key="event_date", range=Range(**event_date_range))
        )

        # Branch 2: event_date is missing AND created_at is in range
        no_event_date_conditions: list = [
            IsEmptyCondition(is_empty=PayloadField(key="event_date")),
            FieldCondition(key="created_at", range=DatetimeRange(**created_at_range)),
        ]

        return Filter(
            should=[
                Filter(must=event_date_conditions),
                Filter(must=no_event_date_conditions),
            ]
        )

    def list_user_ids(self) -> list[str]:
        """Return a sorted list of all distinct user_ids in the collection.

        Scrolls all points (payload only, no vectors) and collects unique
        user_id values. Handles pagination automatically.

        Returns:
            Sorted list of user_id strings.
        """
        user_ids: set[str] = set()
        offset = None
        batch_size = 256

        while True:
            points, next_offset = self._client.scroll(
                collection_name=self.collection_name,
                scroll_filter=None,
                limit=batch_size,
                offset=offset,
                with_payload=["user_id"],
                with_vectors=False,
            )
            for point in points:
                uid = (point.payload or {}).get("user_id")
                if uid:
                    user_ids.add(uid)
            if next_offset is None:
                break
            offset = next_offset

        return sorted(user_ids)

    # ── Result formatting ────────────────────────────────────────────

    @staticmethod
    def _point_to_memory(point) -> dict:
        """Convert a Qdrant point to a standard memory dict.

        Payload structure in Qdrant:
            data, hash, created_at, updated_at, user_id, agent_id,
            role, + custom metadata fields

        Output structure:
            id, memory, hash, created_at, updated_at, user_id, agent_id,
            metadata: {memory_type, categories, importance, pinned, ...}
        """
        payload = point.payload or {}
        memory: dict[str, Any] = {
            "id": str(point.id),
            "memory": payload.get("data", ""),
            "hash": payload.get("hash", ""),
            "created_at": payload.get("created_at"),
            "updated_at": payload.get("updated_at"),
        }
        # Promote standard fields
        for field in ("user_id", "owner_id", "agent_id", "run_id"):
            if field in payload:
                memory[field] = payload[field]
        # Collect remaining fields as metadata
        skip_keys = {
            "data",
            "hash",
            "created_at",
            "updated_at",
            "user_id",
            "owner_id",
            "agent_id",
            "run_id",
            "actor_id",
        }
        metadata = {k: v for k, v in payload.items() if k not in skip_keys}
        if metadata:
            memory["metadata"] = metadata
        return memory


def _session_point_id(session_id: str) -> str:
    """Convert a session ID string to a deterministic UUID for Qdrant.

    Qdrant point IDs must be UUIDs or unsigned integers. Session IDs
    (e.g., ``ses_2e3418b72ffe5gG7rHDeU5aOkv``) are neither, so we
    derive a stable UUID5 from the session ID string.
    """
    import uuid as _uuid

    return str(_uuid.uuid5(_uuid.NAMESPACE_URL, f"mnemory:session:{session_id}"))


class SessionSummaryStore:
    """Persistent session summary storage in Qdrant.

    Stores conversation summaries in the _mnemory_sessions collection,
    separate from the main memory collection. Used by the consolidation
    service to synthesize durable knowledge from raw memories.

    Session summaries are updated on every remember call and persist
    beyond session expiry. The consolidation service reads these
    summaries alongside linked raw memories.
    """

    COLLECTION = "_mnemory_sessions"
    _LIST_BATCH_SIZE = 256
    _LIST_SAFETY_CAP = 5000

    def __init__(self, client: QdrantClient):
        self._client = client

    def upsert(
        self,
        *,
        session_id: str,
        user_id: str,
        agent_id: str | None = None,
        summary: str,
        new_memory_ids: list[str] | None = None,
    ) -> None:
        """Create or update a session summary.

        If the session already exists, updates summary, increments turn_count,
        and appends new memory IDs. If new, creates with initial values.

        Args:
            session_id: Session ID (used as Qdrant point ID).
            user_id: User scope.
            agent_id: Optional agent scope.
            summary: Current conversation summary text.
            new_memory_ids: IDs of raw memories created in this remember call.
        """
        now = datetime.now(timezone.utc).isoformat()

        # Try to read existing
        existing = self.get(session_id)

        if existing:
            # Update existing
            memory_ids = existing.get("memory_ids", [])
            if new_memory_ids:
                memory_ids.extend(new_memory_ids)
            turn_count = existing.get("turn_count", 0) + 1

            # Re-consolidation: reset state to idle when new memories
            # arrive after consolidation, making the session eligible
            # for re-consolidation on the next check cycle.
            # Race condition note: if consolidation is running concurrently,
            # last-writer-wins — both idle and consolidated are valid end
            # states, and the idle threshold prevents immediate re-trigger.
            consolidation_state = existing.get("consolidation_state", "idle")
            if consolidation_state == "consolidated" and new_memory_ids:
                consolidation_state = "idle"

            payload = {
                "session_id": session_id,
                "user_id": user_id,
                "summary": summary,
                "turn_count": turn_count,
                "memory_ids": memory_ids,
                "updated_at": now,
                "created_at": existing.get("created_at", now),
                "consolidation_state": consolidation_state,
            }
            if agent_id:
                payload["agent_id"] = agent_id
            elif existing.get("agent_id"):
                payload["agent_id"] = existing["agent_id"]
        else:
            # Create new
            payload = {
                "session_id": session_id,
                "user_id": user_id,
                "summary": summary,
                "turn_count": 1,
                "memory_ids": new_memory_ids or [],
                "created_at": now,
                "updated_at": now,
                "consolidation_state": "idle",
            }
            if agent_id:
                payload["agent_id"] = agent_id

        self._client.upsert(
            collection_name=self.COLLECTION,
            points=[
                PointStruct(
                    id=_session_point_id(session_id),
                    vector=[0.0],  # dummy vector
                    payload=payload,
                )
            ],
        )

    def get(self, session_id: str) -> dict | None:
        """Get a session summary by ID.

        Returns the session dict, or None if not found.
        """
        try:
            result = self._client.retrieve(
                collection_name=self.COLLECTION,
                ids=[_session_point_id(session_id)],
                with_payload=True,
            )
            if result:
                return dict(result[0].payload or {})
        except Exception:
            logger.debug("Session summary %s not found", session_id)
        return None

    def find_pending(
        self,
        user_id: str,
        idle_threshold_seconds: int = 3600,
    ) -> list[dict]:
        """Find sessions awaiting consolidation.

        Returns sessions where:
        - consolidation_state = "idle"
        - updated_at is older than idle_threshold_seconds
        - user_id matches

        Args:
            user_id: Filter by user.
            idle_threshold_seconds: Minimum idle time before eligible.

        Returns:
            List of session dicts.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=idle_threshold_seconds)

        result = self._client.scroll(
            collection_name=self.COLLECTION,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                    FieldCondition(
                        key="consolidation_state",
                        match=MatchValue(value="idle"),
                    ),
                ]
            ),
            limit=100,
            with_payload=True,
        )

        # Filter by updated_at in Python (simpler than datetime range on string field)
        sessions = []
        for point in result[0]:
            payload = dict(point.payload or {})
            # Skip entries with missing session_id (corrupted/legacy data)
            if not payload.get("session_id"):
                continue
            updated_at_str = payload.get("updated_at", "")
            if updated_at_str:
                try:
                    updated_at = datetime.fromisoformat(updated_at_str)
                    if updated_at.tzinfo is None:
                        updated_at = updated_at.replace(tzinfo=timezone.utc)
                    if updated_at < cutoff:
                        # Only include sessions that have at least one memory
                        if payload.get("memory_ids"):
                            sessions.append(payload)
                except (ValueError, TypeError):
                    pass

        return sessions

    def scan_consolidation_candidates(self) -> list[dict[str, Any]]:
        """Scroll all idle and consolidating session summaries."""
        sessions: list[dict[str, Any]] = []
        offset = None
        payload_fields = [
            "session_id",
            "user_id",
            "memory_ids",
            "updated_at",
            "consolidation_state",
            "consolidated_memory_ids",
        ]
        scroll_filter = Filter(
            must=[
                FieldCondition(
                    key="consolidation_state",
                    match=MatchAny(any=["idle", "consolidating"]),
                )
            ]
        )

        while True:
            points, next_offset = self._client.scroll(
                collection_name=self.COLLECTION,
                scroll_filter=scroll_filter,
                limit=self._LIST_BATCH_SIZE,
                offset=offset,
                with_payload=payload_fields,
                with_vectors=False,
            )
            for point in points:
                payload = dict(point.payload or {})
                if payload.get("session_id") and payload.get("user_id"):
                    sessions.append(payload)
            if next_offset is None:
                break
            offset = next_offset

        return sessions

    def update_consolidation_state(
        self,
        session_id: str,
        state: str,
        *,
        consolidated_memory_ids: list[str] | None = None,
    ) -> None:
        """Update the consolidation state of a session.

        Args:
            session_id: Session to update.
            state: New state ("idle", "consolidating", "consolidated", "failed").
            consolidated_memory_ids: IDs of produced consolidated memories
                (set when transitioning to "consolidated").
        """
        payload_update: dict[str, Any] = {
            "consolidation_state": state,
        }
        if state == "consolidated":
            payload_update["consolidated_at"] = datetime.now(timezone.utc).isoformat()
        if consolidated_memory_ids is not None:
            payload_update["consolidated_memory_ids"] = consolidated_memory_ids

        self._client.set_payload(
            collection_name=self.COLLECTION,
            payload=payload_update,
            points=[_session_point_id(session_id)],
        )

    def delete(self, session_id: str) -> dict | None:
        """Delete a session summary by ID.

        Returns the session payload before deletion, or None if not found.
        The caller can use the returned payload to clean up linked memories.
        """
        # Read the session first so we can return its data
        session = self.get(session_id)

        self._client.delete(
            collection_name=self.COLLECTION,
            points_selector=PointIdsList(points=[_session_point_id(session_id)]),
        )

        return session

    def list_for_user(
        self,
        user_id: str,
        *,
        limit: int = 50,
        consolidation_state: str | None = None,
        offset: int = 0,
        q: str | None = None,
        sort_by: Literal["updated_at", "created_at"] = "updated_at",
        sort_dir: Literal["asc", "desc"] = "desc",
        include_metadata: bool = False,
    ) -> list[dict] | dict[str, Any]:
        """List session summaries for a user.

        Args:
            user_id: Filter by user.
            limit: Maximum results.
            consolidation_state: Optional filter by state.
            offset: Zero-based starting offset into the filtered result set.
            q: Optional case-insensitive substring match against session
                summary, session_id, and agent_id.
            sort_by: Field to sort by.
            sort_dir: Sort direction.
            include_metadata: When True, return paging metadata alongside
                the session page for API consumers.

        Returns:
            Either a list of session dicts, or a metadata envelope when
            include_metadata=True.
        """
        must_conditions = [
            FieldCondition(key="user_id", match=MatchValue(value=user_id)),
        ]
        if consolidation_state:
            must_conditions.append(
                FieldCondition(
                    key="consolidation_state",
                    match=MatchValue(value=consolidation_state),
                )
            )

        sessions: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        scroll_filter = Filter(must=must_conditions)
        scroll_offset = None
        total_truncated = False
        started_at = time.monotonic()
        query = (q or "").strip().lower()

        try:
            while True:
                points, next_offset = self._client.scroll(
                    collection_name=self.COLLECTION,
                    scroll_filter=scroll_filter,
                    limit=self._LIST_BATCH_SIZE,
                    offset=scroll_offset,
                    with_payload=True,
                    with_vectors=False,
                )

                for point in points:
                    payload = dict(point.payload or {})
                    # Skip entries with empty/corrupted payload
                    if (
                        not payload
                        or not payload.get("session_id")
                        or not payload.get("summary")
                    ):
                        logger.debug(
                            "Skipping corrupted session entry (point_id=%s)", point.id
                        )
                        continue

                    sid = payload["session_id"]
                    if sid in seen_ids:
                        continue
                    seen_ids.add(sid)

                    if query:
                        search_haystack = " ".join(
                            [
                                str(payload.get("summary") or ""),
                                str(sid),
                                str(payload.get("agent_id") or ""),
                            ]
                        ).lower()
                        if query not in search_haystack:
                            continue

                    sessions.append(payload)

                    if (
                        len(seen_ids) >= self._LIST_SAFETY_CAP
                        and next_offset is not None
                    ):
                        total_truncated = True
                        break

                if total_truncated or next_offset is None:
                    break
                scroll_offset = next_offset
        except Exception as exc:
            logger.exception("Failed to list session summaries for user %s", user_id)
            raise RuntimeError("Failed to list session summaries") from exc

        elapsed = time.monotonic() - started_at
        if elapsed > 2.0:
            logger.warning(
                "Session summary listing was slow for user %s: %.2fs (%d matched%s)",
                user_id,
                elapsed,
                len(sessions),
                ", truncated" if total_truncated else "",
            )

        if sort_dir == "asc":
            sessions.sort(
                key=lambda session: (
                    not bool(session.get(sort_by)),
                    str(session.get(sort_by) or ""),
                )
            )
        else:
            sessions.sort(
                key=lambda session: (
                    bool(session.get(sort_by)),
                    str(session.get(sort_by) or ""),
                ),
                reverse=True,
            )

        total = len(sessions)
        page = sessions[offset : offset + limit]
        has_more = offset + limit < total

        if include_metadata:
            return {
                "sessions": page,
                "total": total,
                "offset": offset,
                "limit": limit,
                "has_more": has_more,
                "total_truncated": total_truncated,
            }

        return page
