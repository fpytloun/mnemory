"""Configuration management for mnemory.

All settings are loaded from environment variables with sensible defaults.
Defaults are optimized for local development — just set OPENAI_API_KEY and
run. Override for production (Qdrant, S3, API keys).

Data is stored in ~/.mnemory by default (override with DATA_DIR env var).
In Docker, DATA_DIR is set to /data for volume mounting.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger("mnemory")


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, str(default)))


def _env_bool(key: str, default: bool = False) -> bool:
    return os.environ.get(key, str(default)).lower() in ("true", "1", "yes")


def _env_int_or_none(key: str, default: int | None = None) -> int | None:
    """Parse an env var as int, treating empty string and 'null'/'none' as None."""
    raw = os.environ.get(key, "")
    if not raw:
        return default
    if raw.lower() in ("null", "none"):
        return None
    return int(raw)


def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, str(default)))


def _data_dir() -> str:
    """Resolve base data directory.

    Priority: DATA_DIR env var > ~/.mnemory
    In Docker, the Dockerfile sets DATA_DIR=/data for volume mounting.
    Locally, defaults to ~/.mnemory for clean, predictable storage.
    """
    raw = os.environ.get("DATA_DIR", "")
    if raw:
        return raw
    return os.path.join(os.path.expanduser("~"), ".mnemory")


def _llm_api_key() -> str:
    """Resolve LLM API key with fallback chain.

    Priority: LLM_API_KEY > OPENAI_API_KEY
    """
    return _env("LLM_API_KEY") or _env("OPENAI_API_KEY")


def _embed_api_key() -> str:
    """Resolve embedding API key with fallback chain.

    Priority: EMBED_API_KEY > LLM_API_KEY > OPENAI_API_KEY
    """
    return _env("EMBED_API_KEY") or _llm_api_key()


def _llm_base_url() -> str:
    """Resolve LLM base URL with fallback chain.

    Priority: LLM_BASE_URL > OPENAI_API_BASE > default OpenAI URL
    """
    return (
        _env("LLM_BASE_URL") or _env("OPENAI_API_BASE") or "https://api.openai.com/v1"
    )


@dataclass
class VectorConfig:
    """Vector store configuration (Qdrant only).

    Mode detection:
    - If QDRANT_HOST is set → remote mode (connects to Qdrant server)
    - If QDRANT_HOST is not set → local mode (embedded Qdrant, data in QDRANT_PATH)
    """

    qdrant_host: str = field(default_factory=lambda: _env("QDRANT_HOST"))
    qdrant_port: int = field(default_factory=lambda: _env_int("QDRANT_PORT", 6333))
    qdrant_api_key: str = field(default_factory=lambda: _env("QDRANT_API_KEY"))
    collection_name: str = field(
        default_factory=lambda: _env("QDRANT_COLLECTION", "mnemory")
    )

    # Local mode settings (used when QDRANT_HOST is not set)
    qdrant_path: str = field(
        default_factory=lambda: (
            _env("QDRANT_PATH") or os.path.join(_data_dir(), "qdrant")
        )
    )

    @property
    def is_remote(self) -> bool:
        """True if connecting to a remote Qdrant server."""
        return bool(self.qdrant_host)


@dataclass
class LLMConfig:
    """LLM configuration."""

    model: str = field(default_factory=lambda: _env("LLM_MODEL", "gpt-5.4-mini"))
    base_url: str = field(default_factory=_llm_base_url)
    api_key: str = field(default_factory=_llm_api_key)
    temperature: float = 0.1
    reasoning_effort: str | None = field(
        default_factory=lambda: _env("LLM_REASONING_EFFORT") or None
    )


@dataclass
class EmbedConfig:
    """Embedding model configuration."""

    model: str = field(
        default_factory=lambda: _env("EMBED_MODEL", "text-embedding-3-small")
    )
    base_url: str = field(
        default_factory=lambda: _env("EMBED_BASE_URL") or _llm_base_url()
    )
    api_key: str = field(default_factory=_embed_api_key)
    dims: int = field(default_factory=lambda: _env_int("EMBED_DIMS", 1536))


@dataclass
class ArtifactConfig:
    """Artifact store configuration."""

    backend: str = field(default_factory=lambda: _env("ARTIFACT_BACKEND", "filesystem"))

    # S3 / MinIO settings
    s3_endpoint: str = field(
        default_factory=lambda: _env("S3_ENDPOINT", "http://localhost:9000")
    )
    s3_access_key: str = field(default_factory=lambda: _env("S3_ACCESS_KEY"))
    s3_secret_key: str = field(default_factory=lambda: _env("S3_SECRET_KEY"))
    s3_bucket: str = field(default_factory=lambda: _env("S3_BUCKET", "mnemory"))
    s3_region: str = field(default_factory=lambda: _env("S3_REGION"))

    # Filesystem settings (default for local development)
    filesystem_path: str = field(
        default_factory=lambda: (
            _env("ARTIFACT_PATH") or os.path.join(_data_dir(), "artifacts")
        )
    )


def _parse_api_keys(raw: str) -> dict[str, str]:
    """Parse MCP_API_KEYS JSON into a dict mapping key -> user_id.

    Format: {"api-key-1": "username", "api-key-2": "*"}
    A value of "*" means the key authenticates but does not bind to a user_id.
    """
    if not raw:
        return {}
    try:
        keys = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"MCP_API_KEYS is not valid JSON: {e}") from e
    if not isinstance(keys, dict):
        raise ValueError("MCP_API_KEYS must be a JSON object mapping key -> user_id")
    for k, v in keys.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise ValueError("MCP_API_KEYS keys and values must be strings")
    return keys


@dataclass
class ServerConfig:
    """MCP server configuration."""

    host: str = field(default_factory=lambda: _env("MCP_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _env_int("MCP_PORT", 8050))
    api_key: str = field(default_factory=lambda: _env("MCP_API_KEY"))
    api_keys: dict[str, str] = field(
        default_factory=lambda: _parse_api_keys(_env("MCP_API_KEYS"))
    )
    jwt_public_key: str = field(default_factory=lambda: _env("MNEMORY_JWT_PUBLIC_KEY"))
    jwks_url: str = field(default_factory=lambda: _env("MNEMORY_JWKS_URL"))
    enable_delete_all: bool = field(
        default_factory=lambda: _env_bool("ENABLE_DELETE_ALL", False)
    )
    instruction_mode: str = field(
        default_factory=lambda: _env("INSTRUCTION_MODE", "proactive")
    )

    # Management port for /health and /metrics endpoints.
    # When set (non-zero) and different from MCP_PORT, a separate HTTP server
    # runs on this port without authentication. The main port still serves the
    # same endpoints with standard API key auth.
    mgmt_port: int = field(default_factory=lambda: _env_int("MGMT_PORT", 0))
    mgmt_host: str = field(
        default_factory=lambda: _env("MGMT_HOST") or _env("MCP_HOST", "0.0.0.0")
    )

    # Prometheus metrics endpoint
    enable_metrics: bool = field(
        default_factory=lambda: _env_bool("ENABLE_METRICS", True)
    )
    metrics_cache_ttl: int = field(
        default_factory=lambda: _env_int("METRICS_CACHE_TTL", 600)
    )

    # Thread pool size for MCP tool execution.
    # MCP tool functions are async but offload blocking work (LLM calls,
    # embeddings, Qdrant queries) to a thread pool via asyncio.to_thread().
    # This controls the max concurrent blocking operations.
    # Default: min(32, os.cpu_count() + 4) — same as Python's default.
    thread_pool_size: int = field(
        default_factory=lambda: _env_int(
            "MCP_THREAD_POOL_SIZE",
            min(32, (os.cpu_count() or 1) + 4),
        )
    )

    # Download token settings for artifact raw access
    download_token_ttl: int = field(
        default_factory=lambda: _env_int("DOWNLOAD_TOKEN_TTL", 3600)
    )
    download_token_max_ttl: int = field(
        default_factory=lambda: _env_int("DOWNLOAD_TOKEN_MAX_TTL", 86400)
    )
    base_url: str = field(default_factory=lambda: _env("SERVER_BASE_URL"))

    # Exchange session cookie security.  Default True (HTTPS only).
    # Set to False only for localhost development without TLS.
    exchange_cookie_secure: bool = field(
        default_factory=lambda: _env_bool("EXCHANGE_COOKIE_SECURE", True)
    )

    @property
    def has_mgmt_port(self) -> bool:
        """Whether a separate management port is configured."""
        return self.mgmt_port > 0 and self.mgmt_port != self.port


@dataclass
class MemoryConfig:
    """Memory behavior configuration."""

    max_memory_length: int = field(
        default_factory=lambda: _env_int("MAX_MEMORY_LENGTH", 1000)
    )
    max_artifact_size: int = field(
        default_factory=lambda: _env_int("MAX_ARTIFACT_SIZE", 10_485_760)
    )
    max_core_context_length: int = field(
        default_factory=lambda: _env_int("MAX_CORE_CONTEXT_LENGTH", 4000)
    )
    default_recent_days: int = field(
        default_factory=lambda: _env_int("DEFAULT_RECENT_DAYS", 7)
    )
    recent_limit_user: int = field(
        default_factory=lambda: _env_int("RECENT_LIMIT_USER", 25)
    )
    recent_limit_agent: int = field(
        default_factory=lambda: _env_int("RECENT_LIMIT_AGENT", 25)
    )
    auto_classify: bool = field(
        default_factory=lambda: _env_bool("AUTO_CLASSIFY", True)
    )
    classify_cache_ttl: int = field(
        default_factory=lambda: _env_int("CLASSIFY_CACHE_TTL", 300)
    )
    core_memories_cache_ttl: int = field(
        default_factory=lambda: _env_int("CORE_MEMORIES_CACHE_TTL", 300)
    )

    # Core memories: include top-N non-pinned memories by importance in the
    # main sections (User Facts, User Preferences, etc.). Set to 0 to disable
    # (only pinned memories in core sections, original behavior).
    core_top_memories: int = field(
        default_factory=lambda: _env_int("CORE_TOP_MEMORIES", 10)
    )

    # Core memories: minimum importance for non-pinned memories to be included.
    # Only memories at or above this level are considered for top-N inclusion.
    # Options: low, normal, high, critical
    core_min_importance: str = field(
        default_factory=lambda: _env("CORE_MIN_IMPORTANCE", "normal")
    )

    # Core memories: max memories per section (Agent Identity, User Facts, etc.).
    # Pinned memories are always included first; the limit caps the total
    # (pinned + non-pinned) per section. Set to 0 for unlimited.
    core_max_per_section: int = field(
        default_factory=lambda: max(0, _env_int("CORE_MAX_PER_SECTION", 25))
    )

    # Core memories: minimum importance for recent context memories.
    # Filters low-importance session noise from the recent context section.
    # Options: low, normal, high, critical
    core_recent_min_importance: str = field(
        default_factory=lambda: _env("CORE_RECENT_MIN_IMPORTANCE", "normal")
    )

    # TTL defaults by memory type (days, None = permanent)
    ttl_fact: int | None = field(
        default_factory=lambda: _env_int_or_none("TTL_FACT", None)
    )
    ttl_preference: int | None = field(
        default_factory=lambda: _env_int_or_none("TTL_PREFERENCE", None)
    )
    ttl_episodic: int | None = field(
        default_factory=lambda: _env_int_or_none("TTL_EPISODIC", 90)
    )
    ttl_procedural: int | None = field(
        default_factory=lambda: _env_int_or_none("TTL_PROCEDURAL", 60)
    )
    ttl_context: int | None = field(
        default_factory=lambda: _env_int_or_none("TTL_CONTEXT", 7)
    )

    # Access tracking
    track_memory_access: bool = field(
        default_factory=lambda: _env_bool("TRACK_MEMORY_ACCESS", True)
    )

    # Search quality thresholds
    search_score_threshold: float = field(
        default_factory=lambda: _env_float("SEARCH_SCORE_THRESHOLD", 0.30)
    )
    dedup_similarity_threshold: float = field(
        default_factory=lambda: _env_float("DEDUP_SIMILARITY_THRESHOLD", 0.4)
    )

    # Search ranking weights
    # Controls the balance between cosine similarity and importance in the
    # combined search score formula: score = similarity_weight * cosine_sim
    # + (1 - similarity_weight) * importance_weight.
    # Default 0.9 means 90% similarity, 10% importance — importance acts as
    # a tiebreaker rather than a primary ranking factor.
    search_similarity_weight: float = field(
        default_factory=lambda: _env_float("SEARCH_SIMILARITY_WEIGHT", 0.9)
    )

    # DEPRECATED: Replaced by hybrid search (BM25 sparse vectors).
    # Ignored — kept only for backward compatibility so existing configs
    # don't break. Will be removed in a future version.
    # Default changed to 0.0 so deprecation warning only fires when
    # explicitly set via env var.
    search_keyword_weight: float = field(
        default_factory=lambda: _env_float("SEARCH_KEYWORD_WEIGHT", 0.0)
    )

    # Hybrid search: BM25 sparse model for keyword matching via FastEmbed.
    # Default "Qdrant/bm25" is recommended. Can be changed to use a
    # different FastEmbed sparse model.
    search_sparse_model: str = field(
        default_factory=lambda: _env("SEARCH_SPARSE_MODEL", "Qdrant/bm25")
    )

    # Score threshold for hybrid (RRF) search results.
    # RRF score range depends on the Qdrant server's k constant:
    # with Qdrant's default k=1, scores are ~0.1-1.0 (similar to cosine);
    # with k=60, scores are ~0.01-0.03. Default 0.0 disables threshold
    # filtering for hybrid search.
    search_score_threshold_hybrid: float = field(
        default_factory=lambda: _env_float("SEARCH_SCORE_THRESHOLD_HYBRID", 0.0)
    )

    # find_memories: maximum number of search queries the LLM generates from
    # the user's message. The LLM may return fewer (or zero) based on input.
    find_memories_queries: int = field(
        default_factory=lambda: _env_int("FIND_MEMORIES_QUERIES", 5)
    )

    # find/ask pipeline: optional separate LLM model and reasoning effort.
    # Query generation and reranking are simple structured tasks that don't
    # need deep reasoning. Defaults to main LLM_MODEL with "low" reasoning.
    find_model: str = field(default_factory=lambda: _env("FIND_LLM_MODEL", ""))
    find_reasoning_effort: str | None = field(
        default_factory=lambda: _env("FIND_REASONING_EFFORT", "low") or None
    )

    # Default timezone for naive event_date values (no timezone info).
    # IANA timezone name (e.g., "UTC", "Europe/Prague", "America/New_York").
    # Empty string = use server's local timezone.
    default_timezone: str = field(default_factory=lambda: _env("DEFAULT_TIMEZONE", ""))

    # Auto-artifact threshold for remember() endpoint.
    # When the LLM recommends storing an artifact AND content exceeds this
    # length (chars), the original content is saved as an artifact.
    # Set to 0 to disable auto-artifact in remember() entirely.
    remember_artifact_threshold: int = field(
        default_factory=lambda: _env_int("REMEMBER_ARTIFACT_THRESHOLD", 4000)
    )

    # Input length cap (chars) for infer=True and remember().
    # ~100k tokens for gpt-5.4-mini, leaving 50% headroom for output/reasoning.
    max_input_length: int = field(
        default_factory=lambda: _env_int("MAX_INPUT_LENGTH", 400000)
    )

    # Session settings
    memory_session_ttl: int = field(
        default_factory=lambda: _env_int("MEMORY_SESSION_TTL", 86400)
    )
    memory_session_sweep_interval: int = field(
        default_factory=lambda: _env_int("MEMORY_SESSION_SWEEP_INTERVAL", 300)
    )

    # Session persistence backend: "sqlite" (default), "memory", "redis".
    # Auto-detection: if REDIS_URL is set and SESSION_BACKEND is not
    # explicitly set, defaults to "redis".
    session_backend: str = field(
        default_factory=lambda: (
            _env("SESSION_BACKEND") or ("redis" if _env("REDIS_URL") else "sqlite")
        )
    )

    # SQLite session DB path (only for sqlite backend)
    session_path: str = field(
        default_factory=lambda: (
            _env("SESSION_PATH") or os.path.join(_data_dir(), "sessions.db")
        )
    )

    # Redis URL for session backend (e.g., redis://localhost:6379/0)
    redis_url: str = field(default_factory=lambda: _env("REDIS_URL", ""))

    # Recall settings
    recall_max_results: int = field(
        default_factory=lambda: _env_int("RECALL_MAX_RESULTS", 10)
    )

    # Remember settings — rate limit per user per minute (0 = no limit)
    remember_rate_limit: int = field(
        default_factory=lambda: _env_int("REMEMBER_RATE_LIMIT", 10)
    )

    # Remember session context limits.
    # Max extracted memory texts kept in session (FIFO eviction).
    # Older entries are still findable via Stage 2's per-fact vector search.
    remember_max_session_memories: int = field(
        default_factory=lambda: _env_int("REMEMBER_MAX_SESSION_MEMORIES", 50)
    )

    # Conversation summary compaction threshold (chars). When the running
    # summary exceeds this, an LLM call condenses it. Set generously —
    # most sessions never hit this. 0 = disable compaction.
    remember_summary_compaction_threshold: int = field(
        default_factory=lambda: _env_int("REMEMBER_SUMMARY_COMPACTION_THRESHOLD", 10000)
    )

    # Fsck (memory check) settings — how long check results are cached (seconds)
    fsck_cache_ttl: int = field(
        default_factory=lambda: _env_int("FSCK_CACHE_TTL", 86400)
    )

    # Fsck — max concurrent LLM calls during check (1 = sequential)
    fsck_llm_concurrency: int = field(
        default_factory=lambda: _env_int("FSCK_LLM_CONCURRENCY", 4)
    )

    # Fsck — optional separate LLM model for check (empty = use main LLM_MODEL)
    fsck_model: str = field(default_factory=lambda: _env("FSCK_LLM_MODEL", ""))

    # Fsck — reasoning effort for check LLM calls (empty = use main LLM_REASONING_EFFORT).
    # Defaults to "medium" because fsck is a batch operation where accuracy
    # matters more than latency.
    fsck_reasoning_effort: str | None = field(
        default_factory=lambda: _env("FSCK_REASONING_EFFORT", "medium") or None
    )

    # Auto-fsck: periodic background maintenance (0 = disabled)
    # Interval in hours between automatic memory consistency checks.
    fsck_auto_interval: int = field(
        default_factory=lambda: _env_int("FSCK_AUTO_INTERVAL", 0)
    )

    # Auto-fsck: minimum confidence score (0.0–1.0) for a fix to be auto-applied.
    fsck_auto_min_confidence: float = field(
        default_factory=lambda: _env_float("FSCK_AUTO_MIN_CONFIDENCE", 0.95)
    )

    # Auto-fsck: minimum severity for a fix to be auto-applied.
    # Options: low, medium, high
    fsck_auto_min_severity: str = field(
        default_factory=lambda: _env("FSCK_AUTO_MIN_SEVERITY", "medium")
    )

    # Fsck — maximum memories per check run. If a user has more durable
    # memories, a random sample is taken. 0 = no limit.
    fsck_max_memories: int = field(
        default_factory=lambda: _env_int("FSCK_MAX_MEMORIES", 5000)
    )

    # Fsck — maximum LLM calls per check run. When budget is exhausted
    # the pipeline stops gracefully. Unchecked memories are picked up
    # in the next incremental run. 0 = no limit.
    fsck_max_llm_calls: int = field(
        default_factory=lambda: _env_int("FSCK_MAX_LLM_CALLS", 200)
    )

    # ── Consolidation settings ────────────────────────────────────

    # Consolidation — optional separate LLM model (empty = fall back to
    # FSCK_LLM_MODEL, then to main LLM_MODEL). Consolidation benefits from
    # a stronger model since it synthesizes durable knowledge.
    consolidation_model: str = field(
        default_factory=lambda: _env("CONSOLIDATION_LLM_MODEL", "")
    )

    # Consolidation — reasoning effort. Defaults to "low" because
    # consolidation processes pre-extracted content (not raw user input)
    # and benefits more from throughput than deep reasoning.
    consolidation_reasoning_effort: str | None = field(
        default_factory=lambda: _env("CONSOLIDATION_REASONING_EFFORT", "low") or None
    )

    # Minimum idle time (seconds) before a session is eligible for
    # within-session consolidation. Default 1 hour.
    consolidation_idle_threshold: int = field(
        default_factory=lambda: _env_int("CONSOLIDATION_IDLE_THRESHOLD", 3600)
    )

    # How often the consolidation loop checks for idle sessions (seconds).
    # Separate from idle_threshold which controls eligibility.
    consolidation_check_interval: int = field(
        default_factory=lambda: _env_int("CONSOLIDATION_CHECK_INTERVAL", 300)
    )

    # Maximum raw memories per consolidation LLM call. When a session
    # has more raw memories than this, they are split into time-based
    # batches. Each batch is consolidated independently with accumulated
    # context from prior batches.
    consolidation_batch_size: int = field(
        default_factory=lambda: _env_int("CONSOLIDATION_BATCH_SIZE", 100)
    )

    # Maximum raw memories to process per user during cross-session
    # consolidation (fsck Phase 3). Prevents unbounded LLM costs.
    consolidation_max_raw_per_user: int = field(
        default_factory=lambda: _env_int("CONSOLIDATION_MAX_RAW_PER_USER", 500)
    )

    # Maximum clusters to evaluate during cross-session consolidation.
    consolidation_max_clusters: int = field(
        default_factory=lambda: _env_int("CONSOLIDATION_MAX_CLUSTERS", 20)
    )

    # Whether client-facing tools (MCP, REST) can use infer=False.
    # When False, infer is always forced to True for client requests.
    # Internal callers (consolidation, remember) are unaffected.
    allow_client_infer: bool = field(
        default_factory=lambda: _env_bool("ALLOW_CLIENT_INFER", True)
    )

    # Days to retain superseded raw memories before garbage collection.
    # Raw memories with artifacts are always retained regardless.
    consolidation_raw_retention_days: int = field(
        default_factory=lambda: _env_int("CONSOLIDATION_RAW_RETENTION_DAYS", 30)
    )

    # Recall ranking penalties for raw memories (score adjustment).
    recall_raw_penalty: float = field(
        default_factory=lambda: _env_float("RECALL_RAW_PENALTY", 0.05)
    )
    recall_superseded_penalty: float = field(
        default_factory=lambda: _env_float("RECALL_SUPERSEDED_PENALTY", 0.15)
    )

    # Labels: client-provided key-value metadata on memories.
    # Max number of label keys per memory.
    labels_max_fields: int = field(
        default_factory=lambda: _env_int("LABELS_MAX_FIELDS", 20)
    )
    # Max length of a label key (alphanumeric + underscore only).
    labels_max_key_length: int = field(
        default_factory=lambda: _env_int("LABELS_MAX_KEY_LENGTH", 64)
    )
    # Max length of a string label value.
    labels_max_value_length: int = field(
        default_factory=lambda: _env_int("LABELS_MAX_VALUE_LENGTH", 1000)
    )
    # Comma-separated list of label keys to index in Qdrant for fast filtering.
    # Example: LABELS_INDEXES=project,topic,conversation_id
    labels_indexes: list[str] = field(
        default_factory=lambda: [
            s.strip() for s in _env("LABELS_INDEXES", "").split(",") if s.strip()
        ]
    )


@dataclass
class Config:
    """Root configuration container."""

    vector: VectorConfig = field(default_factory=VectorConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    embed: EmbedConfig = field(default_factory=EmbedConfig)
    artifact: ArtifactConfig = field(default_factory=ArtifactConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    def validate(self) -> None:
        """Validate that required configuration is present."""
        if not self.llm.api_key:
            raise ValueError("API key is required. Set LLM_API_KEY or OPENAI_API_KEY.")
        if self.artifact.backend not in ("s3", "filesystem"):
            raise ValueError(f"Unsupported ARTIFACT_BACKEND: {self.artifact.backend}")
        if self.server.instruction_mode not in (
            "passive",
            "proactive",
            "personality",
        ):
            raise ValueError(
                f"Unsupported INSTRUCTION_MODE: "
                f"{self.server.instruction_mode}. "
                "Must be one of: passive, proactive, personality"
            )
        if self.artifact.backend == "s3":
            if not self.artifact.s3_access_key or not self.artifact.s3_secret_key:
                raise ValueError(
                    "S3_ACCESS_KEY and S3_SECRET_KEY are required "
                    "when ARTIFACT_BACKEND=s3"
                )
        if self.memory.session_backend not in ("memory", "sqlite", "redis"):
            raise ValueError(
                f"Unsupported SESSION_BACKEND: {self.memory.session_backend}. "
                "Must be one of: memory, sqlite, redis"
            )
        if self.memory.session_backend == "redis" and not self.memory.redis_url:
            raise ValueError("REDIS_URL is required when SESSION_BACKEND=redis")
        if self.server.jwt_public_key and self.server.jwks_url:
            raise ValueError(
                "Configure only one JWT verifier source: "
                "MNEMORY_JWT_PUBLIC_KEY or MNEMORY_JWKS_URL"
            )
        if self.server.jwt_public_key and not os.path.isfile(
            self.server.jwt_public_key
        ):
            raise ValueError(
                f"MNEMORY_JWT_PUBLIC_KEY={self.server.jwt_public_key} does not exist"
            )


def load_config() -> Config:
    """Load and validate configuration from environment variables.

    Creates the data directory (~/.mnemory by default) if it doesn't exist.
    """
    config = Config()
    config.validate()

    # Ensure data directory exists
    data_dir = _data_dir()
    os.makedirs(data_dir, exist_ok=True)
    logger.debug("Data directory: %s", data_dir)

    return config
