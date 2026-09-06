"""Pydantic request/response models for the REST API.

These models auto-generate the OpenAPI spec that clients (Open WebUI,
Cursor, etc.) consume as native tools.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from mnemory.sanitize import escape_memory_headers


class EvidenceActor(BaseModel):
    """Authenticated actor supplied by Cognis."""

    model_config = {"extra": "forbid"}

    user_id: str = Field(..., min_length=1, max_length=256)
    owner_id: str = Field(..., min_length=1, max_length=256)


class EvidenceEvent(BaseModel):
    """Stable event identity supplied by Cognis."""

    model_config = {"extra": "forbid"}

    id: str = Field(..., min_length=1, max_length=256)
    event_hash: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    cognis_session_id: str = Field(..., min_length=1, max_length=256)
    conversation_id: str = Field(..., min_length=1, max_length=256)
    turn_id: str = Field(..., min_length=1, max_length=256)


class EvidenceMessage(BaseModel):
    """The single user message attached to an evidence event."""

    model_config = {"extra": "forbid"}

    role: Literal["user"]
    content: str = Field(..., min_length=1, max_length=400_000)


class EvidenceRememberRequest(BaseModel):
    """Strict body contract for the trusted evidence endpoint."""

    model_config = {"extra": "forbid"}

    version: Literal[1]
    actor: EvidenceActor
    event: EvidenceEvent
    messages: list[EvidenceMessage] = Field(..., min_length=1, max_length=1)

    @field_validator("messages")
    @classmethod
    def validate_message(cls, value: list[EvidenceMessage]) -> list[EvidenceMessage]:
        """Reject a structurally present but empty user message."""
        if not value or not value[0].content.strip():
            raise ValueError("Evidence message content must be non-empty")
        return value


# ── Memory CRUD ───────────────────────────────────────────────────────


class AddMemoryRequest(BaseModel):
    """Request to add a single memory."""

    content: str = Field(
        ...,
        max_length=400_000,
        description="Memory content (max 1000 chars for infer=False)",
    )
    memory_type: str | None = Field(
        None, description="Memory type: preference, fact, episodic, procedural, context"
    )
    categories: list[str] | None = Field(
        None, description="Category tags from predefined set"
    )
    importance: str | None = Field(
        None, description="Importance: low, normal, high, critical"
    )
    pinned: bool | None = Field(
        None, description="Pin to core memories (loaded at conversation start)"
    )
    infer: bool = Field(
        True,
        description="Use LLM for fact extraction and dedup (True) or store as-is (False)",
    )
    role: Literal["user", "assistant"] = Field(
        "user", description="Who this memory is about: 'user' or 'assistant'"
    )
    ttl_days: int | None = Field(
        None, description="Time-to-live in days (None = use type default)"
    )
    event_date: str | None = Field(
        None, description="ISO 8601 datetime for when the event occurred"
    )
    labels: dict[str, Any] | None = Field(
        None, description="Key-value labels for client-specific metadata"
    )


class BatchMemoryItem(BaseModel):
    """Single item in a batch add request."""

    content: str = Field(..., max_length=400_000)
    memory_type: str | None = None
    categories: list[str] | None = None
    importance: str | None = None
    pinned: bool | None = None
    infer: bool = True
    role: Literal["user", "assistant"] = "user"
    ttl_days: int | None = None
    event_date: str | None = None
    labels: dict[str, Any] | None = None


class AddMemoriesBatchRequest(BaseModel):
    """Request to batch-add multiple memories."""

    memories: list[BatchMemoryItem] = Field(..., description="List of memories to add")


class DeleteMemoriesBatchRequest(BaseModel):
    """Request to batch-delete multiple memories."""

    memory_ids: list[str] = Field(
        ..., min_length=1, max_length=20, description="List of memory IDs to delete"
    )


class MemoryActionResult(BaseModel):
    """Result of one inferred ADD, UPDATE, CONFIRM, or SKIP action."""

    id: str | None = None
    memory: str
    event: str
    operation_id: str | None = None
    lineage_id: str | None = None
    revision: int | None = None
    replayed: bool = False


class ArtifactInfo(BaseModel):
    """Info about an auto-created artifact."""

    id: str
    filename: str
    size: int
    linked_memories: int


class AddMemoryResponse(BaseModel):
    """Response from adding memory/memories."""

    results: list[MemoryActionResult] = Field(default_factory=list)
    artifact: ArtifactInfo | None = None
    error: bool = False
    message: str | None = None


class SearchMemoriesRequest(BaseModel):
    """Request for semantic memory search."""

    query: str = Field(
        ..., max_length=10_000, description="Search query (natural language)"
    )
    memory_type: str | None = Field(None, description="Filter by memory type")
    categories: list[str] | None = Field(None, description="Filter by categories")
    role: str | None = Field(None, description="Filter by role: 'user' or 'assistant'")
    limit: int = Field(10, ge=1, le=100, description="Max results to return")
    include_decayed: bool = Field(False, description="Include expired/decayed memories")
    date_start: str | None = Field(
        None,
        description=(
            "Filter by date range start (YYYY-MM-DD). Matches event_date "
            "or created_at when event_date is not set."
        ),
    )
    date_end: str | None = Field(
        None,
        description=(
            "Filter by date range end (YYYY-MM-DD). Matches event_date "
            "or created_at when event_date is not set."
        ),
    )
    labels: dict[str, Any] | None = Field(
        None, description="Filter by label key-value pairs (AND logic)"
    )


class FindMemoriesRequest(BaseModel):
    """Request for AI-powered multi-query search."""

    question: str = Field(
        ..., max_length=10_000, description="Natural language question"
    )
    memory_type: str | None = Field(None, description="Filter by memory type")
    categories: list[str] | None = Field(None, description="Filter by categories")
    role: str | None = Field(None, description="Filter by role")
    limit: int = Field(10, ge=1, le=100, description="Max results to return")
    include_decayed: bool = Field(False, description="Include expired/decayed memories")
    labels: dict[str, Any] | None = Field(
        None, description="Filter by label key-value pairs (AND logic)"
    )
    context: str | None = Field(
        None,
        max_length=10_000,
        description=(
            "Optional context hint for query generation (e.g., current working "
            "directory, active project). Used to generate additional relevant "
            "queries — does not filter results exclusively to this context."
        ),
    )


class AskMemoriesRequest(BaseModel):
    """Request to ask a question and get a human-readable answer."""

    question: str = Field(
        ..., max_length=10_000, description="Natural language question"
    )
    memory_type: str | None = Field(None, description="Filter by memory type")
    categories: list[str] | None = Field(None, description="Filter by categories")
    role: str | None = Field(None, description="Filter by role")
    limit: int = Field(10, ge=1, le=100, description="Max supporting memories to use")
    include_decayed: bool = Field(False, description="Include expired/decayed memories")
    context: str | None = Field(
        None,
        max_length=10_000,
        description="Optional context hint for query generation",
    )
    include_memories: bool = Field(
        False,
        description=(
            "Include supporting memories in the response. Default false — "
            "returns only the answer text, queries, and stats."
        ),
    )
    labels: dict[str, Any] | None = Field(
        None, description="Filter by label key-value pairs (AND logic)"
    )


class MemoryItem(BaseModel):
    """A memory item in search/list results."""

    id: str
    memory: str
    score: float | None = None
    metadata: dict | None = None
    has_artifacts: bool = False


class AskMemoriesResponse(BaseModel):
    """Response with a human-readable answer and supporting memories."""

    answer: str = Field(..., description="Human-readable answer based on memories")
    results: list["MemoryItem"] = Field(
        default_factory=list,
        description="Supporting memories (when include_memories=true)",
    )
    count: int = Field(0, description="Number of memories used to generate the answer")
    queries: list[str] = Field(
        default_factory=list, description="Generated search queries"
    )
    stats: dict = Field(default_factory=dict, description="Search statistics")


class SearchMemoriesResponse(BaseModel):
    """Response from memory search."""

    results: list[MemoryItem]


class ListMemoriesResponse(BaseModel):
    """Response from listing memories."""

    results: list[MemoryItem]
    next_cursor: str | None = None
    has_more: bool = False


class HistoryRevisionItem(BaseModel):
    """UI-safe immutable revision."""

    id: str
    memory: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    has_artifacts: bool = False
    predecessor_id: str | None = None
    successor_id: str | None = None


class HistoryOperationItem(BaseModel):
    """UI-safe audit event without internal retry or transition data."""

    operation_id: str | None = None
    operation_kind: str
    status: str
    actor_kind: str | None = None
    source_kind: str | None = None
    reason: str | None = None
    created_at_utc: str | None = None
    completed_at_utc: str | None = None


class MemoryHistoryResponse(BaseModel):
    """Authorized immutable lineage history."""

    lineage_id: str
    current_revision_id: str | None = None
    revisions: list[HistoryRevisionItem] = Field(default_factory=list)
    operations: list[HistoryOperationItem] = Field(default_factory=list)
    next_revision_cursor: str | None = None
    next_operation_cursor: str | None = None


class MemoryLinkItem(BaseModel):
    """UI-safe authorized related revision."""

    id: str
    memory: str
    revision: int
    revision_state: str
    memory_type: str | None = None
    memory_layer: str
    has_artifacts: bool = False


class MemoryLinksResponse(BaseModel):
    """Authorized provenance and supersession relationships."""

    revision_id: str
    lineage_id: str
    supersedes: MemoryLinkItem | None = None
    successor: MemoryLinkItem | None = None
    derived_from: list[MemoryLinkItem] = Field(default_factory=list)
    derived_outputs: list[MemoryLinkItem] = Field(default_factory=list)
    provenance_quality: str = "exact"


class GetMemoriesByIdsRequest(BaseModel):
    """Request to fetch multiple memories by their IDs."""

    ids: list[str] = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Memory IDs to fetch (max 500)",
    )


class GetMemoriesByIdsResponse(BaseModel):
    """Response with memories fetched by IDs."""

    results: list[MemoryItem]


class RecentMemoriesResponse(BaseModel):
    """Response from get_recent_memories (formatted text)."""

    text: str = Field(..., description="Formatted recent memories text")


class UpdateMemoryRequest(BaseModel):
    """Request to update an existing memory."""

    content: str | None = Field(None, max_length=1_000, description="New content text")
    memory_type: str | None = Field(None, description="New memory type")
    categories: list[str] | None = Field(
        None, description="New categories (replaces existing)"
    )
    importance: str | None = Field(None, description="New importance level")
    pinned: bool | None = Field(None, description="New pinned state")
    ttl_days: int | None = Field(None, description="New TTL in days")
    event_date: str | None = Field(
        None,
        description=(
            "New event date (ISO 8601, e.g., '2023-05-08'). Pass null to clear."
        ),
    )
    labels: dict[str, Any] | None = Field(
        None,
        description="Updated labels (empty dict clears, omit to keep existing)",
    )
    agent_id: str | None = Field(
        None,
        description=(
            "New agent_id. Non-empty string to set, empty string to clear. "
            "Null (omitted) means no change. When session has an agent_id, "
            "only own agent_id or clearing is allowed."
        ),
    )
    expected_revision: int | None = Field(
        None,
        ge=1,
        description="Expected active revision. A mismatch returns HTTP 409.",
    )
    idempotency_key: str | None = Field(
        None,
        min_length=1,
        max_length=256,
        description="Retry key scoped to this lineage and update operation.",
    )


class CoreMemoriesStatsResponse(BaseModel):
    """Structured stats describing the assembled core context."""

    memory_count: int = Field(..., description="Number of included memories")
    char_count: int = Field(..., description="Character count of the formatted text")
    estimated_tokens: int = Field(
        ..., description="Approximate token count estimated from text length"
    )
    by_type: dict[str, int] = Field(
        default_factory=dict, description="Counts by memory type"
    )
    by_role: dict[str, int] = Field(default_factory=dict, description="Counts by role")
    by_section: dict[str, int] = Field(
        default_factory=dict, description="Counts by stable section key"
    )
    section_labels: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of stable section keys to display labels",
    )
    sections: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Ordered memory IDs grouped by stable section key",
    )
    memory_ids: list[str] = Field(
        default_factory=list,
        description="Ordered flattened memory IDs included in the core text",
    )


class CoreMemoriesResponse(BaseModel):
    """Response from get_core_memories."""

    text: str = Field(..., description="Formatted core memories text")
    stats: CoreMemoriesStatsResponse | None = Field(
        None, description="Optional structured stats for the assembled core memories"
    )


class RecentMemoriesRequest(BaseModel):
    """Query parameters for recent memories (used as query params)."""

    days: int = Field(7, description="How many days back to look")
    scope: str = Field("all", description="Scope: 'all', 'user', or 'agent'")
    limit: int = Field(25, ge=1, le=100, description="Max results per scope")
    include_decayed: bool = Field(False, description="Include expired memories")


class ListMemoriesRequest(BaseModel):
    """Query parameters for listing memories."""

    memory_type: str | None = Field(None, description="Filter by type")
    categories: list[str] | None = Field(None, description="Filter by categories")
    role: str | None = Field(None, description="Filter by role")
    limit: int = Field(50, ge=1, le=500, description="Max results")
    include_decayed: bool = Field(False, description="Include expired memories")
    labels: dict[str, Any] | None = Field(
        None, description="Filter by label key-value pairs (AND logic)"
    )


# ── Artifacts ─────────────────────────────────────────────────────────


class SaveArtifactRequest(BaseModel):
    """Request to save an artifact attached to a memory."""

    # 10MB binary → ~13.3MB base64; 14M chars gives headroom.
    # Actual byte limit is enforced by ArtifactStore after decoding.
    content: str = Field(
        ...,
        max_length=14_000_000,
        description="Text or base64-encoded binary content (up to ~10MB)",
    )
    filename: str = Field("note.md", description="Artifact filename")
    content_type: str = Field("text/markdown", description="MIME type")


class ArtifactMetadataResponse(BaseModel):
    """Artifact metadata in responses."""

    id: str
    filename: str
    content_type: str
    size: int
    created_at: str


class GetArtifactResponse(BaseModel):
    """Response from getting artifact content."""

    content: str
    total_size: int
    has_more: bool


# ── Categories ────────────────────────────────────────────────────────


class CategoryItem(BaseModel):
    """A category with description and count."""

    name: str
    description: str
    count: int


class ListCategoriesResponse(BaseModel):
    """Response from listing categories."""

    categories: list[CategoryItem]


# ── Intelligence Layer ────────────────────────────────────────────────


# Valid message roles for OpenAI chat format.
_VALID_MESSAGE_ROLES = {"user", "assistant", "system", "tool"}


class MessageParam(BaseModel):
    """A single message in OpenAI chat format.

    Extra fields are silently ignored to stay forward-compatible with
    extended message formats (e.g., tool_calls, name, function_call).
    Role is validated against an allowlist to prevent role spoofing.
    """

    model_config = {"extra": "allow"}

    role: str = Field(..., description="Message role: user, assistant, system, tool")
    content: str | None = Field(
        None,
        max_length=100_000,
        description="Message text content (may be null for tool messages)",
    )

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        """Validate role against allowlist to prevent role spoofing."""
        normalized = v.strip().lower()
        if normalized not in _VALID_MESSAGE_ROLES:
            raise ValueError(
                f"Invalid message role '{v}'. "
                f"Valid roles: {', '.join(sorted(_VALID_MESSAGE_ROLES))}"
            )
        return normalized


class RecallRequest(BaseModel):
    """Request for the recall endpoint."""

    session_id: str | None = Field(
        None, description="Session ID from previous call. Null = first call."
    )
    query: str | None = Field(
        None, max_length=10_000, description="Free text search query"
    )
    messages: list[MessageParam] | None = Field(
        None,
        description="OpenAI-format messages. Last user message used as query if query not provided.",
    )
    include_instructions: bool = Field(
        False, description="Include behavioral instructions in response"
    )
    managed: bool = Field(
        False, description="Use managed-mode instructions (plugin-driven)"
    )
    instruction_mode: str | None = Field(
        None, description="Override instruction mode (passive/proactive/personality)"
    )
    search_mode: str | None = Field(
        None,
        description=(
            "Search mode: 'find' (AI-powered, up to N queries, default) "
            "or 'search' (fast single vector search, no LLM). "
            "Applies to every call — the client decides per-request."
        ),
    )
    score_threshold: float | None = Field(
        None,
        description=(
            "Minimum score for search results (0.0-1.0). Results below "
            "this are filtered out. If not set, only the server's default "
            "SEARCH_SCORE_THRESHOLD applies."
        ),
    )
    recent_days: int = Field(7, description="Days of recent context for core memories")
    ttl: int | None = Field(
        None, description="Session TTL in seconds (only used on first call)"
    )
    context: str | None = Field(
        None,
        max_length=10_000,
        description=(
            "Optional context hint for query generation (e.g., current working "
            "directory, active project). Used to generate additional relevant "
            "queries — does not filter results exclusively to this context."
        ),
    )
    labels: dict[str, Any] | None = Field(
        None, description="Filter search results by label key-value pairs"
    )


class RecallStats(BaseModel):
    """Statistics from a recall operation."""

    core_count: int = 0
    search_count: int = 0
    new_count: int = 0
    known_skipped: int = 0
    latency_ms: int = 0


class RecallResponse(BaseModel):
    """Response from the recall endpoint."""

    session_id: str
    instructions: str | None = None
    core_memories: str | None = None
    search_results: list[MemoryItem] = Field(default_factory=list)
    stats: RecallStats = Field(default_factory=RecallStats)


class RememberRequest(BaseModel):
    """Request for the remember endpoint."""

    session_id: str | None = Field(
        None, description="Session ID to update known memory IDs"
    )
    messages: list[MessageParam] = Field(
        ..., description="OpenAI-format messages (typically last 2: user + assistant)"
    )
    context: str | None = Field(
        None,
        max_length=10_000,
        description=(
            "Optional context hint (e.g., current working directory, active "
            "project). Passed to the extraction pipeline to help identify "
            "the project and produce self-contained memories."
        ),
    )
    role: Literal["user", "assistant"] | None = Field(
        None,
        description=(
            "Controls the extraction point of view. "
            "null (default): auto mode — extracts facts from ALL "
            "participants (user and assistant), each stored with its "
            "own role. 'user': extracts only user facts, suppresses "
            "assistant content. 'assistant': extracts only assistant "
            "facts (requires agent_id via X-Agent-Id header)."
        ),
    )
    labels: dict[str, Any] | None = Field(
        None, description="Labels to attach to all extracted memories"
    )


class RememberResponse(BaseModel):
    """Response from the remember endpoint (immediate)."""

    accepted: bool = True


# ── Fsck (Memory Check) ───────────────────────────────────────────────


class FsckRequest(BaseModel):
    """Request to start a memory consistency check."""

    agent_id: str | None = Field(
        None, description="Optional: scope check to specific agent"
    )
    categories: list[str] | None = Field(
        None, description="Optional: scope check to specific categories"
    )
    memory_type: str | None = Field(
        None, description="Optional: scope check to specific memory type"
    )
    include_raw: bool = Field(
        False,
        description=(
            "Whether to include raw provisional memories in the check. "
            "Defaults to false so fsck focuses on durable memories."
        ),
    )
    incremental: bool = Field(
        False,
        description=(
            "When true, only check memories changed since last maintenance. "
            "Defaults to false (full scan)."
        ),
    )


class FsckAuditTarget(BaseModel):
    """One exact revision requested for an audit-only fsck."""

    model_config = {"extra": "forbid"}

    memory_id: str = Field(..., min_length=1, max_length=128)
    revision: int = Field(..., ge=1)


class FsckAuditRequest(BaseModel):
    """Request a bounded, operation-bound, mutation-free fsck audit."""

    model_config = {"extra": "forbid"}

    basis_operation_id: str = Field(..., min_length=1, max_length=128)
    targets: list[FsckAuditTarget] = Field(..., min_length=1, max_length=20)

    @field_validator("targets")
    @classmethod
    def targets_are_unique(cls, value: list[FsckAuditTarget]):
        """Reject duplicate IDs instead of silently changing audit scope."""
        ids = [target.memory_id for target in value]
        if len(ids) != len(set(ids)):
            raise ValueError("Audit target memory IDs must be unique")
        return value


class FsckStartResponse(BaseModel):
    """Response from starting a memory check."""

    check_id: str
    status: str = "running"


class FsckAction(BaseModel):
    """A single action to fix an issue."""

    action: str = Field(..., description="Action type: update, delete, add")
    memory_id: str | None = Field(
        None, description="ID of memory to update/delete, null for add"
    )
    new_content: str | None = Field(
        None, description="New text for update/add, null for delete"
    )
    new_metadata: dict | None = Field(
        None, description="Metadata corrections, null if no metadata changes"
    )


class FsckAffectedMemory(BaseModel):
    """A memory affected by an issue."""

    id: str
    content: str
    metadata: dict | None = None
    agent_id: str | None = None


class FsckIssue(BaseModel):
    """A single issue found during memory check."""

    issue_id: str
    type: str = Field(
        ...,
        description="Issue type: duplicate, quality, split, contradiction, reclassify, security",
    )
    severity: str = Field(..., description="Severity: low, medium, high")
    confidence: float | None = Field(
        None, description="LLM confidence score 0.0-1.0, null for regex-detected issues"
    )
    reasoning: str = Field(
        ..., description="Explanation of the issue and suggested fix"
    )
    affected_memories: list[FsckAffectedMemory]
    actions: list[FsckAction]
    applied: bool = Field(
        False, description="Whether this issue's fixes have already been applied"
    )


class FsckProgress(BaseModel):
    """Progress of a running memory check."""

    phase: str = Field(
        ...,
        description="Current phase: security_scan, duplicate_search, duplicate_eval, quality_check, done",
    )
    total_memories: int = 0
    processed: int = 0
    percent: int = 0
    issues_found: int = Field(0, description="Number of issues found so far")


class FsckSummary(BaseModel):
    """Summary of issues found."""

    duplicate: int = 0
    quality: int = 0
    split: int = 0
    contradiction: int = 0
    reclassify: int = 0
    security: int = 0
    total: int = 0


class FsckStatusResponse(BaseModel):
    """Response from polling a memory check status."""

    check_id: str
    status: str = Field(..., description="Status: running, applying, completed, failed")
    progress: FsckProgress
    summary: FsckSummary | None = None
    issues: list[FsckIssue] | None = None
    error: str | None = None
    created_at: str | None = None
    expires_at: str | None = None
    mode: Literal["scan", "exact_audit"] = "scan"
    target_state: Literal["complete", "absent", "stale"] | None = None


class FsckApplyRequest(BaseModel):
    """Request to apply fixes from a completed check."""

    issue_ids: list[str] | None = Field(
        None, description="IDs of issues to apply. Null or empty = apply all."
    )


class FsckApplyDetail(BaseModel):
    """Result of applying a single issue."""

    issue_id: str
    status: str = Field(..., description="Status: applied, skipped, failed")
    actions_executed: int = 0
    actions_skipped: int = 0
    error: str | None = None


class FsckApplyResponse(BaseModel):
    """Response from applying fixes."""

    applied: int = 0
    skipped: int = 0
    failed: int = 0
    details: list[FsckApplyDetail] = Field(default_factory=list)


class FsckReEvaluateRequest(BaseModel):
    """Compare an old fsck operation with a separately generated check."""

    fresh_check_id: str = Field(..., min_length=1)
    terminalize: bool = False


class FsckReEvaluateResponse(BaseModel):
    """Non-content result of fsck operation re-evaluation."""

    operation_id: str
    outcome: str
    terminalized: bool
    old_action_count: int
    fresh_action_count: int
    stale_target_count: int
    fresh_check_id: str
    fresh_issue_id: str | None = None


class FailedSessionRetryRequest(BaseModel):
    """Bounded explicit retry request for failed sessions."""

    session_ids: list[str] = Field(..., min_length=1, max_length=10)
    dry_run: bool = True
    idempotency_key: str | None = Field(None, min_length=1, max_length=256)
    stop_on_failure: bool = True


# ── Download Tokens ───────────────────────────────────────────────────


class DownloadTokenRequest(BaseModel):
    """Request to generate a short-lived download token for an artifact."""

    ttl: int | None = Field(
        None,
        description=(
            "Token lifetime in seconds. Default and max are server-configured "
            "via DOWNLOAD_TOKEN_TTL and DOWNLOAD_TOKEN_MAX_TTL."
        ),
        ge=1,
    )


class DownloadTokenResponse(BaseModel):
    """Response with a signed download token and URL."""

    token: str = Field(..., description="HMAC-signed download token")
    url: str = Field(..., description="Full URL with token for direct download")
    expires_in: int = Field(..., description="Token lifetime in seconds")
    content_type: str = Field(..., description="Artifact MIME type")
    filename: str = Field(..., description="Artifact filename")
    size: int = Field(..., description="Artifact size in bytes")


# ── Common ────────────────────────────────────────────────────────────


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: bool = True
    message: str


# ── Helpers ───────────────────────────────────────────────────────────


def format_memory_item(item: dict) -> MemoryItem:
    """Convert a MemoryService search result dict to a MemoryItem.

    Used by both CRUD and intelligence endpoints to normalize
    search/find results into the API response format.

    Memory text is escaped to prevent markdown header forgery
    when results are injected into LLM context by plugins.

    Note: _point_to_memory() promotes agent_id to a top-level field
    and excludes it from metadata. We inject it back so the API
    response (and UI) can access it via metadata.agent_id.
    """
    metadata = dict(item.get("metadata") or {})  # copy to avoid mutation
    # Inject agent_id into metadata if present at top level but missing
    agent_id = item.get("agent_id")
    if agent_id and "agent_id" not in metadata:
        metadata["agent_id"] = agent_id
    has_artifacts = bool(metadata.get("artifacts"))
    raw_text = item.get("memory", item.get("text", ""))
    return MemoryItem(
        id=item["id"],
        memory=escape_memory_headers(raw_text),
        score=item.get("score"),
        metadata=metadata if metadata else None,
        has_artifacts=has_artifacts,
    )
