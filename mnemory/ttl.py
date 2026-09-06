"""TTL (Time-To-Live) management for memories.

Provides pure utility functions for calculating expiration, detecting decay,
and building TTL-related metadata. Decay detection is passive/lazy — checked
at query time, not via background jobs.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mnemory.config import MemoryConfig


# Mapping from memory_type to config attribute name
_TTL_CONFIG_ATTRS: dict[str, str] = {
    "fact": "ttl_fact",
    "preference": "ttl_preference",
    "episodic": "ttl_episodic",
    "procedural": "ttl_procedural",
    "context": "ttl_context",
}


def get_default_ttl(memory_type: str | None, config: MemoryConfig) -> int | None:
    """Look up the default TTL (in days) for a memory type from config.

    Returns None (permanent) if the memory type has no default TTL configured,
    or if memory_type is None/unknown.
    """
    if not memory_type:
        return None
    attr = _TTL_CONFIG_ATTRS.get(memory_type)
    if attr is None:
        return None
    return getattr(config, attr, None)


def calculate_expiration(
    ttl_days: int | None, from_time: datetime | None = None
) -> str | None:
    """Calculate an ISO 8601 expiration timestamp from a TTL in days.

    Returns None if ttl_days is None (permanent memory).
    """
    if ttl_days is None:
        return None
    base = from_time or datetime.now(timezone.utc)
    expires = base + timedelta(days=ttl_days)
    return expires.isoformat()


def build_expiry_metadata(
    ttl_days: int | None,
    memory_type: str | None,
    config: MemoryConfig,
) -> dict:
    """Build the TTL metadata fields for a new memory.

    If ttl_days is not explicitly provided, falls back to the default TTL
    for the given memory_type from config.

    Returns a dict with keys: ttl_days, expires_at, decayed_at,
    last_accessed_at, access_count.
    """
    effective_ttl = (
        ttl_days if ttl_days is not None else get_default_ttl(memory_type, config)
    )
    return {
        "ttl_days": effective_ttl,
        "expires_at": calculate_expiration(effective_ttl),
        "decayed_at": None,
        "last_accessed_at": None,
        "access_count": 0,
    }


def is_expired(memory: dict) -> bool:
    """Check if a memory has passed its expiration time.

    A memory is NOT expired if:
    - It has no expires_at (permanent)
    - It is pinned (pinned memories are exempt from TTL)
    - Its expires_at is in the future

    The memory dict should have top-level 'metadata' containing the fields,
    or the fields can be at the top level.
    """
    metadata = memory.get("metadata") or {}
    # Pinned memories are exempt from TTL
    if metadata.get("pinned", False):
        return False
    expires_at = metadata.get("expires_at")
    if not expires_at:
        return False
    try:
        expires_dt = datetime.fromisoformat(expires_at)
        # Ensure timezone-aware comparison
        if expires_dt.tzinfo is None:
            expires_dt = expires_dt.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) >= expires_dt
    except (ValueError, TypeError):
        return False


def is_decayed(memory: dict) -> bool:
    """Check if a memory has been marked as decayed.

    A memory is decayed if its decayed_at field is set (non-null).
    """
    metadata = memory.get("metadata") or {}
    return metadata.get("decayed_at") is not None


def should_exclude(memory: dict, include_decayed: bool = False) -> bool:
    """Determine if a memory should be excluded from results.

    Exclusion rules:
    - Expired memories are excluded (unless include_decayed is True)
    - Already-decayed memories are excluded (unless include_decayed is True)
    - Pinned memories are never excluded (handled by is_expired)

    Returns True if the memory should be filtered out.
    """
    if include_decayed:
        return False
    return is_expired(memory) or is_decayed(memory)


def build_reinforcement_metadata(memory: dict) -> dict:
    """Build updated TTL metadata for access tracking (reinforcement).

    When a memory is accessed in search/recall:
    - last_accessed_at is set to now
    - access_count is incremented
    - If the memory has a TTL, expires_at is reset (reinforcement)

    Returns a dict of metadata fields to update via set_payload.
    """
    metadata = memory.get("metadata") or {}
    now = datetime.now(timezone.utc)

    updates: dict = {
        "last_accessed_at": now.isoformat(),
        "access_count": (metadata.get("access_count") or 0) + 1,
    }

    # Reset TTL if the memory has one (reinforcement)
    ttl_days = metadata.get("ttl_days")
    if ttl_days is not None:
        updates["expires_at"] = calculate_expiration(ttl_days, from_time=now)
        # Clear decayed state if memory was decayed (restore on access)
        if metadata.get("decayed_at") is not None:
            updates["decayed_at"] = None

    return updates


def apply_validation_and_decay_score(memory: dict, config: MemoryConfig) -> float:
    """Return a score with bounded validation boost and optional slow decay."""
    metadata = memory.get("metadata") or {}
    score = float(memory.get("score", 0.0))
    configured_roots = getattr(config, "validation_max_score_roots", 3)
    max_roots = max(configured_roots if isinstance(configured_roots, int) else 3, 1)
    validation_count = max(int(metadata.get("validation_count") or 0), 0)
    validation_strength = min(validation_count, max_roots) / max_roots
    validation_enabled = getattr(config, "validation_enabled", False) is True
    configured_boost = getattr(config, "validation_max_score_boost", 0.10)
    boost = configured_boost if isinstance(configured_boost, (int, float)) else 0.10
    if validation_enabled:
        score *= 1.0 + max(boost, 0.0) * validation_strength

    if getattr(config, "slow_decay_enabled", False) is not True or metadata.get(
        "pinned", False
    ):
        return score

    anchors = [
        metadata.get("revision_created_at_utc"),
        metadata.get("created_at_utc"),
        memory.get("created_at"),
        metadata.get("last_accessed_at"),
        metadata.get("last_validated_at"),
    ]
    parsed: list[datetime] = []
    for value in anchors:
        if not value:
            continue
        try:
            timestamp = datetime.fromisoformat(str(value))
        except (TypeError, ValueError):
            continue
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        parsed.append(timestamp)
    if not parsed:
        return score

    age_days = max((datetime.now(timezone.utc) - max(parsed)).total_seconds(), 0.0)
    age_days /= 86400.0
    configured_half_life = getattr(config, "slow_decay_half_life_days", 30.0)
    base_half_life = max(
        float(configured_half_life)
        if isinstance(configured_half_life, (int, float))
        else 30.0,
        0.001,
    )
    configured_multiplier = getattr(
        config, "slow_decay_validation_half_life_multiplier", 2.0
    )
    validation_multiplier = max(
        float(configured_multiplier)
        if isinstance(configured_multiplier, (int, float))
        else 2.0,
        1.0,
    )
    half_life = base_half_life * (
        1.0 + (validation_multiplier - 1.0) * validation_strength
    )
    configured_floor = getattr(config, "slow_decay_score_floor", 0.25)
    floor = min(
        max(
            float(configured_floor)
            if isinstance(configured_floor, (int, float))
            else 0.25,
            0.0,
        ),
        1.0,
    )
    decay_factor = max(floor, math.pow(2.0, -age_days / half_life))
    return score * decay_factor
