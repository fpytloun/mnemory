"""Stateless HMAC-signed download tokens for artifact raw access.

Tokens are short-lived, scoped to a specific artifact, and verified
without any server-side storage. This allows secure browser-embedded
access (``<img src="...?token=...">``) without exposing long-lived
API keys in URLs.

Token format: ``base64url(json_payload).base64url(hmac_sha256_sig)``

Payload: ``{"u": user_id, "m": memory_id, "a": artifact_id, "e": expiry}``
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import secrets
import time

logger = logging.getLogger("mnemory")

# Grace period (seconds) added to expiry check to tolerate minor clock skew.
_EXPIRY_GRACE_SECONDS = 5


def derive_signing_key(
    api_key: str = "",
    api_keys: dict[str, str] | None = None,
) -> bytes:
    """Derive a signing key from the configured API key material.

    The signing key is derived via HMAC so the raw API key is never used
    directly as a signing secret. If no auth is configured, a random
    32-byte key is generated (ephemeral — tokens won't survive restarts,
    which is acceptable for short-lived tokens).

    Args:
        api_key: Single API key (``MCP_API_KEY``).
        api_keys: Multi-key mapping (``MCP_API_KEYS``).

    Returns:
        32-byte signing key.
    """
    if api_keys:
        # Deterministic: sort keys so the result is stable across restarts.
        material = "|".join(sorted(api_keys.keys()))
    elif api_key:
        material = api_key
    else:
        # No auth configured — generate a random ephemeral key.
        logger.debug("No API key configured; using ephemeral signing key")
        return secrets.token_bytes(32)

    return hmac.new(
        material.encode("utf-8"),
        b"mnemory-download-token",
        hashlib.sha256,
    ).digest()


def generate_download_token(
    signing_key: bytes,
    user_id: str,
    memory_id: str,
    artifact_id: str,
    ttl_seconds: int = 3600,
    owner_id: str | None = None,
    agent_id: str | None = None,
) -> str:
    """Generate a short-lived HMAC-signed download token.

    Args:
        signing_key: 32-byte key from :func:`derive_signing_key`.
        user_id: Owner of the artifact.
        memory_id: Memory the artifact is attached to.
        artifact_id: The artifact to grant access to.
        ttl_seconds: Token lifetime in seconds.

    Returns:
        Token string in the format ``payload_b64.signature_b64``.
    """
    payload = json.dumps(
        {
            "u": user_id,
            "m": memory_id,
            "a": artifact_id,
            "o": owner_id or user_id,
            "g": agent_id,
            "e": int(time.time()) + ttl_seconds,
        },
        separators=(",", ":"),
    )
    payload_b64 = base64.urlsafe_b64encode(payload.encode()).decode("ascii")
    sig = hmac.new(signing_key, payload_b64.encode(), hashlib.sha256).digest()
    sig_b64 = base64.urlsafe_b64encode(sig).decode("ascii")
    return f"{payload_b64}.{sig_b64}"


def validate_download_token(
    signing_key: bytes,
    token: str,
    memory_id: str,
    artifact_id: str,
) -> str | None:
    """Validate a download token and return the user_id if valid.

    Checks:
    1. Token format (payload.signature)
    2. HMAC signature integrity
    3. Expiry (with grace period)
    4. Scope: ``memory_id`` and ``artifact_id`` must match the URL path

    Args:
        signing_key: Same key used to generate the token.
        token: The token string to validate.
        memory_id: Expected memory_id from the URL path.
        artifact_id: Expected artifact_id from the URL path.

    Returns:
        The ``user_id`` from the token payload, or ``None`` if invalid.
    """
    claims = validate_download_token_claims(
        signing_key,
        token,
        memory_id,
        artifact_id,
    )
    return claims["u"] if claims else None


def validate_download_token_claims(
    signing_key: bytes,
    token: str,
    memory_id: str,
    artifact_id: str,
) -> dict[str, str | None] | None:
    """Validate a token and return its signed identity scope."""
    try:
        parts = token.split(".")
        if len(parts) != 2:
            return None

        payload_b64, sig_b64 = parts

        # Verify HMAC signature.
        expected_sig = hmac.new(
            signing_key, payload_b64.encode(), hashlib.sha256
        ).digest()
        actual_sig = base64.urlsafe_b64decode(sig_b64)
        if not hmac.compare_digest(expected_sig, actual_sig):
            return None

        # Decode and parse payload.
        payload_json = base64.urlsafe_b64decode(payload_b64).decode("utf-8")
        payload = json.loads(payload_json)

        # Check expiry.
        if time.time() > payload["e"] + _EXPIRY_GRACE_SECONDS:
            return None

        # Check scope — token must match the requested resource.
        if payload["m"] != memory_id or payload["a"] != artifact_id:
            return None

        return {
            "u": payload["u"],
            "o": payload.get("o") or payload["u"],
            "g": payload.get("g"),
        }

    except Exception:
        logger.debug("Download token validation failed", exc_info=True)
        return None
