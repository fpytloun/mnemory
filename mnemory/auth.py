"""JWT validation helpers for Cognis service authentication.

Supports local PEM validation and remote JWKS validation for Cognis-issued
ES256 service tokens. JWT validation is optional and additive — callers can
fall back to existing API key authentication when validation fails.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any

import jwt
from jwt import InvalidTokenError, PyJWKClient

logger = logging.getLogger("mnemory")

_EXPECTED_ISSUER = "cognis"
_EXPECTED_AUDIENCE = "mnemory"
_JWKS_CACHE_TTL_SECONDS = 300
_JWKS_TIMEOUT_SECONDS = 5
_EVIDENCE_SKEW_SECONDS = 5
_EVIDENCE_MAX_LIFETIME_SECONDS = 60


@dataclass(frozen=True)
class JWTAuthContext:
    """Validated Cognis JWT identity."""

    user_id: str
    agent_id: str | None = None
    owner_id: str | None = None
    token_type: str | None = None
    scopes: frozenset[str] = frozenset()


class CognisJWTValidator:
    """Validate Cognis-issued service JWTs for Mnemory."""

    def __init__(self, *, public_key_path: str = "", jwks_url: str = "") -> None:
        self._public_key_path = public_key_path
        self._jwks_url = jwks_url
        self._public_key: str | None = None
        self._jwks_client: PyJWKClient | None = None

        if self._public_key_path:
            with open(self._public_key_path, encoding="utf-8") as f:
                self._public_key = f.read()

    @property
    def enabled(self) -> bool:
        """Whether JWT validation is configured."""
        return bool(self._public_key_path or self._jwks_url)

    def validate(
        self,
        token: str,
        header_agent_id: str | None = None,
        header_owner_id: str | None = None,
    ) -> JWTAuthContext:
        """Validate a Cognis JWT and resolve the effective agent_id.

        Args:
            token: Bearer token value.
            header_agent_id: Optional X-Agent-Id header fallback.

        Returns:
            Validated auth context.

        Raises:
            InvalidTokenError: If the JWT is invalid.
        """
        claims = self._decode(token)
        user_id = claims.get("sub")
        if not isinstance(user_id, str) or not user_id.strip():
            raise InvalidTokenError("JWT is missing a valid sub claim")

        claim_agent_id = claims.get("agent_id")
        if claim_agent_id is not None and not isinstance(claim_agent_id, str):
            raise InvalidTokenError("JWT agent_id claim must be a string when present")
        claim_owner_id = claims.get("aow")
        if claim_owner_id is not None and not isinstance(claim_owner_id, str):
            raise InvalidTokenError("JWT aow claim must be a string when present")
        token_type = claims.get("typ")
        if token_type is not None and not isinstance(token_type, str):
            raise InvalidTokenError("JWT typ claim must be a string when present")
        raw_scope = claims.get("scope", "")
        if isinstance(raw_scope, str):
            scopes = frozenset(raw_scope.split())
        elif isinstance(raw_scope, list) and all(
            isinstance(item, str) for item in raw_scope
        ):
            scopes = frozenset(raw_scope)
        else:
            raise InvalidTokenError("JWT scope claim must be a string or string list")

        if claim_agent_id and header_agent_id and claim_agent_id != header_agent_id:
            raise InvalidTokenError("X-Agent-Id does not match JWT agent_id claim")
        if claim_owner_id and header_owner_id and claim_owner_id != header_owner_id:
            raise InvalidTokenError("X-Agent-Owner does not match JWT aow claim")

        return JWTAuthContext(
            user_id=user_id,
            agent_id=claim_agent_id or header_agent_id,
            owner_id=claim_owner_id or header_owner_id or user_id,
            token_type=token_type,
            scopes=scopes,
        )

    def decode_claims(self, token: str) -> dict:
        """Decode and validate a JWT, returning all claims.

        Unlike ``validate()``, this returns the raw claims dict including
        custom claims like ``typ``, ``target``, and ``jti``. Used by the
        exchange token endpoint which needs access to these claims.

        Raises:
            InvalidTokenError: If the JWT is invalid, expired, or has
                wrong issuer/audience.
        """
        return self._decode(token)

    def validate_evidence(
        self, token: str, *, now: float | None = None
    ) -> dict[str, Any]:
        """Validate the dedicated, non-fallback Cognis evidence token."""
        header = jwt.get_unverified_header(token)
        if header.get("alg") != "ES256":
            raise InvalidTokenError("Invalid evidence token header")
        claims = self._decode_evidence(token)
        stable_strings = (
            "typ",
            "scope",
            "evop",
            "sub",
            "evt",
            "event_hash",
            "request_hash",
            "evidence_root",
            "cognis_session_id",
            "conversation_id",
            "turn_id",
            "jti",
        )
        for name in stable_strings:
            value = claims.get(name)
            if not isinstance(value, str) or not value.strip():
                raise InvalidTokenError(f"Evidence token is missing {name}")
        for name in (
            "sub",
            "evt",
            "cognis_session_id",
            "conversation_id",
            "turn_id",
            "jti",
        ):
            if len(claims[name]) > 256:
                raise InvalidTokenError(f"Evidence {name} exceeds its size limit")
        if claims["typ"] != "user_event":
            raise InvalidTokenError("Invalid evidence token type")
        if claims["scope"] != "mnemory:evidence":
            raise InvalidTokenError("Invalid evidence scope")
        if claims["evop"] != "remember" or claims.get("ver") != 1:
            raise InvalidTokenError("Invalid evidence operation")
        if not isinstance(claims.get("ver"), int) or isinstance(claims["ver"], bool):
            raise InvalidTokenError("Invalid evidence version")
        if not isinstance(claims.get("aow"), str) or not claims["aow"].strip():
            raise InvalidTokenError("Evidence token requires non-empty aow")
        if "agent_id" in claims:
            raise InvalidTokenError("Evidence tokens cannot contain agent_id")
        for name in ("event_hash", "request_hash", "evidence_root"):
            if not re.fullmatch(r"[0-9a-f]{64}", claims[name]):
                raise InvalidTokenError(f"Evidence {name} must be lowercase SHA-256")
        for name in ("iat", "nbf", "exp"):
            value = claims.get(name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise InvalidTokenError(f"Evidence token requires numeric {name}")
        current = time.time() if now is None else now
        if claims["exp"] - claims["iat"] > _EVIDENCE_MAX_LIFETIME_SECONDS:
            raise InvalidTokenError("Evidence token lifetime exceeds 60 seconds")
        if claims["nbf"] > claims["iat"]:
            raise InvalidTokenError("Evidence nbf cannot follow iat")
        if claims["iat"] > current + _EVIDENCE_SKEW_SECONDS:
            raise InvalidTokenError("Evidence token iat is in the future")
        if claims["exp"] < current - _EVIDENCE_SKEW_SECONDS:
            raise InvalidTokenError("Evidence token is expired")
        if "nbf" in claims and claims["nbf"] > current + _EVIDENCE_SKEW_SECONDS:
            raise InvalidTokenError("Evidence token is not yet valid")
        return claims

    def is_evidence_intent(self, token: str) -> bool:
        """Classify a validly signed evidence-capable token before fallback."""
        try:
            claims = self._decode_signature_only(token)
        except InvalidTokenError:
            return False
        return (
            claims.get("typ") == "user_event"
            or claims.get("scope") == "mnemory:evidence"
            or claims.get("evop") == "remember"
        )

    def _decode(self, token: str) -> dict:
        if self._public_key is not None:
            return self._decode_with_key(token, self._public_key)

        last_error: Exception | None = None
        for force_refresh in (False, True):
            try:
                key = (
                    self._get_jwks_client(force_refresh)
                    .get_signing_key_from_jwt(token)
                    .key
                )
                return self._decode_with_key(token, key)
            except InvalidTokenError as e:
                last_error = e
                if not force_refresh:
                    logger.info(
                        "JWT validation failed with cached JWKS key; refreshing"
                    )
                    continue
                raise
            except Exception as e:  # pragma: no cover - defensive wrapper
                last_error = e
                if not force_refresh:
                    logger.info("JWKS lookup failed; refreshing JWKS cache")
                    continue
                raise InvalidTokenError("Failed to resolve JWT signing key") from e

        raise InvalidTokenError("JWT validation failed") from last_error

    def _decode_evidence(self, token: str) -> dict:
        """Decode an evidence token with the dedicated five-second skew."""
        if self._public_key is not None:
            return self._decode_with_key(
                token,
                self._public_key,
                leeway=5,
                required=("iss", "aud", "iat", "nbf", "exp"),
            )
        try:
            key = self._get_jwks_client(False).get_signing_key_from_jwt(token).key
            return self._decode_with_key(
                token,
                key,
                leeway=5,
                required=("iss", "aud", "iat", "nbf", "exp"),
            )
        except InvalidTokenError:
            raise
        except Exception as exc:
            raise InvalidTokenError("Failed to resolve JWT signing key") from exc

    def _decode_signature_only(self, token: str) -> dict:
        """Verify only ES256 signature for pre-fallback intent classification."""
        if self._public_key is not None:
            return jwt.decode(
                token,
                self._public_key,
                algorithms=["ES256"],
                options={
                    "verify_exp": False,
                    "verify_nbf": False,
                    "verify_aud": False,
                    "verify_iss": False,
                },
            )
        last_error: Exception | None = None
        for force_refresh in (False, True):
            try:
                key = (
                    self._get_jwks_client(force_refresh)
                    .get_signing_key_from_jwt(token)
                    .key
                )
                return jwt.decode(
                    token,
                    key,
                    algorithms=["ES256"],
                    options={
                        "verify_exp": False,
                        "verify_nbf": False,
                        "verify_aud": False,
                        "verify_iss": False,
                    },
                )
            except Exception as exc:
                last_error = exc
                if force_refresh:
                    break
        raise InvalidTokenError("Evidence intent classification failed") from last_error

    @staticmethod
    def _decode_with_key(
        token: str,
        key: object,
        *,
        leeway: int = 0,
        required: tuple[str, ...] = (),
    ) -> dict:
        return jwt.decode(
            token,
            key,
            algorithms=["ES256"],
            issuer=_EXPECTED_ISSUER,
            audience=_EXPECTED_AUDIENCE,
            leeway=leeway,
            options={"require": list(required)},
        )

    def _get_jwks_client(self, force_refresh: bool = False) -> PyJWKClient:
        if force_refresh or self._jwks_client is None:
            if not self._jwks_url:
                raise InvalidTokenError("JWT validation is not configured")
            self._jwks_client = PyJWKClient(
                self._jwks_url,
                cache_jwk_set=True,
                lifespan=_JWKS_CACHE_TTL_SECONDS,
                timeout=_JWKS_TIMEOUT_SECONDS,
            )
        return self._jwks_client


_validator: CognisJWTValidator | None = None
_validator_config: tuple[str, str] | None = None


def get_jwt_validator(
    public_key_path: str = "",
    jwks_url: str = "",
) -> CognisJWTValidator | None:
    """Return a cached validator for the current JWT configuration."""
    global _validator, _validator_config

    if not public_key_path and not jwks_url:
        _validator = None
        _validator_config = None
        return None

    config_key = (public_key_path, jwks_url)
    if _validator is None or _validator_config != config_key:
        _validator = CognisJWTValidator(
            public_key_path=public_key_path,
            jwks_url=jwks_url,
        )
        _validator_config = config_key
    return _validator


def looks_like_jwt(token: str) -> bool:
    """Return True when the token has the shape of a JWT."""
    return token.count(".") == 2
