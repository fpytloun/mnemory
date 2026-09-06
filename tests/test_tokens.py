"""Tests for HMAC-signed download tokens."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from unittest.mock import patch

import pytest

from mnemory.tokens import (
    _EXPIRY_GRACE_SECONDS,
    derive_signing_key,
    generate_download_token,
    validate_download_token,
    validate_download_token_claims,
)

# ── derive_signing_key ────────────────────────────────────────────────


class TestDeriveSigningKey:
    def test_single_api_key(self):
        key = derive_signing_key(api_key="test-key-123")
        assert isinstance(key, bytes)
        assert len(key) == 32

    def test_single_api_key_deterministic(self):
        """Same input produces same key."""
        k1 = derive_signing_key(api_key="my-secret")
        k2 = derive_signing_key(api_key="my-secret")
        assert k1 == k2

    def test_different_keys_produce_different_signing_keys(self):
        k1 = derive_signing_key(api_key="key-a")
        k2 = derive_signing_key(api_key="key-b")
        assert k1 != k2

    def test_multi_key_mapping(self):
        keys = {"key-a": "user1", "key-b": "user2"}
        key = derive_signing_key(api_keys=keys)
        assert isinstance(key, bytes)
        assert len(key) == 32

    def test_multi_key_order_independent(self):
        """Sorted internally, so insertion order doesn't matter."""
        k1 = derive_signing_key(api_keys={"b": "u2", "a": "u1"})
        k2 = derive_signing_key(api_keys={"a": "u1", "b": "u2"})
        assert k1 == k2

    def test_multi_key_takes_precedence_over_single(self):
        """When both are provided, api_keys wins."""
        k_multi = derive_signing_key(api_key="single", api_keys={"a": "u1"})
        k_single = derive_signing_key(api_key="single")
        assert k_multi != k_single

    def test_no_auth_returns_random_key(self):
        """No auth configured — ephemeral random key."""
        k1 = derive_signing_key()
        k2 = derive_signing_key()
        assert isinstance(k1, bytes)
        assert len(k1) == 32
        # Random keys should differ (extremely unlikely to collide)
        assert k1 != k2


# ── generate_download_token ───────────────────────────────────────────


class TestGenerateDownloadToken:
    @pytest.fixture()
    def signing_key(self):
        return derive_signing_key(api_key="test-key")

    def test_format(self, signing_key):
        token = generate_download_token(
            signing_key, "user1", "mem-1", "art-1", ttl_seconds=60
        )
        parts = token.split(".")
        assert len(parts) == 2
        # Both parts should be valid base64url
        base64.urlsafe_b64decode(parts[0])
        base64.urlsafe_b64decode(parts[1])

    def test_payload_contents(self, signing_key):
        token = generate_download_token(
            signing_key, "user1", "mem-1", "art-1", ttl_seconds=300
        )
        payload_b64 = token.split(".")[0]
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        assert payload["u"] == "user1"
        assert payload["m"] == "mem-1"
        assert payload["a"] == "art-1"
        assert payload["e"] > time.time()
        assert payload["e"] <= time.time() + 301  # within 1s tolerance

    def test_signature_is_valid_hmac(self, signing_key):
        token = generate_download_token(signing_key, "user1", "mem-1", "art-1")
        payload_b64, sig_b64 = token.split(".")
        expected_sig = hmac.new(
            signing_key, payload_b64.encode(), hashlib.sha256
        ).digest()
        actual_sig = base64.urlsafe_b64decode(sig_b64)
        assert hmac.compare_digest(expected_sig, actual_sig)


# ── validate_download_token ───────────────────────────────────────────


class TestValidateDownloadToken:
    @pytest.fixture()
    def signing_key(self):
        return derive_signing_key(api_key="test-key")

    def test_valid_token(self, signing_key):
        token = generate_download_token(
            signing_key, "user1", "mem-1", "art-1", ttl_seconds=60
        )
        uid = validate_download_token(signing_key, token, "mem-1", "art-1")
        assert uid == "user1"

    def test_scoped_claims_preserve_owner_and_agent(self, signing_key):
        token = generate_download_token(
            signing_key,
            "user1",
            "mem-1",
            "art-1",
            owner_id="owner-1",
            agent_id="parent:alpha",
        )

        claims = validate_download_token_claims(
            signing_key,
            token,
            "mem-1",
            "art-1",
        )

        assert claims == {
            "u": "user1",
            "o": "owner-1",
            "g": "parent:alpha",
        }

    def test_expired_token(self, signing_key):
        token = generate_download_token(
            signing_key, "user1", "mem-1", "art-1", ttl_seconds=1
        )
        # Simulate time passing beyond expiry + grace
        with patch("mnemory.tokens.time") as mock_time:
            mock_time.time.return_value = time.time() + 2 + _EXPIRY_GRACE_SECONDS + 1
            uid = validate_download_token(signing_key, token, "mem-1", "art-1")
        assert uid is None

    def test_within_grace_period(self, signing_key):
        """Token just expired but within grace period should still work."""
        token = generate_download_token(
            signing_key, "user1", "mem-1", "art-1", ttl_seconds=1
        )
        # Simulate time just past expiry but within grace
        with patch("mnemory.tokens.time") as mock_time:
            mock_time.time.return_value = time.time() + 2  # 1s past expiry
            uid = validate_download_token(signing_key, token, "mem-1", "art-1")
        assert uid == "user1"

    def test_wrong_memory_id(self, signing_key):
        token = generate_download_token(
            signing_key, "user1", "mem-1", "art-1", ttl_seconds=60
        )
        uid = validate_download_token(signing_key, token, "mem-WRONG", "art-1")
        assert uid is None

    def test_wrong_artifact_id(self, signing_key):
        token = generate_download_token(
            signing_key, "user1", "mem-1", "art-1", ttl_seconds=60
        )
        uid = validate_download_token(signing_key, token, "mem-1", "art-WRONG")
        assert uid is None

    def test_tampered_payload(self, signing_key):
        token = generate_download_token(
            signing_key, "user1", "mem-1", "art-1", ttl_seconds=60
        )
        payload_b64, sig_b64 = token.split(".")
        # Tamper with payload
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        payload["u"] = "attacker"
        tampered_b64 = base64.urlsafe_b64encode(
            json.dumps(payload, separators=(",", ":")).encode()
        ).decode()
        tampered_token = f"{tampered_b64}.{sig_b64}"
        uid = validate_download_token(signing_key, tampered_token, "mem-1", "art-1")
        assert uid is None

    def test_tampered_signature(self, signing_key):
        token = generate_download_token(
            signing_key, "user1", "mem-1", "art-1", ttl_seconds=60
        )
        payload_b64, _ = token.split(".")
        fake_sig = base64.urlsafe_b64encode(b"x" * 32).decode()
        tampered_token = f"{payload_b64}.{fake_sig}"
        uid = validate_download_token(signing_key, tampered_token, "mem-1", "art-1")
        assert uid is None

    def test_wrong_signing_key(self, signing_key):
        token = generate_download_token(
            signing_key, "user1", "mem-1", "art-1", ttl_seconds=60
        )
        other_key = derive_signing_key(api_key="different-key")
        uid = validate_download_token(other_key, token, "mem-1", "art-1")
        assert uid is None

    def test_malformed_token_no_dot(self, signing_key):
        uid = validate_download_token(signing_key, "nodothere", "mem-1", "art-1")
        assert uid is None

    def test_malformed_token_too_many_dots(self, signing_key):
        uid = validate_download_token(signing_key, "a.b.c", "mem-1", "art-1")
        assert uid is None

    def test_empty_token(self, signing_key):
        uid = validate_download_token(signing_key, "", "mem-1", "art-1")
        assert uid is None

    def test_invalid_base64(self, signing_key):
        uid = validate_download_token(signing_key, "!!!.!!!", "mem-1", "art-1")
        assert uid is None
