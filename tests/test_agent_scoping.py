"""Tests for agent_id scoping: protection, dual-scope, and ownership verification."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from mnemory.memory import MemoryService

# ── _resolve_agent_id ──────────────────────────────────────────────────


class TestResolveAgentId:
    """Test the _resolve_agent_id function from server.py."""

    def _resolve(self, param_agent_id, session_agent_id=None):
        """Call _resolve_agent_id with a mocked session context."""
        from mnemory.server import _resolve_agent_id, _session_agent_id

        token = _session_agent_id.set(session_agent_id)
        try:
            return _resolve_agent_id(param_agent_id)
        finally:
            _session_agent_id.reset(token)

    def test_no_session_no_param(self):
        assert self._resolve(None, session_agent_id=None) is None

    def test_no_session_with_param(self):
        assert self._resolve("bob", session_agent_id=None) == "bob"

    def test_session_no_param_returns_none(self):
        """LLM omits agent_id → shared memory (None)."""
        assert self._resolve(None, session_agent_id="openwebui") is None

    def test_session_same_param(self):
        """LLM passes same agent_id as session → allowed."""
        assert self._resolve("openwebui", session_agent_id="openwebui") == "openwebui"

    def test_session_different_param_raises(self):
        """LLM passes different agent_id → blocked."""
        with pytest.raises(ValueError, match="Cannot use agent_id"):
            self._resolve("other-agent", session_agent_id="openwebui")


# ── _resolve_agent_id_for_core ─────────────────────────────────────────


class TestResolveAgentIdForCore:
    """Test the _resolve_agent_id_for_core function from server.py."""

    def _resolve(self, param_agent_id, session_agent_id=None):
        from mnemory.server import _resolve_agent_id_for_core, _session_agent_id

        token = _session_agent_id.set(session_agent_id)
        try:
            return _resolve_agent_id_for_core(param_agent_id)
        finally:
            _session_agent_id.reset(token)

    def test_no_session_no_param(self):
        assert self._resolve(None, session_agent_id=None) is None

    def test_no_session_with_param(self):
        assert self._resolve("bob", session_agent_id=None) == "bob"

    def test_session_no_param_auto_injects(self):
        """Unlike _resolve_agent_id, this auto-injects session agent."""
        assert self._resolve(None, session_agent_id="openwebui") == "openwebui"

    def test_session_same_param(self):
        assert self._resolve("openwebui", session_agent_id="openwebui") == "openwebui"

    def test_session_different_param_raises(self):
        with pytest.raises(ValueError, match="Cannot use agent_id"):
            self._resolve("other-agent", session_agent_id="openwebui")

    def test_self_with_session_resolves(self):
        """agent_id='self' with session should resolve to session value."""
        assert self._resolve("self", session_agent_id="openwebui") == "openwebui"

    def test_self_without_session_raises(self):
        """agent_id='self' without session should raise ValueError."""
        with pytest.raises(ValueError, match="requires X-Agent-Id"):
            self._resolve("self", session_agent_id=None)


# ── _get_session_agent_id ──────────────────────────────────────────────


class TestGetSessionAgentId:
    def test_returns_none_by_default(self):
        from mnemory.server import _get_session_agent_id, _session_agent_id

        token = _session_agent_id.set(None)
        try:
            assert _get_session_agent_id() is None
        finally:
            _session_agent_id.reset(token)

    def test_returns_session_value(self):
        from mnemory.server import _get_session_agent_id, _session_agent_id

        token = _session_agent_id.set("claude-code")
        try:
            assert _get_session_agent_id() == "claude-code"
        finally:
            _session_agent_id.reset(token)


# ── verify_memory_access ──────────────────────────────────────────────


class TestVerifyMemoryAccess:
    """Test ownership verification in MemoryService."""

    @staticmethod
    def _make_service(get_by_id_return=None):
        """Create a MemoryService with a mocked vector store."""
        service = object.__new__(MemoryService)
        service.vector = MagicMock()
        service.vector.get_by_id.return_value = get_by_id_return
        service.artifact = MagicMock()
        return service

    def test_no_session_agent_allows_all(self):
        """No session agent → no protection."""
        service = self._make_service()
        # Should not raise, should not even call get_by_id
        service.verify_memory_access("mem-1", session_agent_id=None)
        service.vector.get_by_id.assert_not_called()

    def test_memory_not_found_passes(self):
        """Memory not found → let downstream handle 404."""
        service = self._make_service(get_by_id_return=None)
        service.verify_memory_access("mem-1", session_agent_id="openwebui")
        # No error raised

    def test_shared_memory_allowed(self):
        """Shared memory (no agent_id) → accessible by any agent."""
        mem = {"id": "mem-1", "memory": "test"}
        service = self._make_service(get_by_id_return=mem)
        service.verify_memory_access("mem-1", session_agent_id="openwebui")

    def test_own_agent_memory_allowed(self):
        """Own agent's memory → accessible."""
        mem = {"id": "mem-1", "memory": "test", "agent_id": "openwebui"}
        service = self._make_service(get_by_id_return=mem)
        service.verify_memory_access("mem-1", session_agent_id="openwebui")

    def test_other_agent_memory_blocked(self):
        """Other agent's memory → blocked."""
        mem = {"id": "mem-1", "memory": "test", "agent_id": "claude-code"}
        service = self._make_service(get_by_id_return=mem)
        with pytest.raises(ValueError, match="Cannot access memory"):
            service.verify_memory_access("mem-1", session_agent_id="openwebui")

    def test_agent_id_none_in_memory_allowed(self):
        """Memory with agent_id=None (explicit) → shared, accessible."""
        mem = {"id": "mem-1", "memory": "test", "agent_id": None}
        service = self._make_service(get_by_id_return=mem)
        service.verify_memory_access("mem-1", session_agent_id="openwebui")

    def test_other_tenant_owner_is_blocked(self):
        mem = {
            "id": "mem-1",
            "user_id": "subject",
            "owner_id": "other-owner",
            "agent_id": "openwebui",
        }
        service = self._make_service(get_by_id_return=mem)
        with pytest.raises(ValueError, match="Cannot access memory"):
            service.verify_memory_access(
                "mem-1",
                user_id="subject",
                owner_id="authenticated-owner",
                session_agent_id="openwebui",
            )

    def test_legacy_record_without_owner_fails_closed(self):
        mem = {"id": "mem-1", "user_id": "subject", "agent_id": "openwebui"}
        service = self._make_service(get_by_id_return=mem)
        with pytest.raises(ValueError, match="Cannot access memory"):
            service.verify_memory_access(
                "mem-1",
                user_id="subject",
                owner_id="authenticated-owner",
                session_agent_id="openwebui",
            )

    def test_other_subject_user_is_blocked(self):
        mem = {
            "id": "mem-1",
            "user_id": "other-subject",
            "owner_id": "authenticated-owner",
        }
        service = self._make_service(get_by_id_return=mem)
        with pytest.raises(ValueError, match="Cannot access memory"):
            service.verify_memory_access(
                "mem-1",
                user_id="subject",
                owner_id="authenticated-owner",
                session_agent_id=None,
            )


# ── Dual-scope search ─────────────────────────────────────────────────


class TestSearchMemoriesDualScope:
    """Test dual-scope search merging and deduplication."""

    @staticmethod
    def _make_service():
        service = object.__new__(MemoryService)
        service.vector = MagicMock()
        service.artifact = MagicMock()
        service._config = MagicMock()
        service._config.memory.track_memory_access = False
        service._config.memory.classify_cache_ttl = 300
        service._config.memory.search_score_threshold = 0.30
        service._config.memory.search_score_threshold_hybrid = 0.0
        service._config.memory.search_similarity_weight = 0.9
        service._config.memory.search_keyword_weight = 0.2
        service._sparse = MagicMock()
        service._sparse.embed.return_value = None
        from mnemory.cache import TTLCache

        service._category_cache = TTLCache(ttl_seconds=300)
        service._core_cache = TTLCache(ttl_seconds=300)
        # Mock get_all for _get_available_categories
        service.vector.get_all.return_value = {"results": []}
        return service

    def test_merges_agent_and_shared(self):
        service = self._make_service()
        service.vector.search.side_effect = [
            # Agent-scoped results
            {
                "results": [
                    {"id": "a1", "score": 0.9, "metadata": {"importance": "normal"}},
                ]
            },
            # Shared results
            {
                "results": [
                    {"id": "s1", "score": 0.8, "metadata": {"importance": "normal"}},
                ]
            },
        ]

        results = service.search_memories_dual_scope(
            "test query",
            user_id="filip",
            session_agent_id="openwebui",
        )
        assert len(results) == 2
        ids = {r["id"] for r in results}
        assert ids == {"a1", "s1"}

    def test_deduplicates_by_id(self):
        service = self._make_service()
        service.vector.search.side_effect = [
            {
                "results": [
                    {"id": "dup1", "score": 0.9, "metadata": {"importance": "normal"}},
                ]
            },
            {
                "results": [
                    {"id": "dup1", "score": 0.85, "metadata": {"importance": "normal"}},
                    {"id": "s1", "score": 0.7, "metadata": {"importance": "normal"}},
                ]
            },
        ]

        results = service.search_memories_dual_scope(
            "test query",
            user_id="filip",
            session_agent_id="openwebui",
        )
        assert len(results) == 2
        ids = [r["id"] for r in results]
        assert "dup1" in ids
        assert "s1" in ids

    def test_respects_limit(self):
        service = self._make_service()
        service.vector.search.side_effect = [
            {
                "results": [
                    {
                        "id": f"a{i}",
                        "score": 0.9 - i * 0.01,
                        "metadata": {"importance": "normal"},
                    }
                    for i in range(5)
                ]
            },
            {
                "results": [
                    {
                        "id": f"s{i}",
                        "score": 0.85 - i * 0.01,
                        "metadata": {"importance": "normal"},
                    }
                    for i in range(5)
                ]
            },
        ]

        results = service.search_memories_dual_scope(
            "test query",
            user_id="filip",
            session_agent_id="openwebui",
            limit=3,
        )
        assert len(results) == 3

    def test_category_filter_passed_to_vector_store(self):
        """Category filter should be expanded and passed to vector.search()."""
        service = self._make_service()
        service.vector.search.side_effect = [
            {
                "results": [
                    {
                        "id": "a1",
                        "score": 0.9,
                        "metadata": {"importance": "normal", "categories": ["work"]},
                    },
                ]
            },
            {
                "results": [
                    {
                        "id": "s1",
                        "score": 0.7,
                        "metadata": {"importance": "normal", "categories": ["work"]},
                    },
                ]
            },
        ]

        results = service.search_memories_dual_scope(
            "test query",
            user_id="filip",
            session_agent_id="openwebui",
            categories=["work"],
        )
        # Both search calls should have received the categories parameter
        assert service.vector.search.call_count == 2
        for call in service.vector.search.call_args_list:
            _, kwargs = call
            assert kwargs["categories"] == ["work"]
        # Results should be merged
        ids = {r["id"] for r in results}
        assert ids == {"a1", "s1"}

    def test_shared_search_uses_shared_only(self):
        """The shared (second) search call must pass shared_only=True."""
        service = self._make_service()
        service.vector.search.side_effect = [
            {"results": []},
            {"results": []},
        ]

        service.search_memories_dual_scope(
            "test query",
            user_id="filip",
            session_agent_id="openwebui",
        )

        assert service.vector.search.call_count == 2
        # First call: agent-scoped (no shared_only)
        _, agent_kwargs = service.vector.search.call_args_list[0]
        assert agent_kwargs["agent_id"] == "openwebui"
        assert agent_kwargs.get("shared_only", False) is False
        # Second call: shared (shared_only=True)
        _, shared_kwargs = service.vector.search.call_args_list[1]
        assert shared_kwargs["agent_id"] is None
        assert shared_kwargs["shared_only"] is True

    def test_similarity_weight_passed_to_vector_store(self):
        """search_similarity_weight config should be forwarded to vector.search()."""
        service = self._make_service()
        service._config.memory.search_similarity_weight = 0.85
        service.vector.search.side_effect = [
            {"results": []},
            {"results": []},
        ]

        service.search_memories_dual_scope(
            "test query",
            user_id="filip",
            session_agent_id="openwebui",
        )

        for call in service.vector.search.call_args_list:
            _, kwargs = call
            assert kwargs["similarity_weight"] == 0.85


# ── Dual-scope list ───────────────────────────────────────────────────


class TestListMemoriesDualScope:
    """Test dual-scope list merging and deduplication."""

    @staticmethod
    def _make_service():
        service = object.__new__(MemoryService)
        service.vector = MagicMock()
        service.artifact = MagicMock()
        service._config = MagicMock()
        service._config.memory.track_memory_access = False
        service._sparse = MagicMock()
        service._sparse.embed.return_value = None
        return service

    def test_merges_agent_and_shared(self):
        service = self._make_service()
        service.vector.get_all.side_effect = [
            {"results": [{"id": "a1", "metadata": {}}]},
            {"results": [{"id": "s1", "metadata": {}}]},
        ]

        results = service.list_memories_dual_scope(
            user_id="filip",
            session_agent_id="openwebui",
        )
        assert len(results) == 2

    def test_deduplicates_by_id(self):
        service = self._make_service()
        service.vector.get_all.side_effect = [
            {"results": [{"id": "dup1", "metadata": {}}]},
            {"results": [{"id": "dup1", "metadata": {}}, {"id": "s1", "metadata": {}}]},
        ]

        results = service.list_memories_dual_scope(
            user_id="filip",
            session_agent_id="openwebui",
        )
        assert len(results) == 2
        ids = {r["id"] for r in results}
        assert ids == {"dup1", "s1"}

    def test_respects_limit(self):
        service = self._make_service()
        service.vector.get_all.side_effect = [
            {"results": [{"id": f"a{i}", "metadata": {}} for i in range(5)]},
            {"results": [{"id": f"s{i}", "metadata": {}} for i in range(5)]},
        ]

        results = service.list_memories_dual_scope(
            user_id="filip",
            session_agent_id="openwebui",
            limit=3,
        )
        assert len(results) == 3

    def test_shared_list_uses_shared_only(self):
        """The shared (second) get_all call must pass shared_only=True."""
        service = self._make_service()
        service.vector.get_all.side_effect = [
            {"results": []},
            {"results": []},
        ]

        service.list_memories_dual_scope(
            user_id="filip",
            session_agent_id="openwebui",
        )

        assert service.vector.get_all.call_count == 2
        # First call: agent-scoped (no shared_only)
        _, agent_kwargs = service.vector.get_all.call_args_list[0]
        assert agent_kwargs["agent_id"] == "openwebui"
        assert agent_kwargs.get("shared_only", False) is False
        # Second call: shared (shared_only=True)
        _, shared_kwargs = service.vector.get_all.call_args_list[1]
        assert shared_kwargs["agent_id"] is None
        assert shared_kwargs["shared_only"] is True

    def test_shared_agent_owner_list_filters_to_assistant_identity(self):
        """Owner side of shared-agent list must only expose assistant identity."""
        service = self._make_service()
        service.vector.get_all.side_effect = [
            {"results": [{"id": "owner", "metadata": {"role": "assistant"}}]},
            {"results": [{"id": "grantee", "metadata": {"role": "user"}}]},
        ]

        results = service.list_memories_dual_scope(
            user_id="grantee@example.com",
            owner_id="owner@example.com",
            session_agent_id="shared-agent",
        )

        assert {memory["id"] for memory in results} == {"owner", "grantee"}
        _, owner_kwargs = service.vector.get_all.call_args_list[0]
        assert owner_kwargs["subject_user_id"] == "owner@example.com"
        assert owner_kwargs["filters"]["role"] == "assistant"

    def test_shared_agent_role_user_list_skips_owner_scope(self):
        """role=user list must not query owner memories for shared agents."""
        service = self._make_service()
        service.vector.get_all.return_value = {
            "results": [{"id": "grantee", "metadata": {"role": "user"}}]
        }

        results = service.list_memories_dual_scope(
            user_id="grantee@example.com",
            owner_id="owner@example.com",
            session_agent_id="shared-agent",
            role="user",
        )

        assert [memory["id"] for memory in results] == ["grantee"]
        assert service.vector.get_all.call_count == 1
        _, grantee_kwargs = service.vector.get_all.call_args
        assert grantee_kwargs["subject_user_id"] == "grantee@example.com"
        assert grantee_kwargs["filters"]["role"] == "user"


# ── User identity header priority ─────────────────────────────────────


class TestUserIdentityHeaders:
    """Test the header priority chain for user_id resolution in the middleware."""

    @staticmethod
    def _make_request(headers: dict[str, str]):
        """Create a mock Starlette Request with the given headers."""
        mock_request = MagicMock()
        # Starlette normalizes header names to lowercase; use a real dict
        # so .get() works naturally (dict.get matches the Starlette API).
        mock_request.headers = {k.lower(): v for k, v in headers.items()}
        return mock_request

    def test_x_user_id_takes_priority(self):
        """X-User-Id should be used when both headers are present."""
        from mnemory.server import APIKeyMiddleware, _session_user_id

        middleware = object.__new__(APIKeyMiddleware)
        request = self._make_request(
            {
                "X-User-Id": "filip",
                "X-OpenWebUI-User-Email": "filip@example.com",
            }
        )

        token = _session_user_id.set(None)
        try:
            middleware._set_identity_from_headers(request)
            assert _session_user_id.get() == "filip"
        finally:
            _session_user_id.reset(token)

    def test_openwebui_email_fallback(self):
        """X-OpenWebUI-User-Email should be used when X-User-Id is absent."""
        from mnemory.server import APIKeyMiddleware, _session_user_id

        middleware = object.__new__(APIKeyMiddleware)
        request = self._make_request({"X-OpenWebUI-User-Email": "filip@example.com"})

        token = _session_user_id.set(None)
        try:
            middleware._set_identity_from_headers(request)
            assert _session_user_id.get() == "filip@example.com"
        finally:
            _session_user_id.reset(token)

    def test_empty_x_user_id_falls_through(self):
        """Empty X-User-Id should fall through to X-OpenWebUI-User-Email."""
        from mnemory.server import APIKeyMiddleware, _session_user_id

        middleware = object.__new__(APIKeyMiddleware)
        request = self._make_request(
            {
                "X-User-Id": "  ",
                "X-OpenWebUI-User-Email": "filip@example.com",
            }
        )

        token = _session_user_id.set(None)
        try:
            middleware._set_identity_from_headers(request)
            assert _session_user_id.get() == "filip@example.com"
        finally:
            _session_user_id.reset(token)

    def test_no_identity_headers(self):
        """No identity headers → session user_id stays None."""
        from mnemory.server import APIKeyMiddleware, _session_user_id

        middleware = object.__new__(APIKeyMiddleware)
        request = self._make_request({})

        token = _session_user_id.set(None)
        try:
            middleware._set_identity_from_headers(request)
            assert _session_user_id.get() is None
        finally:
            _session_user_id.reset(token)

    def test_agent_id_still_read(self):
        """X-Agent-Id should still be read alongside email fallback."""
        from mnemory.server import (
            APIKeyMiddleware,
            _session_agent_id,
            _session_user_id,
        )

        middleware = object.__new__(APIKeyMiddleware)
        request = self._make_request(
            {
                "X-OpenWebUI-User-Email": "filip@example.com",
                "X-Agent-Id": "open-webui",
            }
        )

        uid_token = _session_user_id.set(None)
        aid_token = _session_agent_id.set(None)
        try:
            middleware._set_identity_from_headers(request)
            assert _session_user_id.get() == "filip@example.com"
            assert _session_agent_id.get() == "open-webui"
        finally:
            _session_user_id.reset(uid_token)
            _session_agent_id.reset(aid_token)


# ── agent_id="self" sentinel ───────────────────────────────────────────


class TestResolveAgentIdSelf:
    """Test agent_id='self' sentinel resolution."""

    def _resolve(self, param_agent_id, session_agent_id=None):
        from mnemory.server import _resolve_agent_id, _session_agent_id

        token = _session_agent_id.set(session_agent_id)
        try:
            return _resolve_agent_id(param_agent_id)
        finally:
            _session_agent_id.reset(token)

    def test_self_with_session_resolves(self):
        """agent_id='self' with session agent should resolve to session value."""
        assert self._resolve("self", session_agent_id="openwebui") == "openwebui"

    def test_self_without_session_raises(self):
        """agent_id='self' without session agent should raise ValueError."""
        with pytest.raises(ValueError, match="requires X-Agent-Id"):
            self._resolve("self", session_agent_id=None)

    def test_self_is_case_sensitive(self):
        """Only lowercase 'self' is the sentinel, not 'Self' or 'SELF'."""
        # "Self" is treated as a regular agent_id, which differs from session
        with pytest.raises(ValueError, match="Cannot use agent_id"):
            self._resolve("Self", session_agent_id="openwebui")


# ── None metadata safety ──────────────────────────────────────────────


class TestNoneMetadataSafety:
    """Test that memories with metadata=None don't crash."""

    def test_format_memories_with_none_metadata(self):
        """_format_memories should handle memories where metadata is None."""
        import json

        from mnemory.server import _format_memories

        memories = [
            {
                "id": "mem-1",
                "memory": "test fact",
                "created_at": "2025-01-01T00:00:00",
                "metadata": None,
            },
            {
                "id": "mem-2",
                "memory": "another fact",
                "created_at": "2025-01-02T00:00:00",
                # metadata key missing entirely
            },
        ]
        result = json.loads(_format_memories(memories))
        assert result["count"] == 2
        assert result["results"][0]["memory"] == "test fact"
        assert result["results"][1]["memory"] == "another fact"

    def test_format_with_none_metadata_in_search(self):
        """Search results with None metadata should not crash score filtering."""
        from unittest.mock import patch

        mock_config = MagicMock()
        mock_config.memory.max_memory_length = 1000
        mock_config.memory.max_artifact_size = 102400
        mock_config.memory.max_core_context_length = 4000
        mock_config.memory.default_recent_days = 7
        mock_config.memory.recent_limit_user = 25
        mock_config.memory.recent_limit_agent = 25
        mock_config.memory.classify_cache_ttl = 300
        mock_config.memory.core_memories_cache_ttl = 300
        mock_config.memory.auto_classify = False
        mock_config.memory.track_memory_access = False
        mock_config.memory.search_score_threshold = 0.30
        mock_config.memory.search_score_threshold_hybrid = 0.0
        mock_config.memory.search_similarity_weight = 0.9
        mock_config.memory.search_keyword_weight = 0.2

        mock_sparse = MagicMock()
        mock_sparse.embed.return_value = None

        with (
            patch("mnemory.memory.VectorStore"),
            patch("mnemory.memory.ArtifactStore"),
            patch("mnemory.memory.LLMClient"),
            patch("mnemory.memory.SparseEmbeddingClient", return_value=mock_sparse),
        ):
            service = MemoryService(mock_config)

        service.vector = MagicMock()
        service.vector.search.return_value = {
            "results": [
                {"id": "1", "memory": "test", "score": 0.9, "metadata": None},
                {"id": "2", "memory": "test2", "score": 0.8},
            ]
        }
        # Should not crash — score filtering handles missing metadata
        results = service.search_memories(query="test", user_id="filip")
        assert len(results) == 2


# ── add_memories server-level validation ───────────────────────────────


class TestAddMemoriesTool:
    """Test the add_memories tool handler's validation logic."""

    def _call(self, memories, user_id="filip", agent_id=None, infer=True):
        """Call add_memories with mocked globals."""
        import json
        from unittest.mock import patch

        from mnemory.server import _session_agent_id, _session_user_id, add_memories

        uid_token = _session_user_id.set(user_id)
        aid_token = _session_agent_id.set(agent_id)
        try:
            mock_service = MagicMock()
            mock_service.add_memory.return_value = {
                "results": [{"id": "mem-1", "event": "ADD"}]
            }
            with patch("mnemory.server._get_service", return_value=mock_service):
                raw = asyncio.run(add_memories(memories=memories, infer=infer))
            return json.loads(raw), mock_service
        finally:
            _session_user_id.reset(uid_token)
            _session_agent_id.reset(aid_token)

    def test_empty_list_returns_error(self):
        result, _ = self._call([])
        assert result["error"] is True
        assert "empty" in result["message"]

    def test_exceeds_max_batch_size(self):
        memories = [{"content": f"fact {i}"} for i in range(21)]
        result, _ = self._call(memories)
        assert result["error"] is True
        assert "max 20" in result["message"].lower() or "Too many" in result["message"]

    def test_missing_content_field(self):
        memories = [{"memory_type": "fact"}]  # no "content"
        result, _ = self._call(memories)
        assert result["failed"] == 1
        assert result["succeeded"] == 0
        assert "content" in result["errors"][0]["message"].lower()

    def test_mixed_success_and_failure(self):
        memories = [
            {"content": "valid fact"},
            {"no_content_key": True},  # invalid
            {"content": "another valid fact"},
        ]
        result, service = self._call(memories)
        assert result["total"] == 3
        assert result["succeeded"] == 2
        assert result["failed"] == 1
        assert service.add_memory.call_count == 2

    def test_infer_false_passed_to_service(self):
        memories = [{"content": "test"}]
        _, service = self._call(memories, infer=False)
        _, kwargs = service.add_memory.call_args
        assert kwargs["infer"] is False


# ── Sub-agent support ─────────────────────────────────────────────────


class TestSubAgentResolve:
    """Test sub-agent prefix validation in _resolve_agent_id."""

    def _resolve(self, param_agent_id, session_agent_id=None):
        from mnemory.server import _resolve_agent_id, _session_agent_id

        token = _session_agent_id.set(session_agent_id)
        try:
            return _resolve_agent_id(param_agent_id)
        finally:
            _session_agent_id.reset(token)

    def test_sub_agent_allowed(self):
        """Sub-agent with session prefix should be allowed."""
        result = self._resolve("openwebui:bob", session_agent_id="openwebui")
        assert result == "openwebui:bob"

    def test_sub_agent_different_name(self):
        """Different sub-agent names under same parent should work."""
        assert (
            self._resolve("openwebui:alice", session_agent_id="openwebui")
            == "openwebui:alice"
        )

    def test_sub_agent_wrong_prefix_blocked(self):
        """Sub-agent with different prefix should be blocked."""
        with pytest.raises(ValueError, match="Cannot use agent_id"):
            self._resolve("cursor:bob", session_agent_id="openwebui")

    def test_sub_agent_partial_prefix_blocked(self):
        """Partial prefix match should not be allowed (openwebui2:bob)."""
        with pytest.raises(ValueError, match="Cannot use agent_id"):
            self._resolve("openwebui2:bob", session_agent_id="openwebui")

    def test_sub_agent_no_session_passthrough(self):
        """Without session agent, sub-agent IDs pass through."""
        result = self._resolve("openwebui:bob", session_agent_id=None)
        assert result == "openwebui:bob"

    def test_exact_match_still_works(self):
        """Exact session match should still work alongside sub-agents."""
        result = self._resolve("openwebui", session_agent_id="openwebui")
        assert result == "openwebui"

    def test_self_still_resolves_to_session(self):
        """'self' should still resolve to session agent, not sub-agent."""
        result = self._resolve("self", session_agent_id="openwebui")
        assert result == "openwebui"

    def test_none_still_returns_none(self):
        """None should still return None (shared memory)."""
        result = self._resolve(None, session_agent_id="openwebui")
        assert result is None


class TestSubAgentResolveForCore:
    """Test sub-agent prefix validation in _resolve_agent_id_for_core."""

    def _resolve(self, param_agent_id, session_agent_id=None):
        from mnemory.server import _resolve_agent_id_for_core, _session_agent_id

        token = _session_agent_id.set(session_agent_id)
        try:
            return _resolve_agent_id_for_core(param_agent_id)
        finally:
            _session_agent_id.reset(token)

    def test_sub_agent_allowed(self):
        """Sub-agent should be allowed in core memories resolution."""
        result = self._resolve("openwebui:bob", session_agent_id="openwebui")
        assert result == "openwebui:bob"

    def test_sub_agent_wrong_prefix_blocked(self):
        """Wrong prefix should be blocked."""
        with pytest.raises(ValueError, match="Cannot use agent_id"):
            self._resolve("cursor:bob", session_agent_id="openwebui")

    def test_none_auto_injects_session(self):
        """None should auto-inject session agent (not sub-agent)."""
        result = self._resolve(None, session_agent_id="openwebui")
        assert result == "openwebui"


class TestSubAgentVerifyAccess:
    """Test sub-agent access in verify_memory_access."""

    @staticmethod
    def _make_service(get_by_id_return=None):
        service = object.__new__(MemoryService)
        service.vector = MagicMock()
        service.vector.get_by_id.return_value = get_by_id_return
        service.artifact = MagicMock()
        return service

    def test_sub_agent_memory_accessible(self):
        """Session agent should be able to access its sub-agent's memory."""
        mem = {"id": "mem-1", "memory": "test", "agent_id": "openwebui:bob"}
        service = self._make_service(get_by_id_return=mem)
        # Should not raise
        service.verify_memory_access("mem-1", session_agent_id="openwebui")

    def test_other_sub_agent_memory_blocked(self):
        """Sub-agent memory under different parent should be blocked."""
        mem = {"id": "mem-1", "memory": "test", "agent_id": "cursor:bob"}
        service = self._make_service(get_by_id_return=mem)
        with pytest.raises(ValueError, match="Cannot access memory"):
            service.verify_memory_access("mem-1", session_agent_id="openwebui")

    def test_parent_memory_still_accessible(self):
        """Parent agent's own memory should still be accessible."""
        mem = {"id": "mem-1", "memory": "test", "agent_id": "openwebui"}
        service = self._make_service(get_by_id_return=mem)
        service.verify_memory_access("mem-1", session_agent_id="openwebui")

    def test_partial_prefix_blocked(self):
        """Partial prefix match should not grant access."""
        mem = {"id": "mem-1", "memory": "test", "agent_id": "openwebui2:bob"}
        service = self._make_service(get_by_id_return=mem)
        with pytest.raises(ValueError, match="Cannot access memory"):
            service.verify_memory_access("mem-1", session_agent_id="openwebui")
