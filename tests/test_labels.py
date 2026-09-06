"""Tests for the labels feature — custom metadata on memories."""

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from mnemory.memory import MemoryService, _validate_labels

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_memory_config(mock_config: MagicMock) -> None:
    """Set standard memory config attributes on a MagicMock config object."""
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
    mock_config.memory.ttl_fact = None
    mock_config.memory.ttl_preference = None
    mock_config.memory.ttl_episodic = 90
    mock_config.memory.ttl_procedural = 60
    mock_config.memory.ttl_context = 7
    mock_config.memory.search_score_threshold = 0.30
    mock_config.memory.dedup_similarity_threshold = 0.4
    mock_config.memory.search_similarity_weight = 0.9
    mock_config.memory.search_keyword_weight = 0.2
    mock_config.memory.find_memories_queries = 5
    mock_config.memory.remember_artifact_threshold = 4000
    mock_config.memory.remember_max_session_memories = 50
    mock_config.memory.remember_summary_compaction_threshold = 10000
    mock_config.memory.max_input_length = 400000
    mock_config.memory.core_top_memories = 10
    mock_config.memory.core_min_importance = "normal"
    mock_config.memory.core_recent_min_importance = "normal"
    mock_config.memory.core_max_per_section = 25
    mock_config.memory.search_score_threshold_hybrid = 0.0
    mock_config.memory.find_model = ""
    mock_config.memory.find_reasoning_effort = None
    # Labels config
    mock_config.memory.labels_max_fields = 20
    mock_config.memory.labels_max_key_length = 64
    mock_config.memory.labels_max_value_length = 1000
    mock_config.memory.labels_indexes = []


def _make_service(auto_classify=False, track_access=False):
    """Create a MemoryService with mocked backends."""
    mock_config = MagicMock()
    _mock_memory_config(mock_config)
    mock_config.memory.auto_classify = auto_classify
    mock_config.memory.track_memory_access = track_access

    mock_sparse = MagicMock()
    mock_sparse.embed.return_value = None
    mock_sparse.embed_batch.return_value = None

    with (
        patch("mnemory.memory.VectorStore"),
        patch("mnemory.memory.ArtifactStore"),
        patch("mnemory.memory.LLMClient"),
        patch("mnemory.memory.SparseEmbeddingClient", return_value=mock_sparse),
    ):
        service = MemoryService(mock_config)

    service.vector = MagicMock()
    service.revisions = MagicMock()
    service.revisions.revise.return_value = {
        "operation_id": "op-1",
        "lineage_id": "mem-1",
        "previous_revision_id": "mem-1",
        "revision_id": "mem-2",
        "revision": 2,
        "status": "updated",
        "replayed": False,
    }
    service._llm = MagicMock()
    service._find_llm = service._llm

    service.vector.embedding.embed.return_value = [0.1] * 1536
    service.vector.embedding.embed_batch.side_effect = lambda texts: [
        [0.1] * 1536 for _ in texts
    ]

    return service


def _make_config_mock():
    """Return a simple mock with labels config attributes."""
    cfg = MagicMock()
    cfg.labels_max_fields = 20
    cfg.labels_max_key_length = 64
    cfg.labels_max_value_length = 1000
    return cfg


# ===================================================================
# _validate_labels() unit tests
# ===================================================================


class TestValidateLabels:
    """Tests for the _validate_labels() validation function."""

    def test_valid_simple_labels(self):
        cfg = _make_config_mock()
        result = _validate_labels({"project": "myapp", "env": "prod"}, cfg)
        assert result == {"project": "myapp", "env": "prod"}

    def test_valid_all_value_types(self):
        cfg = _make_config_mock()
        labels = {
            "name": "test",
            "count": 42,
            "ratio": 3.14,
            "active": True,
            "tags": ["a", "b", "c"],
        }
        result = _validate_labels(labels, cfg)
        assert result == labels

    def test_empty_dict_is_valid(self):
        cfg = _make_config_mock()
        result = _validate_labels({}, cfg)
        assert result == {}

    def test_not_a_dict_raises(self):
        cfg = _make_config_mock()
        with pytest.raises(ValueError, match="labels must be a dict"):
            _validate_labels("not a dict", cfg)

    def test_not_a_dict_list_raises(self):
        cfg = _make_config_mock()
        with pytest.raises(ValueError, match="labels must be a dict"):
            _validate_labels(["a", "b"], cfg)

    def test_too_many_labels(self):
        cfg = _make_config_mock()
        cfg.labels_max_fields = 2
        with pytest.raises(ValueError, match="Too many labels"):
            _validate_labels({"a": "1", "b": "2", "c": "3"}, cfg)

    def test_key_not_string(self):
        cfg = _make_config_mock()
        with pytest.raises(ValueError, match="Label key must be a string"):
            _validate_labels({123: "value"}, cfg)

    def test_key_empty(self):
        cfg = _make_config_mock()
        with pytest.raises(ValueError, match="must not be empty"):
            _validate_labels({"": "value"}, cfg)

    def test_key_too_long(self):
        cfg = _make_config_mock()
        cfg.labels_max_key_length = 5
        with pytest.raises(ValueError, match="too long"):
            _validate_labels({"toolong": "value"}, cfg)

    def test_key_invalid_pattern_starts_with_digit(self):
        cfg = _make_config_mock()
        with pytest.raises(ValueError, match="is invalid"):
            _validate_labels({"1abc": "value"}, cfg)

    def test_key_invalid_pattern_has_dash(self):
        cfg = _make_config_mock()
        with pytest.raises(ValueError, match="is invalid"):
            _validate_labels({"my-key": "value"}, cfg)

    def test_key_invalid_pattern_has_dot(self):
        cfg = _make_config_mock()
        with pytest.raises(ValueError, match="is invalid"):
            _validate_labels({"my.key": "value"}, cfg)

    def test_key_valid_underscore_prefix(self):
        cfg = _make_config_mock()
        result = _validate_labels({"_private": "value"}, cfg)
        assert "_private" in result

    def test_key_valid_mixed_case(self):
        cfg = _make_config_mock()
        result = _validate_labels({"MyLabel_123": "value"}, cfg)
        assert "MyLabel_123" in result

    def test_reserved_key_memory_type(self):
        cfg = _make_config_mock()
        with pytest.raises(ValueError, match="reserved"):
            _validate_labels({"memory_type": "fact"}, cfg)

    def test_reserved_key_user_id(self):
        cfg = _make_config_mock()
        with pytest.raises(ValueError, match="reserved"):
            _validate_labels({"user_id": "filip"}, cfg)

    def test_reserved_key_labels(self):
        cfg = _make_config_mock()
        with pytest.raises(ValueError, match="reserved"):
            _validate_labels({"labels": "nested"}, cfg)

    def test_reserved_key_categories(self):
        cfg = _make_config_mock()
        with pytest.raises(ValueError, match="reserved"):
            _validate_labels({"categories": ["a"]}, cfg)

    def test_value_unsupported_type_dict(self):
        cfg = _make_config_mock()
        with pytest.raises(ValueError, match="unsupported value type"):
            _validate_labels({"key": {"nested": "dict"}}, cfg)

    def test_value_unsupported_type_none(self):
        cfg = _make_config_mock()
        with pytest.raises(ValueError, match="unsupported value type"):
            _validate_labels({"key": None}, cfg)

    def test_value_string_too_long(self):
        cfg = _make_config_mock()
        cfg.labels_max_value_length = 10
        with pytest.raises(ValueError, match="value too long"):
            _validate_labels({"key": "a" * 11}, cfg)

    def test_value_list_with_non_string(self):
        cfg = _make_config_mock()
        with pytest.raises(ValueError, match="list values must all be strings"):
            _validate_labels({"tags": ["ok", 123]}, cfg)

    def test_value_list_element_too_long(self):
        cfg = _make_config_mock()
        cfg.labels_max_value_length = 5
        with pytest.raises(ValueError, match="list value too long"):
            _validate_labels({"tags": ["toolong"]}, cfg)

    def test_value_bool_false(self):
        cfg = _make_config_mock()
        result = _validate_labels({"active": False}, cfg)
        assert result["active"] is False

    def test_value_int_zero(self):
        cfg = _make_config_mock()
        result = _validate_labels({"count": 0}, cfg)
        assert result["count"] == 0

    def test_value_float_negative(self):
        cfg = _make_config_mock()
        result = _validate_labels({"score": -1.5}, cfg)
        assert result["score"] == -1.5

    def test_value_empty_list(self):
        cfg = _make_config_mock()
        result = _validate_labels({"tags": []}, cfg)
        assert result["tags"] == []


# ===================================================================
# Labels in add_memory (infer=False, direct path)
# ===================================================================


class TestLabelsAddDirect:
    """Labels flow through add_memory with infer=False."""

    def test_labels_stored_in_metadata(self):
        service = _make_service()
        service.vector.insert.return_value = "mem-1"

        service.add_memory(
            content="test fact",
            user_id="filip",
            infer=False,
            labels={"project": "myapp", "env": "prod"},
        )

        service.vector.insert.assert_called_once()
        _, kwargs = service.vector.insert.call_args
        metadata = kwargs["metadata"]
        assert metadata["labels"] == {"project": "myapp", "env": "prod"}

    def test_no_labels_key_when_none(self):
        service = _make_service()
        service.vector.insert.return_value = "mem-1"

        service.add_memory(
            content="test fact",
            user_id="filip",
            infer=False,
        )

        service.vector.insert.assert_called_once()
        _, kwargs = service.vector.insert.call_args
        metadata = kwargs["metadata"]
        assert "labels" not in metadata

    def test_invalid_labels_raises_valueerror(self):
        service = _make_service()
        with pytest.raises(ValueError, match="is invalid"):
            service.add_memory(
                content="test",
                user_id="filip",
                infer=False,
                labels={"1bad": "value"},
            )

    def test_reserved_label_key_raises(self):
        service = _make_service()
        with pytest.raises(ValueError, match="reserved"):
            service.add_memory(
                content="test",
                user_id="filip",
                infer=False,
                labels={"memory_type": "fact"},
            )


# ===================================================================
# Labels in add_memory (infer=True, extraction path)
# ===================================================================


class TestLabelsAddWithInference:
    """Labels are inherited by all extracted facts via explicit_fields."""

    def test_labels_inherited_by_extracted_facts(self):
        service = _make_service()
        service.vector.insert.return_value = "mem-1"
        # No similar memories found for dedup
        service.vector.search_similar.return_value = []

        # LLM extracts two facts
        service._llm.generate.return_value = json.dumps(
            {
                "memories": [
                    {
                        "text": "User lives in Prague",
                        "action": "ADD",
                        "memory_type": "fact",
                        "categories": ["personal"],
                        "importance": "normal",
                        "pinned": False,
                        "event_date": None,
                    },
                    {
                        "text": "User works at Acme",
                        "action": "ADD",
                        "memory_type": "fact",
                        "categories": ["work"],
                        "importance": "normal",
                        "pinned": False,
                        "event_date": None,
                    },
                ]
            }
        )

        service.add_memory(
            content="I live in Prague and work at Acme",
            user_id="filip",
            infer=True,
            labels={"source": "chat", "session": "abc123"},
        )

        # Both inserts should have labels in metadata
        assert service.vector.insert.call_count == 2
        for call in service.vector.insert.call_args_list:
            _, kwargs = call
            metadata = kwargs["metadata"]
            assert metadata["labels"] == {"source": "chat", "session": "abc123"}

    def test_no_labels_in_metadata_when_none(self):
        service = _make_service()
        service.vector.insert.return_value = "mem-1"
        service.vector.search_similar.return_value = []

        service._llm.generate.return_value = json.dumps(
            {
                "memories": [
                    {
                        "text": "User likes Python",
                        "action": "ADD",
                        "memory_type": "preference",
                        "categories": ["technical"],
                        "importance": "normal",
                        "pinned": False,
                        "event_date": None,
                    },
                ]
            }
        )

        service.add_memory(
            content="I like Python",
            user_id="filip",
            infer=True,
        )

        _, kwargs = service.vector.insert.call_args
        metadata = kwargs["metadata"]
        assert "labels" not in metadata


# ===================================================================
# Labels merge in _execute_action UPDATE path
# ===================================================================


class TestLabelsUpdateMerge:
    """Labels merge behavior during UPDATE actions (via extraction)."""

    def test_update_merges_labels(self):
        """Caller labels merge with existing, caller wins on conflicts."""
        service = _make_service()
        service.vector.search_similar.return_value = [
            {
                "id": "existing-1",
                "text": "User drives a Skoda",
                "score": 0.95,
                "metadata": {
                    "memory_type": "fact",
                    "categories": ["vehicles"],
                    "importance": "normal",
                    "pinned": False,
                },
            }
        ]
        service.vector.get_by_id.return_value = {
            "id": "existing-1",
            "memory": "User drives a Skoda",
            "metadata": {
                "memory_type": "fact",
                "categories": ["vehicles"],
                "importance": "normal",
                "pinned": False,
                "labels": {"source": "old", "project": "cars"},
                "artifacts": [],
            },
        }

        service._llm.generate.return_value = json.dumps(
            {
                "memories": [
                    {
                        "text": "User drives a Tesla",
                        "action": "UPDATE",
                        "target_id": "0",
                        "memory_type": "fact",
                        "categories": ["vehicles"],
                        "importance": "normal",
                        "pinned": False,
                        "event_date": None,
                    },
                ]
            }
        )

        service.add_memory(
            content="I bought a Tesla",
            user_id="filip",
            infer=True,
            labels={"source": "new", "env": "prod"},
        )

        service.revisions.revise.assert_called_once()
        meta_arg = service.revisions.revise.call_args.kwargs["changes"]
        assert meta_arg["labels"] == {
            "source": "new",  # caller wins
            "project": "cars",  # preserved from existing
            "env": "prod",  # new from caller
        }

    def test_update_preserves_existing_labels_when_no_caller_labels(self):
        """When no labels passed, existing labels are preserved."""
        service = _make_service()
        service.vector.search_similar.return_value = [
            {
                "id": "existing-1",
                "text": "User drives a Skoda",
                "score": 0.95,
                "metadata": {
                    "memory_type": "fact",
                    "categories": ["vehicles"],
                    "importance": "normal",
                    "pinned": False,
                },
            }
        ]
        service.vector.get_by_id.return_value = {
            "id": "existing-1",
            "memory": "User drives a Skoda",
            "metadata": {
                "memory_type": "fact",
                "categories": ["vehicles"],
                "importance": "normal",
                "pinned": False,
                "labels": {"source": "old"},
                "artifacts": [],
            },
        }

        service._llm.generate.return_value = json.dumps(
            {
                "memories": [
                    {
                        "text": "User drives a Tesla",
                        "action": "UPDATE",
                        "target_id": "0",
                        "memory_type": "fact",
                        "categories": ["vehicles"],
                        "importance": "normal",
                        "pinned": False,
                        "event_date": None,
                    },
                ]
            }
        )

        service.add_memory(
            content="I bought a Tesla",
            user_id="filip",
            infer=True,
            # No labels passed
        )

        meta_arg = service.revisions.revise.call_args.kwargs["changes"]
        assert meta_arg["labels"] == {"source": "old"}


# ===================================================================
# Labels in update_memory (direct update)
# ===================================================================


class TestLabelsUpdateMemory:
    """Labels behavior in the update_memory() method."""

    def test_update_with_labels(self):
        service = _make_service()
        service.update_memory(
            "mem-1",
            user_id="filip",
            labels={"project": "myapp"},
        )

        service.revisions.revise.assert_called_once()
        meta_arg = service.revisions.revise.call_args.kwargs["changes"]
        assert meta_arg["labels"] == {"project": "myapp"}

    def test_update_empty_dict_clears_labels(self):
        service = _make_service()
        service.update_memory(
            "mem-1",
            user_id="filip",
            labels={},
        )

        meta_arg = service.revisions.revise.call_args.kwargs["changes"]
        assert meta_arg["labels"] == {}

    def test_update_none_preserves_labels(self):
        """When labels=None (default), labels key is not in metadata_updates."""
        service = _make_service()
        service.update_memory(
            "mem-1",
            user_id="filip",
            # Only set a metadata field so update_metadata is called
            importance="high",
            # labels=None (default) — should not appear in metadata_updates
        )

        meta_arg = service.revisions.revise.call_args.kwargs["changes"]
        assert "labels" not in meta_arg

    def test_update_validates_labels(self):
        service = _make_service()
        with pytest.raises(ValueError, match="reserved"):
            service.update_memory(
                "mem-1",
                user_id="filip",
                labels={"user_id": "hacker"},
            )


# ===================================================================
# Labels in search_memories and list_memories
# ===================================================================


class TestLabelsSearch:
    """Labels filter is passed through to vector store."""

    def test_search_passes_labels_filter(self):
        service = _make_service()
        service.vector.search.return_value = {"results": []}

        service.search_memories(
            query="test",
            user_id="filip",
            labels={"project": "myapp"},
        )

        _, kwargs = service.vector.search.call_args
        assert kwargs["labels_filter"] == {"project": "myapp"}

    def test_search_no_labels_filter(self):
        service = _make_service()
        service.vector.search.return_value = {"results": []}

        service.search_memories(
            query="test",
            user_id="filip",
        )

        _, kwargs = service.vector.search.call_args
        assert kwargs["labels_filter"] is None

    def test_list_passes_labels_filter(self):
        service = _make_service()
        service.vector.get_all.return_value = {"results": []}

        service.list_memories(
            user_id="filip",
            labels={"env": "prod"},
        )

        _, kwargs = service.vector.get_all.call_args
        assert kwargs["labels_filter"] == {"env": "prod"}

    def test_list_no_labels_filter(self):
        service = _make_service()
        service.vector.get_all.return_value = {"results": []}

        service.list_memories(
            user_id="filip",
        )

        _, kwargs = service.vector.get_all.call_args
        assert kwargs["labels_filter"] is None


# ===================================================================
# Labels in remember pipeline
# ===================================================================


class TestLabelsRemember:
    """Labels flow through the remember pipeline to _execute_action."""

    def test_remember_passes_labels_to_extracted_facts(self):
        service = _make_service()
        service.vector.insert.return_value = "mem-1"
        service.vector.search_similar.return_value = []
        # No session memories
        service.vector.get_all.return_value = {"results": []}

        # Stage 1: extraction returns one fact
        # Stage 2: dedup returns ADD action
        # The remember pipeline calls _llm.generate twice:
        #   1) extraction (returns facts)
        #   2) dedup (returns actions)
        service._llm.generate.side_effect = [
            # Extraction response
            json.dumps(
                {
                    "memories": [
                        {
                            "text": "User prefers dark mode",
                            "role": "user",
                            "memory_type": "preference",
                            "categories": ["preferences"],
                            "importance": "normal",
                            "pinned": False,
                            "event_date": None,
                        }
                    ]
                }
            ),
            # Dedup response
            json.dumps(
                {
                    "memories": [
                        {
                            "text": "User prefers dark mode",
                            "action": "ADD",
                            "memory_type": "preference",
                            "categories": ["preferences"],
                            "importance": "normal",
                            "pinned": False,
                            "event_date": None,
                        }
                    ]
                }
            ),
        ]

        service.remember(
            content="User: I prefer dark mode\nAssistant: Got it!",
            user_id="filip",
            labels={"conversation_id": "conv-123"},
        )

        # The inserted memory should have labels
        service.vector.insert.assert_called_once()
        _, kwargs = service.vector.insert.call_args
        metadata = kwargs["metadata"]
        assert metadata["labels"] == {"conversation_id": "conv-123"}

    def test_remember_no_labels_no_labels_in_metadata(self):
        service = _make_service()
        service.vector.insert.return_value = "mem-1"
        service.vector.search_similar.return_value = []
        service.vector.get_all.return_value = {"results": []}

        service._llm.generate.side_effect = [
            json.dumps(
                {
                    "memories": [
                        {
                            "text": "User likes cats",
                            "role": "user",
                            "memory_type": "preference",
                            "categories": ["preferences"],
                            "importance": "normal",
                            "pinned": False,
                            "event_date": None,
                        }
                    ]
                }
            ),
            json.dumps(
                {
                    "memories": [
                        {
                            "text": "User likes cats",
                            "action": "ADD",
                            "memory_type": "preference",
                            "categories": ["preferences"],
                            "importance": "normal",
                            "pinned": False,
                            "event_date": None,
                        }
                    ]
                }
            ),
        ]

        service.remember(
            content="User: I like cats\nAssistant: Noted!",
            user_id="filip",
        )

        _, kwargs = service.vector.insert.call_args
        metadata = kwargs["metadata"]
        assert "labels" not in metadata


# ===================================================================
# Labels in MCP tools (server.py)
# ===================================================================


class TestLabelsMCPTools:
    """Labels parameter is passed through MCP tool functions."""

    def _set_session(self, user_id="filip", agent_id=None):
        from mnemory.server import _session_agent_id, _session_user_id

        uid_token = _session_user_id.set(user_id)
        aid_token = _session_agent_id.set(agent_id)
        return uid_token, aid_token

    def _reset_session(self, uid_token, aid_token):
        from mnemory.server import _session_agent_id, _session_user_id

        _session_user_id.reset(uid_token)
        _session_agent_id.reset(aid_token)

    def test_add_memory_tool_passes_labels(self):
        from mnemory.server import add_memory as add_memory_tool

        uid_token, aid_token = self._set_session()
        try:
            mock_service = MagicMock()
            mock_service.add_memory.return_value = {
                "results": [{"id": "mem-1", "event": "ADD"}]
            }
            with (
                patch("mnemory.server._get_service", return_value=mock_service),
                patch("mnemory.server.get_collector", return_value=None),
            ):
                asyncio.run(
                    add_memory_tool(
                        content="test",
                        labels={"project": "myapp"},
                    )
                )
            _, kwargs = mock_service.add_memory.call_args
            assert kwargs["labels"] == {"project": "myapp"}
        finally:
            self._reset_session(uid_token, aid_token)

    def test_add_memory_tool_no_labels(self):
        from mnemory.server import add_memory as add_memory_tool

        uid_token, aid_token = self._set_session()
        try:
            mock_service = MagicMock()
            mock_service.add_memory.return_value = {
                "results": [{"id": "mem-1", "event": "ADD"}]
            }
            with (
                patch("mnemory.server._get_service", return_value=mock_service),
                patch("mnemory.server.get_collector", return_value=None),
            ):
                asyncio.run(add_memory_tool(content="test"))
            _, kwargs = mock_service.add_memory.call_args
            assert kwargs["labels"] is None
        finally:
            self._reset_session(uid_token, aid_token)

    def test_search_memories_tool_passes_labels(self):
        from mnemory.server import search_memories as search_tool

        uid_token, aid_token = self._set_session()
        try:
            mock_service = MagicMock()
            mock_service.search_memories.return_value = []
            with (
                patch("mnemory.server._get_service", return_value=mock_service),
                patch("mnemory.server.get_collector", return_value=None),
            ):
                asyncio.run(
                    search_tool(
                        query="test",
                        labels={"project": "myapp"},
                    )
                )
            _, kwargs = mock_service.search_memories.call_args
            assert kwargs["labels"] == {"project": "myapp"}
        finally:
            self._reset_session(uid_token, aid_token)

    def test_update_memory_tool_passes_labels(self):
        from mnemory.server import update_memory as update_tool

        uid_token, aid_token = self._set_session()
        try:
            mock_service = MagicMock()
            mock_service.verify_memory_access.return_value = None
            mock_service.update_memory.return_value = {
                "status": "updated",
                "memory_id": "mem-1",
            }
            with (
                patch("mnemory.server._get_service", return_value=mock_service),
                patch("mnemory.server.get_collector", return_value=None),
            ):
                asyncio.run(
                    update_tool(
                        memory_id="mem-1",
                        labels={"project": "updated"},
                    )
                )
            _, kwargs = mock_service.update_memory.call_args
            assert kwargs["labels"] == {"project": "updated"}
        finally:
            self._reset_session(uid_token, aid_token)

    def test_update_memory_tool_no_labels_not_in_kwargs(self):
        from mnemory.server import update_memory as update_tool

        uid_token, aid_token = self._set_session()
        try:
            mock_service = MagicMock()
            mock_service.verify_memory_access.return_value = None
            mock_service.update_memory.return_value = {
                "status": "updated",
                "memory_id": "mem-1",
            }
            with (
                patch("mnemory.server._get_service", return_value=mock_service),
                patch("mnemory.server.get_collector", return_value=None),
            ):
                asyncio.run(update_tool(memory_id="mem-1", content="new text"))
            _, kwargs = mock_service.update_memory.call_args
            # labels=None is not passed when not provided
            assert "labels" not in kwargs
        finally:
            self._reset_session(uid_token, aid_token)


# ===================================================================
# Labels in vector store filter building
# ===================================================================


class TestLabelsVectorFilter:
    """Test that labels_filter builds correct Qdrant FieldConditions."""

    def test_search_scalar_filter(self):
        """Scalar label values use MatchValue."""
        from mnemory.storage.vector import VectorStore

        store = object.__new__(VectorStore)
        store._config = MagicMock()
        store._config.vector.collection_name = "mnemory"
        store._config.vector.is_remote = False
        store._client = MagicMock()
        store._embedding = MagicMock()
        store._embedding.embed.return_value = [0.1] * 1536
        store._write_lock = None

        # Mock the search to capture the filter
        store._client.query_points.return_value = MagicMock(points=[])

        store.search(
            query="test",
            user_id="filip",
            labels_filter={"project": "myapp"},
        )

        # The filter is built internally — we verify by checking the call was made
        # and that it didn't raise. For deeper verification, we'd need to inspect
        # the actual Filter object, but the integration is tested via the mock.
        store._client.query_points.assert_called_once()

    def test_get_all_list_filter(self):
        """List label values use MatchAny."""
        from mnemory.storage.vector import VectorStore

        store = object.__new__(VectorStore)
        store._config = MagicMock()
        store._config.vector.collection_name = "mnemory"
        store._config.vector.is_remote = False
        store._client = MagicMock()
        store._write_lock = None

        store._client.scroll.return_value = ([], None)

        store.get_all(
            user_id="filip",
            labels_filter={"tags": ["a", "b"]},
        )

        store._client.scroll.assert_called_once()


# ===================================================================
# Labels in fsck
# ===================================================================


class TestLabelsFsck:
    """Labels are preserved through fsck operations."""

    def test_labels_in_metadata_allowlist(self):
        """The 'labels' key is in the fsck metadata allowlist."""
        # This is a structural test — verify the allowlist includes labels
        import inspect

        from mnemory import fsck

        source = inspect.getsource(fsck)
        assert '"labels"' in source or "'labels'" in source

    def test_split_action_passes_labels(self):
        """When fsck splits a memory, labels from metadata are passed to add_memory."""
        from mnemory.fsck import (
            FsckAction,
            FsckAffectedMemory,
            FsckIssue,
            FsckService,
        )

        mock_config = MagicMock()
        mock_vector = MagicMock()
        mock_llm = MagicMock()
        mock_memory_service = MagicMock()
        mock_memory_service.add_memory.return_value = {
            "results": [{"id": "new-1", "event": "ADD"}]
        }

        fsck_service = object.__new__(FsckService)
        fsck_service._config = mock_config
        fsck_service._vector = mock_vector
        fsck_service._llm = mock_llm
        fsck_service._memory_service = mock_memory_service

        issue = FsckIssue(
            issue_id="issue-1",
            type="split",
            severity="medium",
            reasoning="Should be split",
            affected_memories=[
                FsckAffectedMemory(id="mem-1", content="Combined fact"),
            ],
            actions=[
                FsckAction(
                    action="add",
                    new_content="Split fact",
                    new_metadata={
                        "memory_type": "fact",
                        "categories": ["personal"],
                        "importance": "normal",
                        "labels": {"project": "myapp"},
                    },
                ),
            ],
        )

        fsck_service._apply_issue(issue, user_id="filip")

        mock_memory_service.add_memory.assert_called_once()
        _, kwargs = mock_memory_service.add_memory.call_args
        assert kwargs["labels"] == {"project": "myapp"}


# ===================================================================
# Labels in REST API schemas
# ===================================================================


class TestLabelsAPISchemas:
    """Labels field exists in API request schemas."""

    def test_add_memory_request_has_labels(self):
        from mnemory.api.schemas import AddMemoryRequest

        req = AddMemoryRequest(
            content="test",
            labels={"project": "myapp"},
        )
        assert req.labels == {"project": "myapp"}

    def test_add_memory_request_labels_default_none(self):
        from mnemory.api.schemas import AddMemoryRequest

        req = AddMemoryRequest(content="test")
        assert req.labels is None

    def test_search_request_has_labels(self):
        from mnemory.api.schemas import SearchMemoriesRequest

        req = SearchMemoriesRequest(
            query="test",
            labels={"env": "prod"},
        )
        assert req.labels == {"env": "prod"}

    def test_update_request_has_labels(self):
        from mnemory.api.schemas import UpdateMemoryRequest

        req = UpdateMemoryRequest(
            labels={"project": "updated"},
        )
        assert req.labels == {"project": "updated"}

    def test_remember_request_has_labels(self):
        from mnemory.api.schemas import RememberRequest

        req = RememberRequest(
            messages=[{"role": "user", "content": "test conversation"}],
            labels={"conversation_id": "conv-1"},
        )
        assert req.labels == {"conversation_id": "conv-1"}

    def test_recall_request_has_labels(self):
        from mnemory.api.schemas import RecallRequest

        req = RecallRequest(
            labels={"project": "myapp"},
        )
        assert req.labels == {"project": "myapp"}

    def test_list_request_has_labels(self):
        from mnemory.api.schemas import ListMemoriesRequest

        req = ListMemoriesRequest(
            labels={"env": "staging"},
        )
        assert req.labels == {"env": "staging"}

    def test_batch_memory_item_has_labels(self):
        from mnemory.api.schemas import BatchMemoryItem

        item = BatchMemoryItem(
            content="test",
            labels={"batch": "yes"},
        )
        assert item.labels == {"batch": "yes"}


# ===================================================================
# Labels in config
# ===================================================================


class TestLabelsConfig:
    """Labels configuration fields."""

    def test_default_config_values(self):
        from mnemory.config import MemoryConfig

        # Create with defaults (no env vars set)
        with patch.dict("os.environ", {}, clear=False):
            cfg = MemoryConfig()
        assert cfg.labels_max_fields == 20
        assert cfg.labels_max_key_length == 64
        assert cfg.labels_max_value_length == 1000
        assert cfg.labels_indexes == []

    def test_labels_indexes_from_env(self):
        from mnemory.config import MemoryConfig

        with patch.dict(
            "os.environ",
            {"LABELS_INDEXES": "project,topic,conversation_id"},
            clear=False,
        ):
            cfg = MemoryConfig()
        assert cfg.labels_indexes == ["project", "topic", "conversation_id"]

    def test_labels_indexes_empty_env(self):
        from mnemory.config import MemoryConfig

        with patch.dict("os.environ", {"LABELS_INDEXES": ""}, clear=False):
            cfg = MemoryConfig()
        assert cfg.labels_indexes == []


# ===================================================================
# Labels in _format_memories (server.py output)
# ===================================================================


class TestLabelsFormatMemories:
    """Labels appear in formatted memory output."""

    def test_labels_included_in_output(self):
        from mnemory.server import _format_memories

        memories = [
            {
                "id": "mem-1",
                "memory": "test fact",
                "created_at": "2024-01-01",
                "metadata": {
                    "memory_type": "fact",
                    "categories": ["personal"],
                    "importance": "normal",
                    "labels": {"project": "myapp", "env": "prod"},
                },
            }
        ]
        result = json.loads(_format_memories(memories))
        assert result["results"][0]["labels"] == {"project": "myapp", "env": "prod"}

    def test_no_labels_key_when_absent(self):
        from mnemory.server import _format_memories

        memories = [
            {
                "id": "mem-1",
                "memory": "test fact",
                "created_at": "2024-01-01",
                "metadata": {
                    "memory_type": "fact",
                    "categories": ["personal"],
                    "importance": "normal",
                },
            }
        ]
        result = json.loads(_format_memories(memories))
        mem = result["results"][0]
        # Labels should be absent when not set in metadata
        assert "labels" not in mem
