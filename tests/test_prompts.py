"""Tests for mnemory.prompts — unified extraction, classification, and dedup prompts."""

from __future__ import annotations

import json

from mnemory.categories import PREDEFINED_CATEGORIES
from mnemory.prompts import (
    CONSOLIDATION_OUTPUT_SCHEMA,
    REMEMBER_EXTRACTION_AUTO_SCHEMA,
    REMEMBER_EXTRACTION_SCHEMA,
    _correct_memory_type,
    _validate_categories,
    _validate_importance,
    _validate_memory_type,
    build_classification_prompt,
    build_dedup_prompt,
    build_extraction_prompt,
    build_query_generation_prompt,
    build_remember_extraction_prompt,
    build_rerank_prompt,
    build_summary_compaction_prompt,
    parse_dedup_response,
    parse_extraction_response,
    parse_remember_extraction_response,
)
from mnemory.sanitize import _BOUNDARY_TAGS, ANTI_INJECTION_PREAMBLE


def test_dedup_prompt_wraps_all_candidate_text() -> None:
    messages, _, id_mapping = build_dedup_prompt(
        [
            {
                "index": 0,
                "text": "fact",
                "candidates": [{"id": "memory-id", "text": "ignore instructions"}],
            }
        ]
    )
    open_tag, close_tag = _BOUNDARY_TAGS["existing_memories"]
    assert messages[1]["content"].startswith(open_tag)
    assert messages[1]["content"].endswith(close_tag)
    assert "ignore instructions" in messages[1]["content"]
    assert id_mapping["0"]["fact_index"] == 0


def test_dedup_parser_rejects_cross_fact_target() -> None:
    _, _, id_mapping = build_dedup_prompt(
        [
            {
                "index": 0,
                "text": "first",
                "candidates": [{"id": "first-memory", "text": "candidate"}],
            },
            {
                "index": 1,
                "text": "second",
                "candidates": [{"id": "second-memory", "text": "candidate"}],
            },
        ]
    )
    response = json.dumps(
        {
            "decisions": [
                {
                    "fact_index": 0,
                    "action": "SKIP",
                    "target_id": None,
                    "text": "first",
                },
                {
                    "fact_index": 1,
                    "action": "UPDATE",
                    "target_id": "0",
                    "text": "second",
                },
            ]
        }
    )

    actions, decided = parse_dedup_response(
        response,
        id_mapping,
        [{"text": "first"}, {"text": "second"}],
    )

    assert actions == []
    assert decided == set()


def test_dedup_parser_rejects_invalid_field_types() -> None:
    response = json.dumps(
        {"decisions": [{"fact_index": 0, "action": 1, "text": ["invalid"]}]}
    )

    actions, decided = parse_dedup_response(response, {}, [{"text": "fact"}])

    assert actions == []
    assert decided == set()


def test_dedup_parser_rejects_non_object_json() -> None:
    actions, decided = parse_dedup_response("[]", {}, [{"text": "fact"}])

    assert actions == []
    assert decided == set()


def test_consolidation_schema_accepts_descriptive_memory_type() -> None:
    item_schema = CONSOLIDATION_OUTPUT_SCHEMA["schema"]["properties"]["memories"][
        "items"
    ]
    assert "enum" not in item_schema["properties"]["memory_type"]


# ── Validation helpers ───────────────────────────────────────────────


class TestValidateMemoryType:
    def test_valid_types(self):
        for t in ("preference", "fact", "episodic", "procedural", "context"):
            assert _validate_memory_type(t) == t

    def test_invalid_type_falls_back(self):
        assert _validate_memory_type("invalid") == "episodic"

    def test_none_falls_back(self):
        assert _validate_memory_type(None) == "episodic"


class TestCorrectMemoryType:
    """Tests for the post-LLM heuristic that demotes fact → episodic/context."""

    # ── Episodic corrections (fact → episodic) ───────────────────────

    def test_wants_to_becomes_episodic(self):
        assert (
            _correct_memory_type("fact", "User wants to add distributed tracing")
            == "episodic"
        )

    def test_want_to_becomes_episodic(self):
        assert (
            _correct_memory_type("fact", "User want to migrate to Kubernetes")
            == "episodic"
        )

    def test_decided_to_becomes_episodic(self):
        assert (
            _correct_memory_type("fact", "User decided to use PostgreSQL") == "episodic"
        )

    def test_plans_to_becomes_episodic(self):
        assert (
            _correct_memory_type("fact", "User plans to migrate to Kubernetes")
            == "episodic"
        )

    def test_planning_to_becomes_episodic(self):
        assert (
            _correct_memory_type("fact", "User is planning to refactor the auth module")
            == "episodic"
        )

    def test_is_working_on_becomes_episodic(self):
        assert (
            _correct_memory_type("fact", "User is working on the billing service")
            == "episodic"
        )

    def test_intends_to_becomes_episodic(self):
        assert (
            _correct_memory_type("fact", "User intends to deploy next week")
            == "episodic"
        )

    def test_aims_to_becomes_episodic(self):
        assert (
            _correct_memory_type("fact", "User aims to reduce latency by 50%")
            == "episodic"
        )

    # ── Context corrections (fact → context) ─────────────────────────

    def test_currently_lacks_becomes_context(self):
        assert (
            _correct_memory_type("fact", "mnemory currently lacks provenance tracking")
            == "context"
        )

    def test_currently_has_no_becomes_context(self):
        assert (
            _correct_memory_type(
                "fact", "mnemory currently has no provenance tracking system"
            )
            == "context"
        )

    def test_currently_has_becomes_context(self):
        assert (
            _correct_memory_type("fact", "mnemory currently has basic search")
            == "context"
        )

    def test_currently_uses_becomes_context(self):
        assert (
            _correct_memory_type(
                "fact", "The project currently uses Qdrant for vectors"
            )
            == "context"
        )

    def test_does_not_have_becomes_context(self):
        assert (
            _correct_memory_type("fact", "The project does not have rollback mechanism")
            == "context"
        )

    def test_doesnt_have_becomes_context(self):
        assert (
            _correct_memory_type("fact", "The system doesn't have monitoring")
            == "context"
        )

    def test_defaults_to_becomes_context(self):
        assert (
            _correct_memory_type("fact", "The model defaults to gpt-5-mini")
            == "context"
        )

    def test_does_not_support_becomes_context(self):
        assert (
            _correct_memory_type("fact", "The API does not support pagination")
            == "context"
        )

    def test_doesnt_support_becomes_context(self):
        assert (
            _correct_memory_type("fact", "The system doesn't support WebSocket")
            == "context"
        )

    # ── No correction (real facts stay as facts) ─────────────────────

    def test_biographical_name_stays_fact(self):
        assert _correct_memory_type("fact", "User's name is Elena") == "fact"

    def test_biographical_location_stays_fact(self):
        assert _correct_memory_type("fact", "User lives in Prague") == "fact"

    def test_biographical_job_stays_fact(self):
        assert (
            _correct_memory_type("fact", "User works at Google as a software engineer")
            == "fact"
        )

    def test_relationship_stays_fact(self):
        assert _correct_memory_type("fact", "User's partner is Romana") == "fact"

    def test_stable_trait_stays_fact(self):
        assert (
            _correct_memory_type(
                "fact", "User has over 14 years of experience in DevOps"
            )
            == "fact"
        )

    # ── Non-fact types are never touched ─────────────────────────────

    def test_episodic_not_touched(self):
        assert (
            _correct_memory_type("episodic", "User wants to add tracing") == "episodic"
        )

    def test_context_not_touched(self):
        assert (
            _correct_memory_type("context", "The model defaults to gpt-5-mini")
            == "context"
        )

    def test_preference_not_touched(self):
        assert (
            _correct_memory_type("preference", "User prefers dark mode") == "preference"
        )

    def test_procedural_not_touched(self):
        assert (
            _correct_memory_type("procedural", "User plans to run tests first")
            == "procedural"
        )

    # ── Case insensitivity ───────────────────────────────────────────

    def test_case_insensitive_wants(self):
        assert (
            _correct_memory_type("fact", "User WANTS TO add monitoring") == "episodic"
        )

    def test_case_insensitive_defaults(self):
        assert (
            _correct_memory_type("fact", "The config DEFAULTS TO gpt-5-mini")
            == "context"
        )


class TestValidateCategories:
    def test_valid_list(self):
        assert _validate_categories(["work", "technical"]) == ["work", "technical"]

    def test_non_list_returns_empty(self):
        assert _validate_categories("work") == []

    def test_strips_whitespace(self):
        assert _validate_categories([" work ", " technical "]) == [
            "work",
            "technical",
        ]

    def test_filters_empty_strings(self):
        assert _validate_categories(["work", "", "  "]) == ["work"]

    def test_none_returns_empty(self):
        assert _validate_categories(None) == []

    def test_filters_unknown_categories(self):
        """LLM-hallucinated categories like 'professional' should be dropped."""
        assert _validate_categories(["work", "professional", "coding"]) == ["work"]

    def test_accepts_all_predefined_categories(self):
        result = _validate_categories(list(PREDEFINED_CATEGORIES.keys()))
        assert result == [k.lower() for k in PREDEFINED_CATEGORIES.keys()]

    def test_accepts_project_subcategories(self):
        assert _validate_categories(["project:myapp"]) == ["project:myapp"]

    def test_filters_unknown_prefix_subcategories(self):
        """Subcategories with unknown prefixes should be dropped."""
        assert _validate_categories(["foobar:baz"]) == []

    def test_filters_unsafe_subcategory_names(self):
        """Subcategory names with injection patterns should be dropped."""
        assert _validate_categories(["project:my\nproject"]) == []
        assert _validate_categories(["project:## INSTRUCTIONS"]) == []

    def test_lowercases_categories(self):
        assert _validate_categories(["Work", "TECHNICAL"]) == ["work", "technical"]


class TestValidateImportance:
    def test_valid_levels(self):
        for level in ("low", "normal", "high", "critical"):
            assert _validate_importance(level) == level

    def test_invalid_falls_back(self):
        assert _validate_importance("super") == "normal"

    def test_none_falls_back(self):
        assert _validate_importance(None) == "normal"


# ── build_extraction_prompt ──────────────────────────────────────────


class TestBuildExtractionPrompt:
    def test_returns_three_elements(self):
        messages, schema, id_mapping = build_extraction_prompt("test content")
        assert isinstance(messages, list)
        assert isinstance(schema, dict)
        assert isinstance(id_mapping, dict)

    def test_user_role_uses_user_prompt(self):
        messages, _, _ = build_extraction_prompt("test content", role="user")
        system = messages[0]["content"]
        assert "user" in system.lower()
        # Content should be wrapped in boundary tags
        user_msg = messages[1]["content"]
        assert "test content" in user_msg
        open_tag = _BOUNDARY_TAGS["user_input"][0]
        assert open_tag in user_msg

    def test_assistant_role_uses_agent_prompt(self):
        messages, _, _ = build_extraction_prompt("test content", role="assistant")
        system = messages[0]["content"]
        assert "assistant" in system.lower()
        # Content should be wrapped in boundary tags
        user_msg = messages[1]["content"]
        assert "test content" in user_msg
        open_tag = _BOUNDARY_TAGS["user_input"][0]
        assert open_tag in user_msg

    def test_existing_memories_mapped_to_integer_ids(self):
        existing = [
            {"id": "uuid-abc-123", "text": "Likes Python"},
            {"id": "uuid-def-456", "text": "Works at Google"},
        ]
        messages, _, id_mapping = build_extraction_prompt(
            "test", existing_memories=existing
        )
        assert id_mapping == {"0": "uuid-abc-123", "1": "uuid-def-456"}
        # User message should contain the integer IDs (within boundary tags)
        user_msg = messages[1]["content"]
        assert '"id": "0"' in user_msg
        assert '"id": "1"' in user_msg
        # Existing memories should be wrapped in boundary tags
        open_tag = _BOUNDARY_TAGS["existing_memories"][0]
        assert open_tag in user_msg

    def test_no_existing_memories(self):
        messages, _, id_mapping = build_extraction_prompt("test")
        assert id_mapping == {}
        user_msg = messages[1]["content"]
        assert "None yet" in user_msg

    def test_available_categories_in_prompt(self):
        cats = ["personal", "work", "project:myapp"]
        messages, _, _ = build_extraction_prompt("test", available_categories=cats)
        user_msg = messages[1]["content"]
        assert "project:myapp" in user_msg

    def test_explicit_fields_in_prompt(self):
        messages, _, _ = build_extraction_prompt(
            "test", explicit_fields={"memory_type": "fact", "pinned": True}
        )
        user_msg = messages[1]["content"]
        assert "Caller-Provided" in user_msg
        assert '"memory_type"' in user_msg
        assert '"fact"' in user_msg

    def test_schema_has_required_structure(self):
        _, schema, _ = build_extraction_prompt("test")
        assert schema["name"] == "memory_extraction"
        assert "memories" in schema["schema"]["properties"]

    def test_today_date_in_prompt(self):
        messages, _, _ = build_extraction_prompt("test")
        user_msg = messages[1]["content"]
        # Should contain a date like 2026-02-17 in the user message
        import re

        assert re.search(r"\d{4}-\d{2}-\d{2}", user_msg)

    def test_conversation_extraction_rules_in_user_prompt(self):
        """User prompt should guide extraction to focus on user facts
        in user/assistant conversations and filter assistant noise."""
        messages, _, _ = build_extraction_prompt("test", role="user")
        system = messages[0]["content"]
        # Should instruct to extract user facts from conversations
        assert "user and an AI assistant" in system
        # Should warn against extracting assistant reasoning
        assert "assistant's own reasoning" in system.lower() or (
            "assistant" in system and "not facts to remember" in system.lower()
        )
        # Should allow extracting from assistant when confirming user facts
        assert "paraphrase" in system
        # Should preserve multi-person transcript extraction
        assert "multi-person" in system

    def test_conversation_rules_not_in_agent_prompt(self):
        """Agent prompt should NOT contain user/assistant conversation rules."""
        messages, _, _ = build_extraction_prompt("test", role="assistant")
        system = messages[0]["content"]
        # Agent prompt focuses on assistant identity, not conversation filtering
        assert "not facts to remember" not in system.lower()

    def test_english_extraction_required(self):
        """Both user and agent prompts should require English extraction."""
        for role in ("user", "assistant"):
            messages, _, _ = build_extraction_prompt("test", role=role)
            system = messages[0]["content"]
            assert "always write extracted facts in english" in system.lower(), (
                f"role={role}: prompt should require English extraction"
            )
            # Should NOT contain the old same-language rule
            assert "record facts in the same language" not in system.lower(), (
                f"role={role}: prompt should not contain old same-language rule"
            )

    def test_third_party_extraction_guidance(self):
        """User prompt should guide extraction of facts about people and
        things the user mentions (family, pets, possessions, etc.)."""
        messages, _, _ = build_extraction_prompt("test", role="user")
        system = messages[0]["content"]
        # Should mention extracting facts about family/friends/etc.
        assert "family" in system.lower()
        assert "pets" in system.lower()
        # Should instruct relationship-based subjects
        assert "User's mother" in system or "user's mother" in system.lower()
        # Should have a conversation example with third-party extraction
        assert "User's mother" in system


# ── parse_extraction_response ────────────────────────────────────────


class TestParseExtractionResponse:
    def test_add_action(self):
        response = json.dumps(
            {
                "memories": [
                    {
                        "text": "Name is John",
                        "action": "ADD",
                        "target_id": None,
                        "old_memory": None,
                        "memory_type": "fact",
                        "categories": ["personal"],
                        "importance": "normal",
                        "pinned": True,
                    }
                ],
                "store_artifact": False,
            }
        )
        results, store_artifact = parse_extraction_response(response, {})
        assert len(results) == 1
        assert results[0]["action"] == "ADD"
        assert results[0]["text"] == "Name is John"
        assert results[0]["target_id"] is None
        assert results[0]["pinned"] is True
        assert store_artifact is False

    def test_update_action_maps_id(self):
        response = json.dumps(
            {
                "memories": [
                    {
                        "text": "Uses Neovim",
                        "action": "UPDATE",
                        "target_id": "0",
                        "old_memory": "Uses VS Code",
                        "memory_type": "preference",
                        "categories": ["technical"],
                        "importance": "normal",
                        "pinned": False,
                    }
                ],
                "store_artifact": False,
            }
        )
        id_mapping = {"0": "real-uuid-123"}
        results, _ = parse_extraction_response(response, id_mapping)
        assert len(results) == 1
        assert results[0]["action"] == "UPDATE"
        assert results[0]["target_id"] == "real-uuid-123"
        assert results[0]["old_memory"] == "Uses VS Code"

    def test_inference_delete_fails_closed(self):
        response = json.dumps(
            {
                "memories": [
                    {
                        "text": "Old fact",
                        "action": "DELETE",
                        "target_id": "1",
                        "old_memory": None,
                        "memory_type": "fact",
                        "categories": [],
                        "importance": "normal",
                        "pinned": False,
                    }
                ],
                "store_artifact": False,
            }
        )
        id_mapping = {"0": "uuid-a", "1": "uuid-b"}
        results, _ = parse_extraction_response(response, id_mapping)
        assert results == []

    def test_none_alias_is_returned_as_skip(self):
        response = json.dumps(
            {
                "memories": [
                    {
                        "text": "Already known",
                        "action": "NONE",
                        "target_id": None,
                        "old_memory": None,
                        "memory_type": "fact",
                        "categories": [],
                        "importance": "normal",
                        "pinned": False,
                    }
                ],
                "store_artifact": False,
            }
        )
        results, _ = parse_extraction_response(response, {})
        assert results[0]["action"] == "SKIP"

    def test_empty_text_filtered_out(self):
        response = json.dumps(
            {
                "memories": [
                    {
                        "text": "",
                        "action": "ADD",
                        "target_id": None,
                        "old_memory": None,
                        "memory_type": "fact",
                        "categories": [],
                        "importance": "normal",
                        "pinned": False,
                    }
                ],
                "store_artifact": False,
            }
        )
        results, _ = parse_extraction_response(response, {})
        assert len(results) == 0

    def test_unknown_target_id_skipped(self):
        response = json.dumps(
            {
                "memories": [
                    {
                        "text": "Updated fact",
                        "action": "UPDATE",
                        "target_id": "99",
                        "old_memory": "Old",
                        "memory_type": "fact",
                        "categories": [],
                        "importance": "normal",
                        "pinned": False,
                    }
                ],
                "store_artifact": False,
            }
        )
        id_mapping = {"0": "uuid-a"}
        results, _ = parse_extraction_response(response, id_mapping)
        assert len(results) == 0  # Skipped because "99" not in mapping

    def test_invalid_json_returns_empty(self):
        results, store_artifact = parse_extraction_response("not json", {})
        assert results == []
        assert store_artifact is False

    def test_empty_memories_list(self):
        response = json.dumps({"memories": [], "store_artifact": False})
        results, _ = parse_extraction_response(response, {})
        assert results == []

    def test_invalid_action_skipped(self):
        response = json.dumps(
            {
                "memories": [
                    {
                        "text": "test",
                        "action": "INVALID",
                        "target_id": None,
                        "old_memory": None,
                        "memory_type": "fact",
                        "categories": [],
                        "importance": "normal",
                        "pinned": False,
                    }
                ],
                "store_artifact": False,
            }
        )
        results, _ = parse_extraction_response(response, {})
        assert len(results) == 0

    def test_multiple_actions(self):
        response = json.dumps(
            {
                "memories": [
                    {
                        "text": "New fact",
                        "action": "ADD",
                        "target_id": None,
                        "old_memory": None,
                        "memory_type": "fact",
                        "categories": ["personal"],
                        "importance": "normal",
                        "pinned": False,
                    },
                    {
                        "text": "Updated pref",
                        "action": "UPDATE",
                        "target_id": "0",
                        "old_memory": "Old pref",
                        "memory_type": "preference",
                        "categories": [],
                        "importance": "high",
                        "pinned": True,
                    },
                ],
                "store_artifact": False,
            }
        )
        id_mapping = {"0": "uuid-existing"}
        results, _ = parse_extraction_response(response, id_mapping)
        assert len(results) == 2
        assert results[0]["action"] == "ADD"
        assert results[1]["action"] == "UPDATE"
        assert results[1]["target_id"] == "uuid-existing"

    def test_invalid_classification_fields_get_defaults(self):
        response = json.dumps(
            {
                "memories": [
                    {
                        "text": "test fact",
                        "action": "ADD",
                        "target_id": None,
                        "old_memory": None,
                        "memory_type": "invalid_type",
                        "categories": "not_a_list",
                        "importance": "super_important",
                        "pinned": False,
                    }
                ],
                "store_artifact": False,
            }
        )
        results, _ = parse_extraction_response(response, {})
        assert len(results) == 1
        assert results[0]["memory_type"] == "episodic"  # Default (safe fallback)
        assert results[0]["categories"] == []  # Default
        assert results[0]["importance"] == "normal"  # Default

    def test_store_artifact_true(self):
        response = json.dumps(
            {
                "memories": [
                    {
                        "text": "Design uses FastAPI",
                        "action": "ADD",
                        "target_id": None,
                        "old_memory": None,
                        "memory_type": "fact",
                        "categories": ["technical"],
                        "importance": "normal",
                        "pinned": False,
                    }
                ],
                "store_artifact": True,
            }
        )
        results, store_artifact = parse_extraction_response(response, {})
        assert len(results) == 1
        assert store_artifact is True

    def test_store_artifact_missing_defaults_false(self):
        """When store_artifact is missing from response, default to False."""
        response = json.dumps(
            {
                "memories": [
                    {
                        "text": "Simple fact",
                        "action": "ADD",
                        "target_id": None,
                        "old_memory": None,
                        "memory_type": "fact",
                        "categories": [],
                        "importance": "normal",
                        "pinned": False,
                    }
                ]
            }
        )
        results, store_artifact = parse_extraction_response(response, {})
        assert len(results) == 1
        assert store_artifact is False


# ── build_classification_prompt ──────────────────────────────────────


class TestBuildClassificationPrompt:
    def test_empty_fields_returns_empty(self):
        messages, schema = build_classification_prompt("test", missing_fields=set())
        assert messages == []
        assert schema is None

    def test_single_field(self):
        messages, schema = build_classification_prompt(
            "test", missing_fields={"memory_type"}
        )
        assert len(messages) == 2
        assert schema is not None
        assert "memory_type" in schema["schema"]["properties"]
        assert "categories" not in schema["schema"]["properties"]

    def test_all_fields(self):
        messages, schema = build_classification_prompt(
            "test",
            missing_fields={"memory_type", "categories", "importance", "pinned"},
        )
        props = schema["schema"]["properties"]
        assert "memory_type" in props
        assert "categories" in props
        assert "importance" in props
        assert "pinned" in props

    def test_available_categories_in_prompt(self):
        messages, _ = build_classification_prompt(
            "test",
            missing_fields={"categories"},
            available_categories=["personal", "work", "project:myapp"],
        )
        system = messages[0]["content"]
        assert "project:myapp" in system

    def test_content_in_user_message(self):
        messages, _ = build_classification_prompt(
            "I prefer Python", missing_fields={"memory_type"}
        )
        # Content should be wrapped in boundary tags
        assert "I prefer Python" in messages[1]["content"]
        open_tag = _BOUNDARY_TAGS["content"][0]
        assert open_tag in messages[1]["content"]


# ── build_query_generation_prompt ────────────────────────────────────


class TestBuildQueryGenerationPrompt:
    def test_returns_messages_and_schema(self):
        messages, schema = build_query_generation_prompt("What car do I have?")
        assert isinstance(messages, list)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        # Question should be wrapped in boundary tags
        assert "What car do I have?" in messages[1]["content"]
        open_tag = _BOUNDARY_TAGS["user_question"][0]
        assert open_tag in messages[1]["content"]
        assert schema["name"] == "query_generation"

    def test_num_queries_in_system_prompt(self):
        messages, _ = build_query_generation_prompt("test", num_queries=7)
        system = messages[0]["content"]
        assert "7" in system

    def test_without_today_no_date_in_prompt(self):
        messages, _ = build_query_generation_prompt("What happened?")
        system = messages[0]["content"]
        # The dynamic "Today's date is YYYY-MM-DD" line should not appear
        # (static references to "Today's date" in examples are fine)
        assert "Today's date is 20" not in system

    def test_with_today_date_in_prompt(self):
        messages, _ = build_query_generation_prompt(
            "What happened last week?", today="2026-02-19"
        )
        system = messages[0]["content"]
        assert "Today's date is 2026-02-19" in system
        assert "temporal references" in system

    def test_temporal_query_guidance_in_prompt(self):
        messages, _ = build_query_generation_prompt("test")
        system = messages[0]["content"]
        assert "date-specific queries" in system

    def test_up_to_wording_in_prompt(self):
        """Prompt should say 'UP TO' to allow fewer or zero queries."""
        messages, _ = build_query_generation_prompt("test", num_queries=5)
        system = messages[0]["content"]
        assert "UP TO" in system

    def test_skip_guidance_in_prompt(self):
        """Prompt should instruct LLM to return empty queries for
        procedural instructions and acknowledgments."""
        messages, _ = build_query_generation_prompt("test")
        system = messages[0]["content"]
        assert "empty queries list" in system
        assert "procedural instruction" in system
        assert "acknowledgment" in system

    def test_without_context_no_context_in_prompt(self):
        messages, _ = build_query_generation_prompt("test")
        system = messages[0]["content"]
        assert "Additional context" not in system

    def test_with_context_injected_into_prompt(self):
        messages, _ = build_query_generation_prompt(
            "test", context="Working directory: /home/user/src/myapp"
        )
        system = messages[0]["content"]
        assert "Additional context:" in system
        assert "Working directory: /home/user/src/myapp" in system
        assert "Do not limit queries exclusively to this context" in system
        # Context should be wrapped in boundary tags
        open_tag = _BOUNDARY_TAGS["context"][0]
        assert open_tag in system

    def test_without_project_categories_no_categories_in_prompt(self):
        messages, _ = build_query_generation_prompt("test")
        system = messages[0]["content"]
        assert "Known project categories" not in system

    def test_with_project_categories_injected_into_prompt(self):
        messages, _ = build_query_generation_prompt(
            "test",
            project_categories=["project:mnemory", "project:myapp"],
        )
        system = messages[0]["content"]
        assert "Known project categories: project:mnemory, project:myapp" in system
        assert "prefer these exact category names" in system

    def test_context_and_categories_together(self):
        messages, _ = build_query_generation_prompt(
            "test",
            context="Working directory: /src/mnemory",
            project_categories=["project:mnemory"],
        )
        system = messages[0]["content"]
        assert "Additional context:" in system
        assert "Working directory: /src/mnemory" in system
        assert "Known project categories: project:mnemory" in system

    def test_empty_project_categories_not_injected(self):
        messages, _ = build_query_generation_prompt("test", project_categories=[])
        system = messages[0]["content"]
        assert "Known project categories" not in system


# ── build_rerank_prompt ──────────────────────────────────────────────


class TestBuildRerankPrompt:
    def test_returns_messages_and_schema(self):
        memories = [{"id": "1", "memory": "User likes dogs"}]
        messages, schema = build_rerank_prompt("Do I like dogs?", memories)
        assert isinstance(messages, list)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert schema["name"] == "memory_rerank"

    def test_question_in_user_content(self):
        memories = [{"id": "1", "memory": "User likes dogs"}]
        messages, _ = build_rerank_prompt("Do I like dogs?", memories)
        user_content = messages[1]["content"]
        assert "Do I like dogs?" in user_content
        # Question should be wrapped in boundary tags
        open_tag = _BOUNDARY_TAGS["user_question"][0]
        assert open_tag in user_content

    def test_memory_text_in_user_content(self):
        memories = [{"id": "1", "memory": "User likes dogs"}]
        messages, _ = build_rerank_prompt("test", memories)
        user_content = messages[1]["content"]
        assert "User likes dogs" in user_content
        # Memories should be wrapped in boundary tags
        open_tag = _BOUNDARY_TAGS["existing_memories"][0]
        assert open_tag in user_content

    def test_without_today_no_date_in_system(self):
        memories = [{"id": "1", "memory": "test"}]
        messages, _ = build_rerank_prompt("test", memories)
        system = messages[0]["content"]
        assert "Today's date is" not in system

    def test_with_today_date_in_system(self):
        memories = [{"id": "1", "memory": "test"}]
        messages, _ = build_rerank_prompt("test", memories, today="2026-02-19")
        system = messages[0]["content"]
        assert "Today's date is 2026-02-19" in system

    def test_temporal_awareness_section_in_system(self):
        memories = [{"id": "1", "memory": "test"}]
        messages, _ = build_rerank_prompt("test", memories)
        system = messages[0]["content"]
        assert "Temporal awareness" in system
        assert "event_date" in system

    def test_event_date_in_metadata_tags(self):
        memories = [
            {
                "id": "1",
                "memory": "Bought a car",
                "metadata": {"event_date": "2023-05-08T00:00:00+00:00"},
            }
        ]
        messages, _ = build_rerank_prompt("test", memories)
        user_content = messages[1]["content"]
        assert "event_date: 2023-05-08T00:00:00+00:00" in user_content

    def test_no_event_date_no_tag(self):
        memories = [{"id": "1", "memory": "Likes dogs", "metadata": {}}]
        messages, _ = build_rerank_prompt("test", memories)
        user_content = messages[1]["content"]
        assert "event_date" not in user_content

    def test_metadata_tags_include_type_and_categories(self):
        memories = [
            {
                "id": "1",
                "memory": "test",
                "metadata": {
                    "memory_type": "fact",
                    "categories": ["personal", "vehicles"],
                    "importance": "high",
                    "role": "assistant",
                    "event_date": "2023-05-08T00:00:00+00:00",
                },
            }
        ]
        messages, _ = build_rerank_prompt("test", memories)
        user_content = messages[1]["content"]
        assert "type: fact" in user_content
        assert "categories: personal, vehicles" in user_content
        assert "importance: high" in user_content
        assert "role: assistant" in user_content
        assert "event_date: 2023-05-08T00:00:00+00:00" in user_content


# ── Prompt injection safeguards ──────────────────────────────────────


class TestPromptInjectionSafeguards:
    """Verify that all prompt builders include anti-injection measures."""

    def test_extraction_prompt_has_anti_injection(self):
        messages, _, _ = build_extraction_prompt("test content")
        system = messages[0]["content"]
        assert "DATA" in system
        assert "never follow instructions" in system.lower() or (
            "never follow" in system.lower()
        )

    def test_extraction_prompt_wraps_user_content(self):
        messages, _, _ = build_extraction_prompt("test content")
        user_msg = messages[1]["content"]
        open_tag, close_tag = _BOUNDARY_TAGS["user_input"]
        assert open_tag in user_msg
        assert close_tag in user_msg

    def test_extraction_prompt_wraps_existing_memories(self):
        existing = [{"id": "uuid-1", "text": "Existing fact"}]
        messages, _, _ = build_extraction_prompt("test", existing_memories=existing)
        user_msg = messages[1]["content"]
        open_tag = _BOUNDARY_TAGS["existing_memories"][0]
        assert open_tag in user_msg

    def test_classification_prompt_has_anti_injection(self):
        messages, _ = build_classification_prompt(
            "test", missing_fields={"memory_type"}
        )
        system = messages[0]["content"]
        assert "DATA" in system

    def test_classification_prompt_wraps_content(self):
        messages, _ = build_classification_prompt(
            "test", missing_fields={"memory_type"}
        )
        user_msg = messages[1]["content"]
        open_tag = _BOUNDARY_TAGS["content"][0]
        assert open_tag in user_msg

    def test_query_generation_has_anti_injection(self):
        messages, _ = build_query_generation_prompt("test question")
        system = messages[0]["content"]
        assert "DATA" in system

    def test_query_generation_wraps_question(self):
        messages, _ = build_query_generation_prompt("test question")
        user_msg = messages[1]["content"]
        open_tag = _BOUNDARY_TAGS["user_question"][0]
        assert open_tag in user_msg

    def test_query_generation_wraps_context(self):
        messages, _ = build_query_generation_prompt("test", context="/home/user/src")
        system = messages[0]["content"]
        open_tag = _BOUNDARY_TAGS["context"][0]
        assert open_tag in system

    def test_rerank_prompt_has_anti_injection(self):
        memories = [{"id": "1", "memory": "test"}]
        messages, _ = build_rerank_prompt("test question", memories)
        system = messages[0]["content"]
        assert "DATA" in system

    def test_rerank_prompt_wraps_question(self):
        memories = [{"id": "1", "memory": "test"}]
        messages, _ = build_rerank_prompt("test question", memories)
        user_msg = messages[1]["content"]
        open_tag = _BOUNDARY_TAGS["user_question"][0]
        assert open_tag in user_msg

    def test_rerank_prompt_wraps_memories(self):
        memories = [{"id": "1", "memory": "test"}]
        messages, _ = build_rerank_prompt("test question", memories)
        user_msg = messages[1]["content"]
        open_tag = _BOUNDARY_TAGS["existing_memories"][0]
        assert open_tag in user_msg

    def test_anti_injection_preamble_consistent(self):
        """The anti-injection preamble should mention boundary tags."""
        assert "boundary tags" in ANTI_INJECTION_PREAMBLE.lower()
        assert "DATA" in ANTI_INJECTION_PREAMBLE


# ── parse_dedup_response ─────────────────────────────────────────────


class TestParseDedupResponse:
    """Test Stage 2 dedup response parsing."""

    def _make_facts(self, n: int = 3) -> list[dict]:
        return [
            {
                "text": f"Fact {i}",
                "memory_type": "fact",
                "categories": ["personal"],
                "importance": "normal",
                "pinned": False,
                "event_date": None,
            }
            for i in range(n)
        ]

    def test_returns_tuple(self):
        """parse_dedup_response should return (actions, decided_indices) tuple."""
        facts = self._make_facts(2)
        response = json.dumps(
            {
                "decisions": [
                    {"fact_index": 0, "action": "ADD", "text": "Fact 0"},
                    {"fact_index": 1, "action": "ADD", "text": "Fact 1"},
                ]
            }
        )
        actions, decided = parse_dedup_response(response, {}, facts)
        assert isinstance(actions, list)
        assert isinstance(decided, set)
        assert len(actions) == 2
        assert decided == {0, 1}

    def test_skip_is_preserved_as_a_transparent_decision(self):
        facts = self._make_facts(3)
        response = json.dumps(
            {
                "decisions": [
                    {"fact_index": 0, "action": "ADD", "text": "Fact 0"},
                    {"fact_index": 1, "action": "SKIP", "text": "Fact 1"},
                    {"fact_index": 2, "action": "ADD", "text": "Fact 2"},
                ]
            }
        )
        actions, decided = parse_dedup_response(response, {}, facts)
        assert len(actions) == 3
        assert actions[1]["action"] == "SKIP"
        assert decided == {0, 1, 2}  # All three tracked

    def test_update_action_with_id_mapping(self):
        """UPDATE actions should map integer IDs back to real UUIDs."""
        facts = self._make_facts(1)
        id_mapping = {"1": "real-uuid-abc"}
        response = json.dumps(
            {
                "decisions": [
                    {
                        "fact_index": 0,
                        "action": "UPDATE",
                        "text": "Updated fact text",
                        "target_id": 1,
                    },
                ]
            }
        )
        actions, decided = parse_dedup_response(response, id_mapping, facts)
        assert len(actions) == 1
        assert actions[0]["action"] == "UPDATE"
        assert actions[0]["target_id"] == "real-uuid-abc"
        assert actions[0]["text"] == "Updated fact text"
        assert decided == {0}

    def test_update_with_unknown_target_invalidates_response(self):
        """An unknown target must force the caller to retry or fail closed."""
        facts = self._make_facts(1)
        id_mapping = {"1": "real-uuid-abc"}
        response = json.dumps(
            {
                "decisions": [
                    {
                        "fact_index": 0,
                        "action": "UPDATE",
                        "text": "Updated text",
                        "target_id": 999,
                    },
                ]
            }
        )
        actions, decided = parse_dedup_response(response, id_mapping, facts)
        assert len(actions) == 0
        assert decided == set()

    def test_unmentioned_facts_not_in_decided(self):
        """Facts not in the response should not appear in decided_indices."""
        facts = self._make_facts(3)
        response = json.dumps(
            {
                "decisions": [
                    {"fact_index": 0, "action": "ADD", "text": "Fact 0"},
                    # fact_index 1 and 2 not mentioned
                ]
            }
        )
        actions, decided = parse_dedup_response(response, {}, facts)
        assert len(actions) == 1
        assert decided == {0}

    def test_invalid_json_returns_empty(self):
        """Invalid JSON should return empty actions and empty decided set."""
        facts = self._make_facts(1)
        actions, decided = parse_dedup_response("not json", {}, facts)
        assert actions == []
        assert decided == set()

    def test_metadata_from_facts(self):
        """Actions should inherit metadata from the original Stage 1 facts."""
        facts = [
            {
                "text": "User likes Python",
                "memory_type": "preference",
                "categories": ["technical"],
                "importance": "high",
                "pinned": True,
                "event_date": "2025-01-15",
            }
        ]
        response = json.dumps(
            {
                "decisions": [
                    {"fact_index": 0, "action": "ADD", "text": "User likes Python"},
                ]
            }
        )
        actions, decided = parse_dedup_response(response, {}, facts)
        assert len(actions) == 1
        assert actions[0]["memory_type"] == "preference"
        assert actions[0]["categories"] == ["technical"]
        assert actions[0]["importance"] == "high"
        assert actions[0]["pinned"] is True
        assert actions[0]["event_date"] == "2025-01-15"


# ── parse_remember_extraction_response ───────────────────────────────


class TestParseRememberExtractionResponse:
    """Test Stage 1 remember extraction response parsing."""

    def test_basic_extraction(self):
        """Should parse facts, summary, and store_artifact."""
        response = json.dumps(
            {
                "memories": [
                    {
                        "text": "User prefers dark mode",
                        "memory_type": "preference",
                        "categories": ["preferences"],
                        "importance": "normal",
                        "pinned": False,
                    }
                ],
                "summary": "User discussed editor preferences.",
                "store_artifact": False,
            }
        )
        facts, summary, store_artifact = parse_remember_extraction_response(response)
        assert len(facts) == 1
        assert facts[0]["text"] == "User prefers dark mode"
        assert facts[0]["memory_type"] == "preference"
        assert summary == "User discussed editor preferences."
        assert store_artifact is False

    def test_empty_memories(self):
        """Empty memories list should return empty facts."""
        response = json.dumps({"memories": [], "summary": "Chitchat."})
        facts, summary, store_artifact = parse_remember_extraction_response(response)
        assert facts == []
        assert summary == "Chitchat."

    def test_invalid_json(self):
        """Invalid JSON should return empty results."""
        facts, summary, store_artifact = parse_remember_extraction_response("not json")
        assert facts == []
        assert summary == ""
        assert store_artifact is False

    def test_event_date_preserved(self):
        """event_date should be preserved in extracted facts."""
        response = json.dumps(
            {
                "memories": [
                    {
                        "text": "Meeting happened",
                        "memory_type": "episodic",
                        "categories": ["work"],
                        "importance": "normal",
                        "pinned": False,
                        "event_date": "2025-06-15",
                    }
                ],
                "summary": "Discussed a meeting.",
            }
        )
        facts, _, _ = parse_remember_extraction_response(response)
        assert facts[0]["event_date"] == "2025-06-15"

    def test_store_artifact_true(self):
        """store_artifact=true should be parsed correctly."""
        response = json.dumps(
            {
                "memories": [
                    {
                        "text": "Architecture discussion",
                        "memory_type": "episodic",
                        "categories": ["technical"],
                        "importance": "normal",
                        "pinned": False,
                    }
                ],
                "summary": "Detailed architecture review.",
                "store_artifact": True,
            }
        )
        _, _, store_artifact = parse_remember_extraction_response(response)
        assert store_artifact is True


# ── build_remember_extraction_prompt ─────────────────────────────────


class TestBuildRememberExtractionPrompt:
    """Test Stage 1 extraction prompt building."""

    def test_basic_prompt(self):
        """Should produce system + user messages with schema."""
        messages, schema = build_remember_extraction_prompt(
            "User said they like Python"
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert schema is not None

    def test_session_context_included(self):
        """Session context should be included in the system prompt."""
        session_context = {
            "extracted_memories": ["User likes Python"],
            "conversation_summary": "Discussed programming preferences.",
        }
        messages, _ = build_remember_extraction_prompt(
            "User also likes Rust",
            session_context=session_context,
        )
        user_msg = messages[1]["content"]
        assert "User likes Python" in user_msg
        assert "Discussed programming preferences" in user_msg

    def test_context_hint_included(self):
        """Context hint should be in the user message."""
        messages, _ = build_remember_extraction_prompt(
            "Working on the API",
            context="Working directory: /home/user/myproject",
        )
        user_msg = messages[1]["content"]
        assert "myproject" in user_msg

    def test_anti_injection_in_prompt(self):
        """Anti-injection preamble should be in the system prompt."""
        messages, _ = build_remember_extraction_prompt("test content")
        system = messages[0]["content"]
        assert "boundary tags" in system.lower() or "DATA" in system

    def test_user_role_uses_user_prompt(self):
        """role='user' (default) should use the user extraction prompt."""
        messages, _ = build_remember_extraction_prompt(
            "User: I like Python", role="user"
        )
        system = messages[0]["content"]
        # User prompt focuses on user facts
        assert "user" in system.lower()
        # Should NOT contain agent-specific language
        assert "about the assistant" not in system.lower()

    def test_assistant_role_uses_agent_prompt(self):
        """role='assistant' should use the agent extraction prompt."""
        messages, _ = build_remember_extraction_prompt(
            "User: What's your name?\nAssistant: I am Aria.",
            role="assistant",
        )
        system = messages[0]["content"]
        # Agent prompt focuses on assistant facts
        assert "about the assistant" in system.lower()

    def test_default_role_is_auto(self):
        """Omitting role should default to auto extraction prompt (role=None)."""
        messages_default, schema_default = build_remember_extraction_prompt(
            "test content"
        )
        messages_auto, schema_auto = build_remember_extraction_prompt(
            "test content", role=None
        )
        assert messages_default[0]["content"] == messages_auto[0]["content"]
        assert schema_default == schema_auto
        # Auto prompt should differ from user prompt
        messages_user, _ = build_remember_extraction_prompt("test content", role="user")
        assert messages_default[0]["content"] != messages_user[0]["content"]

    def test_assistant_role_normalises_plain_firstperson(self):
        """Plain first-person content should be prefixed with 'assistant: '
        when role='assistant', mirroring build_extraction_prompt behaviour."""
        messages, _ = build_remember_extraction_prompt(
            "I prefer concise answers.", role="assistant"
        )
        user_msg = messages[1]["content"]
        assert "assistant: I prefer concise answers." in user_msg

    def test_assistant_role_no_double_prefix(self):
        """Content already prefixed with 'Assistant:' should not be double-prefixed."""
        messages, _ = build_remember_extraction_prompt(
            "Assistant: I prefer concise answers.", role="assistant"
        )
        user_msg = messages[1]["content"]
        # Original casing preserved, no extra prefix prepended
        assert user_msg.lower().count("assistant: i prefer") == 1

    def test_assistant_role_conversation_not_prefixed(self):
        """Conversation content starting with 'User:' should not be modified."""
        messages, _ = build_remember_extraction_prompt(
            "User: Hello\nAssistant: Hi!", role="assistant"
        )
        user_msg = messages[1]["content"]
        assert "assistant: User: Hello" not in user_msg

    def test_user_role_not_normalised(self):
        """role='user' content should never be prefixed with 'assistant: '."""
        messages, _ = build_remember_extraction_prompt(
            "I prefer concise answers.", role="user"
        )
        user_msg = messages[1]["content"]
        assert "assistant: I prefer" not in user_msg

    def test_auto_prompt_treats_pasted_session_data_as_reference_material(self):
        """Auto prompt should tell the model not to re-ingest pasted session exports."""
        messages, _ = build_remember_extraction_prompt("test content", role=None)
        system = messages[0]["content"]
        assert "REFERENCE MATERIAL" in system
        assert "Do NOT re-extract the underlying quoted facts" in system
        assert "session exports" in system

    def test_user_prompt_treats_pasted_session_data_as_reference_material(self):
        """User prompt should tell the model to extract only new review outcomes."""
        messages, _ = build_remember_extraction_prompt("test content", role="user")
        system = messages[0]["content"]
        assert "REFERENCE MATERIAL" in system
        assert "review, correction, approval, or rejection" in system
        assert "elliptical trainer" in system

    def test_assistant_prompt_treats_pasted_session_data_as_reference_material(self):
        """Assistant prompt should treat pasted quoted assistant facts as reference."""
        messages, _ = build_remember_extraction_prompt("test content", role="assistant")
        system = messages[0]["content"]
        assert "REFERENCE MATERIAL" in system
        assert "diagnosis, conclusion, recommendation, or fix proposal" in system
        assert "Do NOT re-extract the underlying quoted facts" in system

    def test_user_prompt_requires_specific_project_names(self):
        """User prompt should reject project names that just repeat broad categories."""
        messages, _ = build_remember_extraction_prompt("test content", role="user")
        system = messages[0]["content"]
        assert "project:<specific-name>" in system
        assert 'use "home" rather than "project:home"' in system

    def test_auto_prompt_requires_specific_project_names(self):
        """Auto prompt should require specific project-scoped category names."""
        messages, _ = build_remember_extraction_prompt("test content", role=None)
        system = messages[0]["content"]
        assert "project:<specific-name>" in system
        assert "repeat a broad category" in system


# ── build_extraction_prompt content normalisation ─────────────────────


class TestAgentContentNormalization:
    """Test that build_extraction_prompt normalises plain first-person content
    for role='assistant' by prepending 'assistant: '."""

    def test_plain_firstperson_gets_prefix(self):
        """Plain first-person content should be prefixed with 'assistant: '."""
        messages, _, _ = build_extraction_prompt(
            "I prefer concise answers.", role="assistant"
        )
        user_msg = messages[1]["content"]
        assert "assistant: I prefer concise answers." in user_msg

    def test_already_prefixed_not_double_prefixed(self):
        """Content already starting with 'assistant:' should not get a second prefix."""
        messages, _, _ = build_extraction_prompt(
            "assistant: I prefer concise answers.", role="assistant"
        )
        user_msg = messages[1]["content"]
        # Should appear exactly once, not twice
        assert user_msg.count("assistant: I prefer") == 1

    def test_conversation_content_not_prefixed(self):
        """Conversation content with 'User:' prefix should not be modified."""
        messages, _, _ = build_extraction_prompt(
            "User: Hello\nassistant: Hi there!", role="assistant"
        )
        user_msg = messages[1]["content"]
        # Should not prepend another 'assistant: ' since it starts with 'User:'
        assert "assistant: User: Hello" not in user_msg

    def test_user_role_not_normalised(self):
        """role='user' content should never be prefixed."""
        messages, _, _ = build_extraction_prompt(
            "I prefer concise answers.", role="user"
        )
        user_msg = messages[1]["content"]
        assert "assistant: I prefer" not in user_msg


# ── build_summary_compaction_prompt ──────────────────────────────────


class TestBuildSummaryCompactionPrompt:
    """Test summary compaction prompt building."""

    def test_basic_prompt(self):
        """Should produce system + user messages with schema."""
        messages, schema = build_summary_compaction_prompt(
            "Long conversation summary..."
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert schema is not None

    def test_anti_injection_in_prompt(self):
        """Anti-injection preamble should be in the system prompt."""
        messages, _ = build_summary_compaction_prompt("summary text")
        system = messages[0]["content"]
        assert "boundary tags" in system.lower() or "DATA" in system

    def test_summary_wrapped_with_boundary(self):
        """Summary input should be wrapped with boundary tags."""
        messages, _ = build_summary_compaction_prompt("my summary content")
        user_msg = messages[1]["content"]
        # Should contain boundary tags around the summary
        assert "summary" in user_msg.lower()
        assert "my summary content" in user_msg


# ── Auto mode (role=None) prompt and schema tests ────────────────────


class TestAutoModePromptAndSchema:
    """Test three-way role routing in build_remember_extraction_prompt."""

    def test_auto_selects_auto_template(self):
        """role=None should select the auto extraction template."""
        messages, schema = build_remember_extraction_prompt(
            "User: I like Python\nAssistant: Great choice!", role=None
        )
        system = messages[0]["content"]
        # Auto prompt should mention extracting from both participants
        assert "user" in system.lower() and "assistant" in system.lower()
        # Auto schema should have per-fact role field
        mem_props = schema["schema"]["properties"]["memories"]["items"]["properties"]
        assert "role" in mem_props
        assert mem_props["role"]["enum"] == ["user", "assistant"]

    def test_auto_schema_returned_for_none(self):
        """role=None should return REMEMBER_EXTRACTION_AUTO_SCHEMA."""
        _, schema = build_remember_extraction_prompt("test", role=None)
        assert schema is REMEMBER_EXTRACTION_AUTO_SCHEMA

    def test_user_schema_returned_for_user(self):
        """role='user' should return REMEMBER_EXTRACTION_SCHEMA (no per-fact role)."""
        _, schema = build_remember_extraction_prompt("test", role="user")
        assert schema is REMEMBER_EXTRACTION_SCHEMA

    def test_assistant_schema_returned_for_assistant(self):
        """role='assistant' should return REMEMBER_EXTRACTION_SCHEMA (no per-fact role)."""
        _, schema = build_remember_extraction_prompt("test", role="assistant")
        assert schema is REMEMBER_EXTRACTION_SCHEMA

    def test_auto_schema_has_role_field(self):
        """Auto schema should have a 'role' field in each memory item."""
        items = REMEMBER_EXTRACTION_AUTO_SCHEMA["schema"]["properties"]["memories"][
            "items"
        ]
        assert "role" in items["required"]
        assert "role" in items["properties"]
        assert items["properties"]["role"]["enum"] == ["user", "assistant"]

    def test_regular_schema_lacks_role_field(self):
        """Regular schema should NOT have a per-fact 'role' field."""
        items = REMEMBER_EXTRACTION_SCHEMA["schema"]["properties"]["memories"]["items"]
        assert "role" not in items.get("required", [])
        assert "role" not in items["properties"]

    def test_auto_template_differs_from_user(self):
        """Auto template should differ from user template."""
        msgs_auto, _ = build_remember_extraction_prompt("test", role=None)
        msgs_user, _ = build_remember_extraction_prompt("test", role="user")
        assert msgs_auto[0]["content"] != msgs_user[0]["content"]

    def test_auto_template_differs_from_assistant(self):
        """Auto template should differ from assistant template."""
        msgs_auto, _ = build_remember_extraction_prompt("test", role=None)
        msgs_asst, _ = build_remember_extraction_prompt("test", role="assistant")
        assert msgs_auto[0]["content"] != msgs_asst[0]["content"]


class TestParseRememberExtractionAutoRole:
    """Test that parse_remember_extraction_response handles per-fact role."""

    def test_role_preserved_from_auto_output(self):
        """Per-fact role from auto extraction should be preserved."""
        response = json.dumps(
            {
                "memories": [
                    {
                        "text": "User likes Python",
                        "role": "user",
                        "memory_type": "preference",
                        "categories": ["technical"],
                        "importance": "normal",
                        "pinned": False,
                        "event_date": None,
                    },
                    {
                        "text": "Assistant recommended FastAPI",
                        "role": "assistant",
                        "memory_type": "episodic",
                        "categories": ["technical"],
                        "importance": "normal",
                        "pinned": False,
                        "event_date": None,
                    },
                ],
                "summary": "Discussed Python frameworks.",
                "store_artifact": False,
            }
        )
        facts, summary, _ = parse_remember_extraction_response(response)
        assert len(facts) == 2
        assert facts[0]["role"] == "user"
        assert facts[1]["role"] == "assistant"

    def test_missing_role_not_added(self):
        """Facts without a role field (user/assistant explicit mode) should not have role key."""
        response = json.dumps(
            {
                "memories": [
                    {
                        "text": "User likes hiking",
                        "memory_type": "preference",
                        "categories": ["personal"],
                        "importance": "normal",
                        "pinned": False,
                        "event_date": None,
                    }
                ],
                "summary": "Outdoor activities.",
                "store_artifact": False,
            }
        )
        facts, _, _ = parse_remember_extraction_response(response)
        assert len(facts) == 1
        assert "role" not in facts[0]

    def test_invalid_role_defaults_to_user(self):
        """Invalid role value should default to 'user'."""
        response = json.dumps(
            {
                "memories": [
                    {
                        "text": "Some fact",
                        "role": "system",
                        "memory_type": "fact",
                        "categories": ["personal"],
                        "importance": "normal",
                        "pinned": False,
                        "event_date": None,
                    }
                ],
                "summary": "Summary.",
                "store_artifact": False,
            }
        )
        facts, _, _ = parse_remember_extraction_response(response)
        assert facts[0]["role"] == "user"


class TestParseDedupResponseRoleCarry:
    """Test that parse_dedup_response carries per-fact role through to actions."""

    def test_role_carried_from_fact_to_action(self):
        """Role from Stage 1 fact should appear in the action dict."""
        facts = [
            {
                "text": "User likes Python",
                "role": "user",
                "memory_type": "preference",
                "categories": ["technical"],
                "importance": "normal",
                "pinned": False,
                "event_date": None,
            },
            {
                "text": "Assistant recommended FastAPI",
                "role": "assistant",
                "memory_type": "episodic",
                "categories": ["technical"],
                "importance": "normal",
                "pinned": False,
                "event_date": None,
            },
        ]
        response = json.dumps(
            {
                "decisions": [
                    {"fact_index": 0, "action": "ADD", "text": "User likes Python"},
                    {
                        "fact_index": 1,
                        "action": "ADD",
                        "text": "Assistant recommended FastAPI",
                    },
                ]
            }
        )
        actions, decided = parse_dedup_response(response, {}, facts)
        assert len(actions) == 2
        assert actions[0]["role"] == "user"
        assert actions[1]["role"] == "assistant"

    def test_no_role_in_fact_means_no_role_in_action(self):
        """Facts without role (explicit mode) should not add role to action."""
        facts = [
            {
                "text": "User likes hiking",
                "memory_type": "preference",
                "categories": ["personal"],
                "importance": "normal",
                "pinned": False,
                "event_date": None,
            }
        ]
        response = json.dumps(
            {
                "decisions": [
                    {"fact_index": 0, "action": "ADD", "text": "User likes hiking"},
                ]
            }
        )
        actions, _ = parse_dedup_response(response, {}, facts)
        assert len(actions) == 1
        assert "role" not in actions[0]
