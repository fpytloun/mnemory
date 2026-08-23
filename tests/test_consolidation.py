"""Unit tests for the consolidation service."""

from __future__ import annotations

from unittest.mock import MagicMock

from mnemory.consolidation import ConsolidationService


class TestGlobalConsolidationCandidateScan:
    """Tests for maintenance candidate discovery across all users."""

    def test_follows_all_cursors_and_requests_only_required_payload(self):
        from mnemory.storage.vector import SessionSummaryStore

        store = SessionSummaryStore.__new__(SessionSummaryStore)
        store._client = MagicMock()

        def point(index: int) -> MagicMock:
            value = MagicMock()
            value.payload = {
                "session_id": f"s{index}",
                "user_id": f"u{index % 2}",
                "consolidation_state": "idle",
            }
            return value

        store._client.scroll.side_effect = [
            ([point(index) for index in range(120)], "cursor-1"),
            ([point(index) for index in range(120, 263)], None),
        ]

        sessions = store.scan_consolidation_candidates()

        assert len(sessions) == 263
        assert store._client.scroll.call_count == 2
        first = store._client.scroll.call_args_list[0].kwargs
        second = store._client.scroll.call_args_list[1].kwargs
        assert first["with_vectors"] is False
        assert first["with_payload"] == [
            "session_id",
            "user_id",
            "memory_ids",
            "updated_at",
            "consolidation_state",
            "consolidated_memory_ids",
        ]
        assert second["offset"] == "cursor-1"


def _make_service(vector_mock=None):
    """Create a ConsolidationService with mocked dependencies."""
    service = ConsolidationService.__new__(ConsolidationService)
    service._vector = vector_mock or MagicMock()
    service._llm = MagicMock()
    service._embedding = MagicMock()
    service._memory = MagicMock()
    service._sessions = MagicMock()
    service._collector = None
    service._config = MagicMock()
    return service


class TestFetchRawMemories:
    """Tests for ConsolidationService._fetch_raw_memories."""

    def test_includes_memories_without_layer(self):
        """Memories without memory_layer field should be treated as raw."""
        vector = MagicMock()
        vector.get_by_id.return_value = {
            "id": "mem-1",
            "memory": "User likes Python",
            "metadata": {
                "memory_type": "preference",
                # No memory_layer field
            },
        }
        service = _make_service(vector)
        result = service._fetch_raw_memories(["mem-1"], "user-1")
        assert len(result) == 1
        assert result[0]["id"] == "mem-1"

    def test_includes_explicit_raw(self):
        """Memories with memory_layer='raw' should be included."""
        vector = MagicMock()
        vector.get_by_id.return_value = {
            "id": "mem-2",
            "memory": "User prefers dark mode",
            "metadata": {"memory_layer": "raw"},
        }
        service = _make_service(vector)
        result = service._fetch_raw_memories(["mem-2"], "user-1")
        assert len(result) == 1

    def test_excludes_consolidated(self):
        """Memories with memory_layer='consolidated' should be excluded."""
        vector = MagicMock()
        vector.get_by_id.return_value = {
            "id": "mem-3",
            "memory": "User lives in Prague",
            "metadata": {"memory_layer": "consolidated"},
        }
        service = _make_service(vector)
        result = service._fetch_raw_memories(["mem-3"], "user-1")
        assert len(result) == 0

    def test_excludes_superseded(self):
        """Superseded raw memories should be excluded."""
        vector = MagicMock()
        vector.get_by_id.return_value = {
            "id": "mem-4",
            "memory": "User likes Java",
            "metadata": {
                "memory_layer": "raw",
                "superseded_by": "consolidated-1",
            },
        }
        service = _make_service(vector)
        result = service._fetch_raw_memories(["mem-4"], "user-1")
        assert len(result) == 0

    def test_skips_missing_memories(self):
        """Memory IDs that don't exist should be skipped."""
        vector = MagicMock()
        vector.get_by_id.return_value = None
        service = _make_service(vector)
        result = service._fetch_raw_memories(["nonexistent"], "user-1")
        assert len(result) == 0

    def test_handles_exceptions(self):
        """Exceptions from vector.get_by_id should be caught and skipped."""
        vector = MagicMock()
        vector.get_by_id.side_effect = [
            Exception("Connection error"),
            {
                "id": "mem-5",
                "memory": "User likes Rust",
                "metadata": {"memory_layer": "raw"},
            },
        ]
        service = _make_service(vector)
        result = service._fetch_raw_memories(["bad-id", "mem-5"], "user-1")
        assert len(result) == 1
        assert result[0]["id"] == "mem-5"

    def test_mixed_memories(self):
        """Test with a mix of raw, consolidated, superseded, and missing."""
        vector = MagicMock()
        vector.get_by_id.side_effect = [
            # raw, no layer field (should include)
            {"id": "m1", "memory": "fact 1", "metadata": {}},
            # explicit raw (should include)
            {"id": "m2", "memory": "fact 2", "metadata": {"memory_layer": "raw"}},
            # consolidated (should exclude)
            {
                "id": "m3",
                "memory": "fact 3",
                "metadata": {"memory_layer": "consolidated"},
            },
            # superseded raw (should exclude)
            {
                "id": "m4",
                "memory": "fact 4",
                "metadata": {"memory_layer": "raw", "superseded_by": "c1"},
            },
            # missing (should skip)
            None,
        ]
        service = _make_service(vector)
        result = service._fetch_raw_memories(["m1", "m2", "m3", "m4", "m5"], "user-1")
        assert len(result) == 2
        assert {r["id"] for r in result} == {"m1", "m2"}

    def test_deduplicates_memory_ids(self):
        """Duplicate memory IDs should be fetched only once."""
        vector = MagicMock()
        vector.get_by_id.return_value = {
            "id": "m1",
            "memory": "fact 1",
            "metadata": {"memory_layer": "raw"},
        }
        service = _make_service(vector)
        result = service._fetch_raw_memories(["m1", "m1", "m1", "m1"], "user-1")
        assert len(result) == 1
        assert result[0]["id"] == "m1"
        # get_by_id should be called only once despite 4 duplicate IDs
        assert vector.get_by_id.call_count == 1


class TestReConsolidationStateReset:
    """Tests for re-consolidation state reset in SessionSummaryStore.upsert()."""

    def test_state_resets_to_idle_on_new_memories_after_consolidation(self):
        """When new memories arrive after consolidation, state should reset to idle."""
        from mnemory.storage.vector import SessionSummaryStore

        store = SessionSummaryStore.__new__(SessionSummaryStore)
        store._client = MagicMock()

        # Simulate existing consolidated session
        store._client.retrieve.return_value = [
            MagicMock(
                payload={
                    "session_id": "ses-1",
                    "user_id": "user-1",
                    "summary": "old summary",
                    "turn_count": 5,
                    "memory_ids": ["m1", "m2"],
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "updated_at": "2026-01-01T00:00:00+00:00",
                    "consolidation_state": "consolidated",
                }
            )
        ]

        store.upsert(
            session_id="ses-1",
            user_id="user-1",
            summary="new summary",
            new_memory_ids=["m3"],
        )

        # Verify the upsert was called with consolidation_state="idle"
        call_args = store._client.upsert.call_args
        payload = call_args.kwargs["points"][0].payload
        assert payload["consolidation_state"] == "idle"

    def test_state_preserved_when_no_new_memories(self):
        """When no new memories arrive, consolidated state should be preserved."""
        from mnemory.storage.vector import SessionSummaryStore

        store = SessionSummaryStore.__new__(SessionSummaryStore)
        store._client = MagicMock()

        store._client.retrieve.return_value = [
            MagicMock(
                payload={
                    "session_id": "ses-1",
                    "user_id": "user-1",
                    "summary": "old summary",
                    "turn_count": 5,
                    "memory_ids": ["m1"],
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "updated_at": "2026-01-01T00:00:00+00:00",
                    "consolidation_state": "consolidated",
                }
            )
        ]

        store.upsert(
            session_id="ses-1",
            user_id="user-1",
            summary="updated summary",
            new_memory_ids=None,
        )

        call_args = store._client.upsert.call_args
        payload = call_args.kwargs["points"][0].payload
        assert payload["consolidation_state"] == "consolidated"

    def test_idle_state_stays_idle_with_new_memories(self):
        """Idle state should remain idle when new memories arrive."""
        from mnemory.storage.vector import SessionSummaryStore

        store = SessionSummaryStore.__new__(SessionSummaryStore)
        store._client = MagicMock()

        store._client.retrieve.return_value = [
            MagicMock(
                payload={
                    "session_id": "ses-1",
                    "user_id": "user-1",
                    "summary": "summary",
                    "turn_count": 3,
                    "memory_ids": ["m1"],
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "updated_at": "2026-01-01T00:00:00+00:00",
                    "consolidation_state": "idle",
                }
            )
        ]

        store.upsert(
            session_id="ses-1",
            user_id="user-1",
            summary="summary",
            new_memory_ids=["m2"],
        )

        call_args = store._client.upsert.call_args
        payload = call_args.kwargs["points"][0].payload
        assert payload["consolidation_state"] == "idle"


class TestConsolidationPromptWithPrevious:
    """Tests for build_consolidation_prompt with previous_consolidated."""

    def test_prompt_includes_previous_section(self):
        """When previous_consolidated is provided, prompt should include the section."""
        from mnemory.prompts import build_consolidation_prompt

        messages, schema = build_consolidation_prompt(
            summary="Test summary",
            role="user",
            raw_memories=[
                {
                    "id": "m1",
                    "memory": "User likes Python",
                    "metadata": {
                        "memory_type": "preference",
                        "role": "user",
                        "importance": "normal",
                        "categories": [],
                    },
                },
            ],
            previous_consolidated=[
                {
                    "id": "c1",
                    "memory": "User prefers Python for coding",
                    "metadata": {
                        "memory_type": "preference",
                        "role": "user",
                        "importance": "normal",
                    },
                },
            ],
        )

        user_msg = messages[1]["content"]
        assert "Previously Consolidated Memories" in user_msg
        assert "User prefers Python for coding" in user_msg
        assert "Do NOT duplicate" in user_msg

    def test_prompt_without_previous(self):
        """When no previous_consolidated, prompt should not include the section."""
        from mnemory.prompts import build_consolidation_prompt

        messages, schema = build_consolidation_prompt(
            summary="Test summary",
            role="user",
            raw_memories=[
                {
                    "id": "m1",
                    "memory": "User likes Python",
                    "metadata": {
                        "memory_type": "preference",
                        "role": "user",
                        "importance": "normal",
                        "categories": [],
                    },
                },
            ],
        )

        user_msg = messages[1]["content"]
        assert "Previously Consolidated" not in user_msg

    def test_user_role_uses_user_system_prompt(self):
        """role='user' should use the user-focused system prompt."""
        from mnemory.prompts import build_consolidation_prompt

        messages, _ = build_consolidation_prompt(
            summary="Test",
            role="user",
            raw_memories=[
                {
                    "id": "m1",
                    "memory": "User likes X",
                    "metadata": {
                        "memory_type": "preference",
                        "importance": "normal",
                        "categories": [],
                    },
                },
            ],
        )

        system_msg = messages[0]["content"]
        assert "USER" in system_msg
        assert "User decided to" in system_msg or "User prefers" in system_msg

    def test_assistant_role_uses_assistant_system_prompt(self):
        """role='assistant' should use the assistant-focused system prompt."""
        from mnemory.prompts import build_consolidation_prompt

        messages, _ = build_consolidation_prompt(
            summary="Test",
            role="assistant",
            raw_memories=[
                {
                    "id": "m1",
                    "memory": "Assistant implemented X",
                    "metadata": {
                        "memory_type": "episodic",
                        "importance": "high",
                        "categories": [],
                    },
                },
            ],
        )

        system_msg = messages[0]["content"]
        assert "ASSISTANT" in system_msg
        assert (
            "Assistant implemented" in system_msg or "Assistant deployed" in system_msg
        )

    def test_schema_has_no_role_field(self):
        """The output schema should not have a role field (role is implicit)."""
        from mnemory.prompts import build_consolidation_prompt

        _, schema = build_consolidation_prompt(
            summary="Test",
            role="user",
            raw_memories=[
                {
                    "id": "m1",
                    "memory": "Test",
                    "metadata": {
                        "memory_type": "fact",
                        "importance": "normal",
                        "categories": [],
                    },
                },
            ],
        )

        item_props = schema["schema"]["properties"]["memories"]["items"]["properties"]
        assert "role" not in item_props
        item_required = schema["schema"]["properties"]["memories"]["items"]["required"]
        assert "role" not in item_required

    def test_prompt_with_no_raw_memories(self):
        """Prompt should work with empty raw memories (summary-only extraction)."""
        from mnemory.prompts import build_consolidation_prompt

        messages, schema = build_consolidation_prompt(
            summary="User decided to use PostgreSQL for billing. Assistant recommended Redis for caching.",
            role="user",
            raw_memories=[],
        )

        user_msg = messages[1]["content"]
        assert "(no raw memories)" in user_msg
        assert "PostgreSQL" in user_msg
        # System prompt should mention summary-only extraction
        system_msg = messages[0]["content"]
        assert "summary" in system_msg.lower()

    def test_prompt_includes_other_role_context(self):
        """When other_role_consolidated is provided, prompt should include cross-role section."""
        from mnemory.prompts import build_consolidation_prompt

        messages, _ = build_consolidation_prompt(
            summary="Test summary",
            role="assistant",
            raw_memories=[
                {
                    "id": "m1",
                    "memory": "Assistant committed changes as e3a3f72",
                    "metadata": {
                        "memory_type": "episodic",
                        "importance": "normal",
                        "categories": ["technical"],
                    },
                },
            ],
            other_role_consolidated=[
                {"text": "User committed the mnemory changes as commit e3a3f72."},
            ],
        )

        user_msg = messages[1]["content"]
        assert "Already Consolidated" in user_msg
        assert "user role" in user_msg
        assert "e3a3f72" in user_msg
        assert "Do NOT duplicate" in user_msg

    def test_prompt_without_other_role_context(self):
        """When no other_role_consolidated, prompt should not include cross-role section."""
        from mnemory.prompts import build_consolidation_prompt

        messages, _ = build_consolidation_prompt(
            summary="Test summary",
            role="user",
            raw_memories=[
                {
                    "id": "m1",
                    "memory": "User likes Python",
                    "metadata": {
                        "memory_type": "preference",
                        "importance": "normal",
                        "categories": [],
                    },
                },
            ],
        )

        user_msg = messages[1]["content"]
        assert "Already Consolidated" not in user_msg

    def test_prompt_merge_instruction_is_conservative(self):
        """System prompt should instruct to keep separate memories when details differ."""
        from mnemory.prompts import build_consolidation_prompt

        messages, _ = build_consolidation_prompt(
            summary="Test",
            role="user",
            raw_memories=[
                {
                    "id": "m1",
                    "memory": "Test",
                    "metadata": {
                        "memory_type": "fact",
                        "importance": "normal",
                        "categories": [],
                    },
                },
            ],
        )

        system_msg = messages[0]["content"]
        assert "truly redundant" in system_msg
        assert "SEPARATE" in system_msg

    def test_prompt_category_reclassification_instruction(self):
        """System prompt should instruct to reclassify wrong categories."""
        from mnemory.prompts import build_consolidation_prompt

        messages, _ = build_consolidation_prompt(
            summary="Test",
            role="user",
            raw_memories=[
                {
                    "id": "m1",
                    "memory": "Test",
                    "metadata": {
                        "memory_type": "fact",
                        "importance": "normal",
                        "categories": [],
                    },
                },
            ],
        )

        system_msg = messages[0]["content"]
        assert "Reclassify categories" in system_msg
        assert 'use "home" rather than "project:home"' in system_msg

    def test_prompt_distinguishes_recalled_from_new_decision(self):
        """User prompt should avoid turning recalled knowledge into a new decision."""
        from mnemory.prompts import build_consolidation_prompt

        messages, _ = build_consolidation_prompt(
            summary="User recalled a prior pruning plan for the mirobalan tree.",
            role="user",
            raw_memories=[],
        )

        system_msg = messages[0]["content"]
        assert "Distinguish NEW decisions from recalled prior knowledge" in system_msg
        assert "merely recalls or restates" in system_msg

    def test_prompt_prefers_implementation_over_same_session_recommendation(self):
        """Assistant prompt should prefer implementation over same-session recommendation."""
        from mnemory.prompts import build_consolidation_prompt

        messages, _ = build_consolidation_prompt(
            summary="Assistant recommended and then implemented the same fix.",
            role="assistant",
            raw_memories=[],
        )

        system_msg = messages[0]["content"]
        assert "prefer the IMPLEMENTATION memory" in system_msg
        assert "Do not emit both a recommendation and an implementation" in system_msg

    def test_prompt_enforces_one_memory_one_takeaway(self):
        """Prompt should discourage combining multiple durable takeaways."""
        from mnemory.prompts import build_consolidation_prompt

        messages, _ = build_consolidation_prompt(
            summary="User approved the change and stated an ongoing goal.",
            role="user",
            raw_memories=[],
        )

        system_msg = messages[0]["content"]
        assert "One memory = one durable takeaway" in system_msg


class TestConsolidationAlwaysRunsBothPasses:
    """Tests for the always-run-both-passes behavior in consolidation."""

    def test_user_pass_runs_with_no_user_raw_memories(self):
        """User consolidation should run even with 0 user raw memories."""
        import json

        service = _make_service()
        service._config.memory.consolidation_batch_size = 100

        # Mock LLM to return a user decision extracted from summary
        service._llm.generate.return_value = json.dumps(
            {
                "memories": [
                    {
                        "text": "User decided to prune the maple gently",
                        "memory_type": "episodic",
                        "categories": ["home"],
                        "importance": "normal",
                        "pinned": False,
                        "event_date": "2026-03-24",
                    }
                ]
            }
        )

        facts, ids_map = service._consolidate_role(
            session_id="ses-1",
            raw_memories=[],  # No user raw memories
            role="user",
            summary="User decided to prune the maple more gently. Assistant recommended a conservative pruning strategy.",
            artifact_ids=set(),
            previous_consolidated=[],
            session_date="2026-03-24",
        )

        # LLM should have been called (summary-only extraction)
        assert service._llm.generate.called
        assert len(facts) == 1
        assert facts[0]["text"] == "User decided to prune the maple gently"
        assert facts[0]["role"] == "user"

    def test_assistant_pass_receives_user_consolidated_context(self):
        """Assistant consolidation should receive user facts as cross-role context."""
        import json

        service = _make_service()
        service._config.memory.consolidation_batch_size = 100

        service._llm.generate.return_value = json.dumps(
            {
                "memories": [
                    {
                        "text": "Assistant recommended conservative maple pruning",
                        "memory_type": "episodic",
                        "categories": ["home"],
                        "importance": "normal",
                        "pinned": False,
                        "event_date": "2026-03-24",
                    }
                ]
            }
        )

        user_facts = [
            {"text": "User decided to prune the maple gently"},
        ]

        facts, _ = service._consolidate_role(
            session_id="ses-1",
            raw_memories=[
                {
                    "id": "m1",
                    "memory": "Assistant recommended conservative pruning",
                    "metadata": {
                        "memory_type": "episodic",
                        "role": "assistant",
                        "importance": "normal",
                        "categories": ["home"],
                    },
                },
            ],
            role="assistant",
            summary="Test summary",
            artifact_ids=set(),
            previous_consolidated=[],
            session_date="2026-03-24",
            other_role_consolidated=user_facts,
        )

        # Verify the LLM was called with a prompt containing cross-role context
        call_args = service._llm.generate.call_args
        messages = call_args[0][0]
        user_msg = messages[1]["content"]
        assert "Already Consolidated" in user_msg
        assert "User decided to prune the maple gently" in user_msg

    def test_assistant_pass_skipped_without_agent_id(self):
        """When agent_id is None, assistant pass should not run."""
        service = _make_service()
        service._config.memory.consolidation_batch_size = 100

        # Set up session with no agent_id
        service._sessions.get.return_value = {
            "session_id": "ses-1",
            "user_id": "user-1",
            "agent_id": None,
            "memory_ids": ["m1"],
            "summary": "Test summary with enough content to be substantive for the test",
            "created_at": "2026-03-24T00:00:00+00:00",
        }
        service._sessions.list_for_user.return_value = []

        # Only user raw memories
        service._vector.get_by_id.return_value = {
            "id": "m1",
            "memory": "User likes Python",
            "metadata": {
                "memory_layer": "raw",
                "role": "user",
                "memory_type": "preference",
                "importance": "normal",
                "categories": ["technical"],
            },
        }

        import json

        service._llm.generate.return_value = json.dumps({"memories": []})
        service._memory.add_memory.return_value = {"results": []}

        service.consolidate_session("ses-1")

        # LLM should be called exactly once (user pass only, no assistant pass)
        assert service._llm.generate.call_count == 1
