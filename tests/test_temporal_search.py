"""Temporal recall regressions using embedded Qdrant and deterministic LLM output."""

import json
import re
from types import SimpleNamespace
from uuid import uuid4

import pytest
from qdrant_client import QdrantClient, models

from mnemory.storage.vector import VectorStore
from tests.test_memory import _make_service

QUERY_VECTORS = {"Mnemory": [1.0, 0.0, 0.0], "memory search": [0.0, 1.0, 0.0]}


@pytest.fixture
def service():
    service = _make_service()
    client = QdrantClient(":memory:")
    client.create_collection(
        "temporal",
        vectors_config=models.VectorParams(size=3, distance=models.Distance.COSINE),
    )
    vector = VectorStore.__new__(VectorStore)
    vector._client = client
    vector._config = SimpleNamespace(vector=SimpleNamespace(collection_name="temporal"))
    vector._embedding = service.vector.embedding
    vector._embedding.embed.return_value = [1.0, 0.0, 0.0]
    vector._embedding.embed_batch.side_effect = lambda texts: [
        QUERY_VECTORS.get(text, [1.0, 0.0, 0.0]) for text in texts
    ]
    service.vector = vector
    yield service
    client.close()


def insert(
    service,
    *,
    event_date=None,
    created_at="2026-10-01T12:00:00Z",
    vector=(1.0, 0.0, 0.0),
    text="Mnemory supports semantic search and file artifacts.",
):
    memory_id = str(uuid4())
    service.vector._client.upsert(
        "temporal",
        [
            models.PointStruct(
                id=memory_id,
                vector=list(vector),
                payload={
                    "user_id": "alice",
                    "data": text,
                    "event_date": event_date,
                    "created_at": created_at,
                },
            )
        ],
    )
    return memory_id


@pytest.mark.parametrize(
    "source_text", [None, "", "today", "What is Mnemory?", "Mnemory"]
)
def test_generated_date_range_does_not_hide_memories_outside_it(service, source_text):
    """A small model may emit today's date for a timeless question and even
    cite question text as its source; the memory must still be found."""
    memory_id = insert(service)
    date_range = {"start": "2026-11-17", "end": "2026-11-17"}
    if source_text is not None:
        date_range["source_text"] = source_text
    service._find_llm.generate.return_value = json.dumps(
        {"queries": ["Mnemory"], "date_range": date_range}
    )

    result = service.find_memories("What is Mnemory?", user_id="alice")

    assert [m["id"] for m in result["results"]] == [memory_id]


@pytest.mark.parametrize("bad_date", ["17/11/2026", "today", "2026-17-11"])
def test_unparseable_generated_dates_do_not_fail_search(service, bad_date):
    memory_id = insert(service)
    service._find_llm.generate.return_value = json.dumps(
        {
            "queries": ["Mnemory"],
            "date_range": {"source_text": "today", "start": bad_date, "end": bad_date},
        }
    )

    result = service.find_memories(
        "What did I note today about Mnemory?", user_id="alice"
    )

    assert [m["id"] for m in result["results"]] == [memory_id]


def test_unreranked_results_stay_in_score_order_with_date_range(service):
    best = insert(service)
    in_range = insert(
        service, created_at="2026-11-17T09:00:00Z", vector=(0.5, 0.5, 0.7)
    )
    service._find_llm.generate.return_value = json.dumps(
        {
            "queries": ["Mnemory"],
            "date_range": {
                "source_text": "Mnemory",
                "start": "2026-11-17",
                "end": "2026-11-17",
            },
        }
    )

    result = service.find_memories("What is Mnemory?", user_id="alice")

    assert [m["id"] for m in result["results"]] == [best, in_range]


def fake_find_llm(date_range, relevant_text):
    """Query generation with a fixed date range; reranking that scores only
    memories containing `relevant_text` as relevant."""

    def generate(messages, *, operation, **kwargs):
        if operation == "query_gen":
            return json.dumps(
                {"queries": ["Mnemory", "memory search"], "date_range": date_range}
            )
        lines = re.findall(r"^\[(\d+)\] (.*)$", messages[1]["content"], re.M)
        return json.dumps(
            {
                "scored": [
                    {
                        "idx": int(idx),
                        "relevance": 0.9 if relevant_text in text else 0.1,
                    }
                    for idx, text in lines
                ]
            }
        )

    return generate


def test_date_range_adds_in_range_candidates_beyond_unfiltered_top_k(service):
    """In-range memories reach the reranker even when more similar
    out-of-range memories fill the unfiltered search results."""
    for vector in [(1.0, 0.0, 0.0)] * 2 + [(0.0, 1.0, 0.0)] * 2:
        insert(service, event_date="2026-10-01T12:00:00Z", vector=vector)
    in_range = insert(
        service,
        event_date="2026-11-12T15:30:00Z",
        vector=(0.5, 0.5, 0.7),
        text="November note about Mnemory temporal search.",
    )
    service._find_llm.generate.side_effect = fake_find_llm(
        {
            "source_text": "on November 12, 2026",
            "start": "2026-11-12",
            "end": "2026-11-12",
        },
        relevant_text="November",
    )

    result = service.find_memories(
        "What did I learn about Mnemory on November 12, 2026?",
        user_id="alice",
        limit=1,
    )

    assert [m["id"] for m in result["results"]] == [in_range]


def test_wrong_date_range_cannot_crowd_out_best_match(service):
    """Many weak memories from a wrongly generated period must not push
    the best unscoped match out of the reranker input."""
    # Two notes match each generated query best, so the scoped searches
    # retrieve four distinct notes from the wrong period.
    for vector in [(0.6, 0.4, 0.7)] * 2 + [(0.4, 0.6, 0.7)] * 2:
        insert(
            service,
            created_at="2026-11-17T09:00:00Z",
            vector=vector,
            text="Unrelated note from today.",
        )
    best = insert(service)
    service._find_llm.generate.side_effect = fake_find_llm(
        {"source_text": "Mnemory", "start": "2026-11-17", "end": "2026-11-17"},
        relevant_text="Mnemory supports",
    )

    result = service.find_memories("What is Mnemory?", user_id="alice", limit=1)

    assert [m["id"] for m in result["results"]] == [best]


def test_explicit_date_filter_uses_event_date_and_creation_fallback(service):
    expected = {
        insert(service, event_date="2026-11-12T15:30:00Z"),
        insert(service, event_date="2026-11-12T23:59:59.999999Z"),
        insert(service, created_at="2026-11-12T23:59:59.999999Z"),
    }
    insert(service, event_date="2026-11-13T00:00:00Z")
    insert(service, created_at="2026-11-13T00:00:00Z")
    insert(
        service,
        event_date="2026-11-11T12:00:00Z",
        created_at="2026-11-12T12:00:00Z",
    )

    results = service.search_memories(
        "Mnemory",
        user_id="alice",
        date_start="2026-11-12",
        date_end="2026-11-12",
    )

    assert {m["id"] for m in results} == expected


@pytest.mark.parametrize("partial_failure", [False, True])
def test_search_errors_are_not_successful_empty_or_partial_results(
    caplog, partial_failure
):
    service = _make_service()
    service.vector.embedding.embed_batch.side_effect = lambda texts: [
        [1.0, 0.0, 0.0] for _ in texts
    ]
    service._find_llm.generate.return_value = json.dumps(
        {"queries": ["Mnemory", "memory system"], "date_range": None}
    )
    error = ConnectionError("Qdrant unavailable")
    service.vector.search.side_effect = (
        [{"results": []}, error] if partial_failure else error
    )

    with pytest.raises(RuntimeError, match="Failed to search memories") as caught:
        service.find_memories("What is Mnemory?", user_id="alice")

    assert caught.value.__cause__ is error
    assert any(
        record.exc_info and record.exc_info[1] is error for record in caplog.records
    )
