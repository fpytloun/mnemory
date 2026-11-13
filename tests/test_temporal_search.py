"""Temporal recall regressions using embedded Qdrant and deterministic LLM output."""

import json
from types import SimpleNamespace
from uuid import uuid4

import pytest
from qdrant_client import QdrantClient, models

from mnemory.storage.vector import VectorStore
from tests.test_memory import _make_service


@pytest.fixture
def service():
    service = _make_service()
    client = QdrantClient(":memory:")
    client.create_collection(
        "temporal",
        vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
    )
    vector = VectorStore.__new__(VectorStore)
    vector._client = client
    vector._config = SimpleNamespace(vector=SimpleNamespace(collection_name="temporal"))
    vector._embedding = service.vector.embedding
    vector._embedding.embed_batch.side_effect = lambda texts: [
        [1.0, 0.0] for _ in texts
    ]
    service.vector = vector
    yield service
    client.close()


def insert(service, *, event_date=None, created_at="2026-10-01T12:00:00Z"):
    memory_id = str(uuid4())
    service.vector._client.upsert(
        "temporal",
        [
            models.PointStruct(
                id=memory_id,
                vector=[1.0, 0.0],
                payload={
                    "user_id": "alice",
                    "data": "Mnemory supports semantic search and file artifacts.",
                    "event_date": event_date,
                    "created_at": created_at,
                },
            )
        ],
    )
    return memory_id


@pytest.mark.parametrize("source_text", [None, "", "today"])
def test_ungrounded_generated_dates_do_not_hide_memories(service, source_text):
    memory_id = insert(service)
    date_range = {"start": "2026-11-12", "end": "2026-11-12"}
    if source_text is not None:
        date_range["source_text"] = source_text
    service._find_llm.generate.return_value = json.dumps(
        {"queries": ["Mnemory"], "date_range": date_range}
    )

    result = service.find_memories("What is Mnemory?", user_id="alice")

    assert [m["id"] for m in result["results"]] == [memory_id]


def test_temporal_recall_uses_event_date_and_creation_fallback(service):
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
    service._find_llm.generate.return_value = json.dumps(
        {
            "queries": ["Mnemory"],
            "date_range": {
                "source_text": "on November 12, 2026",
                "start": "2026-11-12",
                "end": "2026-11-12",
            },
        }
    )

    result = service.find_memories(
        "What did I learn about Mnemory on November 12, 2026?", user_id="alice"
    )

    assert {m["id"] for m in result["results"]} == expected


@pytest.mark.parametrize("partial_failure", [False, True])
def test_search_errors_are_not_successful_empty_or_partial_results(
    caplog, partial_failure
):
    service = _make_service()
    service.vector.embedding.embed_batch.side_effect = lambda texts: [
        [1.0, 0.0] for _ in texts
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
