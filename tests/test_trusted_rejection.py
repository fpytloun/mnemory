"""Terminal rejection never converts uncertain semantic work into safe fallback."""

import asyncio
import json
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from mnemory.api.evidence import dispatch_trusted_event
from mnemory.revisions import EvidenceConflictError, TrustedBudgetRejection
from mnemory.trusted_events import PROTOCOL
from tests.test_trusted_semantic import expire_claims, llm_responses, run, service
from tests.test_user_event_ingestion import _make_shared_ingest_service


def no_llm(*args, **kwargs):
    pytest.fail("Terminal rejection must not run the LLM")


@pytest.mark.parametrize(
    "reason",
    [
        "input_budget_exceeded",
        "extraction_fact_budget_exceeded",
        "action_limit_exceeded",
        "plan_budget_exceeded",
    ],
)
def test_durable_rejection_replays_across_routes(reason, monkeypatch):
    svc = service()
    text = "Original source remains in the caller's signed queue payload."
    svc._llm = SimpleNamespace(generate=no_llm)
    if reason == "input_budget_exceeded":
        svc._config.memory.max_input_length = 1
    elif reason == "plan_budget_exceeded":
        llm_responses(svc, "A" * 900)
        monkeypatch.setattr("mnemory.revisions.EVIDENCE_MAX_PLAN_BYTES", 1100)
    else:
        facts = (
            [{"text": "x" * 1001}]
            if reason == "extraction_fact_budget_exceeded"
            else [{"text": "Fact"} for _ in range(33)]
        )
        monkeypatch.setattr(
            svc, "_remember_extract", lambda *a, **k: (facts, "", False)
        )
    first = run(svc, "rejected-root", text)
    assert first["status"] == "rejected"
    assert first["outcome"] == "rejected_before_write"
    assert first["reason"] == reason
    assert first["fallback_allowed"] is False
    store = svc.revisions.operations
    row = store.get_evidence_plan(first["operation_id"])
    assert row["status"] == "committed"
    assert row["targets"] == row["checkpoints"] == []
    assert row["claim_epoch"] == 0
    svc._config.memory.max_input_length = 400000
    svc._llm = SimpleNamespace(generate=no_llm)
    assert run(svc, "rejected-root", text, "evidence") == first
    assert svc.vector._client.count("memories").count == 0
    with pytest.raises(EvidenceConflictError):
        run(svc, "rejected-root", text + " changed")


def test_generic_failure_is_not_terminal_rejection(monkeypatch):
    svc = service()
    monkeypatch.setattr(
        svc,
        "_remember_extract",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("provider failure")),
    )
    with pytest.raises(ValueError, match="provider failure"):
        run(svc, "transient-root", "Original")
    records, _ = svc.vector._client.scroll("_mnemory_operations", limit=100)
    assert not any((row.payload or {}).get("protocol") == PROTOCOL for row in records)


@pytest.mark.parametrize("after_write", [False, True])
def test_sealed_failure_resumes_even_if_budget_later_changes(after_write, monkeypatch):
    svc = service()
    llm_responses(svc, "User lives in Prague.")
    execute = svc._execute_action

    def interrupted(*args, **kwargs):
        if after_write:
            execute(*args, **kwargs)
        raise TrustedBudgetRejection("input_budget_exceeded")

    monkeypatch.setattr(svc, "_execute_action", interrupted)
    with pytest.raises(TrustedBudgetRejection):
        run(svc, "sealed-root", "User lives in Prague.")
    records, _ = svc.vector._client.scroll("_mnemory_operations", limit=100)
    row = next(
        row.payload for row in records if row.payload.get("protocol") == PROTOCOL
    )
    assert row["status"] == "claimed"
    assert len(row["targets"]) == 1
    # Even an insert-only rejection request cannot overwrite this plan.
    collision = svc.revisions.operations.seal_evidence_plan(
        protocol=PROTOCOL,
        user_id="user-1",
        owner_id="owner-1",
        evidence_root_id="sealed-root",
        request_fingerprint=row["request_fingerprint"],
        targets=[],
        terminal_rejection="input_budget_exceeded",
    )
    assert collision["status"] == "claimed"
    assert collision["targets"] == row["targets"]
    svc._config.memory.max_input_length = 1
    monkeypatch.setattr(svc, "_execute_action", execute)
    svc._llm = SimpleNamespace(generate=no_llm)
    expire_claims(svc)
    result = run(svc, "sealed-root", "User lives in Prague.", "evidence")
    assert result["status"] == "replayed"
    assert result["result"]["results"][0]["event"] == "ADD"
    assert svc.vector._client.count("memories").count == 1


def race_rejection_and_execution(rejector, executor):
    entered = threading.Event()
    release = threading.Event()
    extract = executor._remember_extract

    def waiting_extract(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        return extract(*args, **kwargs)

    executor._remember_extract = waiting_extract
    llm_responses(executor, "User lives in Prague.")
    rejector._config.memory.max_input_length = 1
    rejector._llm = SimpleNamespace(generate=no_llm)
    with ThreadPoolExecutor(max_workers=2) as pool:
        writing = pool.submit(run, executor, "race-root", "User lives in Prague.")
        assert entered.wait(3)
        rejecting = pool.submit(
            run, rejector, "race-root", "User lives in Prague.", "evidence"
        )
        release.set()
        assert writing.result()["status"] == "accepted"
        assert rejecting.result()["status"] == "replayed"


def test_concurrent_budget_rejection_cannot_override_execution():
    executor = service()
    rejector = _make_shared_ingest_service(executor.vector._client)
    race_rejection_and_execution(rejector, executor)
    assert executor.vector._client.count("memories").count == 1


def test_concurrent_execution_cannot_override_rejection(monkeypatch):
    rejector = service()
    executor = _make_shared_ingest_service(rejector.vector._client)
    rejector._config.memory.max_input_length = 1
    for svc in (rejector, executor):
        svc._llm = SimpleNamespace(generate=no_llm)
    entered = threading.Event()
    release = threading.Event()
    seal = rejector.revisions.operations.seal_evidence_plan

    def waiting_seal(**kwargs):
        assert kwargs["terminal_rejection"] == "input_budget_exceeded"
        entered.set()
        assert release.wait(3)
        return seal(**kwargs)

    monkeypatch.setattr(
        rejector.revisions.operations, "seal_evidence_plan", waiting_seal
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        rejection = pool.submit(run, rejector, "race-rejected", "Original")
        assert entered.wait(3)
        execution = pool.submit(run, executor, "race-rejected", "Original", "evidence")
        release.set()
        assert rejection.result() == execution.result()
    assert executor.vector._client.count("memories").count == 0


@pytest.mark.skipif(
    not os.environ.get("MNEMORY_TEST_QDRANT_URL"), reason="Remote Qdrant required"
)
def test_remote_rejection_execution_fence():
    client = QdrantClient(url=os.environ["MNEMORY_TEST_QDRANT_URL"])
    collection = "trusted_rejection_" + uuid.uuid4().hex
    client.create_collection(
        collection, vectors_config=VectorParams(size=2, distance=Distance.COSINE)
    )
    actors = [_make_shared_ingest_service(client) for _ in range(2)]
    user = "rejection-" + uuid.uuid4().hex
    for svc in actors:
        svc.vector._config.vector.collection_name = collection
        svc.vector._config.vector.is_remote = True
        svc.revisions = type(svc.revisions)(svc.vector)
        svc._test_user = svc._test_owner = user
    try:
        race_rejection_and_execution(*actors)
        assert client.count(collection).count == 1
    finally:
        client.delete_collection(collection)


@pytest.mark.parametrize("route", ["evidence", "ingest"])
def test_dispatch_rejection_is_non_success_http(route):
    from pathlib import Path

    from mnemory.api.schemas import EvidenceRememberRequest

    fixture = Path(__file__).parent / "contract/fixtures/evidence_remember_v1.json"
    body = EvidenceRememberRequest.model_validate(
        json.loads(fixture.read_text())["body"]
    )
    svc = service()
    svc._config.memory.max_input_length = 1
    svc._llm = SimpleNamespace(generate=no_llm)
    with pytest.raises(HTTPException) as first:
        asyncio.run(dispatch_trusted_event(svc, body, route=route))
    assert first.value.status_code == 422
    assert first.value.detail["status"] == "rejected"
    assert first.value.detail["source_retention"] == "caller_queue"
    with pytest.raises(HTTPException) as replay:
        asyncio.run(dispatch_trusted_event(svc, body, route=route))
    assert replay.value.detail == first.value.detail
