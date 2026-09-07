"""Contract tests through extraction, journal and real local Qdrant writes."""

import json
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from mnemory.revisions import EvidenceLeaseLostError
from tests.test_user_event_ingestion import _make_shared_ingest_service


def service():
    client = QdrantClient(location=":memory:")
    client.create_collection(
        collection_name="memories",
        vectors_config=VectorParams(size=2, distance=Distance.COSINE),
    )
    return _make_shared_ingest_service(client)


def llm_responses(svc, text, decision="ADD", equivalent=True):
    def generate(messages, *, operation, **kwargs):
        if operation == "remember_extract":
            return json.dumps(
                {
                    "memories": [{"text": text, "memory_type": "fact"}],
                    "summary": text,
                    "store_artifact": False,
                }
            )
        if operation == "remember_dedup":
            return json.dumps(
                {
                    "decisions": [
                        {
                            "fact_index": 0,
                            "action": decision,
                            "target_id": None if decision == "ADD" else "0",
                            "text": text,
                        }
                    ]
                }
            )
        if operation == "evidence_semantic_equivalence":
            return json.dumps({"equivalent": equivalent})
        raise AssertionError(operation)

    svc._llm = SimpleNamespace(generate=generate)


def run(svc, root, text, route="ingest"):
    return svc.process_trusted_event(
        content=text,
        user_id=getattr(svc, "_test_user", "user-1"),
        owner_id=getattr(svc, "_test_owner", "owner-1"),
        evidence_root_id=root,
        source_event={"event_id": root, "event_hash": root},
        request_hashes={"evidence": root + "-e", "ingest": root + "-i"},
        route=route,
        cancel=threading.Event(),
    )


@pytest.mark.parametrize(
    "second",
    [
        "Qdrant je jediná databáze Mnemory",
        "Mnemory používá výhradně databázi Qdrant.",
        "Qdrant is Mnemory's only database.",
    ],
)
def test_atomic_add_then_independent_confirmation_and_cross_route_replay(second):
    svc = service()
    first = "Qdrant je jediná databáze Mnemory."
    llm_responses(svc, first)
    added = run(svc, "root-a", first)
    memory_id = added["result"]["results"][0]["id"]
    assert svc.vector.get_by_id_strict(memory_id)["metadata"]["validation_count"] == 0
    assert run(svc, "root-a", first, "evidence")["status"] == "replayed"
    llm_responses(svc, second, "CONFIRM")
    confirmed = run(svc, "root-b", second, "evidence")
    assert confirmed["result"]["results"][0]["event"] == "CONFIRM"
    assert svc.vector.get_by_id_strict(memory_id)["metadata"]["validation_count"] == 1
    assert run(svc, "root-b", second)["status"] == "replayed"
    assert svc.vector.get_by_id_strict(memory_id)["metadata"]["validation_count"] == 1


def test_extraction_failure_does_not_store_raw_message():
    svc = service()
    svc._llm = SimpleNamespace(
        generate=lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError())
    )
    with pytest.raises(TimeoutError):
        run(svc, "root-timeout", "A long normal user message")
    assert svc.vector._client.count("memories").count == 0


@pytest.mark.parametrize(
    "text",
    [
        "Qdrant není jediná databáze Mnemory.",
        "Qdrant je jedna z databází Mnemory.",
        "Qdrant byl do roku 2024 jedinou databází Mnemory.",
    ],
)
def test_non_equivalence_is_not_confirmed_even_if_dedup_suggests_confirm(text):
    svc = service()
    llm_responses(svc, "Qdrant je jediná databáze Mnemory.")
    added = run(svc, "initial", "Qdrant je jediná databáze Mnemory.")
    memory_id = added["result"]["results"][0]["id"]
    llm_responses(svc, text, "CONFIRM", equivalent=False)
    assert run(svc, "changed", text)["result"]["results"][0]["event"] == "SKIP"
    assert svc.vector.get_by_id_strict(memory_id)["metadata"]["validation_count"] == 0


def test_revisioned_update_establishes_new_root_and_replays():
    svc = service()
    llm_responses(svc, "User lives in Prague.")
    added = run(svc, "initial", "User lives in Prague.")
    old_id = added["result"]["results"][0]["id"]
    llm_responses(svc, "User lives in Berlin.", "UPDATE")
    updated = run(svc, "moved", "User lives in Berlin.")
    result = updated["result"]["results"][0]
    assert result["event"] == "UPDATE"
    assert result["revision"] == 2
    assert result["id"] != old_id
    metadata = svc.vector.get_by_id_strict(result["id"])["metadata"]
    assert metadata["validation_count"] == 0
    assert metadata["evidence_root_ids"] == ["moved"]
    assert "moved" in metadata["consumed_evidence_root_ids"]
    assert (
        run(svc, "moved", "User lives in Berlin.", "evidence")["status"] == "replayed"
    )
    llm_responses(svc, "User lives in Berlin.", "CONFIRM")
    for ordinal in (1, 2):
        run(svc, f"confirm-move-{ordinal}", "User lives in Berlin.")
        current = svc.vector.get_by_id_strict(result["id"])["metadata"]
        assert current["validation_count"] == ordinal


def expire_claims(svc):
    client = svc.vector._client
    records, _ = client.scroll("_mnemory_operations", limit=1000)
    for point in records:
        if (point.payload or {}).get("status") == "claimed":
            client.set_payload(
                "_mnemory_operations",
                {
                    "lease_expires_at": (
                        datetime.now(timezone.utc) - timedelta(seconds=1)
                    ).isoformat(),
                    "claim_deadline_utc": (
                        datetime.now(timezone.utc) - timedelta(seconds=1)
                    ).isoformat(),
                },
                points=[point.id],
            )


@pytest.mark.parametrize("action", ["ADD", "UPDATE", "CONFIRM"])
def test_crash_after_fact_write_resumes_without_repeating_effect(action, monkeypatch):
    svc = service()
    text = "User lives in Prague."
    if action != "ADD":
        llm_responses(svc, text)
        run(svc, "initial", text)
    next_text = "User lives in Berlin." if action == "UPDATE" else text
    llm_responses(svc, next_text, action)
    operations = svc.revisions.operations
    checkpoint = operations.checkpoint_evidence_plan
    monkeypatch.setattr(
        operations,
        "checkpoint_evidence_plan",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("simulated crash")),
    )
    with pytest.raises(RuntimeError, match="simulated crash"):
        run(svc, "interrupted", next_text)
    expire_claims(svc)
    monkeypatch.setattr(operations, "checkpoint_evidence_plan", checkpoint)
    svc._llm = SimpleNamespace(generate=lambda *a, **k: pytest.fail("replanned event"))
    result = run(svc, "interrupted", next_text)
    assert result["status"] == "replayed"
    points, _ = svc.vector._client.scroll("memories", limit=100)
    active = [point for point in points if point.payload["revision_state"] == "active"]
    assert len(active) == 1
    assert active[0].payload["revision"] == (2 if action == "UPDATE" else 1)
    assert active[0].payload["validation_count"] == (1 if action == "CONFIRM" else 0)


def test_two_instances_serialize_distinct_paraphrases():
    first = service()
    second = _make_shared_ingest_service(first.vector._client)
    second.vector._config = first.vector._config
    second.revisions = type(first.revisions)(second.vector)
    second._test_user = getattr(first, "_test_user", "user-1")
    second._test_owner = getattr(first, "_test_owner", "owner-1")
    texts = ["Qdrant je jediná databáze Mnemory.", "Mnemory uses only Qdrant."]
    for svc, text in zip([first, second], texts):
        llm_responses(svc, text)
        generate = svc._llm.generate

        def deciding(messages, *, operation, _generate=generate, **kwargs):
            if operation == "remember_dedup":
                has_candidate = "Existing memories:" in messages[-1]["content"]
                # Use actual prompt candidate presence, not root ordering.
                response = json.loads(
                    _generate(messages, operation=operation, **kwargs)
                )
                if has_candidate or '"id": "0"' in messages[-1]["content"]:
                    response["decisions"][0].update(action="CONFIRM", target_id="0")
                return json.dumps(response)
            return _generate(messages, operation=operation, **kwargs)

        svc._llm = SimpleNamespace(generate=deciding)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda pair: run(*pair),
                [
                    (first, "parallel-a", texts[0]),
                    (second, "parallel-b", texts[1]),
                ],
            )
        )
    assert sorted(r["result"]["results"][0]["event"] for r in results) == [
        "ADD",
        "CONFIRM",
    ]
    assert first.vector._client.count(first.vector.collection_name).count == 1


def test_cancelled_request_never_mutates():
    svc = service()
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(EvidenceLeaseLostError):
        svc.process_trusted_event(
            content="Signed user content",
            user_id="user-1",
            owner_id="owner-1",
            evidence_root_id="cancelled",
            source_event={},
            request_hashes={},
            route="ingest",
            cancel=cancel,
        )
    assert svc.vector._client.count("memories").count == 0


@pytest.mark.parametrize("old_route", ["ingest", "evidence"])
@pytest.mark.parametrize("new_route", ["ingest", "evidence"])
def test_terminal_legacy_event_is_never_reexecuted(old_route, new_route):
    svc = service()
    store = svc.revisions.operations
    if old_route == "ingest":
        row = store.prepare_user_event_ingestion(
            protocol="mnemory.trusted-evidence.v1",
            user_id="user-1",
            owner_id="owner-1",
            evidence_root_id="historical",
            request_fingerprint="historical-i",
            memory_id="historical-memory",
            source_event={"event_id": "historical"},
        )
        store.complete_user_event_ingestion(
            row["operation_id"],
            request_fingerprint="historical-i",
            result={"results": []},
        )
    else:
        row = store.seal_evidence_plan(
            user_id="user-1",
            owner_id="owner-1",
            evidence_root_id="historical",
            request_fingerprint="historical-e",
            targets=[],
        )
        claim = store.claim_evidence_plan(
            row["operation_id"],
            request_fingerprint="historical-e",
            epoch=1,
            nonce="historical-claim",
        )
        store.commit_evidence_plan(
            row["operation_id"],
            request_fingerprint="historical-e",
            epoch=claim["claim_epoch"],
            nonce=claim["claim_nonce"],
            result={"status": "skipped", "checkpoints": []},
        )
    svc._llm = SimpleNamespace(
        generate=lambda *a, **k: pytest.fail("legacy reexecution")
    )
    result = run(svc, "historical", "Historical message", new_route)
    assert result["status"] == "replayed"
    assert result["operation_id"] == row["operation_id"]
    assert svc.vector._client.count("memories").count == 0


def test_long_message_is_extracted_not_stored_whole():
    svc = service()
    text = "Discussion without another fact. " * 60 + "User lives in Prague."
    llm_responses(svc, "User lives in Prague.")
    result = run(svc, "long-message", text)
    assert result["result"]["results"][0]["memory"] == "User lives in Prague."
    assert svc.vector._client.count("memories").count == 1


@pytest.mark.skipif(
    not os.environ.get("MNEMORY_TEST_QDRANT_URL"),
    reason="Isolated remote Qdrant required",
)
def test_remote_trusted_semantic_concurrency(monkeypatch):
    client = QdrantClient(url=os.environ["MNEMORY_TEST_QDRANT_URL"])
    collection = "trusted_semantic_" + uuid.uuid4().hex
    client.create_collection(
        collection, vectors_config=VectorParams(size=2, distance=Distance.COSINE)
    )
    svc = _make_shared_ingest_service(client)
    svc.vector._config.vector.collection_name = collection
    svc.vector._config.vector.is_remote = True
    svc.revisions = type(svc.revisions)(svc.vector)
    svc._test_user = "semantic-" + uuid.uuid4().hex
    svc._test_owner = svc._test_user
    monkeypatch.setattr("tests.test_trusted_semantic.service", lambda: svc)
    try:
        test_two_instances_serialize_distinct_paraphrases()
    finally:
        client.delete_collection(collection)


def test_scope_release_waits_for_renewal_and_prevents_late_extension():
    from mnemory.trusted_events import TrustedEventLease

    entered = threading.Event()
    unblock = threading.Event()
    calls = []

    def renew(*args, **kwargs):
        entered.set()
        assert unblock.wait(2)
        calls.append("renewed")

    operations = SimpleNamespace(
        renew_user_event_content_claim=renew,
        release_trusted_scope_claim=lambda *a, **k: calls.append("released"),
    )
    lease = TrustedEventLease(
        operations,
        {
            "operation_id": "scope",
            "claim_owner": "worker",
            "claim_epoch": 1,
            "claim_nonce": "nonce",
        },
        threading.Event(),
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        renewal = pool.submit(lease.check)
        assert entered.wait(2)
        release = pool.submit(lease.release)
        unblock.set()
        renewal.result()
        release.result()
    assert calls == ["renewed", "released"]
    with pytest.raises(EvidenceLeaseLostError):
        lease.check()


@pytest.mark.skipif(
    os.environ.get("MNEMORY_TEST_LIVE_SEMANTIC") != "1",
    reason="Opt-in isolated live LLM test",
)
@pytest.mark.parametrize(
    "text,equivalent",
    [
        ("Qdrant je jediná databáze Mnemory", True),
        ("Mnemory používá výhradně databázi Qdrant.", True),
        ("Qdrant is Mnemory's only database.", True),
        ("Qdrant není jediná databáze Mnemory.", False),
        ("Qdrant je jedna z databází Mnemory.", False),
        ("Qdrant byl do roku 2024 jedinou databází Mnemory.", False),
    ],
)
def test_live_llm_semantic_contract(text, equivalent):
    from mnemory.config import LLMConfig
    from mnemory.llm import LLMClient

    svc = service()
    svc._llm = LLMClient(LLMConfig())
    first = "Qdrant je jediná databáze Mnemory."
    added = run(svc, "live-initial", first)
    assert added["result"]["results"][0]["event"] == "ADD"
    result = run(svc, "live-second", text)
    actions = [item["event"] for item in result["result"]["results"]]
    assert ("CONFIRM" in actions) == equivalent
    if equivalent:
        assert svc.vector._client.count("memories").count == 1
