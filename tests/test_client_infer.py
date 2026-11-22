"""Exercise client add-memory inference policy through REST and MCP."""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from mnemory.api import create_api_app
from mnemory.config import MemoryConfig
from tests.test_memory import _make_service

_CONTENT = "Send not confirmed — only the draft was saved.\nCheck with the owner."
_EXTRACTED = "The draft was saved; sending is unconfirmed."


@pytest.mark.parametrize("allow_client_infer", [False, True])
@pytest.mark.parametrize("allow_assistant_verbatim", [False, True])
@pytest.mark.parametrize("role", ["user", "assistant"])
@pytest.mark.parametrize("infer", [False, True])
@pytest.mark.parametrize("transport", ["rest", "rest_batch", "mcp", "mcp_batch"])
def test_client_add_inference_mode(
    monkeypatch: pytest.MonkeyPatch,
    allow_client_infer: bool,
    allow_assistant_verbatim: bool,
    role: str,
    infer: bool,
    transport: str,
) -> None:
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    from mnemory import server

    monkeypatch.setenv("ALLOW_CLIENT_INFER", str(allow_client_infer).lower())
    if allow_assistant_verbatim:
        monkeypatch.setenv("ALLOW_ASSISTANT_VERBATIM", "true")
    else:
        monkeypatch.delenv("ALLOW_ASSISTANT_VERBATIM", raising=False)
    service = _make_service()
    service._config.memory = MemoryConfig(auto_classify=False)
    service.vector.get_all.return_value = {"results": []}
    service.vector.search_similar.return_value = []
    service.vector.insert.return_value = "mem-1"
    service._llm.generate.return_value = json.dumps(
        {
            "memories": [
                {
                    "text": _EXTRACTED,
                    "action": "ADD",
                    "memory_type": "fact",
                    "categories": [],
                    "importance": "normal",
                }
            ]
        }
    )
    monkeypatch.setattr(server, "_get_service", lambda: service)

    effective_infer = infer or not allow_client_infer
    rejected = (
        role == "assistant" and not effective_infer and not allow_assistant_verbatim
    )

    user_token = server._session_user_id.set("test-user")
    agent_token = server._session_agent_id.set("test-agent")
    try:
        if transport.startswith("rest"):
            request = {"content": _CONTENT, "role": role, "infer": infer}
            with TestClient(create_api_app()) as client:
                if transport == "rest":
                    response = client.post("/memories", json=request)
                    assert response.status_code == (422 if rejected else 200)
                    result = response.json()
                    if rejected:
                        assert "infer=False is not allowed" in result["detail"]
                else:
                    response = client.post(
                        "/memories/batch", json={"memories": [request]}
                    )
                    assert response.status_code == 200
                    (result,) = response.json()
                    assert result["error"] is rejected
        elif transport == "mcp":
            result = json.loads(
                asyncio.run(
                    server.add_memory(_CONTENT, agent_id="self", role=role, infer=infer)
                )
            )
            if rejected:
                assert result["error"] is True
                assert "infer=False is not allowed" in result["message"]
        else:
            batch = json.loads(
                asyncio.run(
                    server.add_memories(
                        [{"content": _CONTENT, "role": role}],
                        agent_id="self",
                        infer=infer,
                    )
                )
            )
            if rejected:
                assert "infer" not in batch
                (result,) = batch["errors"]
                assert "infer=False is not allowed" in result["message"]
            else:
                assert batch["infer"] is effective_infer
                (result,) = batch["results"]
    finally:
        server._session_agent_id.reset(agent_token)
        server._session_user_id.reset(user_token)

    if rejected:
        assert result.get("infer") is None
        service.vector.insert.assert_not_called()
    else:
        expected_text = _EXTRACTED if effective_infer else _CONTENT
        assert result["infer"] is effective_infer
        assert result["results"][0]["memory"] == expected_text
        assert service.vector.insert.call_args.kwargs["text"] == expected_text
        assert service.vector.insert.call_args.kwargs["role"] == role


def test_client_opt_in_does_not_change_internal_trust_guard() -> None:
    service = _make_service()
    service._config.memory = MemoryConfig(
        allow_client_infer=True, allow_assistant_verbatim=True, auto_classify=False
    )
    with pytest.raises(ValueError, match="infer=False is not allowed"):
        service.add_memory(
            _CONTENT,
            user_id="test-user",
            agent_id="test-agent",
            role="assistant",
            infer=False,
        )
    with pytest.raises(ValueError, match="agent_id is required"):
        service.add_memory(
            _CONTENT,
            user_id="test-user",
            role="assistant",
            infer=False,
            _client_request=True,
        )
