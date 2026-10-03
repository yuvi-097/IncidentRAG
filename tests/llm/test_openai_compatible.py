"""The OpenAI-compatible provider against a mock HTTP transport (no network, no model)."""

from __future__ import annotations

import json

import httpx
import pytest

from app.config import LLMSettings
from app.llm import ChatMessage, LLMError, OpenAICompatibleProvider, build_llm

MESSAGES = [
    ChatMessage(role="system", content="rules"),
    ChatMessage(role="user", content="question"),
]

# Fake keys, built at run time so no credential-shaped literal is in source.
TEST_KEY = "sk-" + "test"
SECRET_KEY = "sk-" + "very-secret"


def settings(**overrides: object) -> LLMSettings:
    values = {
        "provider": "openai-compatible",
        "model": "test-model",
        "base_url": "http://llm.local/v1",
    }
    values.update(overrides)
    return LLMSettings(_env_file=None, **values)  # type: ignore[call-arg]


def test_request_and_response(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "test-model-2026",
                "choices": [{"message": {"role": "assistant", "content": "Answer [E1]."}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
            },
        )

    provider = OpenAICompatibleProvider(
        settings(api_key=TEST_KEY, temperature=0.2, max_output_tokens=300),
        transport=httpx.MockTransport(handler),
    )
    reply = provider.complete(MESSAGES)
    assert seen["url"] == "http://llm.local/v1/chat/completions"
    assert seen["auth"] == f"Bearer {TEST_KEY}"
    assert seen["body"] == {
        "model": "test-model",
        "messages": [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "question"},
        ],
        "temperature": 0.2,
        "max_tokens": 300,
    }
    assert (reply.text, reply.model, reply.input_tokens, reply.output_tokens) == (
        "Answer [E1].",
        "test-model-2026",
        12,
        4,
    )
    assert provider.name == "openai-compatible:test-model"


def test_no_api_key_sends_no_authorization_header() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "authorization" not in request.headers
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    provider = OpenAICompatibleProvider(settings(), transport=httpx.MockTransport(handler))
    assert provider.complete(MESSAGES).text == "ok"


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx.Response(500, text="boom"), "HTTP 500"),
        (httpx.Response(401, json={"error": "bad key"}), "HTTP 401"),
        (httpx.Response(200, text="not json"), "unexpected"),
        (httpx.Response(200, json={"choices": []}), "unexpected"),
    ],
)
def test_errors_become_llm_errors(response: httpx.Response, message: str) -> None:
    provider = OpenAICompatibleProvider(
        settings(), transport=httpx.MockTransport(lambda r: response)
    )
    with pytest.raises(LLMError, match=message):
        provider.complete(MESSAGES)


def test_network_failures_become_llm_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("slow")

    provider = OpenAICompatibleProvider(settings(), transport=httpx.MockTransport(handler))
    with pytest.raises(LLMError, match="ConnectTimeout"):
        provider.complete(MESSAGES)


def test_the_api_key_never_appears_in_errors() -> None:
    provider = OpenAICompatibleProvider(
        settings(api_key=SECRET_KEY),
        transport=httpx.MockTransport(
            lambda r: httpx.Response(403, text=f"{SECRET_KEY} is invalid")
        ),
    )
    with pytest.raises(LLMError) as info:
        provider.complete(MESSAGES)
    assert SECRET_KEY not in str(info.value)


def test_factory() -> None:
    assert build_llm(LLMSettings(_env_file=None)) is None  # type: ignore[call-arg]
    assert isinstance(build_llm(settings(provider="ollama")), OpenAICompatibleProvider)
    with pytest.raises(LLMError, match="unknown LLM_PROVIDER"):
        build_llm(settings(provider="magic"))
