"""Any server that speaks the OpenAI chat-completions API (OpenAI, Azure-style
gateways, Ollama ``/v1``, vLLM, LM Studio...). Configuration only: LLM_BASE_URL,
LLM_MODEL, LLM_API_KEY (optional for local servers), LLM_TEMPERATURE,
LLM_MAX_OUTPUT_TOKENS, LLM_TIMEOUT_SECONDS. The key is never logged."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from app.config import LLMSettings
from app.llm.base import ChatMessage, LLMError, LLMResponse

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.openai.com/v1"


class OpenAICompatibleProvider:
    def __init__(self, settings: LLMSettings, transport: httpx.BaseTransport | None = None) -> None:
        if not settings.model:
            raise LLMError("LLM_MODEL is required")
        self.model = settings.model
        self.name = f"{settings.provider}:{settings.model}"
        self.temperature = settings.temperature
        self.max_tokens = settings.max_output_tokens
        headers = {"Content-Type": "application/json"}
        if settings.api_key is not None:
            headers["Authorization"] = f"Bearer {settings.api_key.get_secret_value()}"
        self._client = httpx.Client(
            base_url=(settings.base_url or DEFAULT_BASE_URL).rstrip("/"),
            headers=headers,
            timeout=settings.timeout_seconds,
            transport=transport,
        )

    def complete(self, messages: list[ChatMessage]) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.model_dump() for m in messages],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        try:
            response = self._client.post("/chat/completions", json=payload)
        except httpx.HTTPError as exc:
            raise LLMError(f"LLM request failed: {type(exc).__name__}") from exc
        if response.status_code != 200:
            raise LLMError(f"LLM returned HTTP {response.status_code}")
        try:
            body = response.json()
            text = body["choices"][0]["message"]["content"] or ""
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError("unexpected LLM response format") from exc
        usage = {k: v for k, v in (body.get("usage") or {}).items() if isinstance(v, int)}
        logger.info(
            "llm.completed",
            extra={"model": self.model, "output_chars": len(text), **usage},
        )
        return LLMResponse(
            text=text,
            model=str(body.get("model") or self.model),
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            usage=usage,
        )
