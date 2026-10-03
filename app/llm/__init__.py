"""Provider-agnostic LLM access. ``LLM_PROVIDER=none`` (the default) means no LLM:
the agent then answers with an extractive summary of the evidence."""

from __future__ import annotations

from collections.abc import Callable

from app.config import LLMSettings
from app.llm.base import ChatMessage, LLMError, LLMProvider, LLMResponse
from app.llm.openai_compatible import OpenAICompatibleProvider

PROVIDERS: dict[str, Callable[[LLMSettings], LLMProvider]] = {
    "openai": OpenAICompatibleProvider,
    "openai-compatible": OpenAICompatibleProvider,
    "ollama": OpenAICompatibleProvider,  # via its /v1 endpoint (set LLM_BASE_URL)
}


def build_llm(settings: LLMSettings) -> LLMProvider | None:
    if not settings.enabled:
        return None
    factory = PROVIDERS.get(settings.provider)
    if factory is None:
        supported = ", ".join(sorted([*PROVIDERS, "none"]))
        raise LLMError(f"unknown LLM_PROVIDER {settings.provider!r}; supported: {supported}")
    return factory(settings)


__all__ = [
    "PROVIDERS",
    "ChatMessage",
    "LLMError",
    "LLMProvider",
    "LLMResponse",
    "OpenAICompatibleProvider",
    "build_llm",
]
