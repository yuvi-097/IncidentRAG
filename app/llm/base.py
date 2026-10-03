"""The LLM contract. Nothing outside ``app/llm`` knows which provider is configured."""

from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field


class LLMError(RuntimeError):
    pass


class ChatMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Literal["system", "user", "assistant"]
    content: str


class LLMResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    usage: dict[str, int] = Field(default_factory=dict)


class LLMProvider(Protocol):
    name: str  # provider/model, for reporting

    def complete(self, messages: list[ChatMessage]) -> LLMResponse:
        """One chat completion. Raises ``LLMError`` on failure."""
        ...
