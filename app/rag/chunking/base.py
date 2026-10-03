"""Shared chunking types.

Chunkers never copy or rewrite text. They return character spans into the text
they were given, so every chunk is exactly ``text[start:end]`` and can be traced
back to its source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.enums import ChunkingStrategy


class SourceFormat(StrEnum):
    MARKDOWN = "markdown"
    PYTHON = "python"
    YAML = "yaml"
    TEXT = "text"


@dataclass(frozen=True)
class ChunkSpan:
    start: int
    end: int
    section: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ChunkingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    strategy: ChunkingStrategy = ChunkingStrategy.DOCUMENT_AWARE
    chunk_size_tokens: int = Field(default=350, ge=8)
    chunk_overlap_tokens: int = Field(default=50, ge=0)

    @model_validator(mode="after")
    def _overlap_smaller_than_size(self) -> ChunkingConfig:
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")
        return self


class Chunker(Protocol):
    strategy: ChunkingStrategy

    def split(self, text: str, fmt: SourceFormat) -> list[ChunkSpan]: ...


def trim_span(text: str, start: int, end: int) -> tuple[int, int] | None:
    """Shrink ``[start, end)`` to exclude surrounding whitespace; None if nothing remains."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if start < end else None
