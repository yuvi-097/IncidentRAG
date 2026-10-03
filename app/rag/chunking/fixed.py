"""Fixed-size chunking: windows of N tokens with exactly M tokens of overlap."""

from __future__ import annotations

from app.rag.chunking.base import ChunkingConfig, ChunkSpan, SourceFormat
from app.rag.chunking.sections import SectionLocator
from app.rag.chunking.tokens import TokenIndex
from app.schemas.enums import ChunkingStrategy


class FixedSizeChunker:
    strategy = ChunkingStrategy.FIXED

    def __init__(self, config: ChunkingConfig) -> None:
        self.size = config.chunk_size_tokens
        self.overlap = config.chunk_overlap_tokens

    def split(self, text: str, fmt: SourceFormat) -> list[ChunkSpan]:
        tokens = TokenIndex(text)
        if not len(tokens):
            return []
        locator = SectionLocator(text, fmt)
        spans, first = [], 0
        while True:
            last = min(first + self.size, len(tokens))  # exclusive
            start, end = tokens.starts[first], tokens.ends[last - 1]
            spans.append(ChunkSpan(start, end, locator.at(start)))
            if last == len(tokens):
                return spans
            first = last - self.overlap
