"""Chunk size and overlap for fixed and recursive chunking; edge cases for all strategies."""

from __future__ import annotations

import re
from itertools import pairwise

import pytest
from pydantic import ValidationError

from app.config import ChunkingSettings
from app.rag.chunking import ChunkingConfig, SourceFormat, build_chunker, count_tokens
from app.rag.chunking.fixed import FixedSizeChunker
from app.rag.chunking.recursive import RecursiveChunker
from app.schemas.enums import ChunkingStrategy

TOKEN = re.compile(r"\w+|[^\w\s]")
PARAGRAPHS = "\n\n".join(
    f"Paragraph {n}. The payment service timed out waiting for connection {n}; "
    f"responders rolled back release v2.8.{n} and error rates recovered."
    for n in range(40)
)


def _tokens(text: str) -> list[str]:
    return TOKEN.findall(text)


@pytest.mark.parametrize(("size", "overlap"), [(40, 10), (64, 0), (25, 24)])
def test_fixed_chunks_have_exact_size_and_exact_overlap(size: int, overlap: int) -> None:
    config = ChunkingConfig(
        strategy=ChunkingStrategy.FIXED, chunk_size_tokens=size, chunk_overlap_tokens=overlap
    )
    spans = FixedSizeChunker(config).split(PARAGRAPHS, SourceFormat.TEXT)
    chunks = [PARAGRAPHS[s.start : s.end] for s in spans]
    assert all(count_tokens(c) == size for c in chunks[:-1])
    assert 0 < count_tokens(chunks[-1]) <= size
    for left, right in pairwise(chunks):
        if overlap:
            assert _tokens(left)[-overlap:] == _tokens(right)[:overlap]
        else:
            assert PARAGRAPHS.index(right) >= PARAGRAPHS.index(left) + len(left)


def test_fixed_chunks_cover_every_token() -> None:
    config = ChunkingConfig(
        strategy=ChunkingStrategy.FIXED, chunk_size_tokens=50, chunk_overlap_tokens=0
    )
    spans = FixedSizeChunker(config).split(PARAGRAPHS, SourceFormat.TEXT)
    assert sum(count_tokens(PARAGRAPHS[s.start : s.end]) for s in spans) == count_tokens(PARAGRAPHS)


def test_recursive_respects_size_and_breaks_at_paragraphs() -> None:
    config = ChunkingConfig(
        strategy=ChunkingStrategy.RECURSIVE, chunk_size_tokens=80, chunk_overlap_tokens=0
    )
    spans = RecursiveChunker(config).split(PARAGRAPHS, SourceFormat.TEXT)
    assert len(spans) > 1
    for span in spans:
        assert count_tokens(PARAGRAPHS[span.start : span.end]) <= 80
        assert span.end == len(PARAGRAPHS) or PARAGRAPHS[span.end : span.end + 2] == "\n\n"


def test_recursive_overlap_is_bounded_and_used() -> None:
    config = ChunkingConfig(
        strategy=ChunkingStrategy.RECURSIVE, chunk_size_tokens=60, chunk_overlap_tokens=30
    )
    spans = RecursiveChunker(config).split(PARAGRAPHS.replace("\n\n", " "), SourceFormat.TEXT)
    overlaps = [max(0, a.end - b.start) for a, b in pairwise(spans)]
    text = PARAGRAPHS.replace("\n\n", " ")
    assert any(overlaps)
    for a, b in pairwise(spans):
        if a.end > b.start:
            assert count_tokens(text[b.start : a.end]) <= 30


def test_unbreakable_text_is_hard_split_within_limit() -> None:
    text = "-" * 1000  # 1000 punctuation tokens, no separators at all
    for strategy in ChunkingStrategy:
        config = ChunkingConfig(strategy=strategy, chunk_size_tokens=64, chunk_overlap_tokens=8)
        spans = build_chunker(config).split(text, SourceFormat.TEXT)
        assert spans and all(count_tokens(text[s.start : s.end]) <= 64 for s in spans)


@pytest.mark.parametrize("strategy", list(ChunkingStrategy))
@pytest.mark.parametrize("fmt", list(SourceFormat))
def test_empty_and_whitespace_input_yields_no_chunks(
    strategy: ChunkingStrategy, fmt: SourceFormat
) -> None:
    chunker = build_chunker(ChunkingConfig(strategy=strategy))
    assert chunker.split("", fmt) == []
    assert chunker.split("   \n\n\t  \n", fmt) == []


@pytest.mark.parametrize("strategy", list(ChunkingStrategy))
def test_short_text_is_one_exact_chunk(strategy: ChunkingStrategy) -> None:
    text = "# Title\n\nOne short paragraph."
    spans = build_chunker(ChunkingConfig(strategy=strategy)).split(text, SourceFormat.MARKDOWN)
    assert [(s.start, s.end) for s in spans] == [(0, len(text))]


@pytest.mark.parametrize("strategy", list(ChunkingStrategy))
def test_spans_are_valid_offsets(strategy: ChunkingStrategy) -> None:
    config = ChunkingConfig(strategy=strategy, chunk_size_tokens=45, chunk_overlap_tokens=5)
    for span in build_chunker(config).split(PARAGRAPHS, SourceFormat.TEXT):
        assert 0 <= span.start < span.end <= len(PARAGRAPHS)
        assert not PARAGRAPHS[span.start].isspace() and not PARAGRAPHS[span.end - 1].isspace()


def test_invalid_configuration_is_rejected(clean_env: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValidationError):
        ChunkingConfig(chunk_size_tokens=50, chunk_overlap_tokens=50)
    clean_env.setenv("CHUNKING_CHUNK_SIZE_TOKENS", "100")
    clean_env.setenv("CHUNKING_CHUNK_OVERLAP_TOKENS", "150")
    with pytest.raises(ValidationError):
        ChunkingSettings(_env_file=None)  # type: ignore[call-arg]


def test_strategy_is_configurable_from_environment(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("CHUNKING_STRATEGY", "recursive")
    clean_env.setenv("CHUNKING_CHUNK_SIZE_TOKENS", "200")
    settings = ChunkingSettings(_env_file=None)  # type: ignore[call-arg]
    assert settings.strategy is ChunkingStrategy.RECURSIVE and settings.chunk_size_tokens == 200
