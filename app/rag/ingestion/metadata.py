"""Stage 3: chunk-level metadata extraction."""

from __future__ import annotations

import re
from typing import Any

from app.rag.chunking.base import ChunkSpan
from app.rag.ingestion.models import ParsedSource

# Ids of other NovaCart records mentioned in text: the edges later phases hop along.
_MENTION = re.compile(r"\b(?:INC|DEP|RB|DOC|PM|CF)-\d{4}\b|\bPR-\d{4,}\b")
_DIFF_SECTION = re.compile(r"Diff: (\S+)")
_DIFF_HEADING = re.compile(r"^## Diff: (\S+)", re.MULTILINE)


def mentions(text: str, exclude: str | None = None) -> list[str]:
    return sorted({m for m in _MENTION.findall(text) if m != exclude})


def line_range(text: str, start: int, end: int) -> tuple[int, int]:
    """1-based inclusive line numbers of ``text[start:end]``."""
    return text.count("\n", 0, start) + 1, text.count("\n", 0, end) + 1


def chunk_file_path(parsed: ParsedSource, span: ChunkSpan, content: str) -> str | None:
    """The source's path, or, for pull-request chunks, the changed file when the
    chunk covers exactly one file's diff (it may start inside that diff section)."""
    if parsed.file_path:
        return parsed.file_path
    paths = set(_DIFF_HEADING.findall(content))
    if match := _DIFF_SECTION.search(span.section or ""):
        paths.add(match.group(1))
    return paths.pop() if len(paths) == 1 else None


def chunk_metadata(parsed: ParsedSource, span: ChunkSpan, text: str) -> dict[str, Any]:
    content = text[span.start : span.end]
    line_start, line_end = line_range(text, span.start, span.end)
    return {
        **parsed.metadata,
        **span.metadata,
        "format": parsed.format.value,
        "line_start": line_start,
        "line_end": line_end,
        "mentions": mentions(content, exclude=parsed.source_id),
    }
