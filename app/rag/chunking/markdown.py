"""Structure-aware chunking for Markdown.

The text is parsed into blocks (headings, fenced code, tables, lists, paragraphs)
and grouped into sections under their heading path. Whole sections are packed
greedily up to the size limit, so headings stay with their content and code
blocks and tables are never cut. Only a single block larger than the limit is
split, at hunk, blank-line or line boundaries, and the chunk is flagged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.rag.chunking.base import ChunkingConfig, ChunkSpan, SourceFormat, trim_span
from app.rag.chunking.recursive import DIFF_SEPARATORS, SEPARATORS, RecursiveChunker
from app.rag.chunking.tokens import TokenIndex

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_FENCE = re.compile(r"^\s*(```|~~~)\s*([\w+-]*)")


@dataclass
class Block:
    start: int
    end: int
    kind: str  # heading | fence | table | list | paragraph
    level: int = 0
    title: str = ""
    language: str = ""
    malformed: bool = False


@dataclass
class Unit:
    """A section (or part of one) that is kept together when possible."""

    start: int
    end: int
    section: str | None
    blocks: list[Block] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)


def parse_blocks(text: str) -> list[Block]:
    lines = text.split("\n")
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line) + 1)
    blocks: list[Block] = []

    def line_end(i: int) -> int:
        return offsets[i] + len(lines[i])

    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if fence := _FENCE.match(line):
            marker, j = fence.group(1), i + 1
            while j < len(lines) and not lines[j].lstrip().startswith(marker):
                j += 1
            closed = j < len(lines)
            last = j if closed else len(lines) - 1
            blocks.append(
                Block(
                    offsets[i],
                    line_end(last),
                    "fence",
                    language=fence.group(2),
                    malformed=not closed,
                )
            )
            i = last + 1
            continue
        if heading := _HEADING.match(line):
            blocks.append(
                Block(offsets[i], line_end(i), "heading", len(heading.group(1)), heading.group(2))
            )
            i += 1
            continue
        if line.lstrip().startswith("|"):
            kind, predicate = "table", lambda s: s.lstrip().startswith("|")
        elif _LIST_ITEM.match(line):
            kind, predicate = (
                "list",
                lambda s: bool(s.strip()) and (bool(_LIST_ITEM.match(s)) or s.startswith(" ")),
            )
        else:
            kind = "paragraph"

            def predicate(s: str) -> bool:
                return bool(s.strip()) and not (
                    _HEADING.match(s)
                    or _FENCE.match(s)
                    or s.lstrip().startswith("|")
                    or _LIST_ITEM.match(s)
                )

        j = i + 1
        while j < len(lines) and predicate(lines[j]):
            j += 1
        blocks.append(Block(offsets[i], line_end(j - 1), kind))
        i = j
    return blocks


def section_units(blocks: list[Block]) -> list[Unit]:
    units: list[Unit] = []
    stack: list[tuple[int, str]] = []
    for block in blocks:
        if block.kind == "heading":
            while stack and stack[-1][0] >= block.level:
                stack.pop()
            stack.append((block.level, block.title))
            units.append(Unit(block.start, block.end, " > ".join(t for _, t in stack), [block]))
        elif units and (units[-1].blocks[0].kind == "heading" or units[-1].section is None):
            units[-1].blocks.append(block)
            units[-1].end = block.end
        else:
            units.append(Unit(block.start, block.end, None, [block]))
    return units


class MarkdownChunker:
    def __init__(self, config: ChunkingConfig) -> None:
        self.size = config.chunk_size_tokens
        self.fallback = RecursiveChunker(config)

    def split(self, text: str) -> list[ChunkSpan]:
        tokens = TokenIndex(text)
        whole = trim_span(text, 0, len(text))
        if whole is None:
            return []
        blocks = parse_blocks(text)
        units = section_units(blocks)
        if tokens.count(*whole) <= self.size:
            return [
                self._span(
                    text, [Unit(whole[0], whole[1], units[0].section if units else None, blocks)]
                )
            ]
        pieces: list[Unit] = []
        for unit in units:
            if tokens.count(unit.start, unit.end) <= self.size:
                pieces.append(unit)
            else:
                pieces.extend(self._split_unit(text, tokens, unit))
        return self._pack(text, tokens, pieces)

    def _split_unit(self, text: str, tokens: TokenIndex, unit: Unit) -> list[Unit]:
        """Pack a section's blocks; split only blocks that alone exceed the limit."""
        out: list[Unit] = []
        for block in unit.blocks:
            if tokens.count(block.start, block.end) <= self.size:
                out.append(Unit(block.start, block.end, unit.section, [block]))
                continue
            separators = (
                DIFF_SEPARATORS
                if block.language == "diff"
                else SEPARATORS[SourceFormat.TEXT]
                if block.kind != "fence"
                else ("\n\n", "\n", " ", "")
            )
            parts = self.fallback.split_span(text, block.start, block.end, separators)
            for n, (start, end) in enumerate(parts, 1):
                out.append(
                    Unit(
                        start,
                        end,
                        unit.section,
                        [],
                        {"split_block": block.kind, "part": f"{n}/{len(parts)}"},
                    )
                )
        return out  # whole blocks are re-packed together by _pack

    def _pack(self, text: str, tokens: TokenIndex, units: list[Unit]) -> list[ChunkSpan]:
        chunks: list[ChunkSpan] = []
        group: list[Unit] = []
        for unit in units:
            splittable_part = "split_block" in unit.metadata
            if group and (
                splittable_part
                or "split_block" in group[-1].metadata
                or tokens.count(group[0].start, unit.end) > self.size
            ):
                chunks.append(self._span(text, group))
                group = []
            group.append(unit)
        if group:
            chunks.append(self._span(text, group))
        return [c for c in chunks if c.end > c.start]

    @staticmethod
    def _span(text: str, group: list[Unit]) -> ChunkSpan:
        start, end = trim_span(text, group[0].start, group[-1].end) or (
            group[0].start,
            group[0].start,
        )
        sections = list(dict.fromkeys(u.section for u in group if u.section))
        blocks = [b for u in group for b in u.blocks]
        metadata: dict[str, object] = dict(group[0].metadata) if len(group) == 1 else {}
        if len(sections) > 1:
            metadata["sections"] = sections
        languages = sorted({b.language for b in blocks if b.kind == "fence" and b.language})
        if languages:
            metadata["code_languages"] = languages
        if any(b.kind == "table" for b in blocks):
            metadata["has_table"] = True
        if any(b.malformed for b in blocks):
            metadata["malformed"] = "unclosed_code_fence"
        return ChunkSpan(start, end, group[0].section, metadata)
