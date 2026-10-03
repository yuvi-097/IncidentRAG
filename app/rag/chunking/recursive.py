"""Recursive chunking: split on the coarsest separator that yields pieces under the
size limit (paragraphs, then lines, then sentences, then words, then tokens), then
merge adjacent pieces back up to the limit, carrying up to ``overlap`` tokens of
trailing context into the next chunk.
"""

from __future__ import annotations

from itertools import pairwise

from app.rag.chunking.base import ChunkingConfig, ChunkSpan, SourceFormat, trim_span
from app.rag.chunking.sections import SectionLocator
from app.rag.chunking.tokens import TokenIndex
from app.schemas.enums import ChunkingStrategy

# Separators ending in a space/newline split *after* the match; structural prefixes
# (a heading marker, `def`, a diff hunk) split *before* it, so the marker stays
# at the start of the next piece.
SEPARATORS: dict[SourceFormat, tuple[str, ...]] = {
    SourceFormat.MARKDOWN: ("\n## ", "\n### ", "\n#### ", "\n```", "\n\n", "\n", ". ", " ", ""),
    SourceFormat.PYTHON: (
        "\nclass ",
        "\ndef ",
        "\nasync def ",
        "\n    def ",
        "\n    async def ",
        "\n\n",
        "\n",
        " ",
        "",
    ),
    SourceFormat.YAML: ("\n\n", "\n", " ", ""),
    SourceFormat.TEXT: ("\n\n", "\n", ". ", " ", ""),
}
DIFF_SEPARATORS = ("\n@@ ", "\n\n", "\n", " ", "")


def _split_points(text: str, start: int, end: int, separator: str) -> list[int]:
    before = separator.strip() != "" and separator != ". "
    points, index = [], text.find(separator, start, end)
    while index != -1:
        cut = index + 1 if before else index + len(separator)
        if start < cut < end:
            points.append(cut)
        index = text.find(separator, index + len(separator), end)
    return points


class RecursiveChunker:
    strategy = ChunkingStrategy.RECURSIVE

    def __init__(self, config: ChunkingConfig) -> None:
        self.size = config.chunk_size_tokens
        self.overlap = config.chunk_overlap_tokens

    def split(self, text: str, fmt: SourceFormat) -> list[ChunkSpan]:
        locator = SectionLocator(text, fmt)
        return [
            ChunkSpan(s, e, locator.at(s))
            for s, e in self.split_span(text, 0, len(text), SEPARATORS[fmt])
        ]

    def split_span(
        self, text: str, start: int, end: int, separators: tuple[str, ...]
    ) -> list[tuple[int, int]]:
        """Chunk ``text[start:end]``; used directly by document-aware fallbacks."""
        tokens = TokenIndex(text)
        pieces = self._pieces(text, tokens, start, end, separators)
        return self._merge(text, tokens, pieces)

    def _pieces(
        self, text: str, tokens: TokenIndex, start: int, end: int, separators: tuple[str, ...]
    ) -> list[tuple[int, int]]:
        if tokens.count(start, end) <= self.size:
            return [(start, end)]
        for depth, separator in enumerate(separators):
            if separator == "":
                return self._hard_split(tokens, start, end)
            points = _split_points(text, start, end, separator)
            if not points:
                continue
            bounds = [start, *points, end]
            pieces: list[tuple[int, int]] = []
            for a, b in pairwise(bounds):
                if tokens.count(a, b) <= self.size:
                    pieces.append((a, b))
                else:
                    pieces.extend(self._pieces(text, tokens, a, b, separators[depth + 1 :]))
            return pieces
        return self._hard_split(tokens, start, end)

    def _hard_split(self, tokens: TokenIndex, start: int, end: int) -> list[tuple[int, int]]:
        first, last = tokens.first_token_at_or_after(start), tokens.first_token_at_or_after(end)
        cuts = [tokens.starts[i] for i in range(first + self.size, last, self.size)]
        bounds = [start, *cuts, end]
        return list(pairwise(bounds))

    def _merge(
        self, text: str, tokens: TokenIndex, pieces: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        chunks: list[tuple[int, int]] = []
        current: list[tuple[int, int]] = []

        def emit() -> None:
            span = trim_span(text, current[0][0], current[-1][1])
            if span and (not chunks or span != chunks[-1]):
                chunks.append(span)

        for piece in pieces:
            if current and tokens.count(current[0][0], piece[1]) > self.size:
                emit()
                carried: list[tuple[int, int]] = []
                total = 0
                for previous in reversed(current):
                    size = tokens.count(*previous)
                    if total + size > self.overlap:
                        break
                    carried.insert(0, previous)
                    total += size
                current = carried
                while current and tokens.count(current[0][0], piece[1]) > self.size:
                    current.pop(0)
            current.append(piece)
        if current:
            emit()
        return chunks
