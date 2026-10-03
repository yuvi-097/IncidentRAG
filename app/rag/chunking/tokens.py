"""Approximate token counting with exact character offsets.

A token is a run of word characters or a single punctuation character. This
tracks subword tokenisers closely enough for sizing chunks (it undercounts long
identifiers), is deterministic, and needs no model download. Chunk sizes are
therefore configured well below the embedding model's hard limit.
"""

from __future__ import annotations

import bisect
import re

_TOKEN = re.compile(r"\w+|[^\w\s]")


def count_tokens(text: str) -> int:
    return sum(1 for _ in _TOKEN.finditer(text))


class TokenIndex:
    """Token boundaries of one text; counts tokens in any span in O(log n)."""

    def __init__(self, text: str) -> None:
        self.text = text
        matches = list(_TOKEN.finditer(text))
        self.starts = [m.start() for m in matches]
        self.ends = [m.end() for m in matches]

    def __len__(self) -> int:
        return len(self.starts)

    def count(self, start: int, end: int) -> int:
        """Number of tokens that start inside ``[start, end)``."""
        return bisect.bisect_left(self.starts, end) - bisect.bisect_left(self.starts, start)

    def first_token_at_or_after(self, offset: int) -> int:
        return bisect.bisect_left(self.starts, offset)
