"""Text normalisation applied before chunking.

Cleaning is deterministic, and chunk offsets refer to the cleaned text. Re-cleaning
the source therefore reproduces every chunk exactly (see ``source_hash``).

Whitespace can be meaningful in code (a unified diff marks a blank context line
with a single space; string literals may end in spaces), so code is only
minimally normalised, and Markdown is cleaned outside fenced code blocks only.
"""

from __future__ import annotations

import re
import unicodedata

from app.rag.chunking.base import SourceFormat

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_ZERO_WIDTH = re.compile("[\u200b\u200c\u200d\u2060\ufeff]")
_FENCE = re.compile(r"^\s*(```|~~~)")
MAX_BLANK_LINES = 2


def _normalise_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _strip_outer_blank_lines(text: str) -> str:
    lines = text.split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def clean_code(text: str) -> str:
    """Newlines, control characters and outer blank lines only; the code is otherwise untouched."""
    text = _CONTROL.sub("", _normalise_newlines(text)).replace("\ufeff", "")
    return _strip_outer_blank_lines(text)


def clean_markdown(text: str) -> str:
    """NFC, no control or zero-width characters. Outside code fences, also no trailing
    whitespace and at most two consecutive blank lines. Fence contents are kept verbatim."""
    text = _ZERO_WIDTH.sub(
        "", _CONTROL.sub("", _normalise_newlines(unicodedata.normalize("NFC", text)))
    )
    out: list[str] = []
    in_fence, blank_run = False, 0
    for line in text.split("\n"):
        if _FENCE.match(line):
            in_fence = not in_fence
            out.append(line.rstrip())
            blank_run = 0
            continue
        if in_fence:
            out.append(line)
            continue
        line = line.rstrip()
        blank_run = blank_run + 1 if not line else 0
        if blank_run <= MAX_BLANK_LINES:
            out.append(line)
    return _strip_outer_blank_lines("\n".join(out))


def clean_text(text: str, fmt: SourceFormat = SourceFormat.MARKDOWN) -> str:
    return clean_markdown(text) if fmt is SourceFormat.MARKDOWN else clean_code(text)
