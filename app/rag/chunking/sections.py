"""Locate the heading path or code symbol that encloses a span.

Structure-agnostic strategies (fixed, recursive) use this to keep ``section``
metadata, so no strategy loses where in the document a chunk came from.
"""

from __future__ import annotations

import ast
import bisect
import re

from app.rag.chunking.base import SourceFormat

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


def markdown_headings(text: str) -> list[tuple[int, int, str]]:
    """(offset, level, title) of every ATX heading outside code fences."""
    headings, offset, in_fence = [], 0, False
    for line in text.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith(("```", "~~~")):
            in_fence = not in_fence
        elif not in_fence and (match := _HEADING.match(line)):
            headings.append((offset, len(match.group(1)), match.group(2)))
        offset += len(line) + 1
    return headings


def python_symbols(text: str) -> list[tuple[int, int, str]] | None:
    """(start_offset, end_offset, qualified_name) of classes, functions and methods,
    or None when the source does not parse."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return None
    line_starts = [0]
    for line in text.split("\n"):
        line_starts.append(line_starts[-1] + len(line) + 1)
    symbols: list[tuple[int, int, str]] = []

    def visit(nodes: list[ast.stmt], prefix: str) -> None:
        for node in nodes:
            if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                first = min([node.lineno, *(d.lineno for d in node.decorator_list)])
                end = line_starts[min(node.end_lineno or node.lineno, len(line_starts) - 1)]
                name = f"{prefix}{node.name}"
                symbols.append((line_starts[first - 1], end, name))
                if isinstance(node, ast.ClassDef):
                    visit(node.body, f"{name}.")

    visit(tree.body, "")
    return symbols


class SectionLocator:
    def __init__(self, text: str, fmt: SourceFormat) -> None:
        self.fmt = fmt
        self._paths: list[tuple[int, str]] = []
        self._symbols: list[tuple[int, int, str]] = []
        if fmt is SourceFormat.MARKDOWN:
            stack: list[tuple[int, str]] = []
            for offset, level, title in markdown_headings(text):
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, title))
                self._paths.append((offset, " > ".join(t for _, t in stack)))
        elif fmt is SourceFormat.PYTHON:
            self._symbols = python_symbols(text) or []
        self._offsets = [offset for offset, _ in self._paths]

    def at(self, start: int) -> str | None:
        if self._paths:
            index = bisect.bisect_right(self._offsets, start) - 1
            return self._paths[index][1] if index >= 0 else None
        if self.fmt is SourceFormat.PYTHON:
            enclosing = [s for s in self._symbols if s[0] <= start < s[1]]
            if enclosing:
                return max(enclosing, key=lambda s: s[0])[2]  # innermost
            return "<module>"
        return None
