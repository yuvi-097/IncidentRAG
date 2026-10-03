"""Structure-aware chunking for source code and configuration.

Python: the module is cut into contiguous line ranges at top-level definitions,
namely the module header (docstring, imports, constants), each function, and each class.
A class larger than the limit is cut again at its methods. Decorators and the
comments directly above a definition stay with it. Units are packed greedily up
to the size limit. A single function that alone exceeds the limit is the only
thing split mid-body, and its chunks are flagged ``split_symbol``.

YAML: cut at top-level keys (with their leading comments), then packed.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

from app.rag.chunking.base import ChunkingConfig, ChunkSpan, SourceFormat, trim_span
from app.rag.chunking.recursive import SEPARATORS, RecursiveChunker
from app.rag.chunking.tokens import TokenIndex

_YAML_KEY = re.compile(r"^[A-Za-z0-9_.\-\"']+\s*:")


@dataclass
class CodeUnit:
    first_line: int  # 0-based, inclusive
    last_line: int  # 0-based, inclusive
    section: str
    symbols: list[str]


class _Lines:
    def __init__(self, text: str) -> None:
        self.lines = text.split("\n")
        self.starts = [0]
        for line in self.lines:
            self.starts.append(self.starts[-1] + len(line) + 1)

    def span(self, first: int, last: int) -> tuple[int, int]:
        return self.starts[first], self.starts[last] + len(self.lines[last])

    def extend_to_comments(self, line: int, floor: int) -> int:
        """Move ``line`` up over directly preceding comment lines (not past ``floor``)."""
        while line - 1 >= floor and self.lines[line - 1].lstrip().startswith("#"):
            line -= 1
        return line


def _definition_start(node: ast.stmt) -> int:
    decorators = getattr(node, "decorator_list", [])
    return min([node.lineno, *(d.lineno for d in decorators)]) - 1


def _is_definition(node: ast.stmt) -> bool:
    return isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)


class CodeChunker:
    def __init__(self, config: ChunkingConfig) -> None:
        self.size = config.chunk_size_tokens
        self.fallback = RecursiveChunker(config)

    # -- python

    def split_python(self, text: str) -> list[ChunkSpan]:
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError) as exc:
            spans = self.fallback.split(text, SourceFormat.PYTHON)
            reason = f"{type(exc).__name__}: {getattr(exc, 'msg', exc)}"
            return [ChunkSpan(s.start, s.end, s.section, {"parse_error": reason}) for s in spans]
        lines, tokens = _Lines(text), TokenIndex(text)
        units = self._python_units(tree, lines, tokens)
        return self._pack(text, lines, tokens, units)

    def _python_units(self, tree: ast.Module, lines: _Lines, tokens: TokenIndex) -> list[CodeUnit]:
        # (start line, section, symbols) for every cut point; units run to the next cut.
        cuts: list[tuple[int, str, list[str]]] = [(0, "<module>", [])]
        previous_end = 0
        for node in tree.body:
            if _is_definition(node):
                start = lines.extend_to_comments(_definition_start(node), previous_end)
                name = node.name  # type: ignore[attr-defined]
                end = (node.end_lineno or node.lineno) - 1
                first, last = lines.span(start, end)
                if isinstance(node, ast.ClassDef) and tokens.count(first, last) > self.size:
                    cuts.append((start, name, [name]))
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                            method_start = lines.extend_to_comments(
                                _definition_start(item), start + 1
                            )
                            cuts.append(
                                (method_start, f"{name}.{item.name}", [f"{name}.{item.name}"])
                            )
                else:
                    cuts.append((start, name, [name]))
                previous_end = end + 1
            elif cuts[-1][1] != "<module>" and node.lineno - 1 >= previous_end:
                # Statements after a definition (e.g. `app = FastAPI()`) start a module unit.
                cuts.append((node.lineno - 1, "<module>", []))
                previous_end = node.end_lineno or node.lineno
            else:
                previous_end = node.end_lineno or node.lineno
        cuts = sorted({c[0]: c for c in cuts}.values())  # one cut per line, in order
        units = []
        for (start, section, symbols), nxt in zip(
            cuts, [*cuts[1:], (len(lines.lines), "", [])], strict=True
        ):
            last = nxt[0] - 1
            while last > start and not lines.lines[last].strip():
                last -= 1
            if any(lines.lines[i].strip() for i in range(start, last + 1)):
                units.append(CodeUnit(start, last, section, symbols))
        return units

    def _pack(
        self, text: str, lines: _Lines, tokens: TokenIndex, units: list[CodeUnit]
    ) -> list[ChunkSpan]:
        chunks: list[ChunkSpan] = []
        group: list[CodeUnit] = []

        def flush() -> None:
            if not group:
                return
            span = trim_span(text, *lines.span(group[0].first_line, group[-1].last_line))
            if span:
                symbols = [s for u in group for s in u.symbols]
                metadata: dict[str, object] = {"symbols": symbols} if symbols else {}
                chunks.append(ChunkSpan(span[0], span[1], group[0].section, metadata))
            group.clear()

        for unit in units:
            start, end = lines.span(unit.first_line, unit.last_line)
            if tokens.count(start, end) > self.size:
                flush()
                parts = self.fallback.split_span(
                    text, start, end, SEPARATORS[SourceFormat.PYTHON][3:]
                )
                for n, (a, b) in enumerate(parts, 1):
                    chunks.append(
                        ChunkSpan(
                            a,
                            b,
                            unit.section,
                            {
                                "symbols": unit.symbols,
                                "split_symbol": True,
                                "part": f"{n}/{len(parts)}",
                            },
                        )
                    )
                continue
            if group and tokens.count(lines.span(group[0].first_line, 0)[0], end) > self.size:
                flush()
            group.append(unit)
        flush()
        return chunks

    # -- yaml

    def split_yaml(self, text: str) -> list[ChunkSpan]:
        lines, tokens = _Lines(text), TokenIndex(text)
        cuts: list[tuple[int, str]] = []
        for i, line in enumerate(lines.lines):
            if _YAML_KEY.match(line):
                start = lines.extend_to_comments(i, cuts[-1][0] + 1 if cuts else 0)
                cuts.append((start, line.split(":", 1)[0].strip("\"' ")))
        if not cuts or cuts[0][0] != 0:
            cuts.insert(0, (0, "<document>"))
        units = []
        for (start, key), nxt in zip(cuts, [*cuts[1:], (len(lines.lines), "")], strict=True):
            last = nxt[0] - 1
            while last > start and not lines.lines[last].strip():
                last -= 1
            if any(lines.lines[i].strip() for i in range(start, last + 1)):
                units.append(CodeUnit(start, last, key, [key] if key != "<document>" else []))
        chunks = self._pack(text, lines, tokens, units)
        return [
            ChunkSpan(c.start, c.end, c.section, {"keys": c.metadata.get("symbols", [])})
            if "symbols" in c.metadata
            else c
            for c in chunks
        ]
