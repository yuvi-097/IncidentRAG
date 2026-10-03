"""Document-aware chunking keeps headings, code blocks, tables, functions and classes intact."""

from __future__ import annotations

import ast
import textwrap

from app.rag.chunking import ChunkingConfig, SourceFormat, count_tokens
from app.rag.chunking.document_aware import DocumentAwareChunker
from app.synthetic.records import SyntheticDataset


def chunker(size: int = 60, overlap: int = 5) -> DocumentAwareChunker:
    return DocumentAwareChunker(
        ChunkingConfig(chunk_size_tokens=size, chunk_overlap_tokens=overlap)
    )


def para(n: int) -> str:
    return " ".join(f"word{i}" for i in range(n))


MARKDOWN = textwrap.dedent(f"""\
    # Guide

    Intro {para(10)}.

    ## Setup

    {para(30)}

    ```bash
    pip install -r requirements.txt
    uvicorn app.main:app --reload --port 8000
    ```

    ## Tables

    | Setting | Default |
    |---|---|
    | pool_size | 20 |
    | pool_timeout | 3 |

    ## Deep

    ### Deeper

    {para(35)}
    """)


# --- markdown -------------------------------------------------------------------------------


def test_headings_start_chunks_and_carry_their_path() -> None:
    spans = chunker().split(MARKDOWN, SourceFormat.MARKDOWN)
    assert len(spans) > 1
    for span in spans:
        body = MARKDOWN[span.start : span.end]
        assert body.startswith("#"), body[:40]
        assert count_tokens(body) <= 60
    assert "Guide > Deep > Deeper" in {s.section for s in spans} | {
        x for s in spans for x in s.metadata.get("sections", [])
    }


def test_code_blocks_and_tables_are_never_split_when_they_fit() -> None:
    spans = chunker().split(MARKDOWN, SourceFormat.MARKDOWN)
    for span in spans:
        body = MARKDOWN[span.start : span.end]
        assert body.count("```") % 2 == 0
    code_chunks = [s for s in spans if "code_languages" in s.metadata]
    assert code_chunks and code_chunks[0].metadata["code_languages"] == ["bash"]
    table = next(s for s in spans if s.metadata.get("has_table"))
    assert "| pool_size | 20 |" in MARKDOWN[table.start : table.end]
    assert "| pool_timeout | 3 |" in MARKDOWN[table.start : table.end]


def test_oversized_code_block_is_split_and_flagged() -> None:
    lines = "\n".join(f"-    pool_size={n}," for n in range(80))
    text = f"# Change\n\n```diff\n@@ -1,3 +1,3 @@\n{lines}\n@@ -90,2 +90,2 @@\n{lines}\n```"
    spans = chunker(size=80).split(text, SourceFormat.MARKDOWN)
    parts = [s for s in spans if s.metadata.get("split_block") == "fence"]
    assert len(parts) >= 2
    assert all(count_tokens(text[s.start : s.end]) <= 80 for s in spans)


def test_unclosed_code_fence_is_flagged_not_fatal() -> None:
    text = "# Broken\n\nSome text.\n\n```python\ndef f():\n    return 1\n"
    spans = chunker().split(text, SourceFormat.MARKDOWN)
    assert spans and any(s.metadata.get("malformed") == "unclosed_code_fence" for s in spans)


def test_small_document_stays_whole() -> None:
    text = "# Small\n\n## A\n\nalpha\n\n## B\n\nbeta"
    spans = chunker(size=200).split(text, SourceFormat.MARKDOWN)
    assert len(spans) == 1 and (spans[0].start, spans[0].end) == (0, len(text))


# --- python -----------------------------------------------------------------------------------

PYTHON = (
    textwrap.dedent('''\
    """Module docstring."""

    import os
    from typing import Any

    LIMIT = 10


    # Retries with backoff; see RB-0012.
    @decorator(option=True)
    def fetch(url: str) -> dict[str, Any]:
        """Fetch a thing."""
        result = {"url": url, "env": os.environ.get("ENV"), "limit": LIMIT}
        return result


    class Big:
        """A class too large for one chunk."""

        size = 3

        def first(self) -> int:
            values = [n * 2 for n in range(10) if n % 2 == 0 and n > self.size]
            return sum(values) + len(values) - self.size

        @property
        def second(self) -> int:
            mapping = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6}
            return sum(mapping.values()) * self.size

        async def third(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
            data = dict(payload or {})
            data.update({"x": 1, "y": 2, "z": 3, "size": self.size, "ok": True})
            return data


    def huge() -> list[int]:
        out = []
''')
    + "".join(f"    out.append({n} * {n} + {n} - {n} // 2)\n" for n in range(40))
    + "    return out\n"
)


def _definitions(source: str) -> list[tuple[str, int, int]]:
    """(name, first line incl. decorators, last line), 1-based, for functions and methods."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            first = min([node.lineno, *(d.lineno for d in node.decorator_list)])
            found.append((node.name, first, node.end_lineno or node.lineno))
    return found


def _line_span(source: str, start: int, end: int) -> tuple[int, int]:
    return source.count("\n", 0, start) + 1, source.count("\n", 0, end) + 1


def test_functions_are_never_split_when_they_fit() -> None:
    """A definition (with its decorators and leading comments) that fits the limit
    lands in one chunk; one that does not is split and every part is flagged."""
    spans = chunker(size=90).split(PYTHON, SourceFormat.PYTHON)
    ranges = [(_line_span(PYTHON, s.start, s.end), s) for s in spans]
    lines = PYTHON.split("\n")
    for name, first, last in _definitions(PYTHON):
        fits = count_tokens("\n".join(lines[first - 1 : last])) <= 90
        holders = [s for (a, b), s in ranges if a <= first and last <= b]
        touching = [s for (a, b), s in ranges if a <= last and b >= first]
        if fits:
            assert holders, f"{name} fits the limit but was split"
        else:
            assert all(s.metadata.get("split_symbol") for s in touching), name


def test_large_class_is_cut_at_methods_with_qualified_sections() -> None:
    spans = chunker(size=90).split(PYTHON, SourceFormat.PYTHON)
    sections = {s.section for s in spans}
    symbols = {sym for s in spans for sym in s.metadata.get("symbols", [])}
    assert {"Big.first", "Big.second", "Big.third"} <= symbols
    assert any(section and section.startswith("Big") for section in sections)


def test_decorators_and_leading_comments_stay_with_their_function() -> None:
    spans = chunker(size=90).split(PYTHON, SourceFormat.PYTHON)
    holder = next(s for s in spans if "def fetch" in PYTHON[s.start : s.end])
    body = PYTHON[holder.start : holder.end]
    assert "# Retries with backoff" in body and "@decorator(option=True)" in body
    second = next(s for s in spans if "def second" in PYTHON[s.start : s.end])
    assert "@property" in PYTHON[second.start : second.end]


def test_oversized_function_is_split_and_flagged() -> None:
    spans = chunker(size=90).split(PYTHON, SourceFormat.PYTHON)
    parts = [s for s in spans if s.metadata.get("split_symbol")]
    assert len(parts) >= 2 and all(s.section == "huge" for s in parts)
    assert all(count_tokens(PYTHON[s.start : s.end]) <= 90 for s in spans)


def test_every_line_of_code_is_covered() -> None:
    spans = chunker(size=90).split(PYTHON, SourceFormat.PYTHON)
    covered = set()
    for span in spans:
        a, b = _line_span(PYTHON, span.start, span.end)
        covered.update(range(a, b + 1))
    code_lines = {n for n, line in enumerate(PYTHON.split("\n"), 1) if line.strip()}
    assert code_lines <= covered


def test_syntax_error_falls_back_to_recursive_and_is_flagged() -> None:
    broken = "def ok():\n    return 1\n\ndef broken(:\n    pass\n" * 30
    spans = chunker(size=50).split(broken, SourceFormat.PYTHON)
    assert spans and all("parse_error" in s.metadata for s in spans)
    assert all(count_tokens(broken[s.start : s.end]) <= 50 for s in spans)


def test_yaml_is_cut_at_top_level_keys() -> None:
    text = (
        "# manifest\nservice: payment\nreplicas: 6\nenv:\n"
        + "".join(f'  KEY_{n}: "value-{n}"\n' for n in range(30))
        + "secrets:\n  - name: X\n"
    )
    spans = chunker(size=60).split(text, SourceFormat.YAML)
    assert len(spans) > 1
    for span in spans:
        first_line = text[span.start : span.end].split("\n")[0]
        assert not first_line.startswith(" ") or span.metadata.get("split_symbol")
    assert text[spans[0].start : spans[0].end].startswith("# manifest")


def test_real_repository_functions_are_kept_whole(dataset: SyntheticDataset) -> None:
    """Across the NovaCart repo, a function is only split when it alone exceeds the limit."""
    splitter = chunker(size=350, overlap=50)
    for code_file in (f for f in dataset.code_files if f.language == "python"):
        source = code_file.content
        spans = splitter.split(source, SourceFormat.PYTHON)
        ranges = [(_line_span(source, s.start, s.end), s) for s in spans]
        for name, first, last in _definitions(source):
            if any(a <= first and last <= b for (a, b), _ in ranges):
                continue
            touching = [s for (a, b), s in ranges if a <= last and b >= first]
            assert all(s.metadata.get("split_symbol") for s in touching), f"{code_file.path}:{name}"
