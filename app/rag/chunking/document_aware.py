"""Document-aware chunking: pick the structure-preserving splitter for each format."""

from __future__ import annotations

from app.rag.chunking.base import ChunkingConfig, ChunkSpan, SourceFormat
from app.rag.chunking.code import CodeChunker
from app.rag.chunking.markdown import MarkdownChunker
from app.rag.chunking.recursive import RecursiveChunker
from app.schemas.enums import ChunkingStrategy


class DocumentAwareChunker:
    strategy = ChunkingStrategy.DOCUMENT_AWARE

    def __init__(self, config: ChunkingConfig) -> None:
        self.markdown = MarkdownChunker(config)
        self.code = CodeChunker(config)
        self.plain = RecursiveChunker(config)

    def split(self, text: str, fmt: SourceFormat) -> list[ChunkSpan]:
        if fmt is SourceFormat.MARKDOWN:
            return self.markdown.split(text)
        if fmt is SourceFormat.PYTHON:
            return self.code.split_python(text)
        if fmt is SourceFormat.YAML:
            return self.code.split_yaml(text)
        return self.plain.split(text, fmt)  # no structure to preserve
