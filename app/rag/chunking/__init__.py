"""Chunking strategies. All return exact character spans into the input text."""

from app.rag.chunking.base import Chunker, ChunkingConfig, ChunkSpan, SourceFormat
from app.rag.chunking.document_aware import DocumentAwareChunker
from app.rag.chunking.fixed import FixedSizeChunker
from app.rag.chunking.recursive import RecursiveChunker
from app.rag.chunking.tokens import TokenIndex, count_tokens
from app.schemas.enums import ChunkingStrategy

_CHUNKERS = {
    ChunkingStrategy.FIXED: FixedSizeChunker,
    ChunkingStrategy.RECURSIVE: RecursiveChunker,
    ChunkingStrategy.DOCUMENT_AWARE: DocumentAwareChunker,
}


def build_chunker(config: ChunkingConfig) -> Chunker:
    return _CHUNKERS[config.strategy](config)


__all__ = [
    "ChunkSpan",
    "Chunker",
    "ChunkingConfig",
    "SourceFormat",
    "TokenIndex",
    "build_chunker",
    "count_tokens",
]
