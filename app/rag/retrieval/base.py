"""The retriever contract.

    Retriever                     search(query, top_k, filters) -> ranked chunks
    ├── DenseRetriever            embeddings + pgvector
    ├── SparseRetriever           lexical retrieval (interface)
    │   └── BM25Retriever         Okapi BM25 over an in-memory inverted index
    ├── HybridRetriever           fuses several retrievers (RRF or weighted scores)
    └── RetrievalPipeline         first stage (e.g. hybrid) + optional reranker

Every result carries the chunk's provenance (source record, section, file path,
offsets via ``chunk_id``) so later stages can cite and verify it, plus the evidence
for its rank (``score_details``, ``matched_terms``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from app.rag.store import ChunkFilter
from app.schemas.enums import AccessLevel, SourceType

MAX_TOP_K = 100


class RetrievalError(ValueError):
    pass


class RetrievedChunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: str
    document_id: str
    source_type: SourceType
    title: str
    section: str | None
    content: str
    service_id: str | None
    timestamp: datetime
    access_level: AccessLevel
    version: str | None
    doc_type: str | None
    file_path: str | None
    metadata: dict[str, Any]
    score: float  # retriever-specific; higher is more relevant
    rank: int  # 1-based
    retriever: str
    # Why it ranked here: per-stage scores and ranks, e.g. {"dense_rank": 3, "bm25_rank": 1}.
    score_details: dict[str, float] = {}
    matched_terms: tuple[str, ...] = ()  # query terms found in the chunk (lexical retrievers)


class Retriever(ABC):
    name: str

    @abstractmethod
    def search(
        self, query: str, top_k: int = 10, filters: ChunkFilter | None = None
    ) -> list[RetrievedChunk]:
        """Return at most ``top_k`` chunks matching ``filters``, most relevant first."""

    @staticmethod
    def validate(query: str, top_k: int) -> str:
        query = query.strip()
        if not query:
            raise RetrievalError("query must not be empty")
        if not 1 <= top_k <= MAX_TOP_K:
            raise RetrievalError(f"top_k must be between 1 and {MAX_TOP_K}")
        return query
