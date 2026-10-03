"""The reranker contract.

A reranker reorders candidates that a first-stage retriever already selected (and
already filtered). It never adds chunks, so it cannot surface anything the
metadata or access filters excluded.

``PairScorer`` is the model boundary: it scores (query, passage) pairs, and knows
nothing about chunks. Rerankers are tested with fake scorers, and the real
cross-encoder is tested separately.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Protocol

from app.rag.embeddings.text import embedding_text
from app.rag.retrieval.base import RetrievedChunk


class RerankerError(RuntimeError):
    pass


class PairScorer(Protocol):
    name: str

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        """One relevance score per (query, passage) pair; higher is more relevant."""
        ...


def rerank_text(chunk: RetrievedChunk) -> str:
    """The passage a reranker sees: the same header + content that is embedded."""
    return embedding_text(
        {
            "source_type": chunk.source_type,
            "title": chunk.title,
            "section": chunk.section,
            "service_id": chunk.service_id,
            "content": chunk.content,
        }
    )


class Reranker(ABC):
    name: str

    @abstractmethod
    def rerank(
        self, query: str, candidates: Sequence[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        """Return at most ``top_k`` of ``candidates``, most relevant first."""
