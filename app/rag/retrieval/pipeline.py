"""The retrieval pipeline: a first stage proposes candidates, a reranker reorders them.

    query -> first stage (e.g. hybrid: dense + BM25, fused) -> top ``candidates``
          -> reranker (cross-encoder) -> top_k

Filters are applied by the first stage, so the reranker only ever sees chunks the
caller may see.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.observability.metrics import METRICS
from app.rag.retrieval.base import MAX_TOP_K, RetrievedChunk, Retriever
from app.rag.store import ChunkFilter

if TYPE_CHECKING:  # app.rag.reranking imports this package; avoid the cycle at runtime
    from app.rag.reranking.base import Reranker


class RetrievalPipeline(Retriever):
    def __init__(
        self, first_stage: Retriever, reranker: Reranker | None = None, candidates: int = 30
    ) -> None:
        if not 1 <= candidates <= MAX_TOP_K:
            raise ValueError(f"candidates must be between 1 and {MAX_TOP_K}")
        self.first_stage = first_stage
        self.reranker = reranker
        self.candidates = candidates
        self.name = f"{first_stage.name}+rerank" if reranker else first_stage.name

    def search(
        self, query: str, top_k: int = 10, filters: ChunkFilter | None = None
    ) -> list[RetrievedChunk]:
        query = self.validate(query, top_k)
        if self.reranker is None:
            with METRICS.timed("retrieval.first_stage"):
                return self.first_stage.search(query, top_k, filters)
        with METRICS.timed("retrieval.first_stage"):
            pool = self.first_stage.search(query, max(self.candidates, top_k), filters)
        with METRICS.timed("retrieval.rerank"):
            return self.reranker.rerank(query, pool, top_k)


class TimedRetriever(Retriever):
    """Records every search of the wrapped retriever as ``retrieval.search``, in any
    retrieval mode (the agent's tools are given this wrapper)."""

    def __init__(self, inner: Retriever) -> None:
        self.inner = inner
        self.name = inner.name

    def search(
        self, query: str, top_k: int = 10, filters: ChunkFilter | None = None
    ) -> list[RetrievedChunk]:
        with METRICS.timed("retrieval.search"):
            return self.inner.search(query, top_k, filters)
