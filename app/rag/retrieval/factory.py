"""Assemble retrievers from configuration.

Models are loaded only for the components a mode needs (BM25 needs none), and each
component is built once and shared, so the four modes can be compared on exactly
the same components.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy.engine import Engine

from app.config import Settings
from app.rag.embeddings import EmbeddingProvider, build_embedding_provider
from app.rag.reranking import Reranker, build_reranker
from app.rag.retrieval.base import Retriever
from app.rag.retrieval.dense import DenseRetriever
from app.rag.retrieval.hybrid import HybridRetriever, WeightedRetriever
from app.rag.retrieval.pipeline import RetrievalPipeline
from app.rag.retrieval.sparse import BM25Retriever
from app.rag.retrieval.tokenizer import Tokenizer
from app.schemas.enums import RetrievalMode


@dataclass
class RetrievalComponents:
    """Lazily built, shared building blocks for every retrieval mode."""

    engine: Engine
    settings: Settings
    embedding_provider: EmbeddingProvider | None = None
    reranker: Reranker | None = None
    _dense: DenseRetriever | None = field(default=None, init=False)
    _bm25: BM25Retriever | None = field(default=None, init=False)

    @property
    def dense(self) -> DenseRetriever:
        if self._dense is None:
            provider = self.embedding_provider or build_embedding_provider(self.settings.embedding)
            self.embedding_provider = provider
            self._dense = DenseRetriever(
                self.engine, provider, ef_search=self.settings.retrieval.hnsw_ef_search
            )
        return self._dense

    @property
    def bm25(self) -> BM25Retriever:
        if self._bm25 is None:
            config = self.settings.bm25
            self._bm25 = BM25Retriever(
                self.engine,
                Tokenizer(stemming=config.stemming, stopwords=config.stopwords),
                k1=config.k1,
                b=config.b,
                part_weight=config.part_weight,
            )
        return self._bm25

    def hybrid(self) -> HybridRetriever:
        config = self.settings.retrieval
        components = []
        if config.dense_weight > 0:
            components.append(WeightedRetriever(self.dense, config.dense_weight))
        if config.sparse_weight > 0:
            components.append(WeightedRetriever(self.bm25, config.sparse_weight))
        return HybridRetriever(components, config.fusion, config.rrf_k, config.fusion_depth)

    def reranking(self) -> Reranker:
        if self.reranker is None:
            self.reranker = build_reranker(self.settings.reranker)
        if self.reranker is None:
            raise ValueError("reranking needs RERANKER_PROVIDER (it is 'none')")
        return self.reranker

    def retriever(self, mode: RetrievalMode | None = None) -> Retriever:
        mode = RetrievalMode(mode or self.settings.retrieval.mode)
        if mode == RetrievalMode.DENSE:
            return self.dense
        if mode == RetrievalMode.SPARSE:
            return self.bm25
        if mode == RetrievalMode.HYBRID:
            return self.hybrid()
        return RetrievalPipeline(
            self.hybrid(), self.reranking(), self.settings.retrieval.rerank_candidates
        )


def build_retriever(
    engine: Engine, settings: Settings, mode: RetrievalMode | None = None
) -> Retriever:
    """The retriever for ``mode`` (default: ``RETRIEVAL_MODE``)."""
    return RetrievalComponents(engine, settings).retriever(mode)
