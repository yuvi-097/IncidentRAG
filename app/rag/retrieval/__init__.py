"""Retrieval: a common ``Retriever`` interface and its implementations.

``factory`` (building retrievers from settings) is imported from its module
directly; it depends on ``app.rag.reranking``, which depends on this package.
"""

from app.rag.retrieval.base import MAX_TOP_K, RetrievalError, RetrievedChunk, Retriever
from app.rag.retrieval.bm25 import BM25Hit, BM25Index
from app.rag.retrieval.dense import DenseRetriever
from app.rag.retrieval.fusion import (
    FusedHit,
    fuse,
    reciprocal_rank_fusion,
    weighted_score_fusion,
)
from app.rag.retrieval.hybrid import HybridRetriever, WeightedRetriever
from app.rag.retrieval.pipeline import RetrievalPipeline
from app.rag.retrieval.sparse import BM25Retriever, SparseRetriever, lexical_text
from app.rag.retrieval.tokenizer import Tokenizer

__all__ = [
    "MAX_TOP_K",
    "BM25Hit",
    "BM25Index",
    "BM25Retriever",
    "DenseRetriever",
    "FusedHit",
    "HybridRetriever",
    "RetrievalError",
    "RetrievalPipeline",
    "RetrievedChunk",
    "Retriever",
    "SparseRetriever",
    "Tokenizer",
    "WeightedRetriever",
    "fuse",
    "lexical_text",
    "reciprocal_rank_fusion",
    "weighted_score_fusion",
]
