"""Embeddings: providers, the text that is embedded, the batch pipeline and vector indexes."""

from app.rag.embeddings.pipeline import EmbeddingPipeline, EmbeddingReport
from app.rag.embeddings.providers import (
    EmbeddingProvider,
    EmbeddingProviderError,
    HashingEmbeddingProvider,
    build_embedding_provider,
)

__all__ = [
    "EmbeddingPipeline",
    "EmbeddingProvider",
    "EmbeddingProviderError",
    "EmbeddingReport",
    "HashingEmbeddingProvider",
    "build_embedding_provider",
]
