"""Dense retrieval: embed the query with the configured model, find the nearest chunk
embeddings of the same model, return the chunks with provenance."""

from __future__ import annotations

from sqlalchemy.engine import Engine

from app.rag.embeddings.providers import EmbeddingProvider
from app.rag.retrieval.base import RetrievedChunk, Retriever
from app.rag.retrieval.hydrate import hydrate
from app.rag.retrieval.vector_store import VectorStore, vector_store_for
from app.rag.store import ChunkFilter


class DenseRetriever(Retriever):
    name = "dense"

    def __init__(
        self,
        engine: Engine,
        provider: EmbeddingProvider,
        ef_search: int = 100,
        store: VectorStore | None = None,
    ) -> None:
        self.engine = engine
        self.provider = provider
        self.store = store or vector_store_for(engine, provider.name, provider.dimension, ef_search)

    def search(
        self, query: str, top_k: int = 10, filters: ChunkFilter | None = None
    ) -> list[RetrievedChunk]:
        query = self.validate(query, top_k)
        hits = self.store.nearest(self.provider.embed_query(query), top_k, filters)
        return hydrate(self.engine, hits, self.name, filters)
