"""Nearest-neighbour search over ``chunk_embeddings``.

``PgVectorStore`` is the production path (pgvector, HNSW). ``InMemoryVectorStore``
computes exact cosine similarity in NumPy over vectors stored in any SQL database;
it exists so the same retriever runs on SQLite in unit tests. Both apply metadata
filters in SQL through ``filter_conditions``, so filtering behaves identically.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from pgvector.sqlalchemy import Vector
from sqlalchemy import cast, literal, select, text
from sqlalchemy.engine import Engine

from app.database.models import ChunkEmbedding, DocumentChunk
from app.rag.embeddings.index import validate_model_name
from app.rag.store import ChunkFilter, filter_conditions

Hit = tuple[str, float]  # (chunk id, cosine similarity)


class VectorStore(ABC):
    def __init__(self, engine: Engine, model: str, dimension: int) -> None:
        self.engine = engine
        self.model = validate_model_name(model)
        self.dimension = int(dimension)

    @abstractmethod
    def nearest(
        self, vector: list[float], top_k: int, filters: ChunkFilter | None
    ) -> list[Hit]: ...


class PgVectorStore(VectorStore):
    """HNSW approximate search. HNSW scans a fixed-size candidate list and then applies
    the WHERE clause, so a selective filter can return fewer than ``top_k`` rows. When
    that happens with filters present, the query is repeated as an exact scan."""

    def __init__(self, engine: Engine, model: str, dimension: int, ef_search: int = 100) -> None:
        super().__init__(engine, model, dimension)
        self.ef_search = ef_search

    def nearest(self, vector: list[float], top_k: int, filters: ChunkFilter | None) -> list[Hit]:
        e, c = ChunkEmbedding, DocumentChunk
        # Same expression as the partial HNSW index, and the model as a literal,
        # so the planner can match the index.
        distance = cast(e.embedding, Vector(self.dimension)).cosine_distance(vector)
        conditions = filter_conditions(filters)
        query = (
            select(e.chunk_id, distance.label("distance"))
            .join(c, c.id == e.chunk_id)
            .where(e.model == literal(self.model, literal_execute=True), *conditions)
            .order_by(distance)
            .limit(top_k)
        )
        ef_search = max(self.ef_search, 4 * top_k)
        with self.engine.begin() as connection:
            connection.execute(text(f"SET LOCAL hnsw.ef_search = {int(ef_search)}"))
            rows = connection.execute(query).all()
            if conditions and len(rows) < top_k:
                connection.execute(text("SET LOCAL enable_indexscan = off"))  # exact search
                rows = connection.execute(query).all()
        return [(chunk_id, 1.0 - float(d)) for chunk_id, d in rows]


class InMemoryVectorStore(VectorStore):
    """Exact search in NumPy (brute force); for tests and non-PostgreSQL databases."""

    def nearest(self, vector: list[float], top_k: int, filters: ChunkFilter | None) -> list[Hit]:
        e, c = ChunkEmbedding, DocumentChunk
        query = (
            select(e.chunk_id, e.embedding)
            .join(c, c.id == e.chunk_id)
            .where(e.model == self.model, *filter_conditions(filters))
        )
        with self.engine.connect() as connection:
            rows = connection.execute(query).all()
        if not rows:
            return []
        ids = [row[0] for row in rows]
        matrix = np.asarray([row[1] for row in rows], dtype=np.float32)
        q = np.asarray(vector, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1) * (np.linalg.norm(q) or 1.0)
        scores = (matrix @ q) / np.where(norms == 0, 1.0, norms)
        order = sorted(range(len(ids)), key=lambda i: (-float(scores[i]), ids[i]))[:top_k]
        return [(ids[i], float(scores[i])) for i in order]


def vector_store_for(
    engine: Engine, model: str, dimension: int, ef_search: int = 100
) -> VectorStore:
    if engine.dialect.name == "postgresql":
        return PgVectorStore(engine, model, dimension, ef_search)
    return InMemoryVectorStore(engine, model, dimension)
