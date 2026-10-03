"""Sparse (lexical) retrieval.

``SparseRetriever`` is the interface: a retriever that matches query *terms*, can
say which terms a result matched, and keeps an index that must be refreshed when
the chunk store changes. ``BM25Retriever`` implements it with Okapi BM25 over an
in-memory inverted index built from ``document_chunks``. That fits this corpus
(thousands of chunks, index built in about a second). A PostgreSQL or search-engine
backed implementation can replace it behind the same interface.

Filters are never evaluated against the in-memory index: the ids allowed by the
filters, and the rows returned, are read from the database at query time. A stale
index can therefore miss new text, but cannot return a chunk the filters exclude.
"""

from __future__ import annotations

import logging
import threading
import time
from abc import abstractmethod
from collections.abc import Mapping
from typing import Any

from sqlalchemy import select
from sqlalchemy.engine import Engine

from app.database.models import DocumentChunk
from app.rag.embeddings.text import embedding_text
from app.rag.retrieval.base import RetrievedChunk, Retriever
from app.rag.retrieval.bm25 import BM25Index
from app.rag.retrieval.hydrate import hydrate
from app.rag.retrieval.tokenizer import Tokenizer
from app.rag.store import ChunkFilter, filter_conditions

logger = logging.getLogger(__name__)


def lexical_text(chunk: Mapping[str, Any]) -> str:
    """What BM25 indexes: the same header + content that is embedded, plus the chunk's
    identifiers (source record id, file path, version) so that exact references such
    as ``INC-0406`` or ``payment_service/config.py`` match the record itself."""
    identifiers = [chunk.get("document_id"), chunk.get("file_path"), chunk.get("version")]
    extra = " ".join(str(value) for value in identifiers if value)
    return embedding_text(chunk) + ("\n" + extra if extra else "")


class SparseRetriever(Retriever):
    name = "sparse"

    @abstractmethod
    def analyze(self, text: str) -> list[str]:
        """The terms this retriever derives from ``text`` (queries and documents alike)."""

    def analyze_query(self, query: str) -> dict[str, float]:
        """Query terms with their weights (all 1 unless a retriever weights them)."""
        return dict.fromkeys(self.analyze(query), 1.0)

    @abstractmethod
    def refresh(self) -> None:
        """Rebuild the index from the chunk store (call after ingestion)."""


class BM25Retriever(SparseRetriever):
    name = "bm25"

    def __init__(
        self,
        engine: Engine,
        tokenizer: Tokenizer | None = None,
        k1: float = 1.2,
        b: float = 0.75,
        part_weight: float = 0.75,
    ) -> None:
        self.engine = engine
        self.tokenizer = tokenizer or Tokenizer()
        self.k1 = k1
        self.b = b
        self.part_weight = part_weight
        self._index: BM25Index | None = None
        self._lock = threading.Lock()

    @property
    def index(self) -> BM25Index:
        """Built on first use."""
        if self._index is None:
            with self._lock:
                if self._index is None:
                    self._index = self._build()
        return self._index

    def refresh(self) -> None:
        index = self._build()
        with self._lock:
            self._index = index

    def analyze(self, text: str) -> list[str]:
        return self.tokenizer.tokenize(text)

    def analyze_query(self, query: str) -> dict[str, float]:
        return self.tokenizer.query_weights(query, self.part_weight)

    def _build(self) -> BM25Index:
        started = time.perf_counter()
        c = DocumentChunk
        query = (
            select(
                c.id,
                c.document_id,
                c.source_type,
                c.title,
                c.section,
                c.service_id,
                c.file_path,
                c.version,
                c.content,
            )
            .where(*filter_conditions(None))
            .order_by(c.id)
        )  # never confidential
        with self.engine.connect() as connection:
            rows = [dict(row) for row in connection.execute(query).mappings()]
        index = BM25Index.build(
            ((row["id"], self.analyze(lexical_text(row))) for row in rows), self.k1, self.b
        )
        logger.info(
            "bm25.index_built",
            extra={
                "chunks": len(index),
                "terms": len(index.postings),
                "seconds": round(time.perf_counter() - started, 2),
            },
        )
        return index

    def _allowed_ids(self, filters: ChunkFilter) -> set[str]:
        c = DocumentChunk
        with self.engine.connect() as connection:
            return set(
                connection.execute(select(c.id).where(*filter_conditions(filters))).scalars()
            )

    def search(
        self, query: str, top_k: int = 10, filters: ChunkFilter | None = None
    ) -> list[RetrievedChunk]:
        query = self.validate(query, top_k)
        terms = self.analyze_query(query)
        if not terms:
            return []
        filtered = filters is not None and not filters.is_empty
        allowed = self._allowed_ids(filters) if filters is not None and filtered else None
        hits = self.index.search(terms, top_k, allowed)
        return hydrate(
            self.engine,
            [(hit.doc_id, hit.score) for hit in hits],
            self.name,
            filters,
            matched={hit.doc_id: hit.matched_terms for hit in hits},
        )
