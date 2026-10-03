"""chunks -> batching -> embedding -> pgvector.

Work is skipped wherever possible:
- a chunk is embedded only if it has no vector for this model or its embedded
  text changed (``text_hash`` covers the text and the provider fingerprint);
- identical texts within a run are embedded once;
- every batch is committed on its own, so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Iterable, Iterator, Sequence
from typing import Any

from pydantic import BaseModel
from sqlalchemy import delete, insert, inspect, select
from sqlalchemy.engine import Engine

from app.database.models import ChunkEmbedding, DocumentChunk
from app.database.schema import create_schema
from app.rag.embeddings.index import ensure_vector_index, refresh_statistics
from app.rag.embeddings.providers import EmbeddingProvider
from app.rag.embeddings.text import embedding_text
from app.schemas.enums import SourceType

logger = logging.getLogger(__name__)
C, E = DocumentChunk.__table__, ChunkEmbedding.__table__


class EmbeddingReport(BaseModel):
    model: str
    dimension: int
    total_chunks: int
    already_embedded: int
    embedded: int
    reused_identical_text: int
    batches: int
    truncated: int | None  # inputs longer than the model accepts (None: unknown)
    max_input_tokens: int | None
    index: str | None
    seconds: float

    @property
    def chunks_per_second(self) -> float:
        return round(self.embedded / self.seconds, 1) if self.seconds else 0.0


def text_hash(provider: EmbeddingProvider, text: str) -> str:
    return hashlib.sha256(f"{provider.fingerprint}\n{text}".encode()).hexdigest()


def _batches(items: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


class EmbeddingPipeline:
    def __init__(self, provider: EmbeddingProvider, batch_size: int = 32) -> None:
        self.provider = provider
        self.batch_size = batch_size

    def run(
        self, engine: Engine, source_types: Iterable[SourceType] | None = None, force: bool = False
    ) -> EmbeddingReport:
        started = time.perf_counter()
        create_schema(engine)
        if "record_hash" not in {c["name"] for c in inspect(engine).get_columns(C.name)}:
            raise RuntimeError(
                "document_chunks has an outdated schema; run scripts/ingest.py first"
            )
        model = self.provider.name
        query = select(C.c.id, C.c.source_type, C.c.title, C.c.section, C.c.service_id, C.c.content)
        if source_types is not None:
            query = query.where(C.c.source_type.in_(sorted(set(source_types))))
        with engine.connect() as connection:
            chunks = [dict(row) for row in connection.execute(query.order_by(C.c.id)).mappings()]
            existing = dict(
                connection.execute(
                    select(E.c.chunk_id, E.c.text_hash).where(E.c.model == model)
                ).all()
            )

        texts = {chunk["id"]: embedding_text(chunk) for chunk in chunks}
        hashes = {chunk_id: text_hash(self.provider, t) for chunk_id, t in texts.items()}
        todo = [i for i in texts if force or existing.get(i) != hashes[i]]
        unique: dict[str, str] = {}  # text hash -> text, first occurrence
        for chunk_id in todo:
            unique.setdefault(hashes[chunk_id], texts[chunk_id])

        truncated = None
        if unique and self.provider.max_input_tokens:
            counts = self.provider.count_tokens(list(unique.values()))
            if counts is not None:
                truncated = sum(n > self.provider.max_input_tokens for n in counts)

        vectors: dict[str, list[float]] = {}
        batches = 0
        pending = sorted(unique)
        for batch in _batches(pending, self.batch_size):
            batch_vectors = self.provider.embed_documents([unique[h] for h in batch])
            vectors.update(zip(batch, batch_vectors, strict=True))
            batches += 1
            in_batch = set(batch)
            self._write(engine, [i for i in todo if hashes[i] in in_batch], hashes, vectors)
        index = ensure_vector_index(engine, model, self.provider.dimension) if chunks else None
        if todo:
            refresh_statistics(engine)
        report = EmbeddingReport(
            model=model,
            dimension=self.provider.dimension,
            total_chunks=len(chunks),
            already_embedded=len(chunks) - len(todo),
            embedded=len(todo),
            reused_identical_text=len(todo) - len(unique),
            batches=batches,
            truncated=truncated,
            max_input_tokens=self.provider.max_input_tokens,
            index=index,
            seconds=round(time.perf_counter() - started, 2),
        )
        logger.info("embeddings.completed", extra=report.model_dump())
        return report

    def _write(
        self,
        engine: Engine,
        chunk_ids: list[str],
        hashes: dict[str, str],
        vectors: dict[str, list[float]],
    ) -> None:
        if not chunk_ids:
            return
        model = self.provider.name
        rows = [
            {
                "chunk_id": i,
                "model": model,
                "dimension": self.provider.dimension,
                "text_hash": hashes[i],
                "embedding": vectors[hashes[i]],
            }
            for i in chunk_ids
        ]
        with engine.begin() as connection:
            connection.execute(delete(E).where(E.c.model == model, E.c.chunk_id.in_(chunk_ids)))
            connection.execute(insert(E), rows)
