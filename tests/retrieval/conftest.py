from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import pytest
from sqlalchemy.engine import Engine

from app.database.seed import seed_database
from app.rag.chunking import ChunkingConfig
from app.rag.embeddings import EmbeddingPipeline, EmbeddingReport, HashingEmbeddingProvider
from app.rag.ingestion import ChunkRecord, IngestionPipeline
from app.rag.ingestion.persistence import ensure_chunk_table
from app.synthetic.records import SyntheticDataset
from tests.rag.conftest import sqlite_engine_with_fks


class CountingProvider(HashingEmbeddingProvider):
    """Hashing provider that records how many texts were actually embedded."""

    def __init__(self, dimension: int = 64, name: str = "counting-hash") -> None:
        super().__init__(dimension=dimension, name=name)
        self.embedded_texts = 0
        self.calls = 0

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        self.embedded_texts += len(texts)
        return super().embed_documents(texts)


@dataclass
class Corpus:
    engine: Engine
    chunks: list[ChunkRecord]
    dataset: SyntheticDataset


def build_corpus(dataset: SyntheticDataset) -> Corpus:
    engine = sqlite_engine_with_fks()
    ensure_chunk_table(engine)
    corpus = dataset.model_copy(update={"logs": []})
    seed_database(engine, corpus)
    chunks, _ = IngestionPipeline(ChunkingConfig()).run(engine)
    return Corpus(engine, chunks, corpus)


@dataclass
class Embedded(Corpus):
    provider: HashingEmbeddingProvider
    report: EmbeddingReport


@pytest.fixture(scope="module")
def embedded(dataset: SyntheticDataset) -> Iterator[Embedded]:
    """Full corpus in SQLite, embedded with the deterministic hashing provider."""
    corpus = build_corpus(dataset)
    provider = HashingEmbeddingProvider(dimension=256)
    report = EmbeddingPipeline(provider, batch_size=128).run(corpus.engine)
    yield Embedded(corpus.engine, corpus.chunks, corpus.dataset, provider, report)
    corpus.engine.dispose()
