from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine

from app.database.seed import seed_database
from app.rag.chunking import ChunkingConfig
from app.rag.ingestion import ChunkRecord, IngestionPipeline, IngestionReport
from app.rag.ingestion.persistence import ensure_chunk_table
from app.synthetic.records import SyntheticDataset

DEFAULT_CONFIG = ChunkingConfig()


@dataclass
class Ingested:
    engine: Engine
    chunks: list[ChunkRecord]
    report: IngestionReport
    dataset: SyntheticDataset


def sqlite_engine_with_fks() -> Engine:
    engine = create_engine("sqlite://")

    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection, _record) -> None:
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    return engine


@pytest.fixture(scope="module")
def ingested(dataset: SyntheticDataset) -> Iterator[Ingested]:
    """The full corpus seeded into SQLite (logs omitted: not an ingestion source) and ingested."""
    engine = sqlite_engine_with_fks()
    ensure_chunk_table(engine)
    corpus = dataset.model_copy(update={"logs": []})
    seed_database(engine, corpus)
    chunks, report = IngestionPipeline(DEFAULT_CONFIG).run(engine)
    yield Ingested(engine, chunks, report, corpus)
    engine.dispose()
