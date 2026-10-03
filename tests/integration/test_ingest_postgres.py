"""Ingestion against real PostgreSQL + pgvector.

Opt in with OPSRAG_RUN_INTEGRATION_TESTS=1. Seeds the dataset and REPLACES
document_chunks in the configured database; use a development database only.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import load_settings
from app.database.seed import seed_database
from app.database.session import create_db_engine
from app.rag.chunking import ChunkingConfig
from app.rag.ingestion import ChunkRecord, IngestionPipeline
from app.rag.ingestion.persistence import ensure_chunk_table
from app.rag.ingestion.pipeline import source_text
from app.rag.ingestion.sources import load_sources
from app.rag.store import ChunkFilter, filtered_chunks
from app.schemas.enums import AccessLevel, SourceType
from app.security import SOURCE_RESOURCE, principal_for_role
from app.synthetic.records import SyntheticDataset

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("OPSRAG_RUN_INTEGRATION_TESTS") != "1",
        reason="set OPSRAG_RUN_INTEGRATION_TESTS=1 with PostgreSQL running",
    ),
]


@pytest.fixture(scope="module")
def ingested(dataset: SyntheticDataset) -> Iterator[tuple[Engine, list[ChunkRecord]]]:
    settings = load_settings()
    if settings.app.environment == "production":
        pytest.skip("refusing to write synthetic data into production")
    engine = create_db_engine(settings.database)
    ensure_chunk_table(engine)
    seed_database(engine, dataset)
    chunks, report = IngestionPipeline(ChunkingConfig()).run(engine)
    assert [s.source_id for s in report.skipped] == ["DOC-0100"]  # confidential
    yield engine, chunks
    engine.dispose()


def test_all_chunks_are_stored(ingested: tuple[Engine, list[ChunkRecord]]) -> None:
    engine, chunks = ingested
    with engine.connect() as connection:
        stored = connection.execute(text("SELECT count(*) FROM document_chunks")).scalar_one()
        by_type = dict(
            connection.execute(
                text("SELECT source_type, count(*) FROM document_chunks GROUP BY source_type")
            ).all()
        )
    assert stored == len(chunks)
    assert set(by_type) == {t.value for t in SourceType}


def test_stored_chunks_trace_back_to_their_sources(
    ingested: tuple[Engine, list[ChunkRecord]],
) -> None:
    engine, _ = ingested
    with engine.connect() as connection:
        sources = {s.source_id: s for s in load_sources(connection)}
        rows = connection.execute(
            text("SELECT document_id, char_start, char_end, content FROM document_chunks")
        ).all()
    texts: dict[str, str] = {}
    for document_id, start, end, content in rows:
        if document_id not in texts:
            texts[document_id] = source_text(sources[document_id])
        assert texts[document_id][start:end] == content


def test_chunks_do_not_depend_on_the_database_backend(
    ingested: tuple[Engine, list[ChunkRecord]], dataset: SyntheticDataset
) -> None:
    """Same data, same config: PostgreSQL and SQLite must yield byte-identical chunks
    (guards against JSONB key reordering and collation-dependent row order)."""
    _, pg_chunks = ingested
    sqlite = create_engine("sqlite://")
    ensure_chunk_table(sqlite)
    seed_database(sqlite, dataset.model_copy(update={"logs": []}))
    sqlite_chunks, _ = IngestionPipeline(ChunkingConfig()).run(sqlite, dry_run=True)
    sqlite.dispose()

    def key(c: ChunkRecord) -> tuple[str, str, int, int]:
        return (c.id, c.content, c.char_start, c.char_end)

    assert sorted(map(key, pg_chunks)) == sorted(map(key, sqlite_chunks))


def test_postgres_enforces_chunk_provenance(ingested: tuple[Engine, list[ChunkRecord]]) -> None:
    engine, chunks = ingested
    row = chunks[0].model_dump() | {
        "id": "PROBE#999",
        "chunk_index": 999,
        "source_document_id": None,
        "source_incident_id": None,
        "source_deployment_id": None,
        "source_code_file_id": None,
        "source_pull_request_id": None,
    }
    from app.database.models import DocumentChunk

    with pytest.raises(IntegrityError), engine.begin() as connection:
        connection.execute(DocumentChunk.__table__.insert(), [row])


SRE_SOURCES = {
    st for st, r in SOURCE_RESOURCE.items() if principal_for_role("pg-sre", "sre").labels(r)
}


def test_metadata_filters_use_real_columns(ingested: tuple[Engine, list[ChunkRecord]]) -> None:
    engine, chunks = ingested
    query = filtered_chunks(
        ChunkFilter(
            services=frozenset({"payment-service"}),
            versions=frozenset({"v2.8.1"}),
            access=principal_for_role("pg-sre", "sre").chunk_access(),
        )
    )
    with Session(engine) as session:
        found = {c.id for c in session.scalars(query)}
    expected = {
        c.id
        for c in chunks
        if c.service_id == "payment-service"
        and c.version == "v2.8.1"
        and c.access_level in {AccessLevel.PUBLIC, AccessLevel.ENGINEERING, AccessLevel.SRE}
        and c.source_type in SRE_SOURCES
    }
    assert found == expected and found


def test_filter_columns_are_indexed(ingested: tuple[Engine, list[ChunkRecord]]) -> None:
    engine, _ = ingested
    with engine.connect() as connection:
        indexed = {
            row[0].replace('"', "")  # keywords such as "timestamp" are quoted in indexdef
            for row in connection.execute(
                text("SELECT indexdef FROM pg_indexes WHERE tablename = 'document_chunks'")
            )
        }
    for column in (
        "service_id",
        "source_type",
        "timestamp",
        "access_level",
        "version",
        "doc_type",
        "document_id",
    ):
        assert any(
            f"({column})" in definition or f"({column}," in definition for definition in indexed
        ), column
    # Vector indexes live on chunk_embeddings (tests/integration/test_retrieval_postgres.py).
