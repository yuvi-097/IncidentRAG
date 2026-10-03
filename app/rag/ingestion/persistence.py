"""Stage 5: write chunks to ``document_chunks``.

Writes are a diff, not a rewrite. Unchanged chunks (same ``record_hash``) are left
alone, so their embeddings survive re-ingestion. Changed chunks are replaced (their
now-stale embeddings go with them), and chunks that no longer exist are deleted.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass

from sqlalchemy import delete, insert, inspect, select
from sqlalchemy.engine import Engine

from app.database.models import ChunkEmbedding, DocumentChunk
from app.database.schema import create_schema
from app.rag.ingestion.models import ChunkRecord
from app.schemas.enums import SourceType

logger = logging.getLogger(__name__)
CHUNKS = DocumentChunk.__table__
EMBEDDINGS = ChunkEmbedding.__table__
# Derived tables, rebuilt from sources when their schema changes (dependents first).
DERIVED_TABLES = (EMBEDDINGS, CHUNKS)


@dataclass(frozen=True)
class SyncResult:
    inserted: int
    updated: int
    deleted: int
    unchanged: int


def ensure_chunk_table(engine: Engine) -> bool:
    """Create missing tables, and rebuild the derived tables (chunks, embeddings) if
    their columns are outdated. Derived data is re-created by ingestion and embedding,
    so nothing is lost. Returns True when the tables were rebuilt."""
    create_schema(engine)
    inspector = inspect(engine)
    outdated = [
        table.name
        for table in DERIVED_TABLES
        if {c["name"] for c in inspector.get_columns(table.name)} != {c.name for c in table.columns}
    ]
    if not outdated:
        return False
    logger.warning("chunks.tables_rebuilt", extra={"reason": "schema changed", "tables": outdated})
    for table in DERIVED_TABLES:  # embeddings reference chunks: drop them first
        table.drop(engine, checkfirst=True)
    for table in reversed(DERIVED_TABLES):
        table.create(engine)
    return True


def replace_chunks(
    engine: Engine,
    chunks: list[ChunkRecord],
    source_types: Iterable[SourceType],
    batch_size: int = 1000,
) -> SyncResult:
    """Make the stored chunks of ``source_types`` equal ``chunks``, in one transaction."""
    types = sorted(set(source_types))
    wanted = {chunk.id: chunk for chunk in chunks}
    with engine.begin() as connection:
        stored = dict(
            connection.execute(
                select(CHUNKS.c.id, CHUNKS.c.record_hash).where(CHUNKS.c.source_type.in_(types))
            ).all()
        )
        changed = {i for i, h in stored.items() if i in wanted and wanted[i].record_hash != h}
        removed = set(stored) - set(wanted)
        added = set(wanted) - set(stored)
        stale = sorted(changed | removed)
        for start in range(0, len(stale), batch_size):
            connection.execute(
                delete(CHUNKS).where(CHUNKS.c.id.in_(stale[start : start + batch_size]))
            )
        rows = [wanted[i].model_dump() for i in sorted(added | changed)]
        for start in range(0, len(rows), batch_size):
            connection.execute(insert(CHUNKS), rows[start : start + batch_size])
    return SyncResult(
        inserted=len(added),
        updated=len(changed),
        deleted=len(removed),
        unchanged=len(stored) - len(changed) - len(removed),
    )
