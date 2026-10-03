"""Turn ranked chunk ids into ``RetrievedChunk`` results with full provenance.

Rows are always read from the database at query time, with the caller's filters
applied again. A retriever's own index (vectors, BM25 postings) only proposes ids;
whether a chunk still exists and may be returned is decided by the current row.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from sqlalchemy import select
from sqlalchemy.engine import Engine

from app.database.models import DocumentChunk
from app.rag.retrieval.base import RetrievedChunk
from app.rag.store import ChunkFilter, filter_conditions


def fetch_chunks(
    engine: Engine, chunk_ids: Sequence[str], filters: ChunkFilter | None = None
) -> dict[str, Any]:
    """Current rows for ``chunk_ids`` that still exist and satisfy ``filters``."""
    if not chunk_ids:
        return {}
    c = DocumentChunk
    query = select(c.__table__).where(c.id.in_(list(chunk_ids)), *filter_conditions(filters))
    with engine.connect() as connection:
        return {row.id: row for row in connection.execute(query)}


def to_result(
    row: Any,
    *,
    score: float,
    rank: int,
    retriever: str,
    score_details: Mapping[str, float] | None = None,
    matched_terms: Sequence[str] = (),
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=row.id,
        document_id=row.document_id,
        source_type=row.source_type,
        title=row.title,
        section=row.section,
        content=row.content,
        service_id=row.service_id,
        timestamp=row.timestamp,
        access_level=row.access_level,
        version=row.version,
        doc_type=row.doc_type,
        file_path=row.file_path,
        metadata=row.metadata,
        score=round(score, 6),
        rank=rank,
        retriever=retriever,
        score_details=dict(score_details or {}),
        matched_terms=tuple(matched_terms),
    )


def hydrate(
    engine: Engine,
    hits: Sequence[tuple[str, float]],
    retriever: str,
    filters: ChunkFilter | None = None,
    matched: Mapping[str, Sequence[str]] | None = None,
) -> list[RetrievedChunk]:
    """``hits`` are (chunk id, score), best first. Ranks are assigned after dropping
    ids whose rows are gone or no longer match ``filters``."""
    rows = fetch_chunks(engine, [chunk_id for chunk_id, _ in hits], filters)
    results: list[RetrievedChunk] = []
    for chunk_id, score in hits:
        row = rows.get(chunk_id)
        if row is None:
            continue
        results.append(
            to_result(
                row,
                score=score,
                rank=len(results) + 1,
                retriever=retriever,
                score_details={f"{retriever}_score": round(score, 6)},
                matched_terms=(matched or {}).get(chunk_id, ()),
            )
        )
    return results
