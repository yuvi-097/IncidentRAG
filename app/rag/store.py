"""Metadata filters over stored chunks.

``filter_conditions`` is the single definition of what each filter means; plain
listing (``filtered_chunks``), vector search and BM25 all use it. Every filter maps
to an indexed column.

Access: ``access`` lists the (source type, label) pairs the caller may read (see
``Principal.chunk_access``); only chunks matching one of them are selected, and an
empty set selects nothing. Confidential chunks are excluded by every query, with or
without a filter (ingestion does not create them either).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator
from sqlalchemy import ColumnElement, Select, and_, false, or_, select

from app.database.models import DocumentChunk
from app.schemas.enums import AccessLevel, SourceType


class ChunkFilter(BaseModel):
    model_config = ConfigDict(frozen=True)

    services: frozenset[str] | None = None
    source_types: frozenset[SourceType] | None = None
    doc_types: frozenset[str] | None = None
    versions: frozenset[str] | None = None
    access: frozenset[tuple[SourceType, AccessLevel]] | None = None
    since: datetime | None = None  # inclusive
    until: datetime | None = None  # exclusive
    document_ids: frozenset[str] | None = None

    @model_validator(mode="after")
    def _valid_range(self) -> ChunkFilter:
        if self.since and self.until and self.since >= self.until:
            raise ValueError("since must be earlier than until")
        return self

    @property
    def is_empty(self) -> bool:
        return all(value is None for value in self.model_dump().values())


def access_condition(access: frozenset[tuple[SourceType, AccessLevel]]) -> ColumnElement[Any]:
    c = DocumentChunk
    by_source: dict[SourceType, set[AccessLevel]] = {}
    for source_type, level in access:
        if level is not AccessLevel.CONFIDENTIAL:
            by_source.setdefault(source_type, set()).add(level)
    if not by_source:
        return false()
    return or_(
        *(
            and_(c.source_type == source_type, c.access_level.in_(sorted(levels)))
            for source_type, levels in sorted(by_source.items())
        )
    )


def filter_conditions(chunk_filter: ChunkFilter | None) -> list[ColumnElement[Any]]:
    c = DocumentChunk
    conditions: list[ColumnElement[Any]] = [c.access_level != AccessLevel.CONFIDENTIAL]
    if chunk_filter is None:
        return conditions
    if chunk_filter.services is not None:
        conditions.append(c.service_id.in_(sorted(chunk_filter.services)))
    if chunk_filter.source_types is not None:
        conditions.append(c.source_type.in_(sorted(chunk_filter.source_types)))
    if chunk_filter.doc_types is not None:
        conditions.append(c.doc_type.in_(sorted(chunk_filter.doc_types)))
    if chunk_filter.versions is not None:
        conditions.append(c.version.in_(sorted(chunk_filter.versions)))
    if chunk_filter.access is not None:
        conditions.append(access_condition(chunk_filter.access))
    if chunk_filter.since is not None:
        conditions.append(c.timestamp >= chunk_filter.since)
    if chunk_filter.until is not None:
        conditions.append(c.timestamp < chunk_filter.until)
    if chunk_filter.document_ids is not None:
        conditions.append(c.document_id.in_(sorted(chunk_filter.document_ids)))
    return conditions


def filtered_chunks(chunk_filter: ChunkFilter) -> Select[tuple[DocumentChunk]]:
    c = DocumentChunk
    return select(c).where(*filter_conditions(chunk_filter)).order_by(c.document_id, c.chunk_index)
