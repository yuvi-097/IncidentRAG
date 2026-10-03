"""Types passed between ingestion stages."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from app.rag.chunking.base import SourceFormat
from app.schemas.enums import AccessLevel, ChunkingStrategy, SourceType

# Which typed foreign key on document_chunks points at each source type.
SOURCE_FOREIGN_KEY: dict[SourceType, str] = {
    SourceType.INCIDENT: "source_incident_id",
    SourceType.RUNBOOK: "source_document_id",
    SourceType.DOCUMENTATION: "source_document_id",
    SourceType.POSTMORTEM: "source_document_id",
    SourceType.DEPLOYMENT: "source_deployment_id",
    SourceType.CODE: "source_code_file_id",
    SourceType.PULL_REQUEST: "source_pull_request_id",
}


class SkipSource(Exception):
    """A source that cannot be chunked (empty or malformed); ingestion continues."""


@dataclass(frozen=True)
class RawSource:
    """A source record as read from the database (stage 1: raw data)."""

    source_type: SourceType
    source_id: str
    payload: dict[str, Any]


@dataclass
class ParsedSource:
    """Canonical text plus source-level metadata (stages 2-4)."""

    source_type: SourceType
    source_id: str
    title: str
    text: str
    format: SourceFormat
    service_id: str | None
    timestamp: datetime
    access_level: AccessLevel
    version: str | None
    doc_type: str | None
    file_path: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


class ChunkRecord(BaseModel):
    """One row of ``document_chunks`` (stage 5 output). Field names match columns."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    document_id: str
    source_type: SourceType
    chunk_index: int = Field(ge=0)
    title: str
    section: str | None
    content: str = Field(min_length=1)
    service_id: str | None
    timestamp: AwareDatetime
    access_level: AccessLevel
    version: str | None
    doc_type: str | None
    file_path: str | None
    strategy: ChunkingStrategy
    char_start: int = Field(ge=0)
    char_end: int
    token_count: int = Field(gt=0)
    source_hash: str
    record_hash: str  # sha256 of every other field; see ``create``
    metadata: dict[str, Any]
    source_document_id: str | None = None
    source_incident_id: str | None = None
    source_deployment_id: str | None = None
    source_code_file_id: str | None = None
    source_pull_request_id: str | None = None

    @classmethod
    def create(cls, **fields: Any) -> ChunkRecord:
        """Build a record with ``record_hash`` computed from all other fields."""
        draft = cls(record_hash="", **fields)
        digest = hashlib.sha256(draft.model_dump_json(exclude={"record_hash"}).encode()).hexdigest()
        return draft.model_copy(update={"record_hash": digest})
