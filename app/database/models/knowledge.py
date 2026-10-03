"""Documents (runbooks, technical docs, postmortems) and retrieval chunks of all sources."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    CheckConstraint,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, TimestampMixin, enum_column
from app.database.models.identity import User
from app.database.models.services import Service
from app.schemas.enums import AccessLevel, ChunkingStrategy, DocumentType, SourceType


class Document(TimestampMixin, Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(
        String(16), primary_key=True
    )  # "RB-0001", "DOC-0001", "PM-0001"
    doc_type: Mapped[DocumentType] = mapped_column(
        enum_column(DocumentType, "doc_type"), index=True
    )
    title: Mapped[str] = mapped_column(String(256))
    # NULL for platform-wide documents.
    service_id: Mapped[str | None] = mapped_column(ForeignKey("services.id"), index=True)
    content: Mapped[str] = mapped_column(Text)  # markdown
    source_path: Mapped[str] = mapped_column(String(512), unique=True)  # provenance
    content_hash: Mapped[str] = mapped_column(String(64))
    tags: Mapped[list[str]]
    access_level: Mapped[AccessLevel] = mapped_column(
        enum_column(AccessLevel, "access_level"), index=True
    )
    author_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"))
    revision: Mapped[int] = mapped_column(Integer)

    service: Mapped[Service | None] = relationship()
    author: Mapped[User | None] = relationship()
    chunks: Mapped[list[DocumentChunk]] = relationship(
        foreign_keys="DocumentChunk.source_document_id",
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="DocumentChunk.chunk_index",
    )


# Typed provenance columns on document_chunks: exactly one is set, and it equals
# document_id. Deleting a source record deletes its chunks.
CHUNK_SOURCE_COLUMNS = {
    "source_document_id": "documents.id",
    "source_incident_id": "incidents.id",
    "source_deployment_id": "deployments.id",
    "source_code_file_id": "code_files.id",
    "source_pull_request_id": "pull_requests.id",
}
_ONE_SOURCE = (
    " + ".join(f"(CASE WHEN {c} IS NULL THEN 0 ELSE 1 END)" for c in CHUNK_SOURCE_COLUMNS) + " = 1"
)
_SOURCE_MATCHES = f"document_id = COALESCE({', '.join(CHUNK_SOURCE_COLUMNS)})"


class DocumentChunk(Base):
    """A retrievable piece of one source record (document, incident, deployment,
    code file or pull request), written by the ingestion pipeline.

    Provenance guarantees:
    - ``source_type`` + ``document_id`` name the source record. Exactly one typed
      foreign key (``source_*_id``) is set, and the database checks that it equals ``document_id``.
    - ``content == cleaned_source_text[char_start:char_end]``, and ``source_hash`` is the
      sha256 of that cleaned text, so every chunk can be re-derived and verified.
    """

    __tablename__ = "document_chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_document_chunks_document_position"),
        CheckConstraint("char_end > char_start", name="char_span_valid"),
        CheckConstraint("token_count > 0", name="token_count_positive"),
        CheckConstraint(_ONE_SOURCE, name="exactly_one_source"),
        CheckConstraint(_SOURCE_MATCHES, name="source_matches_document_id"),
        Index("ix_document_chunks_service_source_type", "service_id", "source_type"),
    )

    id: Mapped[str] = mapped_column(String(48), primary_key=True)  # "INC-0406#000"
    document_id: Mapped[str] = mapped_column(String(16), index=True)  # source record id
    source_type: Mapped[SourceType] = mapped_column(
        enum_column(SourceType, "source_type"), index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer)
    title: Mapped[str] = mapped_column(String(512))
    section: Mapped[str | None] = mapped_column(String(512))  # heading path or code symbol
    content: Mapped[str] = mapped_column(Text)
    service_id: Mapped[str | None] = mapped_column(ForeignKey("services.id"), index=True)
    timestamp: Mapped[datetime] = mapped_column(index=True)  # when the source happened/changed
    access_level: Mapped[AccessLevel] = mapped_column(
        enum_column(AccessLevel, "access_level"), index=True
    )
    version: Mapped[str | None] = mapped_column(String(32), index=True)  # service release
    doc_type: Mapped[str | None] = mapped_column(String(32), index=True)
    file_path: Mapped[str | None] = mapped_column(String(512))
    strategy: Mapped[ChunkingStrategy] = mapped_column(enum_column(ChunkingStrategy, "strategy"))
    char_start: Mapped[int] = mapped_column(Integer)
    char_end: Mapped[int] = mapped_column(Integer)
    token_count: Mapped[int] = mapped_column(Integer)
    source_hash: Mapped[str] = mapped_column(String(64))
    # sha256 of the whole chunk record; re-ingestion leaves unchanged chunks (and
    # therefore their embeddings) untouched.
    record_hash: Mapped[str] = mapped_column(String(64))
    # Attribute name differs: "metadata" is reserved on SQLAlchemy declarative classes.
    chunk_metadata: Mapped[dict[str, Any]] = mapped_column("metadata")
    source_document_id: Mapped[str | None] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE")
    )
    source_incident_id: Mapped[str | None] = mapped_column(
        ForeignKey("incidents.id", ondelete="CASCADE")
    )
    source_deployment_id: Mapped[str | None] = mapped_column(
        ForeignKey("deployments.id", ondelete="CASCADE")
    )
    source_code_file_id: Mapped[str | None] = mapped_column(
        ForeignKey("code_files.id", ondelete="CASCADE")
    )
    source_pull_request_id: Mapped[str | None] = mapped_column(
        ForeignKey("pull_requests.id", ondelete="CASCADE")
    )
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    document: Mapped[Document | None] = relationship(
        foreign_keys=[source_document_id], back_populates="chunks"
    )
    service: Mapped[Service | None] = relationship()
    embeddings: Mapped[list[ChunkEmbedding]] = relationship(
        back_populates="chunk", cascade="all, delete-orphan", passive_deletes=True
    )


class ChunkEmbedding(Base):
    """One chunk embedded by one model.

    Keyed by (chunk, model), so several models can coexist (e.g. for comparison) and
    changing the configured model never mixes vector spaces. The column is a
    dimension-less ``vector``; each (model, dimension) gets its own partial HNSW
    index, created by the embedding pipeline (``app.rag.embeddings.index``).
    ``text_hash`` identifies the exact text that was embedded: a chunk is only
    re-embedded when that text changes.
    """

    __tablename__ = "chunk_embeddings"

    chunk_id: Mapped[str] = mapped_column(
        ForeignKey("document_chunks.id", ondelete="CASCADE"), primary_key=True
    )
    model: Mapped[str] = mapped_column(String(128), primary_key=True, index=True)
    dimension: Mapped[int] = mapped_column(Integer)
    text_hash: Mapped[str] = mapped_column(String(64))
    embedding: Mapped[list[float]] = mapped_column(Vector())
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    chunk: Mapped[DocumentChunk] = relationship(back_populates="embeddings")
