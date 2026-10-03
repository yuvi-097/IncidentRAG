"""Source code, pull requests and deployments: the change history incidents link back to."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import CheckConstraint, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, TimestampMixin, enum_column
from app.database.models.identity import User
from app.database.models.services import Service
from app.schemas.enums import (
    AccessLevel,
    CodeFileKind,
    DeploymentStatus,
    DeploymentStrategy,
    FileChangeType,
    PullRequestState,
)


class CodeFile(TimestampMixin, Base):
    """A file in the NovaCart monorepo at HEAD."""

    __tablename__ = "code_files"

    id: Mapped[str] = mapped_column(String(16), primary_key=True)  # "CF-0001"
    repository: Mapped[str] = mapped_column(String(64))
    path: Mapped[str] = mapped_column(String(512), unique=True)
    # NULL for shared libraries and repository-level files.
    service_id: Mapped[str | None] = mapped_column(ForeignKey("services.id"), index=True)
    language: Mapped[str] = mapped_column(String(32), index=True)
    kind: Mapped[CodeFileKind] = mapped_column(enum_column(CodeFileKind, "kind"))
    content: Mapped[str] = mapped_column(Text)
    line_count: Mapped[int] = mapped_column(Integer)
    content_hash: Mapped[str] = mapped_column(String(64))  # sha256 of content
    # [{"kind": "class"|"function"|"method", "name": "...", "line": 12}, ...]
    symbols: Mapped[list[dict[str, Any]]]
    last_commit_sha: Mapped[str] = mapped_column(String(40))
    last_modified_at: Mapped[datetime]
    access_level: Mapped[AccessLevel] = mapped_column(enum_column(AccessLevel, "access_level"))

    service: Mapped[Service | None] = relationship(back_populates="code_files")
    changes: Mapped[list[PullRequestFile]] = relationship(back_populates="code_file")


class Deployment(TimestampMixin, Base):
    __tablename__ = "deployments"
    __table_args__ = (
        Index("ix_deployments_service_id_deployed_at", "service_id", "deployed_at"),
        CheckConstraint("duration_seconds >= 0", name="duration_non_negative"),
    )

    id: Mapped[str] = mapped_column(String(16), primary_key=True)  # "DEP-0001"
    service_id: Mapped[str] = mapped_column(ForeignKey("services.id"))
    version: Mapped[str] = mapped_column(String(32), index=True)
    previous_version: Mapped[str | None] = mapped_column(String(32))
    commit_sha: Mapped[str] = mapped_column(String(40), index=True)
    deployed_at: Mapped[datetime]
    author_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    environment: Mapped[str] = mapped_column(String(32))
    strategy: Mapped[DeploymentStrategy] = mapped_column(
        enum_column(DeploymentStrategy, "strategy")
    )
    status: Mapped[DeploymentStatus] = mapped_column(enum_column(DeploymentStatus, "status"))
    is_rollback: Mapped[bool]
    # For rollback deployments: the deployment being reverted.
    rollback_of_id: Mapped[str | None] = mapped_column(ForeignKey("deployments.id"))
    changes: Mapped[str] = mapped_column(Text)  # human-readable changelog
    duration_seconds: Mapped[int] = mapped_column(Integer)

    service: Mapped[Service] = relationship(back_populates="deployments")
    author: Mapped[User] = relationship(back_populates="deployments")
    pull_requests: Mapped[list[PullRequest]] = relationship(back_populates="deployment")
    rollback_of: Mapped[Deployment | None] = relationship(remote_side=[id])


class PullRequest(TimestampMixin, Base):
    __tablename__ = "pull_requests"

    id: Mapped[str] = mapped_column(String(16), primary_key=True)  # "PR-1001"
    number: Mapped[int] = mapped_column(Integer, unique=True)
    service_id: Mapped[str] = mapped_column(ForeignKey("services.id"), index=True)
    title: Mapped[str] = mapped_column(String(256))
    description: Mapped[str] = mapped_column(Text)
    author_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    reviewers: Mapped[list[str]]
    labels: Mapped[list[str]]
    state: Mapped[PullRequestState] = mapped_column(enum_column(PullRequestState, "state"))
    base_branch: Mapped[str] = mapped_column(String(128))
    head_branch: Mapped[str] = mapped_column(String(128))
    opened_at: Mapped[datetime]
    merged_at: Mapped[datetime | None] = mapped_column(index=True)
    merge_commit_sha: Mapped[str | None] = mapped_column(String(40), unique=True)
    # The deployment that first shipped this PR to production.
    deployment_id: Mapped[str | None] = mapped_column(ForeignKey("deployments.id"), index=True)

    author: Mapped[User] = relationship(back_populates="pull_requests")
    deployment: Mapped[Deployment | None] = relationship(back_populates="pull_requests")
    files: Mapped[list[PullRequestFile]] = relationship(back_populates="pull_request")


class PullRequestFile(Base):
    __tablename__ = "pull_request_files"
    __table_args__ = (
        CheckConstraint("additions >= 0 AND deletions >= 0", name="line_counts_non_negative"),
    )

    pull_request_id: Mapped[str] = mapped_column(
        ForeignKey("pull_requests.id", ondelete="CASCADE"), primary_key=True
    )
    code_file_id: Mapped[str] = mapped_column(
        ForeignKey("code_files.id"), primary_key=True, index=True
    )
    change_type: Mapped[FileChangeType] = mapped_column(enum_column(FileChangeType, "change_type"))
    additions: Mapped[int] = mapped_column(Integer)
    deletions: Mapped[int] = mapped_column(Integer)
    patch: Mapped[str] = mapped_column(Text)  # unified diff

    pull_request: Mapped[PullRequest] = relationship(back_populates="files")
    code_file: Mapped[CodeFile] = relationship(back_populates="changes")
