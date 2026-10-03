"""Pieces shared by several tools: input fields, validation, visibility rules."""

from __future__ import annotations

from collections.abc import Collection, Iterable
from datetime import datetime
from typing import Annotated, Any

from pydantic import Field, StringConstraints, model_validator
from sqlalchemy import ColumnElement, and_, exists, false, or_, select
from sqlalchemy.engine import Connection

from app.database.models import CodeFile, Document, PullRequest, PullRequestFile, Service
from app.rag.retrieval.base import RetrievedChunk
from app.rag.store import ChunkFilter
from app.schemas.enums import AccessLevel, DocumentType, Resource, SourceType
from app.security.principal import Principal
from app.tools.base import Evidence, ToolInputError, ToolModel, ToolPermissionError

ServiceId = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9-]{1,63}$")]
IncidentId = Annotated[str, StringConstraints(pattern=r"^INC-\d{4}$")]
DeploymentId = Annotated[str, StringConstraints(pattern=r"^DEP-\d{4}$")]
RunbookId = Annotated[str, StringConstraints(pattern=r"^RB-\d{4}$")]
Version = Annotated[str, StringConstraints(pattern=r"^v?\d+(?:\.\d+){1,3}$")]
QueryText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=2, max_length=500)]


class TimeWindow(ToolModel):
    since: datetime | None = Field(default=None, description="Inclusive, ISO 8601.")
    until: datetime | None = Field(default=None, description="Exclusive, ISO 8601.")

    @model_validator(mode="after")
    def _ordered(self) -> Any:
        if self.since and self.until and self.since >= self.until:
            raise ValueError("since must be earlier than until")
        return self


class AppliedFilters(ToolModel):
    """What the tool actually searched, including the labels the caller may read."""

    source_types: list[SourceType] = Field(default_factory=list)
    services: list[str] = Field(default_factory=list)
    since: datetime | None = None
    until: datetime | None = None
    access: dict[Resource, list[AccessLevel]] = Field(default_factory=dict)
    other: dict[str, Any] = Field(default_factory=dict)


def access_summary(principal: Principal, *resources: Resource) -> dict[Resource, list[AccessLevel]]:
    return {r: principal.visible_levels(r) for r in resources if principal.labels(r)}


def chunk_filter_for(principal: Principal, **fields: Any) -> ChunkFilter:
    """A chunk filter restricted to what ``principal`` may read. Tools build every
    filter through this, and their inputs have no field that could widen it."""
    return ChunkFilter(access=principal.chunk_access(), **fields)


def require_unlabelled_access(principal: Principal, resource: Resource) -> None:
    """Records without their own label (deployments, logs) need a grant for the
    policy's unlabelled level."""
    if not principal.may_read_unlabelled(resource):
        raise ToolPermissionError(f"role {principal.role!r} may not read {resource.value}")


def check_services(connection: Connection, services: Collection[str] | None) -> None:
    if not services:
        return
    known = set(connection.execute(select(Service.id)).scalars())
    unknown = sorted(set(services) - known)
    if unknown:
        raise ToolInputError(
            f"unknown service(s): {', '.join(unknown)}",
            [{"field": "services", "error": f"known services: {', '.join(sorted(known))}"}],
        )


def evidence(chunks: Iterable[RetrievedChunk], max_chars: int) -> list[Evidence]:
    return [Evidence.from_chunk(chunk, max_chars) for chunk in chunks]


_OWN_RESOURCE = {
    DocumentType.RUNBOOK: Resource.RUNBOOKS,
    DocumentType.POSTMORTEM: Resource.INCIDENTS,
}


def document_access(principal: Principal) -> ColumnElement[bool]:
    """Rows of ``documents`` the caller may read: runbooks need a runbooks grant,
    postmortems an incidents grant, everything else a documents grant."""
    d = Document
    parts = [
        and_(d.doc_type == doc_type, d.access_level.in_(principal.visible_levels(resource)))
        for doc_type, resource in _OWN_RESOURCE.items()
        if principal.labels(resource)
    ]
    if principal.labels(Resource.DOCUMENTS):
        parts.append(
            and_(
                d.doc_type.not_in(list(_OWN_RESOURCE)),
                d.access_level.in_(principal.visible_levels(Resource.DOCUMENTS)),
            )
        )
    return or_(*parts) if parts else false()


def visible_documents(
    connection: Connection, ids: Collection[str], principal: Principal
) -> set[str]:
    if not ids:
        return set()
    query = select(Document.id).where(Document.id.in_(sorted(ids)), document_access(principal))
    return set(connection.execute(query).scalars())


def pull_request_levels(principal: Principal) -> list[AccessLevel]:
    """Labels of changed files under which a pull request's metadata (id, title) is
    visible: through a code grant or a deployments grant (change history)."""
    return sorted(principal.labels(Resource.CODE) | principal.labels(Resource.DEPLOYMENTS))


def visible_pull_requests(
    connection: Connection, ids: Collection[str], principal: Principal
) -> set[str]:
    """A pull request is visible when every file it changes is visible."""
    levels = pull_request_levels(principal)
    if not ids or not levels:
        return set()
    hidden_file = (
        select(PullRequestFile.pull_request_id)
        .join(CodeFile, CodeFile.id == PullRequestFile.code_file_id)
        .where(
            PullRequestFile.pull_request_id == PullRequest.id,
            CodeFile.access_level.not_in(levels),
        )
    )
    query = select(PullRequest.id).where(PullRequest.id.in_(sorted(ids)), ~exists(hidden_file))
    return set(connection.execute(query).scalars())
