"""Stage 1: read source records from the database (the system of record)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator
from typing import Any

from sqlalchemy import Connection, select

from app.database.models import (
    CodeFile,
    Deployment,
    Document,
    Incident,
    PullRequest,
    PullRequestFile,
)
from app.rag.ingestion.models import RawSource
from app.schemas.enums import DocumentType, SourceType

DOCUMENT_SOURCE_TYPES = {
    DocumentType.RUNBOOK: SourceType.RUNBOOK,
    DocumentType.POSTMORTEM: SourceType.POSTMORTEM,
}


def document_source_type(doc_type: DocumentType | str) -> SourceType:
    return DOCUMENT_SOURCE_TYPES.get(DocumentType(doc_type), SourceType.DOCUMENTATION)


def _rows(connection: Connection, model: type, order_by: Any) -> list[dict[str, Any]]:
    table = model.__table__  # type: ignore[attr-defined]
    return [dict(row) for row in connection.execute(select(table).order_by(order_by)).mappings()]


def load_sources(
    connection: Connection, source_types: Iterable[SourceType] | None = None
) -> Iterator[RawSource]:
    """Yield every source record of the requested types, in a stable order."""
    wanted = set(source_types or SourceType)

    if wanted & {SourceType.RUNBOOK, SourceType.DOCUMENTATION, SourceType.POSTMORTEM}:
        for row in _rows(connection, Document, Document.id):
            source_type = document_source_type(row["doc_type"])
            if source_type in wanted:
                yield RawSource(source_type, row["id"], row)

    if SourceType.INCIDENT in wanted:
        for row in _rows(connection, Incident, Incident.id):
            yield RawSource(SourceType.INCIDENT, row["id"], row)

    if SourceType.DEPLOYMENT in wanted:
        shipped: dict[str, list[str]] = defaultdict(list)
        for pr_id, deployment_id in connection.execute(
            select(PullRequest.id, PullRequest.deployment_id).order_by(PullRequest.number)
        ):
            if deployment_id:
                shipped[deployment_id].append(pr_id)
        for row in _rows(connection, Deployment, Deployment.id):
            yield RawSource(
                SourceType.DEPLOYMENT, row["id"], {**row, "pull_request_ids": shipped[row["id"]]}
            )

    if SourceType.CODE in wanted:
        for row in _rows(connection, CodeFile, CodeFile.path):
            yield RawSource(SourceType.CODE, row["id"], row)

    if SourceType.PULL_REQUEST in wanted:
        files: dict[str, list[dict[str, Any]]] = defaultdict(list)
        query = (
            select(
                PullRequestFile.pull_request_id,
                PullRequestFile.change_type,
                PullRequestFile.additions,
                PullRequestFile.deletions,
                PullRequestFile.patch,
                CodeFile.path,
                CodeFile.access_level,
                CodeFile.service_id,
            )
            .join(CodeFile, CodeFile.id == PullRequestFile.code_file_id)
            .order_by(PullRequestFile.pull_request_id, CodeFile.path)
        )
        for row in connection.execute(query).mappings():
            files[row["pull_request_id"]].append(dict(row))
        versions = dict(connection.execute(select(Deployment.id, Deployment.version)).all())
        for row in _rows(connection, PullRequest, PullRequest.number):
            yield RawSource(
                SourceType.PULL_REQUEST,
                row["id"],
                {
                    **row,
                    "files": files[row["id"]],
                    "deployment_version": versions.get(row["deployment_id"]),
                },
            )
