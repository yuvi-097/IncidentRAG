"""get_runbook: the full text of one runbook, by id, exact title or search."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator
from sqlalchemy import func, select

from app.database.models import Document
from app.schemas.enums import AccessLevel, DocumentType, Resource, SourceType, ToolPermission
from app.tools.base import ResourceNotFoundError, Tool, ToolContext, ToolModel
from app.tools.common import QueryText, RunbookId, ServiceId, check_services, chunk_filter_for

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


class GetRunbookInput(ToolModel):
    runbook_id: RunbookId | None = None
    title: str | None = Field(default=None, min_length=2, max_length=256)
    query: QueryText | None = Field(default=None, description="Find the best-matching runbook.")
    service: ServiceId | None = Field(default=None, description="Restrict a query search.")

    @model_validator(mode="after")
    def _exactly_one(self) -> GetRunbookInput:
        given = [x for x in (self.runbook_id, self.title, self.query) if x]
        if len(given) != 1:
            raise ValueError("give exactly one of runbook_id, title or query")
        if self.service and not self.query:
            raise ValueError("service only applies to a query search")
        return self


class RunbookRef(ToolModel):
    id: str
    title: str
    service_id: str | None


class Runbook(ToolModel):
    id: str
    title: str
    service_id: str | None
    access_level: AccessLevel
    source_path: str
    revision: int
    updated_at: datetime
    sections: list[str]  # headings, in order
    content: str
    content_truncated: bool


class GetRunbookOutput(ToolModel):
    runbook: Runbook
    matched_by: Literal["id", "title", "search"]
    alternatives: list[RunbookRef]  # other candidates when found by search


class GetRunbookTool(Tool[GetRunbookInput, GetRunbookOutput]):
    name = "get_runbook"
    description = (
        "Fetch one complete runbook (mitigation procedure) by id (RB-0049), exact title, "
        "or the best match for a query, optionally within one service."
    )
    permission = ToolPermission.RUNBOOKS_READ
    input_model = GetRunbookInput
    output_model = GetRunbookOutput

    def _run(self, arguments: GetRunbookInput, context: ToolContext) -> GetRunbookOutput:
        principal = context.principal
        doc = Document
        visible = [
            doc.doc_type == DocumentType.RUNBOOK,
            doc.access_level.in_(principal.visible_levels(Resource.RUNBOOKS)),
        ]
        alternatives: list[str] = []
        with context.engine.connect() as connection:
            if arguments.runbook_id:
                matched_by = "id"
                query = select(doc.__table__).where(doc.id == arguments.runbook_id, *visible)
            elif arguments.title:
                matched_by = "title"
                query = select(doc.__table__).where(
                    func.lower(doc.title) == arguments.title.strip().lower(), *visible
                )
            else:
                matched_by = "search"
                check_services(connection, [arguments.service] if arguments.service else None)
                chunk_filter = chunk_filter_for(
                    principal,
                    source_types=frozenset({SourceType.RUNBOOK}),
                    services=frozenset({arguments.service}) if arguments.service else None,
                )
                assert arguments.query is not None
                chunks = context.require_retriever().search(arguments.query, 10, chunk_filter)
                ranked = list(dict.fromkeys(c.document_id for c in chunks))
                if not ranked:
                    raise ResourceNotFoundError("no accessible runbook matches the query")
                alternatives = ranked[1:4]
                query = select(doc.__table__).where(doc.id == ranked[0], *visible)
            row = connection.execute(query.order_by(doc.id)).first()
            if row is None:  # missing and not permitted look the same
                raise ResourceNotFoundError("no accessible runbook matches")
            refs = list(
                connection.execute(
                    select(doc.id, doc.title, doc.service_id).where(
                        doc.id.in_(alternatives), *visible
                    )
                ).all()
            )
        limit = context.settings.runbook_max_chars
        order = {doc_id: n for n, doc_id in enumerate(alternatives)}
        return GetRunbookOutput(
            runbook=Runbook(
                id=row.id,
                title=row.title,
                service_id=row.service_id,
                access_level=row.access_level,
                source_path=row.source_path,
                revision=row.revision,
                updated_at=row.updated_at,
                sections=[m.group(2) for m in _HEADING.finditer(row.content)],
                content=row.content[:limit],
                content_truncated=len(row.content) > limit,
            ),
            matched_by=matched_by,
            alternatives=[
                RunbookRef(id=r.id, title=r.title, service_id=r.service_id)
                for r in sorted(refs, key=lambda r: order[r.id])
            ],
        )
