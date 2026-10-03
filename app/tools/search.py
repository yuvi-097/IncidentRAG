"""Text-search tools over the chunk store: documents, code, incidents.

Each tool builds the ``ChunkFilter`` itself from its typed input *and* the caller's
grants (``Principal.chunk_access``), so the database only returns chunks the caller may
read. The input has no field that could widen access.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator
from sqlalchemy import select

from app.database.models import Incident
from app.rag.retrieval.base import RetrievedChunk
from app.rag.store import ChunkFilter
from app.schemas.enums import (
    AccessLevel,
    DocumentType,
    IncidentCategory,
    IncidentStatus,
    Resource,
    Severity,
    SourceType,
    ToolPermission,
)
from app.tools.base import Evidence, Tool, ToolContext, ToolModel, truncate
from app.tools.common import (
    AppliedFilters,
    IncidentId,
    QueryText,
    ServiceId,
    TimeWindow,
    access_summary,
    check_services,
    chunk_filter_for,
    evidence,
    visible_documents,
    visible_pull_requests,
)

TopK = Annotated[int, Field(ge=1, le=100)]


def _search(
    context: ToolContext, query: str, top_k: int, chunk_filter: ChunkFilter
) -> list[RetrievedChunk]:
    top_k = min(top_k, context.settings.max_top_k)
    return context.require_retriever().search(query, top_k, chunk_filter)


def _access_filter(context: ToolContext, **fields: object) -> ChunkFilter:
    return chunk_filter_for(context.principal, **fields)


# --- search_documents ---------------------------------------------------------------


class SearchDocumentsInput(TimeWindow):
    query: QueryText
    top_k: TopK = 5
    services: list[ServiceId] | None = Field(default=None, max_length=20)
    doc_types: list[DocumentType] | None = Field(default=None, max_length=10)
    include_postmortems: bool = False


class SearchResultsOutput(ToolModel):
    results: list[Evidence]
    filters: AppliedFilters


class SearchDocumentsTool(Tool[SearchDocumentsInput, SearchResultsOutput]):
    name = "search_documents"
    description = (
        "Search technical documentation and runbooks (architecture, APIs, configuration, "
        "monitoring, troubleshooting, procedures). Returns ranked passages with provenance."
    )
    permission = ToolPermission.DOCUMENTS_READ
    input_model = SearchDocumentsInput
    output_model = SearchResultsOutput

    def _run(self, arguments: SearchDocumentsInput, context: ToolContext) -> SearchResultsOutput:
        with context.engine.connect() as connection:
            check_services(connection, arguments.services)
        source_types = {SourceType.DOCUMENTATION}
        if context.principal.can(ToolPermission.RUNBOOKS_READ):
            source_types.add(SourceType.RUNBOOK)
        if arguments.include_postmortems:
            source_types.add(SourceType.POSTMORTEM)
        chunk_filter = _access_filter(
            context,
            source_types=frozenset(source_types),
            services=frozenset(arguments.services) if arguments.services else None,
            doc_types=frozenset(d.value for d in arguments.doc_types)
            if arguments.doc_types
            else None,
            since=arguments.since,
            until=arguments.until,
        )
        chunks = _search(context, arguments.query, arguments.top_k, chunk_filter)
        return SearchResultsOutput(
            results=evidence(chunks, context.settings.snippet_chars),
            filters=AppliedFilters(
                source_types=sorted(source_types),
                services=arguments.services or [],
                since=arguments.since,
                until=arguments.until,
                access=access_summary(
                    context.principal, Resource.DOCUMENTS, Resource.RUNBOOKS, Resource.INCIDENTS
                ),
                other={"doc_types": [d.value for d in arguments.doc_types or []]},
            ),
        )


# --- search_code ----------------------------------------------------------------------


class SearchCodeInput(TimeWindow):
    query: QueryText
    top_k: TopK = 5
    services: list[ServiceId] | None = Field(default=None, max_length=20)
    include_pull_requests: bool = Field(
        default=False, description="Also search pull-request descriptions and diffs."
    )


class SearchCodeTool(Tool[SearchCodeInput, SearchResultsOutput]):
    name = "search_code"
    description = (
        "Search the source code (functions, classes, config files) and optionally pull "
        "requests. Identifiers such as class names, function names and config keys match "
        "exactly. Returns passages with file path and symbol."
    )
    permission = ToolPermission.CODE_READ
    input_model = SearchCodeInput
    output_model = SearchResultsOutput

    def _run(self, arguments: SearchCodeInput, context: ToolContext) -> SearchResultsOutput:
        with context.engine.connect() as connection:
            check_services(connection, arguments.services)
        source_types = {SourceType.CODE}
        if arguments.include_pull_requests:
            source_types.add(SourceType.PULL_REQUEST)
        chunk_filter = _access_filter(
            context,
            source_types=frozenset(source_types),
            services=frozenset(arguments.services) if arguments.services else None,
            since=arguments.since,
            until=arguments.until,
        )
        chunks = _search(context, arguments.query, arguments.top_k, chunk_filter)
        return SearchResultsOutput(
            results=evidence(chunks, context.settings.snippet_chars),
            filters=AppliedFilters(
                source_types=sorted(source_types),
                services=arguments.services or [],
                since=arguments.since,
                until=arguments.until,
                access=access_summary(context.principal, Resource.CODE),
            ),
        )


# --- search_incidents -----------------------------------------------------------------


class SearchIncidentsInput(TimeWindow):
    query: QueryText | None = None
    incident_ids: list[IncidentId] | None = Field(default=None, max_length=20)
    services: list[ServiceId] | None = Field(default=None, max_length=20)
    severities: list[Severity] | None = None
    categories: list[IncidentCategory] | None = None
    top_k: TopK = 5
    include_postmortems: bool = True
    order: Literal["newest", "oldest"] = Field(
        default="newest", description="By start time, when no text query ranks them."
    )
    overlap: bool = Field(
        default=False,
        description="Match incidents open at any point in [since, until) instead of those "
        "that started in it.",
    )

    @model_validator(mode="after")
    def _something_to_search(self) -> SearchIncidentsInput:
        criteria = (
            self.query,
            self.incident_ids,
            self.services,
            self.severities,
            self.categories,
            self.since,
            self.until,
        )
        if not any(criteria):
            raise ValueError("give a query, incident_ids or at least one filter")
        return self


class IncidentSummary(ToolModel):
    id: str
    title: str
    service_id: str
    root_cause_service_id: str | None
    severity: Severity
    status: IncidentStatus
    category: IncidentCategory
    started_at: datetime
    resolved_at: datetime
    resolution_time_minutes: int
    affected_version: str
    deployment_id: str
    root_cause_deployment_id: str | None
    root_cause_pr_id: str | None  # hidden when the caller may not read the PR
    remediation_deployment_id: str | None
    parent_incident_id: str | None
    runbook_id: str | None  # hidden when the caller may not read it
    postmortem_id: str | None  # hidden when the caller may not read it
    alert_name: str | None
    symptoms: str
    root_cause: str
    resolution: str
    access_level: AccessLevel


class SearchIncidentsOutput(ToolModel):
    incidents: list[IncidentSummary]
    evidence: list[Evidence]
    not_found: list[str]  # requested ids that do not exist or are not accessible
    filters: AppliedFilters


class SearchIncidentsTool(Tool[SearchIncidentsInput, SearchIncidentsOutput]):
    name = "search_incidents"
    description = (
        "Find past incidents and postmortems: by id (INC-0406), by text (symptoms, error "
        "messages, causes), or by filters (service, severity, category, time window). "
        "Returns structured incident summaries plus the matching passages."
    )
    permission = ToolPermission.INCIDENTS_READ
    input_model = SearchIncidentsInput
    output_model = SearchIncidentsOutput

    def _run(self, arguments: SearchIncidentsInput, context: ToolContext) -> SearchIncidentsOutput:
        principal = context.principal
        max_chars = context.settings.snippet_chars
        top_k = min(arguments.top_k, context.settings.max_top_k)
        source_types = {SourceType.INCIDENT}
        if arguments.include_postmortems:
            source_types.add(SourceType.POSTMORTEM)
        conditions = [Incident.access_level.in_(principal.visible_levels(Resource.INCIDENTS))]
        if arguments.services:
            conditions.append(Incident.service_id.in_(arguments.services))
        if arguments.severities:
            conditions.append(Incident.severity.in_(arguments.severities))
        if arguments.categories:
            conditions.append(Incident.category.in_(arguments.categories))
        if arguments.overlap:  # open at some point in the window
            if arguments.since:
                conditions.append(Incident.resolved_at >= arguments.since)
            if arguments.until:
                conditions.append(Incident.started_at < arguments.until)
        else:
            if arguments.since:
                conditions.append(Incident.started_at >= arguments.since)
            if arguments.until:
                conditions.append(Incident.started_at < arguments.until)

        chunks: list[RetrievedChunk] = []
        ordered_ids: list[str] = []
        with context.engine.connect() as connection:
            check_services(connection, arguments.services)
            if arguments.incident_ids:
                ordered_ids = list(dict.fromkeys(arguments.incident_ids))
            elif arguments.query:
                chunk_filter = _access_filter(
                    context,
                    source_types=frozenset(source_types),
                    services=frozenset(arguments.services) if arguments.services else None,
                    since=arguments.since,
                    until=arguments.until,
                )
                chunks = _search(context, arguments.query, max(top_k * 3, 10), chunk_filter)
                postmortems = {
                    c.document_id for c in chunks if c.source_type is SourceType.POSTMORTEM
                }
                by_postmortem = dict(
                    connection.execute(
                        select(Incident.postmortem_id, Incident.id).where(
                            Incident.postmortem_id.in_(sorted(postmortems))
                        )
                    ).all()
                )
                for chunk in chunks:
                    incident = by_postmortem.get(chunk.document_id, chunk.document_id)
                    if incident.startswith("INC-") and incident not in ordered_ids:
                        ordered_ids.append(incident)
            query = select(Incident.__table__).where(*conditions)
            if ordered_ids:
                query = query.where(Incident.id.in_(ordered_ids))
            else:
                ordering = (
                    Incident.started_at.asc()
                    if arguments.order == "oldest"
                    else Incident.started_at.desc()
                )
                query = query.order_by(ordering, Incident.id).limit(top_k)
            rows = {row.id: row for row in connection.execute(query)}
            order = ordered_ids or list(rows)
            found = [rows[i] for i in order if i in rows][:top_k]
            docs = visible_documents(
                connection,
                {d for r in found for d in (r.runbook_id, r.postmortem_id) if d},
                principal,
            )
            prs = visible_pull_requests(
                connection, {r.root_cause_pr_id for r in found if r.root_cause_pr_id}, principal
            )
        kept = {r.id for r in found}
        postmortem_of = {r.postmortem_id: r.id for r in found if r.postmortem_id}
        return SearchIncidentsOutput(
            incidents=[_summary(r, docs, prs, max_chars) for r in found],
            evidence=evidence(
                [c for c in chunks if postmortem_of.get(c.document_id, c.document_id) in kept],
                max_chars,
            ),
            not_found=[i for i in arguments.incident_ids or [] if i not in rows],
            filters=AppliedFilters(
                source_types=sorted(source_types),
                services=arguments.services or [],
                since=arguments.since,
                until=arguments.until,
                access=access_summary(principal, Resource.INCIDENTS),
                other={
                    "severities": [s.value for s in arguments.severities or []],
                    "categories": [c.value for c in arguments.categories or []],
                },
            ),
        )


def _summary(row: Any, docs: set[str], prs: set[str], max_chars: int) -> IncidentSummary:
    return IncidentSummary(
        id=row.id,
        title=row.title,
        service_id=row.service_id,
        root_cause_service_id=row.root_cause_service_id,
        severity=row.severity,
        status=row.status,
        category=row.category,
        started_at=row.started_at,
        resolved_at=row.resolved_at,
        resolution_time_minutes=row.resolution_time_minutes,
        affected_version=row.affected_version,
        deployment_id=row.deployment_id,
        root_cause_deployment_id=row.root_cause_deployment_id,
        root_cause_pr_id=row.root_cause_pr_id if row.root_cause_pr_id in prs else None,
        remediation_deployment_id=row.remediation_deployment_id,
        parent_incident_id=row.parent_incident_id,
        runbook_id=row.runbook_id if row.runbook_id in docs else None,
        postmortem_id=row.postmortem_id if row.postmortem_id in docs else None,
        alert_name=row.alert_name,
        symptoms=truncate(row.symptoms, max_chars),
        root_cause=truncate(row.root_cause, max_chars),
        resolution=truncate(row.resolution, max_chars),
        access_level=row.access_level,
    )


__all__ = [
    "IncidentSummary",
    "SearchCodeInput",
    "SearchCodeTool",
    "SearchDocumentsInput",
    "SearchDocumentsTool",
    "SearchIncidentsInput",
    "SearchIncidentsOutput",
    "SearchIncidentsTool",
    "SearchResultsOutput",
]
