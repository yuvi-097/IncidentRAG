"""search_deployments: releases, versions, rollbacks, the PRs they shipped and the
incidents linked to them."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field
from sqlalchemy import or_, select

from app.database.models import Deployment, Incident, PullRequest
from app.rag.retrieval.base import RetrievedChunk
from app.schemas.enums import (
    DeploymentStatus,
    DeploymentStrategy,
    Resource,
    Severity,
    SourceType,
    ToolPermission,
)
from app.tools.base import Evidence, Tool, ToolContext, ToolModel, truncate
from app.tools.common import (
    AppliedFilters,
    DeploymentId,
    QueryText,
    ServiceId,
    TimeWindow,
    Version,
    access_summary,
    check_services,
    chunk_filter_for,
    evidence,
    require_unlabelled_access,
    visible_pull_requests,
)


class SearchDeploymentsInput(TimeWindow):
    query: QueryText | None = Field(default=None, description="Free text over changelogs.")
    deployment_ids: list[DeploymentId] | None = Field(default=None, max_length=20)
    services: list[ServiceId] | None = Field(default=None, max_length=20)
    versions: list[Version] | None = Field(default=None, max_length=20)
    statuses: list[DeploymentStatus] | None = None
    rollbacks_only: bool = False
    limit: int = Field(default=10, ge=1, le=50)
    order: Literal["newest", "oldest"] = Field(
        default="newest", description="By deploy time, when no text query ranks them."
    )


class PullRequestRef(ToolModel):
    id: str
    title: str
    merged_at: datetime | None


class IncidentRef(ToolModel):
    id: str
    title: str
    severity: Severity
    started_at: datetime
    relation: str  # "during" (live on this deployment) | "root_cause" | "remediation"


class DeploymentRecord(ToolModel):
    id: str
    service_id: str
    version: str
    previous_version: str | None
    status: DeploymentStatus
    strategy: DeploymentStrategy
    deployed_at: datetime
    is_rollback: bool
    rollback_of_id: str | None
    commit_sha: str
    changes: str
    pull_requests: list[PullRequestRef]  # only those within the caller's clearance
    incidents: list[IncidentRef]  # only those within the caller's clearance


class SearchDeploymentsOutput(ToolModel):
    deployments: list[DeploymentRecord]
    evidence: list[Evidence]
    not_found: list[str]
    filters: AppliedFilters


class SearchDeploymentsTool(Tool[SearchDeploymentsInput, SearchDeploymentsOutput]):
    name = "search_deployments"
    description = (
        "Find deployments by id (DEP-0296), service, version (v2.8.1), status, time window "
        "or changelog text. Each result lists the pull requests it shipped and incidents "
        "linked to it (during, caused by, or fixed by the deployment). Newest first "
        "(or oldest first with order=oldest) unless a text query ranks them."
    )
    permission = ToolPermission.DEPLOYMENTS_READ
    input_model = SearchDeploymentsInput
    output_model = SearchDeploymentsOutput

    def _run(
        self, arguments: SearchDeploymentsInput, context: ToolContext
    ) -> SearchDeploymentsOutput:
        principal = context.principal
        require_unlabelled_access(principal, Resource.DEPLOYMENTS)
        d = Deployment
        conditions = []
        if arguments.deployment_ids:
            conditions.append(d.id.in_(arguments.deployment_ids))
        if arguments.services:
            conditions.append(d.service_id.in_(arguments.services))
        if arguments.versions:
            versions = {v if v.startswith("v") else f"v{v}" for v in arguments.versions}
            conditions.append(d.version.in_(sorted(versions)))
        if arguments.statuses:
            conditions.append(d.status.in_(arguments.statuses))
        if arguments.rollbacks_only:
            conditions.append(d.is_rollback.is_(True))
        if arguments.since:
            conditions.append(d.deployed_at >= arguments.since)
        if arguments.until:
            conditions.append(d.deployed_at < arguments.until)

        chunks: list[RetrievedChunk] = []
        with context.engine.connect() as connection:
            check_services(connection, arguments.services)
            query = select(d.__table__).where(*conditions)
            if arguments.query:
                chunk_filter = chunk_filter_for(
                    principal,
                    source_types=frozenset({SourceType.DEPLOYMENT}),
                    services=frozenset(arguments.services) if arguments.services else None,
                    since=arguments.since,
                    until=arguments.until,
                )
                limit = min(max(arguments.limit * 3, 10), context.settings.max_top_k * 3)
                chunks = context.require_retriever().search(arguments.query, limit, chunk_filter)
                ranked = list(dict.fromkeys(c.document_id for c in chunks))
                rows = {r.id: r for r in connection.execute(query.where(d.id.in_(ranked)))}
                found = [rows[i] for i in ranked if i in rows][: arguments.limit]
            else:
                ordering = (
                    (d.deployed_at.asc(), d.id.asc())
                    if arguments.order == "oldest"
                    else (d.deployed_at.desc(), d.id.desc())
                )
                query = query.order_by(*ordering).limit(arguments.limit)
                found = list(connection.execute(query))
            ids = [r.id for r in found]
            prs = list(
                connection.execute(
                    select(
                        PullRequest.id,
                        PullRequest.title,
                        PullRequest.merged_at,
                        PullRequest.deployment_id,
                    )
                    .where(PullRequest.deployment_id.in_(ids))
                    .order_by(PullRequest.id)
                ).all()
            )
            visible_prs = visible_pull_requests(connection, {p.id for p in prs}, principal)
            incidents = list(
                connection.execute(
                    select(Incident.__table__)
                    .where(
                        Incident.access_level.in_(principal.visible_levels(Resource.INCIDENTS)),
                        or_(
                            Incident.deployment_id.in_(ids),
                            Incident.root_cause_deployment_id.in_(ids),
                            Incident.remediation_deployment_id.in_(ids),
                        ),
                    )
                    .order_by(Incident.started_at)
                )
            )
        max_chars = context.settings.snippet_chars
        records = []
        for row in found:
            linked = []
            for incident in incidents:
                relations = [
                    ("root_cause", incident.root_cause_deployment_id),
                    ("remediation", incident.remediation_deployment_id),
                    ("during", incident.deployment_id),
                ]
                relation = next((name for name, dep in relations if dep == row.id), None)
                if relation:
                    linked.append(
                        IncidentRef(
                            id=incident.id,
                            title=incident.title,
                            severity=incident.severity,
                            started_at=incident.started_at,
                            relation=relation,
                        )
                    )
            records.append(
                DeploymentRecord(
                    id=row.id,
                    service_id=row.service_id,
                    version=row.version,
                    previous_version=row.previous_version,
                    status=row.status,
                    strategy=row.strategy,
                    deployed_at=row.deployed_at,
                    is_rollback=row.is_rollback,
                    rollback_of_id=row.rollback_of_id,
                    commit_sha=row.commit_sha,
                    changes=truncate(row.changes, max_chars),
                    pull_requests=[
                        PullRequestRef(id=p.id, title=p.title, merged_at=p.merged_at)
                        for p in prs
                        if p.deployment_id == row.id and p.id in visible_prs
                    ],
                    incidents=linked,
                )
            )
        kept = {r.id for r in found}
        return SearchDeploymentsOutput(
            deployments=records,
            evidence=evidence([c for c in chunks if c.document_id in kept], max_chars),
            not_found=[i for i in arguments.deployment_ids or [] if i not in kept],
            filters=AppliedFilters(
                source_types=[SourceType.DEPLOYMENT],
                services=arguments.services or [],
                since=arguments.since,
                until=arguments.until,
                access=access_summary(principal, Resource.DEPLOYMENTS, Resource.INCIDENTS),
                other={
                    "versions": arguments.versions or [],
                    "statuses": [s.value for s in arguments.statuses or []],
                    "rollbacks_only": arguments.rollbacks_only,
                },
            ),
        )
