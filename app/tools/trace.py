"""trace_change: follow an incident to the code change behind it, one hop at a time.

    incident -> root-cause deployment -> commit (pull request) -> files -> changed lines
    incident -> remediation deployment -> commit (pull request) -> files -> changed lines

Or from a deployment: its commit, pull requests, files, and the incidents it caused.
Every hop is read from the linked records, not searched for, and every hop checks the
caller's grants on its own:

- the incident must be readable, or the tool answers as if it did not exist;
- a deployment needs the deployments grant; a pull request is shown only when every
  file it changes is readable (code or deployments grant); a file and its diff need the
  code grant for the file's label;
- a hop the caller may not read is not silently dropped: it is named in ``withheld``.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from pydantic import Field, model_validator
from sqlalchemy import select
from sqlalchemy.engine import Connection

from app.database.models import CodeFile, Deployment, Incident, PullRequest, PullRequestFile
from app.schemas.enums import AccessLevel, Resource, ToolPermission
from app.security.principal import Principal
from app.tools.base import ResourceNotFoundError, Tool, ToolContext, ToolModel, truncate
from app.tools.common import DeploymentId, IncidentId, visible_pull_requests

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]{3,}")


class TraceChangeInput(ToolModel):
    incident_id: IncidentId | None = None
    deployment_id: DeploymentId | None = None
    max_lines: int = Field(default=12, ge=1, le=60, description="Changed lines per file.")

    @model_validator(mode="after")
    def _one_anchor(self) -> TraceChangeInput:
        if (self.incident_id is None) == (self.deployment_id is None):
            raise ValueError("give exactly one of incident_id or deployment_id")
        return self


class TracedIncident(ToolModel):
    id: str
    title: str
    service_id: str
    category: str
    started_at: datetime
    root_cause: str
    parent_incident_id: str | None
    access_level: AccessLevel


class TracedDeployment(ToolModel):
    id: str
    service_id: str
    version: str
    status: str
    deployed_at: datetime
    commit_sha: str
    is_rollback: bool


class ChangedFile(ToolModel):
    code_file_id: str
    path: str
    kind: str
    change_type: str
    additions: int
    deletions: int
    changed_lines: list[str]  # "-old" / "+new", as in the diff
    matched_terms: list[str]  # words shared with the incident's description
    access_level: AccessLevel


class TracedChange(ToolModel):
    pull_request_id: str
    title: str
    author: str
    merge_commit_sha: str | None
    merged_at: datetime | None
    files: list[ChangedFile]


class TraceChangeOutput(ToolModel):
    incident: TracedIncident | None
    parent: TracedIncident | None  # the upstream incident of a downstream (cascade) one
    deployment: TracedDeployment | None
    changes: list[TracedChange]
    caused_incidents: list[TracedIncident]  # incidents whose recorded root cause it is
    remediation: TracedDeployment | None = None  # the deployment recorded as the fix
    remediation_changes: list[TracedChange] = Field(default_factory=list)
    withheld: list[str]  # hops the caller may not read
    note: str | None = None


def changed_lines(patch: str | None, limit: int) -> list[str]:
    lines = []
    for line in (patch or "").splitlines():
        if line.startswith(("+++", "---")) or not line.startswith(("+", "-")):
            continue
        if line[1:].strip():
            lines.append(f"{line[0]} {line[1:].strip()}")
    return lines[:limit]


class _Walker:
    """Reads the hops of one trace, checking the caller's grants at each."""

    def __init__(self, connection: Connection, principal: Principal, max_lines: int) -> None:
        self.connection = connection
        self.principal = principal
        self.max_lines = max_lines
        self.withheld: list[str] = []
        self.incident_labels = principal.visible_levels(Resource.INCIDENTS)
        self.code_labels = set(principal.visible_levels(Resource.CODE))

    def incident(self, incident_id: str | None) -> Any:
        if not incident_id:
            return None
        return self.connection.execute(
            select(Incident.__table__).where(
                Incident.id == incident_id, Incident.access_level.in_(self.incident_labels)
            )
        ).first()

    def deployment(self, deployment_id: str | None, what: str) -> Any:
        if not deployment_id:
            return None
        if not self.principal.may_read_unlabelled(Resource.DEPLOYMENTS):
            self.withheld.append(f"{what} {deployment_id} (no deployments grant)")
            return None
        return self.connection.execute(
            select(Deployment.__table__).where(Deployment.id == deployment_id)
        ).first()

    def changes(self, pull_request_ids: list[str], words: set[str]) -> list[TracedChange]:
        visible = visible_pull_requests(self.connection, pull_request_ids, self.principal)
        for hidden in [p for p in pull_request_ids if p not in visible]:
            self.withheld.append(f"pull request {hidden} (changes files your role may not read)")
        changes = []
        for pr in self.connection.execute(
            select(PullRequest.__table__)
            .where(PullRequest.id.in_(sorted(visible)))
            .order_by(PullRequest.id)
        ):
            files = []
            rows = self.connection.execute(
                select(
                    CodeFile.id,
                    CodeFile.path,
                    CodeFile.kind,
                    CodeFile.access_level,
                    PullRequestFile.change_type,
                    PullRequestFile.additions,
                    PullRequestFile.deletions,
                    PullRequestFile.patch,
                )
                .join(CodeFile, CodeFile.id == PullRequestFile.code_file_id)
                .where(PullRequestFile.pull_request_id == pr.id)
                .order_by(CodeFile.path)
            )
            for f in rows:
                if f.access_level not in self.code_labels:
                    self.withheld.append(f"diff of {f.path} (no code grant for it)")
                    continue
                lines = changed_lines(f.patch, self.max_lines)
                matched = sorted(words & set(_WORD.findall(" ".join(lines).lower())))
                files.append(
                    ChangedFile(
                        code_file_id=f.id,
                        path=f.path,
                        kind=str(f.kind),
                        change_type=str(f.change_type),
                        additions=f.additions,
                        deletions=f.deletions,
                        changed_lines=lines,
                        matched_terms=matched[:10],
                        access_level=f.access_level,
                    )
                )
            files.sort(key=lambda f: -len(f.matched_terms))  # the most relevant first
            changes.append(
                TracedChange(
                    pull_request_id=pr.id,
                    title=pr.title,
                    author=pr.author_id,
                    merge_commit_sha=pr.merge_commit_sha,
                    merged_at=pr.merged_at,
                    files=files,
                )
            )
        return changes

    def shipped_by(self, deployment_id: str) -> list[str]:
        return list(
            self.connection.execute(
                select(PullRequest.id)
                .where(PullRequest.deployment_id == deployment_id)
                .order_by(PullRequest.id)
            ).scalars()
        )


class TraceChangeTool(Tool[TraceChangeInput, TraceChangeOutput]):
    name = "trace_change"
    description = (
        "Follow an incident to the change behind it: its root-cause deployment, the commit "
        "and pull request that deployment shipped, the files the pull request changed and "
        "the changed lines, and the deployment recorded as its fix. For a downstream "
        "incident, also its upstream incident. From a deployment: its changes and the "
        "incidents it caused. Hops the caller may not read are listed as withheld."
    )
    permission = ToolPermission.INCIDENTS_READ
    input_model = TraceChangeInput
    output_model = TraceChangeOutput

    def _run(self, arguments: TraceChangeInput, context: ToolContext) -> TraceChangeOutput:
        max_chars = context.settings.snippet_chars
        with context.engine.connect() as connection:
            walk = _Walker(connection, context.principal, arguments.max_lines)
            traced = parent = remediation = None
            note = None
            words: set[str] = set()
            changes: list[TracedChange] = []
            remediation_changes: list[TracedChange] = []
            caused: list[Any] = []
            if arguments.incident_id:
                traced = walk.incident(arguments.incident_id)
                if traced is None:  # missing and not permitted look the same
                    raise ResourceNotFoundError(f"{arguments.incident_id} was not found")
                if traced.parent_incident_id:
                    parent = walk.incident(traced.parent_incident_id)
                    if parent is None:
                        walk.withheld.append(f"upstream incident {traced.parent_incident_id}")
                words = set(_WORD.findall(f"{traced.root_cause} {traced.symptoms}".lower()))
                deployment = walk.deployment(traced.root_cause_deployment_id, "deployment")
                if traced.root_cause_deployment_id is None:
                    note = (
                        f"no deployment is recorded as the cause of {traced.id}; the deployment "
                        f"live at the time was {traced.deployment_id}"
                    )
                if traced.root_cause_pr_id:
                    changes = walk.changes([traced.root_cause_pr_id], words)
                fix_id = traced.remediation_deployment_id
                remediation = walk.deployment(fix_id, "remediation deployment")
                if remediation is not None:
                    remediation_changes = walk.changes(walk.shipped_by(remediation.id), words)
            else:
                deployment = walk.deployment(arguments.deployment_id, "deployment")
                if deployment is None and not walk.withheld:
                    raise ResourceNotFoundError(f"{arguments.deployment_id} was not found")
                if deployment is not None:
                    changes = walk.changes(walk.shipped_by(deployment.id), words)
                    caused = list(
                        connection.execute(
                            select(Incident.__table__)
                            .where(
                                Incident.root_cause_deployment_id == deployment.id,
                                Incident.access_level.in_(walk.incident_labels),
                            )
                            .order_by(Incident.started_at, Incident.id)
                        )
                    )
        return TraceChangeOutput(
            incident=_incident(traced, max_chars) if traced else None,
            parent=_incident(parent, max_chars) if parent else None,
            deployment=_deployment(deployment) if deployment else None,
            changes=changes,
            caused_incidents=[_incident(r, max_chars) for r in caused],
            remediation=_deployment(remediation) if remediation else None,
            remediation_changes=remediation_changes,
            withheld=walk.withheld,
            note=note,
        )


def _incident(row: Any, max_chars: int) -> TracedIncident:
    return TracedIncident(
        id=row.id,
        title=row.title,
        service_id=row.service_id,
        category=str(row.category),
        started_at=row.started_at,
        root_cause=truncate(row.root_cause, max_chars),
        parent_incident_id=row.parent_incident_id,
        access_level=row.access_level,
    )


def _deployment(row: Any) -> TracedDeployment:
    return TracedDeployment(
        id=row.id,
        service_id=row.service_id,
        version=row.version,
        status=str(row.status),
        deployed_at=row.deployed_at,
        commit_sha=row.commit_sha,
        is_rollback=row.is_rollback,
    )


__all__ = ["TraceChangeInput", "TraceChangeOutput", "TraceChangeTool", "changed_lines"]
