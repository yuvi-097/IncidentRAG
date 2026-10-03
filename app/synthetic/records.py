"""Pydantic records: the dataset's serialisation format and the boundary between
the generator and the database. Field names match table columns exactly."""

from __future__ import annotations

from typing import Any

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from app.schemas.enums import (
    AccessLevel,
    CodeFileKind,
    DependencyCriticality,
    DependencyProtocol,
    DeploymentStatus,
    DeploymentStrategy,
    DocumentType,
    FileChangeType,
    IncidentCategory,
    IncidentStatus,
    LogLevel,
    PullRequestState,
    ServiceTier,
    Severity,
)


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Timestamped(Record):
    created_at: AwareDatetime
    updated_at: AwareDatetime


class RoleRecord(Timestamped):
    id: str
    name: str
    description: str


class UserRecord(Timestamped):
    id: str
    email: str
    full_name: str
    team: str
    role_id: str
    is_active: bool


class ServiceRecord(Timestamped):
    id: str
    display_name: str
    description: str
    owner_team: str
    tier: ServiceTier
    language: str
    repository_path: str
    port: int
    oncall_channel: str
    datastores: list[str]


class ServiceDependencyRecord(Record):
    service_id: str
    depends_on_id: str
    protocol: DependencyProtocol
    criticality: DependencyCriticality
    description: str


class CodeFileRecord(Timestamped):
    id: str
    repository: str
    path: str
    service_id: str | None
    language: str
    kind: CodeFileKind
    content: str
    line_count: int
    content_hash: str
    symbols: list[dict[str, Any]]
    last_commit_sha: str
    last_modified_at: AwareDatetime
    access_level: AccessLevel


class DeploymentRecord(Timestamped):
    id: str
    service_id: str
    version: str
    previous_version: str | None
    commit_sha: str
    deployed_at: AwareDatetime
    author_id: str
    environment: str
    strategy: DeploymentStrategy
    status: DeploymentStatus
    is_rollback: bool
    rollback_of_id: str | None
    changes: str
    duration_seconds: int = Field(ge=0)


class PullRequestRecord(Timestamped):
    id: str
    number: int
    service_id: str
    title: str
    description: str
    author_id: str
    reviewers: list[str]
    labels: list[str]
    state: PullRequestState
    base_branch: str
    head_branch: str
    opened_at: AwareDatetime
    merged_at: AwareDatetime | None
    merge_commit_sha: str | None
    deployment_id: str | None


class PullRequestFileRecord(Record):
    pull_request_id: str
    code_file_id: str
    change_type: FileChangeType
    additions: int = Field(ge=0)
    deletions: int = Field(ge=0)
    patch: str


class DocumentRecord(Timestamped):
    id: str
    doc_type: DocumentType
    title: str
    service_id: str | None
    content: str
    source_path: str
    content_hash: str
    tags: list[str]
    access_level: AccessLevel
    author_id: str | None
    revision: int


class IncidentRecord(Timestamped):
    id: str
    title: str
    service_id: str
    root_cause_service_id: str | None
    parent_incident_id: str | None
    category: IncidentCategory
    severity: Severity
    status: IncidentStatus
    started_at: AwareDatetime
    detected_at: AwareDatetime
    resolved_at: AwareDatetime
    resolution_time_minutes: int = Field(ge=0)
    affected_version: str
    deployment_id: str
    root_cause_deployment_id: str | None
    root_cause_pr_id: str | None
    remediation_deployment_id: str | None
    runbook_id: str | None
    postmortem_id: str | None
    commander_id: str
    alert_name: str | None
    symptoms: str
    root_cause: str
    resolution: str
    metrics: dict[str, Any]
    tags: list[str]
    access_level: AccessLevel


class LogRecord(Record):
    """``id`` is assigned by the database in insertion (timestamp) order."""

    timestamp: AwareDatetime
    service_id: str
    level: LogLevel
    logger: str
    message: str
    trace_id: str | None
    span_id: str | None
    deployment_id: str | None
    version: str | None
    host: str
    attributes: dict[str, Any]


# Table name -> record type, in foreign-key-safe insertion order.
TABLES: dict[str, type[Record]] = {
    "roles": RoleRecord,
    "users": UserRecord,
    "services": ServiceRecord,
    "service_dependencies": ServiceDependencyRecord,
    "code_files": CodeFileRecord,
    "deployments": DeploymentRecord,
    "pull_requests": PullRequestRecord,
    "pull_request_files": PullRequestFileRecord,
    "documents": DocumentRecord,
    "incidents": IncidentRecord,
    "logs": LogRecord,
}


class DatasetManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    generator_version: str
    seed: int
    window_start: AwareDatetime
    window_end: AwareDatetime
    counts: dict[str, int]
    files: dict[str, str] = Field(default_factory=dict)  # file name -> sha256
    anchors: dict[str, str] = Field(default_factory=dict)  # anchor name -> incident id


class SyntheticDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manifest: DatasetManifest
    roles: list[RoleRecord]
    users: list[UserRecord]
    services: list[ServiceRecord]
    service_dependencies: list[ServiceDependencyRecord]
    code_files: list[CodeFileRecord]
    deployments: list[DeploymentRecord]
    pull_requests: list[PullRequestRecord]
    pull_request_files: list[PullRequestFileRecord]
    documents: list[DocumentRecord]
    incidents: list[IncidentRecord]
    logs: list[LogRecord]

    def table(self, name: str) -> list[Record]:
        return getattr(self, name)
