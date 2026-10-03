"""Mutable in-memory facts produced by the simulation.

Facts reference each other directly (not by id). Ids are assigned only after the
whole timeline exists, in chronological order, and text that mentions ids is
rendered afterwards, so every cross-reference in the output is real.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from app.schemas.enums import (
    DeploymentStatus,
    DeploymentStrategy,
    IncidentCategory,
    Severity,
)
from app.synthetic.changes import FaultPlan, FileEdit

Version = tuple[int, int, int]


@dataclass(eq=False)
class PullRequestFact:
    service_id: str
    title: str
    rationale: str
    author: str
    reviewers: list[str]
    labels: list[str]
    opened_at: datetime
    merged_at: datetime
    head_branch: str
    merge_commit_sha: str
    edits: tuple[FileEdit, ...]
    fault: FaultPlan | None = None
    fixes: IncidentFact | None = None
    deployment: DeploymentFact | None = None
    id: str = ""
    number: int = 0
    description: str = ""


@dataclass(eq=False)
class DeploymentFact:
    service_id: str
    kind: str  # "regular" | "rollback" | "hotfix"
    rel_version: Version
    previous_rel_version: Version | None
    commit_sha: str
    deployed_at: datetime
    author: str
    strategy: DeploymentStrategy
    status: DeploymentStatus
    duration_seconds: int
    pull_requests: list[PullRequestFact] = field(default_factory=list)
    carried_pull_requests: list[PullRequestFact] = field(default_factory=list)  # failed rollouts
    rollback_of: DeploymentFact | None = None
    id: str = ""
    version: str = ""
    previous_version: str | None = None
    changes: str = ""

    @property
    def went_live(self) -> bool:
        return self.status is not DeploymentStatus.FAILED


@dataclass(frozen=True)
class OperationalCause:
    """A non-deployment cause (traffic, infrastructure, provider, manual change)."""

    key: str
    trigger: str
    root_cause: str
    resolution: str
    symptom: str
    exception: str
    error_message: str
    endpoint: str
    status_code: int
    alert_key: str | None
    details: dict[str, Any] = field(default_factory=dict)
    upstream_service_id: str | None = None


@dataclass(eq=False)
class IncidentFact:
    service_id: str
    category: IncidentCategory
    severity: Severity
    started_at: datetime
    detected_at: datetime
    resolved_at: datetime
    commander: str
    alert_key: str | None
    metrics: dict[str, Any]
    root_cause_service_id: str | None = None
    parent: IncidentFact | None = None
    fault: FaultPlan | None = None
    cause: OperationalCause | None = None
    root_cause_deployment: DeploymentFact | None = None
    root_cause_pr: PullRequestFact | None = None
    remediation: DeploymentFact | None = None
    fix_pr: PullRequestFact | None = None
    traffic_event: str | None = None
    anchor: str | None = None
    deployment: DeploymentFact | None = None  # live deployment of service_id at started_at
    children: list[IncidentFact] = field(default_factory=list)
    # rendered after ids are assigned
    id: str = ""
    title: str = ""
    symptoms: str = ""
    root_cause: str = ""
    resolution: str = ""
    alert_name: str | None = None
    runbook_id: str | None = None
    postmortem_id: str | None = None
    tags: list[str] = field(default_factory=list)

    @property
    def is_deployment_related(self) -> bool:
        return self.root_cause_deployment is not None

    @property
    def duration_minutes(self) -> int:
        return int((self.resolved_at - self.started_at).total_seconds() // 60)
