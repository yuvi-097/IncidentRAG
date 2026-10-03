"""Builds the complete synthetic dataset. Deterministic for a given seed."""

from __future__ import annotations

import ast
import random
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from app.schemas.enums import (
    DependencyCriticality,
    DependencyProtocol,
    DocumentType,
    FileChangeType,
    IncidentStatus,
    PullRequestState,
)
from app.synthetic.catalog import PEOPLE, REPOSITORY, ROLES, SERVICES, TOPIC_PRODUCERS
from app.synthetic.code_repo import RepoFile, build_repository
from app.synthetic.documents import (
    DocContext,
    DocumentDraft,
    plan_documents,
    postmortem_draft,
    runbook_for,
)
from app.synthetic.facts import DeploymentFact, IncidentFact, PullRequestFact
from app.synthetic.logs import LogDraft, generate_logs
from app.synthetic.narrative import (
    incident_access_level,
    needs_postmortem,
    render_deployment,
    render_incident,
    render_pull_request,
)
from app.synthetic.operations import LiveDeployments, cascades, operational_incident
from app.synthetic.records import (
    CodeFileRecord,
    DatasetManifest,
    DeploymentRecord,
    DocumentRecord,
    IncidentRecord,
    LogRecord,
    PullRequestFileRecord,
    PullRequestRecord,
    RoleRecord,
    ServiceDependencyRecord,
    ServiceRecord,
    SyntheticDataset,
    UserRecord,
)
from app.synthetic.reports import admin_policies, reliability_reports
from app.synthetic.text import sha256
from app.synthetic.timeline import WINDOW_END, WINDOW_START, hex_id, simulate

GENERATOR_VERSION = "1.0.0"
ORG_CREATED_AT = datetime(2023, 1, 9, 9, 0, tzinfo=UTC)
REPO_IMPORTED_AT = datetime(2024, 3, 4, 10, 0, tzinfo=UTC)


@dataclass(frozen=True)
class GenerationConfig:
    seed: int = 42
    fault_rate: float = 0.2  # share of regular deployments that ship a faulty change
    target_incidents: int = 540
    healthy_logs_per_service_day: int = 4


def generate_dataset(config: GenerationConfig | None = None) -> SyntheticDataset:
    config = config or GenerationConfig()
    repo = build_repository()

    # 1. Release timelines (deployments, PRs, deployment-caused incidents).
    timelines = simulate(repo, config.seed, config.fault_rate)
    deployments = sorted(
        (d for t in timelines for d in t.deployments), key=lambda d: (d.deployed_at, d.service_id)
    )
    pull_requests = sorted(
        (p for t in timelines for p in t.pull_requests), key=lambda p: (p.merged_at, p.service_id)
    )
    incidents: list[IncidentFact] = [i for t in timelines for i in t.incidents]

    # 2. Cascades, then operational incidents up to the target count.
    rng = random.Random(f"{config.seed}:incidents")
    for incident in list(incidents):
        incidents += cascades(incident, rng, force=incident.anchor is not None)
    live = LiveDeployments(deployments)
    earliest = max(live.first_time(s.id) for s in SERVICES) + timedelta(days=1)
    while len(incidents) < config.target_incidents:
        incident = operational_incident(rng, {s.id: earliest for s in SERVICES})
        incidents.append(incident)
        incidents += cascades(incident, rng)
    for incident in incidents:
        incident.deployment = live.at(incident.service_id, incident.started_at)
        if incident.deployment is None:
            raise RuntimeError(
                f"no live deployment for {incident.service_id} at {incident.started_at}"
            )
    incidents.sort(key=lambda i: (i.started_at, i.service_id, i.category.value))

    # 3. Chronological ids.
    for n, deployment in enumerate(deployments, 1):
        deployment.id = f"DEP-{n:04d}"
    for n, pr in enumerate(pull_requests, 1001):
        pr.number, pr.id = n, f"PR-{n}"
    for n, incident in enumerate(incidents, 1):
        incident.id = f"INC-{n:04d}"

    # 4. Documents are planned before incident text so incidents can cite runbooks.
    doc_rng = random.Random(f"{config.seed}:documents")
    drafts = plan_documents(doc_rng)
    for incident in incidents:
        incident.runbook_id = runbook_for(drafts, incident.service_id, incident.category).id
    runbook_titles = {
        d.id: (d.id, d.title) for d in drafts.values() if d.doc_type is DocumentType.RUNBOOK
    }

    # 5. Text that mentions ids.
    text_rng = random.Random(f"{config.seed}:text")
    for incident in incidents:
        render_incident(incident, runbook_titles, text_rng)
    for pr in pull_requests:
        render_pull_request(pr)
    by_remediation = {
        id(i.remediation): i for i in incidents if i.remediation is not None and i.parent is None
    }
    for deployment in deployments:
        render_deployment(deployment, by_remediation)

    # 6. Postmortems, then the remaining document bodies (runbooks list their incidents).
    postmortems: list[DocumentDraft] = []
    for incident in incidents:
        if needs_postmortem(incident):
            draft = postmortem_draft(incident, len(postmortems) + 1, doc_rng)
            incident.postmortem_id = draft.id
            postmortems.append(draft)
    context = DocContext(drafts)
    for incident in incidents:
        if incident.runbook_id:
            context.incidents_by_runbook[incident.runbook_id].append(incident)
    for draft in drafts.values():
        assert draft.render is not None
        draft.content = draft.render(context)

    # 7. Logs.
    logs = generate_logs(
        deployments,
        incidents,
        live,
        random.Random(f"{config.seed}:logs"),
        config.healthy_logs_per_service_day,
    )

    # 8. Operational reports and administrative policies (computed; no randomness).
    documents = [
        *drafts.values(),
        *postmortems,
        *reliability_reports(incidents, deployments),
        *admin_policies(),
    ]
    return _to_records(config, repo, deployments, pull_requests, incidents, documents, logs)


# --- record conversion -------------------------------------------------------------------------


def _symbols(content: str) -> list[dict[str, object]]:
    symbols: list[dict[str, object]] = []
    for node in ast.parse(content).body:
        if isinstance(node, ast.ClassDef):
            symbols.append({"kind": "class", "name": node.name, "line": node.lineno})
            for item in node.body:
                if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                    symbols.append(
                        {"kind": "method", "name": f"{node.name}.{item.name}", "line": item.lineno}
                    )
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            symbols.append({"kind": "function", "name": node.name, "line": node.lineno})
    return symbols


def _service_dependencies() -> list[ServiceDependencyRecord]:
    edges: dict[tuple[str, str], ServiceDependencyRecord] = {}
    for s in SERVICES:
        for d in s.http_dependencies:
            edges[(s.id, d.service)] = ServiceDependencyRecord(
                service_id=s.id,
                depends_on_id=d.service,
                protocol=DependencyProtocol.HTTP,
                criticality=d.criticality,
                description=d.purpose,
            )
        for topic in s.consumes:
            key = (s.id, TOPIC_PRODUCERS[topic])
            existing = edges.get(key)
            if existing is None:
                edges[key] = ServiceDependencyRecord(
                    service_id=s.id,
                    depends_on_id=key[1],
                    protocol=DependencyProtocol.KAFKA,
                    criticality=DependencyCriticality.SOFT,
                    description=f"consumes {topic}",
                )
            else:
                edges[key] = existing.model_copy(
                    update={"description": f"{existing.description}; consumes {topic}"}
                )
    return [edges[key] for key in sorted(edges)]


def _to_records(
    config: GenerationConfig,
    repo: dict[str, RepoFile],
    deployments: list[DeploymentFact],
    pull_requests: list[PullRequestFact],
    incidents: list[IncidentFact],
    documents: list[DocumentDraft],
    logs: list[LogDraft],
) -> SyntheticDataset:
    rng = random.Random(f"{config.seed}:records")
    roles = [
        RoleRecord(
            id=r.id,
            name=r.name,
            description=r.description,
            created_at=ORG_CREATED_AT,
            updated_at=ORG_CREATED_AT,
        )
        for r in ROLES
    ]
    users = []
    for person in PEOPLE:
        joined = ORG_CREATED_AT + timedelta(days=rng.randint(0, 900))
        users.append(
            UserRecord(
                id=person.username,
                email=f"{person.username}@novacart.example",
                full_name=person.full_name,
                team=person.team,
                role_id=person.role,
                is_active=person.active,
                created_at=joined,
                updated_at=joined,
            )
        )
    services = [
        ServiceRecord(
            id=s.id,
            display_name=s.display_name,
            description=s.description,
            owner_team=s.team,
            tier=s.tier,
            language="python",
            repository_path=s.repo_path,
            port=s.port,
            oncall_channel=s.oncall_channel,
            datastores=s.datastores,
            created_at=ORG_CREATED_AT,
            updated_at=ORG_CREATED_AT,
        )
        for s in SERVICES
    ]

    # Code files: HEAD content, with the last commit that touched each file.
    import_sha = hex_id(rng, 40)
    last_touch: dict[str, PullRequestFact] = {}
    for pr in pull_requests:
        for edit in pr.edits:
            last_touch[edit.path] = pr  # pull_requests are sorted by merge time
    file_ids = {path: f"CF-{n:04d}" for n, path in enumerate(sorted(repo), 1)}
    code_files = []
    for path, f in sorted(repo.items()):
        pr = last_touch.get(path)
        modified = pr.merged_at if pr else REPO_IMPORTED_AT
        code_files.append(
            CodeFileRecord(
                id=file_ids[path],
                repository=REPOSITORY,
                path=path,
                service_id=f.service_id,
                language=f.language,
                kind=f.kind,
                content=f.content,
                line_count=f.content.count("\n") + (0 if f.content.endswith("\n") else 1),
                content_hash=sha256(f.content),
                symbols=_symbols(f.content) if f.language == "python" else [],
                last_commit_sha=pr.merge_commit_sha if pr else import_sha,
                last_modified_at=modified,
                access_level=f.access_level,
                created_at=REPO_IMPORTED_AT,
                updated_at=modified,
            )
        )

    deployment_records = [
        DeploymentRecord(
            id=d.id,
            service_id=d.service_id,
            version=d.version,
            previous_version=d.previous_version,
            commit_sha=d.commit_sha,
            deployed_at=d.deployed_at,
            author_id=d.author,
            environment="production",
            strategy=d.strategy,
            status=d.status,
            is_rollback=d.kind == "rollback",
            rollback_of_id=d.rollback_of.id if d.rollback_of else None,
            changes=d.changes,
            duration_seconds=d.duration_seconds,
            created_at=d.deployed_at,
            updated_at=d.deployed_at + timedelta(seconds=d.duration_seconds),
        )
        for d in deployments
    ]

    pr_records, pr_files = [], []
    for pr in pull_requests:
        pr_records.append(
            PullRequestRecord(
                id=pr.id,
                number=pr.number,
                service_id=pr.service_id,
                title=pr.title,
                description=pr.description,
                author_id=pr.author,
                reviewers=pr.reviewers,
                labels=pr.labels,
                state=PullRequestState.MERGED,
                base_branch="main",
                head_branch=pr.head_branch,
                opened_at=pr.opened_at,
                merged_at=pr.merged_at,
                merge_commit_sha=pr.merge_commit_sha,
                deployment_id=pr.deployment.id if pr.deployment else None,
                created_at=pr.opened_at,
                updated_at=pr.merged_at,
            )
        )
        for edit in pr.edits:
            pr_files.append(
                PullRequestFileRecord(
                    pull_request_id=pr.id,
                    code_file_id=file_ids[edit.path],
                    change_type=FileChangeType.MODIFIED,
                    additions=edit.additions,
                    deletions=edit.deletions,
                    patch=edit.patch,
                )
            )

    document_records = [
        DocumentRecord(
            id=d.id,
            doc_type=d.doc_type,
            title=d.title,
            service_id=d.service_id,
            content=d.content,
            source_path=d.source_path,
            content_hash=sha256(d.content),
            tags=d.tags,
            access_level=d.access_level,
            author_id=d.author,
            revision=d.revision,
            created_at=d.created_at,
            updated_at=d.updated_at,
        )
        for d in sorted(documents, key=lambda d: d.id)
    ]

    incident_records = []
    for i in incidents:
        assert i.deployment is not None
        incident_records.append(
            IncidentRecord(
                id=i.id,
                title=i.title,
                service_id=i.service_id,
                root_cause_service_id=i.root_cause_service_id,
                parent_incident_id=i.parent.id if i.parent else None,
                category=i.category,
                severity=i.severity,
                status=IncidentStatus.RESOLVED,
                started_at=i.started_at,
                detected_at=i.detected_at,
                resolved_at=i.resolved_at,
                resolution_time_minutes=i.duration_minutes,
                affected_version=i.deployment.version,
                deployment_id=i.deployment.id,
                root_cause_deployment_id=i.root_cause_deployment.id
                if i.root_cause_deployment
                else None,
                root_cause_pr_id=i.root_cause_pr.id if i.root_cause_pr else None,
                remediation_deployment_id=i.remediation.id
                if i.remediation and i.parent is None
                else None,
                runbook_id=i.runbook_id,
                postmortem_id=i.postmortem_id,
                commander_id=i.commander,
                alert_name=i.alert_name,
                symptoms=i.symptoms,
                root_cause=i.root_cause,
                resolution=i.resolution,
                metrics=i.metrics,
                tags=i.tags,
                access_level=incident_access_level(i),
                created_at=i.detected_at,
                updated_at=i.resolved_at,
            )
        )

    log_records = [
        LogRecord(
            timestamp=line.timestamp,
            service_id=line.service_id,
            level=line.level,
            logger=line.logger,
            message=line.message,
            trace_id=line.trace_id,
            span_id=line.span_id,
            deployment_id=line.deployment.id if line.deployment else None,
            version=line.deployment.version if line.deployment else None,
            host=line.host,
            attributes=line.attributes,
        )
        for line in logs
    ]

    tables = {
        "roles": roles,
        "users": users,
        "services": services,
        "service_dependencies": _service_dependencies(),
        "code_files": code_files,
        "deployments": deployment_records,
        "pull_requests": pr_records,
        "pull_request_files": pr_files,
        "documents": document_records,
        "incidents": incident_records,
        "logs": log_records,
    }
    manifest = DatasetManifest(
        generator_version=GENERATOR_VERSION,
        seed=config.seed,
        window_start=WINDOW_START,
        window_end=WINDOW_END,
        counts={name: len(rows) for name, rows in tables.items()},
        anchors={i.anchor: i.id for i in incidents if i.anchor and i.parent is None},
    )
    return SyntheticDataset(manifest=manifest, **tables)
