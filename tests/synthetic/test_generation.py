"""The generated dataset meets the Phase 1 requirements and supports multi-hop reasoning."""

from __future__ import annotations

import hashlib
from collections import Counter

import yaml

from app.schemas.enums import CodeFileKind, DocumentType, IncidentCategory, LogLevel
from app.synthetic.generator import GenerationConfig, generate_dataset
from app.synthetic.records import TABLES, SyntheticDataset
from app.synthetic.validation import EXPECTED_SERVICES, validate_dataset


def _fingerprint(dataset: SyntheticDataset) -> dict[str, str]:
    return {
        table: hashlib.sha256(
            "".join(r.model_dump_json() + "\n" for r in dataset.table(table)).encode()
        ).hexdigest()
        for table in TABLES
    }


def test_dataset_passes_all_integrity_checks(dataset: SyntheticDataset) -> None:
    assert validate_dataset(dataset) == []


def test_required_volumes(dataset: SyntheticDataset) -> None:
    docs = Counter(d.doc_type for d in dataset.documents)
    technical = sum(
        n for t, n in docs.items() if t not in {DocumentType.RUNBOOK, DocumentType.POSTMORTEM}
    )
    assert len(dataset.incidents) >= 500
    assert docs[DocumentType.RUNBOOK] >= 50
    assert technical >= 100
    assert len(dataset.deployments) >= 300
    assert len(dataset.code_files) >= 100


def test_all_novacart_services_exist(dataset: SyntheticDataset) -> None:
    assert {s.id for s in dataset.services} == EXPECTED_SERVICES


def test_technical_docs_cover_required_topics(dataset: SyntheticDataset) -> None:
    types = {d.doc_type for d in dataset.documents}
    assert {
        DocumentType.ARCHITECTURE,
        DocumentType.API_REFERENCE,
        DocumentType.SERVICE_BEHAVIOR,
        DocumentType.CONFIGURATION,
        DocumentType.DEPLOYMENT,
        DocumentType.DATABASE,
        DocumentType.TROUBLESHOOTING,
        DocumentType.MONITORING,
    } <= types


def test_named_runbooks_exist(dataset: SyntheticDataset) -> None:
    titles = {d.title for d in dataset.documents if d.doc_type is DocumentType.RUNBOOK}
    for expected in (
        "Payment API 500 Errors",
        "Database Connection Pool Exhaustion",
        "Redis Memory Pressure",
        "Kafka Consumer Lag",
        "Authentication Token Failure",
        "API Gateway Timeout",
        "Inventory Synchronization Failure",
    ):
        assert expected in titles


def test_incidents_cover_every_required_category(dataset: SyntheticDataset) -> None:
    assert {i.category for i in dataset.incidents} == set(IncidentCategory)


def test_incidents_have_all_required_fields(dataset: SyntheticDataset) -> None:
    for incident in dataset.incidents:
        assert incident.id.startswith("INC-")
        assert (
            incident.service_id
            and incident.severity
            and incident.affected_version
            and incident.deployment_id
        )
        assert incident.title and incident.symptoms and incident.root_cause and incident.resolution
        assert incident.resolution_time_minutes >= 0


def test_mix_of_deployment_caused_operational_and_cascading_incidents(
    dataset: SyntheticDataset,
) -> None:
    deployment_caused = [
        i for i in dataset.incidents if i.root_cause_deployment_id and not i.parent_incident_id
    ]
    cascades = [i for i in dataset.incidents if i.parent_incident_id]
    operational = [i for i in dataset.incidents if not i.root_cause_deployment_id]
    assert len(deployment_caused) >= 50
    assert len(cascades) >= 30
    assert len(operational) >= 300


def test_generation_is_deterministic(dataset: SyntheticDataset) -> None:
    again = generate_dataset(GenerationConfig())
    assert _fingerprint(again) == _fingerprint(dataset)


def test_different_seed_produces_different_data(dataset: SyntheticDataset) -> None:
    other = generate_dataset(GenerationConfig(seed=7))
    assert _fingerprint(other)["incidents"] != _fingerprint(dataset)["incidents"]
    assert validate_dataset(other) == []


def test_payment_v281_chain_supports_multi_hop_reasoning(dataset: SyntheticDataset) -> None:
    """incident -> deployment -> version -> commit -> pull request -> file -> change."""
    deployments = {d.id: d for d in dataset.deployments}
    prs = {p.id: p for p in dataset.pull_requests}
    files = {f.id: f for f in dataset.code_files}
    incident = next(
        i for i in dataset.incidents if i.id == dataset.manifest.anchors["payment-500s-v2.8.1"]
    )

    assert incident.service_id == "payment-service"
    assert incident.affected_version == "v2.8.1"
    deployment = deployments[incident.root_cause_deployment_id]  # type: ignore[index]
    assert (deployment.service_id, deployment.version) == ("payment-service", "v2.8.1")
    pr = prs[incident.root_cause_pr_id]  # type: ignore[index]
    assert pr.deployment_id == deployment.id
    assert deployment.commit_sha == pr.merge_commit_sha
    changed = [f for f in dataset.pull_request_files if f.pull_request_id == pr.id]
    assert [files[f.code_file_id].path for f in changed] == [
        "services/payment-service/payment_service/db/database.py"
    ]
    assert "-    pool_size=settings.db_pool_size," in changed[0].patch
    # The incident text alone does not name the file: finding it needs the hops above.
    assert "database.py" not in incident.root_cause

    rollback = deployments[incident.remediation_deployment_id]  # type: ignore[index]
    assert (
        rollback.is_rollback
        and rollback.version == "v2.8.0"
        and rollback.rollback_of_id == deployment.id
    )
    fix = next(p for p in dataset.pull_requests if f"Fixes {incident.id}." in p.description)
    assert deployments[fix.deployment_id].version == "v2.8.2"  # type: ignore[index]
    children = {i.service_id for i in dataset.incidents if i.parent_incident_id == incident.id}
    assert "order-service" in children
    errors = [
        line
        for line in dataset.logs
        if line.service_id == "payment-service"
        and incident.started_at <= line.timestamp <= incident.resolved_at
        and line.level is LogLevel.ERROR
    ]
    assert errors and all(
        line.version == "v2.8.1" for line in errors if line.timestamp < rollback.deployed_at
    )
    assert any("QueuePool limit" in line.message for line in errors)


def test_every_deployment_caused_incident_traces_to_changed_code(dataset: SyntheticDataset) -> None:
    prs = {p.id: p for p in dataset.pull_requests}
    changed = Counter(f.pull_request_id for f in dataset.pull_request_files)
    for incident in dataset.incidents:
        if incident.root_cause_pr_id:
            pr = prs[incident.root_cause_pr_id]
            assert pr.deployment_id == incident.root_cause_deployment_id
            assert changed[pr.id] >= 1


def test_cascading_incidents_share_traces_across_services(dataset: SyntheticDataset) -> None:
    services_by_trace: dict[str, set[str]] = {}
    for line in dataset.logs:
        if line.trace_id and line.level is LogLevel.ERROR:
            services_by_trace.setdefault(line.trace_id, set()).add(line.service_id)
    assert sum(len(services) > 1 for services in services_by_trace.values()) >= 50


def test_logs_have_healthy_and_abnormal_patterns(dataset: SyntheticDataset) -> None:
    messages = Counter(line.message.split(" ", 1)[0] for line in dataset.logs)
    assert messages["request.completed"] > 5000
    assert messages["request.failed"] > 1000
    assert messages["deploy.started"] + messages["deploy.rollback_started"] == len(
        dataset.deployments
    )
    for field in ("timestamp", "service_id", "level", "message", "host"):
        assert all(getattr(line, field) for line in dataset.logs[:1000])
    assert all(line.deployment_id and line.version for line in dataset.logs)


def test_code_repository_is_realistic(dataset: SyntheticDataset) -> None:
    kinds = Counter(f.kind for f in dataset.code_files)
    assert (
        kinds[CodeFileKind.TEST] >= 20
        and kinds[CodeFileKind.CONFIG] >= 11
        and kinds[CodeFileKind.DOCS] >= 11
    )
    paths = {f.path for f in dataset.code_files}
    assert "README.md" in paths
    assert sum(p.endswith("/db/database.py") for p in paths) == 7
    python = [f for f in dataset.code_files if f.language == "python"]
    assert len(python) >= 100
    for f in python:
        # compile() is stricter than ast.parse: it also rejects duplicate parameters.
        compile(f.content, f.path, "exec")
    yaml_files = [f for f in dataset.code_files if f.language == "yaml"]
    assert len(yaml_files) >= 11  # one deploy manifest per service, plus platform config
    for f in yaml_files:
        assert isinstance(yaml.safe_load(f.content), dict), f.path
    symbol_kinds = Counter(s["kind"] for f in python for s in f.symbols)
    assert (
        symbol_kinds["class"] >= 50
        and symbol_kinds["function"] >= 50
        and symbol_kinds["method"] >= 100
    )


def test_patches_agree_with_repository_head(dataset: SyntheticDataset) -> None:
    """Fix and maintenance PRs end at HEAD; faulty PRs start from HEAD (their fix restored it).

    A faulty change may only add lines (a leaky buffer) or only remove them (a
    dropped check), so each direction is checked only when present."""
    head = {f.id: set(f.content.splitlines()) for f in dataset.code_files}
    culprits = {i.root_cause_pr_id for i in dataset.incidents if i.root_cause_pr_id}
    for change in dataset.pull_request_files:
        lines = change.patch.splitlines()
        added = [line[1:] for line in lines if line.startswith("+") and not line.startswith("+++")]
        removed = [
            line[1:] for line in lines if line.startswith("-") and not line.startswith("---")
        ]
        assert added or removed, change.pull_request_id
        kept, gone = (removed, added) if change.pull_request_id in culprits else (added, removed)
        # `kept`: lines that exist at HEAD; `gone`: lines HEAD no longer contains.
        assert all(line in head[change.code_file_id] for line in kept), change.pull_request_id
        if gone and any(line.strip() for line in gone):
            assert not all(line in head[change.code_file_id] for line in gone), (
                change.pull_request_id
            )
