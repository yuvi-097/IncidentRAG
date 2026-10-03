"""ORM models: table creation, relationships and constraint enforcement.

Runs on in-memory SQLite (foreign keys enabled). PostgreSQL-specific parts
(pgvector column type, HNSW index, JSONB) are covered by tests/integration.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database.base import Base
from app.database.models import (
    ChunkEmbedding,
    CodeFile,
    Deployment,
    Document,
    DocumentChunk,
    Incident,
    LogEntry,
    PullRequest,
    PullRequestFile,
    Role,
    Service,
    ServiceDependency,
    User,
)
from app.schemas.enums import (
    AccessLevel,
    ChunkingStrategy,
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
    SourceType,
)
from app.synthetic.records import TABLES

T0 = datetime(2026, 6, 16, 13, 40, tzinfo=UTC)
REQUIRED_TABLES = {
    "users",
    "roles",
    "services",
    "incidents",
    "deployments",
    "documents",
    "document_chunks",
    "logs",
    "code_files",
    "pull_requests",
}


def _graph(session: Session) -> None:
    """A minimal incident -> deployment -> PR -> file chain."""
    session.add_all(
        [
            Role(id="sre", name="SRE", description=""),
            User(
                id="alex",
                email="alex@novacart.example",
                full_name="Alex",
                team="sre",
                role_id="sre",
            ),
            Service(
                id="payment-service",
                display_name="Payment Service",
                description="",
                owner_team="payments",
                tier=ServiceTier.TIER_0,
                language="python",
                repository_path="services/payment-service",
                port=8087,
                oncall_channel="#payments-oncall",
                datastores=["postgres:payments"],
            ),
            Service(
                id="order-service",
                display_name="Order Service",
                description="",
                owner_team="commerce",
                tier=ServiceTier.TIER_0,
                language="python",
                repository_path="services/order-service",
                port=8086,
                oncall_channel="#commerce-oncall",
                datastores=[],
            ),
        ]
    )
    session.flush()
    session.add_all(
        [
            ServiceDependency(
                service_id="order-service",
                depends_on_id="payment-service",
                protocol=DependencyProtocol.HTTP,
                criticality=DependencyCriticality.HARD,
                description="payments",
            ),
            CodeFile(
                id="CF-0001",
                repository="novacart",
                path="services/payment-service/payment_service/db/database.py",
                service_id="payment-service",
                language="python",
                kind=CodeFileKind.SOURCE,
                content="x = 1\n",
                line_count=1,
                content_hash="0" * 64,
                symbols=[],
                last_commit_sha="a" * 40,
                last_modified_at=T0,
                access_level=AccessLevel.SRE,
            ),
            Deployment(
                id="DEP-0001",
                service_id="payment-service",
                version="v2.8.1",
                previous_version="v2.8.0",
                commit_sha="a" * 40,
                deployed_at=T0,
                author_id="alex",
                environment="production",
                strategy=DeploymentStrategy.CANARY,
                status=DeploymentStatus.ROLLED_BACK,
                is_rollback=False,
                changes="Release v2.8.1",
                duration_seconds=900,
            ),
        ]
    )
    session.flush()
    session.add_all(
        [
            Deployment(
                id="DEP-0002",
                service_id="payment-service",
                version="v2.8.0",
                previous_version="v2.8.1",
                commit_sha="b" * 40,
                deployed_at=T0 + timedelta(hours=4),
                author_id="alex",
                environment="production",
                strategy=DeploymentStrategy.ROLLING,
                status=DeploymentStatus.SUCCEEDED,
                is_rollback=True,
                rollback_of_id="DEP-0001",
                changes="Rollback",
                duration_seconds=120,
            ),
            PullRequest(
                id="PR-1001",
                number=1001,
                service_id="payment-service",
                title="Reduce idle DB connections",
                description="",
                author_id="alex",
                reviewers=[],
                labels=["performance"],
                state=PullRequestState.MERGED,
                base_branch="main",
                head_branch="alex/pool",
                opened_at=T0 - timedelta(days=1),
                merged_at=T0 - timedelta(hours=2),
                merge_commit_sha="a" * 40,
                deployment_id="DEP-0001",
            ),
            Document(
                id="RB-0001",
                doc_type=DocumentType.RUNBOOK,
                title="Payment API 500 Errors",
                service_id="payment-service",
                content="# Runbook",
                source_path="docs/runbooks/payment.md",
                content_hash="0" * 64,
                tags=["runbook"],
                access_level=AccessLevel.ENGINEERING,
                author_id="alex",
                revision=1,
            ),
        ]
    )
    session.flush()
    session.add_all(
        [
            PullRequestFile(
                pull_request_id="PR-1001",
                code_file_id="CF-0001",
                change_type=FileChangeType.MODIFIED,
                additions=3,
                deletions=2,
                patch="--- a/x\n+++ b/x\n",
            ),
            DocumentChunk(
                id="RB-0001#000",
                document_id="RB-0001",
                source_type=SourceType.RUNBOOK,
                chunk_index=0,
                title="Payment API 500 Errors",
                section=None,
                content="# Runbook",
                service_id="payment-service",
                timestamp=T0,
                access_level=AccessLevel.ENGINEERING,
                version=None,
                doc_type="runbook",
                file_path="docs/runbooks/payment.md",
                strategy=ChunkingStrategy.DOCUMENT_AWARE,
                char_start=0,
                char_end=9,
                token_count=3,
                source_hash="0" * 64,
                record_hash="1" * 64,
                chunk_metadata={},
                source_document_id="RB-0001",
            ),
            _incident("INC-0001", "payment-service", None),
        ]
    )
    session.flush()
    session.add(
        ChunkEmbedding(
            chunk_id="RB-0001#000",
            model="test-model",
            dimension=3,
            text_hash="2" * 64,
            embedding=[0.1, 0.2, 0.3],
        )
    )
    session.flush()
    session.add(_incident("INC-0002", "order-service", "INC-0001"))
    session.flush()


def _incident(incident_id: str, service: str, parent: str | None, **overrides: Any) -> Incident:
    values: dict[str, Any] = dict(
        id=incident_id,
        title="Payment requests returning HTTP 500",
        service_id=service,
        root_cause_service_id="payment-service",
        parent_incident_id=parent,
        category=IncidentCategory.DB_CONNECTION_EXHAUSTION,
        severity=Severity.SEV1,
        status=IncidentStatus.RESOLVED,
        started_at=T0 + timedelta(hours=3),
        detected_at=T0 + timedelta(hours=3, minutes=5),
        resolved_at=T0 + timedelta(hours=5),
        resolution_time_minutes=120,
        affected_version="v2.8.1",
        deployment_id="DEP-0001",
        root_cause_deployment_id="DEP-0001",
        root_cause_pr_id="PR-1001",
        remediation_deployment_id="DEP-0002",
        runbook_id="RB-0001",
        commander_id="alex",
        alert_name=None,
        symptoms="",
        root_cause="",
        resolution="",
        metrics={"peak_error_rate_pct": 31.2},
        tags=[],
        access_level=AccessLevel.ENGINEERING,
    )
    values.update(overrides)
    return Incident(**values)


def test_schema_creates_all_tables(sqlite_engine: Engine) -> None:
    tables = set(inspect(sqlite_engine).get_table_names())
    assert tables >= REQUIRED_TABLES
    assert tables == set(Base.metadata.tables)


def test_records_match_table_columns() -> None:
    """The dataset format and the schema must not drift apart."""
    for table, record in TABLES.items():
        columns = {c.name for c in Base.metadata.tables[table].columns}
        if table == "logs":
            columns.discard("id")  # assigned by the database
        assert set(record.model_fields) == columns, table


def test_relationships_navigate_the_incident_chain(sqlite_engine: Engine) -> None:
    with Session(sqlite_engine) as session:
        _graph(session)
        session.commit()
    with Session(sqlite_engine) as session:
        incident = session.get(Incident, "INC-0001")
        assert incident is not None
        pr = incident.root_cause_pull_request
        assert pr is not None and pr.deployment is incident.root_cause_deployment
        assert pr.files[0].code_file.path.endswith("db/database.py")
        assert incident.remediation_deployment is not None
        assert incident.remediation_deployment.rollback_of is incident.root_cause_deployment
        assert [c.id for c in incident.child_incidents] == ["INC-0002"]
        assert incident.runbook is not None
        embedding = incident.runbook.chunks[0].embeddings[0]
        assert embedding.model == "test-model"
        assert embedding.embedding == pytest.approx([0.1, 0.2, 0.3])
        assert incident.service.dependents[0].service.id == "order-service"
        assert incident.commander.role.id == "sre"


def test_foreign_keys_are_enforced(sqlite_engine: Engine) -> None:
    with Session(sqlite_engine) as session:
        _graph(session)
        session.add(_incident("INC-0003", "ghost-service", None))
        with pytest.raises(IntegrityError):
            session.flush()


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("severity", "'SEV9'"),
        ("category", "'gremlins'"),
        ("detected_at", "'2000-01-01 00:00:00'"),
        ("resolution_time_minutes", "-5"),
    ],
)
def test_check_constraints_reject_invalid_rows(
    sqlite_engine: Engine, column: str, value: str
) -> None:
    with Session(sqlite_engine) as session:
        _graph(session)
        session.commit()
    with sqlite_engine.begin() as connection:
        row = dict(
            connection.execute(text("SELECT * FROM incidents WHERE id = 'INC-0001'"))
            .mappings()
            .one()
        )
    row.update(id="INC-0009", parent_incident_id=None)
    placeholders = ", ".join(f":{name}" if name != column else value for name in row)
    with pytest.raises(IntegrityError), sqlite_engine.begin() as connection:
        connection.execute(
            text(f"INSERT INTO incidents ({', '.join(row)}) VALUES ({placeholders})"),
            {k: v for k, v in row.items() if k != column},
        )


def test_unique_constraints(sqlite_engine: Engine) -> None:
    with Session(sqlite_engine) as session:
        _graph(session)
        session.add(
            Document(
                id="RB-0002",
                doc_type=DocumentType.RUNBOOK,
                title="dup",
                service_id=None,
                content="x",
                source_path="docs/runbooks/payment.md",
                content_hash="0" * 64,
                tags=[],
                access_level=AccessLevel.ENGINEERING,
                author_id=None,
                revision=1,
            )
        )
        with pytest.raises(IntegrityError):
            session.flush()


def test_deleting_a_document_deletes_its_chunks(sqlite_engine: Engine) -> None:
    with Session(sqlite_engine) as session:
        _graph(session)
        session.get(Incident, "INC-0002").runbook_id = None  # type: ignore[union-attr]
        session.get(Incident, "INC-0001").runbook_id = None  # type: ignore[union-attr]
        session.delete(session.get(Document, "RB-0001"))
        session.flush()
        assert session.query(DocumentChunk).count() == 0


def test_log_ids_autoincrement(sqlite_engine: Engine) -> None:
    with Session(sqlite_engine) as session:
        _graph(session)
        for n in range(3):
            session.add(
                LogEntry(
                    timestamp=T0 + timedelta(seconds=n),
                    service_id="payment-service",
                    level=LogLevel.ERROR,
                    logger="payment_service.api.routes",
                    message="request.failed",
                    trace_id="f" * 32,
                    span_id="e" * 16,
                    deployment_id="DEP-0001",
                    version="v2.8.1",
                    host="payment-service-aaaaaaaaaa-00001",
                    attributes={},
                )
            )
        session.flush()
        assert sorted(line.id for line in session.query(LogEntry)) == [1, 2, 3]
