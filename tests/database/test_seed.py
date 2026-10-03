"""Seeding the full dataset (SQLite, foreign keys enforced)."""

from __future__ import annotations

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from app.database.models import Deployment, Incident
from app.database.seed import seed_database, table_counts
from app.synthetic.records import SyntheticDataset


def test_seed_loads_every_record(sqlite_engine: Engine, dataset: SyntheticDataset) -> None:
    inserted = seed_database(sqlite_engine, dataset)
    counts = table_counts(sqlite_engine)
    assert inserted == dataset.manifest.counts
    for table, expected in dataset.manifest.counts.items():
        assert counts[table] == expected, table
    assert counts["document_chunks"] == 0  # written by scripts/ingest.py, not by seeding


def test_seed_is_idempotent(sqlite_engine: Engine, dataset: SyntheticDataset) -> None:
    seed_database(sqlite_engine, dataset)
    first = table_counts(sqlite_engine)
    seed_database(sqlite_engine, dataset)
    assert table_counts(sqlite_engine) == first


def test_seeded_data_supports_orm_traversal(
    sqlite_engine: Engine, dataset: SyntheticDataset
) -> None:
    seed_database(sqlite_engine, dataset)
    anchor = dataset.manifest.anchors["payment-500s-v2.8.1"]
    with Session(sqlite_engine) as session:
        incident = session.get(Incident, anchor)
        assert incident is not None and incident.deployment.version == "v2.8.1"
        paths = [f.code_file.path for f in incident.root_cause_pull_request.files]  # type: ignore[union-attr]
        assert paths == ["services/payment-service/payment_service/db/database.py"]
        assert incident.postmortem is not None and anchor in incident.postmortem.content
        assert {c.service_id for c in incident.child_incidents} >= {"order-service"}
        deployment = session.get(Deployment, incident.root_cause_deployment_id)
        assert deployment is not None and {pr.id for pr in deployment.pull_requests} == {
            incident.root_cause_pr_id
        }
