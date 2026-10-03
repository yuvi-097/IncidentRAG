"""Data integrity: explicit checks on the real dataset, and proof that the
validator catches each kind of corruption."""

from __future__ import annotations

from collections import Counter
from datetime import timedelta
from typing import Any

import pytest

from app.synthetic.records import SyntheticDataset
from app.synthetic.validation import validate_dataset


def _duplicates(values: list[Any]) -> list[Any]:
    return [value for value, count in Counter(values).items() if count > 1]


def test_every_deployment_references_a_valid_service(dataset: SyntheticDataset) -> None:
    services = {s.id for s in dataset.services}
    assert all(d.service_id in services for d in dataset.deployments)


def test_incidents_reference_valid_services(dataset: SyntheticDataset) -> None:
    services = {s.id for s in dataset.services}
    for incident in dataset.incidents:
        assert incident.service_id in services
        assert incident.root_cause_service_id is None or incident.root_cause_service_id in services


def test_ids_are_unique(dataset: SyntheticDataset) -> None:
    for values in (
        [r.id for r in dataset.incidents],
        [r.id for r in dataset.deployments],
        [r.id for r in dataset.pull_requests],
        [r.number for r in dataset.pull_requests],
        [r.id for r in dataset.documents],
        [r.id for r in dataset.code_files],
        [r.path for r in dataset.code_files],
        [r.id for r in dataset.users],
    ):
        assert _duplicates(values) == []


def test_timestamps_are_valid(dataset: SyntheticDataset) -> None:
    start, end = dataset.manifest.window_start, dataset.manifest.window_end
    for incident in dataset.incidents:
        assert start <= incident.started_at <= incident.detected_at <= incident.resolved_at <= end
        assert incident.resolution_time_minutes == int(
            (incident.resolved_at - incident.started_at).total_seconds() // 60
        )
    for deployment in dataset.deployments:
        assert start <= deployment.deployed_at <= end
    for pr in dataset.pull_requests:
        assert pr.merged_at is not None and pr.opened_at <= pr.merged_at
    assert all(start <= line.timestamp < end for line in dataset.logs)


def test_relationships_are_consistent(dataset: SyntheticDataset) -> None:
    deployments = {d.id: d for d in dataset.deployments}
    prs = {p.id: p for p in dataset.pull_requests}
    for incident in dataset.incidents:
        live = deployments[incident.deployment_id]
        assert live.service_id == incident.service_id and live.version == incident.affected_version
        assert live.deployed_at <= incident.started_at
        if incident.root_cause_pr_id:
            assert prs[incident.root_cause_pr_id].deployment_id == incident.root_cause_deployment_id
    for pr in dataset.pull_requests:
        if pr.deployment_id:
            deployment = deployments[pr.deployment_id]
            assert deployment.service_id == pr.service_id and pr.merged_at <= deployment.deployed_at


# --- the validator detects corruption ------------------------------------------------------------


def _corrupt(dataset: SyntheticDataset, table: str, index: int, **changes: Any) -> SyntheticDataset:
    rows = list(dataset.table(table))
    rows[index] = rows[index].model_copy(update=changes)
    return dataset.model_copy(update={table: rows})


def _first(dataset: SyntheticDataset, table: str, predicate: Any) -> int:
    return next(n for n, row in enumerate(dataset.table(table)) if predicate(row))


@pytest.mark.parametrize(
    ("table", "selector", "changes", "expected"),
    [
        (
            "deployments",
            lambda r: True,
            {"service_id": "ghost-service"},
            "service_id='ghost-service' does not exist",
        ),
        (
            "incidents",
            lambda r: True,
            {"service_id": "ghost-service"},
            "service_id='ghost-service' does not exist",
        ),
        ("incidents", lambda r: True, {"affected_version": "v0.0.0"}, "affected_version"),
        (
            "incidents",
            lambda r: True,
            {"resolution_time_minutes": 99999},
            "resolution_time_minutes",
        ),
        (
            "incidents",
            lambda r: r.root_cause_pr_id is not None,
            {"root_cause_pr_id": "PR-1001"},
            "was not shipped by",
        ),
        (
            "incidents",
            lambda r: r.parent_incident_id is not None,
            {"parent_incident_id": "INC-9999"},
            "does not exist",
        ),
        (
            "pull_requests",
            lambda r: r.deployment_id is not None,
            {"service_id": "cart-service"},
            "of another service",
        ),
        (
            "logs",
            lambda r: r.deployment_id is not None,
            {"version": "v9.9.9"},
            "deployment/version inconsistent",
        ),
        ("code_files", lambda r: True, {"content": "tampered\n"}, "content_hash mismatch"),
        ("documents", lambda r: True, {"author_id": "nobody"}, "author_id='nobody' does not exist"),
    ],
)
def test_validator_detects_corruption(
    dataset: SyntheticDataset, table: str, selector: Any, changes: dict[str, Any], expected: str
) -> None:
    corrupted = _corrupt(dataset, table, _first(dataset, table, selector), **changes)
    errors = validate_dataset(corrupted)
    assert any(expected in error for error in errors), errors[:5]


def test_validator_detects_duplicate_ids(dataset: SyntheticDataset) -> None:
    incidents = [*dataset.incidents, dataset.incidents[0]]
    errors = validate_dataset(dataset.model_copy(update={"incidents": incidents}))
    assert any("incidents.id: duplicate" in e for e in errors)


def test_validator_detects_impossible_timeline(dataset: SyntheticDataset) -> None:
    incident = dataset.incidents[0]
    corrupted = _corrupt(
        dataset, "incidents", 0, detected_at=incident.started_at - timedelta(minutes=5)
    )
    assert any("started_at <= detected_at <= resolved_at" in e for e in validate_dataset(corrupted))
