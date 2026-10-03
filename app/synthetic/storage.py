"""Reading and writing datasets.

Layout of a dataset directory::

    manifest.json          seed, generator version, counts, sha256 of every table file
    <table>.jsonl          one JSON record per line (tables in ``records.TABLES``)
    novacart-repo/...      the code repository materialised as files (optional)

Output is byte-for-byte deterministic for a given seed.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from app.synthetic.records import TABLES, DatasetManifest, Record, SyntheticDataset

MANIFEST = "manifest.json"
REPO_DIR = "novacart-repo"


class DatasetIntegrityError(RuntimeError):
    pass


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _jsonl(records: list[Record]) -> str:
    return "".join(record.model_dump_json() + "\n" for record in records)


def write_dataset(
    dataset: SyntheticDataset, directory: Path, *, include_repo: bool = True
) -> DatasetManifest:
    """Write ``dataset`` to ``directory``, replacing files a previous run produced there."""
    directory.mkdir(parents=True, exist_ok=True)
    for stale in [*directory.glob("*.jsonl"), directory / MANIFEST]:
        stale.unlink(missing_ok=True)
    shutil.rmtree(directory / REPO_DIR, ignore_errors=True)

    hashes = {}
    for table in TABLES:
        text = _jsonl(dataset.table(table))
        _write_text(directory / f"{table}.jsonl", text)
        hashes[f"{table}.jsonl"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    manifest = dataset.manifest.model_copy(update={"files": hashes})
    _write_text(directory / MANIFEST, json.dumps(manifest.model_dump(mode="json"), indent=2) + "\n")
    if include_repo:
        for code_file in dataset.code_files:
            _write_text(directory / REPO_DIR / code_file.path, code_file.content)
    return manifest


def load_dataset(directory: Path, *, verify: bool = True) -> SyntheticDataset:
    """Load and validate a dataset written by ``write_dataset``.

    With ``verify`` the sha256 of every table file must match the manifest, so a
    hand-edited or truncated file is rejected instead of silently seeded.
    """
    manifest_path = directory / MANIFEST
    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_path} not found; run scripts/generate_data.py first")
    manifest = DatasetManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    tables: dict[str, list[Record]] = {}
    for table, record_type in TABLES.items():
        path = directory / f"{table}.jsonl"
        raw = path.read_bytes()
        if verify and hashlib.sha256(raw).hexdigest() != manifest.files.get(path.name):
            raise DatasetIntegrityError(f"{path.name} does not match the manifest checksum")
        tables[table] = [
            record_type.model_validate_json(line)
            for line in raw.decode("utf-8").splitlines()
            if line
        ]
    return SyntheticDataset(manifest=manifest, **tables)  # type: ignore[arg-type]


def sample_dataset(
    dataset: SyntheticDataset, anchor: str = "payment-500s-v2.8.1", max_logs: int = 200
) -> SyntheticDataset:
    """A small, self-consistent excerpt for committing to Git: one anchor incident
    and everything it links to (child incidents, deployments, PRs, changed files,
    runbook, postmortem, logs), plus the reference tables (roles, users, services)."""
    root = next(i for i in dataset.incidents if i.id == dataset.manifest.anchors[anchor])
    incidents = [root, *(i for i in dataset.incidents if i.parent_incident_id == root.id)]
    deployment_ids = {
        x
        for i in incidents
        for x in (i.deployment_id, i.root_cause_deployment_id, i.remediation_deployment_id)
        if x
    }
    pr_ids = {i.root_cause_pr_id for i in incidents if i.root_cause_pr_id}
    pr_ids |= {p.id for p in dataset.pull_requests if p.deployment_id in deployment_ids}
    pr_ids |= {p.id for p in dataset.pull_requests if f"Fixes {root.id}." in p.description}
    prs = [p for p in dataset.pull_requests if p.id in pr_ids]
    deployment_ids |= {p.deployment_id for p in prs if p.deployment_id}
    by_id = {d.id: d for d in dataset.deployments}
    pending = list(deployment_ids)
    while pending:  # rollbacks reference the deployment they revert
        target = by_id[pending.pop()].rollback_of_id
        if target and target not in deployment_ids:
            deployment_ids.add(target)
            pending.append(target)
    deployments = [d for d in dataset.deployments if d.id in deployment_ids]
    pr_files = [f for f in dataset.pull_request_files if f.pull_request_id in pr_ids]
    file_ids = {f.code_file_id for f in pr_files}
    doc_ids = {x for i in incidents for x in (i.runbook_id, i.postmortem_id) if x}
    services = {i.service_id for i in incidents} | {d.service_id for d in deployments}
    window = (root.started_at, root.resolved_at)
    logs = [
        line
        for line in dataset.logs
        if line.service_id in services
        and line.deployment_id in deployment_ids
        and window[0] <= line.timestamp <= window[1]
    ]
    tables = {
        "roles": dataset.roles,
        "users": dataset.users,
        "services": dataset.services,
        "service_dependencies": dataset.service_dependencies,
        "code_files": [f for f in dataset.code_files if f.id in file_ids],
        "deployments": deployments,
        "pull_requests": prs,
        "pull_request_files": pr_files,
        "documents": [d for d in dataset.documents if d.id in doc_ids],
        "incidents": incidents,
        "logs": logs[:max_logs],
    }
    manifest = dataset.manifest.model_copy(
        update={
            "counts": {name: len(rows) for name, rows in tables.items()},
            "files": {},
            "anchors": {anchor: root.id},
        }
    )
    return SyntheticDataset(manifest=manifest, **tables)  # type: ignore[arg-type]
