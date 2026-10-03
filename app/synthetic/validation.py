"""Integrity checks for a generated dataset.

``validate_dataset`` returns human-readable problems (empty when the dataset is
consistent). The generator refuses to write a dataset that fails validation, and
the tests run these checks against the real output.
"""

from __future__ import annotations

import bisect
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import datetime

from app.schemas.enums import DeploymentStatus, DocumentType, LogLevel
from app.synthetic.records import (
    DeploymentRecord,
    IncidentRecord,
    SyntheticDataset,
)
from app.synthetic.text import sha256

EXPECTED_SERVICES = frozenset(
    {
        "api-gateway",
        "auth-service",
        "user-service",
        "product-service",
        "inventory-service",
        "cart-service",
        "order-service",
        "payment-service",
        "notification-service",
        "recommendation-service",
        "search-service",
    }
)

MINIMUM_COUNTS = {"incidents": 500, "deployments": 300, "code_files": 100}
MINIMUM_RUNBOOKS = 50
MINIMUM_TECHNICAL_DOCS = 100


def _duplicates(values: Iterable[object]) -> list[object]:
    return [value for value, count in Counter(values).items() if count > 1]


def _semver(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.removeprefix("v").split("."))


class _Checker:
    def __init__(self, ds: SyntheticDataset) -> None:
        self.ds = ds
        self.errors: list[str] = []
        self.services = {s.id for s in ds.services}
        self.users = {u.id for u in ds.users}
        self.roles = {r.id for r in ds.roles}
        self.deployments = {d.id: d for d in ds.deployments}
        self.prs = {p.id: p for p in ds.pull_requests}
        self.files = {f.id: f for f in ds.code_files}
        self.docs = {d.id: d for d in ds.documents}
        self.incidents = {i.id: i for i in ds.incidents}
        self.window = (ds.manifest.window_start, ds.manifest.window_end)

    def fail(self, message: str) -> None:
        self.errors.append(message)

    def ref(
        self, owner: str, field: str, value: str | None, table: dict | set, nullable: bool = True
    ) -> None:
        if value is None:
            if not nullable:
                self.fail(f"{owner}: {field} is required")
            return
        if value not in table:
            self.fail(f"{owner}: {field}={value!r} does not exist")

    def in_window(self, owner: str, field: str, moment: datetime) -> None:
        if not (self.window[0] <= moment <= self.window[1]):
            self.fail(f"{owner}: {field} {moment.isoformat()} outside the dataset window")

    # -- checks

    def counts(self) -> None:
        if self.services != EXPECTED_SERVICES:
            difference = sorted(self.services ^ EXPECTED_SERVICES)
            self.fail(f"services differ from the NovaCart catalog: {difference}")
        for table, minimum in MINIMUM_COUNTS.items():
            if len(self.ds.table(table)) < minimum:
                self.fail(
                    f"{table}: {len(self.ds.table(table))} records, expected at least {minimum}"
                )
        runbooks = sum(d.doc_type is DocumentType.RUNBOOK for d in self.ds.documents)
        technical = sum(
            d.doc_type not in {DocumentType.RUNBOOK, DocumentType.POSTMORTEM}
            for d in self.ds.documents
        )
        if runbooks < MINIMUM_RUNBOOKS:
            self.fail(f"runbooks: {runbooks}, expected at least {MINIMUM_RUNBOOKS}")
        if technical < MINIMUM_TECHNICAL_DOCS:
            self.fail(
                f"technical documents: {technical}, expected at least {MINIMUM_TECHNICAL_DOCS}"
            )
        levels = {line.level for line in self.ds.logs}
        if not {LogLevel.INFO, LogLevel.ERROR} <= levels:
            self.fail("logs must contain both healthy (INFO) and abnormal (ERROR) lines")
        for name, count in self.ds.manifest.counts.items():
            actual = len(self.ds.table(name))
            if actual != count:
                self.fail(f"manifest count for {name} ({count}) does not match {actual} records")

    def uniqueness(self) -> None:
        keys = {
            "roles.id": [r.id for r in self.ds.roles],
            "users.id": [u.id for u in self.ds.users],
            "users.email": [u.email for u in self.ds.users],
            "services.id": [s.id for s in self.ds.services],
            "service_dependencies": [
                (d.service_id, d.depends_on_id) for d in self.ds.service_dependencies
            ],
            "code_files.id": list(self.files)
            if len(self.files) == len(self.ds.code_files)
            else [f.id for f in self.ds.code_files],
            "code_files.path": [f.path for f in self.ds.code_files],
            "deployments.id": [d.id for d in self.ds.deployments],
            "pull_requests.id": [p.id for p in self.ds.pull_requests],
            "pull_requests.number": [p.number for p in self.ds.pull_requests],
            "pull_requests.merge_commit_sha": [
                p.merge_commit_sha for p in self.ds.pull_requests if p.merge_commit_sha
            ],
            "pull_request_files": [
                (f.pull_request_id, f.code_file_id) for f in self.ds.pull_request_files
            ],
            "documents.id": [d.id for d in self.ds.documents],
            "documents.source_path": [d.source_path for d in self.ds.documents],
            "incidents.id": [i.id for i in self.ds.incidents],
        }
        for name, values in keys.items():
            duplicates = _duplicates(values)
            if duplicates:
                self.fail(f"{name}: duplicate values {duplicates[:5]}")

    def references(self) -> None:
        for u in self.ds.users:
            self.ref(f"user {u.id}", "role_id", u.role_id, self.roles, nullable=False)
        for d in self.ds.service_dependencies:
            owner = f"dependency {d.service_id}->{d.depends_on_id}"
            self.ref(owner, "service_id", d.service_id, self.services, nullable=False)
            self.ref(owner, "depends_on_id", d.depends_on_id, self.services, nullable=False)
        for f in self.ds.code_files:
            self.ref(f"code file {f.path}", "service_id", f.service_id, self.services)
        for d in self.ds.deployments:
            self.ref(d.id, "service_id", d.service_id, self.services, nullable=False)
            self.ref(d.id, "author_id", d.author_id, self.users, nullable=False)
            self.ref(d.id, "rollback_of_id", d.rollback_of_id, self.deployments)
        for p in self.ds.pull_requests:
            self.ref(p.id, "service_id", p.service_id, self.services, nullable=False)
            self.ref(p.id, "author_id", p.author_id, self.users, nullable=False)
            self.ref(p.id, "deployment_id", p.deployment_id, self.deployments)
            for reviewer in p.reviewers:
                self.ref(p.id, "reviewers[]", reviewer, self.users)
        for f in self.ds.pull_request_files:
            self.ref(
                f"PR file {f.pull_request_id}",
                "pull_request_id",
                f.pull_request_id,
                self.prs,
                nullable=False,
            )
            self.ref(
                f"PR file {f.pull_request_id}",
                "code_file_id",
                f.code_file_id,
                self.files,
                nullable=False,
            )
        for d in self.ds.documents:
            self.ref(d.id, "service_id", d.service_id, self.services)
            self.ref(d.id, "author_id", d.author_id, self.users)
        for i in self.ds.incidents:
            self.ref(i.id, "service_id", i.service_id, self.services, nullable=False)
            self.ref(i.id, "root_cause_service_id", i.root_cause_service_id, self.services)
            self.ref(i.id, "parent_incident_id", i.parent_incident_id, self.incidents)
            self.ref(i.id, "deployment_id", i.deployment_id, self.deployments, nullable=False)
            self.ref(i.id, "root_cause_deployment_id", i.root_cause_deployment_id, self.deployments)
            self.ref(i.id, "root_cause_pr_id", i.root_cause_pr_id, self.prs)
            self.ref(
                i.id, "remediation_deployment_id", i.remediation_deployment_id, self.deployments
            )
            self.ref(i.id, "runbook_id", i.runbook_id, self.docs)
            self.ref(i.id, "postmortem_id", i.postmortem_id, self.docs)
            self.ref(i.id, "commander_id", i.commander_id, self.users, nullable=False)
        for n, line in enumerate(self.ds.logs):
            self.ref(f"log #{n}", "service_id", line.service_id, self.services, nullable=False)
            self.ref(f"log #{n}", "deployment_id", line.deployment_id, self.deployments)

    def timestamps(self) -> None:
        for record in (
            *self.ds.roles,
            *self.ds.users,
            *self.ds.services,
            *self.ds.code_files,
            *self.ds.deployments,
            *self.ds.pull_requests,
            *self.ds.documents,
            *self.ds.incidents,
        ):
            if record.updated_at < record.created_at:  # type: ignore[attr-defined]
                self.fail(f"{record.id}: updated_at before created_at")  # type: ignore[attr-defined]
        for d in self.ds.deployments:
            self.in_window(d.id, "deployed_at", d.deployed_at)
        for p in self.ds.pull_requests:
            if p.merged_at is None:
                continue
            self.in_window(p.id, "merged_at", p.merged_at)
            if p.opened_at > p.merged_at:
                self.fail(f"{p.id}: opened after it was merged")
            shipped_by = self.deployments.get(p.deployment_id or "")
            if shipped_by is not None and p.merged_at > shipped_by.deployed_at:
                self.fail(f"{p.id}: merged after {p.deployment_id} was deployed")
        for i in self.ds.incidents:
            for field in ("started_at", "detected_at", "resolved_at"):
                self.in_window(i.id, field, getattr(i, field))
            if not (i.started_at <= i.detected_at <= i.resolved_at):
                self.fail(f"{i.id}: expected started_at <= detected_at <= resolved_at")
            if i.resolution_time_minutes != int(
                (i.resolved_at - i.started_at).total_seconds() // 60
            ):
                self.fail(f"{i.id}: resolution_time_minutes does not match started_at/resolved_at")
        previous = None
        for n, line in enumerate(self.ds.logs):
            self.in_window(f"log #{n}", "timestamp", line.timestamp)
            if previous is not None and line.timestamp < previous:
                self.fail(f"log #{n}: logs are not in timestamp order")
                break
            previous = line.timestamp

    def deployment_history(self) -> None:
        by_service: dict[str, list[DeploymentRecord]] = defaultdict(list)
        for d in sorted(self.ds.deployments, key=lambda d: d.deployed_at):
            by_service[d.service_id].append(d)
        for service in self.services:
            if not by_service.get(service):
                self.fail(f"service {service} has no deployments")
        for service, history in by_service.items():
            highest: tuple[int, ...] | None = None
            for d in history:
                if d.is_rollback:
                    target = self.deployments.get(d.rollback_of_id or "")
                    if target is None or target.service_id != service:
                        self.fail(f"{d.id}: rollback must revert a deployment of the same service")
                    elif d.version != target.previous_version:
                        self.fail(
                            f"{d.id}: rollback version {d.version} != {target.id}.previous_version"
                        )
                    elif target.status is not DeploymentStatus.ROLLED_BACK:
                        self.fail(
                            f"{target.id}: reverted by {d.id} but status is {target.status.value}"
                        )
                    continue
                version = _semver(d.version)
                if highest is not None and version <= highest:
                    self.fail(f"{d.id}: version {d.version} does not increase for {service}")
                highest = version
        shipped = defaultdict(list)
        for p in self.ds.pull_requests:
            if p.deployment_id:
                shipped[p.deployment_id].append(p)
                deployment = self.deployments.get(p.deployment_id)
                if deployment and deployment.service_id != p.service_id:
                    self.fail(f"{p.id}: shipped by {deployment.id} of another service")
        for d in self.ds.deployments:
            if d.is_rollback and shipped.get(d.id):
                self.fail(f"{d.id}: a rollback cannot ship pull requests")
            if d.status is DeploymentStatus.FAILED and shipped.get(d.id):
                self.fail(f"{d.id}: a failed rollout cannot be the deployment that shipped a PR")

    def live_deployments(self) -> None:
        live: dict[str, tuple[list[datetime], list[DeploymentRecord]]] = {}
        for service in self.services:
            history = sorted(
                (
                    d
                    for d in self.ds.deployments
                    if d.service_id == service and d.status is not DeploymentStatus.FAILED
                ),
                key=lambda d: d.deployed_at,
            )
            live[service] = ([d.deployed_at for d in history], history)

        def live_at(service: str, moment: datetime) -> DeploymentRecord | None:
            times, history = live[service]
            index = bisect.bisect_right(times, moment) - 1
            return history[index] if index >= 0 else None

        for i in self.ds.incidents:
            actual = self.deployments.get(i.deployment_id)
            if (
                actual is None or i.service_id not in live
            ):  # dangling refs are reported by references()
                continue
            expected = live_at(i.service_id, i.started_at)
            if expected is None or expected.id != actual.id:
                self.fail(
                    f"{i.id}: deployment_id {actual.id} is not the deployment live at started_at"
                )
            if actual.version != i.affected_version:
                self.fail(
                    f"{i.id}: affected_version {i.affected_version} "
                    f"!= {actual.id}.version {actual.version}"
                )
        for n, line in enumerate(self.ds.logs):
            if line.deployment_id is None:
                continue
            d = self.deployments.get(line.deployment_id)
            if d is not None and (d.service_id != line.service_id or d.version != line.version):
                self.fail(f"log #{n}: deployment/version inconsistent with {d.id}")

    def incident_chains(self) -> None:
        touched = defaultdict(list)
        for f in self.ds.pull_request_files:
            touched[f.pull_request_id].append(f)
        for i in self.ds.incidents:
            self._incident_chain(i, touched)

    def _incident_chain(self, i: IncidentRecord, touched: dict[str, list]) -> None:
        if (
            i.deployment_id in self.deployments
            and self.deployments[i.deployment_id].service_id != i.service_id
        ):
            self.fail(f"{i.id}: deployment_id belongs to another service")
        if i.parent_incident_id:
            parent = self.incidents.get(i.parent_incident_id)
            if parent is not None:
                if parent.parent_incident_id is not None:
                    self.fail(f"{i.id}: parent {parent.id} is itself a child incident")
                if parent.started_at > i.started_at:
                    self.fail(f"{i.id}: starts before its parent {parent.id}")
                if parent.service_id == i.service_id:
                    self.fail(f"{i.id}: cascade within the same service as {parent.id}")
        rcd = self.deployments.get(i.root_cause_deployment_id or "")
        rcpr = self.prs.get(i.root_cause_pr_id or "")
        if (i.root_cause_deployment_id is None) != (i.root_cause_pr_id is None):
            self.fail(f"{i.id}: root-cause deployment and pull request must be set together")
        if rcd is not None:
            if rcd.deployed_at > i.started_at:
                self.fail(f"{i.id}: root-cause deployment {rcd.id} is after the incident started")
            if i.root_cause_service_id != rcd.service_id:
                self.fail(f"{i.id}: root_cause_service_id != {rcd.id}.service_id")
        if rcpr is not None:
            if rcpr.deployment_id != i.root_cause_deployment_id:
                self.fail(
                    f"{i.id}: root-cause PR {rcpr.id} was not shipped by "
                    f"{i.root_cause_deployment_id}"
                )
            files = touched.get(rcpr.id, [])
            if not files or not all(f.patch.strip() for f in files):
                self.fail(f"{i.id}: root-cause PR {rcpr.id} has no file changes")
        remediation = self.deployments.get(i.remediation_deployment_id or "")
        if remediation is not None:
            if rcd is None:
                self.fail(f"{i.id}: remediation without a root-cause deployment")
            elif remediation.service_id != rcd.service_id:
                self.fail(f"{i.id}: remediation {remediation.id} is for another service")
            if not (i.started_at <= remediation.deployed_at <= i.resolved_at):
                self.fail(f"{i.id}: remediation {remediation.id} not deployed during the incident")
            if remediation.is_rollback and remediation.rollback_of_id != i.root_cause_deployment_id:
                self.fail(
                    f"{i.id}: rollback {remediation.id} does not revert the root-cause deployment"
                )
        if (
            i.runbook_id
            and self.docs.get(i.runbook_id)
            and self.docs[i.runbook_id].doc_type is not DocumentType.RUNBOOK
        ):
            self.fail(f"{i.id}: runbook_id {i.runbook_id} is not a runbook")
        if i.postmortem_id and self.docs.get(i.postmortem_id):
            postmortem = self.docs[i.postmortem_id]
            if postmortem.doc_type is not DocumentType.POSTMORTEM or i.id not in postmortem.content:
                self.fail(
                    f"{i.id}: postmortem {i.postmortem_id} is not a postmortem about this incident"
                )

    def contents(self) -> None:
        for f in self.ds.code_files:
            if f.content_hash != sha256(f.content):
                self.fail(f"{f.path}: content_hash mismatch")
            if f.language == "python":
                try:
                    # compile(), not ast.parse(): it also rejects e.g. duplicate parameter names.
                    compile(f.content, f.path, "exec")
                except SyntaxError as exc:
                    self.fail(f"{f.path}: invalid Python ({exc.msg} line {exc.lineno})")
        for d in self.ds.documents:
            if d.content_hash != sha256(d.content):
                self.fail(f"{d.id}: content_hash mismatch")
            if not d.content.strip():
                self.fail(f"{d.id}: empty document")
        for f in self.ds.pull_request_files:
            path = self.files[f.code_file_id].path if f.code_file_id in self.files else None
            if path and f"+++ b/{path}" not in f.patch:
                self.fail(f"PR file {f.pull_request_id}: patch does not target {path}")
            pr = self.prs.get(f.pull_request_id)
            code_file = self.files.get(f.code_file_id)
            if pr and code_file and code_file.service_id not in {None, pr.service_id}:
                self.fail(f"{pr.id}: changes {code_file.path} of another service")

    def run(self) -> list[str]:
        for check in (
            self.counts,
            self.uniqueness,
            self.references,
            self.timestamps,
            self.deployment_history,
            self.live_deployments,
            self.incident_chains,
            self.contents,
        ):
            check()
        return self.errors


def validate_dataset(dataset: SyntheticDataset) -> list[str]:
    return _Checker(dataset).run()
