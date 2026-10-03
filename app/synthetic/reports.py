"""Operational reports (MANAGER) and administrative policies (ADMIN).

Reports are computed from the generated incidents and deployments, so every figure in
them can be checked against the structured tables. Nothing here draws random numbers,
so adding these documents leaves the rest of the dataset unchanged.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime, timedelta
from statistics import mean

from app.schemas.enums import AccessLevel, DeploymentStatus, DocumentType, Severity
from app.synthetic.catalog import PEOPLE, ROLES
from app.synthetic.documents import DocumentDraft
from app.synthetic.facts import DeploymentFact, IncidentFact
from app.synthetic.timeline import WINDOW_END, WINDOW_START

REPORT_AUTHOR = "sarah.miller"
POLICY_AUTHOR = "noor.hassan"


def _quarters() -> Iterator[tuple[str, datetime, datetime]]:
    """Calendar quarters overlapping the dataset window, clipped to it."""
    year, quarter = WINDOW_START.year, (WINDOW_START.month - 1) // 3
    while True:
        start = datetime(year, quarter * 3 + 1, 1, tzinfo=UTC)
        if start >= WINDOW_END:
            return
        end_month = quarter * 3 + 4
        end = datetime(year + end_month // 13, (end_month - 1) % 12 + 1, 1, tzinfo=UTC)
        yield f"{year}-Q{quarter + 1}", max(start, WINDOW_START), min(end, WINDOW_END)
        year, quarter = (year + 1, 0) if quarter == 3 else (year, quarter + 1)


def _mttr(incidents: Sequence[IncidentFact]) -> str:
    return f"{mean(i.duration_minutes for i in incidents):.0f}" if incidents else "-"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _reliability_review(
    label: str,
    start: datetime,
    end: datetime,
    incidents: Sequence[IncidentFact],
    deployments: Sequence[DeploymentFact],
) -> str:
    severities = Counter(i.severity for i in incidents)
    major = [i for i in incidents if i.severity in (Severity.SEV1, Severity.SEV2)]
    releases = [d for d in deployments if d.kind != "rollback"]
    rolled_back = sum(d.status is DeploymentStatus.ROLLED_BACK for d in releases)
    failed = sum(d.status is DeploymentStatus.FAILED for d in releases)
    change_failure = (rolled_back + failed) / len(releases) if releases else 0.0
    caused = sum(i.root_cause_deployment is not None for i in incidents)
    last_day = end - timedelta(days=1)
    summary = [
        f"- Incidents: {len(incidents)} ("
        + ", ".join(f"{s.value} {severities.get(s, 0)}" for s in Severity)
        + ").",
        f"- Mean time to resolve: {_mttr(incidents)} minutes; "
        f"SEV1 and SEV2 incidents: {_mttr(major)} minutes.",
        f"- Deployments: {len(releases)} releases ({rolled_back} rolled back, {failed} failed); "
        f"change failure rate {change_failure:.1%}.",
        f"- Incidents caused by a deployment: {caused}.",
    ]
    by_service: dict[str, list[IncidentFact]] = {}
    for incident in incidents:
        by_service.setdefault(incident.service_id, []).append(incident)
    service_rows = [
        [
            service,
            str(len(items)),
            str(sum(i.severity is Severity.SEV1 for i in items)),
            str(sum(i.severity is Severity.SEV2 for i in items)),
            _mttr(items),
        ]
        for service, items in sorted(by_service.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    ]
    categories = Counter(i.category.value for i in incidents).most_common(5)
    sev1 = sorted((i for i in incidents if i.severity is Severity.SEV1), key=lambda i: i.id)
    sev1_lines = [
        f"- {i.id} ({i.started_at:%Y-%m-%d}, {i.service_id}): {i.title}; "
        f"resolved in {i.duration_minutes} minutes."
        for i in sev1
    ] or ["- None."]
    return (
        "\n\n".join(
            [
                f"# Reliability Review {label}",
                f"Period covered: {start:%Y-%m-%d} to {last_day:%Y-%m-%d} (UTC). "
                "Audience: engineering and support management. Figures are computed from the "
                "incident and deployment records.",
                "## Summary",
                "\n".join(summary),
                "## Incidents by service",
                _table(
                    ["Service", "Incidents", "SEV1", "SEV2", "Mean time to resolve (min)"],
                    service_rows,
                ),
                "## Most frequent incident categories",
                _table(["Category", "Incidents"], [[c, str(n)] for c, n in categories]),
                "## SEV1 incidents",
                "\n".join(sev1_lines),
            ]
        )
        + "\n"
    )


def reliability_reports(
    incidents: Sequence[IncidentFact], deployments: Sequence[DeploymentFact]
) -> list[DocumentDraft]:
    drafts = []
    for number, (label, start, end) in enumerate(_quarters(), 1):
        in_period = [i for i in incidents if start <= i.started_at < end]
        released = [d for d in deployments if start <= d.deployed_at < end]
        published = min(end + timedelta(days=5), WINDOW_END)
        drafts.append(
            DocumentDraft(
                key=("report", label),
                id=f"RPT-{number:04d}",
                doc_type=DocumentType.REPORT,
                title=f"Reliability Review {label}",
                service_id=None,
                source_path=f"reports/reliability/{label.lower()}.md",
                author=REPORT_AUTHOR,
                tags=["report", "reliability", label],
                access_level=AccessLevel.MANAGER,
                created_at=published,
                updated_at=published,
                revision=1,
                content=_reliability_review(label, start, end, in_period, released),
            )
        )
    return drafts


def _access_register() -> str:
    role_rows = [
        [r.id, r.description, str(sum(p.role == r.id and p.active for p in PEOPLE))] for r in ROLES
    ]
    member_rows = [
        [p.username, p.team, p.role, "active" if p.active else "deactivated"]
        for p in sorted(PEOPLE, key=lambda p: (p.role, p.username))
    ]
    return (
        "\n\n".join(
            [
                "# Production Access Register",
                "Owner: security team. Reviewed every quarter. Lists who holds which role in "
                "OpsRAG and in the production tooling.",
                "## Roles",
                _table(["Role", "Scope", "Active members"], role_rows),
                "## Members",
                _table(["User", "Team", "Role", "Status"], member_rows),
                "## Review rules",
                "- Access is granted per role; individual exceptions are not allowed.\n"
                "- Leavers and ended contracts are deactivated within one business day; the "
                "accounts remain for audit history.\n"
                "- Admin membership needs approval from two members of the security team.",
            ]
        )
        + "\n"
    )


_BREAK_GLASS = """# Break-glass Production Access Procedure

Use only when a SEV1 incident cannot be mitigated with the normal tooling.

## Procedure

1. The incident commander requests break-glass access in #sec-breakglass, naming the incident id.
2. Two security engineers approve; the approval is recorded in the incident timeline.
3. Credentials are issued for 60 minutes and scoped to the database of one service.
4. Every command run with break-glass access is captured by the session recorder.
5. Within 24 hours the security team reviews the session and rotates the credentials used.

## Rules

- Break-glass access is never granted for convenience or for routine data requests.
- Credentials are never pasted into tickets, chat or documentation.
"""


def admin_policies() -> list[DocumentDraft]:
    created = datetime(2025, 1, 15, 10, 0, tzinfo=UTC)
    updated = datetime(2026, 7, 1, 9, 0, tzinfo=UTC)
    specs = [
        ("access-register", "Production Access Register", _access_register()),
        ("break-glass", "Break-glass Production Access Procedure", _BREAK_GLASS),
    ]
    return [
        DocumentDraft(
            key=("policy", slug),
            id=f"POL-{number:04d}",
            doc_type=DocumentType.POLICY,
            title=title,
            service_id=None,
            source_path=f"docs/policies/{slug}.md",
            author=POLICY_AUTHOR,
            tags=["policy", "access"],
            access_level=AccessLevel.ADMIN,
            created_at=created,
            updated_at=updated,
            revision=4,
            content=content,
        )
        for number, (slug, title, content) in enumerate(specs, 1)
    ]


__all__ = ["admin_policies", "reliability_reports"]
