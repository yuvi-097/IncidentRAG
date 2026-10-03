"""Year-long release simulation per service.

For every service the simulator walks forward in time: regular deployments at
business hours (with a Black Friday change freeze), each shipping a few pull
requests. Some deployments carry a faulty PR; that fault produces an incident,
which is mitigated by a rollback (fix PR ships in a later deployment) or by a
hotfix deployment. Pinned *anchor* storylines guarantee a few fully specified
chains (e.g. payment-service v2.8.1).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import UTC, datetime, time, timedelta
from typing import Any

from app.schemas.enums import (
    DeploymentStatus,
    DeploymentStrategy,
    IncidentCategory,
    ServiceTier,
    Severity,
)
from app.synthetic.catalog import SERVICES_BY_ID, ServiceProfile, team_members
from app.synthetic.changes import (
    FaultPlan,
    Repo,
    build_fault,
    choose_fault,
    choose_maintenance,
)
from app.synthetic.facts import DeploymentFact, IncidentFact, PullRequestFact, Version
from app.synthetic.text import slugify

WINDOW_START = datetime(2025, 9, 1, tzinfo=UTC)
WINDOW_END = datetime(2026, 9, 1, tzinfo=UTC)
CHANGE_FREEZE = (datetime(2025, 11, 24, tzinfo=UTC), datetime(2025, 12, 2, tzinfo=UTC))


@dataclass(frozen=True)
class TrafficEvent:
    name: str
    start: datetime
    days: int
    multiplier: float

    @property
    def end(self) -> datetime:
        return self.start + timedelta(days=self.days)


TRAFFIC_EVENTS: tuple[TrafficEvent, ...] = (
    TrafficEvent("Black Friday", datetime(2025, 11, 28, tzinfo=UTC), 1, 4.5),
    TrafficEvent("Cyber Monday", datetime(2025, 12, 1, tzinfo=UTC), 1, 4.0),
    TrafficEvent("Holiday Gift Rush", datetime(2025, 12, 15, tzinfo=UTC), 5, 2.2),
    TrafficEvent("New Year Sale", datetime(2026, 1, 2, tzinfo=UTC), 3, 2.0),
    TrafficEvent("Spring Sale", datetime(2026, 3, 20, tzinfo=UTC), 3, 2.4),
    TrafficEvent("Summer Mega Sale", datetime(2026, 7, 15, tzinfo=UTC), 2, 3.5),
    TrafficEvent("Back to School", datetime(2026, 8, 20, tzinfo=UTC), 4, 1.8),
)


def traffic_event_at(moment: datetime) -> TrafficEvent | None:
    return next((e for e in TRAFFIC_EVENTS if e.start <= moment < e.end), None)


def in_freeze(moment: datetime) -> bool:
    return CHANGE_FREEZE[0] <= moment < CHANGE_FREEZE[1]


def hex_id(rng: random.Random, length: int) -> str:
    return f"{rng.getrandbits(4 * length):0{length}x}"


def minutes(value: float) -> timedelta:
    return timedelta(minutes=value)


def business_slot(earliest: datetime, rng: random.Random) -> datetime:
    """A random deploy time on or after ``earliest``: weekdays 09:30-16:30 UTC,
    Fridays until 12:30, never inside the change freeze."""
    day = datetime.combine(earliest.date(), time(0), tzinfo=UTC)
    for _ in range(90):
        weekday = day.weekday()
        if weekday < 5 and not in_freeze(day):
            start = day + timedelta(hours=9, minutes=30)
            end = day + timedelta(hours=12 if weekday == 4 else 16, minutes=30)
            low = max(start, earliest)
            if low < end:
                offset = rng.uniform(0, (end - low).total_seconds())
                return (low + timedelta(seconds=offset)).replace(microsecond=0)
        day += timedelta(days=1)
    raise RuntimeError(f"no business slot after {earliest}")


# --- severity, timing and metrics -----------------------------------------------------

_SEVERITY_WEIGHTS = {
    ServiceTier.TIER_0: (0.10, 0.36, 0.36, 0.18),
    ServiceTier.TIER_1: (0.03, 0.24, 0.45, 0.28),
    ServiceTier.TIER_2: (0.01, 0.12, 0.45, 0.42),
}
_SEVERITIES = (Severity.SEV1, Severity.SEV2, Severity.SEV3, Severity.SEV4)


def choose_severity(p: ServiceProfile, rng: random.Random) -> Severity:
    return rng.choices(_SEVERITIES, weights=_SEVERITY_WEIGHTS[p.tier])[0]


def detection_delay(severity: Severity, alert_key: str | None, rng: random.Random) -> timedelta:
    if alert_key is None:  # noticed through customer reports / support tickets
        return minutes(rng.randint(20, 150))
    low, high = {
        Severity.SEV1: (1, 6),
        Severity.SEV2: (2, 10),
        Severity.SEV3: (4, 25),
        Severity.SEV4: (8, 60),
    }[severity]
    return minutes(rng.randint(low, high))


def operational_duration(severity: Severity, rng: random.Random) -> timedelta:
    median = {Severity.SEV1: 70, Severity.SEV2: 105, Severity.SEV3: 165, Severity.SEV4: 250}[
        severity
    ]
    value = rng.lognormvariate(math.log(median), 0.55)
    return minutes(int(min(1440, max(15, value))))


def incident_metrics(
    p: ServiceProfile, severity: Severity, category: IncidentCategory, rng: random.Random
) -> dict[str, Any]:
    err_low, err_high = {
        Severity.SEV1: (25, 75),
        Severity.SEV2: (8, 30),
        Severity.SEV3: (2, 10),
        Severity.SEV4: (0.5, 3),
    }[severity]
    p99_low, p99_high = {
        Severity.SEV1: (6, 18),
        Severity.SEV2: (3, 8),
        Severity.SEV3: (1.8, 4),
        Severity.SEV4: (1.2, 2.5),
    }[severity]
    failed_low, failed_high = {
        Severity.SEV1: (20_000, 160_000),
        Severity.SEV2: (3_000, 30_000),
        Severity.SEV3: (300, 5_000),
        Severity.SEV4: (20, 600),
    }[severity]
    metrics: dict[str, Any] = {
        "peak_error_rate_pct": round(rng.uniform(err_low, err_high), 1),
        "p99_latency_ms": int(p.config.p99_latency_slo_ms * rng.uniform(p99_low, p99_high)),
        "failed_requests": rng.randint(failed_low, failed_high),
    }
    C = IncidentCategory
    if category is C.CACHE_INVALIDATION:
        metrics["peak_error_rate_pct"] = round(rng.uniform(0.0, 0.8), 1)
        metrics["stale_reads_estimated"] = rng.randint(2_000, 90_000)
    elif category is C.KAFKA_CONSUMER_LAG:
        metrics["max_consumer_lag_messages"] = rng.randint(40_000, 2_500_000)
    elif category is C.REDIS_MEMORY_PRESSURE:
        metrics["redis_memory_used_pct"] = round(rng.uniform(92, 100), 1)
        metrics["evicted_keys"] = rng.randint(10_000, 4_000_000)
    elif category is C.MEMORY_LEAK:
        metrics["pod_restarts"] = rng.randint(4, 60)
    elif category is C.CPU_SATURATION:
        metrics["cpu_throttled_pct"] = rng.randint(55, 95)
    elif category is C.RATE_LIMITING:
        metrics["http_429_rate_pct"] = round(rng.uniform(5, 60), 1)
    elif category is C.AUTHENTICATION_FAILURE:
        metrics["http_401_rate_pct"] = round(rng.uniform(4, 55), 1)
    elif category is C.DATABASE_DEADLOCK:
        metrics["deadlocks_detected"] = rng.randint(15, 900)
    elif category is C.DB_CONNECTION_EXHAUSTION:
        metrics["max_pool_waiters"] = rng.randint(20, 400)
    return metrics


def fault_alert_key(fault: FaultPlan) -> str | None:
    by_kind = {
        "pool_shrink": "db_pool",
        "session_leak": "db_pool",
        "request_log_leak": "restarts",
        "orm_column_rename": "db_errors",
        "event_field_rename": "consumer_errors",
        "invalidation_expire": None,
        "ttl_removed": "redis_memory",
        "auth_fault": "auth_failures",
        "timeout_reduced": "latency",
        "gateway_rate_limit": "rate_limited",
        "retry_storm": "provider",
        "consumer_sequential": "consumer_lag",
        "lock_order": "db_deadlock",
        "debug_logging": "cpu",
    }
    if fault.kind == "regression" and fault.service_id == "inventory-service":
        return None  # stock drift is found by reconciliation reports, not alerts
    return by_kind.get(fault.kind, "http_5xx")


# --- anchors ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Anchor:
    """A pinned storyline: this exact deployment carries this fault."""

    name: str
    service_id: str
    deploy_at: datetime
    fault_kind: str
    severity: Severity
    remediation: str  # "rollback" | "hotfix"
    version: Version | None = None
    onset_minutes: int | None = None


ANCHORS: tuple[Anchor, ...] = (
    Anchor(
        "payment-500s-v2.8.1",
        "payment-service",
        datetime(2026, 6, 16, 13, 40, tzinfo=UTC),
        "pool_shrink",
        Severity.SEV1,
        "rollback",
        version=(2, 8, 1),
        onset_minutes=205,
    ),
    Anchor(
        "gateway-jwks-401s",
        "api-gateway",
        datetime(2026, 3, 10, 10, 20, tzinfo=UTC),
        "auth_fault",
        Severity.SEV2,
        "hotfix",
        onset_minutes=1510,
    ),
    Anchor(
        "inventory-wms-drift",
        "inventory-service",
        datetime(2025, 11, 12, 11, 5, tzinfo=UTC),
        "regression",
        Severity.SEV2,
        "rollback",
        onset_minutes=95,
    ),
    Anchor(
        "cart-redis-oom",
        "cart-service",
        datetime(2026, 1, 21, 14, 30, tzinfo=UTC),
        "ttl_removed",
        Severity.SEV2,
        "rollback",
        onset_minutes=3900,
    ),
    Anchor(
        "product-event-rename",
        "product-service",
        datetime(2026, 4, 22, 10, 0, tzinfo=UTC),
        "event_field_rename",
        Severity.SEV3,
        "hotfix",
        onset_minutes=6,
    ),
)


# --- simulation ---------------------------------------------------------------------------


@dataclass
class ServiceTimeline:
    service_id: str
    deployments: list[DeploymentFact] = field(default_factory=list)
    pull_requests: list[PullRequestFact] = field(default_factory=list)
    incidents: list[IncidentFact] = field(default_factory=list)
    anchor_deployments: dict[str, DeploymentFact] = field(default_factory=dict)
    minor_offset: int = 0


_FAULT_LABELS = {
    "config": ["config"],
    "regression": ["feature"],
    "pool_shrink": ["performance"],
    "session_leak": ["performance"],
    "request_log_leak": ["observability"],
    "orm_column_rename": ["refactor"],
    "event_field_rename": ["api"],
    "invalidation_expire": ["performance"],
    "ttl_removed": ["refactor"],
    "auth_fault": ["security"],
    "timeout_reduced": ["resilience"],
    "gateway_rate_limit": ["security"],
    "retry_storm": ["resilience"],
    "consumer_sequential": ["correctness"],
    "lock_order": ["refactor"],
    "debug_logging": ["observability"],
}

_GRADUAL_CLEANUP = {IncidentCategory.REDIS_MEMORY_PRESSURE, IncidentCategory.CACHE_INVALIDATION}


class ServiceSimulator:
    def __init__(
        self,
        p: ServiceProfile,
        repo: Repo,
        rng: random.Random,
        fault_rate: float,
        anchors: list[Anchor],
    ) -> None:
        self.p = p
        self.repo = repo
        self.rng = rng
        self.fault_rate = fault_rate
        self.anchors = sorted(anchors, key=lambda a: a.deploy_at)
        self.team = [m.username for m in team_members(p.team)]
        self.sres = [m.username for m in team_members("sre")]
        self.out = ServiceTimeline(p.id)
        self.pending: list[PullRequestFact] = []  # merged but not yet shipped
        self.max_version: Version | None = None
        self.live: DeploymentFact | None = None
        self.last_time = WINDOW_START
        self.mean_gap_days = {
            ServiceTier.TIER_0: 10.5,
            ServiceTier.TIER_1: 12.0,
            ServiceTier.TIER_2: 14.0,
        }[p.tier]

    # -- building blocks

    def _next_version(self, minor: bool) -> Version:
        if self.max_version is None:
            version = (self.p.start_version[0], 0, 0)
        elif minor:
            version = (self.max_version[0], self.max_version[1] + 1, 0)
        else:
            version = (self.max_version[0], self.max_version[1], self.max_version[2] + 1)
        self.max_version = version
        return version

    def _pr(
        self,
        title: str,
        rationale: str,
        labels: list[str],
        edits: tuple,
        merged_at: datetime,
        author: str | None = None,
        reviewers: list[str] | None = None,
        opened_after: datetime | None = None,
    ) -> PullRequestFact:
        rng = self.rng
        author = author or rng.choice(self.team)
        others = [m for m in self.team if m != author] or self.sres
        reviewers = reviewers or rng.sample(others, k=min(len(others), rng.choice([1, 1, 2])))
        opened = merged_at - timedelta(hours=rng.uniform(1.0, 60.0))
        if opened_after is not None:
            opened = max(opened, opened_after)
        pr = PullRequestFact(
            service_id=self.p.id,
            title=title,
            rationale=rationale,
            author=author,
            reviewers=reviewers,
            labels=labels,
            opened_at=opened,
            merged_at=merged_at,
            head_branch=f"{author.split('.')[0]}/{slugify(title)[:48].rstrip('-')}",
            merge_commit_sha=hex_id(rng, 40),
            edits=edits,
        )
        self.out.pull_requests.append(pr)
        return pr

    def _maintenance_prs(
        self, after: datetime, before: datetime, count: int
    ) -> list[PullRequestFact]:
        span = (before - after).total_seconds()
        times = sorted(
            after + timedelta(seconds=self.rng.uniform(0.1 * span, span - 1800))
            for _ in range(count)
        )
        prs = []
        for merged_at in times:
            change = choose_maintenance(self.p, self.repo, self.rng)
            edits, labels = list(change.edits), list(change.labels)
            if self.rng.random() < 0.35:  # bundle a second, unrelated file (as real PRs often do)
                extra = choose_maintenance(self.p, self.repo, self.rng)
                if all(e.path not in {x.path for x in edits} for e in extra.edits):
                    edits += extra.edits
                    labels += [label for label in extra.labels if label not in labels]
            prs.append(
                self._pr(
                    change.title,
                    change.rationale,
                    labels,
                    tuple(edits),
                    merged_at.replace(microsecond=0),
                    opened_after=None,
                )
            )
        return prs

    def _deploy(
        self,
        at: datetime,
        kind: str,
        version: Version,
        prs: list[PullRequestFact],
        author: str,
        status: DeploymentStatus = DeploymentStatus.SUCCEEDED,
        rollback_of: DeploymentFact | None = None,
        commit_sha: str | None = None,
    ) -> DeploymentFact:
        rng = self.rng
        if kind == "rollback":
            strategy, duration = DeploymentStrategy.ROLLING, rng.randint(120, 300)
        elif self.p.tier is ServiceTier.TIER_0 and kind == "regular":
            strategy, duration = DeploymentStrategy.CANARY, rng.randint(900, 2400)
        else:
            strategy, duration = DeploymentStrategy.ROLLING, rng.randint(180, 480)
        deployment = DeploymentFact(
            service_id=self.p.id,
            kind=kind,
            rel_version=version,
            previous_rel_version=self.live.rel_version if self.live else None,
            commit_sha=commit_sha or (prs[-1].merge_commit_sha if prs else hex_id(rng, 40)),
            deployed_at=at,
            author=author,
            strategy=strategy,
            status=status,
            duration_seconds=duration,
            rollback_of=rollback_of,
        )
        if status is DeploymentStatus.FAILED:
            deployment.carried_pull_requests = prs
        else:
            deployment.pull_requests = prs
            for pr in prs:
                pr.deployment = deployment
        self.out.deployments.append(deployment)
        if deployment.went_live:
            self.live = deployment
        return deployment

    def _commander(self, severity: Severity, service_id: str) -> str:
        if severity in {Severity.SEV1, Severity.SEV2}:
            return self.rng.choice(self.sres)
        return self.rng.choice([m.username for m in team_members(SERVICES_BY_ID[service_id].team)])

    # -- faults

    def _fault_incident(
        self,
        deployment: DeploymentFact,
        culprit: PullRequestFact,
        fault: FaultPlan,
        severity: Severity,
        remediation: str,
        onset: int | None,
        anchor: str | None,
    ) -> IncidentFact:
        rng, p = self.rng, self.p
        prior_live = next(d for d in reversed(self.out.deployments[:-1]) if d.went_live)
        started = deployment.deployed_at + minutes(
            onset if onset is not None else rng.randint(*fault.onset_minutes)
        )
        alert_key = fault_alert_key(fault)
        detected = started + detection_delay(severity, alert_key, rng)
        symptomatic = fault.symptomatic_service_id
        commander = self._commander(severity, symptomatic)
        incident = IncidentFact(
            service_id=symptomatic,
            category=fault.category,
            severity=severity,
            started_at=started,
            detected_at=detected,
            resolved_at=detected,
            commander=commander,
            alert_key=alert_key,
            metrics=incident_metrics(SERVICES_BY_ID[symptomatic], severity, fault.category, rng),
            root_cause_service_id=p.id,
            fault=fault,
            root_cause_deployment=deployment,
            root_cause_pr=culprit,
            traffic_event=(e.name if (e := traffic_event_at(started)) else None),
            anchor=anchor,
        )
        if remediation == "rollback":
            rollback = self._deploy(
                detected + minutes(rng.randint(12, 45)),
                "rollback",
                prior_live.rel_version,
                [],
                commander if commander in self.sres else rng.choice(self.sres),
                rollback_of=deployment,
                commit_sha=prior_live.commit_sha,
            )
            deployment.status = DeploymentStatus.ROLLED_BACK
            fix_merged = rollback.deployed_at + timedelta(hours=rng.uniform(3, 40))
            fix = self._pr(
                fault.fix_title,
                "",
                ["bugfix", "incident"],
                fault.fix_edits,
                fix_merged.replace(microsecond=0),
                author=culprit.author if rng.random() < 0.6 else None,
                opened_after=rollback.deployed_at,
            )
            self.pending.append(fix)
            remediation_deployment = rollback
        else:
            fix_opened = detected + minutes(rng.randint(5, 20))
            fix_merged = fix_opened + minutes(rng.randint(20, 80))
            fix = self._pr(
                fault.fix_title,
                "",
                ["bugfix", "incident", "hotfix"],
                fault.fix_edits,
                fix_merged.replace(microsecond=0),
                author=culprit.author if rng.random() < 0.6 else None,
                reviewers=[rng.choice(self.sres)],
                opened_after=fix_opened,
            )
            fix.opened_at = fix_opened.replace(microsecond=0)
            remediation_deployment = self._deploy(
                (fix_merged + minutes(rng.randint(8, 25))).replace(microsecond=0),
                "hotfix",
                self._next_version(minor=False),
                [fix],
                fix.author,
            )
        fix.fixes = incident
        incident.fix_pr = fix
        incident.remediation = remediation_deployment
        recovered = remediation_deployment.deployed_at + timedelta(
            seconds=remediation_deployment.duration_seconds
        )
        needs_cleanup = fault.category in _GRADUAL_CLEANUP or (
            fault.kind == "regression"
            and fault.service_id == "inventory-service"  # stock reconciliation
        )
        cleanup = minutes(rng.randint(30, 150)) if needs_cleanup else minutes(0)
        incident.resolved_at = (recovered + minutes(rng.randint(5, 25)) + cleanup).replace(
            microsecond=0
        )
        self.last_time = max(self.last_time, remediation_deployment.deployed_at, fix.merged_at)
        self.out.incidents.append(incident)
        return incident

    def _regular_deployment(
        self,
        at: datetime,
        minor: bool | None = None,
        fault: FaultPlan | None = None,
        severity: Severity | None = None,
        remediation: str | None = None,
        onset: int | None = None,
        anchor: str | None = None,
        allow_failure: bool = True,
    ) -> DeploymentFact:
        rng = self.rng
        first = self.live is None
        count = 1 if first else rng.choice([1, 1, 2, 2, 2, 3])
        prs = self.pending + self._maintenance_prs(self.last_time, at, count)
        self.pending = []
        prs.sort(key=lambda pr: pr.merged_at)
        if minor is None:
            minor = not first and rng.random() < 0.16
        version = self._next_version(minor)
        author = prs[-1].author
        if allow_failure and not first and fault is None and rng.random() < 0.04:
            failed = self._deploy(
                at, "regular", version, prs, author, status=DeploymentStatus.FAILED
            )
            self.pending = list(prs)  # re-shipped by the next deployment
            self.last_time = at + timedelta(seconds=failed.duration_seconds)
            return failed
        if fault is not None:
            culprit = (
                prs[-1]
                if prs[-1].fixes is None
                else self._maintenance_prs(self.last_time, at, 1)[0]
            )
            if culprit not in prs:
                prs.append(culprit)
                prs.sort(key=lambda pr: pr.merged_at)
            culprit.title, culprit.rationale, culprit.edits = (
                fault.pr_title,
                fault.pr_rationale,
                fault.edits,
            )
            culprit.labels = _FAULT_LABELS.get(fault.kind, ["feature"])
            culprit.fault = fault
            culprit.head_branch = (
                f"{culprit.author.split('.')[0]}/{slugify(fault.pr_title)[:48].rstrip('-')}"
            )
        deployment = self._deploy(at, "regular", version, prs, author)
        self.last_time = at + timedelta(seconds=deployment.duration_seconds)
        if anchor:
            self.out.anchor_deployments[anchor] = deployment
        if fault is not None:
            assert severity is not None and remediation is not None
            self._fault_incident(deployment, culprit, fault, severity, remediation, onset, anchor)
        return deployment

    def _gap(self) -> timedelta:
        days = self.rng.lognormvariate(math.log(self.mean_gap_days), 0.45)
        return timedelta(days=min(30.0, max(3.0, days)))

    def run(self) -> ServiceTimeline:
        rng, p = self.rng, self.p
        at = business_slot(WINDOW_START + timedelta(hours=rng.uniform(2, 30)), rng)
        self._regular_deployment(at, minor=False, allow_failure=False)
        while True:
            at = business_slot(self.last_time + self._gap(), rng)
            anchor = self.anchors[0] if self.anchors else None
            if anchor is not None and at >= anchor.deploy_at - timedelta(days=2):
                self.anchors.pop(0)
                self._run_anchor(anchor)
                continue
            if at > WINDOW_END - timedelta(days=4):
                break
            near_anchor = anchor is not None and at >= anchor.deploy_at - timedelta(days=9)
            if not near_anchor and rng.random() < self.fault_rate:
                fault = choose_fault(p, self.repo, rng)
                remediation = "hotfix" if rng.random() < 0.4 else "rollback"
                self._regular_deployment(
                    at, fault=fault, severity=choose_severity(p, rng), remediation=remediation
                )
            else:
                self._regular_deployment(at, allow_failure=not near_anchor)
        if self.pending:  # ship merged fixes still waiting for a release
            at = business_slot(self.last_time + timedelta(hours=20), rng)
            if at < WINDOW_END:
                self._regular_deployment(at, allow_failure=False)
        if self.anchors:
            raise RuntimeError(f"unplaced anchors for {p.id}: {[a.name for a in self.anchors]}")
        return self.out

    def _run_anchor(self, anchor: Anchor) -> None:
        fault = build_fault(anchor.fault_kind, self.p, self.repo, self.rng)
        if fault is None:
            raise ValueError(
                f"anchor {anchor.name}: fault {anchor.fault_kind} does not apply to {self.p.id}"
            )
        if anchor.version is not None:
            pre = business_slot(
                max(self.last_time + timedelta(hours=3), anchor.deploy_at - timedelta(days=2)),
                self.rng,
            )
            if pre >= anchor.deploy_at - timedelta(hours=2):
                raise RuntimeError(f"anchor {anchor.name}: no room for the preceding minor release")
            self._regular_deployment(pre, minor=True, allow_failure=False)
        elif self.last_time >= anchor.deploy_at:
            raise RuntimeError(f"anchor {anchor.name}: timeline already past {anchor.deploy_at}")
        self._regular_deployment(
            anchor.deploy_at,
            minor=False if anchor.version else None,
            fault=fault,
            severity=anchor.severity,
            remediation=anchor.remediation,
            onset=anchor.onset_minutes,
            anchor=anchor.name,
        )


def finalize_versions(timeline: ServiceTimeline, p: ServiceProfile) -> None:
    """Turn relative versions into release strings, honouring version anchors."""
    offset = p.start_version[1]
    for anchor in ANCHORS:
        if anchor.service_id == p.id and anchor.version is not None:
            deployment = timeline.anchor_deployments[anchor.name]
            major, minor, patch = deployment.rel_version
            if (major, patch) != (anchor.version[0], anchor.version[2]):
                raise RuntimeError(f"anchor {anchor.name}: simulated {deployment.rel_version}")
            offset = anchor.version[1] - minor
            if offset < 0:
                raise RuntimeError(f"anchor {anchor.name}: needs a negative starting minor version")
    timeline.minor_offset = offset

    def fmt(version: Version | None) -> str | None:
        return None if version is None else f"v{version[0]}.{version[1] + offset}.{version[2]}"

    for deployment in timeline.deployments:
        deployment.version = fmt(deployment.rel_version) or ""
        deployment.previous_version = fmt(deployment.previous_rel_version)


def simulate(repo: Repo, seed: int, fault_rate: float) -> list[ServiceTimeline]:
    timelines = []
    for index, p in enumerate(SERVICES_BY_ID.values()):
        rng = random.Random(f"{seed}:{index}:{p.id}")
        anchors = [a for a in ANCHORS if a.service_id == p.id]
        timeline = ServiceSimulator(p, repo, rng, fault_rate, anchors).run()
        finalize_versions(timeline, p)
        timelines.append(timeline)
    return timelines
