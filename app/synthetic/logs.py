"""Structured application logs: healthy baselines, deployment events and the
abnormal patterns of every incident.

Each line carries the deployment and version that were live in its service at
that moment, and a pod name derived from the deployment, so logs alone reveal
when a new version started serving. Errors in cascading incidents share
``trace_id`` values across services.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from app.schemas.enums import (
    DeploymentStatus,
    DeploymentStrategy,
    IncidentCategory,
    LogLevel,
    Severity,
)
from app.synthetic.catalog import SERVICES, SERVICES_BY_ID, ServiceProfile
from app.synthetic.facts import DeploymentFact, IncidentFact
from app.synthetic.operations import LiveDeployments
from app.synthetic.timeline import WINDOW_END, WINDOW_START, hex_id

C = IncidentCategory
_HOUR_WEIGHTS = [
    0.4,
    0.3,
    0.3,
    0.3,
    0.3,
    0.4,
    0.6,
    0.8,
    1.0,
    1.1,
    1.2,
    1.3,
    1.4,
    1.4,
    1.4,
    1.5,
    1.6,
    1.8,
    1.9,
    1.9,
    1.7,
    1.4,
    1.0,
    0.6,
]
_LINES_PER_INCIDENT = {Severity.SEV1: 40, Severity.SEV2: 28, Severity.SEV3: 18, Severity.SEV4: 10}


@dataclass
class LogDraft:
    timestamp: datetime
    service_id: str
    level: LogLevel
    logger: str
    message: str
    trace_id: str | None = None
    span_id: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    deployment: DeploymentFact | None = None
    host: str = ""


class LogWriter:
    def __init__(self, live: LiveDeployments, rng: random.Random) -> None:
        self.live = live
        self.rng = rng
        self.lines: list[LogDraft] = []

    def pod(self, deployment: DeploymentFact | None, service_id: str) -> str:
        p = SERVICES_BY_ID[service_id]
        replicaset = deployment.commit_sha[:10] if deployment else "0000000000"
        index = self.rng.randrange(p.config.replicas)
        suffix = f"{random.Random(f'{replicaset}:{index}').getrandbits(20):05x}"
        return f"{service_id}-{replicaset}-{suffix}"

    def emit(
        self,
        moment: datetime,
        service_id: str,
        level: LogLevel,
        logger: str,
        message: str,
        attributes: dict[str, Any] | None = None,
        trace_id: str | None = None,
        deployment: DeploymentFact | None = None,
    ) -> LogDraft | None:
        moment = moment.replace(microsecond=(moment.microsecond // 1000) * 1000)
        if not (WINDOW_START <= moment < WINDOW_END):
            return None
        deployment = deployment or self.live.at(service_id, moment)
        line = LogDraft(
            moment,
            service_id,
            level,
            logger,
            message,
            trace_id,
            hex_id(self.rng, 16) if trace_id else None,
            attributes or {},
            deployment,
            self.pod(deployment, service_id),
        )
        self.lines.append(line)
        return line


def _route(endpoint: str) -> tuple[str, str]:
    method, _, path = endpoint.partition(" ")
    return method, path


# --- healthy baseline ---------------------------------------------------------------------


def _healthy_line(w: LogWriter, p: ServiceProfile, moment: datetime) -> None:
    rng = w.rng
    kinds = (
        ["request"] * 5
        + (["pool"] if p.postgres_db else [])
        + (["cache"] if p.redis_cluster else [])
        + (["consumer"] if p.consumes else [])
        + ["retry"]
    )
    kind = rng.choice(kinds)
    if kind == "request":
        ep = rng.choice(p.endpoints)
        status = 201 if ep.method == "POST" and rng.random() < 0.5 else 200
        duration = round(p.config.p99_latency_slo_ms * rng.uniform(0.05, 0.6), 1)
        w.emit(
            moment,
            p.id,
            LogLevel.INFO,
            f"{p.package}.api.routes",
            f"request.completed {ep.method} {ep.path} {status} in {duration}ms",
            {
                "http.method": ep.method,
                "http.route": ep.path,
                "http.status_code": status,
                "duration_ms": duration,
            },
            trace_id=hex_id(rng, 32),
        )
    elif kind == "pool":
        size, overflow = p.config.db_pool_size, p.config.db_max_overflow
        used = rng.randint(1, max(2, size // 2))
        w.emit(
            moment,
            p.id,
            LogLevel.INFO,
            f"{p.package}.db.database",
            f"db.pool.stats checked_out={used} size={size} overflow=0/{overflow} waiters=0",
            {"db.pool.checked_out": used, "db.pool.size": size, "db.pool.waiters": 0},
        )
    elif kind == "cache":
        ratio = round(rng.uniform(0.86, 0.98), 2)
        memory = rng.randint(38, 72)
        w.emit(
            moment,
            p.id,
            LogLevel.INFO,
            f"{p.package}.cache",
            f"cache.stats cluster={p.redis_cluster} hit_ratio={ratio} used_memory_pct={memory}",
            {"cache.hit_ratio": ratio, "redis.used_memory_pct": memory},
        )
    elif kind == "consumer":
        topic = rng.choice(p.consumes)
        batch, lag = rng.randint(20, p.config.kafka_max_poll_records), rng.randint(0, 900)
        w.emit(
            moment,
            p.id,
            LogLevel.INFO,
            f"{p.package}.events.consumer",
            f"consumer.batch_committed group={p.consumer_group} topic={topic} messages={batch} lag={lag}",
            {"kafka.topic": topic, "kafka.batch_size": batch, "kafka.lag": lag},
        )
    else:
        target = (
            rng.choice(p.http_dependencies).service
            if p.http_dependencies
            else (p.external[0].name if p.external else "postgres")
        )
        w.emit(
            moment,
            p.id,
            LogLevel.WARNING,
            "novacart_common.http",
            f"upstream.retry target={target} attempt=1 reason=ReadTimeout outcome=succeeded",
            {"upstream": target, "attempt": 1},
            trace_id=hex_id(rng, 32),
        )


def healthy_logs(w: LogWriter, per_day: int) -> None:
    rng = w.rng
    days = (WINDOW_END - WINDOW_START).days
    for p in SERVICES:
        start = w.live.first_time(p.id)
        for day in range(days):
            base = WINDOW_START + timedelta(days=day)
            for _ in range(per_day):
                hour = rng.choices(range(24), weights=_HOUR_WEIGHTS)[0]
                moment = base + timedelta(hours=hour, seconds=rng.uniform(0, 3600))
                if moment >= start:
                    _healthy_line(w, p, moment)


# --- deployments -----------------------------------------------------------------------------------


def deployment_logs(w: LogWriter, deployments: list[DeploymentFact]) -> None:
    for d in deployments:
        attrs = {"deployment.id": d.id, "version": d.version, "strategy": d.strategy.value}
        if d.kind == "rollback":
            assert d.rollback_of is not None
            w.emit(
                d.deployed_at,
                d.service_id,
                LogLevel.WARNING,
                "deployctl",
                f"deploy.rollback_started from={d.rollback_of.version} to={d.version} reverting={d.rollback_of.id}",
                attrs,
                deployment=d,
            )
        else:
            w.emit(
                d.deployed_at,
                d.service_id,
                LogLevel.INFO,
                "deployctl",
                f"deploy.started version={d.version} previous={d.previous_version} strategy={d.strategy.value}",
                attrs,
                deployment=d,
            )
        if d.strategy is DeploymentStrategy.CANARY:
            w.emit(
                d.deployed_at + timedelta(seconds=d.duration_seconds * 0.3),
                d.service_id,
                LogLevel.INFO if d.went_live else LogLevel.ERROR,
                "deployctl",
                (
                    "deploy.canary_step weight=5% analysis=passed"
                    if d.went_live
                    else "deploy.canary_step weight=5% analysis=failed error_rate_ratio=2.7"
                ),
                attrs,
                deployment=d,
            )
        end = d.deployed_at + timedelta(seconds=d.duration_seconds)
        if d.status is DeploymentStatus.FAILED:
            w.emit(
                end,
                d.service_id,
                LogLevel.ERROR,
                "deployctl",
                f"deploy.aborted version={d.version} reason=canary_analysis_failed",
                attrs,
                deployment=d,
            )
        else:
            w.emit(
                end,
                d.service_id,
                LogLevel.INFO,
                "deployctl",
                f"deploy.completed version={d.version} duration_s={d.duration_seconds}",
                attrs,
                deployment=d,
            )
        w.emit(
            end + timedelta(seconds=w.rng.uniform(5, 40)),
            d.service_id,
            LogLevel.INFO,
            f"{SERVICES_BY_ID[d.service_id].package}.main",
            f"service.started version={d.version}",
            {"version": d.version},
            deployment=d,
        )


# --- incidents -----------------------------------------------------------------------------------------


def _error_spec(incident: IncidentFact) -> tuple[str, str, str, int]:
    if incident.parent is None and incident.fault is not None:
        f = incident.fault
        return f.exception, f.error_message, f.endpoint, f.status_code
    if incident.cause is not None:
        c = incident.cause
        return c.exception, c.error_message, c.endpoint, c.status_code
    if incident.fault is not None:  # schema cascade into another consumer
        f = incident.fault
        return f.exception, f.error_message, f"kafka {f.details['topic']}", 0
    parent = SERVICES_BY_ID[incident.parent.service_id] if incident.parent else None
    p = SERVICES_BY_ID[incident.service_id]
    ep = p.endpoints[0]
    return (
        "UpstreamUnavailable",
        f"POST {parent.id if parent else 'upstream'} failed: 500 Internal Server Error",
        f"{ep.method} {ep.path}",
        502,
    )


def _signal_line(w: LogWriter, incident: IncidentFact, moment: datetime) -> None:
    rng, p = w.rng, SERVICES_BY_ID[incident.service_id]
    details = {
        **(incident.fault.details if incident.fault else {}),
        **(incident.cause.details if incident.cause else {}),
    }
    cat, pkg = incident.category, p.package
    if cat is C.DB_CONNECTION_EXHAUSTION and p.postgres_db:
        size = details.get("new_pool_size", p.config.db_pool_size)
        overflow = details.get("new_max_overflow", p.config.db_max_overflow)
        waiters = rng.randint(15, 400)
        w.emit(
            moment,
            p.id,
            LogLevel.WARNING,
            f"{pkg}.db.database",
            f"db.pool.saturated checked_out={size + overflow} size={size} overflow={overflow}/{overflow} waiters={waiters}",
            {
                "db.pool.checked_out": size + overflow,
                "db.pool.size": size,
                "db.pool.waiters": waiters,
            },
        )
    elif cat is C.REDIS_MEMORY_PRESSURE and p.redis_cluster:
        pct = round(rng.uniform(93, 100), 1)
        w.emit(
            moment,
            p.id,
            LogLevel.WARNING,
            f"{pkg}.cache",
            f"cache.stats cluster={p.redis_cluster} used_memory_pct={pct} evicted_keys={rng.randint(1000, 90000)}",
            {"redis.used_memory_pct": pct},
        )
    elif cat is C.KAFKA_CONSUMER_LAG and p.consumes:
        topic = details.get("topic") or rng.choice(p.consumes)
        lag = rng.randint(20_000, 2_000_000)
        w.emit(
            moment,
            p.id,
            LogLevel.WARNING,
            f"{pkg}.events.consumer",
            f"consumer.lag group={p.consumer_group} topic={topic} lag={lag}",
            {"kafka.topic": topic, "kafka.lag": lag},
        )
    elif cat is C.MEMORY_LEAK:
        w.emit(
            moment,
            p.id,
            LogLevel.ERROR,
            "k8s.events",
            f"pod OOMKilled container={p.id} limit={p.config.memory} restart_count={rng.randint(2, 40)}",
            {"k8s.reason": "OOMKilled", "k8s.memory_limit": p.config.memory},
        )
    elif cat is C.CPU_SATURATION:
        pct = rng.randint(55, 95)
        w.emit(
            moment,
            p.id,
            LogLevel.WARNING,
            "k8s.events",
            f"container {p.id} CPU throttled in {pct}% of periods (limit {p.config.cpu})",
            {"k8s.cpu_throttled_pct": pct},
        )
    elif cat is C.DATABASE_DEADLOCK:
        w.emit(
            moment,
            p.id,
            LogLevel.ERROR,
            f"{pkg}.db.repository",
            f"deadlock detected on {details.get('table', 'table')}: Process {rng.randint(1000, 9999)} waits for "
            f"ShareLock on transaction {rng.randint(10**6, 10**7)}; blocked by process {rng.randint(1000, 9999)}",
            {"db.error": "DeadlockDetected"},
        )
    elif cat in {C.API_TIMEOUT, C.NETWORK_TIMEOUT}:
        target = details.get("upstream") or (
            p.http_dependencies[0].service if p.http_dependencies else "postgres"
        )
        w.emit(
            moment,
            p.id,
            LogLevel.WARNING,
            "novacart_common.http",
            f"upstream.timeout target={target} timeout_s={details.get('new_timeout', p.config.http_timeout_seconds)} "
            f"attempt={rng.randint(1, 3)}",
            {"upstream": target},
            trace_id=hex_id(rng, 32),
        )
    elif cat is C.AUTHENTICATION_FAILURE:
        kid = details.get("kid", f"key-{hex_id(rng, 8)}")
        w.emit(
            moment,
            p.id,
            LogLevel.WARNING,
            f"{pkg}.auth_middleware" if p.id == "api-gateway" else f"{pkg}.tokens",
            f"auth.token_rejected kid={kid} reason=signature_or_claims_invalid",
            {"auth.kid": kid},
            trace_id=hex_id(rng, 32),
        )
    elif cat is C.RATE_LIMITING:
        w.emit(
            moment,
            p.id,
            LogLevel.INFO,
            f"{pkg}.rate_limiter" if p.id == "api-gateway" else "novacart_common.http",
            "ratelimit.rejected client_id=mobile-app limit_rps=50"
            if p.id == "api-gateway"
            else f"provider throttled request status=429 provider={details.get('provider', 'provider')}",
            {"http.status_code": 429},
            trace_id=hex_id(rng, 32),
        )
    elif cat is C.DEPENDENCY_FAILURE:
        target = (
            incident.parent.service_id
            if incident.parent
            else details.get("upstream") or details.get("provider", "upstream")
        )
        w.emit(
            moment,
            p.id,
            LogLevel.WARNING,
            "novacart_common.circuit_breaker",
            f"circuit_breaker.opened target={target} failure_ratio={round(rng.uniform(0.5, 0.95), 2)}",
            {"upstream": target},
        )
    elif cat is C.CACHE_INVALIDATION:
        w.emit(
            moment,
            p.id,
            LogLevel.WARNING,
            f"{pkg}.cache",
            "cache.stale_read cached_version older than source_of_truth",
            {"cache.stale": True},
        )


def incident_logs(w: LogWriter, incident: IncidentFact) -> None:
    rng = w.rng
    p = SERVICES_BY_ID[incident.service_id]
    exception, message, endpoint, status = _error_spec(incident)
    count = _LINES_PER_INCIDENT[incident.severity]
    span = max(60.0, (incident.resolved_at - incident.started_at).total_seconds())
    # Precursors: gradual problems are visible before the incident is declared.
    deployment = incident.root_cause_deployment
    if (
        deployment is not None
        and incident.parent is None
        and deployment.service_id == incident.service_id
        and incident.category
        in {C.MEMORY_LEAK, C.REDIS_MEMORY_PRESSURE, C.DB_CONNECTION_EXHAUSTION}
    ):
        lead = (incident.started_at - deployment.deployed_at).total_seconds()
        for k in range(1, 5):
            _signal_line(w, incident, deployment.deployed_at + timedelta(seconds=lead * k / 5))
    for _ in range(count):
        moment = incident.started_at + timedelta(seconds=rng.uniform(0, span))
        if rng.random() < 0.3:
            _signal_line(w, incident, moment)
            continue
        trace = hex_id(rng, 32)
        if endpoint.startswith("kafka "):
            topic = endpoint.removeprefix("kafka ")
            if status == 0 and exception == "ConsumerLag":
                _signal_line(w, incident, moment)
                continue
            w.emit(
                moment,
                p.id,
                LogLevel.ERROR,
                f"{p.package}.events.consumer",
                f"consumer.handler_failed topic={topic}: {exception}: {message}",
                {"kafka.topic": topic, "error.type": exception},
                trace_id=trace,
            )
            continue
        method, route = _route(endpoint)
        if status == 200:
            w.emit(
                moment,
                p.id,
                LogLevel.WARNING,
                f"{p.package}.api.routes",
                f"request.completed {method} {route} 200 served stale data: {message}",
                {"http.route": route, "http.status_code": 200, "error.type": exception},
                trace_id=trace,
            )
            continue
        duration = round(rng.uniform(0.8, 1.0) * 1000 * (3.0 if status in {500, 504} else 0.2), 1)
        level = LogLevel.ERROR if status >= 500 or status == 0 else LogLevel.WARNING
        w.emit(
            moment,
            p.id,
            level,
            f"{p.package}.api.routes",
            f"request.failed {method} {route} {status}: {exception}: {message}",
            {
                "http.method": method,
                "http.route": route,
                "http.status_code": status,
                "error.type": exception,
                "duration_ms": duration,
            },
            trace_id=trace,
        )
        if incident.parent is not None and incident.category is not C.SCHEMA_MISMATCH:
            # The same request, as seen by the upstream that caused the failure.
            parent_exception, parent_message, parent_endpoint, parent_status = _error_spec(
                incident.parent
            )
            parent = SERVICES_BY_ID[incident.parent.service_id]
            if not parent_endpoint.startswith("kafka "):
                parent_method, parent_route = _route(parent_endpoint)
                w.emit(
                    moment - timedelta(milliseconds=rng.randint(5, 900)),
                    parent.id,
                    LogLevel.ERROR,
                    f"{parent.package}.api.routes",
                    f"request.failed {parent_method} {parent_route} {parent_status}: {parent_exception}: {parent_message}",
                    {
                        "http.method": parent_method,
                        "http.route": parent_route,
                        "http.status_code": parent_status,
                        "error.type": parent_exception,
                        "caller": p.id,
                    },
                    trace_id=trace,
                )
    # A healthy request shortly after recovery.
    ep = p.endpoints[0]
    w.emit(
        incident.resolved_at + timedelta(minutes=rng.uniform(1, 6)),
        p.id,
        LogLevel.INFO,
        f"{p.package}.api.routes",
        f"request.completed {ep.method} {ep.path} 200 in {round(rng.uniform(5, 60), 1)}ms",
        {"http.method": ep.method, "http.route": ep.path, "http.status_code": 200},
        trace_id=hex_id(rng, 32),
    )


def generate_logs(
    deployments: list[DeploymentFact],
    incidents: list[IncidentFact],
    live: LiveDeployments,
    rng: random.Random,
    healthy_per_day: int,
) -> list[LogDraft]:
    writer = LogWriter(live, rng)
    healthy_logs(writer, healthy_per_day)
    deployment_logs(writer, deployments)
    for incident in incidents:
        incident_logs(writer, incident)
    writer.lines.sort(key=lambda line: (line.timestamp, line.service_id, line.message))
    return writer.lines
