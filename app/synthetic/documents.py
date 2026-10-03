"""Runbooks, technical documentation and postmortems.

Documents are planned first (ids, titles, coverage) so incidents can link to
runbooks; bodies are rendered afterwards so they can cite real incident,
deployment and pull request ids.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from app.schemas.enums import (
    AccessLevel,
    DependencyCriticality,
    DocumentType,
    IncidentCategory,
    Severity,
)
from app.synthetic.catalog import (
    KAFKA_BOOTSTRAP,
    SERVICES,
    SERVICES_BY_ID,
    TOPIC_PARTITIONS,
    TOPIC_PRODUCERS,
    ServiceProfile,
    alert_names,
    consumers_of,
    dependents_of,
    team_members,
)
from app.synthetic.code_repo import (
    EVENT_FIELDS,
    TABLE_MODELS,
    database_url,
    dependency_url_setting,
    primary_table,
    redis_url,
    service_url,
)
from app.synthetic.facts import IncidentFact
from app.synthetic.text import bullet_list, markdown_table, slugify
from app.synthetic.timeline import WINDOW_END, WINDOW_START

C = IncidentCategory
D = DocumentType


@dataclass
class DocumentDraft:
    key: tuple[str, ...]
    id: str
    doc_type: DocumentType
    title: str
    service_id: str | None
    source_path: str
    author: str
    tags: list[str]
    access_level: AccessLevel = AccessLevel.ENGINEERING
    categories: tuple[IncidentCategory, ...] = ()
    link_services: tuple[str, ...] = ()
    created_at: datetime = WINDOW_START
    updated_at: datetime = WINDOW_START
    revision: int = 1
    content: str = ""
    render: Callable[[DocContext], str] | None = None


@dataclass
class DocContext:
    drafts: dict[tuple[str, ...], DocumentDraft]
    incidents_by_runbook: dict[str, list[IncidentFact]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def ref(self, *key: str | IncidentCategory) -> str:
        draft = self.drafts[tuple(str(k) for k in key)]
        return f"{draft.id} ({draft.title})"


def _h(title: str, body: str) -> str:
    return f"## {title}\n\n{body.strip()}\n"


def _steps(items: list[str]) -> str:
    return "\n".join(f"{i}. {item}" for i, item in enumerate(items, 1))


def _header(title: str, rows: list[tuple[str, str]]) -> str:
    return f"# {title}\n\n" + markdown_table(["Field", "Value"], [[k, v] for k, v in rows]) + "\n"


# --- generic category runbooks ------------------------------------------------------------

CATEGORY_RUNBOOKS: dict[IncidentCategory, dict[str, object]] = {
    C.DB_CONNECTION_EXHAUSTION: {
        "title": "Database Connection Pool Exhaustion",
        "when": "Requests fail with `QueuePool limit ... reached, connection timed out` or PgBouncer reports waiting clients.",
        "symptoms": [
            "HTTP 500 on database-backed endpoints after a fixed wait (the pool timeout)",
            "`*DBPoolSaturated` alert: checked-out connections at pool size for 5 minutes",
            "PgBouncer `SHOW POOLS` shows `cl_waiting` > 0",
        ],
        "diagnosis": [
            "Check whether the service was deployed in the last 24h (`deployctl history <service> --limit 5`). A pool change or a leaked session in a new release is the most common cause.",
            "Compare `db_pool_checked_out` with `db_pool_size + db_max_overflow` on the service dashboard. Pinned at the maximum means exhaustion; slowly growing means a leak.",
            "Look for long-running transactions: `SELECT pid, now() - xact_start AS age, state, query FROM pg_stat_activity WHERE datname = '<db>' ORDER BY age DESC LIMIT 20;`",
            "Check PgBouncer: `psql -p 6432 pgbouncer -c 'SHOW POOLS;'` (cl_waiting, sv_active, maxwait).",
            "Confirm capacity: replicas x (pool_size + max_overflow) must stay below PgBouncer `default_pool_size` for the database.",
        ],
        "mitigation": [
            "If the onset follows a deploy, roll back first (`deployctl rollback <service>`), then investigate.",
            "If a single query holds locks, terminate it: `SELECT pg_terminate_backend(<pid>);`",
            "If traffic-driven, scale out the service and raise PgBouncer `default_pool_size` within the database's `max_connections` budget.",
            "Restarting pods only helps for leaks and buys minutes; treat it as a stopgap.",
        ],
        "prevention": [
            "Pool size comes from configuration (`db_pool_size`), never hard-coded",
            "Every session is opened with `session_scope()` so it is closed on all paths",
        ],
    },
    C.REDIS_MEMORY_PRESSURE: {
        "title": "Redis Memory Pressure",
        "when": "`*RedisMemoryHigh` fires or writes fail with `OOM command not allowed when used memory > 'maxmemory'`.",
        "symptoms": [
            "Write errors from the cache or store layer",
            "Evicted keys spike and the cache hit ratio drops",
            "Latency rises as requests fall through to the database",
        ],
        "diagnosis": [
            "`redis-cli -h <cluster> INFO memory`: compare used_memory with maxmemory and check mem_fragmentation_ratio.",
            "`redis-cli -h <cluster> INFO keyspace`: compare total keys with keys that have an expiry. A growing gap means keys are written without a TTL.",
            "`redis-cli -h <cluster> --bigkeys` to find oversized values.",
            "Check the eviction policy (`CONFIG GET maxmemory-policy`). With `volatile-lru`, keys without a TTL are never evicted.",
            "Check recent deploys of the owning service for changes to cache writes.",
        ],
        "mitigation": [
            "Roll back a release that changed TTL handling.",
            "Delete keys without a TTL in batches (`SCAN` + `TTL` == -1 + `UNLINK`); never use `KEYS *` in production.",
            "Scale the cluster to the next node size if growth is organic.",
        ],
        "prevention": [
            "All cache writes set an explicit TTL",
            "Alert at 80% of maxmemory, not 95%",
        ],
    },
    C.KAFKA_CONSUMER_LAG: {
        "title": "Kafka Consumer Lag",
        "when": "`*ConsumerLagHigh` fires or downstream effects (emails, stock updates, search index) are delayed.",
        "symptoms": [
            "Lag grows continuously instead of oscillating",
            "Events are processed minutes or hours late",
            "Repeated rebalances in consumer logs",
        ],
        "diagnosis": [
            "`kafka-consumer-groups --bootstrap-server kafka-1:9092 --describe --group <group>`: is lag spread across all partitions or stuck on one?",
            "One stuck partition usually means a poison message. Look for repeated errors on the same offset.",
            "All partitions growing means throughput is too low: check processing time per message, batch concurrency, and recent deploys.",
            "Look for rebalance storms: frequent `Revoking previously assigned partitions` log lines during pod churn.",
        ],
        "mitigation": [
            "Poison message: copy it to `<topic>.dlq` and skip the offset (`--reset-offsets --shift-by 1` for that partition only).",
            "Throughput regression after a deploy: roll back.",
            "Organic volume: scale consumers up to the partition count; beyond that, more replicas do not help.",
        ],
        "prevention": [
            "Consumers use cooperative-sticky assignment with static membership",
            "Deserialisation failures go to a DLQ instead of blocking the partition",
        ],
    },
    C.API_TIMEOUT: {
        "title": "API Timeouts and Latency Spikes",
        "when": "`*LatencyP99High` fires, clients see 504s, or upstream calls log `ReadTimeout`.",
        "symptoms": [
            "p99 latency above SLO while error rate is still low",
            "504 from the gateway",
            "Thread/worker saturation",
        ],
        "diagnosis": [
            "Use traces to locate the slow span: is the time spent in the service itself, the database, or an upstream call?",
            "For database time, check `pg_stat_statements` for the top queries by mean time, and check for table bloat.",
            "For upstream time, compare the caller's timeout with the upstream's p99. A timeout below the upstream p99 causes retries and a retry storm.",
            "Check for recent changes to `HTTP_TIMEOUT_SECONDS` or `HTTP_MAX_RETRIES` in the deploy manifest.",
        ],
        "mitigation": [
            "Roll back timeout or retry changes.",
            "Shed load: enable the circuit breaker or the degraded mode for soft dependencies.",
            "Scale the slow component if it is capacity-bound.",
        ],
        "prevention": [
            "Timeouts are set per dependency from its measured p99.9",
            "The retry budget is at most 2 with jittered backoff",
        ],
    },
    C.AUTHENTICATION_FAILURE: {
        "title": "Authentication Token Failures",
        "when": "`ApiGatewayAuthFailuresHigh` or `AuthServiceAuthFailuresHigh` fires, or customers report being signed out.",
        "symptoms": [
            "Spike in HTTP 401 at the gateway",
            "`unknown signing key` or `token is not yet valid` in logs",
            "Logins succeed but subsequent calls fail",
        ],
        "diagnosis": [
            "Compare the `kid` of rejected tokens with the kids served at `/v1/auth/.well-known/jwks.json`.",
            "Check the signing-key rotation job history (`kubectl -n prod logs job/signing-key-rotation`).",
            "Check node clock skew: `chronyc tracking` on affected nodes (the gateway allows 30s of leeway).",
            "Check recent deploys of auth-service and api-gateway for changes to token validation or JWKS caching.",
        ],
        "mitigation": [
            "Force a JWKS refresh on the gateway (`deployctl restart api-gateway` or `POST /admin/jwks/refresh`).",
            "Roll back validation changes.",
            "If keys were rotated too early, re-activate the previous key (see the Signing Key Rotation runbook).",
        ],
        "prevention": [
            "New keys are published 24h before activation",
            "The gateway refreshes JWKS when it sees an unknown kid",
        ],
    },
    C.DEPLOYMENT_REGRESSION: {
        "title": "Deployment Regression Triage and Rollback",
        "when": "Errors or latency change within minutes to hours of a deployment.",
        "symptoms": [
            "Error rate steps up right after a rollout",
            "New exception types in logs",
            "Only some client versions are affected",
        ],
        "diagnosis": [
            "List recent deploys: `deployctl history <service> --limit 5`.",
            "Diff the release against the previous one: `deployctl diff <service> <prev> <current>` lists the included PRs.",
            "Group errors by exception and route to find the broken path.",
            "Check whether dependent services had incidents at the same time.",
        ],
        "mitigation": [
            "Roll back first, debug second: `deployctl rollback <service> --to <previous-version>`.",
            "Ship forward only when the fix is small, reviewed, and faster than a rollback.",
        ],
        "prevention": [
            "Canary analysis gates tier_0 deploys at 5% and 25%",
            "Contract tests with older client payloads",
        ],
    },
    C.CONFIGURATION_ERROR: {
        "title": "Configuration Error Recovery",
        "when": "Failures start right after a manifest, feature flag or ConfigMap change.",
        "symptoms": [
            "Connection errors to hosts that do not exist in production",
            "Only writes (or only one path) fail",
            "Readiness probes pass while real traffic fails",
        ],
        "diagnosis": [
            "Diff the live environment with Git: `kubectl -n prod get deploy <service> -o yaml` against `deploy/<service>.yaml`.",
            "Check recent flag changes in LaunchPad (audit log).",
            "Resolve the hosts in connection errors: `kubectl -n prod exec deploy/<service> -- getent hosts <host>`.",
        ],
        "mitigation": [
            "Revert the manifest change (roll back the deployment) or the flag.",
            "Restore a ConfigMap from Git; direct edits are blocked by admission policy.",
        ],
        "prevention": [
            "Manifests are validated against the production host allow-list in CI",
            "Flag changes above 10% need a second approver",
        ],
    },
    C.DATABASE_DEADLOCK: {
        "title": "Database Deadlocks",
        "when": "`*DBDeadlocksDetected` fires or logs show `deadlock detected`.",
        "symptoms": [
            "A fraction of writes fail and succeed on retry",
            "Spikes during batch jobs or migrations",
        ],
        "diagnosis": [
            "Read the deadlock report in the PostgreSQL log. It names both statements and the locks involved.",
            "Check whether code paths lock rows in different orders (for example sorted in one path and caller order in another).",
            "Check running migrations and batch jobs: `SELECT * FROM pg_stat_activity WHERE query ILIKE '%index%' OR application_name LIKE '%job%';`",
        ],
        "mitigation": [
            "Pause the conflicting batch job",
            "Roll back a release that changed lock ordering",
            "Re-run index migrations with CONCURRENTLY",
        ],
        "prevention": [
            "Multi-row updates lock rows in primary-key order",
            "Migrations use CREATE INDEX CONCURRENTLY",
        ],
    },
    C.MEMORY_LEAK: {
        "title": "Memory Leaks and OOMKilled Pods",
        "when": "`*PodRestartsHigh` or `*MemoryNearLimit` fires, or pods show `OOMKilled` as their last state.",
        "symptoms": [
            "A sawtooth memory graph: linear growth, then a restart",
            "Requests dropped during restarts (502/503)",
        ],
        "diagnosis": [
            "`kubectl -n prod get pods -l app=<service>` shows restart counts; `kubectl describe pod` shows the reason.",
            "Correlate the start of growth with deploys. Growth proportional to traffic points to per-request retention.",
            "Take a heap snapshot on one pod (`py-spy dump`, `tracemalloc` endpoint) before it restarts.",
        ],
        "mitigation": [
            "Roll back if growth began with a release",
            "Raise the memory limit temporarily and stagger restarts to protect capacity",
        ],
        "prevention": [
            "No unbounded module-level collections",
            "Memory growth alerts at 80% of the limit",
        ],
    },
    C.CPU_SATURATION: {
        "title": "CPU Saturation and Throttling",
        "when": "`*CPUThrottlingHigh` fires or latency rises with CPU pinned at the limit.",
        "symptoms": [
            "CPU at the limit and throttled periods above 50%",
            "Latency rises across all endpoints",
        ],
        "diagnosis": [
            "Check whether the HPA is at `max_replicas`.",
            "Check the log level and log volume (DEBUG in production is a common cause).",
            "Check traffic sources at the edge for bot patterns.",
        ],
        "mitigation": [
            "Raise the HPA maximum and CPU limits",
            "Set the log level back to INFO",
            "Block abusive traffic at the edge",
        ],
        "prevention": [
            "Capacity tests before peak events",
            "Log level changes in production go through the deploy pipeline with an expiry",
        ],
    },
    C.DEPENDENCY_FAILURE: {
        "title": "Upstream Dependency Failure",
        "when": "`*UpstreamErrorRate` or `*ProviderErrorRate` fires: a service we call is failing.",
        "symptoms": [
            "Errors concentrated on calls to one upstream",
            "The circuit breaker opens",
            "Another team's incident is active",
        ],
        "diagnosis": [
            "Find the failing upstream from traces or `UpstreamUnavailable` logs.",
            "Check whether the upstream has an active incident (#incidents channel), or the provider's status page.",
            "Check the dependency criticality in the service's behaviour doc: a hard dependency means we fail, a soft one means we degrade.",
        ],
        "mitigation": [
            "Let the circuit breaker shed load; do not raise retries",
            "Enable fallback or degraded mode for soft dependencies",
            "Coordinate with the upstream's incident commander",
        ],
        "prevention": [
            "Every hard dependency has a documented failure mode",
            "Retries use jittered backoff and a retry budget",
        ],
    },
    C.NETWORK_TIMEOUT: {
        "title": "Network Timeouts and DNS Failures",
        "when": "Connect timeouts or name-resolution errors across several services at once.",
        "symptoms": [
            "`ConnectTimeout` or `Temporary failure in name resolution`",
            "Errors concentrated in one availability zone",
        ],
        "diagnosis": [
            "Group errors by the source node's zone (`topology.kubernetes.io/zone`).",
            "Check CoreDNS CPU usage and query latency.",
            "Check `nf_conntrack` usage on nodes: `conntrack -S`.",
        ],
        "mitigation": [
            "Cordon affected nodes or zones",
            "Scale CoreDNS or enable NodeLocal DNSCache",
            "Raise `nf_conntrack_max`",
        ],
        "prevention": [
            "Keep-alive for internal HTTP calls",
            "Multi-zone spread for all tier_0 services",
        ],
    },
    C.RATE_LIMITING: {
        "title": "Rate Limiting (HTTP 429) Spikes",
        "when": "`ApiGateway429RateHigh` or `*ProviderErrorRate` fires with 429 responses.",
        "symptoms": ["Clients receive 429 with `Retry-After`", "A provider throttles our API key"],
        "diagnosis": [
            "Top rejected clients: `logs query 'service:api-gateway message:ratelimit.rejected' | top client_id`.",
            "Check whether the limiter key changed (client id vs IP) in a recent gateway deploy.",
            "For providers, compare our request rate with the contracted limit.",
        ],
        "mitigation": [
            "Raise the limit temporarily for affected clients",
            "Roll back limiter changes",
            "Enable request smoothing for provider calls",
        ],
        "prevention": [
            "Limits are keyed by authenticated client id",
            "Clients back off exponentially",
        ],
    },
    C.CACHE_INVALIDATION: {
        "title": "Stale Cache and Invalidation Issues",
        "when": "Customers report outdated data (prices, profiles, stock) after updates.",
        "symptoms": [
            "Data is correct in the database but wrong in API responses",
            "Stale data expires on its own after the TTL",
        ],
        "diagnosis": [
            "Compare the cached value (`redis-cli GET <prefix>:v3:<id>`) with the database row.",
            "Check whether invalidation deletes keys or only extends them (`invalidate()` in `cache.py`).",
            "Check the ordering of invalidation relative to the database commit.",
        ],
        "mitigation": [
            "Delete the affected keys (SCAN + UNLINK)",
            "Roll back invalidation changes",
        ],
        "prevention": [
            "Invalidate after commit",
            "Version cache keys when the payload shape changes",
        ],
    },
    C.SCHEMA_MISMATCH: {
        "title": "Schema Mismatch after Migrations or Event Changes",
        "when": "Errors such as `UndefinedColumn`, `NotNullViolation` or consumer `KeyError` on event fields.",
        "symptoms": [
            "All queries touching one table fail",
            "Consumers fail on every new message of one topic",
        ],
        "diagnosis": [
            "Compare ORM models with the live schema: `\\d+ <table>` in psql.",
            "Check applied migrations (`alembic current`) against the release.",
            "For events, compare the producer's `events/schemas.py` with consumer handlers.",
        ],
        "mitigation": [
            "Roll back the producer or application release",
            "Add the missing column or default (expand phase)",
            "Replay failed events from the DLQ after the fix",
        ],
        "prevention": [
            "Expand/contract migrations only",
            "Event fields are never renamed in place: bump SCHEMA_VERSION and dual-publish",
        ],
    },
}


def _category_runbook_body(category: IncidentCategory) -> Callable[[DocContext], str]:
    spec = CATEGORY_RUNBOOKS[category]

    def body(ctx: DocContext) -> str:
        draft = ctx.drafts[("category", str(category))]
        return (
            _header(
                f"Runbook: {spec['title']}",
                [
                    ("Owner", "sre"),
                    ("Scope", "All services"),
                    ("Last reviewed", f"{draft.updated_at:%Y-%m-%d}"),
                ],
            )
            + "\n"
            + _h("When to use", str(spec["when"]))
            + "\n"
            + _h("Symptoms", bullet_list(list(spec["symptoms"])))  # type: ignore[arg-type]
            + "\n"
            + _h("Diagnosis", _steps(list(spec["diagnosis"])))  # type: ignore[arg-type]
            + "\n"
            + _h("Mitigation", _steps(list(spec["mitigation"])))  # type: ignore[arg-type]
            + "\n"
            + _h("Prevention", bullet_list(list(spec["prevention"])))  # type: ignore[arg-type]
            + "\n"
            + _h(
                "Escalation",
                "Page the owning team's on-call. For SEV1, open an incident channel and "
                "assign an incident commander from SRE (see the Incident Management Process).",
            )
            + _past_incidents(ctx, draft.id)
        )

    return body


def _past_incidents(ctx: DocContext, runbook_id: str) -> str:
    incidents = sorted(ctx.incidents_by_runbook.get(runbook_id, []), key=lambda i: i.started_at)[
        -6:
    ]
    if not incidents:
        return ""
    rows = [f"{i.id} ({i.started_at:%Y-%m-%d}, {i.severity.value}): {i.title}" for i in incidents]
    return "\n" + _h("Recent incidents that used this runbook", bullet_list(rows))


# --- service runbooks --------------------------------------------------------------------------


def _service_runbook(p: ServiceProfile, kind: str) -> Callable[[DocContext], str]:
    alerts = alert_names(p)
    ns = "kubectl -n prod"

    def body(ctx: DocContext) -> str:
        draft = ctx.drafts[("service", p.id, kind)]
        deps = [d.service for d in p.http_dependencies]
        if kind == "5xx":
            title = f"{p.display_name}: High 5xx Error Rate"
            when = f"`{alerts['http_5xx']}` fires (5xx ratio above 2% for 5 minutes on any {p.id} route)."
            diagnosis = [
                f"Check recent deploys: `deployctl history {p.id} --limit 5`. If the onset is within a few hours of a deploy, prefer a rollback.",
                f"Group errors by exception: `logs query 'service:{p.id} level:ERROR' --since 30m | top exception`.",
            ]
            if p.postgres_db:
                diagnosis.append(
                    f"`sqlalchemy.exc.TimeoutError` (QueuePool): follow {ctx.ref('category', C.DB_CONNECTION_EXHAUSTION)}."
                )
                diagnosis.append(
                    "`UndefinedColumn` or `NotNullViolation`: schema mismatch between code and database."
                )
            if p.redis_cluster:
                diagnosis.append(
                    f"Redis `ConnectionError` or `OOM`: check `{p.redis_cluster}` (`redis-cli -h {p.redis_cluster} INFO memory`)."
                )
            if deps:
                diagnosis.append(
                    f"`UpstreamUnavailable` from {', '.join(deps)}: check those services' dashboards and active incidents."
                )
            diagnosis.append(
                f"Verify the environment matches Git: `{ns} get deploy {p.id} -o yaml | diff - deploy/{p.id}.yaml`."
            )
            mitigation = [
                f"Roll back: `deployctl rollback {p.id} --to <previous-version>`.",
                f"If a dependency is failing, see {ctx.ref('category', C.DEPENDENCY_FAILURE)}.",
                f"Scale out if the errors are capacity-related: `{ns} scale deploy {p.id} --replicas={p.config.replicas * 2}`.",
            ]
        elif kind == "latency":
            title = f"{p.display_name}: Elevated Latency"
            when = f"`{alerts['latency']}` fires (p99 above {p.config.p99_latency_slo_ms} ms for 10 minutes) or `{alerts['cpu']}` fires."
            diagnosis = [
                "Open a slow trace from the service dashboard and find the dominant span.",
                f"Check CPU throttling and HPA state: `{ns} get hpa {p.id}` (max {p.config.replicas * 3} replicas).",
            ]
            if deps:
                diagnosis.append(
                    f"Compare upstream p99 with the client timeout ({p.config.http_timeout_seconds}s, {p.config.http_max_retries} retries)."
                )
            if p.postgres_db:
                diagnosis.append(
                    f"Check slow queries on `{p.postgres_db}`: `SELECT query, mean_exec_time FROM pg_stat_statements ORDER BY mean_exec_time DESC LIMIT 10;`"
                )
            diagnosis.append(
                f"Check the log level: `{ns} get deploy {p.id} -o jsonpath='{{..env}}' | grep LOG_LEVEL` (must be INFO)."
            )
            mitigation = [
                f"Scale out: `{ns} scale deploy {p.id} --replicas=<n>`.",
                f"Roll back recent timeout, retry or log-level changes; see {ctx.ref('category', C.API_TIMEOUT)}.",
            ]
        else:
            title = f"{p.display_name}: Pods Restarting / OOMKilled"
            when = f"`{alerts['restarts']}` or `{alerts['memory']}` fires."
            diagnosis = [
                f"`{ns} get pods -l app={p.id}`: check restart counts and `lastState.terminated.reason`.",
                f"Memory limit is {p.config.memory}. Linear growth up to the limit means a leak (see {ctx.ref('category', C.MEMORY_LEAK)}).",
                f"Check liveness probe failures: `{ns} describe pod <pod>` (events section).",
            ]
            mitigation = [
                "Roll back if the growth started with a release.",
                f"Temporarily raise the memory limit above {p.config.memory} and stagger restarts.",
            ]
        return (
            _header(
                f"Runbook: {title}",
                [
                    ("Owner", p.team),
                    ("Service", p.id),
                    ("On-call", p.oncall_channel),
                    ("Last reviewed", f"{draft.updated_at:%Y-%m-%d}"),
                ],
            )
            + "\n"
            + _h("When to use", when)
            + "\n"
            + _h("Diagnosis", _steps(diagnosis))
            + "\n"
            + _h("Mitigation", _steps(mitigation))
            + "\n"
            + _h(
                "Escalation",
                f"Page `{p.oncall_channel}`. SEV1/SEV2: SRE incident commander via #incidents.",
            )
            + _past_incidents(ctx, draft.id)
        )

    return body


# --- bespoke runbooks ------------------------------------------------------------------------------


@dataclass(frozen=True)
class BespokeRunbook:
    name: str
    service_id: str | None
    categories: tuple[IncidentCategory, ...]
    link_services: tuple[str, ...]
    when: str
    symptoms: tuple[str, ...]
    diagnosis: tuple[str, ...]
    mitigation: tuple[str, ...]
    access_level: AccessLevel = AccessLevel.ENGINEERING


def _bespoke_runbooks() -> list[BespokeRunbook]:
    pay, gw = SERVICES_BY_ID["payment-service"], SERVICES_BY_ID["api-gateway"]
    pay_capacity = pay.config.replicas * (pay.config.db_pool_size + pay.config.db_max_overflow)
    return [
        BespokeRunbook(
            "Payment API 500 Errors",
            "payment-service",
            (
                C.DB_CONNECTION_EXHAUSTION,
                C.DEPLOYMENT_REGRESSION,
                C.CONFIGURATION_ERROR,
                C.SCHEMA_MISMATCH,
            ),
            ("payment-service",),
            "`PaymentServiceHigh5xxRate` fires, or order-service logs `checkout.payment_failed` for many orders.",
            (
                "`POST /v1/payments` returns HTTP 500; the gateway returns 502 on `/v1/payments`",
                "order-service checkouts fail with `payment unavailable`",
                "Payment success rate on the Payments dashboard drops below 97%",
            ),
            (
                "Check recent payment-service deploys: `deployctl history payment-service --limit 5`. Most payment 500s in the last year followed a release.",
                "Group errors by exception: `logs query 'service:payment-service level:ERROR' --since 30m | top exception`.",
                f"`sqlalchemy.exc.TimeoutError: QueuePool limit ...`: connection pool exhaustion. Normal sizing is `db_pool_size={pay.config.db_pool_size}` and `db_max_overflow={pay.config.db_max_overflow}` per pod x {pay.config.replicas} pods = {pay_capacity} connections. Check `payment_service/db/database.py` in the running release for pool changes.",
                "`KeyError` in `processor.create_payment`: request contract regression (older app versions still send `card_token`).",
                "`ReadOnlySqlTransaction`: `PAYMENT_SERVICE_DATABASE_URL` points at the read-only pooler.",
                "`ProviderUnavailable`: PayFlux degradation; use the PayFlux Provider Degradation runbook.",
                "Check PgBouncer for `payments`: `psql -p 6432 pgbouncer -c 'SHOW POOLS;'` (cl_waiting should be 0).",
            ),
            (
                "If errors started after a deploy, roll back immediately: `deployctl rollback payment-service --to <previous-version>`.",
                "Pool exhaustion without a deploy: scale to 10 replicas and confirm PgBouncer `default_pool_size` for `payments` stays at or above the new total.",
                "Tell #commerce-oncall that checkout failures are payment-caused, so order-service does not roll back needlessly.",
                "After recovery, reconcile authorisations with PayFlux (`payments-reconcile --since <incident start>`).",
            ),
            AccessLevel.ENGINEERING,
        ),
        BespokeRunbook(
            "PayFlux Provider Degradation",
            "payment-service",
            (C.DEPENDENCY_FAILURE, C.RATE_LIMITING),
            ("payment-service",),
            "`PaymentServiceProviderErrorRate` fires: PayFlux returns 5xx, 429, or times out.",
            (
                "`ProviderUnavailable` in payment-service logs",
                "Card authorisations fail while other payment routes work",
            ),
            (
                "Check the PayFlux status page and the #vendor-payflux channel.",
                "429 responses mean we exceeded our rate limit. Check the client retry settings in `clients/payflux_client.py` (backoff 0.2s, max 2 retries).",
                "5xx or timeouts: compare our p99 to PayFlux with the 2.5s timeout.",
            ),
            (
                "Do not increase retries. That turns a degradation into a rate-limit outage.",
                "Ask PayFlux support (contract P1 line) for a temporary rate-limit increase if we are throttled.",
                "Enable the 'payment retry later' checkout banner via the `checkout_payment_retry_banner` flag.",
            ),
        ),
        BespokeRunbook(
            "Checkout Failures",
            "order-service",
            (C.DEPENDENCY_FAILURE, C.DEPLOYMENT_REGRESSION, C.API_TIMEOUT),
            ("order-service",),
            "`OrderServiceHigh5xxRate` or `OrderServiceUpstreamErrorRate` fires; the order success rate drops.",
            (
                "`POST /v1/orders` fails",
                "`checkout.payment_failed` or inventory reservation errors in logs",
            ),
            (
                "Checkout depends on cart-service, inventory-service and payment-service (all hard dependencies). Find the failing hop in traces.",
                "If a dependency has an active incident, do not roll back order-service. Coordinate instead.",
                "Check order-service deploys for changes to `checkout.py`.",
            ),
            (
                "Mitigate the failing dependency first.",
                "Roll back order-service only if the errors originate in its own code.",
                "Orders stuck in `pending` after recovery are retried by the saga reaper every 5 minutes.",
            ),
        ),
        BespokeRunbook(
            "Inventory Synchronization Failure",
            "inventory-service",
            (C.DEPLOYMENT_REGRESSION, C.KAFKA_CONSUMER_LAG, C.DEPENDENCY_FAILURE),
            ("inventory-service",),
            "The daily reconciliation report shows drift between NovaCart stock and StockSync WMS, or `reconciliation.mismatch` errors appear.",
            (
                "Oversold SKUs (orders for stock that does not exist)",
                "Items shown out of stock that are available in the warehouse",
                "`wms.sync_applied` counts much higher than the deltas received",
            ),
            (
                "Compare `applied` with `deltas` in `wms.sync_applied` logs. Applied > received means deltas were double-applied.",
                "Check `warehouse_sync.py` in the running release: deltas must be de-duplicated by per-warehouse sequence.",
                "Check StockSync API health and `inventory-service` consumer lag for `order.created`.",
                "Check recent inventory-service deploys.",
            ),
            (
                "Pause the sync job: `deployctl scale inventory-service-sync --replicas=0`.",
                "Roll back a release that changed sync logic.",
                "Run a full reconciliation: `inventory-reconcile --source stocksync --apply` (takes about 20 minutes).",
                "Notify #commerce-oncall about oversold orders so support can contact customers.",
            ),
        ),
        BespokeRunbook(
            "API Gateway Timeout",
            "api-gateway",
            (C.API_TIMEOUT, C.NETWORK_TIMEOUT, C.DEPENDENCY_FAILURE),
            ("api-gateway",),
            "`ApiGatewayLatencyP99High` fires, or clients receive 504 from the gateway.",
            (
                "504 Gateway Timeout for one route prefix or all routes",
                "`gateway.upstream_error` logs",
            ),
            (
                "Identify the route prefix: per-upstream timeouts are in `routing.py` (orders/payments 5s, search 1.5s, recommendations 0.8s, others 2s).",
                "504s on one prefix: the upstream is slow. Follow its latency runbook.",
                "504s on all prefixes: check gateway CPU, the connection pool (max 500 connections) and the network (see Network Timeouts).",
                f"Check gateway replicas: {gw.config.replicas} minimum, HPA max {gw.config.replicas * 3}.",
            ),
            (
                "Scale the gateway if CPU-bound.",
                "Mitigate the slow upstream.",
                "Do not raise gateway timeouts during an incident. It holds connections longer and makes saturation worse.",
            ),
        ),
        BespokeRunbook(
            "Authentication Token Failure",
            "auth-service",
            (C.AUTHENTICATION_FAILURE,),
            ("auth-service", "api-gateway"),
            "HTTP 401 spikes at the gateway or auth-service; customers are signed out unexpectedly.",
            (
                "`unknown signing key` in api-gateway logs",
                "`ImmatureSignatureError` or `ExpiredSignatureError` in auth-service logs",
                "Login succeeds but the next API call returns 401",
            ),
            (
                "Decode a failing token header (`jwt decode --no-verify`) and compare its `kid` with `/v1/auth/.well-known/jwks.json`.",
                "If the kid is in JWKS but the gateway rejects it: the gateway's JWKS cache is stale (`JWKS_CACHE_TTL_SECONDS` in `auth_middleware.py` should be 300, and unknown kids must trigger a refresh).",
                "`not yet valid (iat)`: check node clocks (`chronyc tracking`) and the validation leeway (30s).",
                "Check signing-key rotation job logs and recent deploys of both services.",
            ),
            (
                "Force a JWKS refresh: `deployctl restart api-gateway` (rolling).",
                "Roll back token-validation changes.",
                "If the new key is bad, re-activate the previous key (Signing Key Rotation runbook, restricted).",
            ),
            AccessLevel.SRE,
        ),
        BespokeRunbook(
            "Signing Key Rotation",
            "auth-service",
            (C.AUTHENTICATION_FAILURE,),
            (),
            "Scheduled weekly rotation, emergency key revocation, or rollback of a newly activated key.",
            ("Tokens with an unexpected `kid`", "Rotation job failures"),
            (
                "List keys: `SELECT kid, status, activated_at FROM signing_keys ORDER BY activated_at DESC LIMIT 5;` (database `auth`).",
                "A new key must be `retiring`-published for 24h before it becomes `active`.",
                "Private keys are in Vault at `secret/auth-service/signing-keys/<kid>`; never copy them locally.",
            ),
            (
                "Emergency re-activation: `authctl keys activate <previous-kid>` (requires security on-call approval).",
                "Revocation: `authctl keys revoke <kid>` then force a JWKS refresh on the gateway.",
                "Record the change in #security-changes.",
            ),
            AccessLevel.SRE,
        ),
        BespokeRunbook(
            "Gateway 429 Rate Limit Spikes",
            "api-gateway",
            (C.RATE_LIMITING,),
            ("api-gateway",),
            "`ApiGateway429RateHigh` fires.",
            ("Clients receive 429 with `Retry-After: 1`", "`ratelimit.rejected` logs"),
            (
                "Top rejected keys: `logs query 'service:api-gateway message:ratelimit.rejected' | top client_id`.",
                "Limits are 50 rps and a burst of 100 per client id, with partner overrides in `rate_limiter.py`. The bucket key must be `ratelimit:<client_id>`; keying by IP throttles carrier-NAT users together.",
                "Check whether a mobile release is retrying without backoff (group by `user-agent`).",
            ),
            (
                "Roll back limiter changes.",
                "Temporarily raise the limit for a client: `gatewayctl limits set <client_id> <rps>`.",
                "Coordinate with the partner or mobile team on backoff.",
            ),
        ),
        BespokeRunbook(
            "Cart Redis Memory Pressure",
            "cart-service",
            (C.REDIS_MEMORY_PRESSURE,),
            ("cart-service",),
            "`CartServiceRedisMemoryHigh` fires for `redis-carts` (8 GB, `volatile-lru`).",
            (
                "`OOM command not allowed` on add-to-cart",
                "Memory climbs steadily over days instead of following traffic",
            ),
            (
                "`redis-cli -h redis-carts INFO keyspace`: `expires` should be close to `keys`. Carts expire after 7 days of inactivity.",
                "Sample keys: `redis-cli -h redis-carts --scan --pattern 'cart:*' | head -1000 | xargs -n1 redis-cli -h redis-carts TTL | sort | uniq -c`. Many `-1` values mean carts were written without an expiry.",
                "Check recent cart-service deploys for changes to `CartStore.add` in `cart_store.py`.",
            ),
            (
                "Roll back the release that dropped the expiry.",
                "Backfill expiries: `cart-maintenance set-missing-ttl --ttl 604800` (SCAN-based).",
                "If writes are failing now, scale `redis-carts` to 16 GB (takes about 10 minutes, no downtime).",
            ),
        ),
        BespokeRunbook(
            "Search Indexing Lag",
            "search-service",
            (C.KAFKA_CONSUMER_LAG, C.SCHEMA_MISMATCH),
            ("search-service",),
            "`SearchServiceConsumerLagHigh` fires; product or price changes are missing from search results.",
            ("Searches return old prices", "`indexer.bulk_errors` or `KeyError` in consumer logs"),
            (
                "Consumer group `search-service-consumer` on `product.updated` and `price.changed`: check lag per partition.",
                "`KeyError` on event fields: product-service changed its event schema. Compare with `product_service/events/schemas.py`.",
                "Check OpenSearch cluster health (`GET _cluster/health`) and bulk rejections.",
            ),
            (
                "Fix or roll back the producer; replay failed events from `<topic>.dlq`.",
                "Trigger a full reindex from product-service if drift is large (`search-reindex --full`, about 40 minutes).",
            ),
        ),
        BespokeRunbook(
            "Notification Delivery Delays",
            "notification-service",
            (C.KAFKA_CONSUMER_LAG, C.DEPENDENCY_FAILURE, C.RATE_LIMITING),
            ("notification-service",),
            "`NotificationServiceConsumerLagHigh` or `NotificationServiceProviderErrorRate` fires; customers report missing order emails.",
            ("Lag on `order.created` / `payment.completed`", "MailRelay or TextBridge 429/5xx"),
            (
                "Check provider status (MailRelay, TextBridge).",
                "Check consumer errors: template rendering (`KeyError`) blocks whole batches.",
                "Check delivery attempts: `SELECT status, count(*) FROM delivery_attempts WHERE attempted_at > now() - interval '1 hour' GROUP BY 1;`",
            ),
            (
                "Provider outage: messages are retried with backoff up to 5 attempts; nothing to do unless it exceeds 1h.",
                "Consumer failures: roll back and replay from the DLQ.",
            ),
        ),
        BespokeRunbook(
            "Recommendation Fallback Mode",
            "recommendation-service",
            (C.DEPENDENCY_FAILURE, C.REDIS_MEMORY_PRESSURE, C.DEPLOYMENT_REGRESSION),
            ("recommendation-service",),
            "Recommendation carousels are empty, or `ranker.fallback_popular` logs spike.",
            ("Personalised carousels show generic popular items", "HTTP 500 for new visitors"),
            (
                "Check `redis-reco` health; `FeatureStoreUnavailable` triggers the popular-items fallback (expected degradation).",
                "HTTP 500 instead of fallback means a regression: check the handling of missing feature vectors in `ranker.py`.",
            ),
            (
                "Fallback is acceptable for up to 24h (tier_2).",
                "Roll back regressions; restore Redis capacity.",
            ),
        ),
        BespokeRunbook(
            "Product Price Inconsistency",
            "product-service",
            (C.CACHE_INVALIDATION, C.SCHEMA_MISMATCH),
            ("product-service",),
            "Customers or merchandisers report prices that differ between product page, cart and search.",
            (
                "Product page shows the old price after an update",
                "Search shows a different price than the product page",
            ),
            (
                "Compare the `prices` row, the `redis-catalog` entry (`product:v3:<id>`) and the search document for the SKU.",
                "If the cache is stale, check `ProductCache.invalidate` (must delete the key).",
                "If search is stale, check search-service consumer lag on `price.changed`.",
            ),
            (
                "Delete stale cache keys; replay `price.changed` for affected SKUs.",
                "Roll back invalidation changes.",
            ),
        ),
        BespokeRunbook(
            "Emergency Rollback Procedure",
            None,
            (),
            (),
            "Any production incident where a recent deploy is a plausible cause.",
            ("Error rate or latency changed within hours of a deploy",),
            (
                "`deployctl history <service> --limit 5` to find the last good version.",
                "Check for database migrations in the release. Expand-only migrations are safe to roll back over; contract migrations are not.",
            ),
            (
                "`deployctl rollback <service> --to <version>`: a rolling rollback takes 2-5 minutes and skips canary analysis.",
                "Announce in #incidents with the deployment id and the reason.",
                "Freeze further deploys of the service until a fix is reviewed (`deployctl freeze <service>`).",
            ),
        ),
    ]


def _bespoke_body(rb: BespokeRunbook) -> Callable[[DocContext], str]:
    def body(ctx: DocContext) -> str:
        draft = ctx.drafts[("bespoke", rb.name)]
        p = SERVICES_BY_ID.get(rb.service_id or "")
        rows = [("Owner", p.team if p else "sre"), ("Service", rb.service_id or "All services")]
        if p:
            alerts = alert_names(p)
            rows.append(("Alerts", ", ".join(f"`{a}`" for a in list(alerts.values())[:4])))
            rows.append(("On-call", p.oncall_channel))
        rows.append(("Last reviewed", f"{draft.updated_at:%Y-%m-%d}"))
        related = [ctx.ref("category", c) for c in rb.categories[:3]]
        return (
            _header(f"Runbook: {rb.name}", rows)
            + "\n"
            + _h("When to use", rb.when)
            + "\n"
            + _h("Symptoms", bullet_list(list(rb.symptoms)))
            + "\n"
            + _h("Diagnosis", _steps(list(rb.diagnosis)))
            + "\n"
            + _h("Mitigation", _steps(list(rb.mitigation)))
            + ("\n" + _h("Related runbooks", bullet_list(related)) if related else "")
            + _past_incidents(ctx, draft.id)
        )

    return body


# --- per-service technical documentation ------------------------------------------------------


def _architecture(p: ServiceProfile) -> Callable[[DocContext], str]:
    def body(ctx: DocContext) -> str:
        deps = markdown_table(
            ["Dependency", "Type", "Criticality", "Purpose"],
            [[d.service, "HTTP", d.criticality.value, d.purpose] for d in p.http_dependencies]
            + [[x.name, "external HTTPS", "hard", x.purpose] for x in p.external]
            + [[TOPIC_PRODUCERS[t], f"Kafka `{t}`", "soft", "event consumer"] for t in p.consumes],
        )
        dependents = [d.id for d in dependents_of(p.id)] + [c.id for c in consumers_of(p.id)]
        modules = [f"`{p.package}/{m}.py`" for m in p.domain_modules] + [
            f"`{p.package}/api/routes.py` (HTTP layer)"
        ]
        if p.postgres_db:
            modules.append(f"`{p.package}/db/` (SQLAlchemy engine, models, repository)")
        if p.redis_cluster:
            modules.append(f"`{p.package}/cache.py` (Redis, {p.redis_usage})")
        if p.consumes:
            modules.append(
                f"`{p.package}/events/consumer.py` (Kafka consumer group `{p.consumer_group}`)"
            )
        if p.produces:
            modules.append(
                f"`{p.package}/events/producer.py` + `schemas.py` (publishes {', '.join(p.produces)})"
            )
        return (
            f"# {p.display_name} Architecture\n\n{p.description}\n\n"
            + _h("Components", bullet_list(modules))
            + "\n"
            + _h(
                "Dependencies",
                deps if (p.http_dependencies or p.external or p.consumes) else "None.",
            )
            + "\n"
            + _h(
                "Depended on by",
                bullet_list([f"`{d}`" for d in sorted(set(dependents))])
                or "No internal callers (edge service).",
            )
            + "\n"
            + _h("Data stores", bullet_list([f"`{s}`" for s in p.datastores]) or "Stateless.")
            + "\n"
            + _h(
                "Scaling",
                f"{p.config.replicas} replicas minimum, HPA up to {p.config.replicas * 3} at 65% CPU. "
                f"Requests: {p.config.cpu} CPU / {p.config.memory} memory per pod.",
            )
            + "\n"
            + _h(
                "Related",
                bullet_list(
                    [
                        ctx.ref("service", p.id, "api_reference"),
                        ctx.ref("service", p.id, "service_behavior"),
                        ctx.ref("service", p.id, "monitoring"),
                    ]
                ),
            )
        )

    return body


def _api_reference(p: ServiceProfile) -> Callable[[DocContext], str]:
    def body(ctx: DocContext) -> str:
        blocks = []
        for ep in p.endpoints:
            errors = [
                "400 invalid request",
                "404 not found",
                "429 rate limited (via gateway)",
                "500 internal error",
                "503 dependency unavailable",
            ]
            if ep.method != "GET":
                errors.insert(1, "409 conflict (idempotency or state)")
            blocks.append(
                f"### `{ep.method} {ep.path}`\n\n{ep.summary}.\n\n"
                f"- Handler: `{p.package}.api.routes.{ep.handler}`\n- Auth: bearer token validated by api-gateway\n"
                f"- Errors: {', '.join(errors)}\n"
            )
        return (
            f"# {p.display_name} API Reference\n\nBase URL (in cluster): `{service_url(p.id)}`. "
            "Public traffic arrives through api-gateway. All responses are JSON; errors use "
            '`{"error": {"code": ..., "message": ...}}`.\n\n'
            + "\n".join(blocks)
            + "\n"
            + _h("Related", ctx.ref("service", p.id, "architecture"))
        )

    return body


def _service_behavior(p: ServiceProfile) -> Callable[[DocContext], str]:
    def body(ctx: DocContext) -> str:
        c = p.config
        rows = []
        for d in p.http_dependencies:
            failure = (
                "request fails (5xx)"
                if d.criticality is DependencyCriticality.HARD
                else "degrades gracefully (fallback)"
            )
            rows.append(
                [
                    d.service,
                    f"{c.http_timeout_seconds}s",
                    str(c.http_max_retries),
                    "50% over 50 calls",
                    failure,
                ]
            )
        for x in p.external:
            rows.append(
                [
                    x.name,
                    f"{c.http_timeout_seconds}s",
                    str(c.http_max_retries),
                    "50% over 50 calls",
                    "request fails (5xx)",
                ]
            )
        parts = [
            f"# {p.display_name} Service Behavior: Timeouts, Retries and Degradation\n\n"
            "All outbound HTTP calls use `novacart_common.http.ResilientHttpClient`: per-call timeout, retries only for "
            "connect errors / timeouts / 502-504 with exponential backoff and full jitter, and a circuit breaker.\n"
        ]
        if rows:
            parts.append(
                _h(
                    "Outbound calls",
                    markdown_table(
                        ["Upstream", "Timeout", "Retries", "Breaker opens at", "When it fails"],
                        rows,
                    ),
                )
            )
        if p.postgres_db:
            capacity = c.replicas * (c.db_pool_size + c.db_max_overflow)
            parts.append(
                _h(
                    "Database connections",
                    f"Pool per pod: `db_pool_size={c.db_pool_size}`, `db_max_overflow={c.db_max_overflow}`, "
                    f"`pool_timeout={c.db_pool_timeout_seconds}s`. At {c.replicas} replicas the service can open up to "
                    f"{capacity} connections through PgBouncer. A request that cannot get a connection within the "
                    "pool timeout fails with `sqlalchemy.exc.TimeoutError` (HTTP 500). Sessions are always opened "
                    "via `session_scope()` so they are returned on every code path.",
                )
            )
        if p.redis_cluster:
            parts.append(
                _h(
                    "Caching",
                    f"`{p.redis_cluster}` holds {p.redis_usage}. TTL {c.cache_ttl_seconds}s; updates "
                    "invalidate by deleting the key. If Redis is unavailable, reads fall through to the source of truth.",
                )
            )
        if p.consumes:
            parts.append(
                _h(
                    "Event processing",
                    f"Consumer group `{p.consumer_group}` polls up to {c.kafka_max_poll_records} "
                    f"messages and processes them with concurrency {c.consumer_concurrency}; offsets are committed after "
                    "each batch (at-least-once), so handlers must be idempotent.",
                )
            )
        if p.produces:
            parts.append(
                _h(
                    "Published events",
                    bullet_list([f"`{t}`: {', '.join(EVENT_FIELDS[t])}" for t in p.produces])
                    + "\n\nRenaming a field is a breaking change: bump `SCHEMA_VERSION` and dual-publish.",
                )
            )
        parts.append(
            _h(
                "Related",
                bullet_list(
                    [
                        ctx.ref("service", p.id, "configuration"),
                        ctx.ref("platform", "resilience-standards"),
                    ]
                ),
            )
        )
        return "\n".join(parts)

    return body


def _configuration(p: ServiceProfile) -> Callable[[DocContext], str]:
    def body(ctx: DocContext) -> str:
        c, pre = p.config, p.env_prefix
        rows = [
            [f"`{pre}_LOG_LEVEL`", "INFO", "Never DEBUG in production (CPU cost)"],
            [f"`{pre}_HTTP_TIMEOUT_SECONDS`", str(c.http_timeout_seconds), "Per outbound call"],
            [
                f"`{pre}_HTTP_MAX_RETRIES`",
                str(c.http_max_retries),
                "Retries on connect errors, timeouts, 502-504",
            ],
        ]
        if p.postgres_db:
            rows += [
                [
                    f"`{pre}_DATABASE_URL`",
                    f"`{database_url(p)}`",
                    "Primary PgBouncer; never the `-ro` pooler",
                ],
                [f"`{pre}_DB_POOL_SIZE`", str(c.db_pool_size), "Connections kept per pod"],
                [
                    f"`{pre}_DB_MAX_OVERFLOW`",
                    str(c.db_max_overflow),
                    "Extra connections under burst",
                ],
                [
                    f"`{pre}_DB_POOL_TIMEOUT_SECONDS`",
                    str(c.db_pool_timeout_seconds),
                    "Wait for a connection before failing",
                ],
            ]
        if p.redis_cluster:
            rows += [
                [f"`{pre}_REDIS_URL`", f"`{redis_url(p)}`", "Production cluster"],
                [f"`{pre}_CACHE_TTL_SECONDS`", str(c.cache_ttl_seconds), "Applied to every write"],
            ]
        if p.consumes:
            rows += [
                [
                    f"`{pre}_KAFKA_MAX_POLL_RECORDS`",
                    str(c.kafka_max_poll_records),
                    "Batch size per poll",
                ],
                [f"`{pre}_KAFKA_CONSUMER_GROUP`", f"`{p.consumer_group}`", ""],
                [f"`{pre}_KAFKA_BOOTSTRAP_SERVERS`", f"`{KAFKA_BOOTSTRAP}`", ""],
            ]
        if p.id != "api-gateway":
            for d in p.http_dependencies:
                rows.append(
                    [
                        f"`{pre}_{dependency_url_setting(d.service).upper()}`",
                        f"`{service_url(d.service)}`",
                        f"{d.service} base URL",
                    ]
                )
        secrets = [
            f"`{pre}_{x.settings_prefix.upper()}_API_KEY` (Vault `secret/{p.id}/...`)"
            for x in p.external
        ]
        if p.postgres_db:
            secrets.append(f"`{pre}_DATABASE_PASSWORD` (Vault)")
        return (
            f"# {p.display_name} Configuration Reference\n\nSettings are environment variables read by "
            f"`{p.package}/config.py`; production values are set in `deploy/{p.id}.yaml`. Changing them requires a "
            "deploy (no hot reload).\n\n"
            + _h("Variables", markdown_table(["Variable", "Production value", "Notes"], rows))
            + "\n"
            + _h("Secrets", bullet_list(secrets) or "None.")
            + "\n"
            + _h("Related", ctx.ref("category", C.CONFIGURATION_ERROR))
        )

    return body


def _deployment_doc(p: ServiceProfile) -> Callable[[DocContext], str]:
    def body(ctx: DocContext) -> str:
        strategy = (
            "canary (5% -> 25% -> 100%, 10 minutes per step, automated error-rate analysis)"
            if p.tier.value == "tier_0"
            else "rolling update (maxUnavailable 0, maxSurge 25%)"
        )
        migrations = (
            (
                "Database migrations under `migrations/versions` run as a pre-deploy job and must be "
                "expand-only; contract steps ship in a later release."
            )
            if p.postgres_db
            else "No database migrations."
        )
        return (
            f"# {p.display_name} Deployment Guide\n\n"
            + _h(
                "Strategy",
                f"{p.id} deploys with {strategy}. Deploy windows: weekdays 09:30-16:30 UTC, Fridays until 12:30; "
                "no deploys during the Black Friday freeze.",
            )
            + "\n"
            + _h(
                "Steps",
                _steps(
                    [
                        f"Merge to `main`; CI runs `pytest services/{p.id}/tests`.",
                        f"`deployctl deploy {p.id} --version <vX.Y.Z>` (release notes list the included PRs).",
                        "Watch the canary dashboard; analysis aborts automatically on a 2x error-rate increase.",
                        "Announce in #deploys with the deployment id.",
                    ]
                ),
            )
            + "\n"
            + _h(
                "Rollback",
                f"`deployctl rollback {p.id} --to <previous-version>` redeploys the previous image with a "
                f"rolling strategy in 2-5 minutes. See {ctx.ref('bespoke', 'Emergency Rollback Procedure')}.",
            )
            + "\n"
            + _h("Migrations", migrations)
        )

    return body


def _data_doc(p: ServiceProfile) -> Callable[[DocContext], str]:
    def body(ctx: DocContext) -> str:
        parts = [f"# {p.display_name} Data Model\n"]
        if p.postgres_db:
            parts.append(
                f"PostgreSQL database `{p.postgres_db}` (via PgBouncer, transaction pooling).\n"
            )
            for table in p.tables:
                model, columns = TABLE_MODELS[table]
                rows = (
                    [["id", "String(36)", "primary key"]]
                    + [[n, t, ""] for n, t in columns]
                    + [
                        ["created_at", "timestamptz", "default now()"],
                        ["updated_at", "timestamptz", ""],
                    ]
                )
                parts.append(
                    _h(f"`{table}` ({model})", markdown_table(["Column", "Type", "Notes"], rows))
                )
            parts.append(
                _h(
                    "Indexes",
                    f"See `migrations/versions/0002_add_indexes.py`; `{primary_table(p)}` is indexed on "
                    "`(status, created_at)`.",
                )
            )
        if p.redis_cluster:
            parts.append(
                _h(
                    f"Redis `{p.redis_cluster}`",
                    f"Key pattern `{p.short.lower()}:v3:<id>` holding {p.redis_usage}; "
                    f"TTL {p.config.cache_ttl_seconds}s.",
                )
            )
        if "opensearch:search-products" in p.extra_datastores:
            parts.append(
                _h(
                    "OpenSearch",
                    "Index `products-v7` (title^3, brand^2, description; `title_suggest` completion field).",
                )
            )
        if p.produces:
            parts.append(
                _h(
                    "Events",
                    bullet_list(
                        [
                            f"`{t}` ({TOPIC_PARTITIONS[t]} partitions): {', '.join(EVENT_FIELDS[t])}"
                            for t in p.produces
                        ]
                    ),
                )
            )
        if len(parts) == 1:
            parts.append("Stateless service.")
        return "\n".join(parts)

    return body


def _monitoring(p: ServiceProfile) -> Callable[[DocContext], str]:
    def body(ctx: DocContext) -> str:
        alerts = alert_names(p)
        conditions = {
            "http_5xx": "5xx ratio > 2% for 5m",
            "latency": f"p99 > {p.config.p99_latency_slo_ms} ms for 10m",
            "restarts": "> 3 restarts in 15m",
            "memory": "working set > 90% of limit for 10m",
            "cpu": "throttled periods > 50% for 10m",
            "upstream": "upstream error ratio > 5% for 5m",
            "db_pool": "checked-out == pool size + overflow for 5m",
            "db_deadlock": "> 5 deadlocks in 10m",
            "db_errors": "query error ratio > 1% for 5m",
            "redis_memory": "used_memory > 85% of maxmemory",
            "cache_hit": "hit ratio < 60% for 15m",
            "consumer_lag": "lag > 50k messages or growing for 15m",
            "consumer_errors": "handler error ratio > 1% for 5m",
            "provider": "provider 5xx/429 ratio > 5% for 5m",
            "auth_failures": "401 ratio > 3% for 5m",
            "rate_limited": "429 ratio > 5% for 5m",
        }
        rows = [
            [f"`{name}`", conditions[key], _alert_runbook(ctx, p, key)]
            for key, name in alerts.items()
        ]
        return (
            f"# {p.display_name} Monitoring and Alerts\n\n"
            + _h(
                "SLOs",
                f"Availability {p.config.availability_slo}% (30-day window); p99 latency {p.config.p99_latency_slo_ms} ms.",
            )
            + "\n"
            + _h(
                "Dashboards",
                bullet_list(
                    [
                        f"Grafana: {p.display_name} / Overview",
                        f"Grafana: {p.display_name} / Dependencies",
                    ]
                    + ([f"Grafana: {p.display_name} / Database"] if p.postgres_db else [])
                    + (
                        [f"Grafana: Kafka / Consumer group {p.consumer_group}"]
                        if p.consumes
                        else []
                    )
                ),
            )
            + "\n"
            + _h("Alerts", markdown_table(["Alert", "Condition", "Runbook"], rows))
            + "\n"
            + _h(
                "Useful log queries",
                bullet_list(
                    [
                        f"`service:{p.id} level:ERROR | top exception`",
                        f"`service:{p.id} trace_id:<id>` (follow a request across services)",
                    ]
                ),
            )
        )

    return body


def _alert_runbook(ctx: DocContext, p: ServiceProfile, alert_key: str) -> str:
    mapping: dict[str, tuple[str, ...]] = {
        "http_5xx": ("service", p.id, "5xx"),
        "latency": ("service", p.id, "latency"),
        "cpu": ("service", p.id, "latency"),
        "restarts": ("service", p.id, "restarts"),
        "memory": ("service", p.id, "restarts"),
        "upstream": ("category", str(C.DEPENDENCY_FAILURE)),
        "provider": ("category", str(C.DEPENDENCY_FAILURE)),
        "db_pool": ("category", str(C.DB_CONNECTION_EXHAUSTION)),
        "db_deadlock": ("category", str(C.DATABASE_DEADLOCK)),
        "db_errors": ("category", str(C.SCHEMA_MISMATCH)),
        "redis_memory": ("category", str(C.REDIS_MEMORY_PRESSURE)),
        "cache_hit": ("category", str(C.CACHE_INVALIDATION)),
        "consumer_lag": ("category", str(C.KAFKA_CONSUMER_LAG)),
        "consumer_errors": ("category", str(C.KAFKA_CONSUMER_LAG)),
        "auth_failures": ("bespoke", "Authentication Token Failure"),
        "rate_limited": ("bespoke", "Gateway 429 Rate Limit Spikes"),
    }
    overrides = {
        ("payment-service", "http_5xx"): ("bespoke", "Payment API 500 Errors"),
        ("payment-service", "provider"): ("bespoke", "PayFlux Provider Degradation"),
        ("api-gateway", "latency"): ("bespoke", "API Gateway Timeout"),
        ("cart-service", "redis_memory"): ("bespoke", "Cart Redis Memory Pressure"),
        ("search-service", "consumer_lag"): ("bespoke", "Search Indexing Lag"),
        ("notification-service", "consumer_lag"): ("bespoke", "Notification Delivery Delays"),
        ("order-service", "upstream"): ("bespoke", "Checkout Failures"),
    }
    key = overrides.get((p.id, alert_key), mapping[alert_key])
    return ctx.drafts[key].id


def _troubleshooting(p: ServiceProfile) -> Callable[[DocContext], str]:
    def body(ctx: DocContext) -> str:
        items = [
            (
                "Errors started right after a deploy",
                f"Roll back first. See {ctx.ref('category', C.DEPLOYMENT_REGRESSION)}.",
            )
        ]
        if p.postgres_db:
            items += [
                (
                    "`QueuePool limit of size N overflow M reached`",
                    f"Connection pool exhausted; expected pool is {p.config.db_pool_size}+{p.config.db_max_overflow}. "
                    f"See {ctx.ref('category', C.DB_CONNECTION_EXHAUSTION)}.",
                ),
                ("`deadlock detected`", f"See {ctx.ref('category', C.DATABASE_DEADLOCK)}."),
                (
                    "`column ... does not exist` / `NotNullViolation`",
                    f"Schema drift. See {ctx.ref('category', C.SCHEMA_MISMATCH)}.",
                ),
            ]
        if p.redis_cluster:
            items += [
                (
                    "`OOM command not allowed`",
                    f"`{p.redis_cluster}` is full. See {ctx.ref('category', C.REDIS_MEMORY_PRESSURE)}.",
                ),
                (
                    "Customers see outdated data",
                    f"See {ctx.ref('category', C.CACHE_INVALIDATION)}.",
                ),
            ]
        if p.consumes:
            items.append(
                ("Events processed late", f"See {ctx.ref('category', C.KAFKA_CONSUMER_LAG)}.")
            )
        if p.http_dependencies or p.external:
            items.append(
                (
                    "`UpstreamUnavailable` / `ReadTimeout`",
                    f"See {ctx.ref('category', C.DEPENDENCY_FAILURE)} and "
                    f"{ctx.ref('category', C.API_TIMEOUT)}.",
                )
            )
        items.append(
            ("Pods restarting with `OOMKilled`", f"See {ctx.ref('service', p.id, 'restarts')}.")
        )
        body_text = "\n\n".join(f"### {symptom}\n\n{answer}" for symptom, answer in items)
        return f"# {p.display_name} Troubleshooting Guide\n\nFirst stop: {ctx.ref('service', p.id, '5xx')}.\n\n{body_text}\n"

    return body


SERVICE_DOCS: dict[
    DocumentType, tuple[str, Callable[[ServiceProfile], Callable[[DocContext], str]]]
] = {
    D.ARCHITECTURE: ("Architecture", _architecture),
    D.API_REFERENCE: ("API Reference", _api_reference),
    D.SERVICE_BEHAVIOR: ("Service Behavior: Timeouts, Retries and Degradation", _service_behavior),
    D.CONFIGURATION: ("Configuration Reference", _configuration),
    D.DEPLOYMENT: ("Deployment Guide", _deployment_doc),
    D.DATABASE: ("Data Model", _data_doc),
    D.MONITORING: ("Monitoring and Alerts", _monitoring),
    D.TROUBLESHOOTING: ("Troubleshooting Guide", _troubleshooting),
}


# --- platform documentation ---------------------------------------------------------------------


def _platform_docs() -> list[
    tuple[str, str, DocumentType, AccessLevel, str, Callable[[DocContext], str]]
]:
    def static(text: str) -> Callable[[DocContext], str]:
        return lambda ctx: text

    service_rows = [[s.id, s.team, s.tier.value, ", ".join(s.datastores) or "-"] for s in SERVICES]
    edges = [
        [s.id, d.service, "HTTP", d.criticality.value]
        for s in SERVICES
        for d in s.http_dependencies
    ]
    edges += [
        [s.id, TOPIC_PRODUCERS[t], f"Kafka {t}", "soft"] for s in SERVICES for t in s.consumes
    ]
    pg = [s for s in SERVICES if s.postgres_db]
    pg_rows = [
        [
            s.postgres_db or "",
            s.id,
            str(s.config.replicas),
            f"{s.config.db_pool_size}+{s.config.db_max_overflow}",
            str(s.config.replicas * (s.config.db_pool_size + s.config.db_max_overflow)),
        ]
        for s in pg
    ]
    return [
        (
            "system-architecture",
            "NovaCart System Architecture",
            D.ARCHITECTURE,
            AccessLevel.ENGINEERING,
            "platform",
            static(
                "# NovaCart System Architecture\n\nNovaCart runs eleven Python/FastAPI services on Kubernetes (namespace `prod`, "
                "three availability zones). Clients reach them only through `api-gateway`. Synchronous calls use HTTP through "
                "`ResilientHttpClient`; asynchronous flows use Kafka.\n\n"
                + _h(
                    "Services",
                    markdown_table(["Service", "Team", "Tier", "Data stores"], service_rows),
                )
                + "\n"
                + _h(
                    "Checkout flow",
                    _steps(
                        [
                            "Client -> api-gateway (auth, rate limit) -> order-service `POST /v1/orders`.",
                            "order-service loads the cart from cart-service, which prices it via product-service.",
                            "order-service reserves stock in inventory-service (SELECT ... FOR UPDATE in SKU order).",
                            "order-service authorises the payment through payment-service -> PayFlux.",
                            "order-service publishes `order.created`; notification-service sends the confirmation.",
                        ]
                    ),
                )
            ),
        ),
        (
            "service-dependency-map",
            "Service Dependency Map",
            D.ARCHITECTURE,
            AccessLevel.ENGINEERING,
            "platform",
            static(
                "# Service Dependency Map\n\nA hard dependency means the caller fails when the callee fails; a soft one means it "
                "degrades. Incidents in a service usually cascade to its hard dependents.\n\n"
                + markdown_table(["Caller", "Callee", "Via", "Criticality"], edges)
            ),
        ),
        (
            "deployment-pipeline",
            "Deployment Pipeline and Canary Analysis",
            D.DEPLOYMENT,
            AccessLevel.ENGINEERING,
            "platform",
            static(
                "# Deployment Pipeline and Canary Analysis\n\n`deployctl` builds an image per merge to `main` and deploys a "
                "versioned release (semantic versioning per service). tier_0 services use canary analysis: 5% of pods for 10 "
                "minutes, then 25%, then 100%. The step fails if the canary's 5xx ratio exceeds twice the baseline. Failed "
                "canaries are rolled back automatically and the release is marked `failed`.\n\n"
                + _h(
                    "Rollbacks",
                    "`deployctl rollback` redeploys a previous version with a rolling strategy and records a "
                    "deployment of kind `rollback`. The previous commit is redeployed; `main` is not reverted, so a fix PR "
                    "is still required.",
                )
                + "\n"
                + _h(
                    "Change freeze",
                    "No regular deploys from 24 November to 1 December (Black Friday / Cyber Monday). "
                    "Hotfixes and rollbacks are allowed with SRE approval.",
                )
            ),
        ),
        (
            "incident-management",
            "Incident Management Process",
            D.TROUBLESHOOTING,
            AccessLevel.ENGINEERING,
            "sre",
            static(
                "# Incident Management Process\n\n"
                + _h(
                    "Severity",
                    markdown_table(
                        ["Severity", "Definition", "Response"],
                        [
                            [
                                "SEV1",
                                "Critical customer impact (checkout, payments or login down)",
                                "Page SRE + owning team; IC from SRE; postmortem required",
                            ],
                            [
                                "SEV2",
                                "Major degradation of a tier_0/tier_1 feature",
                                "Page owning team; SRE IC; postmortem if deployment-related",
                            ],
                            [
                                "SEV3",
                                "Partial degradation, workaround exists",
                                "Owning team during business hours",
                            ],
                            ["SEV4", "Minor or internal-only impact", "Ticket"],
                        ],
                    ),
                )
                + "\n"
                + _h(
                    "During an incident",
                    _steps(
                        [
                            "Declare in #incidents; the bot opens a channel and an incident record.",
                            "Mitigate first (roll back, scale, fail over); root cause later.",
                            "Link child incidents in dependent services to the parent incident.",
                            "Record the triggering deployment, if any, on the incident.",
                        ]
                    ),
                )
                + "\n"
                + _h(
                    "After",
                    "Postmortems are blameless, due within 5 business days, and list action items with owners.",
                )
            ),
        ),
        (
            "oncall-handbook",
            "On-call Handbook",
            D.TROUBLESHOOTING,
            AccessLevel.ENGINEERING,
            "sre",
            static(
                "# On-call Handbook\n\nEach team runs a weekly rotation (`#<team>-oncall`). SRE runs a follow-the-sun rotation and "
                "provides incident commanders for SEV1/SEV2.\n\n"
                + _h(
                    "First five minutes",
                    _steps(
                        [
                            "Acknowledge the page; open the alert's runbook (linked in the alert).",
                            "Check `deployctl history <service>`: most incidents follow a change.",
                            "Check #incidents for related incidents upstream.",
                            "Decide: roll back, mitigate, or escalate.",
                        ]
                    ),
                )
            ),
        ),
        (
            "observability-stack",
            "Observability Stack",
            D.MONITORING,
            AccessLevel.ENGINEERING,
            "platform",
            static(
                "# Observability Stack\n\n- Logs: JSON lines with `timestamp`, `service`, `level`, `logger`, `message`, `trace_id`, "
                "`span_id`, `version`; queried with `logs query`.\n- Metrics: Prometheus; dashboards in Grafana per service.\n"
                "- Traces: W3C `traceparent` propagated by `novacart_common.tracing`; the same `trace_id` appears in the logs of "
                "every service a request touched.\n- Alerts: Alertmanager -> PagerDuty; each alert links a runbook.\n"
            ),
        ),
        (
            "postgresql-platform",
            "PostgreSQL Platform Guide",
            D.DATABASE,
            AccessLevel.ENGINEERING,
            "platform",
            static(
                "# PostgreSQL Platform Guide\n\nEach service owns one database behind PgBouncer (transaction pooling, port 6432). "
                "Read replicas are exposed as `pgbouncer-<db>-ro` and are read-only.\n\n"
                + _h(
                    "Connection budgets",
                    markdown_table(
                        [
                            "Database",
                            "Service",
                            "Replicas",
                            "Pool per pod",
                            "Max client connections",
                        ],
                        pg_rows,
                    )
                    + "\n\nThe sum must stay below PgBouncer `default_pool_size` (200) for the database. Hard-coding smaller pools "
                    "causes pool exhaustion at peak.",
                )
                + "\n"
                + _h(
                    "Rules",
                    bullet_list(
                        [
                            "Migrations are expand/contract; `CREATE INDEX CONCURRENTLY` only",
                            "Long reporting queries run on the replica",
                            "Row locks are taken in primary-key order",
                        ]
                    ),
                )
            ),
        ),
        (
            "redis-platform",
            "Redis Platform Guide",
            D.DATABASE,
            AccessLevel.ENGINEERING,
            "platform",
            static(
                "# Redis Platform Guide\n\nClusters: "
                + ", ".join(f"`{s.redis_cluster}`" for s in SERVICES if s.redis_cluster)
                + ". `redis-carts` and `redis-payments` use `volatile-lru` (keys without a TTL are never evicted); others use "
                "`allkeys-lru`. Every write must set a TTL. `FLUSHDB`/`KEYS` are disabled via rename-command.\n"
            ),
        ),
        (
            "kafka-platform",
            "Kafka Platform Guide",
            D.DATABASE,
            AccessLevel.ENGINEERING,
            "platform",
            static(
                "# Kafka Platform Guide\n\n"
                + markdown_table(
                    ["Topic", "Partitions", "Producer", "Consumers"],
                    [
                        [
                            t,
                            str(TOPIC_PARTITIONS[t]),
                            TOPIC_PRODUCERS[t],
                            ", ".join(s.id for s in SERVICES if t in s.consumes) or "-",
                        ]
                        for t in sorted(TOPIC_PRODUCERS)
                    ],
                )
                + "\n\nConsumers must be idempotent (at-least-once delivery) and send undeserialisable messages to `<topic>.dlq`. "
                "Consumer throughput is bounded by partition count.\n"
            ),
        ),
        (
            "gateway-routing",
            "API Gateway Routing and Rate Limits",
            D.API_REFERENCE,
            AccessLevel.PUBLIC,
            "platform",
            static(
                "# API Gateway Routing and Rate Limits\n\nPublic API: `https://api.novacart.example/v1/<service>/<path>`. "
                "Requests need a bearer token from `/v1/auth/token` except for catalogue and search reads.\n\n"
                "Rate limits: 50 requests/second per client id with a burst of 100; partners have contracted limits. Exceeding "
                "them returns HTTP 429 with `Retry-After`. Clients must retry with exponential backoff.\n"
            ),
        ),
        (
            "auth-design",
            "Authentication and Token Design",
            D.ARCHITECTURE,
            AccessLevel.SRE,
            "identity",
            static(
                "# Authentication and Token Design\n\nAccess tokens are RS256 JWTs (15-minute TTL) signed by auth-service; "
                "refresh tokens are opaque, single-use and rotate on every refresh (30-day TTL). The gateway validates tokens "
                "locally with the JWKS from auth-service, cached for 300s and refreshed on unknown `kid`. Validation allows 30s "
                "of clock-skew leeway. Signing keys rotate weekly; new keys are published 24h before activation.\n"
            ),
        ),
        (
            "secrets-management",
            "Secrets Management",
            D.CONFIGURATION,
            AccessLevel.CONFIDENTIAL,
            "security",
            static(
                "# Secrets Management\n\nAll credentials live in Vault under `secret/<service>/...` and are injected at pod start. "
                "Provider API keys (PayFlux, RiskShield, MailRelay, TextBridge, StockSync) rotate quarterly; signing keys weekly. "
                "Break-glass access requires two security engineers and is audited. Secrets must never appear in logs, tickets or "
                "this knowledge base.\n"
            ),
        ),
        (
            "migration-policy",
            "Database Migration Policy",
            D.DATABASE,
            AccessLevel.ENGINEERING,
            "platform",
            static(
                "# Database Migration Policy\n\n1. Expand: add nullable columns or new tables; deploy code that writes both.\n"
                "2. Migrate: backfill.\n3. Contract: remove old columns in a later release.\n\nRenaming a column or ORM attribute "
                "in place is forbidden; it breaks running pods during rollout and makes rollbacks unsafe.\n"
            ),
        ),
        (
            "event-schema-policy",
            "Event Schema Evolution Policy",
            D.SERVICE_BEHAVIOR,
            AccessLevel.ENGINEERING,
            "platform",
            static(
                "# Event Schema Evolution Policy\n\nEvents are versioned dataclasses (`SCHEMA_VERSION`). Adding optional fields is "
                "safe. Renaming or removing a field requires a new schema version, dual-publishing both shapes until every consumer "
                "has migrated, and a consumer contract test in CI.\n"
            ),
        ),
        (
            "resilience-standards",
            "Resilience Standards: Timeouts, Retries, Circuit Breakers",
            D.SERVICE_BEHAVIOR,
            AccessLevel.ENGINEERING,
            "platform",
            static(
                "# Resilience Standards\n\n- Timeout per dependency above the callee's p99.9, never below its p99.\n"
                "- At most 2 retries, only for idempotent failures, exponential backoff with full jitter (`backoff * 2**attempt`).\n"
                "- Circuit breaker opens at 50% failures over the last 50 calls; 10s cooldown.\n"
                "- Soft dependencies must have a fallback; hard dependencies must be documented in the service behaviour doc.\n"
                "- Lowering a timeout while raising retries is a classic retry-storm trigger.\n"
            ),
        ),
        (
            "feature-flags",
            "Feature Flags",
            D.CONFIGURATION,
            AccessLevel.ENGINEERING,
            "platform",
            static(
                "# Feature Flags\n\nFlags live in LaunchPad. Production changes above 10% of traffic need a second approver and "
                "are announced in #deploys. Flag changes appear in the incident timeline tool alongside deployments.\n"
            ),
        ),
    ]


# --- postmortems --------------------------------------------------------------------------------


def render_postmortem(incident: IncidentFact) -> str:
    p = SERVICES_BY_ID[incident.service_id]
    timeline: list[tuple[datetime, str]] = []
    deployment, pr, fault = incident.root_cause_deployment, incident.root_cause_pr, incident.fault
    if deployment and pr:
        timeline.append(
            (
                deployment.deployed_at,
                f"{deployment.id} deploys {deployment.service_id} {deployment.version} "
                f'({deployment.strategy.value}), including {pr.id} "{pr.title}".',
            )
        )
    timeline.append((incident.started_at, "First errors observed."))
    timeline.append(
        (
            incident.detected_at,
            f"`{incident.alert_name}` fires; on-call paged."
            if incident.alert_name
            else "Customer support escalates reports to on-call.",
        )
    )
    for child in incident.children:
        timeline.append(
            (child.started_at, f"Downstream impact in {child.service_id} ({child.id}).")
        )
    if incident.remediation:
        verb = "Rollback" if incident.remediation.kind == "rollback" else "Hotfix"
        timeline.append(
            (
                incident.remediation.deployed_at,
                f"{verb} {incident.remediation.id} to {incident.remediation.version} starts.",
            )
        )
    timeline.append((incident.resolved_at, "Metrics back to baseline; incident resolved."))
    if incident.fix_pr and incident.fix_pr.merged_at > incident.resolved_at:
        timeline.append((incident.fix_pr.merged_at, f"Permanent fix {incident.fix_pr.id} merged."))
    timeline.sort(key=lambda item: item[0])
    lines = [f"- {moment:%Y-%m-%d %H:%M} UTC: {text}" for moment, text in timeline]

    if fault and pr and deployment:
        diff = "\n".join(pr.edits[0].patch.splitlines()[:24])
        root = (
            f'{fault.root_cause}\n\nThe change was introduced by {pr.id} ("{pr.title}", author @{pr.author}, '
            f"commit `{pr.merge_commit_sha[:7]}`) and shipped in {deployment.id} ({deployment.version}). "
            f"The author's intent: {pr.rationale}\n\n```diff\n{diff}\n```"
        )
        went_wrong = [
            "The PR was reviewed as a low-risk change; nothing flagged the production impact",
            "Canary analysis did not catch the problem before full rollout"
            if deployment.strategy.value == "canary"
            else "No canary for this service",
        ]
        actions = [
            [f"Add a regression test for `{fault.primary_path.rsplit('/', 1)[-1]}`", pr.author],
            ["Add a CI check or lint rule that catches this class of change", "platform"],
            ["Review alert thresholds that fired late", incident.commander],
        ]
    else:
        assert incident.cause is not None
        root = incident.cause.root_cause
        went_wrong = [
            "Capacity and guard rails did not account for this trigger",
            "Detection relied on symptoms, not causes",
        ]
        actions = [
            ["Add alerting on the underlying cause", p.team],
            ["Document the mitigation in the runbook", incident.commander],
        ]
    impact = incident.metrics
    return (
        f"# Postmortem: {incident.title}\n\n"
        + markdown_table(
            ["Field", "Value"],
            [
                ["Incident", incident.id],
                ["Severity", incident.severity.value],
                ["Service", incident.service_id],
                ["Date", f"{incident.started_at:%Y-%m-%d}"],
                ["Duration", f"{incident.duration_minutes} minutes"],
                ["Incident commander", f"@{incident.commander}"],
                ["Related incidents", ", ".join(c.id for c in incident.children) or "none"],
            ],
        )
        + "\n\n"
        + _h("Summary", f"{incident.symptoms}")
        + "\n"
        + _h(
            "Impact",
            bullet_list(
                [
                    f"{k.replace('_', ' ')}: {v:,}"
                    if isinstance(v, int)
                    else f"{k.replace('_', ' ')}: {v}"
                    for k, v in impact.items()
                ]
            ),
        )
        + "\n"
        + _h("Timeline", "\n".join(lines))
        + "\n"
        + _h("Root cause", root)
        + "\n"
        + _h("Resolution", incident.resolution)
        + "\n"
        + _h(
            "What went well",
            bullet_list(["Detection and paging worked", "Mitigation was applied quickly"]),
        )
        + "\n"
        + _h("What went wrong", bullet_list(went_wrong))
        + "\n"
        + _h(
            "Action items",
            markdown_table(
                ["Action", "Owner", "Status"],
                [[a, f"@{o}" if "." in o else o, "done"] for a, o in actions],
            ),
        )
    )


# --- planning -------------------------------------------------------------------------------------


def plan_documents(rng: random.Random) -> dict[tuple[str, ...], DocumentDraft]:
    drafts: dict[tuple[str, ...], DocumentDraft] = {}
    sres = [m.username for m in team_members("sre")]

    def dates() -> tuple[datetime, datetime, int]:
        created = datetime(2023, 6, 1, tzinfo=UTC) + timedelta(days=rng.randint(0, 780))
        updated = WINDOW_START + timedelta(
            days=rng.randint(0, (WINDOW_END - WINDOW_START).days - 20)
        )
        return created, max(created, updated), rng.randint(2, 14)

    def add(key: tuple[str, ...], prefix: str, **kwargs: object) -> None:
        created, updated, revision = dates()
        number = sum(1 for d in drafts.values() if d.id.startswith(prefix)) + 1
        drafts[key] = DocumentDraft(
            key=key,
            id=f"{prefix}-{number:04d}",
            created_at=created,
            updated_at=updated,
            revision=revision,
            **kwargs,
        )  # type: ignore[arg-type]

    for category, spec in CATEGORY_RUNBOOKS.items():
        title = str(spec["title"])
        add(
            ("category", str(category)),
            "RB",
            doc_type=D.RUNBOOK,
            title=title,
            service_id=None,
            source_path=f"docs/runbooks/{slugify(title)}.md",
            author=rng.choice(sres),
            tags=["runbook", category.value],
            categories=(category,),
            render=_category_runbook_body(category),
            access_level=AccessLevel.SRE
            if category is C.AUTHENTICATION_FAILURE
            else AccessLevel.ENGINEERING,
        )
    kinds = {
        "5xx": (
            C.DEPLOYMENT_REGRESSION,
            C.CONFIGURATION_ERROR,
            C.SCHEMA_MISMATCH,
            C.DEPENDENCY_FAILURE,
        ),
        "latency": (C.API_TIMEOUT, C.NETWORK_TIMEOUT, C.CPU_SATURATION),
        "restarts": (C.MEMORY_LEAK,),
    }
    labels = {
        "5xx": "High 5xx Error Rate",
        "latency": "Elevated Latency",
        "restarts": "Pods Restarting / OOMKilled",
    }
    for p in SERVICES:
        for kind, categories in kinds.items():
            title = f"{p.display_name}: {labels[kind]}"
            add(
                ("service", p.id, kind),
                "RB",
                doc_type=D.RUNBOOK,
                title=title,
                service_id=p.id,
                source_path=f"docs/runbooks/{p.id}/{slugify(labels[kind])}.md",
                author=rng.choice([m.username for m in team_members(p.team)]),
                tags=["runbook", p.id, kind],
                categories=categories,
                link_services=(p.id,),
                render=_service_runbook(p, kind),
                access_level=AccessLevel.SRE if p.id == "auth-service" else AccessLevel.ENGINEERING,
            )
    for rb in _bespoke_runbooks():
        owner = SERVICES_BY_ID[rb.service_id].team if rb.service_id else "sre"
        add(
            ("bespoke", rb.name),
            "RB",
            doc_type=D.RUNBOOK,
            title=rb.name,
            service_id=rb.service_id,
            source_path=f"docs/runbooks/{slugify(rb.name)}.md",
            author=rng.choice([m.username for m in team_members(owner)]),
            tags=["runbook", *(c.value for c in rb.categories)],
            categories=rb.categories,
            link_services=rb.link_services,
            render=_bespoke_body(rb),
            access_level=rb.access_level,
        )
    for p in SERVICES:
        for doc_type, (label, builder) in SERVICE_DOCS.items():
            level = AccessLevel.ENGINEERING
            if doc_type is D.CONFIGURATION and p.access_level is AccessLevel.SRE:
                level = AccessLevel.SRE
            add(
                ("service", p.id, doc_type.value),
                "DOC",
                doc_type=doc_type,
                title=f"{p.display_name} {label}",
                service_id=p.id,
                source_path=f"docs/services/{p.id}/{doc_type.value.replace('_', '-')}.md",
                author=rng.choice([m.username for m in team_members(p.team)]),
                tags=[p.id, doc_type.value],
                render=builder(p),
                access_level=level,
            )
    for slug, title, doc_type, level, owner, builder in _platform_docs():
        add(
            ("platform", slug),
            "DOC",
            doc_type=doc_type,
            title=title,
            service_id=None,
            source_path=f"docs/platform/{slug}.md",
            author=rng.choice([m.username for m in team_members(owner)]),
            tags=["platform", doc_type.value],
            render=builder,
            access_level=level,
        )
    return drafts


def runbook_for(
    drafts: dict[tuple[str, ...], DocumentDraft], service_id: str, category: IncidentCategory
) -> DocumentDraft:
    runbooks = [d for d in drafts.values() if d.doc_type is D.RUNBOOK]
    for predicate in (
        lambda d: (
            d.key[0] == "bespoke" and service_id in d.link_services and category in d.categories
        ),
        lambda d: d.key[0] == "service" and d.service_id == service_id and category in d.categories,
        lambda d: d.key[0] == "category" and category in d.categories,
    ):
        match = next((d for d in runbooks if predicate(d)), None)
        if match is not None:
            return match
    raise LookupError(f"no runbook for {service_id} / {category}")


def postmortem_draft(incident: IncidentFact, number: int, rng: random.Random) -> DocumentDraft:
    created = incident.resolved_at + timedelta(days=rng.randint(2, 6))
    updated = created + timedelta(days=rng.randint(0, 3))
    restricted = (
        incident.category is C.AUTHENTICATION_FAILURE or incident.service_id == "auth-service"
    )
    return DocumentDraft(
        key=("postmortem", incident.id),
        id=f"PM-{number:04d}",
        doc_type=D.POSTMORTEM,
        title=f"Postmortem: {incident.title} ({incident.id})",
        service_id=incident.service_id,
        source_path=f"docs/postmortems/{incident.started_at:%Y-%m-%d}-{incident.id.lower()}.md",
        author=incident.commander,
        tags=["postmortem", incident.category.value, incident.service_id, incident.severity.value],
        access_level=AccessLevel.SRE if restricted else AccessLevel.ENGINEERING,
        created_at=min(created, WINDOW_END),
        updated_at=min(max(updated, created), WINDOW_END),
        revision=rng.randint(1, 3),
        content=render_postmortem(incident),
    )


__all__ = [
    "DocContext",
    "DocumentDraft",
    "Severity",
    "plan_documents",
    "postmortem_draft",
    "runbook_for",
]
