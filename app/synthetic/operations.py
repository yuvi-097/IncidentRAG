"""Incidents not caused by a deployment (traffic, infrastructure, providers,
manual changes), cascades into dependent services, and live-deployment lookup."""

from __future__ import annotations

import bisect
import random
from collections.abc import Callable
from datetime import datetime, timedelta

from app.schemas.enums import DependencyCriticality, IncidentCategory, ServiceTier, Severity
from app.synthetic.catalog import (
    SERVICES,
    SERVICES_BY_ID,
    TOPIC_PARTITIONS,
    ServiceProfile,
    dependents_of,
    team_members,
)
from app.synthetic.code_repo import TABLE_MODELS, primary_table
from app.synthetic.facts import DeploymentFact, IncidentFact, OperationalCause
from app.synthetic.timeline import (
    TRAFFIC_EVENTS,
    WINDOW_END,
    choose_severity,
    detection_delay,
    hex_id,
    incident_metrics,
    minutes,
    operational_duration,
    traffic_event_at,
)

C = IncidentCategory
CauseBuilder = Callable[[ServiceProfile, random.Random, str, float], OperationalCause | None]


def _endpoint(p: ServiceProfile, write: bool = False) -> str:
    for ep in p.endpoints:
        if (ep.method != "GET") == write:
            return f"{ep.method} {ep.path}"
    return f"{p.endpoints[0].method} {p.endpoints[0].path}"


# --- cause builders: (profile, rng, trigger, traffic multiplier) -> cause -------------------


def db_exhaustion(
    p: ServiceProfile, rng: random.Random, trigger: str, mult: float
) -> OperationalCause | None:
    if not p.postgres_db:
        return None
    c, table = p.config, primary_table(p)
    variant = rng.choice(["traffic", "long_query", "pgbouncer"])
    message = (
        f"QueuePool limit of size {c.db_pool_size} overflow {c.db_max_overflow} reached, "
        f"connection timed out, timeout {c.db_pool_timeout_seconds:.2f}"
    )
    if variant == "traffic":
        new_replicas = c.replicas + rng.choice([2, 4, 6])
        return OperationalCause(
            "db_exhaustion_traffic",
            trigger,
            f"Traffic on {p.id} reached {mult:.1f}x baseline during {trigger}. With {c.replicas} pods x "
            f"({c.db_pool_size}+{c.db_max_overflow}) connections the pools were saturated and requests waited "
            f"{c.db_pool_timeout_seconds:.0f}s for a connection before failing.",
            f"Scaled {p.id} from {c.replicas} to {new_replicas} replicas and raised PgBouncer default_pool_size for "
            f"`{p.postgres_db}` from 80 to 120; errors stopped as new pods took traffic.",
            "database-backed requests failed with pool timeouts",
            "sqlalchemy.exc.TimeoutError",
            message,
            _endpoint(p, write=True),
            500,
            "db_pool",
            {"new_replicas": new_replicas},
        )
    if variant == "long_query":
        held = rng.randint(6, 40)
        return OperationalCause(
            "db_exhaustion_long_query",
            "ad-hoc reporting query",
            f"An ad-hoc reporting query against `{table}` ran on the primary and held row locks for {held} "
            "minutes. Request transactions blocked behind it kept their pooled connections until the pool was "
            "exhausted.",
            "Found the blocking backend in pg_stat_activity and terminated it with pg_terminate_backend; "
            "reporting access now goes to the read replica.",
            "requests hung and then failed with pool timeouts",
            "sqlalchemy.exc.TimeoutError",
            message,
            _endpoint(p, write=True),
            500,
            "db_pool",
            {"lock_minutes": held, "table": table},
        )
    return OperationalCause(
        "db_exhaustion_pgbouncer",
        "node drain",
        f"A node drain rescheduled all {p.id} pods within a minute; each new pod opened its full pool at once "
        f"and PgBouncer for `{p.postgres_db}` hit max_client_conn.",
        "Raised max_client_conn and added a PodDisruptionBudget so drains move pods gradually.",
        "new pods failed readiness and requests errored while connecting to PgBouncer",
        "psycopg.OperationalError",
        "FATAL: no more connections allowed (max_client_conn)",
        _endpoint(p),
        503,
        "db_pool",
    )


def redis_pressure(
    p: ServiceProfile, rng: random.Random, trigger: str, mult: float
) -> OperationalCause | None:
    if not p.redis_cluster:
        return None
    variant = rng.choice(["growth", "big_keys", "bgsave"])
    base = OperationalCause(
        "",
        trigger,
        "",
        "",
        "Redis writes failed and cache hit ratio dropped",
        "redis.exceptions.ResponseError",
        "OOM command not allowed when used memory > 'maxmemory'.",
        _endpoint(p, write=True),
        500,
        "redis_memory",
        {"cluster": p.redis_cluster},
    )
    if variant == "growth":
        growth = rng.randint(40, 180)
        return OperationalCause(
            "redis_growth",
            trigger,
            f"Key count on `{p.redis_cluster}` grew {growth}% during {trigger}; used_memory reached maxmemory and "
            "hot keys were evicted.",
            f"Scaled `{p.redis_cluster}` to the next node size and shortened TTLs for `{p.short.lower()}:*` keys.",
            base.symptom,
            base.exception,
            base.error_message,
            base.endpoint,
            500,
            "redis_memory",
            {"cluster": p.redis_cluster, "growth_pct": growth},
        )
    if variant == "big_keys":
        size = rng.randint(4, 40)
        return OperationalCause(
            "redis_big_keys",
            "oversized values",
            f"A batch job wrote values of up to {size} MB to `{p.redis_cluster}`; memory fragmentation "
            "(mem_fragmentation_ratio > 1.8) pushed the instance to maxmemory.",
            "Deleted the oversized keys, enabled activedefrag and added a 512 KB value-size guard.",
            base.symptom,
            base.exception,
            base.error_message,
            base.endpoint,
            500,
            "redis_memory",
            {"cluster": p.redis_cluster, "value_mb": size},
        )
    return OperationalCause(
        "redis_bgsave",
        "RDB snapshot",
        f"A BGSAVE on `{p.redis_cluster}` started during peak write load; copy-on-write doubled resident memory "
        "and the instance hit maxmemory.",
        "Moved RDB persistence to the replica only and restarted the primary during low traffic.",
        base.symptom,
        base.exception,
        base.error_message,
        base.endpoint,
        500,
        "redis_memory",
        {"cluster": p.redis_cluster},
    )


def kafka_lag(
    p: ServiceProfile, rng: random.Random, trigger: str, mult: float
) -> OperationalCause | None:
    if not p.consumes:
        return None
    topic = rng.choice(p.consumes)
    partitions = TOPIC_PARTITIONS[topic]
    lag = rng.randint(50_000, 2_000_000)
    variant = rng.choice(["rebalance", "poison", "volume"])
    message = f"consumer group {p.consumer_group} lag {lag} on {topic}"
    if variant == "rebalance":
        return OperationalCause(
            "kafka_rebalance_storm",
            "HPA scale events",
            f"HPA scale events restarted consumer pods repeatedly; group `{p.consumer_group}` rebalanced "
            f"{rng.randint(8, 30)} times in 15 minutes and made little progress, so lag on `{topic}` reached {lag:,} messages.",
            "Switched to the cooperative-sticky assignor with static membership (group.instance.id) and added a "
            "10-minute HPA scale-down stabilisation window.",
            f"processing of {topic} events stalled",
            "ConsumerLag",
            message,
            f"kafka {topic}",
            0,
            "consumer_lag",
            {"topic": topic, "lag": lag},
        )
    if variant == "poison":
        partition = rng.randrange(partitions)
        return OperationalCause(
            "kafka_poison_message",
            "malformed message",
            f"A malformed message on `{topic}` partition {partition} failed deserialisation; the consumer retried "
            "the same batch in a loop and the partition stopped advancing.",
            f"Copied the message to `{topic}.dlq`, skipped its offset, and added dead-letter handling for "
            "deserialisation errors.",
            f"one partition of {topic} stopped advancing",
            "json.JSONDecodeError",
            f"failed to decode message on {topic}[{partition}]",
            f"kafka {topic}",
            0,
            "consumer_lag",
            {"topic": topic, "partition": partition, "lag": lag},
        )
    return OperationalCause(
        "kafka_volume",
        trigger,
        f"{trigger} produced {mult:.1f}x normal volume on `{topic}`; {p.config.replicas} consumer pods could not "
        "keep up.",
        f"Scaled consumers to {partitions} replicas (one per partition of `{topic}`); lag drained within the hour.",
        f"{topic} events were processed late",
        "ConsumerLag",
        message,
        f"kafka {topic}",
        0,
        "consumer_lag",
        {"topic": topic, "lag": lag},
    )


def api_timeout(
    p: ServiceProfile, rng: random.Random, trigger: str, mult: float
) -> OperationalCause | None:
    options = []
    if p.postgres_db:
        options.append("bloat")
    if p.http_dependencies:
        options.append("upstream")
    if not options:
        return None
    if rng.choice(options) == "bloat":
        table = primary_table(p)
        return OperationalCause(
            "timeout_table_bloat",
            "table bloat",
            f"Autovacuum fell behind on `{table}`; bloat pushed the main lookup to a sequential scan and p99 "
            "latency exceeded client timeouts.",
            f"Ran VACUUM (ANALYZE) on `{table}` and lowered autovacuum_vacuum_scale_factor for it; latency "
            "returned to baseline.",
            "p99 latency rose far above SLO and clients timed out",
            "httpx.ReadTimeout",
            "upstream request timeout",
            _endpoint(p),
            504,
            "latency",
            {"table": table},
        )
    dep = rng.choice(p.http_dependencies)
    return OperationalCause(
        "timeout_upstream_slow",
        "noisy neighbour",
        f"`{dep.service}` pods were co-located with a CPU-heavy batch job; its p99 rose above {p.id}'s "
        f"{p.config.http_timeout_seconds}s client timeout and requests piled up.",
        f"Cordoned the noisy node and rescheduled `{dep.service}` pods; added anti-affinity for the batch job.",
        f"calls from {p.id} to {dep.service} timed out",
        "httpx.ReadTimeout",
        f"timed out after {p.config.http_timeout_seconds}s calling {dep.service}",
        _endpoint(p),
        504,
        "upstream",
        {"upstream": dep.service},
        upstream_service_id=dep.service,
    )


def auth_failure(
    p: ServiceProfile, rng: random.Random, trigger: str, mult: float
) -> OperationalCause | None:
    if p.id not in {"api-gateway", "auth-service"}:
        return None
    variant = rng.choice(
        ["rotation", "clock_skew", "stuffing"]
        if p.id == "auth-service"
        else ["rotation", "clock_skew"]
    )
    if variant == "rotation":
        kid = f"key-{hex_id(rng, 8)}"
        return OperationalCause(
            "auth_key_rotation",
            "signing key rotation",
            f"The weekly signing-key rotation activated `{kid}` before it had reached every JWKS cache; tokens "
            "signed with it were rejected by gateway pods holding the previous key set.",
            "Forced a JWKS refresh on all gateway pods and changed the rotation job to publish new keys 24h "
            "before activating them.",
            "newly issued tokens were rejected with HTTP 401",
            "AuthenticationError",
            f"unknown signing key: {kid}",
            "GET /v1/{service}/{path}",
            401,
            "auth_failures",
            {"kid": kid},
        )
    if variant == "clock_skew":
        skew = rng.randint(20, 95)
        return OperationalCause(
            "auth_clock_skew",
            "NTP failure",
            f"NTP sync stopped on several nodes after a network ACL change; clock skew of up to {skew}s made fresh "
            "tokens fail iat/nbf validation.",
            "Restored NTP egress, restarted chrony and cordoned skewed nodes until clocks converged.",
            "intermittent HTTP 401 for valid tokens",
            "jwt.exceptions.ImmatureSignatureError",
            "The token is not yet valid (iat)",
            "POST /v1/auth/introspect",
            401,
            "auth_failures",
            {"skew_seconds": skew},
        )
    rate = rng.randint(800, 6000)
    return OperationalCause(
        "auth_credential_stuffing",
        "credential stuffing",
        f"A credential-stuffing attack ({rate} login attempts/s from rotating IPs) tripped login throttling, "
        "which also blocked legitimate customers sharing carrier-NAT IPs.",
        "Enabled a bot challenge at the edge and scoped throttling to username plus device fingerprint.",
        "legitimate logins were throttled with HTTP 401/429",
        "LoginThrottled",
        "login throttled for source",
        "POST /v1/auth/token",
        401,
        "auth_failures",
        {"attempts_per_second": rate},
    )


def config_error(
    p: ServiceProfile, rng: random.Random, trigger: str, mult: float
) -> OperationalCause | None:
    variant = rng.choice(["flag", "configmap", "certificate"])
    if variant == "flag":
        flag = f"{p.short.lower()}_{rng.choice(['new_pricing_path', 'async_writes', 'v2_serializer', 'batch_lookups'])}"
        return OperationalCause(
            "config_feature_flag",
            "feature flag change",
            f"A LaunchPad flag change enabled `{flag}` for 100% of traffic instead of the planned 5% cohort; the "
            "new code path was not ready for full load.",
            f"Reverted `{flag}` to 5%; production flag changes above 10% now need a second approver.",
            "error rate rose immediately after a feature flag change",
            "RuntimeError",
            f"unhandled error in {flag} path",
            _endpoint(p),
            500,
            "http_5xx",
            {"flag": flag},
        )
    if variant == "configmap":
        return OperationalCause(
            "config_manual_configmap",
            "manual ConfigMap edit",
            f"An engineer edited the `{p.id}` ConfigMap directly (kubectl edit) to test a setting; pods restarted "
            "overnight picked up the change.",
            "Restored the ConfigMap from Git; direct edits in production are now blocked by an admission policy.",
            "a subset of pods behaved differently after restarting",
            "ValueError",
            "invalid configuration value",
            _endpoint(p),
            500,
            "http_5xx",
        )
    return OperationalCause(
        "config_cert_expiry",
        "certificate expiry",
        f"The internal TLS certificate for `{p.id}.novacart.svc` expired; cert-manager renewal had been failing "
        "silently for 10 days.",
        "Renewed the certificate manually, fixed the issuer's permissions and added expiry alerting at 14 days.",
        "callers failed TLS handshakes",
        "ssl.SSLCertVerificationError",
        "certificate has expired",
        _endpoint(p),
        503,
        "http_5xx",
    )


def deadlock(
    p: ServiceProfile, rng: random.Random, trigger: str, mult: float
) -> OperationalCause | None:
    if not p.postgres_db:
        return None
    table = primary_table(p)
    if rng.random() < 0.6:
        job = f"{table}_archival"
        return OperationalCause(
            "deadlock_batch_job",
            "nightly batch job",
            f"The `{job}` job updated `{table}` rows in descending id order while request traffic locked them in "
            "ascending order; concurrent transactions deadlocked.",
            "Paused the job, changed it to lock rows in primary-key order and re-ran it off-peak.",
            f"writes to {table} intermittently failed",
            "psycopg.errors.DeadlockDetected",
            "deadlock detected",
            _endpoint(p, write=True),
            500,
            "db_deadlock",
            {"job": job, "table": table},
        )
    return OperationalCause(
        "deadlock_index_migration",
        "online migration",
        f"A migration creating an index on `{table}` without CONCURRENTLY took locks that combined with request "
        "transactions into deadlocks.",
        "Cancelled the migration and re-ran it with CREATE INDEX CONCURRENTLY during low traffic.",
        f"writes to {table} failed while the migration ran",
        "psycopg.errors.DeadlockDetected",
        "deadlock detected",
        _endpoint(p, write=True),
        500,
        "db_deadlock",
        {"table": table},
    )


def memory_leak(
    p: ServiceProfile, rng: random.Random, trigger: str, mult: float
) -> OperationalCause | None:
    if rng.random() < 0.5:
        return OperationalCause(
            "memory_malloc_arenas",
            "base image upgrade",
            f"A base-image upgrade changed glibc malloc arena behaviour; {p.id} RSS grew steadily until pods hit "
            f"their {p.config.memory} limit and were OOMKilled.",
            "Set MALLOC_ARENA_MAX=2 in the deployment and rolled the pods; memory is flat since.",
            "pods restarted every few hours with OOMKilled",
            "OOMKilled",
            f"container {p.id} exceeded memory limit {p.config.memory}",
            _endpoint(p),
            503,
            "restarts",
        )
    return OperationalCause(
        "memory_http_client_per_request",
        trigger,
        f"A rarely used code path created an httpx client per request; under {trigger} it became hot and "
        "connection pools accumulated until pods were OOMKilled.",
        "Rolled the pods as mitigation and switched the path to the shared client.",
        "memory climbed until pods were OOMKilled",
        "OOMKilled",
        f"container {p.id} exceeded memory limit {p.config.memory}",
        _endpoint(p),
        503,
        "restarts",
    )


def cpu_saturation(
    p: ServiceProfile, rng: random.Random, trigger: str, mult: float
) -> OperationalCause | None:
    if mult > 1.5 or rng.random() < 0.5:
        max_replicas = p.config.replicas * 3
        return OperationalCause(
            "cpu_traffic",
            trigger,
            f"{trigger} traffic ({mult:.1f}x) pushed {p.id} CPU to its {p.config.cpu} limit; the HPA was already at "
            f"its maximum of {max_replicas} replicas.",
            f"Raised HPA max replicas to {max_replicas + p.config.replicas} and increased CPU requests; latency recovered.",
            "CPU throttling and rising latency across endpoints",
            "CPUThrottling",
            f"container {p.id} CPU throttled above 80% of periods",
            _endpoint(p),
            504,
            "cpu",
            {"max_replicas": max_replicas},
        )
    rate = rng.randint(2_000, 15_000)
    return OperationalCause(
        "cpu_bot_scraping",
        "bot scraping",
        f"A scraping botnet sent about {rate} requests/s to {_endpoint(p)}, saturating {p.id} CPU.",
        "Blocked the bot ASNs at the edge and tightened limits for unauthenticated clients.",
        "CPU saturation from unwanted traffic",
        "CPUThrottling",
        f"container {p.id} CPU throttled above 80% of periods",
        _endpoint(p),
        504,
        "cpu",
        {"bot_rps": rate},
    )


def dependency_failure(
    p: ServiceProfile, rng: random.Random, trigger: str, mult: float
) -> OperationalCause | None:
    options = [("external", x) for x in p.external] + [("internal", d) for d in p.http_dependencies]
    if not options:
        return None
    kind, target = rng.choice(options)
    if kind == "external":
        pct = rng.randint(15, 90)
        return OperationalCause(
            "dependency_provider_outage",
            f"{target.name} outage",
            f"{target.name} ({target.purpose}) had a partial outage in its eu-west region; {pct}% of calls "
            "returned 503.",
            f"Waited for {target.name} to recover (their status page confirmed the incident) and replayed "
            "failed operations from the retry queue.",
            f"calls to {target.name} failed",
            "ProviderUnavailable",
            f"{target.name} returned 503 Service Unavailable",
            _endpoint(p, write=True),
            502,
            "provider",
            {"provider": target.name, "failure_pct": pct},
        )
    dep = SERVICES_BY_ID[target.service]
    crit = "hard" if target.criticality is DependencyCriticality.HARD else "soft"
    return OperationalCause(
        "dependency_internal_brownout",
        "node pool upgrade",
        f"`{dep.id}` returned intermittent 503s while its pods were evicted during a node pool upgrade; {p.id} "
        f"has a {crit} dependency on it ({target.purpose}).",
        f"Paused the node pool upgrade and let `{dep.id}` stabilise; {'degraded mode kept core flows working' if crit == 'soft' else 'errors stopped once it recovered'}.",
        f"requests depending on {dep.id} failed",
        "UpstreamUnavailable",
        f"GET {dep.id} failed: 503 Service Unavailable",
        _endpoint(p),
        502 if crit == "hard" else 200,
        "upstream",
        {"upstream": dep.id},
        upstream_service_id=dep.id,
    )


def network_timeout(
    p: ServiceProfile, rng: random.Random, trigger: str, mult: float
) -> OperationalCause | None:
    variant = rng.choice(["az", "dns", "conntrack"])
    if variant == "az":
        pct = rng.randint(3, 30)
        return OperationalCause(
            "network_az_packet_loss",
            "cloud network event",
            f"Packet loss of {pct}% between availability zones us-east-1a and us-east-1c made cross-zone calls from "
            f"{p.id} time out.",
            "The cloud provider resolved the network event; traffic was pinned to healthy zones meanwhile.",
            "intermittent connect timeouts on cross-zone calls",
            "httpx.ConnectTimeout",
            "connection attempt timed out",
            _endpoint(p),
            504,
            "upstream",
            {"packet_loss_pct": pct},
        )
    if variant == "dns":
        return OperationalCause(
            "network_dns",
            "CoreDNS saturation",
            "CoreDNS pods were CPU throttled after a cluster-wide spike in DNS queries; in-cluster lookups timed "
            "out intermittently.",
            "Scaled CoreDNS and enabled NodeLocal DNSCache on all nodes.",
            "sporadic name resolution failures for in-cluster services",
            "httpx.ConnectError",
            "[Errno -3] Temporary failure in name resolution",
            _endpoint(p),
            502,
            "upstream",
        )
    return OperationalCause(
        "network_conntrack",
        "conntrack exhaustion",
        f"The conntrack table on nodes running {p.id} filled up (nf_conntrack: table full), dropping new "
        "connections.",
        "Raised nf_conntrack_max and reduced connection churn by enabling HTTP keep-alive for internal calls.",
        "new connections were dropped",
        "httpx.ConnectTimeout",
        "connection attempt timed out",
        _endpoint(p),
        504,
        "latency",
    )


def rate_limiting(
    p: ServiceProfile, rng: random.Random, trigger: str, mult: float
) -> OperationalCause | None:
    if p.id == "api-gateway":
        if rng.random() < 0.5:
            partner = rng.choice(["partner-marketplace", "partner-affiliates"])
            rps = rng.randint(900, 3000)
            return OperationalCause(
                "ratelimit_partner",
                "partner sync job",
                f"`{partner}` started an uncoordinated catalogue sync at {rps} req/s, far above its allocation; its "
                "requests were throttled and its checkout integration failed.",
                f"Temporarily raised the limit for `{partner}` and agreed a nightly sync window with them.",
                "partner traffic received HTTP 429",
                "RateLimitExceeded",
                "rate limit exceeded",
                "GET /v1/{service}/{path}",
                429,
                "rate_limited",
                {"partner": partner, "rps": rps},
            )
        app_version = f"{rng.randint(5, 7)}.{rng.randint(0, 20)}.0"
        return OperationalCause(
            "ratelimit_mobile_retries",
            f"mobile app {app_version}",
            f"Mobile app {app_version} retried failed requests without backoff, tripling request volume per client; "
            "many customers hit the per-client limit.",
            "Raised per-client limits temporarily; the app hotfix added exponential backoff.",
            "mobile clients received HTTP 429",
            "RateLimitExceeded",
            "rate limit exceeded",
            "GET /v1/{service}/{path}",
            429,
            "rate_limited",
            {"app_version": app_version},
        )
    if not p.external:
        return None
    provider = rng.choice(p.external)
    return OperationalCause(
        "ratelimit_provider",
        trigger,
        f"{provider.name} rate limited NovaCart's API key during {trigger}; request volume exceeded the contracted limit.",
        f"Requested a temporary limit increase from {provider.name} and enabled request smoothing in the client.",
        f"{provider.name} answered HTTP 429",
        "httpx.HTTPStatusError",
        f"429 Too Many Requests from {provider.name}",
        _endpoint(p, write=True),
        503,
        "provider",
        {"provider": provider.name},
    )


def cache_invalidation(
    p: ServiceProfile, rng: random.Random, trigger: str, mult: float
) -> OperationalCause | None:
    if not p.redis_cluster:
        return None
    if rng.random() < 0.6:
        return OperationalCause(
            "cache_race",
            "invalidation race",
            "Update events were handled before the database transaction committed; readers repopulated the cache "
            f"with the old value, which then lived for the full TTL ({p.config.cache_ttl_seconds}s).",
            "Flushed the affected keys and moved invalidation into an after-commit hook.",
            "customers saw outdated data after updates",
            "StaleCacheRead",
            "cache.stale_read: cached version older than source of truth",
            _endpoint(p),
            200,
            None,
        )
    return OperationalCause(
        "cache_flush_stampede",
        "manual cache flush",
        f"A manual FLUSHDB on `{p.redis_cluster}` during maintenance emptied the cache at peak; the miss storm "
        "overloaded the backing store.",
        "Added single-flight cache refills and a pre-warm script for hot keys; FLUSHDB is now disabled via rename-command.",
        "latency spiked after the cache was emptied",
        "CacheMissStorm",
        "cache hit ratio dropped below 20%",
        _endpoint(p),
        504,
        "cache_hit",
    )


def schema_mismatch(
    p: ServiceProfile, rng: random.Random, trigger: str, mult: float
) -> OperationalCause | None:
    if not p.postgres_db:
        return None
    table = primary_table(p)
    column = next(name for name, _ in TABLE_MODELS[table][1] if name != "status")
    revision = f"{rng.randint(3, 30):04d}"
    return OperationalCause(
        "schema_manual_migration",
        "manual migration",
        f"Migration {revision} for `{p.postgres_db}` was applied by hand ahead of the release that needed it; the "
        f"running version inserted rows into `{table}` without the new NOT NULL column.",
        f"Added a server default to the new column in `{table}`; migrations now run only through the deploy pipeline.",
        f"inserts into {table} failed",
        "psycopg.errors.NotNullViolation",
        f'null value in column "{column}_v2" of relation "{table}" violates not-null constraint',
        _endpoint(p, write=True),
        500,
        "db_errors",
        {"table": table, "revision": revision},
    )


CAUSES: dict[IncidentCategory, tuple[CauseBuilder, float]] = {
    C.API_TIMEOUT: (api_timeout, 1.3),
    C.DEPENDENCY_FAILURE: (dependency_failure, 1.2),
    C.DB_CONNECTION_EXHAUSTION: (db_exhaustion, 1.1),
    C.NETWORK_TIMEOUT: (network_timeout, 1.0),
    C.CPU_SATURATION: (cpu_saturation, 1.0),
    C.KAFKA_CONSUMER_LAG: (kafka_lag, 1.1),
    C.REDIS_MEMORY_PRESSURE: (redis_pressure, 1.0),
    C.MEMORY_LEAK: (memory_leak, 0.8),
    C.AUTHENTICATION_FAILURE: (auth_failure, 2.5),
    C.CONFIGURATION_ERROR: (config_error, 0.7),
    C.DATABASE_DEADLOCK: (deadlock, 0.7),
    C.RATE_LIMITING: (rate_limiting, 1.2),
    C.CACHE_INVALIDATION: (cache_invalidation, 0.7),
    C.SCHEMA_MISMATCH: (schema_mismatch, 0.4),
}

_TIER_WEIGHT = {ServiceTier.TIER_0: 1.4, ServiceTier.TIER_1: 1.0, ServiceTier.TIER_2: 0.8}
_PEAK_HOURS = list(range(24))
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


def _commander(severity: Severity, service_id: str, rng: random.Random) -> str:
    if severity in {Severity.SEV1, Severity.SEV2}:
        return rng.choice([m.username for m in team_members("sre")])
    return rng.choice([m.username for m in team_members(SERVICES_BY_ID[service_id].team)])


def _sample_start(rng: random.Random, earliest: datetime) -> tuple[datetime, float, str]:
    if rng.random() < 0.18:
        event = rng.choice(TRAFFIC_EVENTS)
        start = event.start + timedelta(seconds=rng.uniform(0, event.days * 86400))
        return start.replace(microsecond=0), event.multiplier, event.name
    span = (WINDOW_END - timedelta(days=2) - earliest).total_seconds()
    day = earliest + timedelta(seconds=rng.uniform(0, span))
    hour = rng.choices(_PEAK_HOURS, weights=_HOUR_WEIGHTS)[0]
    start = day.replace(
        hour=hour, minute=rng.randrange(60), second=rng.randrange(60), microsecond=0
    )
    event = traffic_event_at(start)
    if event:
        return start, event.multiplier, event.name
    return (
        start,
        rng.uniform(1.3, 2.2),
        rng.choice(
            [
                "the evening traffic peak",
                "a marketing email campaign",
                "a flash sale",
                "the lunchtime peak",
            ]
        ),
    )


def operational_incident(rng: random.Random, earliest: dict[str, datetime]) -> IncidentFact:
    while True:
        p = rng.choices(SERVICES, weights=[_TIER_WEIGHT[s.tier] for s in SERVICES])[0]
        categories = [
            (cat, w)
            for cat, (builder, w) in CAUSES.items()
            if builder(p, random.Random(0), "probe", 1.0) is not None
        ]
        cats, weights = zip(*categories, strict=True)
        category = rng.choices(cats, weights=weights)[0]
        started, mult, trigger = _sample_start(rng, earliest[p.id])
        if started < earliest[p.id]:
            continue
        cause = CAUSES[category][0](p, rng, trigger, mult)
        if cause is None:
            continue
        severity = choose_severity(p, rng)
        detected = started + detection_delay(severity, cause.alert_key, rng)
        resolved = max(detected + minutes(10), started + operational_duration(severity, rng))
        return IncidentFact(
            service_id=p.id,
            category=category,
            severity=severity,
            started_at=started,
            detected_at=detected,
            resolved_at=resolved,
            commander=_commander(severity, p.id, rng),
            alert_key=cause.alert_key,
            metrics=incident_metrics(p, severity, category, rng),
            root_cause_service_id=cause.upstream_service_id or p.id,
            cause=cause,
            traffic_event=(e.name if (e := traffic_event_at(started)) else None),
        )


_LATENCY_CATEGORIES = {
    C.API_TIMEOUT,
    C.NETWORK_TIMEOUT,
    C.CPU_SATURATION,
    C.DB_CONNECTION_EXHAUSTION,
}
_LOWER = {
    Severity.SEV1: Severity.SEV2,
    Severity.SEV2: Severity.SEV3,
    Severity.SEV3: Severity.SEV4,
    Severity.SEV4: Severity.SEV4,
}


def cascades(parent: IncidentFact, rng: random.Random, force: bool = False) -> list[IncidentFact]:
    """Child incidents in services that depend on ``parent.service_id``."""
    children: list[IncidentFact] = []
    if parent.parent is not None:
        return children
    fault = parent.fault
    if fault is not None and fault.kind == "event_field_rename":
        others = [c for c in fault.details["consumers"] if c != parent.service_id]
        candidates = [(SERVICES_BY_ID[c], C.SCHEMA_MISMATCH) for c in others]
        probability = 1.0
    else:
        probability = {Severity.SEV1: 0.75, Severity.SEV2: 0.3}.get(parent.severity, 0.0)
        category = (
            C.API_TIMEOUT
            if parent.category in _LATENCY_CATEGORIES and rng.random() < 0.5
            else C.DEPENDENCY_FAILURE
        )
        candidates = [(d, category) for d in dependents_of(parent.service_id)]
    for dependent, category in candidates:
        if not force and rng.random() >= probability:
            continue
        started = parent.started_at + minutes(rng.randint(2, 8))
        if started >= parent.resolved_at - minutes(5):
            continue
        severity = parent.severity if rng.random() < 0.4 else _LOWER[parent.severity]
        detected = started + minutes(rng.randint(1, 5))
        child = IncidentFact(
            service_id=dependent.id,
            category=category,
            severity=severity,
            started_at=started,
            detected_at=detected,
            resolved_at=parent.resolved_at + minutes(rng.randint(0, 15)),
            commander=parent.commander
            if severity in {Severity.SEV1, Severity.SEV2}
            else _commander(severity, dependent.id, rng),
            alert_key="consumer_errors" if category is C.SCHEMA_MISMATCH else "upstream",
            metrics=incident_metrics(dependent, severity, category, rng),
            root_cause_service_id=parent.root_cause_service_id or parent.service_id,
            parent=parent,
            root_cause_deployment=parent.root_cause_deployment,
            root_cause_pr=parent.root_cause_pr,
            fault=parent.fault if category is C.SCHEMA_MISMATCH else None,
            traffic_event=parent.traffic_event,
            anchor=parent.anchor,
        )
        parent.children.append(child)
        children.append(child)
    return children


# --- live deployment lookup -------------------------------------------------------------------


class LiveDeployments:
    """Which deployment of a service was serving traffic at a given moment."""

    def __init__(self, deployments: list[DeploymentFact]) -> None:
        self._by_service: dict[str, tuple[list[datetime], list[DeploymentFact]]] = {}
        for service_id in {d.service_id for d in deployments}:
            live = sorted(
                (d for d in deployments if d.service_id == service_id and d.went_live),
                key=lambda d: d.deployed_at,
            )
            self._by_service[service_id] = ([d.deployed_at for d in live], live)

    def at(self, service_id: str, moment: datetime) -> DeploymentFact | None:
        times, live = self._by_service[service_id]
        index = bisect.bisect_right(times, moment) - 1
        return live[index] if index >= 0 else None

    def first_time(self, service_id: str) -> datetime:
        return self._by_service[service_id][0][0]
