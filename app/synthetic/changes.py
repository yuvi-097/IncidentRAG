"""Code changes carried by pull requests.

* Faults: plausible changes that break production (``FaultPlan``). Their diff goes
  HEAD -> faulty; the matching fix PR diffs faulty -> HEAD, so the repository at
  HEAD always reflects the fixed state.
* Maintenance: routine changes whose *after* state equals HEAD (dependency bumps,
  tuning, tests, logging, docs).

All diffs are real ``difflib`` unified diffs against the rendered repository.
"""

from __future__ import annotations

import ast
import difflib
import random
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from app.schemas.enums import FileChangeType, IncidentCategory
from app.synthetic.catalog import SERVICES_BY_ID, ServiceProfile, consumers_of
from app.synthetic.code_repo import (
    EVENT_FIELDS,
    TABLE_MODELS,
    RepoFile,
    database_url,
    dependency_url_setting,
    event_class,
    primary_table,
    redis_url,
    service_url,
)
from app.synthetic.domain_code import (
    AUTH_FAULTS,
    GATEWAY_RATE_LIMIT_FAULT,
    REGRESSIONS,
    CodeFault,
)

Repo = dict[str, RepoFile]
C = IncidentCategory


@dataclass(frozen=True)
class Replacement:
    old: str
    new: str
    after: str | None = None  # anchor: replace the first ``old`` following it


def apply_replacements(content: str, replacements: list[Replacement], path: str) -> str:
    for rep in replacements:
        start = 0
        if rep.after is not None:
            start = content.find(rep.after)
            if start < 0:
                raise ValueError(f"{path}: anchor not found: {rep.after!r}")
        index = content.find(rep.old, start)
        if index < 0:
            raise ValueError(f"{path}: snippet not found: {rep.old!r}")
        if rep.after is None and content.count(rep.old) != 1:
            raise ValueError(f"{path}: snippet is ambiguous: {rep.old!r}")
        content = content[:index] + rep.new + content[index + len(rep.old) :]
    return content


@dataclass(frozen=True)
class FileEdit:
    path: str
    before: str
    after: str
    change_type: FileChangeType = FileChangeType.MODIFIED

    @property
    def patch(self) -> str:
        diff = difflib.unified_diff(
            self.before.splitlines(keepends=True),
            self.after.splitlines(keepends=True),
            fromfile=f"a/{self.path}",
            tofile=f"b/{self.path}",
        )
        return "".join(diff)

    @property
    def additions(self) -> int:
        return sum(
            1
            for line in self.patch.splitlines()
            if line.startswith("+") and not line.startswith("+++")
        )

    @property
    def deletions(self) -> int:
        return sum(
            1
            for line in self.patch.splitlines()
            if line.startswith("-") and not line.startswith("---")
        )

    def reversed(self) -> FileEdit:
        return FileEdit(self.path, self.after, self.before, self.change_type)


def edit(repo: Repo, path: str, replacements: list[Replacement]) -> FileEdit:
    head = repo[path].content
    return FileEdit(path, head, apply_replacements(head, replacements, path))


@dataclass(frozen=True)
class FaultPlan:
    """A faulty change and everything needed to narrate its consequences."""

    kind: str
    category: IncidentCategory
    service_id: str  # where the change was deployed
    edits: tuple[FileEdit, ...]
    pr_title: str
    pr_rationale: str
    root_cause: str  # mechanism, without incident-specific ids
    fix_title: str
    symptom: str
    exception: str
    error_message: str
    endpoint: str
    status_code: int
    impacted_service_id: str | None = None  # where symptoms appear, if elsewhere
    onset_minutes: tuple[int, int] = (3, 25)
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def symptomatic_service_id(self) -> str:
        return self.impacted_service_id or self.service_id

    @property
    def fix_edits(self) -> tuple[FileEdit, ...]:
        return tuple(e.reversed() for e in self.edits)

    @property
    def primary_path(self) -> str:
        return self.edits[0].path


# --- helpers --------------------------------------------------------------------------


def _pkg(p: ServiceProfile) -> str:
    return f"{p.repo_path}/{p.package}"


def _deploy(p: ServiceProfile) -> str:
    return f"{p.repo_path}/deploy/{p.id}.yaml"


def _write_endpoint(p: ServiceProfile) -> str:
    for ep in p.endpoints:
        if ep.method in {"POST", "PUT", "PATCH"}:
            return f"{ep.method} {ep.path}"
    return f"{p.endpoints[0].method} {p.endpoints[0].path}"


def _read_endpoint(p: ServiceProfile) -> str:
    for ep in p.endpoints:
        if ep.method == "GET":
            return f"GET {ep.path}"
    return _write_endpoint(p)


def _env_line(p: ServiceProfile, key: str, value: str) -> str:
    return f'  {p.env_prefix}_{key}: "{value}"'


def _from_code_fault(
    repo: Repo,
    fault: CodeFault,
    category: IncidentCategory,
    kind: str,
    onset: tuple[int, int] = (3, 25),
) -> FaultPlan:
    p = SERVICES_BY_ID[fault.service_id]
    path = f"{_pkg(p)}/{fault.module}.py"
    return FaultPlan(
        kind=kind,
        category=category,
        service_id=p.id,
        edits=(edit(repo, path, [Replacement(old, new) for old, new in fault.replacements]),),
        pr_title=fault.pr_title,
        pr_rationale=fault.pr_rationale,
        root_cause=fault.root_cause,
        fix_title=fault.fix_title,
        symptom=fault.symptom,
        exception=fault.exception,
        error_message=fault.error_message,
        endpoint=fault.endpoint,
        status_code=fault.status_code,
        onset_minutes=onset,
    )


# --- fault builders -------------------------------------------------------------------
# Each returns None when the fault does not apply to the service.

FaultBuilder = Callable[[ServiceProfile, Repo, random.Random], FaultPlan | None]


def pool_shrink(p: ServiceProfile, repo: Repo, rng: random.Random) -> FaultPlan | None:
    if not p.postgres_db:
        return None
    c = p.config
    new_pool = rng.choice([4, 5])
    path = f"{_pkg(p)}/db/database.py"
    return FaultPlan(
        kind="pool_shrink",
        category=C.DB_CONNECTION_EXHAUSTION,
        service_id=p.id,
        edits=(
            edit(
                repo,
                path,
                [
                    Replacement(
                        "    pool_size=settings.db_pool_size,\n    max_overflow=settings.db_max_overflow,\n",
                        "    # Fewer idle connections per pod so we stay under the PgBouncer client limit.\n"
                        f"    pool_size={new_pool},\n    max_overflow=0,\n",
                    )
                ],
            ),
        ),
        pr_title="Reduce idle DB connections per pod",
        pr_rationale=f"PgBouncer for `{p.postgres_db}` is close to max_client_conn; each pod keeps more idle connections than it needs.",
        root_cause=(
            f"`db/database.py` hard-coded `pool_size={new_pool}` and `max_overflow=0`, bypassing the configured "
            f"`db_pool_size={c.db_pool_size}` / `db_max_overflow={c.db_max_overflow}`. Each pod could hold at most "
            f"{new_pool} connections, so at peak traffic requests queued for a connection and failed after the "
            f"{c.db_pool_timeout_seconds:.0f}s pool timeout."
        ),
        fix_title="Restore configurable DB pool size (revert hard-coded pool_size)",
        symptom=f"requests needing the database failed with HTTP 500 after waiting {c.db_pool_timeout_seconds:.0f}s for a pooled connection",
        exception="sqlalchemy.exc.TimeoutError",
        error_message=f"QueuePool limit of size {new_pool} overflow 0 reached, connection timed out, timeout {c.db_pool_timeout_seconds:.2f}",
        endpoint=_write_endpoint(p),
        status_code=500,
        onset_minutes=(60, 300),
        details={
            "old_pool_size": c.db_pool_size,
            "old_max_overflow": c.db_max_overflow,
            "new_pool_size": new_pool,
            "new_max_overflow": 0,
            "pool_timeout": c.db_pool_timeout_seconds,
        },
    )


def session_leak(p: ServiceProfile, repo: Repo, rng: random.Random) -> FaultPlan | None:
    if not p.postgres_db:
        return None
    model = TABLE_MODELS[primary_table(p)][0]
    path = f"{_pkg(p)}/db/repository.py"
    return FaultPlan(
        kind="session_leak",
        category=C.DB_CONNECTION_EXHAUSTION,
        service_id=p.id,
        edits=(
            edit(
                repo,
                path,
                [
                    Replacement(
                        f"from {p.package}.db.database import session_scope\n",
                        f"from {p.package}.db.database import SessionLocal, session_scope\n",
                    ),
                    Replacement(
                        "        with session_scope() as session:\n"
                        f"            row = session.get({model}, record_id)\n"
                        "            return row.to_dict() if row else None\n",
                        "        session = SessionLocal()\n"
                        f"        row = session.get({model}, record_id)\n"
                        "        if row is None:\n"
                        "            return None\n"
                        "        result = row.to_dict()\n"
                        "        session.close()\n"
                        "        return result\n",
                    ),
                ],
            ),
        ),
        pr_title=f"Short-circuit {model}Repository.get for missing records",
        pr_rationale="Avoids opening a transaction scope for lookups that miss; most misses come from client retries.",
        root_cause=(
            f"`{model}Repository.get` opened a session without a context manager and returned early on a miss "
            "without closing it. Every lookup of a missing record leaked one pooled connection until the pool "
            "was exhausted and all database-backed requests timed out."
        ),
        fix_title=f"Close session on every path in {model}Repository.get",
        symptom="database-backed requests progressively slowed and then failed with pool timeouts; restarts helped temporarily",
        exception="sqlalchemy.exc.TimeoutError",
        error_message=f"QueuePool limit of size {p.config.db_pool_size} overflow {p.config.db_max_overflow} reached, connection timed out, timeout {p.config.db_pool_timeout_seconds:.2f}",
        endpoint=_read_endpoint(p),
        status_code=500,
        onset_minutes=(90, 480),
        details={"model": model},
    )


def request_log_leak(p: ServiceProfile, repo: Repo, rng: random.Random) -> FaultPlan | None:
    path = f"{_pkg(p)}/main.py"
    return FaultPlan(
        kind="request_log_leak",
        category=C.MEMORY_LEAK,
        service_id=p.id,
        edits=(
            edit(
                repo,
                path,
                [
                    Replacement(
                        "logger = get_logger(__name__)\n",
                        "logger = get_logger(__name__)\n\n"
                        "# Recent requests, exposed on /debug/requests to help reproduce customer reports.\n"
                        "_RECENT_REQUESTS: list[dict[str, object]] = []\n",
                    ),
                    Replacement(
                        '    request.state.request_id = request.headers.get("x-request-id") or new_request_id()\n',
                        '    request.state.request_id = request.headers.get("x-request-id") or new_request_id()\n'
                        "    _RECENT_REQUESTS.append(\n"
                        '        {"id": request.state.request_id, "path": request.url.path, "headers": dict(request.headers)}\n'
                        "    )\n",
                    ),
                ],
            ),
        ),
        pr_title="Keep recent requests in memory for /debug/requests",
        pr_rationale="Support asked for a way to look up the last requests a pod served when reproducing customer reports.",
        root_cause=(
            "The request middleware appended every request (including headers) to a module-level list that was "
            f"never trimmed. Pod memory grew linearly with traffic until pods hit their {p.config.memory} limit "
            "and were OOMKilled, causing restarts and dropped in-flight requests."
        ),
        fix_title="Bound the debug request buffer (deque maxlen=200)",
        symptom=f"{p.id} pods were repeatedly OOMKilled; in-flight requests failed with HTTP 502/503 during restarts",
        exception="OOMKilled",
        error_message=f"container {p.id} exceeded memory limit {p.config.memory}",
        endpoint=_read_endpoint(p),
        status_code=503,
        onset_minutes=(360, 2400),
        details={"memory_limit": p.config.memory},
    )


def config_fault(p: ServiceProfile, repo: Repo, rng: random.Random) -> FaultPlan | None:
    variants = []
    if p.postgres_db:
        variants.append("db_readonly")
    if p.redis_cluster:
        variants.append("redis_staging")
    if p.id != "api-gateway" and p.http_dependencies:
        variants.append("dependency_port")
    if not variants:
        return None
    variant = rng.choice(variants)
    deploy = _deploy(p)
    if variant == "db_readonly":
        url = database_url(p)
        bad = url.replace(f"pgbouncer-{p.postgres_db}.", f"pgbouncer-{p.postgres_db}-ro.")
        return FaultPlan(
            kind="config_db_readonly",
            category=C.CONFIGURATION_ERROR,
            service_id=p.id,
            edits=(
                edit(
                    repo,
                    deploy,
                    [
                        Replacement(
                            _env_line(p, "DATABASE_URL", url), _env_line(p, "DATABASE_URL", bad)
                        )
                    ],
                ),
            ),
            pr_title=f"Point {p.id} at the regional PgBouncer endpoint",
            pr_rationale="Part of the regional failover work: services should use the region-local pooler.",
            root_cause=(
                f"`deploy/{p.id}.yaml` set `{p.env_prefix}_DATABASE_URL` to `pgbouncer-{p.postgres_db}-ro`, the "
                "read-only replica pooler. Reads worked, so readiness probes passed, but every write failed."
            ),
            fix_title=f"Point {p.env_prefix}_DATABASE_URL back at the primary pooler",
            symptom="all write requests failed with HTTP 500 while reads kept working",
            exception="psycopg.errors.ReadOnlySqlTransaction",
            error_message="cannot execute INSERT in a read-only transaction",
            endpoint=_write_endpoint(p),
            status_code=500,
            onset_minutes=(2, 12),
            details={"bad_host": f"pgbouncer-{p.postgres_db}-ro"},
        )
    if variant == "redis_staging":
        url = redis_url(p)
        bad = url.replace(".novacart.internal", ".staging.novacart.internal")
        host = f"{p.redis_cluster}.staging.novacart.internal"
        return FaultPlan(
            kind="config_redis_staging",
            category=C.CONFIGURATION_ERROR,
            service_id=p.id,
            edits=(
                edit(
                    repo,
                    deploy,
                    [Replacement(_env_line(p, "REDIS_URL", url), _env_line(p, "REDIS_URL", bad))],
                ),
            ),
            pr_title=f"Consolidate Redis settings for {p.id}",
            pr_rationale="Moves the Redis URL into the shared manifest layout used by other services.",
            root_cause=(
                f"The production manifest was copied from the staging overlay: `{p.env_prefix}_REDIS_URL` pointed at "
                f"`{host}`, which does not resolve from the production network."
            ),
            fix_title=f"Use the production Redis host for {p.id}",
            symptom="requests depending on Redis failed and cache-backed paths fell through to slower code paths",
            exception="redis.exceptions.ConnectionError",
            error_message=f"Error -2 connecting to {host}:6379. Name or service not known.",
            endpoint=_read_endpoint(p),
            status_code=500,
            onset_minutes=(2, 10),
            details={"bad_host": host},
        )
    dep = rng.choice(p.http_dependencies)
    good = service_url(dep.service)
    bad = good.rsplit(":", 1)[0] + ":8080"
    key = dependency_url_setting(dep.service).upper()
    return FaultPlan(
        kind="config_dependency_port",
        category=C.CONFIGURATION_ERROR,
        service_id=p.id,
        edits=(edit(repo, deploy, [Replacement(_env_line(p, key, good), _env_line(p, key, bad))]),),
        pr_title=f"Standardise service URLs in {p.id} manifest",
        pr_rationale="Aligns all in-cluster URLs on the platform's default service port.",
        root_cause=(
            f"`{p.env_prefix}_{key}` was changed to port 8080, but `{dep.service}` listens on "
            f"{SERVICES_BY_ID[dep.service].port}. Every call to {dep.service} was refused."
        ),
        fix_title=f"Restore {dep.service} port in {p.id} manifest",
        symptom=f"calls from {p.id} to {dep.service} failed with connection refused",
        exception="httpx.ConnectError",
        error_message="[Errno 111] Connection refused",
        endpoint=_write_endpoint(p),
        status_code=502,
        onset_minutes=(2, 10),
        details={"dependency": dep.service, "bad_port": 8080},
    )


def orm_column_rename(p: ServiceProfile, repo: Repo, rng: random.Random) -> FaultPlan | None:
    if not p.postgres_db:
        return None
    table = primary_table(p)
    model = TABLE_MODELS[table][0]
    line = "    status: Mapped[str] = mapped_column(String(32), nullable=False)\n"
    return FaultPlan(
        kind="orm_column_rename",
        category=C.SCHEMA_MISMATCH,
        service_id=p.id,
        edits=(
            edit(
                repo,
                f"{_pkg(p)}/db/models.py",
                [
                    Replacement(
                        line, line.replace("status:", "state:"), after=f"class {model}(Base):"
                    )
                ],
            ),
        ),
        pr_title=f"Rename {model}.status to state to match the public API",
        pr_rationale="The API already calls this field `state`; aligning the model removes a mapping layer.",
        root_cause=(
            f"The ORM attribute `{model}.status` was renamed to `state` without a migration, so SQLAlchemy issued "
            f"queries against column `{table}.state`, which does not exist."
        ),
        fix_title=f"Revert {model}.state rename (needs expand/contract migration)",
        symptom=f"every query touching {table} failed with HTTP 500",
        exception="psycopg.errors.UndefinedColumn",
        error_message=f"column {table}.state does not exist",
        endpoint=_read_endpoint(p),
        status_code=500,
        onset_minutes=(1, 8),
        details={"table": table, "model": model},
    )


_EVENT_RENAMES = {
    "amount_minor": "amount_cents",
    "total_minor": "total_cents",
    "order_id": "order_ref",
    "sku": "sku_code",
    "product_id": "product_ref",
    "user_id": "customer_id",
    "status": "state",
}


def event_field_rename(p: ServiceProfile, repo: Repo, rng: random.Random) -> FaultPlan | None:
    options = []
    for topic in p.produces:
        consumers = [c for c in consumers_of(p.id) if topic in c.consumes]
        for name in EVENT_FIELDS[topic]:
            if name in _EVENT_RENAMES and consumers:
                options.append((topic, name, consumers))
    if not options:
        return None
    topic, old_name, consumers = rng.choice(options)
    new_name = _EVENT_RENAMES[old_name]
    impacted = max(consumers, key=lambda c: (c.tier == "tier_0", c.tier == "tier_1", c.id))
    typ = "int" if old_name.endswith("_minor") else "str"
    return FaultPlan(
        kind="event_field_rename",
        category=C.SCHEMA_MISMATCH,
        service_id=p.id,
        edits=(
            edit(
                repo,
                f"{_pkg(p)}/events/schemas.py",
                [
                    Replacement(
                        f"    {old_name}: {typ}\n",
                        f"    {new_name}: {typ}\n",
                        after=f"class {event_class(topic)}:",
                    )
                ],
            ),
        ),
        pr_title=f"Rename {old_name} to {new_name} in {topic} events",
        pr_rationale="Aligns event field names with the public API naming guidelines.",
        root_cause=(
            f"`{event_class(topic)}.{old_name}` was renamed to `{new_name}` in place, without bumping "
            f"SCHEMA_VERSION or dual-publishing. Consumers of `{topic}` "
            f'({", ".join(c.id for c in consumers)}) read `event["{old_name}"]` and failed on every new message.'
        ),
        fix_title=f"Restore {old_name} in {topic} payload (dual-publish {new_name})",
        symptom=f"consumers of {topic} failed to process new events and consumer lag grew continuously",
        exception="KeyError",
        error_message=f"KeyError: '{old_name}'",
        endpoint=f"kafka {topic}",
        status_code=0,
        impacted_service_id=impacted.id,
        onset_minutes=(2, 15),
        details={
            "topic": topic,
            "old_field": old_name,
            "new_field": new_name,
            "consumers": [c.id for c in consumers],
        },
    )


def regression(p: ServiceProfile, repo: Repo, rng: random.Random) -> FaultPlan | None:
    fault = REGRESSIONS.get(p.id)
    if fault is None:
        return None
    onset = (30, 240) if p.id == "inventory-service" else (3, 25)
    return _from_code_fault(repo, fault, C.DEPLOYMENT_REGRESSION, "regression", onset)


def invalidation_expire(p: ServiceProfile, repo: Repo, rng: random.Random) -> FaultPlan | None:
    if not p.redis_cluster:
        return None
    return FaultPlan(
        kind="invalidation_expire",
        category=C.CACHE_INVALIDATION,
        service_id=p.id,
        edits=(
            edit(
                repo,
                f"{_pkg(p)}/cache.py",
                [
                    Replacement(
                        "        self._redis.delete(self._key(entity_id))\n",
                        "        # Refresh the TTL instead of deleting to avoid a cache stampede on hot keys.\n"
                        "        self._redis.expire(self._key(entity_id), self._ttl_seconds)\n",
                    )
                ],
            ),
        ),
        pr_title=f"Avoid cache stampede on {p.short.lower()} invalidation",
        pr_rationale="Deleting hot keys caused bursts of cache misses; extending the TTL keeps them warm.",
        root_cause=(
            f"`{p.short}Cache.invalidate` stopped deleting keys and only refreshed their TTL "
            f"({p.config.cache_ttl_seconds}s), so updated records kept being served from stale cache entries."
        ),
        fix_title="Delete cache entries on invalidation again",
        symptom="customers saw stale data after updates (old prices, profiles or stock)",
        exception="StaleCacheRead",
        error_message="cache.stale_read: cached version older than source of truth",
        endpoint=_read_endpoint(p),
        status_code=200,
        onset_minutes=(60, 1200),
        details={"ttl": p.config.cache_ttl_seconds},
    )


def ttl_removed(p: ServiceProfile, repo: Repo, rng: random.Random) -> FaultPlan | None:
    if not p.redis_cluster:
        return None
    if p.id == "cart-service":
        path, rep = (
            f"{_pkg(p)}/cart_store.py",
            Replacement(
                "        pipe.expire(self._key(cart_id), settings.cache_ttl_seconds)\n", ""
            ),
        )
        what = "`CartStore.add` no longer refreshed the 7-day expiry on cart hashes, so abandoned carts never expired"
    else:
        path, rep = (
            f"{_pkg(p)}/cache.py",
            Replacement(
                "        self._redis.set(self._key(entity_id), payload, ex=self._ttl_seconds)\n",
                "        self._redis.set(self._key(entity_id), payload)\n",
            ),
        )
        what = (
            f"`{p.short}Cache.set` stopped passing `ex=` so cache keys were written without a TTL"
        )
    policy_note = (
        "Because the cluster uses `volatile-lru`, keys without a TTL are never evicted"
        if p.id in {"cart-service", "payment-service"}
        else "Memory filled with keys that never expired"
    )
    return FaultPlan(
        kind="ttl_removed",
        category=C.REDIS_MEMORY_PRESSURE,
        service_id=p.id,
        edits=(edit(repo, path, [rep]),),
        pr_title=f"Simplify {p.short.lower()} Redis writes",
        pr_rationale="Expiry is handled by the cluster eviction policy, so explicit TTLs are redundant.",
        root_cause=f"{what}. {policy_note}; `{p.redis_cluster}` reached maxmemory and writes began failing with OOM errors.",
        fix_title=f"Restore TTL on {p.short.lower()} Redis writes",
        symptom=f"`{p.redis_cluster}` memory climbed steadily until writes failed with OOM errors",
        exception="redis.exceptions.ResponseError",
        error_message="OOM command not allowed when used memory > 'maxmemory'.",
        endpoint=_write_endpoint(p),
        status_code=500,
        onset_minutes=(1440, 5760),
        details={"cluster": p.redis_cluster},
    )


def auth_fault(p: ServiceProfile, repo: Repo, rng: random.Random) -> FaultPlan | None:
    fault = AUTH_FAULTS.get(p.id)
    if fault is None:
        return None
    onset = (60, 4000) if p.id == "api-gateway" else (5, 60)
    return _from_code_fault(repo, fault, C.AUTHENTICATION_FAILURE, "auth_fault", onset)


def timeout_reduced(p: ServiceProfile, repo: Repo, rng: random.Random) -> FaultPlan | None:
    if not (p.http_dependencies or p.external):
        return None
    c = p.config
    new_timeout = round(max(0.2, c.http_timeout_seconds * rng.choice([0.2, 0.25, 0.3])), 2)
    new_retries = c.http_max_retries + 2
    target = p.external[0].name if p.external else p.http_dependencies[0].service
    return FaultPlan(
        kind="timeout_reduced",
        category=C.API_TIMEOUT,
        service_id=p.id,
        edits=(
            edit(
                repo,
                _deploy(p),
                [
                    Replacement(
                        _env_line(p, "HTTP_TIMEOUT_SECONDS", str(c.http_timeout_seconds)),
                        _env_line(p, "HTTP_TIMEOUT_SECONDS", str(new_timeout)),
                    ),
                    Replacement(
                        _env_line(p, "HTTP_MAX_RETRIES", str(c.http_max_retries)),
                        _env_line(p, "HTTP_MAX_RETRIES", str(new_retries)),
                    ),
                ],
            ),
        ),
        pr_title=f"Fail fast on slow upstreams in {p.id}",
        pr_rationale="Lower timeouts free workers sooner; extra retries compensate for transient failures.",
        root_cause=(
            f"`HTTP_TIMEOUT_SECONDS` dropped from {c.http_timeout_seconds}s to {new_timeout}s and retries rose from "
            f"{c.http_max_retries} to {new_retries}. The new timeout was below {target}'s normal p99, so healthy "
            "but slow calls timed out and were retried, multiplying load on the upstream (retry storm)."
        ),
        fix_title=f"Restore {p.id} upstream timeout and retry budget",
        symptom=f"requests from {p.id} to {target} timed out; latency and 504s rose sharply",
        exception="httpx.ReadTimeout",
        error_message=f"timed out after {new_timeout}s calling {target}",
        endpoint=_write_endpoint(p),
        status_code=504,
        onset_minutes=(10, 180),
        details={
            "old_timeout": c.http_timeout_seconds,
            "new_timeout": new_timeout,
            "old_retries": c.http_max_retries,
            "new_retries": new_retries,
            "upstream": target,
        },
    )


def rate_limit_fault(p: ServiceProfile, repo: Repo, rng: random.Random) -> FaultPlan | None:
    if p.id == "api-gateway":
        return _from_code_fault(
            repo, GATEWAY_RATE_LIMIT_FAULT, C.RATE_LIMITING, "gateway_rate_limit", (5, 90)
        )
    if not p.external:
        return None
    provider = p.external[0]
    path = f"{_pkg(p)}/clients/{provider.client_module}.py"
    return FaultPlan(
        kind="retry_storm",
        category=C.RATE_LIMITING,
        service_id=p.id,
        edits=(
            edit(
                repo,
                path,
                [
                    Replacement(
                        "            max_retries=settings.http_max_retries,\n",
                        "            max_retries=5,\n",
                    ),
                    Replacement(
                        "            backoff_seconds=0.2,\n", "            backoff_seconds=0.0,\n"
                    ),
                ],
            ),
        ),
        pr_title=f"Retry {provider.name} calls immediately",
        pr_rationale=f"Backoff added latency to {provider.purpose} calls; transient errors usually clear on the next try.",
        root_cause=(
            f"`{provider.client_class}` retried up to 5 times with no backoff. During a brief {provider.name} slowdown "
            f"the retries multiplied request volume and {provider.name} started rate limiting NovaCart's API key."
        ),
        fix_title=f"Restore jittered backoff for {provider.name} retries",
        symptom=f"{provider.name} returned HTTP 429 to most calls from {p.id}",
        exception="httpx.HTTPStatusError",
        error_message=f"429 Too Many Requests from {provider.name}",
        endpoint=_write_endpoint(p),
        status_code=503,
        onset_minutes=(20, 600),
        details={"provider": provider.name},
    )


def consumer_sequential(p: ServiceProfile, repo: Repo, rng: random.Random) -> FaultPlan | None:
    if not p.consumes:
        return None
    n = p.config.consumer_concurrency
    return FaultPlan(
        kind="consumer_sequential",
        category=C.KAFKA_CONSUMER_LAG,
        service_id=p.id,
        edits=(
            edit(
                repo,
                f"{_pkg(p)}/events/consumer.py",
                [
                    Replacement(
                        f"        self._concurrency = {n}\n",
                        "        # Process sequentially so events for the same entity are handled in order.\n"
                        "        self._concurrency = 1\n",
                    )
                ],
            ),
        ),
        pr_title=f"Process {p.id} events sequentially",
        pr_rationale="Rare out-of-order handling was observed for events of the same entity.",
        root_cause=(
            f"Batch concurrency dropped from {n} to 1, cutting consumer throughput by roughly {n}x. At peak "
            f"volume on {', '.join(p.consumes)} the group `{p.consumer_group}` fell further behind every minute."
        ),
        fix_title="Restore concurrent processing (order per key via partition affinity)",
        symptom=f"consumer lag for `{p.consumer_group}` grew continuously; downstream effects were delayed",
        exception="ConsumerLag",
        error_message=f"consumer group {p.consumer_group} lag above threshold",
        endpoint=f"kafka {p.consumes[0]}",
        status_code=0,
        onset_minutes=(30, 240),
        details={"old_concurrency": n, "topics": list(p.consumes)},
    )


def lock_order(p: ServiceProfile, repo: Repo, rng: random.Random) -> FaultPlan | None:
    if not p.postgres_db:
        return None
    table = primary_table(p)
    return FaultPlan(
        kind="lock_order",
        category=C.DATABASE_DEADLOCK,
        service_id=p.id,
        edits=(
            edit(
                repo,
                f"{_pkg(p)}/db/repository.py",
                [
                    Replacement(
                        "            for record_id in sorted(record_ids):\n",
                        "            for record_id in record_ids:  # keep caller order for the audit trail\n",
                    )
                ],
            ),
        ),
        pr_title="Preserve caller order in bulk status updates",
        pr_rationale="Audit events should be written in the order the caller supplied.",
        root_cause=(
            f"`bulk_update_status` stopped locking `{table}` rows in primary-key order. Concurrent batches touching "
            "overlapping rows acquired locks in opposite orders and deadlocked; PostgreSQL aborted one transaction "
            "of each pair."
        ),
        fix_title="Lock rows in primary-key order again in bulk_update_status",
        symptom=f"a fraction of writes to {table} failed with deadlock errors under concurrent load",
        exception="psycopg.errors.DeadlockDetected",
        error_message="deadlock detected",
        endpoint=_write_endpoint(p),
        status_code=500,
        onset_minutes=(60, 720),
        details={"table": table},
    )


def debug_logging(p: ServiceProfile, repo: Repo, rng: random.Random) -> FaultPlan | None:
    return FaultPlan(
        kind="debug_logging",
        category=C.CPU_SATURATION,
        service_id=p.id,
        edits=(
            edit(
                repo,
                _deploy(p),
                [
                    Replacement(
                        _env_line(p, "LOG_LEVEL", "INFO"), _env_line(p, "LOG_LEVEL", "DEBUG")
                    )
                ],
            ),
        ),
        pr_title=f"Temporarily enable debug logging in {p.id}",
        pr_rationale="Need request-level detail to investigate a customer report.",
        root_cause=(
            f"DEBUG logging in production serialised full request/response payloads on every request. CPU usage hit "
            f"the {p.config.cpu} limit, pods were throttled, and latency rose across all endpoints."
        ),
        fix_title=f"Set {p.id} log level back to INFO",
        symptom=f"{p.id} CPU throttled at its limit and p99 latency rose several-fold",
        exception="CPUThrottling",
        error_message=f"container {p.id} CPU throttled above 80% of periods",
        endpoint=_read_endpoint(p),
        status_code=504,
        onset_minutes=(5, 60),
        details={"cpu_limit": p.config.cpu},
    )


FAULT_BUILDERS: dict[str, tuple[FaultBuilder, float]] = {
    # kind: (builder, relative weight)
    "regression": (regression, 2.2),
    "config": (config_fault, 1.0),
    "pool_shrink": (pool_shrink, 1.5),
    "session_leak": (session_leak, 1.0),
    "request_log_leak": (request_log_leak, 1.0),
    "orm_column_rename": (orm_column_rename, 0.8),
    "event_field_rename": (event_field_rename, 0.8),
    "invalidation_expire": (invalidation_expire, 0.8),
    "ttl_removed": (ttl_removed, 0.8),
    "auth_fault": (auth_fault, 1.5),
    "timeout_reduced": (timeout_reduced, 1.0),
    "rate_limit": (rate_limit_fault, 0.8),
    "consumer_sequential": (consumer_sequential, 1.0),
    "lock_order": (lock_order, 0.8),
    "debug_logging": (debug_logging, 0.6),
}


def build_fault(kind: str, p: ServiceProfile, repo: Repo, rng: random.Random) -> FaultPlan | None:
    return FAULT_BUILDERS[kind][0](p, repo, rng)


def choose_fault(p: ServiceProfile, repo: Repo, rng: random.Random) -> FaultPlan:
    candidates = [
        (kind, weight)
        for kind, (builder, weight) in FAULT_BUILDERS.items()
        if builder(p, repo, random.Random(0)) is not None
    ]
    kinds, weights = zip(*candidates, strict=True)
    kind = rng.choices(kinds, weights=weights)[0]
    plan = build_fault(kind, p, repo, rng)
    assert plan is not None
    return plan


# --- maintenance changes (after == HEAD) ----------------------------------------------------


@dataclass(frozen=True)
class Maintenance:
    title: str
    rationale: str
    labels: tuple[str, ...]
    edits: tuple[FileEdit, ...]


def _reverse_edit(repo: Repo, path: str, head_snippet: str, older_snippet: str) -> FileEdit:
    head = repo[path].content
    return FileEdit(
        path, apply_replacements(head, [Replacement(head_snippet, older_snippet)], path), head
    )


def _bump_dependencies(p: ServiceProfile, repo: Repo, rng: random.Random) -> Maintenance | None:
    path = f"{p.repo_path}/requirements.txt"
    lines = [
        line
        for line in repo[path].content.splitlines()
        if "==" in line and not line.startswith("novacart")
    ]
    line = rng.choice(lines)
    name, version = line.split("==")
    parts = version.split(".")
    parts[-1] = str(max(0, int(re.sub(r"\D", "", parts[-1]) or 0) - rng.randint(1, 3)))
    older = f"{name}=={'.'.join(parts)}"
    if older == line:
        return None
    return Maintenance(
        f"Bump {name.split('[')[0]} to {version}",
        "Routine dependency update (security and bug fixes).",
        ("dependencies",),
        (_reverse_edit(repo, path, line + "\n", older + "\n"),),
    )


def _deploy_tuning(p: ServiceProfile, repo: Repo, rng: random.Random) -> Maintenance | None:
    path = _deploy(p)
    content = repo[path].content
    # Leading newline: "replicas: N" is also a suffix of "min_replicas: N".
    options = [("replicas", f"\nreplicas: {p.config.replicas}\n")]
    for key in (
        "HTTP_TIMEOUT_SECONDS",
        "DB_POOL_SIZE",
        "CACHE_TTL_SECONDS",
        "KAFKA_MAX_POLL_RECORDS",
    ):
        match = re.search(rf'  {p.env_prefix}_{key}: "([\d.]+)"\n', content)
        if match:
            options.append((key, match.group(0)))
    key, head_line = rng.choice(options)
    value = re.search(r"([\d.]+)\"?\n$", head_line).group(1)  # type: ignore[union-attr]
    is_float = "." in value
    old_value = (
        float(value) * rng.choice([0.5, 0.75, 1.5])
        if is_float
        else max(1, int(int(value) * rng.choice([0.5, 0.75, 1.25])))
    )
    old_value_str = f"{old_value:.1f}" if is_float else str(old_value)
    if old_value_str == value:
        return None
    older_line = head_line.replace(value, old_value_str, 1)
    title = f"Tune {key.lower().replace('_', ' ')} for {p.id} ({old_value_str} -> {value})"
    return Maintenance(
        title,
        "Adjusted after reviewing last month's capacity and latency dashboards.",
        ("config", "capacity"),
        (_reverse_edit(repo, path, head_line, older_line),),
    )


def _add_test(p: ServiceProfile, repo: Repo, rng: random.Random) -> Maintenance | None:
    tests = [f for f in repo.values() if f.service_id == p.id and f.kind.value == "test"]
    test = rng.choice(tests)
    functions = [n for n in ast.parse(test.content).body if isinstance(n, ast.FunctionDef)]
    if len(functions) < 2:
        return None
    target = functions[-1]
    lines = test.content.splitlines(keepends=True)
    block = "".join(lines[target.lineno - 1 - len(target.decorator_list) : target.end_lineno])
    before = (
        test.content.replace("\n\n\n" + block, "\n", 1)
        if ("\n\n\n" + block) in test.content
        else None
    )
    if before is None:
        return None
    return Maintenance(
        f"Add test: {target.name.removeprefix('test_').replace('_', ' ')}",
        "Covers a path that was only exercised manually.",
        ("tests",),
        (FileEdit(test.path, before, test.content),),
    )


def _domain_statement(p: ServiceProfile, repo: Repo, rng: random.Random) -> Maintenance | None:
    module = rng.choice(p.domain_modules)
    path = f"{_pkg(p)}/{module}.py"
    source = repo[path].content
    tree = ast.parse(source)
    candidates: list[tuple[str, str, ast.stmt]] = []
    for cls in [n for n in tree.body if isinstance(n, ast.ClassDef)] + [tree]:
        owner = cls.name if isinstance(cls, ast.ClassDef) else None
        for fn in [n for n in cls.body if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)]:
            for stmt in fn.body[1:]:  # keep the first statement (often a docstring)
                if isinstance(stmt, ast.Expr | ast.Raise | ast.Assign) and not isinstance(
                    getattr(stmt, "value", None), ast.Constant
                ):
                    name = f"{owner}.{fn.name}" if owner else fn.name
                    candidates.append((name, type(stmt).__name__, stmt))
    if not candidates:
        return None
    name, kind, stmt = rng.choice(candidates)
    lines = source.splitlines(keepends=True)
    before = "".join(lines[: stmt.lineno - 1] + lines[stmt.end_lineno :])  # type: ignore[operator]
    segment = "".join(lines[stmt.lineno - 1 : stmt.end_lineno])
    if "logger." in segment:
        title = f"Add structured logging to {name}"
    elif kind == "Raise":
        title = f"Reject invalid input earlier in {name}"
    else:
        title = f"{rng.choice(['Refactor', 'Simplify', 'Clean up', 'Tidy'])} {name}"
    return Maintenance(
        title,
        f"Small improvement to `{name}` in `{module}.py`.",
        ("maintenance",),
        (FileEdit(path, before, source),),
    )


def _readme(p: ServiceProfile, repo: Repo, rng: random.Random) -> Maintenance | None:
    path = f"{p.repo_path}/README.md"
    head = repo[path].content
    section = head[head.index("## Running locally") :]
    return Maintenance(
        f"Document local development for {p.id}",
        "Onboarding feedback: the README lacked setup steps.",
        ("docs",),
        (FileEdit(path, head.replace(section, ""), head),),
    )


MAINTENANCE_BUILDERS: tuple[
    tuple[Callable[[ServiceProfile, Repo, random.Random], Maintenance | None], float], ...
] = (
    (_domain_statement, 4.0),
    (_deploy_tuning, 2.0),
    (_bump_dependencies, 2.0),
    (_add_test, 1.0),
    (_readme, 0.3),
)


def choose_maintenance(p: ServiceProfile, repo: Repo, rng: random.Random) -> Maintenance:
    builders, weights = zip(*MAINTENANCE_BUILDERS, strict=True)
    for _ in range(20):
        change = rng.choices(builders, weights=weights)[0](p, repo, rng)
        if change is not None and all(e.patch for e in change.edits):
            return change
    raise RuntimeError(f"could not build a maintenance change for {p.id}")
