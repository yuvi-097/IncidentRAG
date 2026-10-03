"""Renders human-readable text once ids exist: incident fields, pull request
descriptions and deployment changelogs.

Incident root-cause text deliberately varies in how much it spells out. Some name
the PR, commit and file; others only the version, so answering "which change
caused this?" requires following deployment -> pull request -> file links.
"""

from __future__ import annotations

import random
from collections.abc import Mapping

from app.schemas.enums import AccessLevel, DependencyCriticality, IncidentCategory, Severity
from app.synthetic.catalog import SERVICES_BY_ID, alert_names
from app.synthetic.facts import DeploymentFact, IncidentFact, PullRequestFact
from app.synthetic.text import slugify

C = IncidentCategory

IMPACT = {
    "payment-service": "Customers could not complete card payments; checkout conversion dropped sharply.",
    "order-service": "Checkouts failed and orders were not created.",
    "api-gateway": "Client traffic through the public API was affected across all features.",
    "auth-service": "Customers could not sign in or refresh their sessions.",
    "cart-service": "Customers could not view or modify their carts.",
    "inventory-service": "Stock reservations and availability checks were affected.",
    "product-service": "Product pages and pricing were affected.",
    "user-service": "Account pages and profile lookups were affected.",
    "search-service": "Product search was degraded.",
    "recommendation-service": "Recommendation carousels were empty or generic.",
    "notification-service": "Order and payment emails and SMS were delayed.",
}

TITLES: dict[IncidentCategory, list[str]] = {
    C.DB_CONNECTION_EXHAUSTION: [
        "{display} requests failing with database connection pool timeouts",
        "Elevated HTTP 500s on {endpoint} ({service})",
        "{display}: DB connection pool exhausted",
    ],
    C.REDIS_MEMORY_PRESSURE: [
        "{cluster} at maxmemory; {service} writes failing",
        "Redis OOM errors in {service}",
        "{display} degraded by Redis memory pressure",
    ],
    C.KAFKA_CONSUMER_LAG: [
        "{service} consumer lag growing",
        "Delayed event processing in {service}",
        "{display}: Kafka consumer group falling behind",
    ],
    C.API_TIMEOUT: [
        "{display} latency spike and timeouts on {endpoint}",
        "Timeouts on {endpoint} ({service})",
        "{service} p99 latency far above SLO",
    ],
    C.AUTHENTICATION_FAILURE: [
        "Spike in HTTP 401 responses from {service}",
        "Token validation failures at {service}",
        "{display}: customers unexpectedly signed out",
    ],
    C.DEPLOYMENT_REGRESSION: [
        "{display}: {endpoint} returning HTTP {status}",
        "Errors on {endpoint} after {service} release",
        "Elevated HTTP {status}s from {service}",
    ],
    C.CONFIGURATION_ERROR: [
        "{display} failing after configuration change",
        "{service}: HTTP {status}s due to misconfiguration",
        "Errors on {endpoint} ({service})",
    ],
    C.DATABASE_DEADLOCK: [
        "Deadlocks on {table} in {service}",
        "{display}: intermittent write failures (deadlock detected)",
    ],
    C.MEMORY_LEAK: [
        "{service} pods OOMKilled repeatedly",
        "{display} memory growth and pod restarts",
    ],
    C.CPU_SATURATION: [
        "{service} CPU saturation and elevated latency",
        "{display} throttled at CPU limit",
    ],
    C.DEPENDENCY_FAILURE: [
        "{display} errors caused by {upstream} failures",
        "{service}: upstream {upstream} unavailable",
    ],
    C.NETWORK_TIMEOUT: [
        "Network timeouts affecting {service}",
        "{display}: intermittent connect timeouts",
    ],
    C.RATE_LIMITING: ["HTTP 429 spike affecting {service}", "{display} requests rate limited"],
    C.CACHE_INVALIDATION: [
        "Stale data served by {service}",
        "{display}: updates not visible to customers",
    ],
    C.SCHEMA_MISMATCH: [
        "{service} failing on schema mismatch",
        "{display}: {subject} contract mismatch errors",
    ],
}

SHORT_MECHANISM = {
    C.DB_CONNECTION_EXHAUSTION: "the release changed how database connections are pooled and pods ran out of connections under load",
    C.MEMORY_LEAK: "memory grew without bound after the release until pods were OOMKilled",
    C.CONFIGURATION_ERROR: "a value in the deploy manifest was wrong for production",
    C.SCHEMA_MISMATCH: "code and the data contract it depends on went out of sync",
    C.DEPLOYMENT_REGRESSION: "a code change broke a request path that existing clients still use",
    C.CACHE_INVALIDATION: "cache entries stopped being invalidated after updates",
    C.REDIS_MEMORY_PRESSURE: "Redis keys stopped expiring after the release",
    C.AUTHENTICATION_FAILURE: "token validation behaviour changed in the release",
    C.API_TIMEOUT: "upstream timeout and retry settings changed in the release",
    C.RATE_LIMITING: "retry or rate-limit behaviour changed and requests were throttled",
    C.KAFKA_CONSUMER_LAG: "consumer throughput dropped after the release",
    C.DATABASE_DEADLOCK: "row-lock ordering changed in the release",
    C.CPU_SATURATION: "logging verbosity was raised in production and saturated CPU",
}


def _sentence(text: str) -> str:
    text = text.strip()
    return (text[0].upper() + text[1:]).rstrip(".") + "."


def _endpoint_and_status(incident: IncidentFact) -> tuple[str, int]:
    if incident.fault and incident.parent is None:
        return incident.fault.endpoint, incident.fault.status_code
    if incident.cause:
        return incident.cause.endpoint, incident.cause.status_code
    p = SERVICES_BY_ID[incident.service_id]
    return f"{p.endpoints[0].method} {p.endpoints[0].path}", 502


def render_title(incident: IncidentFact, rng: random.Random) -> str:
    p = SERVICES_BY_ID[incident.service_id]
    endpoint, status = _endpoint_and_status(incident)
    details = {
        **(incident.fault.details if incident.fault else {}),
        **(incident.cause.details if incident.cause else {}),
    }
    upstream = (
        incident.parent.service_id
        if incident.parent
        else (incident.root_cause_service_id or "upstream")
    )
    if (
        upstream == incident.service_id
        and incident.cause
        and incident.cause.details.get("provider")
    ):
        upstream = incident.cause.details["provider"]
    values = {
        "display": p.display_name,
        "service": p.id,
        "endpoint": endpoint,
        "status": status or 500,
        "cluster": p.redis_cluster or "Redis",
        "table": details.get("table", "primary table"),
        "upstream": upstream,
        "subject": details.get("topic") or details.get("table") or "data",
    }
    deployment = incident.root_cause_deployment
    if (
        deployment
        and incident.parent is None
        and deployment.service_id == incident.service_id
        and status >= 500
        and (incident.anchor or rng.random() < 0.35)
    ):
        return f"{p.short} requests returning HTTP {status} after {deployment.version} deploy"
    return rng.choice(TITLES[incident.category]).format(**values)


def render_symptoms(incident: IncidentFact, rng: random.Random) -> str:
    p = SERVICES_BY_ID[incident.service_id]
    parts = []
    if incident.alert_name:
        parts.append(f"Alert `{incident.alert_name}` fired at {incident.detected_at:%H:%M} UTC.")
    else:
        parts.append(
            f"Reported through customer support at {incident.detected_at:%H:%M} UTC "
            f"({rng.randint(4, 60)} tickets) before any alert fired."
        )
    if incident.parent is not None:
        upstream = SERVICES_BY_ID[incident.parent.service_id]
        parts.append(
            f"Calls from {p.id} to {upstream.id} started failing at {incident.started_at:%H:%M} UTC."
        )
    elif incident.fault:
        parts.append(_sentence(incident.fault.symptom))
    elif incident.cause:
        parts.append(_sentence(incident.cause.symptom))
    m = incident.metrics
    if incident.category is C.KAFKA_CONSUMER_LAG and "max_consumer_lag_messages" in m:
        parts.append(f"Consumer lag peaked at {m['max_consumer_lag_messages']:,} messages.")
    elif incident.category is C.CACHE_INVALIDATION:
        parts.append(
            f"An estimated {m.get('stale_reads_estimated', 0):,} reads returned stale data; error rate stayed near zero."
        )
    else:
        parts.append(
            f"Peak error rate {m['peak_error_rate_pct']}%, p99 latency {m['p99_latency_ms']:,} ms "
            f"(SLO {p.config.p99_latency_slo_ms} ms), {m['failed_requests']:,} failed requests."
        )
    parts.append(IMPACT[p.id])
    deployment = incident.root_cause_deployment
    if deployment and incident.parent is None and (incident.anchor or rng.random() < 0.6):
        parts.append(
            f"Most recent change to {deployment.service_id}: {deployment.id} ({deployment.version}) "
            f"deployed at {deployment.deployed_at:%Y-%m-%d %H:%M} UTC."
        )
    if incident.traffic_event:
        parts.append(f"Occurred during {incident.traffic_event}.")
    return " ".join(parts)


def render_root_cause(incident: IncidentFact, rng: random.Random) -> str:
    if incident.parent is not None:
        parent = incident.parent
        upstream = SERVICES_BY_ID[parent.service_id]
        if incident.category is C.SCHEMA_MISMATCH and incident.fault:
            return (
                f"Same root cause as {parent.id}: {incident.fault.root_cause} {incident.service_id} is one of "
                f"the consumers of `{incident.fault.details['topic']}`."
            )
        dependency = next(
            (
                d
                for d in SERVICES_BY_ID[incident.service_id].http_dependencies
                if d.service == upstream.id
            ),
            None,
        )
        relation = (
            f"{incident.service_id} has a {dependency.criticality.value} dependency on {upstream.id} "
            f"({dependency.purpose})."
            if dependency
            else ""
        )
        return f"Downstream impact of {parent.id} in {upstream.id} ({parent.title}). {relation}".strip()
    if incident.fault:
        fault, deployment, pr = (
            incident.fault,
            incident.root_cause_deployment,
            incident.root_cause_pr,
        )
        assert deployment is not None and pr is not None
        style = (
            "version"
            if incident.anchor
            else rng.choices(["full", "version", "minimal"], weights=[45, 35, 20])[0]
        )
        if style == "full":
            return (
                f"Deployment {deployment.id} ({deployment.service_id} {deployment.version}) shipped {pr.id} "
                f'"{pr.title}" (commit {pr.merge_commit_sha[:7]}), which changed `{fault.primary_path}`. '
                f"{fault.root_cause}"
            )
        if style == "version":
            return (
                f"Regression introduced by {deployment.service_id} {deployment.version} ({deployment.id}): "
                f"{SHORT_MECHANISM[incident.category]}. Details in the linked pull request."
            )
        return f"Introduced by deployment {deployment.id}; {SHORT_MECHANISM[incident.category]}."
    assert incident.cause is not None
    return incident.cause.root_cause


_CLEANUP = {
    C.REDIS_MEMORY_PRESSURE: "Removed keys written without a TTL using a SCAN-based cleanup script.",
    C.CACHE_INVALIDATION: "Flushed the stale cache entries once the fix was live.",
}


def render_resolution(
    incident: IncidentFact, runbook: tuple[str, str] | None, rng: random.Random
) -> str:
    parts: list[str] = []
    if incident.parent is not None:
        parent = incident.parent
        if incident.category is C.SCHEMA_MISMATCH and parent.remediation:
            parts.append(
                f"Recovered after the fix for {parent.id} ({parent.remediation.id}); failed messages were "
                "replayed from the dead-letter topic."
            )
        else:
            dependency = next(
                (
                    d
                    for d in SERVICES_BY_ID[incident.service_id].http_dependencies
                    if d.service == parent.service_id
                ),
                None,
            )
            if dependency and dependency.criticality is DependencyCriticality.SOFT:
                parts.append(
                    f"Served degraded responses (fallback path) until {parent.id} was mitigated."
                )
            else:
                parts.append(
                    f"No change needed in {incident.service_id}; errors stopped when {parent.id} was "
                    f"mitigated at {parent.resolved_at:%H:%M} UTC."
                )
    elif incident.fault:
        remediation, fix = incident.remediation, incident.fix_pr
        assert remediation is not None and fix is not None
        if remediation.kind == "rollback":
            parts.append(
                f"Rolled back {remediation.service_id} to {remediation.version} ({remediation.id}) at "
                f"{remediation.deployed_at:%H:%M} UTC; metrics recovered by {incident.resolved_at:%H:%M} UTC."
            )
            if fix.deployment is not None:
                parts.append(
                    f'Permanent fix {fix.id} ("{fix.title}") shipped in {fix.deployment.id} '
                    f"({fix.deployment.version})."
                )
            else:
                parts.append(f'Permanent fix {fix.id} ("{fix.title}") merged; awaiting release.')
        else:
            parts.append(
                f"Shipped hotfix {remediation.id} ({remediation.version}) containing {fix.id} "
                f'("{fix.title}"); recovered by {incident.resolved_at:%H:%M} UTC.'
            )
        if incident.category in _CLEANUP:
            parts.append(_CLEANUP[incident.category])
        if incident.fault.kind == "regression" and incident.fault.service_id == "inventory-service":
            parts.append(
                "Ran a full stock reconciliation against StockSync to correct drifted levels."
            )
    else:
        assert incident.cause is not None
        parts.append(incident.cause.resolution)
    if runbook and rng.random() < 0.6:
        parts.append(f"Responders followed {runbook[0]} ({runbook[1]}).")
    return " ".join(parts)


def render_incident(
    incident: IncidentFact, runbooks: Mapping[str, tuple[str, str]], rng: random.Random
) -> None:
    p = SERVICES_BY_ID[incident.service_id]
    incident.alert_name = alert_names(p).get(incident.alert_key) if incident.alert_key else None
    incident.title = render_title(incident, rng)
    incident.symptoms = render_symptoms(incident, rng)
    incident.root_cause = render_root_cause(incident, rng)
    runbook = runbooks.get(incident.runbook_id) if incident.runbook_id else None
    incident.resolution = render_resolution(
        incident, (incident.runbook_id, runbook[1]) if runbook else None, rng
    )
    tags = [incident.category.value, incident.service_id]
    if incident.is_deployment_related:
        tags.append("deployment")
    if incident.parent is not None:
        tags.append("cascade")
    if incident.alert_name is None:
        tags.append("customer-reported")
    if incident.traffic_event:
        tags.append(slugify(incident.traffic_event))
    incident.tags = tags


def incident_access_level(incident: IncidentFact) -> AccessLevel:
    if incident.category is C.AUTHENTICATION_FAILURE or incident.service_id == "auth-service":
        return AccessLevel.SRE
    return AccessLevel.ENGINEERING


def needs_postmortem(incident: IncidentFact) -> bool:
    if incident.parent is not None:
        return False
    return incident.severity is Severity.SEV1 or (
        incident.severity is Severity.SEV2 and incident.is_deployment_related
    )


# --- pull requests and deployments -------------------------------------------------------------


def render_pull_request(pr: PullRequestFact) -> None:
    files = "\n".join(f"- `{e.path}` (+{e.additions}/-{e.deletions})" for e in pr.edits)
    if pr.fixes is not None:
        incident, fault = pr.fixes, pr.fixes.fault
        assert fault is not None and incident.root_cause_pr is not None
        culprit = incident.root_cause_pr
        deployment = incident.root_cause_deployment
        pr.description = (
            f"Fixes {incident.id}.\n\n{fault.root_cause}\n\n"
            f'The problem was introduced by {culprit.id} ("{culprit.title}"), shipped in '
            f"{deployment.id} ({deployment.service_id} {deployment.version}).\n\n"
            f"## Changes\n{files}\n\n## Testing\n- Reproduced the failure in staging before the change\n"
            "- Verified the fix in staging under replayed production traffic\n"
        )
    else:
        testing = (
            "- Unit tests pass\n- Verified in staging"
            if "tests" not in pr.labels
            else "- New test passes locally and in CI"
        )
        pr.description = (
            f"{_sentence(pr.rationale)}\n\n## Changes\n{files}\n\n## Testing\n{testing}\n"
        )


def render_deployment(
    deployment: DeploymentFact, incident_by_remediation: Mapping[int, IncidentFact]
) -> None:
    incident = incident_by_remediation.get(id(deployment))
    if deployment.kind == "rollback":
        target = deployment.rollback_of
        assert target is not None
        reason = f" for {incident.id}" if incident else ""
        deployment.changes = (
            f"Rollback to {deployment.version}{reason}, reverting {target.id} ({target.version}).\n"
            f"Executed by @{deployment.author}."
        )
        return
    prs = deployment.pull_requests or deployment.carried_pull_requests
    lines = [f"- {pr.id}: {pr.title} (@{pr.author})" for pr in prs]
    if deployment.kind == "hotfix":
        header = f"Hotfix {deployment.version}" + (f" for {incident.id}" if incident else "")
    else:
        header = f"Release {deployment.version}"
    body = header + "\n" + "\n".join(lines)
    if not deployment.went_live:
        shipped = prs[0].deployment if prs else None
        body += "\nRollout aborted at the 5% canary step: error-rate analysis failed. " + (
            f"Changes re-shipped in {shipped.id}." if shipped else ""
        )
    deployment.changes = body
