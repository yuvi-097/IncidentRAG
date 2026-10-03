"""Deterministic NL -> SQL for common aggregate questions.

Covers counting, averaging resolution time, "top N" and per-service / per-month
breakdowns over incidents, deployments, rollbacks and pull requests, filtered by the
services, time range and severities found in the question. Anything else returns
None: the agent then says it cannot translate the question instead of guessing.
An LLM planner can replace this later; its SQL passes the same guard.

Literal values come from validated entities only (service ids from the catalog,
severities from an enum, timestamps the code formats), and are quoted anyway.
The generated SQL is shown to the user as evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.agents.entities import QueryEntities


@dataclass(frozen=True)
class SqlPlan:
    sql: str
    description: str  # e.g. "number of incidents for payment-service, 2026-08-01 to 2026-09-01"


@dataclass(frozen=True)
class _Subject:
    name: str
    table: str
    time_column: str
    extra: tuple[str, ...] = ()  # always-on conditions


SUBJECTS = [
    (
        re.compile(r"\broll(?:ed)?[\s-]?backs?\b|\brolled back\b", re.I),
        _Subject(
            "rolled-back deployments", "deployments", "deployed_at", ("status = 'rolled_back'",)
        ),
    ),
    (
        re.compile(r"\bpull requests?\b|\bPRs?\b", re.I),
        _Subject("merged pull requests", "pull_requests", "merged_at", ("merged_at IS NOT NULL",)),
    ),
    (
        re.compile(r"\bdeploy(?:s|ments?)?\b|\breleases?\b", re.I),
        _Subject("deployments", "deployments", "deployed_at"),
    ),
    (
        re.compile(r"\bincidents?\b|\boutages?\b", re.I),
        _Subject("incidents", "incidents", "started_at"),
    ),
]
_AVERAGE_RESOLUTION = re.compile(
    r"\b(?:average|avg|mean|median)\b.*"
    r"\b(?:resolution|resolve|recovery|time to (?:resolve|recover))\b"
    r"|\bMTTR\b",
    re.I,
)
_COUNT = re.compile(r"\bhow many\b|\bcount\b|\bnumber of\b|\btally\b", re.I)
_TOP = re.compile(r"\btop\s+(\d+)\b|\b(?:most|highest)\b", re.I)
_PER_SERVICE = re.compile(r"\b(?:per|by|each)\s+service\b|\bwhich services?\b", re.I)
_PER_MONTH = re.compile(r"\b(?:per|by|each)\s+month\b|\bmonthly\b", re.I)
_PER_SEVERITY = re.compile(r"\b(?:per|by|each)\s+severity\b", re.I)


def _quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _timestamp(value: object, dialect: str) -> str:
    text = value.strftime("%Y-%m-%d %H:%M:%S")  # type: ignore[attr-defined]
    return _quote(text if dialect == "sqlite" else text + "+00")


def _month(column: str, dialect: str) -> str:
    if dialect == "sqlite":
        return f"strftime('%Y-%m', {column})"
    return f"to_char({column}, 'YYYY-MM')"


def build_sql(question: str, entities: QueryEntities, dialect: str) -> SqlPlan | None:
    subject = next((s for pattern, s in SUBJECTS if pattern.search(question)), None)
    if subject is None:
        return None
    conditions = list(subject.extra)
    scope: list[str] = []
    if entities.services:
        conditions.append(f"service_id IN ({', '.join(_quote(s) for s in entities.services)})")
        scope.append(", ".join(entities.services))
    if entities.severities and subject.table == "incidents":
        conditions.append(
            f"severity IN ({', '.join(_quote(s.value) for s in entities.severities)})"
        )
        scope.append("/".join(s.value for s in entities.severities))
    if entities.time_range:
        window = entities.time_range
        conditions.append(f"{subject.time_column} >= {_timestamp(window.since, dialect)}")
        conditions.append(f"{subject.time_column} < {_timestamp(window.until, dialect)}")
        scope.append(f"{window.since:%Y-%m-%d} to {window.until:%Y-%m-%d} ({window.expression})")
    where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
    scope_text = f" ({'; '.join(scope)})" if scope else ""

    if _AVERAGE_RESOLUTION.search(question):
        if subject.table != "incidents":
            return None
        if _PER_SERVICE.search(question):
            return SqlPlan(
                f"SELECT service_id, round(avg(resolution_time_minutes), 1) AS avg_minutes, "
                f"count(*) AS incidents FROM incidents{where} "
                f"GROUP BY service_id ORDER BY avg_minutes DESC",
                f"average resolution time in minutes per service{scope_text}",
            )
        return SqlPlan(
            f"SELECT round(avg(resolution_time_minutes), 1) AS avg_minutes, count(*) AS incidents "
            f"FROM incidents{where}",
            f"average resolution time of incidents in minutes{scope_text}",
        )
    top = _TOP.search(question)
    if top and (_PER_SERVICE.search(question) or re.search(r"\bservices?\b", question, re.I)):
        limit = int(top.group(1)) if top.group(1) else 1
        limit = max(1, min(limit, 50))
        return SqlPlan(
            f"SELECT service_id, count(*) AS n FROM {subject.table}{where} "
            f"GROUP BY service_id ORDER BY n DESC, service_id LIMIT {limit}",
            f"services with the most {subject.name}{scope_text}",
        )
    if _PER_MONTH.search(question):
        month = _month(subject.time_column, dialect)
        return SqlPlan(
            f"SELECT {month} AS month, count(*) AS n FROM {subject.table}{where} "
            f"GROUP BY month ORDER BY month",
            f"{subject.name} per month{scope_text}",
        )
    if _PER_SEVERITY.search(question) and subject.table == "incidents":
        return SqlPlan(
            f"SELECT severity, count(*) AS n FROM incidents{where} "
            "GROUP BY severity ORDER BY severity",
            f"incidents per severity{scope_text}",
        )
    if _PER_SERVICE.search(question):
        return SqlPlan(
            f"SELECT service_id, count(*) AS n FROM {subject.table}{where} "
            f"GROUP BY service_id ORDER BY n DESC, service_id",
            f"{subject.name} per service{scope_text}",
        )
    if _COUNT.search(question):
        return SqlPlan(
            f"SELECT count(*) AS n FROM {subject.table}{where}",
            f"number of {subject.name}{scope_text}",
        )
    return None
