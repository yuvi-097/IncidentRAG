"""Stage 2: turn a raw record into canonical text plus source-level metadata.

Documents and code files are used verbatim. Incidents, deployments and pull
requests are rendered as Markdown (a field table plus sections), so they chunk with the
same structure-aware logic and every fact in a chunk is labelled. Rendering is
deterministic: re-parsing a record reproduces the text that chunk offsets refer to.

Malformed records raise ``SkipSource`` with a reason; they never abort ingestion.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from app.rag.chunking.base import SourceFormat
from app.rag.ingestion.models import ParsedSource, RawSource, SkipSource
from app.rag.ingestion.sources import document_source_type
from app.schemas.enums import UNLABELLED_LEVEL, AccessLevel, SourceType, most_restrictive

_CODE_FORMATS = {
    "python": SourceFormat.PYTHON,
    "yaml": SourceFormat.YAML,
    "markdown": SourceFormat.MARKDOWN,
}


def _require(payload: dict[str, Any], *fields: str) -> None:
    missing = [f for f in fields if payload.get(f) in (None, "")]
    if missing:
        raise SkipSource(f"missing required field(s): {', '.join(missing)}")


def _timestamp(value: Any, field: str) -> datetime:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError as exc:
            raise SkipSource(f"invalid {field}: {value!r}") from exc
    if not isinstance(value, datetime):
        raise SkipSource(f"invalid {field}: {value!r}")
    return value if value.tzinfo else value.replace(tzinfo=UTC)  # SQLite drops tzinfo


def _access(value: Any) -> AccessLevel:
    try:
        return AccessLevel(value)
    except ValueError as exc:
        raise SkipSource(f"invalid access_level: {value!r}") from exc


def _utc(moment: datetime | None) -> str:
    return f"{moment:%Y-%m-%d %H:%M} UTC" if moment else ""


def _field_table(rows: list[tuple[str, Any]]) -> str:
    lines = ["| Field | Value |", "|---|---|"]
    lines += [f"| {name} | {value} |" for name, value in rows if value not in (None, "", [])]
    return "\n".join(lines)


def _value(item: Any) -> Any:
    return getattr(item, "value", item)


# --- per source type -------------------------------------------------------------------------


def parse_document(raw: RawSource) -> ParsedSource:
    p = raw.payload
    _require(p, "id", "title", "doc_type", "updated_at", "access_level", "source_path")
    if not str(p.get("content") or "").strip():
        raise SkipSource("empty document content")
    return ParsedSource(
        source_type=document_source_type(p["doc_type"]),
        source_id=p["id"],
        title=p["title"],
        text=p["content"],
        format=SourceFormat.MARKDOWN,
        service_id=p.get("service_id"),
        timestamp=_timestamp(p["updated_at"], "updated_at"),
        access_level=_access(p["access_level"]),
        version=None,
        doc_type=_value(p["doc_type"]),
        file_path=p["source_path"],
        metadata={
            "source_path": p["source_path"],
            "revision": p.get("revision"),
            "tags": p.get("tags") or [],
            "author_id": p.get("author_id"),
            "content_hash": p.get("content_hash"),
        },
    )


def parse_incident(raw: RawSource) -> ParsedSource:
    p = raw.payload
    _require(p, "id", "service_id", "started_at", "access_level")
    started = _timestamp(p["started_at"], "started_at")
    resolved = _timestamp(p["resolved_at"], "resolved_at") if p.get("resolved_at") else None
    sections = [
        (name, str(p.get(key) or "").strip())
        for name, key in (
            ("Symptoms", "symptoms"),
            ("Root cause", "root_cause"),
            ("Resolution", "resolution"),
        )
    ]
    if not any(body for _, body in sections):
        raise SkipSource("incident has no symptoms, root cause or resolution text")
    title = p.get("title") or f"{_value(p.get('category')) or 'incident'} in {p['service_id']}"
    table = _field_table(
        [
            ("Service", p["service_id"]),
            ("Severity", _value(p.get("severity"))),
            ("Category", _value(p.get("category"))),
            ("Status", _value(p.get("status"))),
            ("Started", _utc(started)),
            (
                "Detected",
                _utc(_timestamp(p["detected_at"], "detected_at")) if p.get("detected_at") else None,
            ),
            (
                "Resolved",
                f"{_utc(resolved)} ({p.get('resolution_time_minutes')} min)" if resolved else None,
            ),
            (
                "Affected version",
                f"{p.get('affected_version')} ({p.get('deployment_id')})"
                if p.get("affected_version")
                else None,
            ),
            ("Root cause service", p.get("root_cause_service_id")),
            ("Root cause deployment", p.get("root_cause_deployment_id")),
            ("Root cause pull request", p.get("root_cause_pr_id")),
            ("Remediation deployment", p.get("remediation_deployment_id")),
            ("Parent incident", p.get("parent_incident_id")),
            ("Alert", p.get("alert_name")),
            ("Runbook", p.get("runbook_id")),
            ("Postmortem", p.get("postmortem_id")),
            ("Incident commander", f"@{p['commander_id']}" if p.get("commander_id") else None),
        ]
    )
    body = "\n\n".join(f"## {name}\n\n{text}" for name, text in sections if text)
    metrics = p.get("metrics") or {}
    if metrics:
        # Sorted: PostgreSQL JSONB reorders object keys, and the canonical text
        # (and therefore chunk boundaries) must not depend on the database backend.
        body += "\n\n## Metrics\n\n" + "\n".join(f"- {k}: {v}" for k, v in sorted(metrics.items()))
    keys = (
        "severity",
        "category",
        "status",
        "deployment_id",
        "root_cause_service_id",
        "root_cause_deployment_id",
        "root_cause_pr_id",
        "remediation_deployment_id",
        "parent_incident_id",
        "runbook_id",
        "postmortem_id",
        "alert_name",
        "commander_id",
        "resolution_time_minutes",
    )
    return ParsedSource(
        source_type=SourceType.INCIDENT,
        source_id=p["id"],
        title=f"{p['id']}: {title}",
        text=f"# {p['id']}: {title}\n\n{table}\n\n{body}",
        format=SourceFormat.MARKDOWN,
        service_id=p["service_id"],
        timestamp=started,
        access_level=_access(p["access_level"]),
        version=p.get("affected_version"),
        doc_type=SourceType.INCIDENT.value,
        file_path=None,
        metadata={
            **{k: _value(p.get(k)) for k in keys if p.get(k) is not None},
            "tags": p.get("tags") or [],
        },
    )


def parse_deployment(raw: RawSource) -> ParsedSource:
    p = raw.payload
    _require(p, "id", "service_id", "version", "deployed_at")
    deployed = _timestamp(p["deployed_at"], "deployed_at")
    heading = f"{p['id']}: {p['service_id']} {p['version']}"
    table = _field_table(
        [
            ("Service", p["service_id"]),
            (
                "Version",
                f"{p['version']} (previous {p['previous_version']})"
                if p.get("previous_version")
                else p["version"],
            ),
            ("Status", _value(p.get("status"))),
            ("Strategy", _value(p.get("strategy"))),
            ("Deployed", f"{_utc(deployed)} ({p.get('duration_seconds')} s)"),
            ("Commit", p.get("commit_sha")),
            ("Author", f"@{p['author_id']}" if p.get("author_id") else None),
            ("Rollback of", p.get("rollback_of_id")),
            ("Pull requests", ", ".join(p.get("pull_request_ids") or [])),
        ]
    )
    changes = str(p.get("changes") or "").strip() or "(no changelog)"
    return ParsedSource(
        source_type=SourceType.DEPLOYMENT,
        source_id=p["id"],
        title=heading,
        text=f"# {heading}\n\n{table}\n\n## Changes\n\n{changes}",
        format=SourceFormat.MARKDOWN,
        service_id=p["service_id"],
        timestamp=deployed,
        access_level=UNLABELLED_LEVEL,
        version=p["version"],
        doc_type=SourceType.DEPLOYMENT.value,
        file_path=None,
        metadata={
            "status": _value(p.get("status")),
            "strategy": _value(p.get("strategy")),
            "is_rollback": bool(p.get("is_rollback")),
            "rollback_of_id": p.get("rollback_of_id"),
            "previous_version": p.get("previous_version"),
            "commit_sha": p.get("commit_sha"),
            "author_id": p.get("author_id"),
            "pull_request_ids": list(p.get("pull_request_ids") or []),
        },
    )


def parse_code_file(raw: RawSource) -> ParsedSource:
    p = raw.payload
    _require(p, "id", "path", "last_modified_at", "access_level")
    if not str(p.get("content") or "").strip():
        raise SkipSource("empty file")
    language = str(p.get("language") or "text")
    return ParsedSource(
        source_type=SourceType.CODE,
        source_id=p["id"],
        title=p["path"],
        text=p["content"],
        format=_CODE_FORMATS.get(language, SourceFormat.TEXT),
        service_id=p.get("service_id"),
        timestamp=_timestamp(p["last_modified_at"], "last_modified_at"),
        access_level=_access(p["access_level"]),
        version=None,
        doc_type=SourceType.CODE.value,
        file_path=p["path"],
        metadata={
            "language": language,
            "kind": _value(p.get("kind")),
            "repository": p.get("repository"),
            "last_commit_sha": p.get("last_commit_sha"),
            "content_hash": p.get("content_hash"),
        },
    )


def parse_pull_request(raw: RawSource) -> ParsedSource:
    p = raw.payload
    _require(p, "id", "service_id", "title", "opened_at")
    # Sorted here (code-point order), not by SQL ORDER BY: PostgreSQL collations
    # ignore punctuation, so row order and the canonical text would vary by backend.
    files = sorted(p.get("files") or [], key=lambda f: str(f.get("path", "")))
    moment = _timestamp(p.get("merged_at") or p["opened_at"], "merged_at")
    # A PR is as sensitive as the most sensitive file it changes.
    access = most_restrictive([_access(f["access_level"]) for f in files] + [UNLABELLED_LEVEL])
    heading = f"{p['id']}: {p['title']}"
    table = _field_table(
        [
            ("Service", p["service_id"]),
            ("State", _value(p.get("state"))),
            ("Author", f"@{p['author_id']}" if p.get("author_id") else None),
            ("Reviewers", ", ".join(f"@{r}" for r in p.get("reviewers") or [])),
            ("Labels", ", ".join(p.get("labels") or [])),
            ("Opened", _utc(_timestamp(p["opened_at"], "opened_at"))),
            ("Merged", _utc(moment) if p.get("merged_at") else None),
            ("Merge commit", p.get("merge_commit_sha")),
            (
                "Shipped in",
                f"{p['deployment_id']} ({p.get('deployment_version')})"
                if p.get("deployment_id")
                else None,
            ),
        ]
    )
    parts = [
        f"# {heading}",
        table,
        "## Description",
        str(p.get("description") or "").strip() or "(no description)",
    ]
    for f in files:
        parts.append(f"## Diff: {f['path']} (+{f['additions']}/-{f['deletions']})")
        parts.append(f"```diff\n{str(f.get('patch') or '').rstrip()}\n```")
    return ParsedSource(
        source_type=SourceType.PULL_REQUEST,
        source_id=p["id"],
        title=heading,
        text="\n\n".join(parts),
        format=SourceFormat.MARKDOWN,
        service_id=p["service_id"],
        timestamp=moment,
        access_level=access,
        version=p.get("deployment_version"),
        doc_type=SourceType.PULL_REQUEST.value,
        file_path=None,
        metadata={
            "number": p.get("number"),
            "author_id": p.get("author_id"),
            "labels": p.get("labels") or [],
            "merge_commit_sha": p.get("merge_commit_sha"),
            "deployment_id": p.get("deployment_id"),
            "files": [f["path"] for f in files],
            "additions": sum(f["additions"] for f in files),
            "deletions": sum(f["deletions"] for f in files),
        },
    )


PARSERS: dict[SourceType, Callable[[RawSource], ParsedSource]] = {
    SourceType.INCIDENT: parse_incident,
    SourceType.RUNBOOK: parse_document,
    SourceType.DOCUMENTATION: parse_document,
    SourceType.POSTMORTEM: parse_document,
    SourceType.DEPLOYMENT: parse_deployment,
    SourceType.CODE: parse_code_file,
    SourceType.PULL_REQUEST: parse_pull_request,
}


def parse_source(raw: RawSource) -> ParsedSource:
    if not isinstance(raw.payload, dict):
        raise SkipSource("payload is not a mapping")
    try:
        return PARSERS[raw.source_type](raw)
    except SkipSource:
        raise
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise SkipSource(f"malformed {raw.source_type.value}: {type(exc).__name__}: {exc}") from exc
