"""search_logs: structured search over application log lines.

Logs are large, so every search is bounded: by a time window (at most
``TOOLS_LOGS_MAX_WINDOW_DAYS``), a deployment id or a trace id, and by a result
limit. Text matching is a case-insensitive substring match with LIKE wildcards
escaped, so ``%`` and ``_`` in the text are literal.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Annotated

from pydantic import Field, StringConstraints, model_validator
from sqlalchemy import func, select

from app.database.models import LogEntry
from app.schemas.enums import LogLevel, Resource, ToolPermission
from app.tools.base import Tool, ToolContext, ToolInputError, ToolModel, truncate
from app.tools.common import (
    DeploymentId,
    ServiceId,
    TimeWindow,
    check_services,
    require_unlabelled_access,
)

TraceId = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{32}$")]
LEVEL_ORDER = list(LogLevel)


class SearchLogsInput(TimeWindow):
    services: list[ServiceId] | None = Field(default=None, max_length=20)
    min_level: LogLevel | None = Field(default=None, description="e.g. WARNING = WARNING and up")
    levels: list[LogLevel] | None = None
    text: str | None = Field(default=None, min_length=2, max_length=200)
    deployment_id: DeploymentId | None = None
    trace_id: TraceId | None = None
    limit: int = Field(default=50, ge=1, le=500)
    newest_first: bool = True

    @model_validator(mode="after")
    def _bounded(self) -> SearchLogsInput:
        if self.levels and self.min_level:
            raise ValueError("use either levels or min_level, not both")
        if not (self.since or self.until or self.deployment_id or self.trace_id):
            raise ValueError("give a time window (since/until), a deployment_id or a trace_id")
        return self


class LogRecord(ToolModel):
    timestamp: datetime
    service_id: str
    level: LogLevel
    logger: str
    message: str
    trace_id: str | None
    deployment_id: str | None
    version: str | None
    host: str


class SearchLogsOutput(ToolModel):
    entries: list[LogRecord]
    total_matched: int
    truncated: bool
    level_counts: dict[str, int]  # over all matches, not just the returned entries
    since: datetime | None
    until: datetime | None


class SearchLogsTool(Tool[SearchLogsInput, SearchLogsOutput]):
    name = "search_logs"
    description = (
        "Search application logs by service, level, time window, deployment id, trace id "
        "and message text. Returns matching log lines plus counts per level."
    )
    permission = ToolPermission.LOGS_READ
    input_model = SearchLogsInput
    output_model = SearchLogsOutput

    def _run(self, arguments: SearchLogsInput, context: ToolContext) -> SearchLogsOutput:
        require_unlabelled_access(context.principal, Resource.LOGS)
        max_window = timedelta(days=context.settings.logs_max_window_days)
        since, until = arguments.since, arguments.until
        if since and not until:
            until = since + max_window
        elif until and not since:
            since = until - max_window
        if since and until and until - since > max_window:
            raise ToolInputError(
                f"time window longer than {context.settings.logs_max_window_days} days",
                [{"field": "since", "error": "narrow the window"}],
            )
        log = LogEntry
        conditions = []
        if since:
            conditions.append(log.timestamp >= since)
        if until:
            conditions.append(log.timestamp < until)
        if arguments.services:
            conditions.append(log.service_id.in_(arguments.services))
        if arguments.levels:
            conditions.append(log.level.in_(arguments.levels))
        elif arguments.min_level:
            index = LEVEL_ORDER.index(arguments.min_level)
            conditions.append(log.level.in_(LEVEL_ORDER[index:]))
        if arguments.text:
            conditions.append(
                func.lower(log.message).contains(arguments.text.lower(), autoescape=True)
            )
        if arguments.deployment_id:
            conditions.append(log.deployment_id == arguments.deployment_id)
        if arguments.trace_id:
            conditions.append(log.trace_id == arguments.trace_id)

        limit = min(arguments.limit, context.settings.logs_max_results)
        order = log.timestamp.desc() if arguments.newest_first else log.timestamp.asc()
        with context.engine.connect() as connection:
            check_services(connection, arguments.services)
            counts = dict(
                connection.execute(
                    select(log.level, func.count()).where(*conditions).group_by(log.level)
                ).all()
            )
            rows = connection.execute(
                select(log.__table__).where(*conditions).order_by(order, log.id).limit(limit)
            )
            entries = [
                LogRecord(
                    timestamp=r.timestamp,
                    service_id=r.service_id,
                    level=r.level,
                    logger=r.logger,
                    message=truncate(r.message, context.settings.snippet_chars),
                    trace_id=r.trace_id,
                    deployment_id=r.deployment_id,
                    version=r.version,
                    host=r.host,
                )
                for r in rows
            ]
        total = sum(counts.values())
        return SearchLogsOutput(
            entries=entries,
            total_matched=total,
            truncated=total > len(entries),
            level_counts={LogLevel(k).value: v for k, v in sorted(counts.items())},
            since=since,
            until=until,
        )
