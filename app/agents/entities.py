"""Deterministic extraction of the things a query refers to.

Record ids, versions, services, time ranges, code identifiers, log levels and
quoted phrases. The router uses them as routing evidence, and the planner uses them
to fill tool arguments. Everything is pattern-based and explainable; nothing is
guessed.
"""

from __future__ import annotations

import calendar
import re
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta

from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.engine import Engine

from app.schemas.enums import LogLevel, Severity


class TimeRange(BaseModel):
    model_config = ConfigDict(frozen=True)

    since: datetime
    until: datetime
    expression: str  # the text it was derived from


class QueryEntities(BaseModel):
    model_config = ConfigDict(frozen=True)

    incident_ids: list[str] = []
    deployment_ids: list[str] = []
    pull_request_ids: list[str] = []
    document_ids: list[str] = []  # RB-, DOC-, PM-
    code_file_ids: list[str] = []
    versions: list[str] = []
    services: list[str] = []
    severities: list[Severity] = []
    log_levels: list[LogLevel] = []
    trace_ids: list[str] = []
    file_paths: list[str] = []
    code_identifiers: list[str] = []  # CamelCase, snake_case, Class.method
    config_keys: list[str] = []  # UPPER_SNAKE_CASE
    quoted: list[str] = []
    time_range: TimeRange | None = None

    @property
    def record_ids(self) -> list[str]:
        return [
            *self.incident_ids,
            *self.deployment_ids,
            *self.pull_request_ids,
            *self.document_ids,
            *self.code_file_ids,
        ]


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


_ID = {
    "incident_ids": re.compile(r"\bINC-\d{1,6}\b", re.IGNORECASE),
    "deployment_ids": re.compile(r"\bDEP-\d{1,6}\b", re.IGNORECASE),
    "pull_request_ids": re.compile(r"\bPR-\d{1,6}\b", re.IGNORECASE),
    "document_ids": re.compile(r"\b(?:RB|DOC|PM)-\d{1,6}\b", re.IGNORECASE),
    "code_file_ids": re.compile(r"\bCF-\d{1,6}\b", re.IGNORECASE),
}
_VERSION = re.compile(r"(?<![\w.])v\d+(?:\.\d+){1,3}\b", re.IGNORECASE)
_TRACE = re.compile(r"\b[0-9a-f]{32}\b")
_SEVERITY = re.compile(r"\bsev\s?([1-4])\b", re.IGNORECASE)
_LEVELS = {
    "debug": LogLevel.DEBUG,
    "info": LogLevel.INFO,
    "warn": LogLevel.WARNING,
    "warning": LogLevel.WARNING,
    "warnings": LogLevel.WARNING,
    "error": LogLevel.ERROR,
    "critical": LogLevel.CRITICAL,
    "fatal": LogLevel.CRITICAL,
}
# A level counts only in log context ("error logs", "WARN lines", "level ERROR").
_LEVEL_CONTEXT = re.compile(
    r"\b(debug|info|warn|warning|warnings|error|critical|fatal)\s+(?:level\s+)?"
    r"(?:logs?|lines?|entries|messages)\b|\blevel\s*[=:]?\s*(debug|info|warn|warning|error|critical)\b",
    re.IGNORECASE,
)
_FILE_PATH = re.compile(r"\b[\w.-]+(?:/[\w.-]+)*\.(?:py|ya?ml|toml|json|md|txt|cfg|ini|sql)\b")
_DOTTED = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\.[a-z_][A-Za-z0-9_]*\b")
_CAMEL = re.compile(r"\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+\b")
_SNAKE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")
_UPPER_SNAKE = re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b")
_QUOTED = re.compile(r"'([^']{2,200})'|\"([^\"]{2,200})\"|`([^`]{2,200})`")

_MONTHS = {name.lower(): n for n, name in enumerate(calendar.month_name) if name}
_MONTHS.update({name.lower(): n for n, name in enumerate(calendar.month_abbr) if name})
_UNITS = {"minute": 1 / 1440, "hour": 1 / 24, "day": 1, "week": 7, "month": 30, "year": 365}


def _month_start(year: int, month: int) -> datetime:
    return datetime(year, month, 1, tzinfo=UTC)


def _next_month(start: datetime) -> datetime:
    return _month_start(start.year + (start.month // 12), start.month % 12 + 1)


def extract_time_range(text: str, now: datetime) -> TimeRange | None:
    """Relative and absolute expressions; ``now`` is injected (tests fix it)."""
    lowered = text.lower()
    today = datetime(now.year, now.month, now.day, tzinfo=UTC)
    match = re.search(
        r"\b(?:last|past|previous)\s+(\d{1,3})\s+(minute|hour|day|week|month|year)s?\b", lowered
    )
    if match:
        span = timedelta(days=int(match.group(1)) * _UNITS[match.group(2)])
        return TimeRange(since=now - span, until=now, expression=match.group())
    match = re.search(
        r"\b(?:last|past|previous)\s+(hour|day|week|month|year)\b|\b(yesterday|today|this\s+(?:week|month|year))\b",
        lowered,
    )
    if match:
        phrase = match.group()
        this_month = _month_start(now.year, now.month)
        ranges = {
            "hour": (now - timedelta(hours=1), now),
            "day": (now - timedelta(days=1), now),
            "week": (now - timedelta(days=7), now),
            "month": (
                _month_start(now.year - (now.month == 1), (now.month - 2) % 12 + 1),
                this_month,
            ),
            "year": (
                datetime(now.year - 1, 1, 1, tzinfo=UTC),
                datetime(now.year, 1, 1, tzinfo=UTC),
            ),
            "yesterday": (today - timedelta(days=1), today),
            "today": (today, now),
            "this week": (today - timedelta(days=today.weekday()), now),
            "this month": (this_month, now),
            "this year": (datetime(now.year, 1, 1, tzinfo=UTC), now),
        }
        key = match.group(1) or re.sub(r"\s+", " ", match.group(2))
        since, until = ranges[key]
        return TimeRange(since=since, until=until, expression=phrase)
    match = re.search(r"\b(?:since|after|from)\s+(\d{4}-\d{2}-\d{2})\b", lowered)
    if match:
        since = datetime.fromisoformat(match.group(1)).replace(tzinfo=UTC)
        return TimeRange(since=since, until=now, expression=match.group())
    names = "|".join(sorted(_MONTHS, key=len, reverse=True))
    match = re.search(rf"\b(?:in\s+)?({names})\.?\s+(\d{{4}})\b", lowered)
    if match:
        start = _month_start(int(match.group(2)), _MONTHS[match.group(1)])
        return TimeRange(since=start, until=_next_month(start), expression=match.group())
    match = re.search(r"\b(?:in|during)\s+(20\d{2})\b", lowered)
    if match:
        year = int(match.group(1))
        return TimeRange(
            since=datetime(year, 1, 1, tzinfo=UTC),
            until=datetime(year + 1, 1, 1, tzinfo=UTC),
            expression=match.group(),
        )
    return None


class ServiceCatalog:
    """Recognises services by id ("payment-service"), name ("payment service") and a
    distinctive short alias ("payment", "payments"). Aliases that are ordinary
    English words ("order", "user", "search", "product") are not used alone,
    because "in order to" or "search the logs" must not select a service."""

    AMBIGUOUS = frozenset({"order", "user", "search", "product", "api"})

    def __init__(self, service_ids: Iterable[str]) -> None:
        self.aliases: dict[str, str] = {}
        for service in service_ids:
            self.aliases[service] = service
            words = service.replace("-", " ")
            self.aliases[words] = service
            base = service.removesuffix("-service")
            if base != service:
                self.aliases[f"{base} svc"] = service
                if base not in self.AMBIGUOUS:
                    self.aliases[base] = service
                    self.aliases[f"{base}s"] = service
            if service == "api-gateway":
                self.aliases["gateway"] = service
        names = sorted(self.aliases, key=len, reverse=True)
        self._pattern = re.compile(
            r"\b(" + "|".join(re.escape(n) for n in names) + r")\b", re.IGNORECASE
        )

    @classmethod
    def from_engine(cls, engine: Engine) -> ServiceCatalog:
        from app.database.models import Service

        with engine.connect() as connection:
            return cls(connection.execute(select(Service.id)).scalars())

    def find(self, text: str) -> list[str]:
        return _unique(self.aliases[m.group(1).lower()] for m in self._pattern.finditer(text))


def extract_entities(text: str, services: ServiceCatalog | None, now: datetime) -> QueryEntities:
    fields: dict[str, list[str]] = {
        name: _unique(m.upper() for m in pattern.findall(text)) for name, pattern in _ID.items()
    }
    quoted = [next(g for g in groups if g) for groups in _QUOTED.findall(text)]
    unquoted = _QUOTED.sub(" ", text)
    paths = _unique(_FILE_PATH.findall(unquoted))
    dotted = [d for d in _DOTTED.findall(unquoted) if not any(d in p for p in paths)]
    identifiers = _unique([*dotted, *_CAMEL.findall(unquoted), *_SNAKE.findall(unquoted)])
    levels: list[LogLevel] = []
    for match in _LEVEL_CONTEXT.finditer(text):
        word = (match.group(1) or match.group(2)).lower()
        levels.append(_LEVELS[word])
    return QueryEntities(
        **fields,
        versions=_unique(v.lower() for v in _VERSION.findall(text)),
        services=services.find(text) if services else [],
        severities=_unique(Severity(f"SEV{n}") for n in _SEVERITY.findall(text)),
        log_levels=_unique(levels),
        trace_ids=_unique(_TRACE.findall(text)),
        file_paths=paths,
        code_identifiers=[i for i in identifiers if not i.lower().startswith(("http", "www"))],
        config_keys=_unique(_UPPER_SNAKE.findall(unquoted)),
        quoted=_unique(quoted),
        time_range=extract_time_range(text, now),
    )
