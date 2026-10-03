"""Structured logging on top of the standard library.

Log calls use a stable event name as the message and pass fields via ``extra``:

    logger.info("request.completed", extra={"status_code": 200, "duration_ms": 3.1})

``json`` format emits one JSON object per line (for log aggregation);
``console`` format emits a readable ``key=value`` line for local development.

Secrets are never logged: ``RedactingFilter`` replaces the value of any field whose
name says it is a secret (``api_key``, ``password``, ``authorization``...) and any
credential found in the message or other fields; both formatters also redact the
final line (covering exception tracebacks).
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

from app.observability.context import get_request_id
from app.security.secrets import REDACTED, SECRET_FIELD, redact_secrets

# Attributes present on every LogRecord; anything else came from `extra=`.
# `color_message` is an ANSI-coloured duplicate that uvicorn attaches.
_STANDARD_RECORD_ATTRS = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()
    | {"message", "asctime", "request_id", "color_message"}
)

# Marks handlers installed by configure_logging so re-configuration replaces
# only our own handlers (and leaves e.g. pytest's capture handlers alone).
_HANDLER_MARKER = "_opsrag_handler"


class RequestContextFilter(logging.Filter):
    """Attach the current request id (if any) to every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id()
        return True


def _redact_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact_secrets(value)[0]
    if isinstance(value, dict):
        return {
            k: REDACTED if isinstance(k, str) and SECRET_FIELD.match(k) else _redact_value(v)
            for k, v in value.items()
        }
    if isinstance(value, list | tuple):
        return type(value)(_redact_value(v) for v in value)
    return value


class RedactingFilter(logging.Filter):
    """Remove credentials from a record before any handler formats it."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact_secrets(record.msg)[0]
        if record.args:
            if isinstance(record.args, dict):
                record.args = _redact_value(record.args)
            else:
                record.args = tuple(_redact_value(a) for a in record.args)
        for key in list(record.__dict__):
            if key in _STANDARD_RECORD_ATTRS or key.startswith("_"):
                continue
            value = record.__dict__[key]
            record.__dict__[key] = REDACTED if SECRET_FIELD.match(key) else _redact_value(value)
        return True


def _extra_fields(record: logging.LogRecord) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.__dict__.items()
        if key not in _STANDARD_RECORD_ATTRS and not key.startswith("_")
    }


def _timestamp(record: logging.LogRecord) -> str:
    return datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="milliseconds")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": _timestamp(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = getattr(record, "request_id", None)
        if request_id:
            payload["request_id"] = request_id
        payload.update(_extra_fields(record))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return redact_secrets(json.dumps(payload, default=str, ensure_ascii=False))[0]


def _single_line(value: object) -> str:
    # One event per line; also stops field values from forging extra log lines.
    return str(value).replace("\r", "\\r").replace("\n", "\\n")


class ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        fields = _extra_fields(record)
        request_id = getattr(record, "request_id", None)
        if request_id:
            fields["request_id"] = request_id
        rendered = " ".join(f"{key}={_single_line(value)}" for key, value in fields.items())
        line = (
            f"{_timestamp(record)} {record.levelname:<8} {record.name}: "
            f"{_single_line(record.getMessage())}"
        )
        if rendered:
            line = f"{line} {rendered}"
        if record.exc_info:
            line = f"{line}\n{self.formatException(record.exc_info)}"
        return redact_secrets(line)[0]


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    """Install a single structured handler on the root logger. Idempotent."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, _HANDLER_MARKER, False):
            root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if fmt == "json" else ConsoleFormatter())
    handler.addFilter(RequestContextFilter())
    handler.addFilter(RedactingFilter())
    setattr(handler, _HANDLER_MARKER, True)
    root.addHandler(handler)
    root.setLevel(level)

    # Route uvicorn's own logs through our handler. Its access log is disabled
    # because RequestContextMiddleware emits a structured access event instead.
    for name in ("uvicorn", "uvicorn.error"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True
    access_logger = logging.getLogger("uvicorn.access")
    access_logger.handlers.clear()
    access_logger.propagate = False
