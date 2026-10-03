from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterator

import pytest

from app.observability.context import reset_request_id, set_request_id
from app.observability.structured_logging import (
    ConsoleFormatter,
    JsonFormatter,
    RequestContextFilter,
    configure_logging,
)


@pytest.fixture
def restore_root_logger() -> Iterator[None]:
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)


def _record(message: str = "incident.lookup", **extra: object) -> logging.LogRecord:
    record = logging.getLogger("test").makeRecord(
        "test", logging.INFO, __file__, 1, message, (), None, extra=extra
    )
    RequestContextFilter().filter(record)
    return record


def test_json_formatter_emits_fields_extras_and_request_id() -> None:
    token = set_request_id("req-123")
    try:
        record = _record(service="payment-service", error_rate=0.42)
    finally:
        reset_request_id(token)

    payload = json.loads(JsonFormatter().format(record))

    assert payload["message"] == "incident.lookup"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "test"
    assert payload["request_id"] == "req-123"
    assert payload["service"] == "payment-service"
    assert payload["error_rate"] == 0.42
    assert payload["timestamp"].endswith("+00:00")


def test_json_formatter_omits_request_id_outside_requests() -> None:
    payload = json.loads(JsonFormatter().format(_record()))
    assert "request_id" not in payload


def test_json_formatter_serialises_exceptions_and_non_json_values() -> None:
    try:
        raise ValueError("bad deploy")
    except ValueError:
        record = logging.getLogger("test").makeRecord(
            "test",
            logging.ERROR,
            __file__,
            1,
            "deploy.failed",
            (),
            sys.exc_info(),
            extra={"payload": object()},
        )

    payload = json.loads(JsonFormatter().format(record))

    assert "ValueError: bad deploy" in payload["exception"]
    assert payload["payload"].startswith("<object object")


def test_console_formatter_renders_key_values() -> None:
    line = ConsoleFormatter().format(_record(service="cart-service", status_code=500))

    assert "INFO" in line
    assert "incident.lookup" in line
    assert "service=cart-service" in line
    assert "status_code=500" in line


def test_console_formatter_keeps_each_event_on_one_line() -> None:
    line = ConsoleFormatter().format(_record(error="first\nFAKE 2026 CRITICAL forged"))

    assert "\n" not in line
    assert "error=first\\nFAKE" in line


def test_formatters_drop_uvicorn_color_message() -> None:
    record = _record(color_message="\x1b[36mcoloured\x1b[0m")

    assert "color_message" not in JsonFormatter().format(record)
    assert "color_message" not in ConsoleFormatter().format(record)


@pytest.mark.usefixtures("restore_root_logger")
def test_configure_logging_is_idempotent() -> None:
    configure_logging("DEBUG", "console")
    configure_logging("WARNING", "json")

    root = logging.getLogger()
    ours = [h for h in root.handlers if getattr(h, "_opsrag_handler", False)]
    assert len(ours) == 1
    assert isinstance(ours[0].formatter, JsonFormatter)
    assert root.level == logging.WARNING
    assert logging.getLogger("uvicorn.access").propagate is False
