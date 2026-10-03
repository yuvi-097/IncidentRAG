"""Entity extraction: record ids, versions, services, time ranges, code identifiers."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.agents.entities import ServiceCatalog, extract_entities, extract_time_range
from app.schemas.enums import LogLevel, Severity

NOW = datetime(2026, 9, 15, 12, 0, tzinfo=UTC)
SERVICES = ServiceCatalog(
    [
        "api-gateway",
        "auth-service",
        "user-service",
        "product-service",
        "inventory-service",
        "cart-service",
        "order-service",
        "payment-service",
        "notification-service",
        "recommendation-service",
        "search-service",
    ]
)


def entities(text: str):
    return extract_entities(text, SERVICES, NOW)


def test_record_ids_and_versions() -> None:
    e = entities("Did inc-0421 follow DEP-0296 (PR-1501, RB-0049, PM-0039, CF-0164) in v2.8.1?")
    assert e.incident_ids == ["INC-0421"] and e.deployment_ids == ["DEP-0296"]
    assert e.pull_request_ids == ["PR-1501"] and e.document_ids == ["RB-0049", "PM-0039"]
    assert e.code_file_ids == ["CF-0164"] and e.versions == ["v2.8.1"]
    assert e.record_ids[:2] == ["INC-0421", "DEP-0296"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("payment-service latency", ["payment-service"]),
        ("How many payment incidents?", ["payment-service"]),
        ("the payments database", ["payment-service"]),
        ("Why is the API gateway slow?", ["api-gateway"]),
        ("notification service delays", ["notification-service"]),
        ("cart and inventory", ["cart-service", "inventory-service"]),
        ("in order to search the logs as a user", []),  # ordinary words are not services
        ("order-service and search-service", ["order-service", "search-service"]),
    ],
)
def test_services_by_id_name_and_distinctive_alias(text: str, expected: list[str]) -> None:
    assert entities(text).services == expected


@pytest.mark.parametrize(
    ("text", "since", "until"),
    [
        ("last month", datetime(2026, 8, 1, tzinfo=UTC), datetime(2026, 9, 1, tzinfo=UTC)),
        ("in the last 90 days", datetime(2026, 6, 17, 12, tzinfo=UTC), NOW),
        ("past 24 hours", datetime(2026, 9, 14, 12, tzinfo=UTC), NOW),
        ("yesterday", datetime(2026, 9, 14, tzinfo=UTC), datetime(2026, 9, 15, tzinfo=UTC)),
        ("this year", datetime(2026, 1, 1, tzinfo=UTC), NOW),
        ("last year", datetime(2025, 1, 1, tzinfo=UTC), datetime(2026, 1, 1, tzinfo=UTC)),
        ("in June 2026", datetime(2026, 6, 1, tzinfo=UTC), datetime(2026, 7, 1, tzinfo=UTC)),
        ("during Dec 2025", datetime(2025, 12, 1, tzinfo=UTC), datetime(2026, 1, 1, tzinfo=UTC)),
        ("incidents in 2025", datetime(2025, 1, 1, tzinfo=UTC), datetime(2026, 1, 1, tzinfo=UTC)),
        ("since 2026-06-01", datetime(2026, 6, 1, tzinfo=UTC), NOW),
    ],
)
def test_time_expressions(text: str, since: datetime, until: datetime) -> None:
    window = extract_time_range(text, NOW)
    assert window is not None and (window.since, window.until) == (since, until)


def test_last_month_in_january_wraps_to_december() -> None:
    window = extract_time_range("last month", datetime(2026, 1, 10, tzinfo=UTC))
    assert window is not None and window.since == datetime(2025, 12, 1, tzinfo=UTC)
    assert extract_time_range("no time here", NOW) is None


def test_code_identifiers_config_keys_paths_and_quotes() -> None:
    e = entities(
        "Is PaymentProcessor.create_payment in payment_service/processor.py reading "
        "PAYMENT_SERVICE_DB_POOL_SIZE? Logs say 'QueuePool limit reached'."
    )
    assert "PaymentProcessor.create_payment" in e.code_identifiers
    assert "payment_service/processor.py" in e.file_paths
    assert e.config_keys == ["PAYMENT_SERVICE_DB_POOL_SIZE"]
    assert e.quoted == ["QueuePool limit reached"]
    assert "QueuePool" not in e.code_identifiers  # quoted text is not an identifier


def test_severities_levels_and_traces() -> None:
    e = entities(
        "SEV1 and sev 2; show ERROR logs and WARN lines for trace 4bf92f3577b34da6a3ce929d0e0e4736"
    )
    assert e.severities == [Severity.SEV1, Severity.SEV2]
    assert e.log_levels == [LogLevel.ERROR, LogLevel.WARNING]
    assert e.trace_ids == ["4bf92f3577b34da6a3ce929d0e0e4736"]
    assert entities("an error occurred").log_levels == []  # a level needs log context
