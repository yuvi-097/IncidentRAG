"""NL -> SQL templates: the right shape, safe literals, and None when unsure."""

from __future__ import annotations

import pytest

from app.agents.entities import extract_entities
from app.agents.sql_templates import build_sql
from app.tools.sql_guard import check_sql
from tests.agents.test_entities import SERVICES
from tests.tools.conftest import NOW

TABLES = {"incidents", "deployments", "pull_requests", "services", "logs"}


def plan(question: str, dialect: str = "sqlite"):
    return build_sql(question, extract_entities(question, SERVICES, NOW), dialect)


@pytest.mark.parametrize(
    ("question", "starts", "contains"),
    [
        (
            "How many payment incidents happened last month?",
            "SELECT count(*) AS n FROM incidents",
            [
                "service_id IN ('payment-service')",
                "started_at >= '2026-08-01 00:00:00'",
                "started_at < '2026-09-01 00:00:00'",
            ],
        ),
        (
            "How many deployments were rolled back this year?",
            "SELECT count(*) AS n FROM deployments",
            ["status = 'rolled_back'", "deployed_at >= '2026-01-01 00:00:00'"],
        ),
        (
            "How many pull requests were merged for cart-service in June 2026?",
            "SELECT count(*) AS n FROM pull_requests",
            [
                "merged_at IS NOT NULL",
                "service_id IN ('cart-service')",
                "merged_at < '2026-07-01 00:00:00'",
            ],
        ),
        (
            "Which service had the most incidents in the last 90 days?",
            "SELECT service_id, count(*) AS n FROM incidents",
            ["ORDER BY n DESC, service_id LIMIT 1"],
        ),
        (
            "List the top 5 services by number of deployments.",
            "SELECT service_id, count(*) AS n FROM deployments",
            ["LIMIT 5"],
        ),
        (
            "What is the average resolution time for SEV1 incidents?",
            "SELECT round(avg(resolution_time_minutes), 1)",
            ["severity IN ('SEV1')"],
        ),
        (
            "Count SEV2 incidents per month in 2025.",
            "SELECT strftime('%Y-%m', started_at) AS month",
            ["GROUP BY month", "severity IN ('SEV2')"],
        ),
        (
            "Number of incidents per severity",
            "SELECT severity, count(*) AS n FROM incidents",
            ["GROUP BY severity"],
        ),
    ],
)
def test_templates(question: str, starts: str, contains: list[str]) -> None:
    result = plan(question)
    assert result is not None and result.sql.startswith(starts), result
    for fragment in contains:
        assert fragment in result.sql, (fragment, result.sql)
    assert check_sql(result.sql, TABLES).tables  # always passes the SQL guard
    assert result.description


def test_postgres_dialect_uses_explicit_utc_and_to_char() -> None:
    result = plan("Count incidents per month in 2025.", "postgresql")
    assert result is not None
    assert (
        "to_char(started_at, 'YYYY-MM')" in result.sql and "'2025-01-01 00:00:00+00'" in result.sql
    )
    check_sql(result.sql, TABLES)


@pytest.mark.parametrize(
    "question",
    [
        "What percentage of incidents were caused by deployments?",  # no template for ratios
        "How many users logged in yesterday?",  # no supported subject
        "What is the average deployment duration?",  # average only of resolution time
    ],
)
def test_unsupported_questions_return_none(question: str) -> None:
    assert plan(question) is None
