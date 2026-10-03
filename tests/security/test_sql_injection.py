"""Security test 5: SQL injection.

Three surfaces: SQL written against ``query_database`` (the model or a user tries to
reach other tables or rows), values that end up in SQL (the agent's SQL templates,
tool arguments), and free text passed to retrieval. None can read rows outside the
caller's grants or change anything.
"""

from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy import func, select

from app.agents.entities import QueryEntities
from app.agents.sql_templates import build_sql
from app.database.base import Base
from app.database.models import Incident
from app.schemas.enums import AccessLevel, Resource, ToolPermission
from app.tools import ToolInputError, UnsafeSqlError, build_registry
from app.tools.logs import SearchLogsTool
from app.tools.runbooks import GetRunbookTool
from app.tools.search import SearchDocumentsTool, SearchIncidentsTool
from app.tools.sql_tool import QueryDatabaseTool
from tests.agents.conftest import make_agent
from tests.tools.conftest import ToolEnv, custom_principal, principal

BASIC = {AccessLevel.PUBLIC, AccessLevel.ENGINEERING}
LIMITED = custom_principal({Resource.INCIDENTS: BASIC}, frozenset({ToolPermission.SQL_READ}))


def table_counts(env: ToolEnv) -> dict[str, int]:
    with env.engine.connect() as connection:
        return {
            name: connection.execute(select(func.count()).select_from(table)).scalar_one()
            for name, table in Base.metadata.tables.items()
        }


def visible_incidents(env: ToolEnv) -> int:
    with env.engine.connect() as connection:
        return connection.execute(
            select(func.count())
            .select_from(Incident)
            .where(Incident.access_level.in_(["public", "engineering"]))
        ).scalar_one()


def run(env: ToolEnv, sql: str, who: Any = LIMITED):
    return QueryDatabaseTool().execute({"sql": sql}, env.context(who))


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT count(*) FROM incidents WHERE id = 'x' OR '1'='1'",
        "SELECT count(*) FROM incidents WHERE access_level = 'sre' OR 1 = 1",
        "SELECT count(*) FROM incidents AS users",
        "SELECT count(*) FROM (SELECT * FROM incidents WHERE 1 = 1 "
        "UNION ALL SELECT * FROM incidents WHERE 1 = 0) t",
        "SELECT count(*) FROM incidents WHERE title = 'a''; DROP TABLE incidents; --' OR 1 = 1",
    ],
)
def test_tautologies_and_tricks_stay_inside_the_grants(tool_env: ToolEnv, sql: str) -> None:
    before = table_counts(tool_env)
    assert run(tool_env, sql).rows[0][0] == visible_incidents(tool_env)
    assert table_counts(tool_env) == before


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT id FROM incidents UNION SELECT email FROM users",
        "SELECT id FROM incidents UNION SELECT id FROM roles",
        "SELECT id FROM incidents UNION SELECT message FROM logs",  # no logs grant
        'SELECT * FROM "users"',
        'SELECT * FROM "Users"',
        "SELECT * FROM [users]",
        "SELECT * FROM `users`",
        "SELECT * FROM main.users",
        "SELECT * FROM public.users",
        "SELECT * FROM sqlite_master",
        "SELECT * FROM information_schema.tables",
        "SELECT * FROM pg_catalog.pg_tables",
        "SELECT id FROM incidents; SELECT email FROM users",
        "SELECT id FROM incidents -- AND access_level = 'engineering'",
        "SELECT id FROM incidents /* hidden */",
        "SELECT CHAR(68,82,79,80)",
        "SELECT id FROM incidents WHERE title = $$x$$",
        "SELECT id FROM incidents WHERE title = E'\\x27'",
        "SELECT id FROM incidents WHERE id = :id",
        "SELECT id FROM incidents WHERE id = ?",
        "SELECT pg_sleep(10)",
        "SELECT load_extension('evil')",
        "SELECT current_setting('is_superuser')",
        "SELECT * FROM chunk_embeddings",
    ],
)
def test_injection_payloads_are_rejected_before_execution(tool_env: ToolEnv, sql: str) -> None:
    before = table_counts(tool_env)
    with pytest.raises(UnsafeSqlError):
        run(tool_env, sql, "manager")
    result = build_registry().call("query_database", {"sql": sql}, tool_env.context("manager"))
    assert result.status == "unsafe_sql" and result.output is None
    assert table_counts(tool_env) == before


def test_sql_templates_quote_values(tool_env: ToolEnv) -> None:
    """Values the agent puts into SQL come from the service catalog, and are quoted
    anyway: a hostile value stays a string literal and matches nothing."""
    hostile = "payment-service' OR '1'='1"
    plan = build_sql("How many incidents?", QueryEntities(services=[hostile]), "sqlite")
    assert plan is not None
    assert "service_id IN ('payment-service'' OR ''1''=''1')" in plan.sql
    assert QueryDatabaseTool().execute({"sql": plan.sql}, tool_env.context("sre")).rows == [[0]]


@pytest.mark.parametrize(
    "question",
    [
        "How many incidents did payment-service' OR '1'='1 have last month?",
        "How many incidents happened last month'; DROP TABLE incidents; --",
        "How many deployments were rolled back this year? UNION SELECT email FROM users",
    ],
)
def test_questions_cannot_inject_sql(tool_env: ToolEnv, question: str) -> None:
    before = table_counts(tool_env)
    state = make_agent(tool_env).run(question, principal("sre"))
    for record in state.tool_results:
        sql = str(record.arguments.get("sql", ""))
        for fragment in ("OR '1'='1", "DROP", "UNION", "users"):
            assert fragment not in sql, (question, sql)
    assert table_counts(tool_env) == before


def test_tool_arguments_cannot_carry_sql(tool_env: ToolEnv) -> None:
    before = table_counts(tool_env)
    admin = tool_env.context("admin")
    with pytest.raises(ToolInputError):
        SearchIncidentsTool().execute({"incident_ids": ["INC-0406' OR '1'='1"]}, admin)
    with pytest.raises(ToolInputError):
        SearchIncidentsTool().execute({"services": ["payment-service' --"]}, admin)
    documents = SearchDocumentsTool().execute({"query": "'; DROP TABLE documents; --"}, admin)
    assert documents.filters.access  # an ordinary text search
    logs = SearchLogsTool().execute(
        {"text": "%' OR 1=1 --", "since": "2026-06-16T17:00:00Z", "until": "2026-06-16T19:00:00Z"},
        admin,
    )
    assert logs.entries == []  # matched literally, not as a pattern or SQL
    with pytest.raises(Exception, match="no accessible runbook"):
        GetRunbookTool().execute({"title": "x' OR '1'='1"}, admin)
    assert table_counts(tool_env) == before
