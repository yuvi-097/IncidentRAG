"""Security test 3: unauthorized logs.

Only SREs and admins may read logs. Every path to log lines is closed for other
roles: the log tool (permission check before it runs), the ``logs`` table through
SQL (not queryable, and empty even if reached), and the agent, which cannot plan the
call and says so.
"""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI

from app.agents.state import EvidenceKind
from app.schemas.enums import AccessLevel, Resource
from app.tools import ToolPermissionError, UnsafeSqlError, build_registry
from app.tools.logs import SearchLogsTool
from app.tools.sql_tool import QueryDatabaseTool, guarded_sql
from tests.agents.conftest import make_agent
from tests.security.conftest import api_client, ask_api, user_of
from tests.tools.conftest import ToolEnv, custom_principal, principal

WINDOW = {"since": "2026-06-16T17:00:00Z", "until": "2026-06-16T19:00:00Z"}
QUESTION = "Why did payment-service fail after deployment v2.8.1?"
LOGGED_ERROR = "QueuePool limit"  # the error payment-service logged during INC-0406


@pytest.mark.parametrize("role", ["developer", "manager"])
def test_the_log_tool_refuses_roles_without_a_logs_grant(tool_env: ToolEnv, role: str) -> None:
    with pytest.raises(ToolPermissionError, match="logs:read"):
        SearchLogsTool().execute(
            {**WINDOW, "services": ["payment-service"]}, tool_env.context(role)
        )
    result = build_registry().call("search_logs", WINDOW, tool_env.context(role))
    assert result.status == "permission_denied" and result.output is None


def test_a_logs_grant_without_the_label_is_not_enough(tool_env: ToolEnv) -> None:
    """Log lines carry no label of their own; they count as engineering material."""
    only_sre = custom_principal({Resource.LOGS: {AccessLevel.SRE}})
    with pytest.raises(ToolPermissionError, match="may not read logs"):
        SearchLogsTool().execute(WINDOW, tool_env.context(only_sre))


def test_sql_cannot_read_logs_without_the_grant(tool_env: ToolEnv) -> None:
    for sql in (
        "SELECT message FROM logs",
        "SELECT id FROM incidents UNION SELECT message FROM logs",
    ):
        with pytest.raises(UnsafeSqlError, match="not available"):
            QueryDatabaseTool().execute({"sql": sql}, tool_env.context("manager"))
    statement, _ = guarded_sql(
        "SELECT count(*) FROM incidents", tool_env.engine, principal("manager")
    )
    # empty even if a reference slipped through, without touching the real table
    assert "logs AS (SELECT NULL AS unavailable WHERE 1 = 0)" in statement
    sre = QueryDatabaseTool().execute({"sql": "SELECT count(*) FROM logs"}, tool_env.context("sre"))
    assert sre.rows[0][0] == len(tool_env.dataset.logs)  # positive control


def test_the_agent_does_not_plan_log_searches_for_other_roles(tool_env: ToolEnv) -> None:
    agent = make_agent(tool_env)
    for role in ("developer", "manager"):
        state = agent.run(QUESTION, principal(role))
        assert "search_logs" not in {r.tool for r in state.tool_results}, role
        assert EvidenceKind.LOGS not in {i.kind for i in state.retrieved_documents}, role
        assert LOGGED_ERROR not in state.final_answer, role
        assert any("search_logs is not permitted" in n for n in state.limitations), role
    sre = agent.run(QUESTION, principal("sre"))
    assert "search_logs" in {r.tool for r in sre.tool_results}  # positive control
    assert LOGGED_ERROR in sre.final_answer


def test_the_api_returns_no_log_lines_to_other_roles(app: FastAPI, tool_env: ToolEnv) -> None:
    client = api_client(app, tool_env)
    body = ask_api(client, user_of(tool_env, "developer"), QUESTION)
    assert LOGGED_ERROR not in json.dumps(body)
    assert all(e["kind"] != "logs" for e in body["evidence"])
    assert "search_logs" not in {t["tool"] for t in body["tools"]}
