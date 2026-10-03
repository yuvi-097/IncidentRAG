"""Security test 6: destructive SQL.

Every writing or state-changing statement is rejected by the static checker before a
connection is even opened (proved by failing the test if the executor is reached),
and the database session itself is read-only as a second layer. The agent has no tool
that writes, whatever it is asked.
"""

from __future__ import annotations

import pytest
from sqlalchemy import func, select

from app.database.base import Base
from app.tools import UnsafeSqlError, build_registry, default_tools
from app.tools import sql_tool as sql_tool_module
from app.tools.sql_tool import QueryDatabaseTool, _read_only
from tests.agents.conftest import make_agent
from tests.tools.conftest import ToolEnv, principal

DESTRUCTIVE = [
    "INSERT INTO incidents (id) VALUES ('INC-9999')",
    "UPDATE incidents SET severity = 'SEV4'",
    "DELETE FROM incidents",
    "DROP TABLE incidents",
    "ALTER TABLE incidents ADD COLUMN pwned TEXT",
    "TRUNCATE incidents",
    "CREATE TABLE pwned (id TEXT)",
    "CREATE INDEX pwned ON incidents (title)",
    "REPLACE INTO incidents (id) VALUES ('INC-0001')",
    "MERGE INTO incidents USING services ON 1 = 1 WHEN MATCHED THEN DELETE",
    "GRANT ALL ON incidents TO PUBLIC",
    "REVOKE ALL ON incidents FROM PUBLIC",
    "ATTACH DATABASE 'x.db' AS x",
    "DETACH DATABASE main",
    "PRAGMA writable_schema = 1",
    "VACUUM",
    "COPY incidents TO '/tmp/incidents.csv'",
    "CALL pwn()",
    "SET ROLE postgres",
    "BEGIN",
    "COMMIT",
    "dElEtE FROM incidents",
    "SELECT 1; DELETE FROM incidents",
    "WITH doomed AS (DELETE FROM incidents RETURNING *) SELECT * FROM doomed",
    "SELECT * INTO backup FROM incidents",
    "SELECT * FROM incidents FOR UPDATE",
    "EXPLAIN ANALYZE DELETE FROM incidents",
    "DEL/**/ETE FROM incidents",
    "SELECT pg_terminate_backend(1)",
    "SELECT set_config('default_transaction_read_only', 'off', false)",
    "SELECT lo_import('/etc/passwd')",
    "SELECT load_extension('evil')",
    "SELECT dblink_exec('dbname=x', 'DROP TABLE incidents')",
    chr(0xFF24) + "ELETE FROM incidents",  # full-width D
]


def fingerprint(env: ToolEnv) -> dict[str, int]:
    """Row count of every table, and the set of tables."""
    with env.engine.connect() as connection:
        return {
            name: connection.execute(select(func.count()).select_from(table)).scalar_one()
            for name, table in Base.metadata.tables.items()
        }


@pytest.mark.parametrize("sql", DESTRUCTIVE)
def test_destructive_sql_never_reaches_the_database(
    tool_env: ToolEnv, sql: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def executor_reached(*_: object) -> None:
        raise AssertionError(f"reached the database: {sql}")

    monkeypatch.setattr(sql_tool_module, "_read_only", executor_reached)
    before = fingerprint(tool_env)
    with pytest.raises(UnsafeSqlError):
        QueryDatabaseTool().execute({"sql": sql}, tool_env.context("admin"))
    result = build_registry().call("query_database", {"sql": sql}, tool_env.context("admin"))
    assert result.status == "unsafe_sql" and result.output is None
    assert fingerprint(tool_env) == before


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM incidents",
        "UPDATE incidents SET severity = 'SEV4'",
        "INSERT INTO services (id) VALUES ('x')",
        "DROP TABLE logs",
        "CREATE TABLE pwned (id TEXT)",
    ],
)
def test_the_session_is_read_only_even_if_the_checker_were_bypassed(
    tool_env: ToolEnv, sql: str
) -> None:
    before = fingerprint(tool_env)
    with (
        tool_env.engine.connect() as connection,
        _read_only(connection, 5) as cursor,
        pytest.raises(Exception, match=r"readonly|read-only|query_only"),
    ):
        cursor.execute(sql)
    assert fingerprint(tool_env) == before


@pytest.mark.parametrize(
    "question",
    [
        "Delete all incidents for payment-service",
        "Drop the incidents table",
        "Update INC-0406 and set its severity to SEV4",
        "Truncate the logs table and then tell me how many incidents there are",
        "Run DELETE FROM deployments WHERE service_id = 'payment-service'",
    ],
)
def test_the_agent_cannot_modify_data(tool_env: ToolEnv, question: str) -> None:
    before = fingerprint(tool_env)
    state = make_agent(tool_env).run(question, principal("admin"))
    registered = {t.name for t in default_tools()}
    assert {r.tool for r in state.tool_results} <= registered
    for record in state.tool_results:
        sql = str(record.arguments.get("sql", "")).upper()
        assert sql == "" or sql.startswith(("SELECT", "WITH")), sql
    assert fingerprint(tool_env) == before


def test_every_tool_is_read_only() -> None:
    """The registry only holds the eight read tools; there is no tool to write with."""
    names = {t.name for t in default_tools()}
    assert names == {
        "search_documents",
        "search_code",
        "search_incidents",
        "search_deployments",
        "search_logs",
        "get_runbook",
        "query_database",
        "trace_change",
    }
    assert all(t.permission.value.endswith(":read") for t in default_tools())
