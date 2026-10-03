"""query_database on a real (SQLite) database: results, filtering by the caller's
grants, and the runtime guards behind the static checker."""

from __future__ import annotations

import pytest
from sqlalchemy import func, select

from app.database.models import CodeFile, Document, Incident
from app.schemas.enums import AccessLevel, DocumentType, Resource, ToolPermission
from app.security import Principal
from app.tools import ToolExecutionError, UnsafeSqlError
from app.tools.sql_tool import (
    QueryDatabaseTool,
    _read_only,
    guarded_sql,
    readable_tables,
    shadow_views,
)
from tests.tools.conftest import ToolEnv, custom_principal, principal

TOOL = QueryDatabaseTool()
BASIC = {AccessLevel.PUBLIC, AccessLevel.ENGINEERING}
# SQL over non-sensitive incidents, documents and code only.
LIMITED = custom_principal(
    {Resource.INCIDENTS: BASIC, Resource.DOCUMENTS: BASIC, Resource.CODE: BASIC},
    frozenset({ToolPermission.SQL_READ}),
)


def run(env: ToolEnv, sql: str, role: str | Principal = "sre", **settings: object):
    return TOOL.execute({"sql": sql}, env.context(role, **settings))


def count_where(env: ToolEnv, *conditions: object) -> int:
    with env.engine.connect() as connection:
        return connection.execute(
            select(func.count()).select_from(Incident).where(*conditions)
        ).scalar_one()


def test_results_match_the_database(tool_env: ToolEnv) -> None:
    out = run(
        tool_env,
        "SELECT service_id, count(*) AS n FROM incidents WHERE access_level = 'engineering' "
        "GROUP BY service_id ORDER BY n DESC, service_id LIMIT 3",
    )
    assert out.columns == ["service_id", "n"] and out.row_count == 3 and out.tables == ["incidents"]
    for service, n in out.rows:
        assert n == count_where(
            tool_env, Incident.service_id == service, Incident.access_level == "engineering"
        )


@pytest.mark.parametrize(
    ("role", "levels"),
    [
        (LIMITED, BASIC),
        ("sre", BASIC | {AccessLevel.SRE}),
        ("manager", BASIC | {AccessLevel.SRE, AccessLevel.MANAGER}),
        ("admin", set(AccessLevel) - {AccessLevel.CONFIDENTIAL}),
    ],
    ids=["limited", "sre", "manager", "admin"],
)
def test_rows_the_caller_may_not_read_are_invisible(
    tool_env: ToolEnv, role: str | Principal, levels: set[AccessLevel]
) -> None:
    out = run(tool_env, "SELECT access_level, count(*) FROM incidents GROUP BY access_level", role)
    seen = {AccessLevel(level): n for level, n in out.rows}
    assert set(seen) <= levels
    for level, n in seen.items():
        assert n == count_where(tool_env, Incident.access_level == level)
    total = run(tool_env, "SELECT count(*) FROM incidents", role).rows[0][0]
    assert total == count_where(tool_env, Incident.access_level.in_(sorted(levels)))
    assert out.role == (role.role if isinstance(role, Principal) else role)


def test_every_way_of_referencing_a_table_is_filtered(tool_env: ToolEnv) -> None:
    visible = count_where(tool_env, Incident.access_level.in_(["public", "engineering"]))
    assert visible < count_where(tool_env)  # some incidents are hidden from LIMITED
    for sql in (
        "SELECT count(*) FROM incidents",
        "SELECT count(*) FROM (SELECT * FROM incidents) t",
        "SELECT (SELECT count(*) FROM incidents)",
        "WITH x AS (SELECT * FROM incidents) SELECT count(*) FROM x",
        "SELECT count(*) FROM incidents a "
        "WHERE EXISTS (SELECT 1 FROM incidents b WHERE b.id = a.id)",
        "SELECT count(*) FROM (SELECT id FROM incidents UNION SELECT id FROM incidents) u",
        "SELECT count(*) FROM incidents WHERE access_level = 'sre' OR 1 = 1",
    ):
        assert run(tool_env, sql, LIMITED).rows[0][0] == visible, sql


def test_documents_code_and_pull_requests_follow_the_grants(tool_env: ToolEnv) -> None:
    basic = ["public", "engineering"]
    with tool_env.engine.connect() as connection:
        docs = connection.execute(
            select(func.count())
            .select_from(Document)
            .where(Document.access_level.in_(basic), Document.doc_type != DocumentType.RUNBOOK)
        ).scalar_one()
        code = connection.execute(
            select(func.count()).select_from(CodeFile).where(CodeFile.access_level.in_(basic))
        ).scalar_one()
    # LIMITED has no runbooks grant (runbooks hidden); postmortems follow its incidents grant.
    assert run(tool_env, "SELECT count(*) FROM documents", LIMITED).rows[0][0] == docs
    assert run(tool_env, "SELECT count(*) FROM code_files", LIMITED).rows[0][0] == code
    hidden_diffs = run(
        tool_env,
        "SELECT count(*) FROM pull_request_files f JOIN code_files c ON c.id = f.code_file_id "
        "WHERE c.access_level NOT IN ('public', 'engineering')",
        LIMITED,
    )
    assert hidden_diffs.rows[0][0] == 0
    limited_prs = run(tool_env, "SELECT count(*) FROM pull_requests", LIMITED).rows[0][0]
    admin_prs = run(tool_env, "SELECT count(*) FROM pull_requests", "admin").rows[0][0]
    assert 0 < limited_prs < admin_prs  # PRs touching SRE-labelled files are hidden


def test_labels_are_compartments_in_sql(tool_env: ToolEnv) -> None:
    """Managers read MANAGER reports but not SRE documents; SREs the reverse; nobody
    reads confidential rows, not even an admin."""
    sql = (
        "SELECT DISTINCT access_level FROM documents "
        "WHERE doc_type NOT IN ('runbook', 'postmortem')"
    )
    manager = {row[0] for row in run(tool_env, sql, "manager").rows}
    sre = {row[0] for row in run(tool_env, sql, "sre").rows}
    admin = {row[0] for row in run(tool_env, sql, "admin").rows}
    assert "manager" in manager and "sre" not in manager
    assert "sre" in sre and "manager" not in sre
    assert {"admin", "manager", "sre"} <= admin and "confidential" not in admin
    secret = "SELECT count(*) FROM documents WHERE access_level = 'confidential'"
    assert all(run(tool_env, secret, r).rows[0][0] == 0 for r in ("sre", "manager", "admin"))


def test_tables_depend_on_the_grants() -> None:
    assert "logs" in readable_tables(principal("sre"))
    assert "logs" not in readable_tables(principal("manager"))  # no logs grant
    assert "code_files" not in readable_tables(principal("sre"))  # no code grant
    assert "deployments" not in readable_tables(principal("manager"))
    assert {"code_files", "logs", "deployments"} <= set(readable_tables(principal("admin")))
    no_catalog = custom_principal({Resource.INCIDENTS: BASIC})
    assert "services" not in readable_tables(no_catalog)  # deny by default


def test_tables_outside_the_allowlist_are_empty_even_if_reached(tool_env: ToolEnv) -> None:
    views = shadow_views(tool_env.engine, principal("sre"))
    assert views["users"] == views["chunk_embeddings"] == "SELECT NULL AS unavailable WHERE 1 = 0"
    sql, _ = guarded_sql("SELECT count(*) FROM incidents", tool_env.engine, principal("sre"))
    # Bypass the checker on purpose: the executed statement still sees an empty table.
    with tool_env.engine.connect() as connection, _read_only(connection, 5) as cursor:
        cursor.execute(sql.replace("SELECT count(*) FROM incidents", "SELECT count(*) FROM users"))
        assert cursor.fetchone()[0] == 0


def test_the_database_itself_refuses_writes(tool_env: ToolEnv) -> None:
    before = count_where(tool_env)
    with (
        tool_env.engine.connect() as connection,
        _read_only(connection, 5) as cursor,
        pytest.raises(Exception, match=r"readonly|read-only|query_only"),
    ):
        cursor.execute("DELETE FROM incidents")
    assert count_where(tool_env) == before


def test_unsafe_sql_never_executes(tool_env: ToolEnv) -> None:
    before = count_where(tool_env)
    for sql in ("DELETE FROM incidents", "SELECT 1; DELETE FROM incidents", "DROP TABLE incidents"):
        with pytest.raises(UnsafeSqlError):
            run(tool_env, sql)
    assert count_where(tool_env) == before
    with pytest.raises(UnsafeSqlError, match="not available"):
        run(tool_env, "SELECT email FROM users")


def test_row_limits_and_truncation(tool_env: ToolEnv) -> None:
    out = TOOL.execute(
        {"sql": "SELECT id FROM incidents ORDER BY id", "max_rows": 5}, tool_env.context()
    )
    assert out.row_count == 5 and out.truncated
    capped = TOOL.execute(
        {"sql": "SELECT id FROM incidents", "max_rows": 100}, tool_env.context(sql_max_rows=7)
    )
    assert capped.row_count == 7 and capped.truncated
    small = run(tool_env, "SELECT id FROM services")
    assert not small.truncated and small.row_count == len(tool_env.dataset.services)


def test_values_are_json_safe_and_long_text_is_cut(tool_env: ToolEnv) -> None:
    out = TOOL.execute(
        {"sql": "SELECT started_at, symptoms FROM incidents ORDER BY id LIMIT 1"},
        tool_env.context(snippet_chars=100),
    )
    started, symptoms = out.rows[0]
    assert isinstance(started, str) and started.startswith("20")
    assert len(symptoms) <= 100
    out.model_dump_json()


def test_slow_queries_time_out(tool_env: ToolEnv) -> None:
    with pytest.raises(ToolExecutionError, match=r"interrupted|timeout|canceling"):
        TOOL.execute(
            {"sql": "SELECT count(*) FROM logs a, logs b"},
            tool_env.context(sql_timeout_seconds=0.2),
        )
    assert run(tool_env, "SELECT count(*) FROM services").rows[0][0] > 0  # connection still usable


def test_database_errors_are_reported_not_raised_raw(tool_env: ToolEnv) -> None:
    with pytest.raises(ToolExecutionError, match="query failed"):
        run(tool_env, "SELECT no_such_column FROM incidents")


def test_permission_is_required(tool_env: ToolEnv) -> None:
    from app.tools import ToolPermissionError

    assert not principal("developer").can(ToolPermission.SQL_READ)
    with pytest.raises(ToolPermissionError, match="sql:read"):
        run(tool_env, "SELECT 1", "developer")
