"""query_database: read-only SQL over the operational tables, filtered per caller.

Layers, in order:
1. ``sql_guard.check_sql``: static checks, before anything reaches the database.
2. Access-filtered views: the query runs behind a ``WITH`` clause that defines a
   CTE for *every* table in the schema, named like the table, so every unqualified
   reference resolves to the CTE instead of the table:
   - tables the caller may read are filtered by the grant for the kind of data
     they hold (``incidents`` rows to the caller's incident labels, ``logs`` only
     with a logs grant, ``documents`` per document type);
   - all other tables (users, roles, chunks, embeddings...) become empty.
   Even a reference the static checker missed cannot reach unfiltered rows.
3. Execution: exactly one statement (a server-side prepared statement on
   PostgreSQL), in a read-only transaction locked by a first query (PostgreSQL) or
   ``PRAGMA query_only`` (SQLite), with a statement timeout and a row limit.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from pydantic import Field
from sqlalchemy import inspect, text
from sqlalchemy.engine import Connection, Engine

from app.database import models as _models  # noqa: F401  (registers every table)
from app.database.base import Base
from app.schemas.enums import AccessLevel, DocumentType, Resource, ToolPermission
from app.security.principal import Principal
from app.tools.base import (
    Tool,
    ToolContext,
    ToolExecutionError,
    ToolModel,
    truncate,
)
from app.tools.common import pull_request_levels
from app.tools.sql_guard import check_sql

# How each readable table is filtered: the kind of data it holds, and the rule.
LABELLED = "labelled"  # its own access_level column, against the grant for its kind
UNLABELLED = "unlabelled"  # no labels: readable whole with a grant for the unlabelled level
DOCUMENT_ROWS = "documents"  # runbooks, postmortems and other documents have own grants
PULL_REQUEST = "pull_request"  # visible if every changed file is (code or deployments grant)
PULL_REQUEST_FILE = "pull_request_file"  # a diff is as sensitive as its file
READABLE_TABLES: dict[str, tuple[Resource, str]] = {
    "services": (Resource.CATALOG, UNLABELLED),
    "service_dependencies": (Resource.CATALOG, UNLABELLED),
    "incidents": (Resource.INCIDENTS, LABELLED),
    "deployments": (Resource.DEPLOYMENTS, UNLABELLED),
    "pull_requests": (Resource.DEPLOYMENTS, PULL_REQUEST),
    "pull_request_files": (Resource.CODE, PULL_REQUEST_FILE),
    "code_files": (Resource.CODE, LABELLED),
    "documents": (Resource.DOCUMENTS, DOCUMENT_ROWS),
    "logs": (Resource.LOGS, UNLABELLED),
}
_DOCUMENT_GRANTS = (
    (DocumentType.RUNBOOK, Resource.RUNBOOKS),
    (DocumentType.POSTMORTEM, Resource.INCIDENTS),
)


def _readable(principal: Principal, resource: Resource, rule: str) -> bool:
    if rule == UNLABELLED:
        return principal.may_read_unlabelled(resource)
    if rule == PULL_REQUEST:
        return bool(pull_request_levels(principal))
    if rule == DOCUMENT_ROWS:
        return any(
            principal.labels(r) for r in (Resource.DOCUMENTS, Resource.RUNBOOKS, Resource.INCIDENTS)
        )
    return bool(principal.labels(resource))


def readable_tables(principal: Principal) -> list[str]:
    """Tables this caller may query at all (rows are filtered further by label)."""
    return [
        table
        for table, (resource, rule) in READABLE_TABLES.items()
        if _readable(principal, resource, rule)
    ]


def _in(levels: list[AccessLevel]) -> str:
    # Enum values only (never user input), so inlining them is safe.
    return "(" + ", ".join(f"'{level.value}'" for level in levels) + ")" if levels else "(NULL)"


def _document_rows(principal: Principal) -> str:
    own = [doc_type for doc_type, _ in _DOCUMENT_GRANTS]
    parts = [
        f"(doc_type = '{doc_type.value}' AND "
        f"access_level IN {_in(principal.visible_levels(resource))})"
        for doc_type, resource in _DOCUMENT_GRANTS
        if principal.labels(resource)
    ]
    if principal.labels(Resource.DOCUMENTS):
        others = ", ".join(f"'{d.value}'" for d in own)
        parts.append(
            f"(doc_type NOT IN ({others}) AND "
            f"access_level IN {_in(principal.visible_levels(Resource.DOCUMENTS))})"
        )
    return " OR ".join(parts) or "1 = 0"


def _schema_prefix(engine: Engine) -> str:
    if engine.dialect.name == "sqlite":
        return "main."
    schema = inspect(engine).default_schema_name
    return f"{schema}." if schema else ""


def shadow_views(engine: Engine, principal: Principal) -> dict[str, str]:
    """table -> the SELECT that replaces it for this caller."""
    q = _schema_prefix(engine)
    allowed = set(readable_tables(principal))
    views: dict[str, str] = {}
    for table in sorted(Base.metadata.tables):
        resource, rule = READABLE_TABLES.get(table, (None, None))
        if table not in allowed or resource is None:
            # No reference to the real table: nothing to read, and no privilege needed.
            views[table] = "SELECT NULL AS unavailable WHERE 1 = 0"
        elif rule == LABELLED:
            levels = _in(principal.visible_levels(resource))
            views[table] = f"SELECT * FROM {q}{table} WHERE access_level IN {levels}"
        elif rule == DOCUMENT_ROWS:
            views[table] = f"SELECT * FROM {q}{table} WHERE {_document_rows(principal)}"
        elif rule == PULL_REQUEST:
            levels = _in(pull_request_levels(principal))
            views[table] = (
                f"SELECT * FROM {q}pull_requests pr WHERE NOT EXISTS ("
                f"SELECT 1 FROM {q}pull_request_files f JOIN {q}code_files c "
                f"ON c.id = f.code_file_id WHERE f.pull_request_id = pr.id "
                f"AND c.access_level NOT IN {levels})"
            )
        elif rule == PULL_REQUEST_FILE:
            levels = _in(principal.visible_levels(Resource.CODE))
            views[table] = (
                f"SELECT f.* FROM {q}pull_request_files f JOIN {q}code_files c "
                f"ON c.id = f.code_file_id WHERE c.access_level IN {levels}"
            )
        else:  # UNLABELLED, already checked by readable_tables
            views[table] = f"SELECT * FROM {q}{table}"
    return views


def guarded_sql(sql: str, engine: Engine, principal: Principal) -> tuple[str, list[str]]:
    """The statement actually executed, and the tables it reads."""
    checked = check_sql(sql, readable_tables(principal), Base.metadata.tables)
    views = shadow_views(engine, principal)
    prefix = ",\n".join(f"{name} AS ({view})" for name, view in views.items())
    body = checked.sql
    if checked.starts_with_with:
        body = ", " + body[len("with") :].lstrip()  # continue our WITH list
    return f"WITH {prefix}\n{body}", sorted(checked.tables)


class QueryDatabaseInput(ToolModel):
    sql: str = Field(min_length=1, max_length=5000, description="One read-only SELECT query.")
    max_rows: int = Field(default=50, ge=1, le=5000)


class QueryDatabaseOutput(ToolModel):
    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    truncated: bool  # more rows existed than max_rows
    tables: list[str]  # tables the query read
    role: str  # rows this role may not read were invisible to the query
    elapsed_ms: float


def _json_value(value: Any, max_chars: int) -> Any:
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, bytes | bytearray | memoryview):
        return "<binary>"
    if isinstance(value, str):
        return truncate(value, max_chars)
    return value


@contextmanager
def _read_only(connection: Connection, timeout_seconds: float) -> Iterator[Any]:
    """A DB-API cursor inside a read-only, time-limited transaction (rolled back)."""
    raw = connection.connection.dbapi_connection
    assert raw is not None
    dialect = connection.dialect.name
    transaction = connection.begin()
    try:
        cursor = raw.cursor()
        if dialect == "postgresql":
            cursor.execute("SET TRANSACTION READ ONLY")
            cursor.execute(f"SET LOCAL statement_timeout = {int(timeout_seconds * 1000)}")
            # After the first query PostgreSQL refuses to switch the transaction back to
            # read-write ("must be set before any query").
            cursor.execute("SELECT 1")
            cursor.fetchall()
            yield cursor
        elif dialect == "sqlite":
            deadline = time.monotonic() + timeout_seconds
            cursor.execute("PRAGMA query_only = ON")
            raw.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
            try:
                yield cursor
            finally:
                raw.set_progress_handler(None, 0)
                cursor.execute("PRAGMA query_only = OFF")
        else:  # pragma: no cover - only PostgreSQL and SQLite are supported
            raise ToolExecutionError(f"query_database does not support {dialect}")
    finally:
        transaction.rollback()


def ensure_sql_reader(admin: Engine, role: str, password: str) -> None:
    """Create or update the read-only login ``query_database`` connects as, and grant it
    SELECT on exactly the tables the tool exposes (nothing on users, roles, chunks or
    embeddings). Run with a connection allowed to create roles; idempotent."""
    if admin.dialect.name != "postgresql":
        raise ToolExecutionError("a dedicated SQL role needs PostgreSQL")
    if not re.fullmatch(r"[a-z_][a-z0-9_]{0,62}", role):
        raise ValueError(f"invalid role name: {role!r}")
    with admin.begin() as connection:
        exists = connection.execute(
            text("SELECT 1 FROM pg_roles WHERE rolname = :role"), {"role": role}
        ).first()
        # A password cannot be a bind parameter in DDL; quote it as a literal.
        literal = "'" + password.replace("'", "''") + "'"
        if exists:
            # PostgreSQL only lets a role restate attributes it holds itself (SUPERUSER,
            # CREATEDB...), so an update sets the password alone; attributes are checked
            # below instead.
            connection.execute(text(f"ALTER ROLE {role} LOGIN PASSWORD {literal}"))
        else:
            connection.execute(
                text(
                    f"CREATE ROLE {role} LOGIN PASSWORD {literal} NOSUPERUSER NOCREATEDB "
                    "NOCREATEROLE NOINHERIT NOREPLICATION NOBYPASSRLS"
                )
            )
        privileged = connection.execute(
            text(
                "SELECT rolsuper OR rolreplication OR rolbypassrls OR rolcreaterole "
                "OR rolcreatedb FROM pg_roles WHERE rolname = :role"
            ),
            {"role": role},
        ).scalar_one()
        if privileged:
            raise ToolExecutionError(f"role {role} is privileged; it cannot be the SQL reader")
        connection.execute(text(f"ALTER ROLE {role} SET default_transaction_read_only = on"))
        database = connection.execute(text("SELECT current_database()")).scalar_one()
        connection.execute(text(f'GRANT CONNECT ON DATABASE "{database}" TO {role}'))
    grant_sql_reader(admin, role)


def grant_sql_reader(admin: Engine, role: str) -> None:
    """(Re)apply the reader's table grants; needed again after tables are recreated."""
    schema = _schema_prefix(admin).rstrip(".") or "public"
    with admin.begin() as connection:
        connection.execute(text(f"GRANT USAGE ON SCHEMA {schema} TO {role}"))
        connection.execute(text(f"REVOKE ALL ON ALL TABLES IN SCHEMA {schema} FROM {role}"))
        for table in READABLE_TABLES:
            connection.execute(text(f"GRANT SELECT ON {schema}.{table} TO {role}"))


def execute_one(cursor: Any, sql: str, dialect: str) -> None:
    """Run exactly one statement, enforced by the driver or the server: PostgreSQL gets
    it as a prepared statement, which it refuses to build from several statements;
    SQLite's driver refuses several statements itself."""
    if dialect == "postgresql":
        cursor.execute(sql, prepare=True)
    else:
        cursor.execute(sql)


class QueryDatabaseTool(Tool[QueryDatabaseInput, QueryDatabaseOutput]):
    name = "query_database"
    description = (
        "Run one read-only SELECT (or WITH ... SELECT) over the operational tables: "
        "services, service_dependencies, incidents, deployments, pull_requests, "
        "pull_request_files, code_files, documents, logs. Use it for counts, "
        "aggregates and rankings. No writes, comments, parameters or schema-qualified "
        "names; rows your role may not read are invisible."
    )
    permission = ToolPermission.SQL_READ
    input_model = QueryDatabaseInput
    output_model = QueryDatabaseOutput

    def _run(self, arguments: QueryDatabaseInput, context: ToolContext) -> QueryDatabaseOutput:
        engine = context.sql_engine or context.engine
        sql, tables = guarded_sql(arguments.sql, engine, context.principal)
        limit = min(arguments.max_rows, context.settings.sql_max_rows)
        started = time.perf_counter()
        try:
            timeout = context.settings.sql_timeout_seconds
            with (
                engine.connect() as connection,
                _read_only(connection, timeout) as cursor,
            ):
                execute_one(cursor, sql, connection.dialect.name)
                columns = [c[0] for c in cursor.description or []]
                fetched = cursor.fetchmany(limit + 1)
        except ToolExecutionError:
            raise
        except Exception as exc:  # DB errors: report the database's message only
            message = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
            raise ToolExecutionError(f"query failed: {message}") from exc
        max_chars = context.settings.snippet_chars
        rows = [[_json_value(v, max_chars) for v in row] for row in fetched[:limit]]
        return QueryDatabaseOutput(
            columns=columns,
            rows=rows,
            row_count=len(rows),
            truncated=len(fetched) > limit,
            tables=tables,
            role=context.principal.role,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
        )
