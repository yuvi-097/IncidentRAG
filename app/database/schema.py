"""Schema lifecycle: extensions, create, drop, and drift detection.

Tables are created with ``metadata.create_all`` (idempotent; existing tables are
left untouched), which cannot change an existing table. ``schema_drift`` finds what it
cannot fix (columns added or removed, enum values changed in CHECK constraints, as
between Phase 7 and Phase 8), and ``prepare_schema`` then recreates the tables. That is
safe for the dataset tables because seeding replaces their contents anyway; tables in
``PERSISTENT_TABLES`` (API tokens) are kept unless they drifted themselves.
"""

from __future__ import annotations

import logging
import re

from sqlalchemy import Table, inspect, text
from sqlalchemy.engine import Engine

import app.database.models  # noqa: F401  (registers all tables on Base.metadata)
from app.database.base import Base

logger = logging.getLogger(__name__)

POSTGRES_EXTENSIONS = ("vector",)


def ensure_extensions(engine: Engine) -> None:
    """Enable required PostgreSQL extensions (needs CREATE privilege on the database)."""
    if engine.dialect.name != "postgresql":
        return
    with engine.begin() as connection:
        for extension in POSTGRES_EXTENSIONS:
            connection.execute(text(f'CREATE EXTENSION IF NOT EXISTS "{extension}"'))


# Tables that are not part of the synthetic dataset: never cleared by seeding.
PERSISTENT_TABLES = frozenset({"api_tokens"})
_LITERAL = re.compile(r"'((?:[^']|'')*)'")


def create_schema(engine: Engine) -> None:
    ensure_extensions(engine)
    Base.metadata.create_all(engine)
    logger.info("schema.created", extra={"tables": len(Base.metadata.tables)})


def drop_schema(engine: Engine, keep: frozenset[str] = PERSISTENT_TABLES) -> None:
    tables = [t for t in Base.metadata.sorted_tables if t.name not in keep]
    Base.metadata.drop_all(engine, tables=tables)
    logger.info("schema.dropped", extra={"tables": len(tables), "kept": sorted(keep)})


def _enum_drift(table: Table, checks: list[dict[str, object]]) -> list[str]:
    by_name = {str(c.get("name")): str(c.get("sqltext")) for c in checks}
    problems = []
    for column in table.columns:
        enums = getattr(column.type, "enums", None)
        if not enums:
            continue
        sqltext = by_name.get(f"ck_{table.name}_{column.name}")
        if sqltext is None:
            problems.append(f"{table.name}.{column.name}: CHECK constraint missing")
            continue
        allowed = {v.replace("''", "'") for v in _LITERAL.findall(sqltext)}
        if allowed != set(enums):
            problems.append(
                f"{table.name}.{column.name}: allowed values differ "
                f"(missing {sorted(set(enums) - allowed)}, obsolete {sorted(allowed - set(enums))})"
            )
    return problems


def schema_drift(engine: Engine) -> dict[str, list[str]]:
    """table -> differences ``create_all`` cannot fix. Missing tables are not drift."""
    inspector = inspect(engine)
    existing = set(inspector.get_table_names())
    drift: dict[str, list[str]] = {}
    for table in Base.metadata.sorted_tables:
        if table.name not in existing:
            continue
        problems = []
        columns = {c["name"] for c in inspector.get_columns(table.name)}
        wanted = {c.name for c in table.columns}
        if columns != wanted:
            problems.append(
                f"{table.name}: columns differ (missing {sorted(wanted - columns)}, "
                f"obsolete {sorted(columns - wanted)})"
            )
        problems += _enum_drift(table, inspector.get_check_constraints(table.name))
        if problems:
            drift[table.name] = problems
    return drift


def prepare_schema(engine: Engine, recreate: bool = False) -> list[str]:
    """Create missing tables; recreate them all when ``recreate`` or when the schema
    drifted. Returns the drift that triggered a rebuild (empty if none)."""
    drift = schema_drift(engine)
    problems = [p for table in drift.values() for p in table]
    if recreate or drift:
        keep = PERSISTENT_TABLES - set(drift)
        if drift:
            logger.warning("schema.drift", extra={"problems": problems[:20], "kept": sorted(keep)})
        drop_schema(engine, keep=keep)
    create_schema(engine)
    return problems


def table_names() -> list[str]:
    """Tables in dependency order (parents before children)."""
    return [table.name for table in Base.metadata.sorted_tables]
