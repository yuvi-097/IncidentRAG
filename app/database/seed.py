"""Load a synthetic dataset into the database.

Seeding *replaces* the contents of every OpsRAG table in one transaction, so it
is idempotent (running it twice leaves the same state) and atomic (a failure
leaves the previous contents untouched). It never drops or alters tables; use
``schema.drop_schema`` / ``create_schema`` for that.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator, Sequence
from typing import Any

from sqlalchemy import Connection, delete, insert, text
from sqlalchemy.engine import Engine

from app.database.base import Base
from app.database.models import (
    CodeFile,
    Deployment,
    Document,
    Incident,
    LogEntry,
    PullRequest,
    PullRequestFile,
    Role,
    Service,
    ServiceDependency,
    User,
)
from app.database.schema import PERSISTENT_TABLES
from app.synthetic.records import SyntheticDataset

logger = logging.getLogger(__name__)

# Table name in the dataset -> model, in foreign-key-safe insertion order.
LOAD_ORDER: tuple[tuple[str, type[Base]], ...] = (
    ("roles", Role),
    ("users", User),
    ("services", Service),
    ("service_dependencies", ServiceDependency),
    ("code_files", CodeFile),
    ("deployments", Deployment),
    ("pull_requests", PullRequest),
    ("pull_request_files", PullRequestFile),
    ("documents", Document),
    ("incidents", Incident),
    ("logs", LogEntry),
)


def _batches(rows: Sequence[dict[str, Any]], size: int) -> Iterator[Sequence[dict[str, Any]]]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def clear_tables(connection: Connection) -> None:
    """Remove all rows from every dataset table (including derived document chunks).
    Tables that are not part of the dataset (API tokens) are left alone."""
    tables = [t for t in Base.metadata.sorted_tables if t.name not in PERSISTENT_TABLES]
    if connection.dialect.name == "postgresql":
        names = ", ".join(f'"{table.name}"' for table in tables)
        connection.execute(text(f"TRUNCATE {names} RESTART IDENTITY"))
        return
    for table in reversed(tables):
        connection.execute(delete(table))


def seed_database(
    engine: Engine, dataset: SyntheticDataset, batch_size: int = 2000
) -> dict[str, int]:
    """Replace all table contents with ``dataset``. Returns rows inserted per table."""
    started = time.perf_counter()
    counts: dict[str, int] = {}
    with engine.begin() as connection:
        clear_tables(connection)
        for name, model in LOAD_ORDER:
            rows = [record.model_dump() for record in dataset.table(name)]
            for batch in _batches(rows, batch_size):
                connection.execute(insert(model), list(batch))
            counts[name] = len(rows)
    logger.info(
        "seed.completed",
        extra={"rows": sum(counts.values()), "seconds": round(time.perf_counter() - started, 2)},
    )
    return counts


def table_counts(engine: Engine) -> dict[str, int]:
    with engine.connect() as connection:
        return {
            table.name: connection.execute(
                text(f'SELECT count(*) FROM "{table.name}"')
            ).scalar_one()
            for table in Base.metadata.sorted_tables
        }
