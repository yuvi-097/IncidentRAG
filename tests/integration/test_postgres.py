"""Checks against a real PostgreSQL + pgvector (e.g. `docker compose up -d db`).

Opt in with OPSRAG_RUN_INTEGRATION_TESTS=1; connection settings come from the
environment / .env exactly as the API would read them.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.config import load_settings
from app.database import check_database, create_db_engine
from app.main import create_app
from app.schemas.health import ComponentStatus

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("OPSRAG_RUN_INTEGRATION_TESTS") != "1",
        reason="set OPSRAG_RUN_INTEGRATION_TESTS=1 with PostgreSQL running",
    ),
]


@pytest.fixture(scope="module")
def engine() -> Iterator[Engine]:
    db_engine = create_db_engine(load_settings().database)
    yield db_engine
    db_engine.dispose()


def test_database_is_reachable(engine: Engine) -> None:
    assert check_database(engine).status is ComponentStatus.OK


def test_pgvector_extension_is_installed(engine: Engine) -> None:
    with engine.connect() as connection:
        version = connection.execute(
            text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        ).scalar_one_or_none()
    assert version is not None, "pgvector missing: was scripts/db/*.sql run on first start?"


def test_health_endpoint_reports_ok() -> None:
    with TestClient(create_app(load_settings())) as client:
        body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["checks"]["database"]["status"] == "ok"
