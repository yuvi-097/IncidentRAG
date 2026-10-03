"""The dedicated read-only SQL role, and schema drift, on real PostgreSQL.

Opt in with OPSRAG_RUN_INTEGRATION_TESTS=1. The configured user must be allowed to
create roles (the compose database user is). Uses a development database only.
"""

from __future__ import annotations

import os
import secrets
from collections.abc import Iterator

import pytest
from pydantic import SecretStr
from sqlalchemy import func, select, text
from sqlalchemy.engine import Engine

from app.config import ToolSettings, load_settings
from app.database.models import Incident
from app.database.schema import create_schema, drop_schema, prepare_schema, schema_drift
from app.database.seed import seed_database
from app.database.session import create_db_engine
from app.security import principal_for_role
from app.synthetic.records import SyntheticDataset
from app.tools import ToolContext
from app.tools.sql_tool import QueryDatabaseTool, ensure_sql_reader, grant_sql_reader

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("OPSRAG_RUN_INTEGRATION_TESTS") != "1",
        reason="set OPSRAG_RUN_INTEGRATION_TESTS=1 with PostgreSQL running",
    ),
]
ROLE = "opsrag_sql_reader_test"


@pytest.fixture(scope="module")
def engines(dataset: SyntheticDataset) -> Iterator[tuple[Engine, Engine]]:
    settings = load_settings()
    if settings.app.environment == "production":
        pytest.skip("refusing to write synthetic data into production")
    admin = create_db_engine(settings.database)
    prepare_schema(admin)
    seed_database(admin, dataset)
    password = secrets.token_urlsafe(24)
    ensure_sql_reader(admin, ROLE, password)
    reader = create_db_engine(
        settings.database.model_copy(update={"user": ROLE, "password": SecretStr(password)})
    )
    yield admin, reader
    reader.dispose()
    with admin.begin() as connection:  # the reader owns nothing; drop its grants, then it
        database = connection.execute(text("SELECT current_database()")).scalar_one()
        connection.execute(text(f"REVOKE ALL ON ALL TABLES IN SCHEMA public FROM {ROLE}"))
        connection.execute(text(f"REVOKE ALL ON SCHEMA public FROM {ROLE}"))
        connection.execute(text(f'REVOKE ALL ON DATABASE "{database}" FROM {ROLE}'))
        connection.execute(text(f"DROP ROLE IF EXISTS {ROLE}"))
    admin.dispose()


def context(admin: Engine, reader: Engine, role: str = "sre") -> ToolContext:
    return ToolContext(
        engine=admin,
        principal=principal_for_role(f"pg-{role}", role),
        settings=ToolSettings(_env_file=None),  # type: ignore[call-arg]
        sql_engine=reader,
    )


def test_the_tool_runs_as_the_reader(engines: tuple[Engine, Engine]) -> None:
    admin, reader = engines
    out = QueryDatabaseTool().execute(
        {"sql": "SELECT current_user, count(*) FROM incidents"}, context(admin, reader)
    )
    with admin.connect() as connection:
        total = connection.execute(select(func.count()).select_from(Incident)).scalar_one()
    assert out.rows == [[ROLE, total]]


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM incidents",
        "UPDATE incidents SET severity = 'SEV4'",
        "INSERT INTO services (id) VALUES ('x')",
        "CREATE TABLE pwned (id text)",
        "TRUNCATE incidents",
    ],
)
def test_the_reader_cannot_write_even_outside_a_read_only_transaction(
    engines: tuple[Engine, Engine], sql: str
) -> None:
    _, reader = engines
    with reader.connect() as connection:
        connection.execute(text("SET default_transaction_read_only = off"))  # its own setting
        with pytest.raises(Exception, match=r"permission denied|read-only"):
            connection.execute(text(sql))
        connection.rollback()


@pytest.mark.parametrize(
    "table", ["users", "roles", "api_tokens", "document_chunks", "chunk_embeddings"]
)
def test_the_reader_cannot_read_what_the_tool_hides(
    engines: tuple[Engine, Engine], table: str
) -> None:
    _, reader = engines
    with reader.connect() as connection, pytest.raises(Exception, match="permission denied"):
        connection.execute(text(f"SELECT * FROM {table} LIMIT 1"))


def test_grants_are_reapplied_after_tables_are_recreated(
    engines: tuple[Engine, Engine], dataset: SyntheticDataset
) -> None:
    admin, reader = engines
    drop_schema(admin)
    create_schema(admin)
    seed_database(admin, dataset)
    with reader.connect() as connection, pytest.raises(Exception, match="permission denied"):
        connection.execute(text("SELECT count(*) FROM incidents"))
    grant_sql_reader(admin, ROLE)
    out = QueryDatabaseTool().execute(
        {"sql": "SELECT count(*) FROM incidents"}, context(admin, reader)
    )
    assert out.rows[0][0] == len(dataset.incidents)


def test_a_schema_from_an_earlier_phase_is_detected_and_rebuilt(
    engines: tuple[Engine, Engine], dataset: SyntheticDataset
) -> None:
    """Phase 7 labels (internal/restricted) and the old roles column: seed_db's
    prepare_schema notices both and recreates the tables."""
    admin, _ = engines
    old_labels = "('public', 'internal', 'restricted', 'confidential')"
    with admin.begin() as connection:
        connection.execute(text("ALTER TABLE incidents DROP CONSTRAINT ck_incidents_access_level"))
        connection.execute(
            text(
                "ALTER TABLE incidents ADD CONSTRAINT ck_incidents_access_level "
                f"CHECK (access_level IN {old_labels}) NOT VALID"
            )
        )
        connection.execute(text("ALTER TABLE roles ADD COLUMN max_access_level varchar(12)"))
    drift = schema_drift(admin)
    assert set(drift) == {"incidents", "roles"}
    assert "internal" in " ".join(drift["incidents"])
    problems = prepare_schema(admin)
    assert problems and schema_drift(admin) == {}
    seed_database(admin, dataset)
    grant_sql_reader(admin, ROLE)
