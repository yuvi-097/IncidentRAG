"""Phase 8 security on real PostgreSQL: grants in SQL and chunk queries, the read-only
transaction, SQL injection and destructive SQL, and quarantine of planted injections.

Opt in with OPSRAG_RUN_INTEGRATION_TESTS=1. Seeds the (tainted) synthetic dataset into
the configured database; use a development database only.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from sqlalchemy import func, select
from sqlalchemy.engine import Engine

from app.config import ToolSettings, load_settings
from app.database.base import Base
from app.database.models import DocumentChunk
from app.database.seed import seed_database
from app.database.session import create_db_engine
from app.rag.chunking import ChunkingConfig
from app.rag.ingestion import IngestionPipeline
from app.rag.ingestion.persistence import ensure_chunk_table
from app.rag.retrieval import BM25Retriever
from app.schemas.enums import AccessLevel
from app.security import SOURCE_RESOURCE, principal_for_role
from app.synthetic.records import SyntheticDataset
from app.tools import ToolContext, UnsafeSqlError
from app.tools.search import SearchDocumentsTool
from app.tools.sql_tool import QueryDatabaseTool, _read_only, execute_one
from tests.agents.conftest import make_agent
from tests.security.conftest import INJECTION_DOC, taint
from tests.tools.conftest import NOW, ToolEnv, principal

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("OPSRAG_RUN_INTEGRATION_TESTS") != "1",
        reason="set OPSRAG_RUN_INTEGRATION_TESTS=1 with PostgreSQL running",
    ),
]


@pytest.fixture(scope="module")
def env(dataset: SyntheticDataset) -> Iterator[ToolEnv]:
    settings = load_settings()
    if settings.app.environment == "production":
        pytest.skip("refusing to write synthetic data into production")
    engine = create_db_engine(settings.database)
    tainted = taint(dataset)
    ensure_chunk_table(engine)
    seed_database(engine, tainted)
    IngestionPipeline(ChunkingConfig()).run(engine)
    bm25 = BM25Retriever(engine)
    from app.tools import build_registry

    yield ToolEnv(engine, tainted, bm25, build_registry(), bm25)  # type: ignore[arg-type]
    engine.dispose()


def context(env: ToolEnv, role: str) -> ToolContext:
    return ToolContext(
        engine=env.engine,
        principal=principal_for_role(f"pg-{role}", role),
        settings=ToolSettings(_env_file=None),  # type: ignore[call-arg]
        retriever=env.bm25,
        clock=lambda: NOW,
    )


def fingerprint(engine: Engine) -> dict[str, int]:
    with engine.connect() as connection:
        return {
            name: connection.execute(select(func.count()).select_from(table)).scalar_one()
            for name, table in Base.metadata.tables.items()
            if name != "chunk_embeddings"
        }


def test_confidential_sources_are_not_indexed(env: ToolEnv) -> None:
    with env.engine.connect() as connection:
        levels = dict(
            connection.execute(
                select(DocumentChunk.access_level, func.count()).group_by(
                    DocumentChunk.access_level
                )
            ).all()
        )
    assert AccessLevel.CONFIDENTIAL not in levels and "confidential" not in levels
    assert {"manager", "admin", "sre", "engineering"} <= {str(k) for k in levels}


@pytest.mark.parametrize("role", ["developer", "sre", "manager", "admin"])
def test_searches_return_only_granted_pairs(env: ToolEnv, role: str) -> None:
    ctx = context(env, role)
    for query in (
        "payment service configuration reference",
        "reliability review incidents by service",
        "break-glass production access",
        "customers unexpectedly signed out",
    ):
        out = SearchDocumentsTool().execute(
            {"query": query, "top_k": 20, "include_postmortems": True}, ctx
        )
        for r in out.results:
            assert ctx.principal.may_read(SOURCE_RESOURCE[r.source_type], r.access_level), (
                role,
                r.chunk_id,
            )


def test_sql_views_are_compartments_on_postgres(env: ToolEnv) -> None:
    sql = {
        "sql": "SELECT DISTINCT access_level FROM documents "
        "WHERE doc_type NOT IN ('runbook', 'postmortem') ORDER BY access_level"
    }

    def labels(role: str) -> list[str]:
        return [row[0] for row in QueryDatabaseTool().execute(sql, context(env, role)).rows]

    assert labels("manager") == ["engineering", "manager", "public"]
    assert labels("sre") == ["engineering", "public", "sre"]
    assert labels("admin") == ["admin", "engineering", "manager", "public", "sre"]


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM incidents",
        "WITH d AS (DELETE FROM incidents RETURNING *) SELECT * FROM d",
        "SELECT pg_sleep(5)",
        "SELECT set_config('default_transaction_read_only', 'off', false)",
        "SELECT current_setting('is_superuser')",
        "SELECT * FROM pg_catalog.pg_roles",
        "SELECT * FROM information_schema.tables",
        "SELECT dblink_exec('dbname=opsrag', 'DROP TABLE incidents')",
        "COPY incidents TO PROGRAM 'id'",
        "SELECT id FROM incidents UNION SELECT email FROM users",
        'SELECT * FROM "users"',
        "SELECT lo_import('/etc/passwd')",
    ],
)
def test_injection_and_destructive_sql_are_rejected(env: ToolEnv, sql: str) -> None:
    before = fingerprint(env.engine)
    with pytest.raises(UnsafeSqlError):
        QueryDatabaseTool().execute({"sql": sql}, context(env, "admin"))
    assert fingerprint(env.engine) == before


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM incidents",
        "UPDATE incidents SET severity = 'SEV4'",
        "INSERT INTO services (id) VALUES ('x')",
        "CREATE TABLE pwned (id text)",
        "SET TRANSACTION READ WRITE",
    ],
)
def test_the_postgres_transaction_refuses_writes(env: ToolEnv, sql: str) -> None:
    """The second layer, reached only if the checker were bypassed."""
    before = fingerprint(env.engine)
    with (
        env.engine.connect() as connection,
        _read_only(connection, 5) as cursor,
        pytest.raises(Exception, match=r"read-only transaction|must be set before any query"),
    ):
        cursor.execute(sql)
    assert fingerprint(env.engine) == before


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1; DELETE FROM incidents",
        "COMMIT; DELETE FROM incidents",
        "SELECT 1; SET TRANSACTION READ WRITE; DELETE FROM incidents",
    ],
)
def test_postgres_executes_exactly_one_statement(env: ToolEnv, sql: str) -> None:
    """Also a second layer: stacked statements are refused by the server."""
    before = fingerprint(env.engine)
    with (
        env.engine.connect() as connection,
        _read_only(connection, 5) as cursor,
        pytest.raises(Exception, match="multiple commands"),
    ):
        execute_one(cursor, sql, "postgresql")
    assert fingerprint(env.engine) == before


def test_planted_injection_is_quarantined_on_postgres(env: ToolEnv) -> None:
    question = "How do I troubleshoot payment-service checkout timeouts with the PayFlux provider?"
    state = make_agent(env).run(question, principal("sre"))
    assert INJECTION_DOC in {q.source_id for q in state.security.quarantined}
    assert "IGNORE ALL PREVIOUS" not in state.final_answer
    assert "attacker.example" not in state.final_answer
