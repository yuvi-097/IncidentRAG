"""The tool layer on real PostgreSQL: read-only transactions, statement timeouts,
per-caller shadow views, and the search tools.

Opt in with OPSRAG_RUN_INTEGRATION_TESTS=1. Seeds and ingests into the configured
database; use a development database only.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from sqlalchemy import func, select
from sqlalchemy.engine import Engine

from app.config import ToolSettings, load_settings
from app.database.models import Document, Incident
from app.database.seed import seed_database
from app.database.session import create_db_engine
from app.rag.chunking import ChunkingConfig
from app.rag.ingestion import IngestionPipeline
from app.rag.ingestion.persistence import ensure_chunk_table
from app.rag.retrieval import BM25Retriever
from app.security import load_principal, principal_for_role
from app.synthetic.records import SyntheticDataset
from app.tools import ToolContext, ToolExecutionError, UnsafeSqlError, build_registry
from app.tools.sql_tool import QueryDatabaseTool, _read_only, guarded_sql

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("OPSRAG_RUN_INTEGRATION_TESTS") != "1",
        reason="set OPSRAG_RUN_INTEGRATION_TESTS=1 with PostgreSQL running",
    ),
]


@pytest.fixture(scope="module")
def engine(dataset: SyntheticDataset) -> Iterator[Engine]:
    settings = load_settings()
    if settings.app.environment == "production":
        pytest.skip("refusing to write synthetic data into production")
    engine = create_db_engine(settings.database)
    ensure_chunk_table(engine)
    seed_database(engine, dataset)
    IngestionPipeline(ChunkingConfig()).run(engine)
    yield engine
    engine.dispose()


def context(engine: Engine, role: str, **settings: object) -> ToolContext:
    return ToolContext(
        engine=engine,
        principal=principal_for_role(f"pg-{role}", role),
        settings=ToolSettings(_env_file=None, **settings),  # type: ignore[call-arg]
        retriever=BM25Retriever(engine),
    )


def test_sql_results_follow_the_grants(engine: Engine) -> None:
    tool = QueryDatabaseTool()
    sql = {"sql": "SELECT count(*) FROM documents WHERE doc_type NOT IN ('runbook', 'postmortem')"}
    manager = tool.execute(sql, context(engine, "manager")).rows[0][0]
    sre = tool.execute(sql, context(engine, "sre")).rows[0][0]
    with engine.connect() as connection:

        def documents(*levels: str) -> int:
            return connection.execute(
                select(func.count())
                .select_from(Document)
                .where(
                    Document.access_level.in_(levels),
                    Document.doc_type.not_in(["runbook", "postmortem"]),
                )
            ).scalar_one()

        assert manager == documents("public", "engineering", "manager")
        assert sre == documents("public", "engineering", "sre")
        total = connection.execute(select(func.count()).select_from(Incident)).scalar_one()
    incidents = tool.execute({"sql": "SELECT count(*) FROM incidents"}, context(engine, "manager"))
    assert incidents.rows[0][0] == total  # managers read all incidents


def test_views_use_the_postgres_schema(engine: Engine) -> None:
    sql, tables = guarded_sql(
        "SELECT count(*) FROM incidents", engine, principal_for_role("x", "sre")
    )
    assert "FROM public.incidents WHERE access_level IN ('engineering', 'public', 'sre')" in sql
    assert "users AS (SELECT NULL AS unavailable WHERE 1 = 0)" in sql
    assert "logs AS (SELECT * FROM public.logs)" in sql  # an SRE may read logs
    assert tables == ["incidents"]


def test_postgres_refuses_writes_in_the_tool_transaction(engine: Engine) -> None:
    with (
        engine.connect() as connection,
        _read_only(connection, 5) as cursor,
        pytest.raises(Exception, match="read-only transaction"),
    ):
        cursor.execute("DELETE FROM incidents")
    with engine.connect() as connection:
        assert connection.execute(select(func.count()).select_from(Incident)).scalar_one() > 0


def test_statement_timeout_cancels_slow_queries(engine: Engine) -> None:
    ctx = context(engine, "sre", sql_timeout_seconds=0.5)
    with pytest.raises(ToolExecutionError, match=r"statement timeout|canceling"):
        QueryDatabaseTool().execute({"sql": "SELECT count(*) FROM logs a, logs b"}, ctx)
    assert QueryDatabaseTool().execute({"sql": "SELECT 1"}, ctx).rows == [[1]]


def test_unsafe_and_unavailable_sql_on_postgres(engine: Engine) -> None:
    ctx = context(engine, "sre")
    for sql in (
        "SELECT * FROM users",
        "SELECT * FROM public.incidents",
        "SELECT pg_sleep(1)",
        "DELETE FROM logs",
    ):
        with pytest.raises(UnsafeSqlError):
            QueryDatabaseTool().execute({"sql": sql}, ctx)


def test_jsonb_and_timestamps_come_back_json_safe(engine: Engine) -> None:
    out = QueryDatabaseTool().execute(
        {"sql": "SELECT started_at, metrics, tags FROM incidents WHERE id = 'INC-0406'"},
        context(engine, "sre"),
    )
    started, metrics, tags = out.rows[0]
    assert started.startswith("2026-06-16") and isinstance(metrics, dict) and isinstance(tags, list)
    out.model_dump_json()


def test_registry_tools_on_postgres(engine: Engine, dataset: SyntheticDataset) -> None:
    registry = build_registry()
    user = next(u for u in dataset.users if u.role_id == "sre" and u.is_active)
    ctx = ToolContext(
        engine=engine, principal=load_principal(engine, user.id), retriever=BM25Retriever(engine)
    )
    incidents = registry.call("search_incidents", {"incident_ids": ["INC-0406"]}, ctx)
    deployments = registry.call(
        "search_deployments", {"versions": ["v2.8.1"], "services": ["payment-service"]}, ctx
    )
    logs = registry.call(
        "search_logs",
        {
            "services": ["payment-service"],
            "min_level": "ERROR",
            "since": "2026-06-16T17:00:00Z",
            "until": "2026-06-16T19:00:00Z",
        },
        ctx,
    )
    runbook = registry.call(
        "get_runbook", {"query": "payment api 500 errors", "service": "payment-service"}, ctx
    )
    docs = registry.call("search_documents", {"query": "PaymentServiceDBPoolSaturated alert"}, ctx)
    for result in (incidents, deployments, logs, runbook, docs):
        assert result.ok, (result.tool, result.error)
    assert (
        incidents.model_dump()["output"]["incidents"][0]["root_cause_deployment_id"] == "DEP-0296"
    )
    assert logs.model_dump()["output"]["total_matched"] > 0
