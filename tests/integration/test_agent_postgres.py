"""The agent end to end on real PostgreSQL (BM25 retrieval, no models).

Checks what SQLite cannot: PostgreSQL SQL from the templates (to_char, explicit UTC
timestamps), timezone-aware windows for the log search, and the read-only guarantee.
Opt in with OPSRAG_RUN_INTEGRATION_TESTS=1; seeds the configured development database.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from sqlalchemy import func, select
from sqlalchemy.engine import Engine

from app.agents.entities import ServiceCatalog
from app.agents.graph import Agent
from app.agents.router import RuleBasedRouter
from app.agents.state import AnswerConfidence, EvidenceKind, EvidenceStatus
from app.config import load_settings
from app.database.models import Incident, LogEntry
from app.database.seed import seed_database
from app.database.session import create_db_engine
from app.rag.chunking import ChunkingConfig
from app.rag.ingestion import IngestionPipeline
from app.rag.ingestion.persistence import ensure_chunk_table
from app.rag.retrieval import BM25Retriever
from app.security import principal_for_role
from app.synthetic.records import SyntheticDataset
from app.tools import build_registry

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("OPSRAG_RUN_INTEGRATION_TESTS") != "1",
        reason="set OPSRAG_RUN_INTEGRATION_TESTS=1 with PostgreSQL running",
    ),
]
NOW = datetime(2026, 9, 1, tzinfo=UTC)
SRE = principal_for_role("pg-sre", "sre")


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


@pytest.fixture(scope="module")
def agent(engine: Engine) -> Agent:
    bm25 = BM25Retriever(engine)
    return Agent(
        engine=engine,
        registry=build_registry(),
        router=RuleBasedRouter(ServiceCatalog.from_engine(engine), clock=lambda: NOW),
        retriever=bm25,
        term_stats=bm25.index,
        clock=lambda: NOW,
    )


def test_sql_question_on_postgres(agent: Agent, engine: Engine) -> None:
    state = agent.run("How many incidents per month in 2026?", SRE)
    assert state.evidence_status is EvidenceStatus.SUFFICIENT, state.errors
    sql = state.tool_results[0].arguments["sql"]
    assert "to_char(started_at, 'YYYY-MM')" in sql and "'2026-01-01 00:00:00+00'" in sql
    with engine.connect() as connection:
        june = connection.execute(
            select(func.count())
            .select_from(Incident)
            .where(
                Incident.started_at >= datetime(2026, 6, 1, tzinfo=UTC),
                Incident.started_at < datetime(2026, 7, 1, tzinfo=UTC),
            )
        ).scalar_one()
    assert f"month=2026-06, n={june}" in state.final_answer


def test_multi_source_question_on_postgres(agent: Agent) -> None:
    state = agent.run("Why did payment-service fail after deployment v2.8.1?", SRE)
    assert state.confidence is AnswerConfidence.HIGH, (state.errors, state.limitations)
    kinds = {c.source_type for c in state.citations}
    assert {EvidenceKind.INCIDENT, EvidenceKind.DEPLOYMENT, EvidenceKind.LOGS} <= kinds
    logs = next(r for r in state.tool_results if r.tool == "search_logs")
    assert logs.results > 0  # the timezone-aware incident window found the error lines


def test_the_agent_leaves_postgres_unchanged(agent: Agent, engine: Engine) -> None:
    def counts() -> tuple[int, int]:
        with engine.connect() as connection:
            return (
                connection.execute(select(func.count()).select_from(Incident)).scalar_one(),
                connection.execute(select(func.count()).select_from(LogEntry)).scalar_one(),
            )

    before = counts()
    questions = (
        "Delete all incidents",
        "What caused INC-0406?",
        "How many deployments were rolled back this year?",
    )
    for question in questions:
        agent.run(question, SRE)
    assert counts() == before
