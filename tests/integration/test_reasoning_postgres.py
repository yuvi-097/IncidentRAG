"""Phase 9 on real PostgreSQL: temporal plans, trace_change and conflicts.

What SQLite cannot show: timezone-aware timestamps compared with date anchors and with
the clock, boundaries of [since, until) windows in SQL, and the trace tool's joins.
Opt in with OPSRAG_RUN_INTEGRATION_TESTS=1; seeds the configured development database.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from sqlalchemy.engine import Engine

from app.agents.entities import ServiceCatalog
from app.agents.graph import Agent
from app.agents.router import RuleBasedRouter
from app.agents.state import AgentState
from app.config import load_settings
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
ADMIN = principal_for_role("pg-admin", "admin")
DEVELOPER = principal_for_role("pg-developer", "developer")


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


def _utc(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=UTC)


def first_line(state: AgentState) -> str:
    assert state.errors == [], state.errors
    return state.final_answer.splitlines()[0]


def test_nearest_deployment_before_an_incident(agent: Agent) -> None:
    state = agent.run("Which rollout landed just before INC-0406?", ADMIN)
    assert state.plan == "temporal" and "DEP-0296" in first_line(state)


def test_a_date_anchor_against_timezone_aware_timestamps(
    agent: Agent, dataset: SyntheticDataset
) -> None:
    state = agent.run("What was the latest deployment of payment-service before 2026-03-01?", ADMIN)
    cutoff = datetime(2026, 3, 1, tzinfo=UTC)
    rows = [
        d
        for d in dataset.deployments
        if d.service_id == "payment-service" and _utc(d.deployed_at) < cutoff
    ]
    expected = max(rows, key=lambda d: (_utc(d.deployed_at), d.id))
    assert state.plan == "temporal" and expected.id in first_line(state)


def test_incidents_open_during_an_incident(agent: Agent, dataset: SyntheticDataset) -> None:
    anchor = next(i for i in dataset.incidents if i.id == "INC-0229")
    start, end = _utc(anchor.started_at), _utc(anchor.resolved_at)
    expected = {
        i.id
        for i in dataset.incidents
        if i.id != anchor.id and _utc(i.started_at) <= end and _utc(i.resolved_at) >= start
    }
    state = agent.run("Which other incidents were open during INC-0229?", ADMIN)
    listed = state.final_answer.splitlines()[0]
    assert expected and all(i in listed for i in expected)


def test_the_chain_and_the_fix_on_postgres(agent: Agent, dataset: SyntheticDataset) -> None:
    incident = next(i for i in dataset.incidents if i.id == "INC-0033")
    state = agent.run("Which commit and file caused INC-0033, and what changed?", ADMIN)
    assert state.plan == "chain" and state.errors == []
    assert incident.root_cause_deployment_id in state.final_answer
    assert "payment-service.yaml" in state.final_answer
    fixed = next(i for i in dataset.incidents if i.remediation_deployment_id)
    state = agent.run(f"Which commit fixed {fixed.id}?", ADMIN)
    assert f"was fixed by {fixed.remediation_deployment_id}" in first_line(state)


def test_withheld_hops_on_postgres(agent: Agent) -> None:
    state = agent.run("Which commit and file caused INC-0033?", DEVELOPER)
    assert state.plan == "chain"
    assert any("withheld" in note for note in state.limitations)
    assert "DEP-0043" not in {c.source_id for c in state.citations}


def test_conflicting_sources_on_postgres(agent: Agent) -> None:
    state = agent.run("How did the payment-service HTTP timeout change in PM-0003?", ADMIN)
    conflict = next(c for c in state.conflicts if c.key == "HTTP_TIMEOUT_SECONDS")
    assert {"0.62", "2.5"} <= {v.value for v in conflict.values}
    assert conflict.preferred == "2.5" and state.security.output_removed == []
