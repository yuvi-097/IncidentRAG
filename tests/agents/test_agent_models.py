"""The agent with the real configured models: embedder, hybrid retrieval with the
cross-encoder, and cross-encoder evidence reranking (extractive synthesis).

Opt in with OPSRAG_RUN_MODEL_TESTS=1. Embeds the whole test corpus with the configured
model first (several minutes on CPU). Assertions are about behaviour (routes, tools,
what gets cited), not exact wording.
"""

from __future__ import annotations

import os

import pytest

from app.agents.graph import Agent
from app.agents.state import AnswerConfidence, EvidenceKind, EvidenceStatus
from app.config import load_settings
from app.rag.embeddings import EmbeddingPipeline, build_embedding_provider
from app.schemas.enums import QueryType
from app.services.agent import build_agent
from tests.tools.conftest import NOW, ToolEnv, principal

pytestmark = [
    pytest.mark.model,
    pytest.mark.skipif(
        os.getenv("OPSRAG_RUN_MODEL_TESTS") != "1",
        reason="set OPSRAG_RUN_MODEL_TESTS=1 to run tests with the real models",
    ),
]


@pytest.fixture(scope="module")
def real_agent(tool_env: ToolEnv) -> Agent:
    settings = load_settings(env_file=None)
    provider = build_embedding_provider(settings.embedding)
    EmbeddingPipeline(provider, settings.embedding.batch_size).run(tool_env.engine)
    return build_agent(tool_env.engine, settings, clock=lambda: NOW)


def test_document_question(real_agent: Agent) -> None:
    state = real_agent.run("How does the API gateway validate JWT tokens?", principal("sre"))
    assert state.query_type is QueryType.DOCUMENT_SEARCH and len(state.tool_results) == 1
    assert state.citations and "jwks" in state.final_answer.lower()


def test_incident_question(real_agent: Agent) -> None:
    state = real_agent.run("What caused INC-0406?", principal("sre"))
    assert {c.source_id for c in state.citations} == {"INC-0406"}
    assert state.confidence is AnswerConfidence.HIGH


def test_code_question(real_agent: Agent) -> None:
    state = real_agent.run(
        "Where is the token bucket rate limiter implemented?", principal("developer")
    )
    assert "rate_limiter.py" in state.final_answer


def test_sql_question(real_agent: Agent) -> None:
    state = real_agent.run("How many payment incidents happened last month?", principal("sre"))
    assert [c.source_type for c in state.citations] == [EvidenceKind.SQL_RESULT]


def test_multi_source_question(real_agent: Agent) -> None:
    question = "Why did payment-service fail after deployment v2.8.1?"
    state = real_agent.run(question, principal("sre"))
    assert {"INC-0406", "DEP-0296"} <= {c.source_id for c in state.citations}
    assert len(state.tool_results) < 6


def test_no_answer_question(real_agent: Agent) -> None:
    question = "What is the refund policy for the Mars colony warehouse?"
    state = real_agent.run(question, principal("sre"))
    assert state.evidence_status is EvidenceStatus.INSUFFICIENT and state.citations == []
