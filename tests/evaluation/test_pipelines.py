"""Phase 10 ablation pipelines on the test corpus (BM25 stands in for the model
retrievers, so this checks the plumbing of methods B, E and F, not model quality)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.agents.synthesis import ExtractiveSynthesizer
from app.evaluation.eval_set import Category, EvalQuestion, check_answer, load_eval_set
from app.evaluation.pipelines import CONTEXT_K, Retrievers, Runner
from tests.agents.conftest import make_agent
from tests.tools.conftest import NOW, ToolEnv

EVAL_SET = Path(__file__).resolve().parents[2] / "data" / "evaluation" / "eval_set.jsonl"


@pytest.fixture(scope="module")
def questions() -> dict[str, list[EvalQuestion]]:
    by_category: dict[str, list[EvalQuestion]] = {}
    for q in load_eval_set(EVAL_SET):
        by_category.setdefault(q.category.value, []).append(q)
    return by_category


@pytest.fixture
def runner(tool_env: ToolEnv) -> Runner:
    agent = make_agent(tool_env)
    retrievers = Retrievers(
        dense=None, bm25=tool_env.bm25, hybrid=None, hybrid_rerank=tool_env.bm25
    )
    return Runner(retrievers, agent, ExtractiveSynthesizer(tool_env.bm25.index), lambda: NOW)


def test_retrieve_then_read_answers_from_the_top_chunks(
    runner: Runner, questions: dict[str, list[EvalQuestion]]
) -> None:
    q = questions[Category.SEMANTIC.value][0]
    out = runner.run("B", q)
    assert len(out.ranked) == 10 and len(out.context) == CONTEXT_K
    assert [c.label for c in out.context] == [f"E{n}" for n in range(1, CONTEXT_K + 1)]
    assert "[E1]" in out.answer and not out.declined and out.latency_ms > 0


def test_verification_can_decline_and_the_agent_records_its_plan(
    runner: Runner, questions: dict[str, list[EvalQuestion]]
) -> None:
    unanswerable = questions[Category.NO_ANSWER.value][0]  # the Mars colony question
    verified = runner.run("E", unanswerable)
    assert check_answer(unanswerable, verified.answer).abstained
    plain = runner.run("B", unanswerable)
    assert not check_answer(unanswerable, plain.answer).abstained  # A-D never decline
    temporal = questions[Category.TEMPORAL.value][0]
    full = runner.run("F", temporal)
    assert full.plan == "temporal" and full.tools and full.query_type
    assert full.context and all(item.label.startswith("E") for item in full.context)


def test_retrieval_respects_the_questions_role(
    runner: Runner, questions: dict[str, list[EvalQuestion]]
) -> None:
    denied = next(
        q
        for q in questions[Category.PERMISSION.value]
        if q.role == "developer" and q.checks.abstain
    )
    out = runner.run("B", denied)
    assert not check_answer(denied, out.answer).forbidden_found  # filtered in the query
