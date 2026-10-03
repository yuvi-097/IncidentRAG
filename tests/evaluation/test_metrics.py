# ruff: noqa: E501  (test data)
"""Phase 10 metrics, error classification, judge plumbing and report building."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from app.evaluation.errors import ErrorClass, classify
from app.evaluation.eval_set import Category, Checks, EvalQuestion, check_answer
from app.evaluation.judge import LLMJudge
from app.evaluation.metrics import (
    claims,
    context_item,
    generation_scores,
    grounded,
    nli_faithfulness,
    retrieval_scores,
    ungrounded_tokens,
)
from app.evaluation.pipelines import MethodOutput
from app.evaluation.report import Record, ablation_table, error_table, plots, summarize
from app.llm.base import ChatMessage, LLMError, LLMResponse
from app.schemas.enums import QueryType


def question(**overrides: object) -> EvalQuestion:
    fields: dict[str, object] = {
        "id": "EV-1",
        "category": Category.DIRECT,
        "question": "What is the payment timeout?",
        "query_type": QueryType.DOCUMENT_SEARCH,
        "difficulty": "easy",
        "expected_answer": "2.5 seconds",
        "expected_sources": ["DOC-0060"],
        "supporting_sources": ["CF-0154"],
        "checks": Checks(facts=[["2.5"]]),
        "origin": "test",
    }
    fields.update(overrides)
    return EvalQuestion.model_validate(fields)


# --- retrieval ------------------------------------------------------------------------------------


def test_retrieval_scores_use_records_and_graded_relevance() -> None:
    relevance = {"A": 2, "B": 1}
    s = retrieval_scores(["X", "A", "A", "Y", "B", "Z"], relevance)
    assert s.recall[1] == 0.0 and s.recall[5] == 1.0 and s.hit[5] == 1.0
    assert s.precision_5 == 3 / 5  # chunks, not records: A counts twice
    assert s.reciprocal_rank == 0.5 and s.first_relevant_rank == 2
    dcg = 3 / math.log2(3) + 1 / math.log2(6)
    ideal = 3 / math.log2(2) + 1 / math.log2(3)
    assert s.ndcg_10 == pytest.approx(dcg / ideal)
    perfect = retrieval_scores(["A", "B"], relevance)
    assert perfect.ndcg_10 == pytest.approx(1.0) and perfect.recall[1] == 1.0
    assert retrieval_scores([], relevance).reciprocal_rank == 0.0


# --- generation --------------------------------------------------------------------------------------


CONTEXT = [
    context_item(
        "E1",
        "DOC-0060",
        "Payment config",
        "PAYMENT_SERVICE_HTTP_TIMEOUT_SECONDS is 2.5 per outbound call",
    ),
    context_item(
        "E2", "INC-0033", "Latency incident", "Regression introduced by DEP-0043 in payment-service"
    ),
]


def test_claims_and_grounding() -> None:
    parts = claims("The timeout is 2.5 [E1]. DEP-0043 caused it [E2].\nUncited line.")
    assert [c.labels for c in parts] == [["E1"], ["E2"], []]
    joined = "\n".join(c.text for c in CONTEXT)
    assert grounded("PAYMENT_SERVICE_HTTP_TIMEOUT_SECONDS is 2.5 per outbound call", joined, "")
    assert ungrounded_tokens("DEP-0999 caused it", joined, "") == ["DEP-0999"]
    assert not grounded("The payment processor was replaced by quantum hardware", joined, "")


def test_generation_scores_catch_hallucinations_and_bad_citations() -> None:
    q = question()
    good = generation_scores(
        q, "PAYMENT_SERVICE_HTTP_TIMEOUT_SECONDS is 2.5 per outbound call [E1].", CONTEXT, False
    )
    assert good.faithfulness == 1.0 and good.hallucinated is False and good.citation_correct == 1.0
    assert good.context_relevance == 0.5  # E1 is gold, E2 neither gold nor holds a fact
    bad = generation_scores(
        q, "DEP-0999 set the timeout to 9.9 [E2]. It is 2.5 [E7].", CONTEXT, False
    )
    assert bad.hallucinated is True and set(bad.hallucinated_tokens) >= {"DEP-0999", "9.9"}
    assert bad.invalid_citations == 1 and bad.citation_correct == 0.0
    declined = generation_scores(q, "I don't have sufficient evidence.", CONTEXT, True)
    assert declined.claims == 0 and declined.faithfulness is None and declined.hallucinated is None


def test_nli_faithfulness_asks_the_cited_items() -> None:
    class FakeNLI:
        name = "fake"

        def __init__(self) -> None:
            self.pairs: list[tuple[str, str]] = []

        def predict(self, pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
            self.pairs += pairs
            return [{"entailment": 0.9 if "2.5" in p and "2.5" in h else 0.1} for p, h in pairs]

    nli = FakeNLI()
    score = nli_faithfulness(nli, "The timeout is 2.5 [E1]. It was 7 [E2].", CONTEXT, False)
    assert score == 0.5
    assert all(p.startswith("DOC-0060") or p.startswith("INC-0033") for p, _ in nli.pairs)
    assert nli_faithfulness(nli, "anything", CONTEXT, True) is None


# --- errors -------------------------------------------------------------------------------------------


def output(**overrides: object) -> MethodOutput:
    fields: dict[str, object] = {
        "method": "F",
        "question_id": "EV-1",
        "answer": "",
        "ranked": [],
        "candidates": [],
        "context": [],
    }
    fields.update(overrides)
    return MethodOutput.model_validate(fields)


def classify_answer(q: EvalQuestion, out: MethodOutput) -> ErrorClass | None:
    check = check_answer(q, out.answer)
    scores = generation_scores(q, out.answer, out.context, check.abstained)
    return classify(q, out, check, scores).primary


def test_error_classes_follow_the_documented_order() -> None:
    q = question()
    assert (
        classify_answer(q, output(answer="Nothing relevant.", ranked=["X"])) is ErrorClass.RETRIEVAL
    )
    assert (
        classify_answer(
            q, output(answer="Nothing relevant.", candidates=["DOC-0060"], ranked=["X"])
        )
        is ErrorClass.RERANKING
    )
    routed = output(answer="Nothing relevant.", query_type="SQL_QUERY", context=CONTEXT)
    assert classify_answer(q, routed) is ErrorClass.ROUTING
    reasoning = output(answer="Payment config exists [E1].", context=CONTEXT, ranked=["DOC-0060"])
    assert classify_answer(q, reasoning) is ErrorClass.REASONING
    invented = output(answer="The timeout is 2.5 set by DEP-0999 [E1].", context=CONTEXT)
    assert classify_answer(q, invented) is ErrorClass.HALLUCINATION
    leak = question(category=Category.PERMISSION, checks=Checks(forbidden=["secret plan"]))
    assert classify_answer(leak, output(answer="the secret plan")) is ErrorClass.PERMISSION
    temporal = question(category=Category.TEMPORAL, checks=Checks(facts=[["DEP-0002"]]))
    # The wrong deployment is in the evidence (not invented): the ordering went wrong.
    timeline = [*CONTEXT, context_item("E3", "DEP-0001", "cart v1", "DEP-0001 deployed")]
    wrong = output(answer="DEP-0001 deployed [E3].", context=timeline, ranked=["DOC-0060"])
    assert classify_answer(temporal, wrong) is ErrorClass.TEMPORAL
    sql = question(category=Category.SQL, expected_sources=[], supporting_sources=[])
    assert classify_answer(sql, output(method="B", answer="Some passage.")) is ErrorClass.RETRIEVAL
    right = output(
        answer="PAYMENT_SERVICE_HTTP_TIMEOUT_SECONDS is 2.5 per outbound call [E1].",
        context=CONTEXT,
    )
    check = check_answer(q, right.answer)
    assert not classify(q, right, check, generation_scores(q, right.answer, CONTEXT, False)).failed


# --- judge ----------------------------------------------------------------------------------------------


class FakeLLM:
    name = "fake/judge"

    def __init__(self, reply: str | Exception) -> None:
        self.reply = reply
        self.messages: list[ChatMessage] = []

    def complete(self, messages: list[ChatMessage]) -> LLMResponse:
        self.messages = messages
        if isinstance(self.reply, Exception):
            raise self.reply
        return LLMResponse(text=self.reply, model="fake")


def test_the_judge_parses_scores_and_reports_failures() -> None:
    llm = FakeLLM('Sure: {"correctness": 1.0, "faithfulness": 0.5, "context_relevance": 0.25}')
    result = LLMJudge(llm).grade(question(), "It is 2.5 [E1].", CONTEXT)
    assert result.scores is not None and result.scores.faithfulness == 0.5
    assert llm.messages[0].role == "system" and '"answer"' in llm.messages[1].content
    assert (
        LLMJudge(FakeLLM("no json here")).grade(question(), "x", CONTEXT).error
        == "unparseable reply"
    )
    assert (
        LLMJudge(FakeLLM('{"correctness": 3}')).grade(question(), "x", CONTEXT).error
        == "invalid scores"
    )
    failed = LLMJudge(FakeLLM(LLMError("down"))).grade(question(), "x", CONTEXT)
    assert failed.scores is None and failed.error and failed.error.startswith("llm_error")


# --- report ----------------------------------------------------------------------------------------------


def record(
    method: str, correct: bool, category: str = "direct_retrieval", failed: bool = False
) -> Record:
    return Record(
        question_id=f"EV-{method}",
        category=category,
        difficulty="easy",
        origin="generated",
        role="admin",
        method=method,
        question="q",
        answer="a",
        correct=correct,
        abstained=False,
        checks={"has_positive": True},
        retrieval={
            "recall": {"1": 1.0, "5": 1.0, "10": 1.0},
            "hit": {"1": 1.0 if correct else 0.0, "5": 1.0, "10": 1.0},
            "precision_5": 0.2,
            "reciprocal_rank": 1.0,
            "ndcg_10": 1.0,
        },
        generation={
            "faithfulness": 1.0,
            "citation_correct": 1.0,
            "hallucinated": False,
            "context_relevance": 0.5,
        },
        nli_faithfulness=None,
        judge=None,
        error={
            "failed": failed,
            "primary": "retrieval_failure" if failed else None,
            "classes": ["retrieval_failure"] if failed else [],
            "detail": "",
        },
        ranked=[],
        context=[],
        query_type=None,
        plan=None,
        tools=[],
        confidence=None,
        security={},
        latency_ms=10.0,
    )


def test_summaries_tables_and_plots(tmp_path: Path) -> None:
    records = [record("B", True), record("F", False, failed=True)]
    summary = summarize(records, [question()])
    assert (
        summary["methods"]["B"]["correctness"] == 1.0
        and summary["methods"]["F"]["correctness"] == 0.0
    )
    assert summary["errors"]["F"]["primary"]["retrieval_failure"] == 1
    assert summary["methods"]["B"]["hit_1"] == 1.0 and summary["methods"]["F"]["hit_1"] == 0.0
    table = ablation_table(summary)
    assert "B. BM25" in table and "F. Full agentic system" in table and "100.0%" in table
    assert "retrieval_failure | 0 | 1" in error_table(summary)
    made = plots(summary, tmp_path)
    assert {p.name for p in made} == {
        "retrieval_ablation.png",
        "generation_ablation.png",
        "correctness_by_category.png",
        "error_classes.png",
    }
    assert all(p.stat().st_size > 1000 for p in made)
