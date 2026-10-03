"""Benchmark definition and metric arithmetic (no model needed)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from app.evaluation.comparison import compare, to_markdown
from app.evaluation.retrieval import (
    BenchmarkQuestion,
    RelevanceSelector,
    chunk_catalog,
    evaluate,
    load_benchmark,
    ndcg_at,
    resolve_relevant,
)
from app.rag.retrieval import RetrievedChunk, Retriever
from app.rag.store import ChunkFilter
from tests.retrieval.conftest import Embedded

BENCHMARK = (
    Path(__file__).resolve().parents[2] / "data" / "evaluation" / "retrieval_benchmark.jsonl"
)


def test_benchmark_has_enough_well_formed_questions() -> None:
    questions = load_benchmark(BENCHMARK)
    assert len(questions) >= 30
    assert len({q.question for q in questions}) == len(questions)
    categories = {q.category for q in questions}
    assert {"runbook", "documentation", "code", "incident", "exact-match"} <= categories
    assert sum(q.category == "exact-match" for q in questions) >= 20


def test_every_question_resolves_to_existing_records(embedded: Embedded) -> None:
    catalog = chunk_catalog(embedded.engine)
    for question in load_benchmark(BENCHMARK):
        assert resolve_relevant(question, catalog), question.id


def test_selector_semantics() -> None:
    row = {
        "document_id": "CF-1",
        "source_type": "code",
        "service_id": "payment-service",
        "version": None,
        "doc_type": "code",
        "title": "x",
        "file_path": "services/payment-service/payment_service/config.py",
        "metadata": {"category": "c"},
    }
    assert RelevanceSelector(
        source_type="code", file_path_endswith="payment_service/config.py"
    ).matches(row)
    assert not RelevanceSelector(source_type="code", service_id="cart-service").matches(row)
    assert RelevanceSelector(metadata={"category": "c"}).matches(row)
    with pytest.raises(ValueError):
        RelevanceSelector()


class _Scripted(Retriever):
    """Returns pre-set document ids in order (to check metric arithmetic)."""

    name = "scripted"

    def __init__(self, answers: dict[str, list[str]], template: RetrievedChunk) -> None:
        self.answers, self.template = answers, template

    def search(
        self, query: str, top_k: int = 10, filters: ChunkFilter | None = None
    ) -> list[RetrievedChunk]:
        return [
            self.template.model_copy(update={"document_id": doc, "rank": n})
            for n, doc in enumerate(self.answers[query][:top_k], 1)
        ]


def test_metrics_are_computed_correctly(embedded: Embedded) -> None:
    from app.rag.retrieval import DenseRetriever

    template = DenseRetriever(embedded.engine, embedded.provider).search("x", top_k=1)[0]
    questions = [
        BenchmarkQuestion(
            id="A",
            question="a",
            category="t",
            relevant=[RelevanceSelector(document_ids=["RB-0001"])],
        ),
        BenchmarkQuestion(
            id="B",
            question="b",
            category="t",
            relevant=[RelevanceSelector(document_ids=["RB-0002", "RB-0003"])],
        ),
        BenchmarkQuestion(
            id="C",
            question="c",
            category="t",
            relevant=[RelevanceSelector(document_ids=["RB-0004"])],
        ),
    ]
    answers = {
        "a": ["RB-0001", "X", "X", "X", "X"],  # hit at rank 1
        "b": ["X", "X", "RB-0003", "RB-0003", "X"],  # one of two relevant, rank 3 (duplicate chunk)
        "c": ["X"] * 10,  # miss
    }
    report = evaluate(_Scripted(answers, template), embedded.engine, questions, ks=(1, 5))
    assert report.recall[1] == pytest.approx((1 + 0 + 0) / 3, abs=1e-4)
    assert report.recall[5] == pytest.approx((1 + 0.5 + 0) / 3, abs=1e-4)
    assert report.hit[5] == pytest.approx(2 / 3, abs=1e-4)  # reports round to 4 decimals
    assert report.mrr == pytest.approx((1 + 1 / 3 + 0) / 3, abs=1e-4)
    assert [r.first_relevant_rank for r in report.results] == [1, 3, None]
    # NDCG@5: A = 1/log2(2) / 1; B = 1/log2(4) / (1/log2(2) + 1/log2(3)); the duplicate
    # RB-0003 chunk at rank 4 gains nothing; C = 0.
    ndcg_b = (1 / math.log2(4)) / (1 + 1 / math.log2(3))
    assert [r.ndcg[5] for r in report.results] == pytest.approx([1.0, ndcg_b, 0.0])
    assert report.ndcg[5] == pytest.approx((1 + ndcg_b) / 3, abs=1e-4)
    assert report.by_category["t"].questions == 3


def test_ndcg_rewards_earlier_relevant_results() -> None:
    relevant = {"R1", "R2"}
    assert ndcg_at(["R1", "R2", "X"], relevant, 3) == pytest.approx(1.0)
    assert ndcg_at(["X", "R1", "R2"], relevant, 3) < ndcg_at(["R1", "X", "R2"], relevant, 3) < 1
    assert ndcg_at(["R1", "R1", "R1"], relevant, 3) < 1  # repeats of one record count once
    assert ndcg_at(["X", "X"], relevant, 2) == 0.0
    assert ndcg_at(["R1"], {"R1", "R2", "R3"}, 1) == 1.0  # ideal is capped at k


def test_comparison_tables(embedded: Embedded) -> None:
    from app.rag.retrieval import DenseRetriever

    template = DenseRetriever(embedded.engine, embedded.provider).search("x", top_k=1)[0]
    questions = [
        BenchmarkQuestion(
            id="Q1",
            question="q1",
            category="exact-match",
            relevant=[RelevanceSelector(document_ids=["RB-0001"])],
        ),
        BenchmarkQuestion(
            id="Q2",
            question="q2",
            category="runbook",
            relevant=[RelevanceSelector(document_ids=["RB-0002"])],
        ),
    ]
    good = _Scripted({"q1": ["RB-0001"], "q2": ["RB-0002"]}, template)
    bad = _Scripted({"q1": ["X", "RB-0001"], "q2": ["X"]}, template)
    reports = {
        name: evaluate(retriever, embedded.engine, questions)
        for name, retriever in {"good": good, "bad": bad}.items()
    }
    comparison = compare(reports, {"good": "Good", "bad": "Bad"})
    assert comparison.subsets["exact-match"]["bad"].mrr == 0.5
    assert comparison.subsets["natural-language"]["good"].recall[10] == 1.0
    assert comparison.subsets["all"]["bad"].questions == 2
    assert comparison.first_relevant_rank == {
        "Q1": {"good": 1, "bad": 2},
        "Q2": {"good": 1, "bad": None},
    }
    markdown = to_markdown(comparison)
    assert "### exact-match (1 questions)" in markdown
    assert "| Good | 1.000 | 1.000 | 1.000 | 1.000 |" in markdown
    assert "| Q2 | 1 | - | q2 |" in markdown
    with pytest.raises(ValueError, match="different questions"):
        compare({"good": reports["good"], "short": evaluate(good, embedded.engine, questions[:1])})
