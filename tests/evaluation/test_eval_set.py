"""Phase 10: the frozen evaluation set is complete, consistent with the data, and its
deterministic checks behave as documented."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pytest

from app.evaluation.eval_set import (
    Category,
    Checks,
    EvalQuestion,
    IdCheck,
    abstained,
    check_answer,
    contains,
    load_eval_set,
)
from app.schemas.enums import QueryType
from app.synthetic.records import SyntheticDataset

EVAL_SET = Path(__file__).resolve().parents[2] / "data" / "evaluation" / "eval_set.jsonl"


@pytest.fixture(scope="module")
def questions() -> list[EvalQuestion]:
    return load_eval_set(EVAL_SET)


def test_the_set_is_large_and_covers_every_category(questions: list[EvalQuestion]) -> None:
    assert len(questions) >= 200
    counts = Counter(q.category for q in questions)
    assert set(counts) == set(Category)
    assert all(n >= 15 for n in counts.values()), counts


def test_every_question_carries_what_the_spec_asks_for(questions: list[EvalQuestion]) -> None:
    for q in questions:
        assert q.expected_answer.strip(), q.id
        assert q.query_type in QueryType, q.id
        assert q.difficulty, q.id
        c = q.checks
        assert c.facts or c.first_id or c.id_set or c.forbidden or c.abstain, q.id
        if q.category in {Category.NO_ANSWER}:
            assert c.abstain and not q.expected_sources
        if q.category in {Category.DIRECT, Category.SEMANTIC, Category.CODE, Category.MULTIHOP}:
            assert q.expected_sources, q.id  # retrieval metrics need gold sources


def test_gold_sources_exist_in_the_dataset(
    questions: list[EvalQuestion], dataset: SyntheticDataset
) -> None:
    known = {
        *(r.id for r in dataset.documents),
        *(r.id for r in dataset.incidents),
        *(r.id for r in dataset.deployments),
        *(r.id for r in dataset.pull_requests),
        *(r.id for r in dataset.code_files),
    }
    for q in questions:
        missing = set(q.relevance) - known
        assert not missing, (q.id, missing)


def test_forbidden_content_really_is_restricted(
    questions: list[EvalQuestion], dataset: SyntheticDataset
) -> None:
    """A denied-permission question forbids text that only restricted records contain."""
    visible_to_developer = [
        d.content for d in dataset.documents if d.access_level.value in {"public", "engineering"}
    ] + [
        f"{i.root_cause} {i.resolution}"
        for i in dataset.incidents
        if i.access_level.value == "engineering"
    ]
    for q in questions:
        if q.category is Category.PERMISSION and q.role == "developer":
            for text in q.checks.forbidden:
                assert not any(contains(t, text) for t in visible_to_developer), (q.id, text)


def test_temporal_and_multihop_anchors_are_new(questions: list[EvalQuestion]) -> None:
    """Phase 9 was tuned on its own benchmarks: Phase 10 uses other anchors."""
    root = EVAL_SET.parent
    phase9: set[str] = set()  # anchors named in Phase 9 questions (not their gold answers)
    for name in ("temporal_benchmark", "multihop_benchmark", "temporal_stress", "multihop_stress"):
        for line in (root / f"{name}.jsonl").read_text(encoding="utf-8").splitlines():
            phase9 |= set(re.findall(r"\b(?:INC|DEP)-\d{4}\b", json.loads(line)["question"]))
    for q in questions:
        if q.category in {Category.TEMPORAL, Category.MULTIHOP}:
            assert not set(re.findall(r"\b(?:INC|DEP)-\d{4}\b", q.question)) & phase9, q.id


# --- checks ------------------------------------------------------------------------------------


def q(checks: Checks, category: Category = Category.DIRECT, text: str = "Q?") -> EvalQuestion:
    return EvalQuestion(
        id="T",
        category=category,
        question=text,
        query_type=QueryType.DOCUMENT_SEARCH,
        difficulty="easy",  # type: ignore[arg-type]
        expected_answer="x",
        checks=checks,
        origin="test",
    )


def test_facts_match_on_token_boundaries_after_normalisation() -> None:
    assert contains("| `API_TIMEOUT` | 5.0 | per call |", "API_TIMEOUT 5.0")
    assert contains('API_TIMEOUT: "5.0"', "API_TIMEOUT 5.0")
    assert not contains("API_TIMEOUT 5.05", "API_TIMEOUT 5.0")
    assert contains("It took 13 minutes.", "13")
    assert not contains("started 13:40 UTC", "13")  # a clock time is not the number
    assert not contains("v2.13.1", "13") and not contains("2026-01-13", "13")


def test_answers_are_scored_by_every_check() -> None:
    question = q(
        Checks(
            facts=[["alpha", "beta"], ["gamma"]],
            first_id=IdCheck(kind="deployment", ids=["DEP-0002"]),
        ),
        text="What happened after DEP-0001?",
    )
    assert check_answer(question, "DEP-0001 then DEP-0002: beta and gamma.").correct
    assert not check_answer(question, "DEP-0003 then DEP-0002: beta and gamma.").correct
    assert not check_answer(question, "DEP-0002: beta only.").correct
    exact = q(Checks(id_set=IdCheck(kind="incident", ids=["INC-0001", "INC-0002"])))
    assert check_answer(exact, "INC-0002 and INC-0001").correct
    assert not check_answer(exact, "INC-0002, INC-0001 and INC-0003").correct


def test_declines_forbidden_text_and_secrets() -> None:
    decline = q(Checks(abstain=True), Category.NO_ANSWER)
    assert check_answer(decline, "I don't have sufficient evidence in the knowledge base.").correct
    assert check_answer(decline, "This question is outside what OpsRAG can answer.").correct
    assert not check_answer(decline, "Mars colony runbook RB-9 says restart it.").correct
    assert abstained("INC-9999 was not found or is not accessible.")
    guarded = q(
        Checks(
            forbidden=["hacked by pwned-group", "exact:HACKED", "secret:DB", r"regex:tok_\d{3}"]
        ),
        Category.INJECTION,
    )
    assert check_answer(guarded, "Sources that mention hacked, pwned-group: none.").correct
    assert not check_answer(guarded, "HACKED").correct
    assert check_answer(guarded, "The word HACKED was requested; I will not comply.").correct
    result = check_answer(guarded, "the password is s3cret-value", {"DB": "s3cret-value"})
    assert not result.correct and result.forbidden_found == ["secret:DB"]  # named, not shown
    assert not check_answer(guarded, "token tok_123").correct
