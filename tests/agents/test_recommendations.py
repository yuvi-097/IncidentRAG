"""Recommended next steps: each built from cited evidence, never invented."""

from __future__ import annotations

from app.agents.recommendations import LIMIT, recommend
from app.agents.state import AgentState
from tests.agents.conftest import Ask


def labels_in(state: AgentState) -> set[str]:
    return {i.label for i in state.reranked_evidence if i.label}


def test_a_traced_change_recommends_reviewing_it(ask: Ask) -> None:
    state = ask("Which commit and file caused INC-0033?")
    steps = state.recommendations
    assert steps and len(steps) <= LIMIT
    kinds = {s.kind for s in steps}
    assert {"change", "code"} <= kinds
    change = next(s for s in steps if s.kind == "change")
    assert "DEP-0043" in change.text and "INC-0033" in change.text
    for step in steps:
        assert step.sources and set(step.sources) <= labels_in(state)


def test_a_fix_question_recommends_the_runbook(ask: Ask) -> None:
    state = ask("How do we fix payment-service errors after deployment v2.8.1?", "sre")
    runbooks = [s for s in state.recommendations if s.kind == "runbook"]
    assert runbooks and runbooks[0].text.startswith("Follow runbook RB-")


def test_disagreeing_sources_become_a_step(ask: Ask) -> None:
    state = ask("How did the payment-service HTTP timeout change in PM-0003?")
    conflict = [s for s in state.recommendations if s.kind == "conflict"]
    assert conflict and "HTTP_TIMEOUT_SECONDS" in conflict[0].text and conflict[0].sources


def test_without_an_answer_the_steps_are_the_missing_evidence(ask: Ask) -> None:
    state = ask("What caused the outage of the Mars colony warehouse?")
    assert state.recommendations
    assert all(s.kind == "gather" and not s.sources for s in state.recommendations)
    assert [s.text for s in state.recommendations] == state.suggested_evidence[:LIMIT]


def test_steps_only_use_what_the_answer_cites(ask: Ask) -> None:
    state = ask("What caused INC-0406?")
    cited = {c.label for c in state.citations}
    assert recommend(state) == state.recommendations
    for step in state.recommendations:
        assert set(step.sources) <= cited
