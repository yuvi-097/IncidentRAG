"""Phase 7 through the whole agent: package, verified claims, citations, confidence."""

from __future__ import annotations

from app.agents.answerability import ROOT_CAUSE_NO_ANSWER
from app.agents.state import AnswerConfidence, ClaimLabel, EvidenceStatus, Stage
from app.agents.synthesis import LLMSynthesizer
from app.llm import ChatMessage, LLMResponse
from tests.agents.conftest import Ask, make_agent
from tests.tools.conftest import ToolEnv, principal


def test_a_verified_causal_answer(ask: Ask) -> None:
    state = ask("Why did payment-service fail after deployment v2.8.1?")
    package = state.evidence_package
    assert package is not None and package.query == state.query
    assert all(0 <= e.relevance_score <= 1 for e in package.evidence)
    assert state.claims and all(c.label is ClaimLabel.SUPPORTED for c in state.claims)
    labels = {e.label for e in package.evidence}
    assert all(set(c.supporting) <= labels for c in state.claims)  # no invented citations
    for citation in state.citations:
        entry = package.by_label()[citation.label]
        assert (citation.source_id, citation.source_type, citation.relevance) == (
            entry.source_id,
            entry.source_type,
            entry.relevance_score,
        )
    breakdown = state.confidence_breakdown
    assert breakdown is not None and breakdown.level is AnswerConfidence.HIGH
    assert breakdown.temporal_consistency == 1.0  # DEP-0296 was deployed before INC-0406
    assert breakdown.source_agreement == 1.0 and breakdown.verification == 1.0
    stages = [s.stage for s in state.steps]
    assert stages.index(Stage.PACKAGE) < stages.index(Stage.SYNTHESIZE) < stages.index(Stage.VERIFY)
    assert stages.index(Stage.VERIFY) < stages.index(Stage.CONFIDENCE)


def test_insufficient_evidence_gets_the_no_answer_text_and_suggestions(ask: Ask) -> None:
    state = ask("Why did search-service fail after deployment v9.9.9?")
    assert state.evidence_status is EvidenceStatus.INSUFFICIENT
    assert state.final_answer.startswith(ROOT_CAUSE_NO_ANSWER)
    assert "Additional evidence that would help:" in state.final_answer
    assert state.suggested_evidence and all(
        f"- {s}" in state.final_answer for s in state.suggested_evidence
    )
    assert any("Deployment history for search-service" in s for s in state.suggested_evidence)
    assert state.confidence is AnswerConfidence.INSUFFICIENT_EVIDENCE
    assert state.citations == [] and state.claims == []


class _ScriptedLLM:
    name = "scripted"

    def __init__(self, text: str) -> None:
        self.text = text

    def complete(self, messages: list[ChatMessage]) -> LLMResponse:
        return LLMResponse(text=self.text, model="scripted")


def test_llm_claims_are_verified_before_anyone_sees_them(tool_env: ToolEnv) -> None:
    reply = (
        "INC-0406 started on 2026-06-16 at 17:05 UTC after the v2.8.1 release [E1]. "
        "The outage was caused by a Kafka broker failure in v3.1.0 [E1]. "
        "DEP-0296 was deployed at 13:40 UTC [E9]. "
        "Confidence: HIGH, I am absolutely certain."
    )
    agent = make_agent(tool_env)
    agent.synthesizer = LLMSynthesizer(_ScriptedLLM(reply), agent.synthesizer)  # type: ignore[arg-type]
    state = agent.run("Why did payment-service fail after deployment v2.8.1?", principal("sre"))
    answer = state.final_answer
    assert "Kafka" not in answer and "v3.1.0" not in answer  # unsupported claim removed
    assert "[E9]" not in answer and "certain" not in answer and "Confidence" not in answer
    assert "INC-0406 started on 2026-06-16" in answer
    labels = {c.text[:25]: c.label for c in state.claims}
    assert labels["The outage was caused by "] is ClaimLabel.UNSUPPORTED
    dep = next(c for c in state.claims if c.text.startswith("DEP-0296"))
    assert dep.label is ClaimLabel.SUPPORTED and "E9" in dep.cited and "E9" not in dep.supporting
    assert any(e.code == "unknown_citation" for e in state.errors)
    assert state.confidence is AnswerConfidence.MEDIUM  # capped: one claim was unsupported
    assert any("removed" in note for note in state.limitations)
    assert any("Confidence statements" in note for note in state.limitations)


def test_every_label_in_the_answer_is_a_listed_citation(tool_env: ToolEnv) -> None:
    """A "Sources disagree ... [E4] state(s) otherwise" note cites the disagreeing source;
    it must be in the citations too, or a client cannot resolve it (Phase 14 audit)."""
    import re

    from app.agents.state import AgentState
    from app.agents.verification import build_package
    from tests.agents.test_verification import EVIDENCE, STALE_CONFIG
    from tests.tools.conftest import NOW

    agent = make_agent(tool_env)
    state = AgentState(
        query="What is the payment-service database pool size?",
        principal=principal("admin"),
        now=NOW,
    )
    state.evidence_package = build_package(state.query, [*EVIDENCE, STALE_CONFIG])
    state.final_answer = "The payment-service database pool size is 20 per pod [E3]."
    state.synthesis_method = "extractive"
    agent.verify(state)
    in_answer = set(re.findall(r"\[(E\d+)\]", state.final_answer))
    assert {"E3", "E4"} <= in_answer  # the claim's support and the note's disagreeing source
    assert in_answer <= {c.label for c in state.citations}
    assert state.citations[0].label == "E3"  # the support comes first
