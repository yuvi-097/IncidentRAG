"""The security stages fail closed: if screening breaks, no evidence goes further and
no model is called; if output validation breaks, the answer is withheld."""

from __future__ import annotations

import pytest

from app.agents.guard import WITHHELD, EvidenceScreen, OutputGuard
from app.agents.state import AnswerConfidence
from app.agents.synthesis import LLMSynthesizer
from tests.agents.conftest import make_agent
from tests.security.conftest import RecordingLLM
from tests.tools.conftest import ToolEnv, principal

QUESTION = "What caused INC-0406?"


def _explode(*_: object, **__: object) -> None:
    raise RuntimeError("screening crashed")


def test_a_failing_screen_drops_all_evidence(
    tool_env: ToolEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(EvidenceScreen, "screen", _explode)
    llm = RecordingLLM("INC-0406 was caused by v2.8.1 [E1].")
    state = make_agent(tool_env, synthesizer=LLMSynthesizer(llm)).run(QUESTION, principal("sre"))
    assert state.tool_results  # the tools ran...
    assert state.retrieved_documents == [] and state.reranked_evidence == []  # ...nothing passed
    assert llm.calls == []  # the model was never called
    assert any(e.code == "screening_failed" for e in state.errors)
    assert state.confidence is AnswerConfidence.INSUFFICIENT_EVIDENCE


def test_a_failing_output_guard_withholds_the_answer(
    tool_env: ToolEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(OutputGuard, "check", _explode)
    state = make_agent(tool_env).run(QUESTION, principal("sre"))
    assert state.final_answer == WITHHELD
    assert "INC-0406 (SEV1" not in state.final_answer
    assert any(e.code == "output_validation_failed" for e in state.errors)
    assert state.citations == []
