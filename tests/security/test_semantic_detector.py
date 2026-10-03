"""The second, model-based injection detector: what it is asked about, and how it is
used. A deterministic stand-in classifier here; ``test_semantic_detector_model.py``
runs the real one."""

from __future__ import annotations

from collections.abc import Sequence

from app.agents.synthesis import LLMSynthesizer
from app.config import SecuritySettings
from app.security.injection import REMOVED, scan
from app.security.injection_model import SemanticDetector
from tests.agents.conftest import make_agent
from tests.security.conftest import REWORDED_ATTACK, REWORDED_DOC, RecordingLLM, Tainted
from tests.tools.conftest import principal

QUESTION = "How does inventory-service stock sync with StockSync work?"


class StandIn:
    """Scores 1.0 for text asking to set aside earlier instructions, 0 otherwise, and
    records what it was asked about."""

    name = "stand-in"

    def __init__(self) -> None:
        self.seen: list[str] = []

    def score(self, texts: Sequence[str]) -> list[float]:
        self.seen += texts
        return [1.0 if "set aside everything" in t else 0.0 for t in texts]


def test_only_sentences_addressed_to_an_ai_are_classified() -> None:
    classifier = StandIn()
    detector = SemanticDetector(classifier)
    text = f"Restart the pods. Check the pool size.\n{REWORDED_ATTACK}\nIf you see 500s, roll back."
    spans = detector.flagged(text)
    assert [text[a:b].strip() for a, b in spans] == [REWORDED_ATTACK]
    assert classifier.seen == [REWORDED_ATTACK, "If you see 500s, roll back."]
    stripped = detector.strip(text)
    assert REMOVED in stripped and "set aside" not in stripped
    assert "Restart the pods." in stripped and "roll back" in stripped


def test_the_rules_miss_the_reworded_attack() -> None:
    assert not scan(REWORDED_ATTACK).flagged  # why the second detector exists


def test_the_second_detector_quarantines_what_the_rules_miss(tainted: Tainted) -> None:
    llm = RecordingLLM("Stock sync runs every 5 minutes [E1].")
    without = make_agent(tainted.env).run(QUESTION, principal("sre"))
    assert REWORDED_DOC in {i.source_id for i in without.retrieved_documents}  # reached
    with_model = make_agent(
        tainted.env, synthesizer=LLMSynthesizer(llm), semantic=SemanticDetector(StandIn())
    ).run(QUESTION, principal("sre"))
    quarantined = {q.source_id: q.categories for q in with_model.security.quarantined}
    assert quarantined.get(REWORDED_DOC) == ["semantic_injection"]
    assert "set aside everything" not in llm.sent()


def test_redact_mode_keeps_the_rest_of_the_document(tainted: Tainted) -> None:
    llm = RecordingLLM("Stock sync runs every 5 minutes [E1].")
    agent = make_agent(
        tainted.env,
        synthesizer=LLMSynthesizer(llm),
        semantic=SemanticDetector(StandIn()),
        security=SecuritySettings(_env_file=None, injection_action="redact"),  # type: ignore[call-arg]
    )
    state = agent.run(QUESTION, principal("sre"))
    item = next(i for i in state.retrieved_documents if i.source_id == REWORDED_DOC)
    assert "set aside everything" not in item.text and "every 5 minutes" in item.text
    assert item.security_flags == ("semantic_injection",)
    assert "set aside everything" not in llm.sent()
