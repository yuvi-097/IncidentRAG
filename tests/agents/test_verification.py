"""Evidence package, claim verification, citations, confidence and no-answer behaviour.

Claims and evidence are written out here, so each label has a known right answer.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any

import pytest

from app.agents.answerability import ROOT_CAUSE_NO_ANSWER, no_answer_text, suggest_evidence
from app.agents.confidence import compute_confidence, temporal_checks
from app.agents.entities import TimeRange
from app.agents.state import (
    AgentState,
    AnswerConfidence,
    ClaimLabel,
    ClaimVerdict,
    EvidenceItem,
    EvidenceKind,
    EvidenceStatus,
    ToolCallRecord,
)
from app.agents.verification import (
    CrossEncoderNLI,
    Verifier,
    _quote,
    build_package,
    citations_for,
)
from app.config import VerificationSettings
from app.schemas.enums import AccessLevel, QueryType, SourceTrust
from tests.tools.conftest import principal

T0 = datetime(2026, 6, 16, 17, 5, tzinfo=UTC)


def settings(**overrides: Any) -> VerificationSettings:
    return VerificationSettings(_env_file=None, nli_model=None, **overrides)  # type: ignore[call-arg]


def ev(
    label: str, text: str, kind: EvidenceKind = EvidenceKind.DOCUMENT, **fields: Any
) -> EvidenceItem:
    return EvidenceItem(
        kind=kind,
        source_id=fields.pop("source_id", f"SRC-{label}"),
        title=fields.pop("title", f"Source {label}"),
        text=text,
        access_level=AccessLevel.ENGINEERING,
        tool="test",
        label=label,
        relevance=fields.pop("relevance", 0.9),
        **fields,
    )


INCIDENT = ev(
    "E1",
    "INC-0406: Payment requests returning HTTP 500 after v2.8.1 deploy. "
    "Started 2026-06-16 17:05 UTC. "
    "Root cause: the v2.8.1 release changed how database connections are pooled and pods ran out "
    "of connections under load.",
    EvidenceKind.INCIDENT,
    source_id="INC-0406",
    title="Payment requests returning HTTP 500 after v2.8.1 deploy",
    timestamp=T0,
)
DEPLOYMENT = ev(
    "E2",
    "DEP-0296: payment-service v2.8.1, status rolled_back, deployed 2026-06-16 13:40 UTC. "
    "Changes: PR-1501 Reduce idle DB connections per pod.",
    EvidenceKind.DEPLOYMENT,
    source_id="DEP-0296",
    title="payment-service v2.8.1",
    timestamp=datetime(2026, 6, 16, 13, 40, tzinfo=UTC),
    facts={"incidents": "INC-0406 (root_cause)"},
)
CONFIG = ev(
    "E3",
    "Payment Service Configuration Reference. "
    "The payment-service database pool size is 20 per pod.",
    source_id="DOC-0060",
    title="Payment Service Configuration Reference",
    section="Variables",
    location="docs/services/payment-service/configuration.md",
    relevance=0.8,
)
STALE_CONFIG = ev(
    "E4",
    "Payment Service Tuning Notes. The payment-service database pool size is 5 per pod.",
    source_id="DOC-0099",
    title="Payment Service Tuning Notes",
    relevance=0.6,
)
EVIDENCE = [INCIDENT, DEPLOYMENT, CONFIG]


def verify(answer: str, evidence: list[EvidenceItem] = EVIDENCE, **overrides: Any):
    return Verifier(settings(**overrides)).verify(answer, build_package("q", evidence))


# --- the evidence package ------------------------------------------------------------------


def test_the_evidence_package_has_the_specified_shape() -> None:
    package = build_package("Why did payments fail?", EVIDENCE)
    payload = package.model_dump(mode="json")
    assert payload["query"] == "Why did payments fail?"
    first = payload["evidence"][0]
    assert {"source_id", "source_type", "content", "relevance_score"} <= set(first)
    assert (first["source_id"], first["source_type"], first["relevance_score"]) == (
        "INC-0406",
        "incident",
        0.9,
    )
    assert all(0.0 <= e["relevance_score"] <= 1.0 for e in payload["evidence"])


# --- claim labels ----------------------------------------------------------------------------


def test_supported_claim() -> None:
    result = verify("The v2.8.1 release changed how database connections are pooled [E1].")
    (claim,) = result.claims
    assert (
        claim.label is ClaimLabel.SUPPORTED
        and claim.supporting == ["E1"]
        and claim.action == "kept"
    )
    assert result.text == "The v2.8.1 release changed how database connections are pooled [E1]."


def test_unsupported_claim_is_removed_by_default() -> None:
    result = verify("The outage was caused by a Kafka broker failure in v3.1.0 [E1].")
    (claim,) = result.claims
    assert claim.label is ClaimLabel.UNSUPPORTED and claim.action == "removed"
    assert "v3.1.0" in claim.reason
    assert result.text == "" and result.citations == []


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        (
            "hedge",
            "Not confirmed by the available evidence: "
            "The outage was caused by a Kafka broker failure.",
        ),
        ("label", "The outage was caused by a Kafka broker failure [unsupported]."),
    ],
)
def test_unsupported_claims_can_be_hedged_or_labelled_but_never_stated_as_fact(
    policy: str, expected: str
) -> None:
    result = verify(
        "The outage was caused by a Kafka broker failure [E1].", unsupported_policy=policy
    )
    assert result.text == expected
    assert result.claims[0].action == {"hedge": "hedged", "label": "labelled"}[policy]
    assert result.citations == []  # an unsupported claim never carries a citation


def test_partially_supported_claim() -> None:
    result = verify(
        "The v2.8.1 release changed connection pooling and also doubled the CPU limits [E1]."
    )
    (claim,) = result.claims
    assert claim.label is ClaimLabel.PARTIALLY_SUPPORTED
    assert claim.action == "labelled" and "(partially supported) [E1]" in result.text
    hedged = verify(
        "The v2.8.1 release changed connection pooling and also doubled the CPU limits [E1].",
        partial_policy="hedge",
    )
    assert hedged.text.startswith("The evidence only partly supports this:")


def test_a_wrong_number_is_not_supported() -> None:
    result = verify("INC-0406 started at 2026-06-16 19:45 UTC [E1].")
    assert result.claims[0].label is not ClaimLabel.SUPPORTED
    assert (
        "19:45" in result.claims[0].reason
        or result.claims[0].label is ClaimLabel.PARTIALLY_SUPPORTED
    )


def test_conflicting_sources() -> None:
    result = verify(
        "The payment-service database pool size is 20 per pod [E3].", [*EVIDENCE, STALE_CONFIG]
    )
    (claim,) = result.claims
    assert claim.label is ClaimLabel.SUPPORTED and claim.supporting == ["E3"]
    assert claim.conflicting == ["E4"]
    assert result.notes and "[E4]" in result.notes[0]


def test_no_evidence() -> None:
    result = verify("The database pool size is 20 [E1].", [])
    (claim,) = result.claims
    assert claim.label is ClaimLabel.UNSUPPORTED and result.text == "" and result.citations == []


# --- citations -----------------------------------------------------------------------------


def test_wrong_citations_are_corrected_and_missing_ones_added() -> None:
    corrected = verify("DEP-0296 deployed payment-service v2.8.1 at 13:40 UTC [E1].")
    assert (
        corrected.claims[0].supporting == ["E2"]
        and "citation corrected" in corrected.claims[0].reason
    )
    assert corrected.text.endswith("[E2].")
    added = verify("DEP-0296 deployed payment-service v2.8.1 at 13:40 UTC.")
    assert (
        added.claims[0].supporting == ["E2"]
        and added.claims[0].reason == "citation added by verification"
    )


def test_citations_to_evidence_that_does_not_exist_are_never_kept() -> None:
    result = verify("The v2.8.1 release changed how database connections are pooled [E1][E9].")
    assert "[E9]" not in result.text and result.citations == ["E1"]
    assert "E9" in result.claims[0].cited  # recorded, then dropped
    fabricated = verify("The payment pool size is 20 per pod [E7].", [INCIDENT])
    assert fabricated.citations == [] and fabricated.claims[0].label is ClaimLabel.UNSUPPORTED


def test_citation_metadata() -> None:
    package = build_package("q", EVIDENCE)
    (citation,) = citations_for(["E3", "E3", "E99"], package)
    assert citation.model_dump() == {
        "label": "E3",
        "source_id": "DOC-0060",
        "title": "Payment Service Configuration Reference",
        "source_type": EvidenceKind.DOCUMENT,
        "timestamp": None,
        "section": "Variables",
        "file_path": "docs/services/payment-service/configuration.md",
        "trust": SourceTrust.USER_CONTENT,  # not screened in this unit test
        "relevance": 0.8,
        "chunk_id": None,
    }


def test_limitations_pass_and_model_confidence_is_dropped() -> None:
    answer = (
        "The v2.8.1 release changed how database connections are pooled [E1]. "
        "I am highly confident in this answer. Confidence: HIGH. "
        "The evidence does not say who approved the release."
    )
    result = verify(answer)
    assert result.dropped_confidence_statements == 2
    assert "confident" not in result.text.lower() and "Confidence:" not in result.text
    assert "does not say who approved" in result.text
    assert len(result.claims) == 1  # the limitation is not a factual claim


class _FakeNLI:
    name = "fake-nli"

    def __init__(self, table: dict[str, dict[str, float]]) -> None:
        self.table = table

    def predict(self, pairs: Any) -> list[dict[str, float]]:
        return [self.table.get(h.rstrip("."), {"neutral": 1.0}) for _, h in pairs]


def test_nli_entailment_supports_a_paraphrase_and_contradiction_flags_it() -> None:
    paraphrase = "After the v2.8.1 rollout the pods exhausted their database connections"
    nli = _FakeNLI({paraphrase: {"entailment": 0.97, "contradiction": 0.01, "neutral": 0.02}})
    lexical = Verifier(settings()).verify(f"{paraphrase} [E1].", build_package("q", EVIDENCE))
    semantic = Verifier(settings(), nli).verify(f"{paraphrase} [E1].", build_package("q", EVIDENCE))
    assert lexical.claims[0].label is not ClaimLabel.SUPPORTED
    assert semantic.claims[0].label is ClaimLabel.SUPPORTED
    wrong = "The v2.8.1 release increased the connection pool"
    contradicting = _FakeNLI({wrong: {"entailment": 0.0, "contradiction": 0.99, "neutral": 0.01}})
    result = Verifier(settings(), contradicting).verify(
        f"{wrong} [E1].", build_package("q", EVIDENCE)
    )
    assert (
        result.claims[0].label is not ClaimLabel.SUPPORTED and "E1" in result.claims[0].conflicting
    )


# --- confidence --------------------------------------------------------------------------


def state_with(
    claims: list[ClaimVerdict], evidence: list[EvidenceItem], **fields: Any
) -> AgentState:
    state = AgentState(query="q", principal=principal("sre"), now=T0)
    state.reranked_evidence = evidence
    state.evidence_package = build_package("q", evidence)
    state.claims = claims
    state.goals = fields.pop("goals", {"incident": True, "change": True})
    state.evidence_status = fields.pop("status", EvidenceStatus.SUFFICIENT)
    for name, value in fields.items():
        setattr(state, name, value)
    return state


def claim(
    label: ClaimLabel, supporting: list[str], conflicting: list[str] | None = None
) -> ClaimVerdict:
    return ClaimVerdict(
        text="x",
        label=label,
        cited=supporting,
        supporting=supporting if label is not ClaimLabel.UNSUPPORTED else [],
        conflicting=conflicting or [],
        support_score=1.0,
        action="removed" if label is ClaimLabel.UNSUPPORTED else "kept",
    )


S, P, U = ClaimLabel.SUPPORTED, ClaimLabel.PARTIALLY_SUPPORTED, ClaimLabel.UNSUPPORTED


def test_confidence_components_and_levels() -> None:
    high = compute_confidence(
        state_with([claim(S, ["E1"]), claim(S, ["E2"])], [INCIDENT, DEPLOYMENT]), settings()
    )
    assert high.level is AnswerConfidence.HIGH
    assert (high.retrieval_quality, high.source_agreement, high.verification) == (0.9, 1.0, 1.0)
    assert high.temporal_consistency == 1.0 and "1 temporal check(s) passed" in high.reasons
    assert high.overall == pytest.approx(
        sum([0.2 * 0.9, 0.2 * 1.0, 0.2 * 1.0, 0.15 * 1.0, 0.25 * 1.0]), abs=1e-3
    )


def test_confidence_is_capped_by_conflicts_unsupported_claims_and_failures() -> None:
    conflict = compute_confidence(
        state_with([claim(S, ["E3"], ["E4"])], [CONFIG, STALE_CONFIG]), settings()
    )
    assert conflict.source_agreement == 0.3 and conflict.level is not AnswerConfidence.HIGH
    unsupported = compute_confidence(
        state_with([claim(S, ["E1"]), claim(S, ["E2"]), claim(U, [])], [INCIDENT, DEPLOYMENT]),
        settings(),
    )
    assert unsupported.verification == pytest.approx(2 / 3, abs=1e-3)
    assert unsupported.level is AnswerConfidence.MEDIUM
    failed = state_with([claim(S, ["E1"]), claim(S, ["E2"])], [INCIDENT, DEPLOYMENT])
    failed.tool_results = [
        ToolCallRecord(
            tool="search_logs", purpose="", arguments={}, status="execution_error", duration_ms=1
        )
    ]
    assert compute_confidence(failed, settings()).level is AnswerConfidence.MEDIUM


def test_temporal_inconsistency_lowers_confidence() -> None:
    late = DEPLOYMENT.model_copy(update={"timestamp": datetime(2026, 6, 16, 20, 0, tzinfo=UTC)})
    state = state_with([claim(S, ["E1"]), claim(S, ["E2"])], [INCIDENT, late])
    passed, total, problems = temporal_checks(state)
    assert (passed, total) == (0, 1) and "deployed after INC-0406" in problems[0]
    breakdown = compute_confidence(state, settings())
    assert breakdown.temporal_consistency == 0.0 and breakdown.level is AnswerConfidence.LOW


def test_evidence_outside_the_asked_window_is_inconsistent() -> None:
    from app.agents.entities import QueryEntities

    window = TimeRange(
        since=datetime(2026, 8, 1, tzinfo=UTC),
        until=datetime(2026, 9, 1, tzinfo=UTC),
        expression="last month",
    )
    state = state_with([claim(S, ["E1"])], [INCIDENT], entities=QueryEntities(time_range=window))
    assert temporal_checks(state)[:2] == (0, 1)


def test_no_supported_claim_means_insufficient_evidence() -> None:
    assert (
        compute_confidence(state_with([claim(U, [])], [INCIDENT]), settings()).level
        is AnswerConfidence.INSUFFICIENT_EVIDENCE
    )
    insufficient = state_with([claim(S, ["E1"])], [INCIDENT], status=EvidenceStatus.INSUFFICIENT)
    assert (
        compute_confidence(insufficient, settings()).level is AnswerConfidence.INSUFFICIENT_EVIDENCE
    )


# --- no-answer behaviour -------------------------------------------------------------------


def test_no_answer_text_and_suggestions() -> None:
    state = AgentState(query="Why did search-service fail?", principal=principal("sre"), now=T0)
    state.query_type = QueryType.INCIDENT_SEARCH
    state.goals = {"incident": False}
    state.unknown_terms = ["quantum"]
    state.limitations = [
        "search_logs is not permitted for role 'support_agent'; that source was not searched."
    ]
    assert no_answer_text(state) == ROOT_CAUSE_NO_ANSWER
    ideas = suggest_evidence(state)
    assert any(i.startswith("Incident records or postmortems") for i in ideas)
    assert any("quantum" in i for i in ideas) and any(
        i.startswith("Access to search_logs") for i in ideas
    )
    other = AgentState(query="What is the retention policy?", principal=principal("sre"), now=T0)
    other.query_type = QueryType.DOCUMENT_SEARCH
    assert no_answer_text(other).endswith("to answer this question.")


@pytest.mark.model
@pytest.mark.skipif(
    os.getenv("OPSRAG_RUN_MODEL_TESTS") != "1",
    reason="set OPSRAG_RUN_MODEL_TESTS=1 for the real NLI model",
)
def test_real_nli_model() -> None:
    nli = CrossEncoderNLI(VerificationSettings(_env_file=None).nli_model or "")  # type: ignore[call-arg]
    verifier = Verifier(settings(), nli)
    package = build_package("q", EVIDENCE)
    paraphrase = verifier.verify(
        "After the v2.8.1 release, payment pods exhausted their database connections [E1].", package
    )
    assert paraphrase.claims[0].label is ClaimLabel.SUPPORTED
    wrong = verifier.verify(
        "The v2.8.1 release increased the database connection pool size [E1].", package
    )
    assert wrong.claims[0].label is not ClaimLabel.SUPPORTED


class _ContradictsOthers:
    """Entails when the premise is the claim's own record; 'contradicts' everything else,
    the way small NLI models treat passages about other files or incidents."""

    name = "contradicts-others"

    def predict(self, pairs: Any) -> list[dict[str, float]]:
        return [
            {"entailment": 0.95, "contradiction": 0.02, "neutral": 0.03}
            if "DEP-0296" in premise
            else {"entailment": 0.01, "contradiction": 0.98, "neutral": 0.01}
            for premise, _ in pairs
        ]


def test_nli_contradictions_about_other_subjects_do_not_count() -> None:
    verifier = Verifier(settings(), _ContradictsOthers())
    package = build_package("q", EVIDENCE)
    deployment = verifier.verify(
        "DEP-0296 deployed payment-service v2.8.1 at 13:40 UTC [E2].", package
    )
    assert deployment.claims[0].label is ClaimLabel.SUPPORTED
    assert deployment.claims[0].conflicting == []  # INC-0406 and DOC-0060 are other subjects
    path = verifier.verify("docs/services/payment-service/configuration.md [E3].", package)
    assert path.claims[0].conflicting == []  # a file path states nothing to contradict
    assert path.notes == [] and deployment.notes == []


def test_disagreement_notes_quote_whole_words() -> None:
    claim = (
        "The deployment recorded as the cause of INC-0033 is DEP-0043: payment-service "
        "v2.6.4 (rolled_back), deployed 2025-10-09 14:37 UTC."
    )
    quoted = _quote(claim)
    assert quoted.endswith("…") and "v2.6.4" not in quoted
    assert quoted[:-1].endswith("payment-service")  # never "payment-service v2"
    assert _quote("DEP-0043 was rolled back.") == "DEP-0043 was rolled back"
