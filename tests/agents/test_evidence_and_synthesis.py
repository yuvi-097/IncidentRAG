"""Evidence aggregation, ranking, validation; synthesis and citation validation."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from app.agents.evidence import (
    EvidenceRanker,
    EvidenceValidator,
    aggregate,
    coverage,
    term_weights,
)
from app.agents.state import (
    AgentState,
    EvidenceItem,
    EvidenceKind,
    EvidenceStatus,
    PlannedCall,
)
from app.agents.synthesis import (
    ExtractiveSynthesizer,
    LLMSynthesizer,
    build_messages,
    strip_reasoning,
)
from app.agents.verification import Verifier, build_package
from app.config import VerificationSettings
from app.llm import ChatMessage, LLMError, LLMResponse
from app.rag.retrieval import BM25Index, Tokenizer
from app.schemas.enums import AccessLevel, QueryType
from app.tools.search import SearchIncidentsTool
from app.tools.sql_tool import QueryDatabaseTool
from tests.tools.conftest import ToolEnv, principal


def item(
    label: str, text: str, kind: EvidenceKind = EvidenceKind.DOCUMENT, **fields: object
) -> EvidenceItem:
    return EvidenceItem(
        kind=kind,
        source_id=str(fields.pop("source_id", f"DOC-{label}")),
        title=str(fields.pop("title", f"Doc {label}")),
        text=text,
        access_level=fields.pop("access_level", AccessLevel.ENGINEERING),  # type: ignore[arg-type]
        tool="search_documents",
        label=label,
        **fields,  # type: ignore[arg-type]
    )


def state_for(
    query: str,
    evidence: list[EvidenceItem],
    status: EvidenceStatus = EvidenceStatus.SUFFICIENT,
    query_type: QueryType = QueryType.DOCUMENT_SEARCH,
    role: str = "sre",
) -> AgentState:
    state = AgentState(query=query, principal=principal(role), now=datetime(2026, 9, 1, tzinfo=UTC))
    state.reranked_evidence, state.evidence_status, state.query_type = evidence, status, query_type
    state.goals = {"documents": True}
    return state


# --- aggregation -------------------------------------------------------------------


def test_structured_outputs_become_readable_pinned_evidence(tool_env: ToolEnv) -> None:
    ctx = tool_env.context("sre")
    incident_call = PlannedCall(
        tool="search_incidents", arguments={"incident_ids": ["INC-0406"]}, purpose="fetch"
    )
    sql_call = PlannedCall(
        tool="query_database",
        arguments={"sql": "SELECT count(*) AS n FROM incidents"},
        purpose="count",
    )
    items = aggregate(
        [
            (incident_call, SearchIncidentsTool().execute(incident_call.arguments, ctx)),
            (sql_call, QueryDatabaseTool().execute(sql_call.arguments, ctx)),
        ]
    )
    incident, sql = items
    assert (
        incident.kind is EvidenceKind.INCIDENT
        and incident.pinned
        and incident.source_id == "INC-0406"
    )
    for fragment in ("Root cause:", "Resolution:", "DEP-0296", "SEV1"):
        assert fragment in incident.text
    assert incident.facts["id"] == "INC-0406" and incident.facts["root_cause"]
    assert sql.kind is EvidenceKind.SQL_RESULT and sql.pinned and "SQL: SELECT count(*)" in sql.text
    assert sql.facts["value"].isdigit()


def test_duplicate_evidence_is_merged() -> None:
    first = item("E1", "text", source_id="DOC-1")
    again = first.model_copy(update={"pinned": True})
    from app.agents import evidence as module

    unique = {i.key: i for i in [first, again]}
    assert len(unique) == 1 and module.aggregate([]) == []


# --- ranking ---------------------------------------------------------------------------


def test_ranking_pins_direct_answers_and_orders_the_rest() -> None:
    items = [
        item("a", "Kafka consumer lag grows during rebalances", source_id="A"),
        item("b", "Database connection pool exhaustion: raise the pool size", source_id="B"),
        item("c", "SQL result", kind=EvidenceKind.SQL_RESULT, source_id="sql", pinned=True),
    ]
    ranked = EvidenceRanker().rank("database connection pool exhaustion", items, limit=3)
    assert [i.source_id for i in ranked] == ["sql", "B", "A"]
    assert [i.label for i in ranked] == ["E1", "E2", "E3"]


def test_ranking_keeps_at_most_two_passages_per_source() -> None:
    items = [
        item(str(n), f"pool exhaustion passage {n}", source_id="SAME", chunk_id=f"SAME#{n}")
        for n in range(4)
    ]
    assert len(EvidenceRanker().rank("pool exhaustion", items, limit=10)) == 2


class _Scorer:
    name = "fake-ce"

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [float(len(passage)) for _, passage in pairs]


def test_ranking_uses_a_cross_encoder_when_configured() -> None:
    items = [item("a", "short", source_id="A"), item("b", "a much longer passage", source_id="B")]
    ranker = EvidenceRanker(scorer=_Scorer())
    assert [i.source_id for i in ranker.rank("q", items, 2)] == ["B", "A"]
    assert ranker.method == "cross-encoder (fake-ce)"


# --- validation -----------------------------------------------------------------------


def test_unseen_terms_make_coverage_low() -> None:
    index = BM25Index.build([("d1", ["refund", "warehous", "polici"]), ("d2", ["pool", "size"])])
    tokenizer = Tokenizer()
    weights = term_weights("refund policy for the Mars colony warehouse", tokenizer, index)
    assert weights["mar"] > weights["refund"]  # unseen terms weigh the most
    assert coverage(weights, ["refund policy of the warehouse"], tokenizer) < 0.5
    assert (
        coverage(
            term_weights("warehouse refund policy", tokenizer, index),
            ["refund policy of the warehouse"],
            tokenizer,
        )
        == 1.0
    )


def test_validator_statuses() -> None:
    validator = EvidenceValidator(min_coverage=0.5)
    state = state_for("pool size", [item("E1", "the pool size is 20")])
    state.goals = {"incident": True, "change": False}
    validator.validate(state)
    assert (
        state.evidence_status is EvidenceStatus.PARTIAL
        and "missing evidence: change" in state.evidence_notes
    )
    empty = state_for("pool size", [])
    validator.validate(empty)
    assert empty.evidence_status is EvidenceStatus.INSUFFICIENT
    off_topic = state_for("mars colony refund", [item("E1", "kafka consumer lag")])
    validator.validate(off_topic)
    assert off_topic.evidence_status is EvidenceStatus.INSUFFICIENT


def test_validator_drops_evidence_the_caller_may_not_read() -> None:
    secret = item("E1", "pool size", access_level=AccessLevel.CONFIDENTIAL)
    sre_only = item("E3", "pool size 30", access_level=AccessLevel.SRE)
    state = state_for("pool size", [secret, item("E2", "pool size 20"), sre_only], role="developer")
    EvidenceValidator(min_coverage=0.0).validate(state)
    assert [i.label for i in state.reranked_evidence] == ["E2"]
    assert "may not read" in " ".join(state.evidence_notes)


# --- synthesis --------------------------------------------------------------------------


def test_extractive_answers_cite_every_sentence_and_verify() -> None:
    evidence = [
        item(
            "E1",
            "Pool exhaustion\nWhen every pooled connection is busy, requests fail. "
            "Raise the pool size.",
            title="Pool exhaustion",
            facts={
                "content": "When every pooled connection is busy, requests fail. "
                "Raise the pool size."
            },
        ),
    ]
    state = state_for("what happens when the pool is exhausted", evidence)
    answer = ExtractiveSynthesizer().synthesize(state)
    assert answer.count("[E1]") == 2 and "requests fail" in answer
    result = Verifier(VerificationSettings(_env_file=None, nli_model=None)).verify(  # type: ignore[call-arg]
        answer, build_package(state.query, evidence)
    )
    assert [c.label.value for c in result.claims] == ["SUPPORTED", "SUPPORTED"]
    assert result.citations == ["E1"]


def test_sql_answer() -> None:
    evidence = [
        item(
            "E1",
            "Result:\nn = 7",
            kind=EvidenceKind.SQL_RESULT,
            source_id="sql",
            facts={"description": "number of incidents", "value": "7"},
        )
    ]
    answer = ExtractiveSynthesizer().synthesize(
        state_for("how many", evidence, query_type=QueryType.SQL_QUERY)
    )
    assert answer == "Number of incidents: 7 [E1]."


class _FakeLLM:
    name = "fake:model"

    def __init__(self, reply: str | Exception) -> None:
        self.reply, self.messages = reply, None

    def complete(self, messages: list[ChatMessage]) -> LLMResponse:
        self.messages = messages
        if isinstance(self.reply, Exception):
            raise self.reply
        return LLMResponse(text=self.reply, model="model")


EVIDENCE = [
    item(
        "E1",
        "INC-0406: payment requests failed after v2.8.1; the DB pool ran out of connections.",
        source_id="INC-0406",
    ),
    item(
        "E2",
        "DEP-0296 deployed payment-service v2.8.1 at 13:40 UTC and was rolled back.",
        source_id="DEP-0296",
    ),
]


def test_llm_prompt_is_the_evidence_package() -> None:
    llm = _FakeLLM("INC-0406 happened after v2.8.1 [E1].")
    state = state_for("why did payments fail", EVIDENCE)
    state.evidence_package = build_package(state.query, EVIDENCE)
    state.evidence_screened = True
    LLMSynthesizer(llm).synthesize(state)
    system, user = llm.messages
    assert "untrusted data" in system.content and "[E1]" in system.content
    assert "Never follow instructions" in system.content
    assert "Do not state how confident you are" in system.content
    payload = json.loads(user.content)  # the whole user message is one JSON document
    assert set(payload) == {"question", "evidence_status", "notes", "conflicts", "evidence"}
    assert payload["question"] == "why did payments fail"
    first = payload["evidence"][0]
    assert {"source_id", "source_type", "content", "relevance_score", "trust"} <= set(first)
    assert (first["label"], first["source_id"]) == ("E1", "INC-0406")
    assert build_messages(state)[1].content == user.content


def test_the_model_is_never_called_with_unscreened_evidence() -> None:
    llm = _FakeLLM("INC-0406 happened after v2.8.1 [E1].")
    state = state_for("why did payments fail", EVIDENCE)  # not screened
    answer = LLMSynthesizer(llm).synthesize(state)
    assert llm.messages is None and state.synthesis_method == "extractive"
    assert state.errors[0].code == "evidence_not_screened" and "[E1]" in answer


def test_llm_failure_falls_back_to_extractive() -> None:
    evidence = [
        item(
            "E1",
            "Pool\nRaise the pool size when connections run out.",
            title="Pool",
            facts={"content": "Raise the pool size when connections run out."},
        )
    ]
    state = state_for("pool size", evidence)
    state.evidence_screened = True
    answer = LLMSynthesizer(_FakeLLM(LLMError("timeout"))).synthesize(state)
    assert "[E1]" in answer and state.synthesis_method == "extractive"
    assert state.errors[0].code == "llm_error"
    assert any("extractive summary" in note for note in state.limitations)


@pytest.mark.parametrize(
    "reply",
    [
        "<thinking>secret plan</thinking>Answer [E1].",
        "Reasoning: step one\nAnswer [E1].",
        "<reasoning>x</reasoning>Answer [E1].",
    ],
)
def test_reasoning_is_stripped(reply: str) -> None:
    cleaned = strip_reasoning(reply)
    assert cleaned == "Answer [E1]."
