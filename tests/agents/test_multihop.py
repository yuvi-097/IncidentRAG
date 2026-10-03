# ruff: noqa: E501  (question tables are data)
"""Phase 9: multi-hop questions (parsing, per-hop evidence, answers, end to end).

Phrasings here are written for the tests; the benchmark questions are not reused.
"""

from __future__ import annotations

import pytest

from app.agents.entities import QueryEntities
from app.agents.multihop import ChainQuery, answer_parts, chain_items, parse_chain
from app.agents.state import AgentState, EvidenceKind
from app.tools.trace import TraceChangeOutput
from tests.agents.conftest import Ask
from tests.tools.conftest import ToolEnv


def cited_sources(state: AgentState) -> set[str]:
    return {c.source_id for c in state.citations}


@pytest.mark.parametrize(
    ("question", "entities", "expected"),
    [
        ("Which file did the change behind INC-0100 modify?", {"incident_ids": ["INC-0100"]}, ("INC-0100", "cause")),
        ("Show me the diff responsible for INC-0100.", {"incident_ids": ["INC-0100"]}, ("INC-0100", "cause")),
        ("Who authored the regression behind INC-0100?", {"incident_ids": ["INC-0100"]}, ("INC-0100", "cause")),
        ("What upstream outage triggered INC-0100?", {"incident_ids": ["INC-0100"]}, ("INC-0100", "cause")),
        ("Which commit fixed INC-0100?", {"incident_ids": ["INC-0100"]}, ("INC-0100", "fix")),
        ("Which deployment remediated INC-0100?", {"incident_ids": ["INC-0100"]}, ("INC-0100", "fix")),
        ("What incidents resulted from DEP-0050?", {"deployment_ids": ["DEP-0050"]}, ("DEP-0050", "impact")),
        ("Which incidents were triggered by DEP-0050?", {"deployment_ids": ["DEP-0050"]}, ("DEP-0050", "impact")),
    ],
)  # fmt: skip
def test_chain_questions_are_recognised(
    question: str, entities: dict[str, list[str]], expected: tuple[str, str]
) -> None:
    chain = parse_chain(question, QueryEntities(**entities))
    assert chain is not None and (chain.anchor, chain.direction) == expected


@pytest.mark.parametrize(
    ("question", "entities"),
    [
        ("What caused INC-0421?", {"incident_ids": ["INC-0421"]}),  # the broad investigation
        ("Which deployment caused INC-0033?", {"incident_ids": ["INC-0033"]}),
        ("Summarise INC-0100.", {"incident_ids": ["INC-0100"]}),
        (
            "Which incidents followed DEP-0050 within a day?",
            {"deployment_ids": ["DEP-0050"]},
        ),  # temporal
    ],
)
def test_other_questions_are_not_chains(question: str, entities: dict[str, list[str]]) -> None:
    assert parse_chain(question, QueryEntities(**entities)) is None


def _trace(env: ToolEnv, **arguments: str) -> TraceChangeOutput:
    result = env.registry.call("trace_change", arguments, env.context())
    assert isinstance(result.output, TraceChangeOutput)
    return result.output


def test_every_hop_is_its_own_evidence_item(tool_env: ToolEnv) -> None:
    out = _trace(tool_env, incident_id="INC-0033")
    items = chain_items(out)
    hops = [(i.facts["hop"], i.kind) for i in items]
    assert hops[0] == ("incident", EvidenceKind.INCIDENT)
    assert ("deployment", EvidenceKind.DEPLOYMENT) in hops
    assert ("file", EvidenceKind.CODE) in hops
    assert all(i.pinned for i in items)
    code = next(i for i in items if i.facts["hop"] == "file")
    assert code.access_level == out.changes[0].files[0].access_level  # the file's own label
    assert code.facts["valid"] == (
        "true" if out.deployment and out.deployment.status == "succeeded" else "false"
    )


def test_answers_follow_the_chain_in_order(tool_env: ToolEnv) -> None:
    items = [
        i.model_copy(update={"label": f"E{n}"})
        for n, i in enumerate(chain_items(_trace(tool_env, incident_id="INC-0033")), 1)
    ]
    parts = answer_parts(items, ChainQuery(incident_id="INC-0033"))
    text = "\n".join(p for p, _ in parts)
    out = _trace(tool_env, incident_id="INC-0033")
    assert out.deployment is not None
    assert text.index(out.deployment.id) < text.index(out.changes[0].files[0].path)
    assert out.deployment.commit_sha[:7] in text and out.changes[0].author in text
    assert all(cited for _, cited in parts)  # every sentence names the records it rests on


def test_the_fix_direction_leads_with_the_remediation(tool_env: ToolEnv) -> None:
    incident = next(
        i
        for i in tool_env.dataset.incidents
        if i.remediation_deployment_id and i.root_cause_deployment_id
    )
    items = chain_items(_trace(tool_env, incident_id=incident.id))
    parts = answer_parts(items, ChainQuery(incident_id=incident.id, direction="fix"))
    assert parts[0][0].startswith(
        f"{incident.id} was fixed by {incident.remediation_deployment_id}"
    )


def test_missing_links_are_stated_not_invented(tool_env: ToolEnv) -> None:
    incident = next(
        i
        for i in tool_env.dataset.incidents
        if not i.root_cause_deployment_id
        and not i.remediation_deployment_id
        and not i.parent_incident_id
    )
    items = chain_items(_trace(tool_env, incident_id=incident.id))
    cause = "\n".join(p for p, _ in answer_parts(items, ChainQuery(incident_id=incident.id)))
    assert "No deployment is recorded as the cause" in cause
    fix = "\n".join(
        p for p, _ in answer_parts(items, ChainQuery(incident_id=incident.id, direction="fix"))
    )
    assert "No remediation deployment is recorded" in fix


# --- end to end -------------------------------------------------------------------------


def test_an_incident_is_traced_to_its_file_and_lines(ask: Ask, tool_env: ToolEnv) -> None:
    state = ask("Which file did the change behind INC-0033 modify, and how?")
    assert state.plan == "chain"
    incident = next(i for i in tool_env.dataset.incidents if i.id == "INC-0033")
    assert incident.root_cause_deployment_id in state.final_answer
    assert "payment-service.yaml" in state.final_answer and "Changed lines" in state.final_answer
    assert {incident.root_cause_deployment_id, "INC-0033"} & cited_sources(state)
    assert state.goals == {"chain": True}


def test_a_deployment_is_traced_to_the_incidents_it_caused(ask: Ask) -> None:
    state = ask("What incidents resulted from DEP-0031?")
    assert state.plan == "chain"
    assert "INC-0027" in state.final_answer and "INC-0028" in state.final_answer


def test_withheld_hops_become_limitations(ask: Ask) -> None:
    state = ask("Which commit and file caused INC-0033?", role="developer")
    assert state.plan == "chain"
    assert any("withheld" in note and "deployment" in note for note in state.limitations)
    assert "DEP-0043" not in {c.source_id for c in state.citations}


def test_a_withheld_fix_is_not_reported_as_missing(ask: Ask, tool_env: ToolEnv) -> None:
    # The developer may not read deployments: the fix exists but is withheld, and the
    # answer must not claim that none is recorded.
    incident = next(
        i
        for i in tool_env.dataset.incidents
        if i.remediation_deployment_id and i.access_level.value in {"public", "engineering"}
    )
    state = ask(f"Which commit fixed {incident.id}?", role="developer")
    assert state.plan == "chain"
    assert "No remediation deployment is recorded" not in state.final_answer
    assert "withheld" in state.final_answer
    assert incident.remediation_deployment_id not in state.final_answer
