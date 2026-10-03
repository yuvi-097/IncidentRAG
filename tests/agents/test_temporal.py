# ruff: noqa: E501  (question tables are data)
"""Phase 9: temporal questions (parsing, planning, the timeline item, end to end).

Phrasings here are written for the tests; the benchmark questions are not reused.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.agents.entities import QueryEntities
from app.agents.guard import may_read_item
from app.agents.state import AgentState, EvidenceKind
from app.agents.temporal import (
    SECOND,
    Anchor,
    Relation,
    TemporalQuery,
    fixed_anchor,
    parse_temporal,
    single,
    target_call,
    timeline_item,
)
from app.schemas.enums import AccessLevel, Resource
from app.tools.deployments import SearchDeploymentsOutput
from app.tools.search import SearchIncidentsOutput
from tests.agents.conftest import Ask
from tests.tools.conftest import ToolEnv, custom_principal, principal

R = Relation
T0 = datetime(2026, 5, 1, 12, 0, tzinfo=UTC)


def cited_sources(state: AgentState) -> set[str]:
    return {c.source_id for c in state.citations}


def parse(question: str, **entities: list[str]) -> TemporalQuery | None:
    return parse_temporal(question, QueryEntities(**entities))


@pytest.mark.parametrize(
    ("question", "entities", "relation", "target", "anchor"),
    [
        ("Which rollout landed just before INC-0100?", {"incident_ids": ["INC-0100"]}, R.IMMEDIATELY_BEFORE, "deployment", "incident"),
        ("What shipped right after INC-0100 began?", {"incident_ids": ["INC-0100"]}, R.IMMEDIATELY_AFTER, "deployment", "incident"),
        ("Which deployment preceded DEP-0050?", {"deployment_ids": ["DEP-0050"]}, R.PREVIOUS, "deployment", "deployment"),
        ("What was the next incident after INC-0100?", {"incident_ids": ["INC-0100"]}, R.NEXT, "incident", "incident"),
        ("Which version of cart-service was live when INC-0100 hit?", {"incident_ids": ["INC-0100"], "services": ["cart-service"]}, R.AT_TIME_OF, "deployment", "incident"),
        ("Which incidents were ongoing at the time of DEP-0050?", {"deployment_ids": ["DEP-0050"]}, R.AT_TIME_OF, "incident", "deployment"),
        ("List deployments made during INC-0100.", {"incident_ids": ["INC-0100"]}, R.DURING, "deployment", "incident"),
        ("Which incidents opened in the 6 hours after DEP-0050?", {"deployment_ids": ["DEP-0050"]}, R.AFTER, "incident", "deployment"),
        ("What was deployed in the 3 days before INC-0100?", {"incident_ids": ["INC-0100"]}, R.BEFORE, "deployment", "incident"),
        ("What is the most recent release of order-service?", {"services": ["order-service"]}, R.LATEST, "deployment", "now"),
        ("INC-0100: what was the deployment right before it?", {"incident_ids": ["INC-0100"]}, R.IMMEDIATELY_BEFORE, "deployment", "incident"),
        ("After INC-0100, what was the first deployment of its service?", {"incident_ids": ["INC-0100"]}, R.IMMEDIATELY_AFTER, "deployment", "incident"),
        ("Which deployments of cart-service happened on 2026-03-02?", {"services": ["cart-service"]}, R.DURING, "deployment", "date"),
        ("What was the last cart-service deployment before 2026-03-02?", {"services": ["cart-service"]}, R.IMMEDIATELY_BEFORE, "deployment", "date"),
    ],
)  # fmt: skip
def test_temporal_questions_are_parsed(
    question: str, entities: dict[str, list[str]], relation: Relation, target: str, anchor: str
) -> None:
    query = parse(question, **entities)
    assert query is not None, question
    assert (query.relation, query.target, query.anchor_kind) == (relation, target, anchor)


@pytest.mark.parametrize(
    ("question", "entities"),
    [
        ("What caused INC-0421?", {"incident_ids": ["INC-0421"]}),
        ("Which deployment caused INC-0033?", {"incident_ids": ["INC-0033"]}),
        ("How do I restart payment-service?", {"services": ["payment-service"]}),
        ("Show the runbook for cart-service latency.", {"services": ["cart-service"]}),
        ("Explain INC-0100 to me.", {"incident_ids": ["INC-0100"]}),
    ],
)
def test_other_questions_are_not_temporal(question: str, entities: dict[str, list[str]]) -> None:
    assert parse(question, **entities) is None


def test_plural_nearest_becomes_a_window_and_windows_are_read() -> None:
    query = parse("Which deployments went out right before INC-0100?", incident_ids=["INC-0100"])
    assert query and query.relation is R.BEFORE and query.window == timedelta(hours=24)
    query = parse(
        "Which incidents began within 2 weeks after DEP-0050?", deployment_ids=["DEP-0050"]
    )
    assert query and query.relation is R.AFTER and query.window == timedelta(weeks=2)


def test_status_filter_scope_and_unapplied_qualifiers_are_recorded() -> None:
    query = parse(
        "Which deployment came just before INC-0100, counting only failed deployments?",
        incident_ids=["INC-0100"],
    )
    assert query and query.statuses == ["failed"] and query.unhandled == []
    query = parse(
        "Which deployment on any service went out right before INC-0100?", incident_ids=["INC-0100"]
    )
    assert query and query.all_services
    query = parse(
        "Which deployment came just before INC-0100, excluding hotfixes?", incident_ids=["INC-0100"]
    )
    assert query and query.unhandled == ["excluding"]


def _anchor(kind: str = "incident", service: str | None = "cart-service") -> Anchor:
    return Anchor(
        kind=kind,  # type: ignore[arg-type]
        id="INC-0100" if kind == "incident" else "DEP-0050",
        name="INC-0100" if kind == "incident" else "DEP-0050",
        service_id=service,
        start=T0,
        end=T0 + timedelta(hours=2),
        label="INC-0100 (cart-service)",
        resource=Resource.INCIDENTS if kind == "incident" else Resource.DEPLOYMENTS,
        access_level=AccessLevel.ENGINEERING,
    )


def _query(relation: Relation, target: str = "deployment", **extra: object) -> TemporalQuery:
    return TemporalQuery(
        relation=relation, target=target, anchor_kind="incident", anchor_id="INC-0100", **extra
    )  # type: ignore[arg-type]


def test_target_calls_use_the_anchor_timestamps() -> None:
    iso = datetime.isoformat
    before = target_call(_query(R.IMMEDIATELY_BEFORE), _anchor()).arguments
    assert before["until"] == iso(T0) and before["limit"] == 1 and before["order"] == "newest"
    assert before["services"] == ["cart-service"] and before["since"] is None
    after = target_call(_query(R.IMMEDIATELY_AFTER), _anchor()).arguments
    assert after["since"] == iso(T0 + SECOND) and after["order"] == "oldest"
    live = target_call(_query(R.AT_TIME_OF), _anchor()).arguments
    assert live["until"] == iso(T0 + SECOND) and live["statuses"] == ["succeeded", "rolled_back"]
    during = target_call(_query(R.DURING), _anchor()).arguments
    assert during["since"] == iso(T0) and during["until"] == iso(T0 + timedelta(hours=2) + SECOND)
    window = target_call(_query(R.BEFORE, window=timedelta(days=3)), _anchor()).arguments
    assert window["since"] == iso(T0 - timedelta(days=3)) and window["limit"] == 50
    overlap = target_call(_query(R.DURING, target="incident"), _anchor()).arguments
    assert overlap["overlap"] is True and overlap["services"] is None  # incidents: any service
    anywhere = target_call(_query(R.IMMEDIATELY_BEFORE, all_services=True), _anchor()).arguments
    assert anywhere["services"] is None
    only = target_call(_query(R.IMMEDIATELY_BEFORE, statuses=["succeeded"]), _anchor()).arguments
    assert only["statuses"] == ["succeeded"]


def test_incidents_open_at_a_moment_are_all_listed() -> None:
    assert single(_query(R.AT_TIME_OF)) and not single(_query(R.AT_TIME_OF, target="incident"))


def test_fixed_anchors_for_now_and_dates() -> None:
    now = datetime(2026, 9, 1, tzinfo=UTC)
    latest = TemporalQuery(relation=R.LATEST, target="deployment", anchor_kind="now")
    assert fixed_anchor(latest, now).start == now
    day = TemporalQuery(
        relation=R.DURING,
        target="deployment",
        anchor_kind="date",
        anchor_id="2026-03-02",
        window=timedelta(days=1),
    )
    anchor = fixed_anchor(day, now)
    assert anchor.start == datetime(2026, 3, 2, tzinfo=UTC) and anchor.end == datetime(
        2026, 3, 3, tzinfo=UTC
    )


def test_timeline_states_the_order_and_needs_every_grant(tool_env: ToolEnv) -> None:
    anchor = _anchor()
    incident = tool_env.registry.call(
        "search_incidents", {"incident_ids": ["INC-0406"]}, tool_env.context()
    ).output
    assert isinstance(incident, SearchIncidentsOutput)
    deployments = tool_env.registry.call(
        "search_deployments", {"deployment_ids": ["DEP-0296"]}, tool_env.context()
    ).output
    assert isinstance(deployments, SearchDeploymentsOutput)
    item = timeline_item(_query(R.IMMEDIATELY_BEFORE), anchor, deployments)
    assert item.kind is EvidenceKind.TIMELINE and item.pinned
    assert "DEP-0296" in item.text and "before INC-0100" in item.text
    assert item.facts["records"] == "DEP-0296"
    assert set(item.facts["requires"].split(",")) == {
        "deployments:engineering",
        "incidents:engineering",
    }
    assert may_read_item(principal("admin"), item)
    no_deployments = custom_principal({Resource.INCIDENTS: {AccessLevel.ENGINEERING}})
    assert not may_read_item(no_deployments, item)  # derived from records it may not read
    empty = timeline_item(
        _query(R.DURING), anchor, SearchDeploymentsOutput.model_construct(deployments=[])
    )
    assert empty.text.endswith("none found.")


# --- end to end -------------------------------------------------------------------------


def test_nearest_deployment_before_an_incident_is_chosen_by_time(ask: Ask) -> None:
    state = ask("Which rollout landed just before INC-0406?")
    assert state.plan == "temporal"
    assert "DEP-0296" in state.final_answer.splitlines()[0]
    assert {"DEP-0296", "timeline:immediately_before:INC-0406"} <= cited_sources(state)
    assert state.goals == {"anchor": True, "timeline": True}


def test_latest_is_relative_to_the_clock(ask: Ask, tool_env: ToolEnv) -> None:
    state = ask("What is the most recent release of order-service?")
    assert state.plan == "temporal"
    rows = [d for d in tool_env.dataset.deployments if d.service_id == "order-service"]
    newest = max((d for d in rows if d.deployed_at <= state.now), key=lambda d: d.deployed_at)
    assert newest.id in state.final_answer.splitlines()[0]


def test_an_unapplied_qualifier_is_reported_and_caps_confidence(ask: Ask) -> None:
    state = ask("Which deployment came just before INC-0406, excluding hotfixes?")
    assert any('"excluding"' in note for note in state.limitations)
    assert state.confidence.value in {"MEDIUM", "LOW"}


def test_without_the_deployments_grant_the_timeline_is_not_built(ask: Ask) -> None:
    # Developers may not search deployments: the temporal plan is not started, and a
    # limitation says why.
    state = ask("Which rollout landed just before INC-0406?", role="developer")
    assert state.plan == "routed"
    assert "search_deployments" not in {r.tool for r in state.tool_results}
    assert not any(i.kind is EvidenceKind.TIMELINE for i in state.reranked_evidence)
    assert any("temporal plan was not used" in note for note in state.limitations)
