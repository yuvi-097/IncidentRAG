"""End-to-end: question in, validated and cited answer out.

The six required scenarios, then planning, stopping, failure handling, permissions
and the read-only guarantee. Expected facts come from the dataset, not from the agent.
"""

from __future__ import annotations

import re
from typing import Any

import pytest
from sqlalchemy import func, select

from app.agents.state import AnswerConfidence, EvidenceKind, EvidenceStatus, Stage
from app.database.models import Deployment, Incident, LogEntry
from app.schemas.enums import QueryType
from app.tools import Tool, ToolContext, ToolModel, ToolRegistry, default_tools
from app.tools.search import SearchCodeTool
from tests.agents.conftest import Ask, make_agent, response
from tests.tools.conftest import NOW, ToolEnv, principal

ALL_STAGES = [s for s in Stage if s is not Stage.DONE]


def cited_sources(state: Any) -> set[str]:
    return {c.source_id for c in state.citations}


def assert_every_line_cited(answer: str) -> None:
    for line in answer.splitlines():
        assert re.search(r"\[E\d+\]", line), line


# --- the six scenarios ---------------------------------------------------------------


def test_document_question(ask: Ask) -> None:
    state = ask("How does the API gateway validate bearer tokens and cache the JWKS?")
    assert state.query_type is QueryType.DOCUMENT_SEARCH
    assert [r.tool for r in state.tool_results] == ["search_documents"]
    assert state.evidence_status is EvidenceStatus.SUFFICIENT
    assert state.citations and {c.source_type for c in state.citations} <= {
        EvidenceKind.DOCUMENT,
        EvidenceKind.RUNBOOK,
    }
    assert "jwks" in state.final_answer.lower()
    assert_every_line_cited(state.final_answer)
    assert state.confidence in {AnswerConfidence.HIGH, AnswerConfidence.MEDIUM}


def test_incident_question(ask: Ask, tool_env: ToolEnv) -> None:
    state = ask("What caused INC-0406?")
    incident = next(i for i in tool_env.dataset.incidents if i.id == "INC-0406")
    assert state.query_type is QueryType.INCIDENT_SEARCH
    assert [r.tool for r in state.tool_results] == ["search_incidents"]  # one call suffices
    assert cited_sources(state) == {"INC-0406"}
    first_root_cause_sentence = incident.root_cause.split(":")[0]
    assert first_root_cause_sentence in state.final_answer
    assert state.confidence is AnswerConfidence.HIGH


def test_code_question(ask: Ask) -> None:
    state = ask("Where is the token bucket rate limiter implemented?")
    assert state.query_type is QueryType.CODE_SEARCH
    assert [r.tool for r in state.tool_results] == ["search_code"]
    assert "services/api-gateway/api_gateway/rate_limiter.py" in state.final_answer
    assert "TokenBucketRateLimiter" in state.final_answer
    assert all(c.source_type is EvidenceKind.CODE for c in state.citations)


def test_sql_question(ask: Ask, tool_env: ToolEnv) -> None:
    state = ask("How many payment incidents happened last month?")
    with tool_env.engine.connect() as connection:
        expected = connection.execute(
            select(func.count())
            .select_from(Incident)
            .where(
                Incident.service_id == "payment-service",
                Incident.started_at >= NOW.replace(month=8),
                Incident.started_at < NOW,
                Incident.access_level != "confidential",  # admin reads all others
            )
        ).scalar_one()
    assert state.query_type is QueryType.SQL_QUERY
    assert [r.tool for r in state.tool_results] == ["query_database"]
    assert re.search(rf": {expected} \[E1\]", state.final_answer), state.final_answer
    assert [c.source_type for c in state.citations] == [EvidenceKind.SQL_RESULT]
    sql = state.tool_results[0].arguments["sql"]
    assert sql.startswith("SELECT count(*)") and "payment-service" in sql


def test_multi_source_incident_question(ask: Ask) -> None:
    state = ask("Why did payment-service fail after deployment v2.8.1?")
    tools = [r.tool for r in state.tool_results]
    assert state.query_type is QueryType.MULTI_SOURCE
    assert tools[0] == "search_deployments"  # anchored on the named change
    assert {"search_deployments", "search_incidents", "search_logs"} <= set(tools)
    assert len(tools) < 6  # stopped before the budget
    assert state.goals == {"incident": True, "change": True, "corroboration": True}
    assert {"INC-0406", "DEP-0296"} <= cited_sources(state)
    assert {c.source_type for c in state.citations} >= {
        EvidenceKind.INCIDENT,
        EvidenceKind.DEPLOYMENT,
        EvidenceKind.LOGS,
    }
    assert "QueuePool limit" in state.final_answer  # the logged error, quoted from the logs
    assert_every_line_cited(state.final_answer)
    assert state.confidence is AnswerConfidence.HIGH


def test_no_answer_question(ask: Ask) -> None:
    state = ask("What is the refund policy for the Mars colony warehouse?")
    assert state.evidence_status is EvidenceStatus.INSUFFICIENT
    assert state.final_answer.startswith("I don't have sufficient evidence")
    assert state.citations == [] and state.confidence is AnswerConfidence.INSUFFICIENT_EVIDENCE
    assert any("Mars" in note and "colony" in note for note in state.limitations)


def test_out_of_scope_question_calls_no_tools(ask: Ask) -> None:
    state = ask("What's the weather in Paris?")
    assert state.query_type is QueryType.UNKNOWN and state.tool_results == []
    assert "outside what OpsRAG can answer" in state.final_answer
    assert state.confidence is AnswerConfidence.INSUFFICIENT_EVIDENCE


# --- planning and stopping -----------------------------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        "What caused INC-0406?",
        "Where is the token bucket rate limiter implemented?",
        "What changed in payment-service v2.8.1?",
        "How many deployments were rolled back this year?",
    ],
)
def test_simple_questions_use_one_tool(ask: Ask, question: str) -> None:
    assert len(ask(question).tool_results) == 1


def test_later_calls_use_earlier_results(ask: Ask, tool_env: ToolEnv) -> None:
    """The log search is scoped by the incident found earlier, not by the question."""
    state = ask("Why did payment-service fail after deployment v2.8.1?")
    incident = next(i for i in tool_env.dataset.incidents if i.id == "INC-0406")
    logs = next(r for r in state.tool_results if r.tool == "search_logs")
    assert logs.arguments["services"] == [incident.service_id]
    assert logs.arguments["until"].startswith(incident.resolved_at.strftime("%Y-%m-%dT%H:%M"))
    incidents = next(r for r in state.tool_results if r.tool == "search_incidents")
    assert "INC-0406" in incidents.arguments["incident_ids"]  # linked to DEP-0296


def test_incident_anchored_causal_question_fetches_its_deployment(ask: Ask) -> None:
    state = ask("What changed before INC-0406 started, and why did it fail?")
    tools = [(r.tool, r.arguments) for r in state.tool_results]
    assert tools[0] == ("search_incidents", {"incident_ids": ["INC-0406"]})
    assert ("search_deployments", {"deployment_ids": ["DEP-0296"]}) in tools
    assert "DEP-0296" in cited_sources(state)


def test_deployments_right_before_an_incident_are_selected_by_time(ask: Ask) -> None:
    # Phase 9: a question about order in time is answered from timestamps: the incident
    # (the anchor) first, then the deployments of its service before its start.
    state = ask("What deployments happened right before INC-0406 started?")
    assert state.plan == "temporal"
    tools = [(r.tool, r.arguments) for r in state.tool_results]
    assert tools[0][1]["incident_ids"] == ["INC-0406"]
    follow = tools[1][1]
    assert tools[1][0] == "search_deployments"
    assert follow["services"] == ["payment-service"] and follow["until"].startswith("2026-06-16")
    assert "DEP-0296" in cited_sources(state)


def test_budget_limits_tool_calls(tool_env: ToolEnv) -> None:
    from app.config import AgentSettings

    agent = make_agent(tool_env, settings=AgentSettings(_env_file=None, max_tool_calls=1))  # type: ignore[call-arg]
    state = agent.run("Why did payment-service fail after deployment v2.8.1?", principal("sre"))
    assert len(state.tool_results) == 1
    assert state.evidence_status is not EvidenceStatus.SUFFICIENT
    assert any("budget" in note for note in state.limitations)


# --- failures and permissions --------------------------------------------------------


class _Broken(SearchCodeTool):
    def _run(self, arguments: Any, context: ToolContext) -> Any:
        raise RuntimeError("index offline")


def test_a_failing_tool_is_reported_not_papered_over(tool_env: ToolEnv) -> None:
    tools: list[Tool[Any, Any]] = [t for t in default_tools() if t.name != "search_code"]
    agent = make_agent(tool_env, registry=ToolRegistry([*tools, _Broken()]))
    state = agent.run("Where is the token bucket rate limiter implemented?", principal("admin"))
    (record,) = state.tool_results
    assert record.status == "execution_error" and "RuntimeError" in (record.error or "")
    assert "index offline" not in (record.error or "")  # internals are not exposed
    assert any(e.tool == "search_code" and e.stage is Stage.EXECUTE for e in state.errors)
    assert any("search_code failed" in note for note in state.limitations)
    assert state.final_answer.startswith("I don't have sufficient evidence")
    assert state.citations == [] and state.confidence is AnswerConfidence.INSUFFICIENT_EVIDENCE


class _NoLogs(Tool[ToolModel, ToolModel]):
    name = "search_logs"
    description = "log store that is down"
    permission = default_tools()[4].permission
    input_model = default_tools()[4].input_model
    output_model = default_tools()[4].output_model

    def _run(self, arguments: ToolModel, context: ToolContext) -> ToolModel:
        raise RuntimeError("log store unavailable")


def test_the_agent_continues_after_a_failure_and_says_so(tool_env: ToolEnv) -> None:
    """Logs fail; the root-cause code change still corroborates the incident."""
    tools = [t for t in default_tools() if t.name != "search_logs"]
    state = make_agent(tool_env, registry=ToolRegistry([*tools, _NoLogs()])).run(
        "Why did payment-service fail after deployment v2.8.1?", principal("admin")
    )
    calls = [(r.tool, r.status) for r in state.tool_results]
    assert ("search_logs", "execution_error") in calls
    assert calls[-1] == ("search_code", "ok")  # the next planned source was still used
    assert state.goals == {"incident": True, "change": True, "corroboration": True}
    assert {"INC-0406", "DEP-0296"} <= cited_sources(state)
    assert EvidenceKind.LOGS not in {c.source_type for c in state.citations}  # nothing invented
    assert any("search_logs failed" in note for note in state.limitations)
    assert "1 tool call(s) failed" in state.confidence_reasons
    assert state.confidence is AnswerConfidence.MEDIUM  # a source was missing


def test_permissions_shape_the_plan(ask: Ask) -> None:
    state = ask("Why did payment-service fail after deployment v2.8.1?", role="developer")
    tools = {r.tool for r in state.tool_results}
    assert not tools & {"search_logs", "search_deployments", "query_database"}
    assert any("search_logs" in note and "not permitted" in note for note in state.limitations)
    developer = ask("How many incidents happened last month?", role="developer")
    assert "query_database" not in {r.tool for r in developer.tool_results}
    assert any("query_database" in note for note in developer.limitations)


def test_labels_limit_the_evidence(ask: Ask) -> None:
    """Auth-service runbooks are SRE-labelled: an SRE's answer uses them, a developer's
    evidence never contains them (they are filtered in the tools' queries)."""
    question = "Why were customers unexpectedly signed out by auth-service?"
    developer, sre = ask(question, role="developer"), ask(question, role="sre")
    assert developer.retrieved_documents and developer.citations
    for item in developer.retrieved_documents:
        assert item.access_level.value in {"public", "engineering"}
    assert "sre" in {i.access_level.value for i in sre.reranked_evidence}
    assert not {i.source_id for i in sre.reranked_evidence if i.access_level.value == "sre"} & {
        i.source_id for i in developer.retrieved_documents
    }


def test_the_agent_never_writes(ask: Ask, tool_env: ToolEnv) -> None:
    def counts() -> tuple[int, int, int]:
        with tool_env.engine.connect() as connection:
            return tuple(  # type: ignore[return-value]
                connection.execute(select(func.count()).select_from(t)).scalar_one()
                for t in (Incident, Deployment, LogEntry)
            )

    before = counts()
    for question in (
        "Delete all incidents for payment-service",
        "DROP TABLE incidents; how many incidents are there?",
        "Why did payment-service fail after deployment v2.8.1?",
    ):
        state = ask(question)
        assert {r.tool for r in state.tool_results} <= set(build_names())
    assert counts() == before


def build_names() -> list[str]:
    return [t.name for t in default_tools()]


# --- the state and the response --------------------------------------------------------


def test_state_carries_every_required_field(ask: Ask) -> None:
    state = ask("Why did payment-service fail after deployment v2.8.1?")
    assert state.query and state.user_id == "test-admin" and state.role == "admin"
    assert state.query_type is QueryType.MULTI_SOURCE
    assert state.selected_tools and state.tool_results
    assert state.retrieved_documents and state.reranked_evidence
    assert state.evidence_status and state.final_answer and state.citations
    assert state.confidence is not AnswerConfidence.INSUFFICIENT_EVIDENCE
    assert set(state.latency_ms) == {s.value for s in ALL_STAGES} | {"total"}
    assert state.errors == []
    assert [s.stage for s in state.steps][:3] == [Stage.UNDERSTAND, Stage.ROUTE, Stage.SELECT_TOOLS]
    assert state.steps[-1].stage is Stage.RESPOND


def test_response_exposes_summaries_not_reasoning(ask: Ask) -> None:
    body = response(ask("Why did payment-service fail after deployment v2.8.1?")).model_dump(
        mode="json"
    )
    assert set(body) == {
        "question",
        "answer",
        "query_type",
        "confidence",
        "confidence_reasons",
        "evidence_status",
        "citations",
        "evidence",
        "tools",
        "reasoning_summary",
        "limitations",
        "errors",
        "synthesis",
        "latency_ms",
        "claims",
        "confidence_breakdown",
        "suggested_evidence",
        "security",
        "conflicts",
        "recommendations",
        "plan",
    }
    assert all(len(line) <= 330 for line in body["reasoning_summary"])
    text = repr(body).lower()
    for leaked in ("system prompt", "you are opsrag", "<think", "chain of thought"):
        assert leaked not in text
    assert all(e["snippet"] and len(e["snippet"]) <= 320 for e in body["evidence"])


def test_roles_without_code_access_get_the_documentation(ask: Ask) -> None:
    """SREs and managers may not read code; a code question falls back to the documents
    they may read, and says so."""
    question = "What does the Payment Service Configuration Reference say about DB_POOL_SIZE?"
    sre, manager = ask(question, role="sre"), ask(question, role="manager")
    for state in (sre, manager):
        assert state.query_type is QueryType.CODE_SEARCH
        assert [r.tool for r in state.tool_results] == ["search_documents"]
        assert any("source code was not searched" in n for n in state.limitations)
        assert all(c.source_type is not EvidenceKind.CODE for c in state.citations)
    assert "DOC-0060" in {i.source_id for i in sre.retrieved_documents}  # SRE-labelled
    assert "DOC-0060" not in {i.source_id for i in manager.retrieved_documents}


def test_roles_without_deployments_anchor_on_the_incident(ask: Ask) -> None:
    """Developers and managers may not read deployments or logs, but may read the incident:
    the multi-source plan starts from the failure instead, and says what is missing."""
    for role in ("developer", "manager"):
        state = ask("Why did payment-service fail after deployment v2.8.1?", role=role)
        assert state.tool_results[0].tool == "search_incidents", role
        assert "INC-0406" in cited_sources(state), role
        assert state.goals["incident"] and not state.goals["change"]
        assert state.confidence is AnswerConfidence.MEDIUM  # a source was out of reach
        assert any("search_deployments is not permitted" in n for n in state.limitations)


def test_operational_reports_reach_managers_only(ask: Ask) -> None:
    question = "What is in the Reliability Review 2026-Q2 report?"
    manager = ask(question, role="manager")
    assert manager.query_type is QueryType.DOCUMENT_SEARCH
    assert "RPT-0004" in cited_sources(manager)
    for role in ("developer", "sre"):
        state = ask(question, role=role)
        assert not any(i.source_id.startswith("RPT-") for i in state.retrieved_documents), role
