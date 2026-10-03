"""Security test 2: unauthorized incidents.

Developers read non-sensitive incidents only; SREs, managers and admins read all of
them (the policy in app/security/policy.json). Sensitive (SRE-labelled) incidents
are filtered in every query that can return them: by id, by text, by filters, through
postmortems, SQL and the agent. A hidden incident looks exactly like a missing one.
"""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI

from app.agents.answerability import ROOT_CAUSE_NO_ANSWER
from app.schemas.enums import AccessLevel, IncidentCategory
from app.tools.search import SearchDocumentsTool, SearchIncidentsTool
from app.tools.sql_tool import QueryDatabaseTool
from tests.agents.conftest import make_agent
from tests.security.conftest import api_client, ask_api, user_of
from tests.tools.conftest import ToolEnv, principal

SENSITIVE_ROLES = ("sre", "manager", "admin")


def sensitive_incident(env: ToolEnv) -> str:
    """An SRE-labelled incident with a postmortem (auth-service sign-outs)."""
    return next(
        i.id
        for i in env.dataset.incidents
        if i.access_level is AccessLevel.SRE and i.postmortem_id and i.service_id == "auth-service"
    )


def test_by_id_hidden_looks_like_missing(tool_env: ToolEnv) -> None:
    incident = sensitive_incident(tool_env)
    out = SearchIncidentsTool().execute(
        {"incident_ids": [incident, "INC-9999"]}, tool_env.context("developer")
    )
    assert out.incidents == [] and out.not_found == [incident, "INC-9999"]
    for role in SENSITIVE_ROLES:
        found = SearchIncidentsTool().execute({"incident_ids": [incident]}, tool_env.context(role))
        assert [i.id for i in found.incidents] == [incident], role


def test_text_search_and_filters_never_return_sensitive_incidents(tool_env: ToolEnv) -> None:
    context = tool_env.context("developer")
    by_text = SearchIncidentsTool().execute(
        {"query": "customers unexpectedly signed out auth-service tokens", "top_k": 20}, context
    )
    by_category = SearchIncidentsTool().execute(
        {"categories": [IncidentCategory.AUTHENTICATION_FAILURE.value], "top_k": 50}, context
    )
    by_service = SearchIncidentsTool().execute({"services": ["auth-service"], "top_k": 50}, context)
    for out in (by_text, by_category, by_service):
        assert all(i.access_level is AccessLevel.ENGINEERING for i in out.incidents)
        assert all(e.access_level is not AccessLevel.SRE for e in out.evidence)
    # every authentication failure is sensitive, so a developer sees none of them
    assert by_category.incidents == []
    sre = SearchIncidentsTool().execute(
        {"categories": [IncidentCategory.AUTHENTICATION_FAILURE.value], "top_k": 50},
        tool_env.context("sre"),
    )
    assert sre.incidents and all(i.access_level is AccessLevel.SRE for i in sre.incidents)


def test_postmortems_of_sensitive_incidents_are_hidden(tool_env: ToolEnv) -> None:
    incident = next(i for i in tool_env.dataset.incidents if i.id == sensitive_incident(tool_env))
    postmortem = next(d for d in tool_env.dataset.documents if d.id == incident.postmortem_id)
    assert postmortem.access_level is AccessLevel.SRE

    def found(role: str) -> set[str]:
        out = SearchDocumentsTool().execute(
            {"query": postmortem.title, "top_k": 20, "include_postmortems": True},
            tool_env.context(role, retriever=tool_env.bm25),
        )
        return {r.document_id for r in out.results}

    assert postmortem.id not in found("developer")
    assert postmortem.id in found("manager")  # managers read all incidents and reports on them


def test_sql_counts_follow_the_incident_grants(tool_env: ToolEnv) -> None:
    sql = {"sql": "SELECT access_level, count(*) FROM incidents GROUP BY access_level"}
    for role in SENSITIVE_ROLES:
        levels = {row[0] for row in QueryDatabaseTool().execute(sql, tool_env.context(role)).rows}
        assert "sre" in levels, role
    with pytest.raises(Exception, match="sql:read"):
        QueryDatabaseTool().execute(sql, tool_env.context("developer"))


def test_the_agent_does_not_answer_about_hidden_incidents(tool_env: ToolEnv) -> None:
    incident = sensitive_incident(tool_env)
    agent = make_agent(tool_env)
    developer = agent.run(f"What caused {incident}?", principal("developer"))
    assert incident not in {i.source_id for i in developer.retrieved_documents}
    assert developer.citations == []
    assert developer.final_answer.startswith(ROOT_CAUSE_NO_ANSWER)
    assert any(f"{incident} was not found or is not accessible" in n for n in developer.limitations)
    source = next(i for i in tool_env.dataset.incidents if i.id == incident)
    assert source.root_cause not in developer.final_answer
    manager = agent.run(f"What caused {incident}?", principal("manager"))
    assert incident in {c.source_id for c in manager.citations}  # positive control


def test_the_api_hides_sensitive_incidents(app: FastAPI, tool_env: ToolEnv) -> None:
    incident = sensitive_incident(tool_env)
    source = next(i for i in tool_env.dataset.incidents if i.id == incident)
    client = api_client(app, tool_env)
    body = ask_api(client, user_of(tool_env, "developer"), f"What caused {incident}?")
    text = json.dumps(body)
    assert source.root_cause[:60] not in text and source.symptoms[:60] not in text
    assert body["confidence"] == "INSUFFICIENT_EVIDENCE" and body["citations"] == []
