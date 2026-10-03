"""The API behind the frontend: explorer, identity, evaluation and metrics endpoints."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.agents.graph import Agent
from app.api.dependencies import get_engine
from app.api.routes.agent import get_agent
from app.config import Settings
from app.main import create_app
from app.observability.metrics import METRICS
from app.security.auth import issue_token
from tests.tools.conftest import ToolEnv


def user_of(env: ToolEnv, role: str) -> str:
    return next(u.id for u in env.dataset.users if u.role_id == role and u.is_active)


def auth(env: ToolEnv, role: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {issue_token(env.engine, user_of(env, role), 'ui tests')}"}


def client_for(app: FastAPI, agent: Agent, env: ToolEnv) -> TestClient:
    app.dependency_overrides[get_engine] = lambda: env.engine
    app.dependency_overrides[get_agent] = lambda: agent
    return TestClient(app)


@pytest.fixture
def api(app: FastAPI, agent: Agent, tool_env: ToolEnv) -> TestClient:
    return client_for(app, agent, tool_env)


def test_every_endpoint_needs_authentication(api: TestClient) -> None:
    for path in ("/api/me", "/api/services", "/api/incidents", "/api/evaluation", "/api/metrics"):
        assert api.get(path).status_code == 401, path


def test_identity_and_catalog(api: TestClient, tool_env: ToolEnv) -> None:
    me = api.get("/api/me", headers=auth(tool_env, "developer")).json()
    assert me["role"] == "developer" and "search_deployments" not in me["tools"]
    assert "sre" not in me["grants"]["incidents"]
    services = api.get("/api/services", headers=auth(tool_env, "sre")).json()
    assert {s["id"] for s in services} >= {"payment-service", "api-gateway"}
    assert all({"name", "tier", "owner_team"} <= set(s) for s in services)


def test_incidents_are_filtered(api: TestClient, tool_env: ToolEnv) -> None:
    headers = auth(tool_env, "sre")
    body = api.get(
        "/api/incidents",
        params={"severity": ["SEV1"], "service": ["payment-service"], "limit": 100},
        headers=headers,
    ).json()
    assert body["count"] > 0
    assert all(
        i["severity"] == "SEV1" and i["service_id"] == "payment-service" for i in body["incidents"]
    )
    window = api.get(
        "/api/incidents",
        params={"since": "2026-06-16", "until": "2026-06-16", "limit": 100},
        headers=headers,
    ).json()
    assert window["count"] and all(
        i["started_at"].startswith("2026-06-16") for i in window["incidents"]
    )
    browse = api.get("/api/incidents", params={"limit": 100}, headers=headers).json()
    cap = 20  # TOOLS_MAX_TOP_K default: the response says which limit was applied
    assert browse["limit"] == cap and browse["count"] == cap
    newest = api.get("/api/incidents", params={"limit": 5}, headers=headers).json()["incidents"]
    assert [i["started_at"] for i in newest] == sorted(
        (i["started_at"] for i in newest), reverse=True
    )
    text = api.get(
        "/api/incidents",
        params={"q": "database connection pool timeouts", "limit": 5},
        headers=headers,
    ).json()
    assert text["count"] == 5 and any("pool" in i["title"].lower() for i in text["incidents"])
    bad = api.get("/api/incidents", params={"severity": ["SEV9"]}, headers=headers)
    too_many = api.get("/api/incidents", params={"limit": 500}, headers=headers)
    assert bad.status_code == too_many.status_code == 422


def test_the_explorer_applies_the_callers_grants(api: TestClient, tool_env: ToolEnv) -> None:
    restricted = next(i for i in tool_env.dataset.incidents if i.access_level.value == "sre")
    developer, sre = auth(tool_env, "developer"), auth(tool_env, "sre")
    listed = api.get("/api/incidents", params={"limit": 100}, headers=developer).json()["incidents"]
    assert all(i["access_level"] != "sre" for i in listed)
    assert api.get(f"/api/incidents/{restricted.id}", headers=developer).status_code == 404
    assert api.get(f"/api/incidents/{restricted.id}", headers=sre).status_code == 200
    assert api.get(f"/api/incidents/{restricted.id}/trace", headers=developer).status_code == 404


def test_incident_detail_and_trace(api: TestClient, tool_env: ToolEnv) -> None:
    detail = api.get("/api/incidents/INC-0406", headers=auth(tool_env, "sre")).json()
    assert detail["incident"]["root_cause_deployment_id"] == "DEP-0296"
    assert detail["incident"]["postmortem_id"] == "PM-0039"
    trace = api.get("/api/incidents/INC-0033/trace", headers=auth(tool_env, "admin")).json()
    assert trace["deployment"]["id"] == "DEP-0043" and trace["changes"][0]["files"]
    withheld = api.get("/api/incidents/INC-0033/trace", headers=auth(tool_env, "developer")).json()
    assert withheld["deployment"] is None and withheld["withheld"]


def test_answers_carry_recommendations_and_full_evidence(
    api: TestClient, tool_env: ToolEnv
) -> None:
    body = api.post(
        "/api/agent/ask",
        json={"question": "Which commit and file caused INC-0033?"},
        headers=auth(tool_env, "admin"),
    ).json()
    evidence = {e["label"]: e for e in body["evidence"]}
    assert all(e["content"] and e["access_level"] for e in evidence.values())
    assert body["recommendations"]
    for r in body["recommendations"]:
        assert set(r["sources"]) <= set(evidence)  # every step points at evidence it came from
    assert any(
        r["kind"] == "code" and "payment-service.yaml" in r["text"] for r in body["recommendations"]
    )


def test_metrics_count_requests_answers_and_tools(api: TestClient, tool_env: ToolEnv) -> None:
    METRICS.reset()
    headers = auth(tool_env, "sre")
    api.post("/api/agent/ask", json={"question": "What caused INC-0406?"}, headers=headers)
    api.get("/api/incidents/INC-0406", headers=headers)
    api.get("/api/incidents/INC-9999", headers=headers)
    snapshot = api.get("/api/metrics", headers=headers).json()
    assert snapshot["agent"]["requests"] == 1 and snapshot["agent"]["latency_ms"]["p50"] is not None
    assert "tool_execution" in snapshot["agent"]["stages"]
    assert snapshot["tools"]["search_incidents"]["calls"] >= 1
    routes = snapshot["http"]["by_route"]
    assert routes.get("GET /api/incidents/{incident_id}") == 2  # one label per endpoint
    assert snapshot["http"]["by_status"].get("404") == 1
    assert snapshot["llm"]["completion"]["count"] == 0 and snapshot["configuration"]["llm"] is None


def test_evaluation_serves_the_latest_run(
    settings: Settings, agent: Agent, tool_env: ToolEnv, tmp_path: Path
) -> None:
    app = create_app(
        settings.model_copy(
            update={"app": settings.app.model_copy(update={"evaluation_dir": tmp_path})}
        )
    )
    api = client_for(app, agent, tool_env)
    headers = auth(tool_env, "manager")
    assert api.get("/api/evaluation", headers=headers).status_code == 404  # no run yet
    run = tmp_path / "run-1"
    run.mkdir()
    (tmp_path / "LATEST").write_text("run-1\n")
    summary = {"methods": {"B": {"correctness": 0.5}, "F": {"correctness": 0.75}}}
    manifest = {
        "run_id": "run-1",
        "started_at": "2026-10-01T00:00:00+00:00",
        "finished_at": "2026-10-01T01:00:00+00:00",
        "questions": 2,
        "eval_set": {"sha256": "abc"},
        "dataset": {"generator_version": "1.0.0", "seed": 42, "window": ["a", "b"], "files": {}},
        "models": {"embedding": {"model": "m", "revision": "r"}},
        "judge": "not run: no LLM configured",
        "code_sha256": "def",
        "configuration": {"database": {"password": "**********"}},
    }
    (run / "summary.json").write_text(json.dumps(summary))
    (run / "manifest.json").write_text(json.dumps(manifest))
    record = {
        "origin": "generated",
        "method": "B",
        "retrieval": {
            "recall": {"5": 1.0},
            "hit": {"5": 1.0},
            "reciprocal_rank": 0.5,
            "ndcg_10": 0.7,
        },
    }
    (run / "per_question.jsonl").write_text(json.dumps(record) + "\n")
    body = api.get("/api/evaluation", headers=headers).json()
    assert body["run"]["id"] == "run-1" and body["methods"] == {
        "B": "BM25",
        "F": "Full agentic system",
    }
    assert body["summary"]["methods"]["F"]["correctness"] == 0.75
    assert body["retrieval_by_origin"]["generated"]["B"]["mrr"] == 0.5
    assert "configuration" not in body["run"]  # nothing beyond provenance is served
