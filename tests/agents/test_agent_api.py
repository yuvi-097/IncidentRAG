"""POST /api/agent/ask: authentication, and a response with evidence but no reasoning."""

from __future__ import annotations

import logging
from datetime import timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.agents.graph import Agent
from app.api.dependencies import get_engine
from app.api.routes.agent import get_agent
from app.config import load_settings
from app.main import create_app
from app.security.auth import issue_token
from tests.tools.conftest import ToolEnv

QUESTION = {"question": "What caused INC-0406?"}


@pytest.fixture
def api(app: FastAPI, agent: Agent, tool_env: ToolEnv) -> TestClient:
    app.dependency_overrides[get_engine] = lambda: tool_env.engine
    app.dependency_overrides[get_agent] = lambda: agent
    return TestClient(app)


def user_of(env: ToolEnv, role: str) -> str:
    return next(u.id for u in env.dataset.users if u.role_id == role and u.is_active)


def bearer(env: ToolEnv, role: str, **kwargs: object) -> dict[str, str]:
    token = issue_token(env.engine, user_of(env, role), "api tests", **kwargs)  # type: ignore[arg-type]
    return {"Authorization": f"Bearer {token}"}


def test_ask_returns_a_cited_answer(api: TestClient, tool_env: ToolEnv) -> None:
    reply = api.post(
        "/api/agent/ask",
        json={"question": "Why did payment-service fail after deployment v2.8.1?"},
        headers=bearer(tool_env, "sre"),
    )
    assert reply.status_code == 200
    body = reply.json()
    assert body["query_type"] == "MULTI_SOURCE" and body["confidence"] == "HIGH"
    assert {"INC-0406", "DEP-0296"} <= {c["source_id"] for c in body["citations"]}
    assert [t["tool"] for t in body["tools"]][:2] == ["search_deployments", "search_incidents"]
    assert body["reasoning_summary"][0].startswith("query_understanding:")
    assert "total" in body["latency_ms"] and body["errors"] == []
    for key in ("prompt", "messages", "raw", "outputs", "chain_of_thought", "thoughts"):
        assert key not in body


def test_authentication_is_required(api: TestClient, tool_env: ToolEnv) -> None:
    missing = api.post("/api/agent/ask", json=QUESTION)
    assert missing.status_code == 401 and missing.headers["WWW-Authenticate"] == "Bearer"
    wrong_scheme = api.post(
        "/api/agent/ask", json=QUESTION, headers={"Authorization": "Basic YWRtaW46YWRtaW4="}
    )
    assert wrong_scheme.status_code == 401
    header_only = api.post(
        "/api/agent/ask", json=QUESTION, headers={"X-OpsRAG-User": user_of(tool_env, "admin")}
    )
    assert header_only.status_code == 401  # the header is off by default


def test_the_token_decides_the_role(api: TestClient, tool_env: ToolEnv) -> None:
    developer = api.post(
        "/api/agent/ask",
        json={"question": "How many incidents happened last month?"},
        headers=bearer(tool_env, "developer"),
    )
    assert developer.status_code == 200  # a developer may not run SQL reports
    assert any("not permitted" in note for note in developer.json()["limitations"])
    assert "query_database" not in {t["tool"] for t in developer.json()["tools"]}


def test_expired_tokens_are_refused(api: TestClient, tool_env: ToolEnv) -> None:
    expired = bearer(tool_env, "sre", ttl=timedelta(seconds=-1))
    assert api.post("/api/agent/ask", json=QUESTION, headers=expired).status_code == 401


def test_requests_are_validated(api: TestClient, tool_env: ToolEnv) -> None:
    headers = bearer(tool_env, "developer")
    assert api.post("/api/agent/ask", json={"question": ""}, headers=headers).status_code == 422
    assert (
        api.post("/api/agent/ask", json={"question": "x" * 1001}, headers=headers).status_code
        == 422
    )
    extra = {"question": "What caused INC-0406?", "role": "admin"}
    assert api.post("/api/agent/ask", json=extra, headers=headers).status_code == 422


def test_tokens_work_outside_local_environments(
    clean_env: pytest.MonkeyPatch, agent: Agent, tool_env: ToolEnv
) -> None:
    clean_env.setenv("OPSRAG_ENVIRONMENT", "staging")
    app = create_app(load_settings(env_file=None))
    app.dependency_overrides[get_engine] = lambda: tool_env.engine
    app.dependency_overrides[get_agent] = lambda: agent
    client = TestClient(app)
    ok = client.post("/api/agent/ask", json=QUESTION, headers=bearer(tool_env, "sre"))
    assert ok.status_code == 200
    header = {"X-OpsRAG-User": user_of(tool_env, "admin")}
    assert client.post("/api/agent/ask", json=QUESTION, headers=header).status_code == 401


def test_the_user_header_is_a_local_opt_in(
    clean_env: pytest.MonkeyPatch, agent: Agent, tool_env: ToolEnv
) -> None:
    clean_env.setenv("OPSRAG_ENVIRONMENT", "local")
    clean_env.setenv("SECURITY_ALLOW_USER_HEADER", "true")
    app = create_app(load_settings(env_file=None))
    app.dependency_overrides[get_engine] = lambda: tool_env.engine
    app.dependency_overrides[get_agent] = lambda: agent
    client = TestClient(app)
    header = {"X-OpsRAG-User": user_of(tool_env, "sre")}
    assert client.post("/api/agent/ask", json=QUESTION, headers=header).status_code == 200
    inactive = {"X-OpsRAG-User": "contractor.docs"}
    assert client.post("/api/agent/ask", json=QUESTION, headers=inactive).status_code == 403
    clean_env.setenv("OPSRAG_ENVIRONMENT", "staging")
    with pytest.raises(ValidationError, match="SECURITY_ALLOW_USER_HEADER"):
        load_settings(env_file=None)


def test_tokens_are_never_logged(
    api: TestClient, tool_env: ToolEnv, caplog: pytest.LogCaptureFixture
) -> None:
    headers = bearer(tool_env, "sre")
    secret = headers["Authorization"].split()[1]
    with caplog.at_level(logging.DEBUG):
        api.post("/api/agent/ask", json=QUESTION, headers=headers)
        api.post("/api/agent/ask", json=QUESTION, headers={"Authorization": f"Bearer {secret}x"})
    logged = "\n".join(r.getMessage() + " " + str(r.__dict__) for r in caplog.records)
    # opsrag_<id>_<secret>: the secret itself may contain "_", so split from the left
    assert secret not in logged and secret.split("_", 2)[2] not in logged
    assert any(r.getMessage() == "auth.failed" for r in caplog.records)
