"""Every endpoint of the API, checked the same way (Phase 14 audit).

The table below lists every operation; a test fails if the API gains one that is not in
it, so a new endpoint cannot ship without being exercised here.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.agents.graph import Agent
from app.api.dependencies import get_database_probe
from app.schemas.health import ComponentHealth, ComponentStatus
from tests.api.test_dashboard_api import auth, client_for
from tests.tools.conftest import ToolEnv

# (method, path template, concrete path, JSON body, protected, expected status when allowed)
ENDPOINTS: list[tuple[str, str, str, dict[str, Any] | None, bool, int]] = [
    ("GET", "/api/health", "/api/health", None, False, 200),
    ("GET", "/api/ready", "/api/ready", None, False, 200),
    (
        "POST",
        "/api/agent/ask",
        "/api/agent/ask",
        {"question": "What caused INC-0406?"},
        True,
        200,
    ),
    ("GET", "/api/services", "/api/services", None, True, 200),
    ("GET", "/api/incidents", "/api/incidents", None, True, 200),
    ("GET", "/api/incidents/{incident_id}", "/api/incidents/INC-0406", None, True, 200),
    (
        "GET",
        "/api/incidents/{incident_id}/trace",
        "/api/incidents/INC-0033/trace",
        None,
        True,
        200,
    ),
    ("GET", "/api/me", "/api/me", None, True, 200),
    ("GET", "/api/evaluation", "/api/evaluation", None, True, 200),
    ("GET", "/api/metrics", "/api/metrics", None, True, 200),
]
PROTECTED = [e for e in ENDPOINTS if e[4]]


def _ok() -> ComponentHealth:
    return ComponentHealth(status=ComponentStatus.OK, latency_ms=0.1)


@pytest.fixture
def api(app: FastAPI, agent: Agent, tool_env: ToolEnv) -> TestClient:
    client = client_for(app, agent, tool_env)
    app.dependency_overrides[get_database_probe] = lambda: _ok
    return client


def call(api: TestClient, method: str, path: str, body: Any, headers: dict[str, str]) -> Any:
    return api.request(method, path, json=body, headers=headers)


def test_the_table_covers_every_operation_of_the_api(api: TestClient) -> None:
    documented = {
        (method.upper(), path)
        for path, operations in api.get("/api/openapi.json").json()["paths"].items()
        for method in operations
    }
    assert documented == {(m, template) for m, template, *_ in ENDPOINTS}


@pytest.mark.parametrize(("method", "template", "path", "body", "_", "status"), PROTECTED)
def test_protected_endpoints_refuse_anonymous_and_forged_callers(
    api: TestClient, method: str, template: str, path: str, body: Any, _: bool, status: int
) -> None:
    anonymous = call(api, method, path, body, {})
    forged_token = "opsrag_" + "bad_" + "tokenvalue"  # built at run time (secret scanner)
    forged = call(api, method, path, body, {"Authorization": f"Bearer {forged_token}"})
    header = call(api, method, path, body, {"X-OpsRAG-User": "noor.hassan"})  # header off
    for response in (anonymous, forged, header):
        assert response.status_code == 401, (template, response.status_code)
        assert response.headers["WWW-Authenticate"] == "Bearer"
        assert "noor.hassan" not in response.text


@pytest.mark.parametrize(("method", "template", "path", "body", "protected", "status"), ENDPOINTS)
def test_every_endpoint_answers_an_allowed_caller(
    api: TestClient,
    tool_env: ToolEnv,
    method: str,
    template: str,
    path: str,
    body: Any,
    protected: bool,
    status: int,
) -> None:
    headers = auth(tool_env, "admin") if protected else {}
    response = call(api, method, path, body, headers)
    assert response.status_code == status, (template, response.text[:300])
    assert response.headers["content-type"].startswith("application/json")
    assert len(response.headers["X-Request-ID"]) >= 16  # correlates with the logs


def test_docs_and_schema_are_served(api: TestClient) -> None:
    assert api.get("/api/docs").status_code == 200
    schema = api.get("/api/openapi.json").json()
    assert schema["info"]["title"] == "OpsRAG"
    # Every protected operation declares its security scheme in the schema.
    for _method, template, *_rest, protected, _status in ENDPOINTS:
        operation = schema["paths"][template][_method.lower()]
        assert bool(operation.get("security")) == protected, template


def test_wrong_methods_and_malformed_bodies_are_refused(api: TestClient, tool_env: ToolEnv) -> None:
    headers = auth(tool_env, "admin")
    assert api.get("/api/agent/ask", headers=headers).status_code == 405
    assert api.post("/api/health").status_code == 405
    assert api.delete("/api/incidents/INC-0406", headers=headers).status_code == 405
    for body in ({}, {"question": ""}, {"question": "x" * 100_000}, {"question": 42}, []):
        response = api.post("/api/agent/ask", json=body, headers=headers)
        assert response.status_code == 422, body
    assert api.get("/api/incidents/NOT-AN-ID", headers=headers).status_code in (404, 422)
    assert api.get("/api/incidents", params={"limit": 0}, headers=headers).status_code == 422
