"""Frontend tests: every page runs headless (Streamlit's AppTest) against a fake client
that replays JSON captured from the real API, so the tests also check that the UI and
the API agree on the response shapes."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.agents.graph import Agent
from app.api.dependencies import get_engine
from app.api.routes.agent import get_agent
from app.observability.metrics import METRICS
from app.security.auth import issue_token
from tests.agents.conftest import agent  # noqa: F401  (shared fixtures)
from tests.tools.conftest import ToolEnv, tool_env  # noqa: F401

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
if str(FRONTEND) not in sys.path:
    sys.path.insert(0, str(FRONTEND))  # the pages import opsrag_ui, as under `streamlit run`


class FakeClient:
    """Replays captured API responses and records what the pages asked for."""

    def __init__(self, captured: dict[str, Any]) -> None:
        self.captured = captured
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def _get(self, name: str, **params: Any) -> Any:
        self.calls.append((name, params))
        return self.captured[name]

    def health(self) -> dict[str, Any]:
        return self._get("health")

    def me(self) -> dict[str, Any]:
        return self._get("me")

    def ask(self, question: str) -> dict[str, Any]:
        return self._get("ask", question=question)

    def services(self) -> list[dict[str, Any]]:
        return self._get("services")

    def incidents(self, **params: Any) -> dict[str, Any]:
        return self._get("incidents", **params)

    def incident(self, incident_id: str) -> dict[str, Any]:
        return self._get("incident", incident_id=incident_id)

    def trace(self, incident_id: str) -> dict[str, Any]:
        return self._get("trace", incident_id=incident_id)

    def evaluation(self) -> dict[str, Any]:
        return self._get("evaluation")

    def metrics(self) -> dict[str, Any]:
        return self._get("metrics")


@pytest.fixture(scope="module")
def captured(agent: Agent, tool_env: ToolEnv) -> dict[str, Any]:  # noqa: F811
    from app.config import load_settings
    from app.main import create_app

    settings = load_settings(env_file=None)
    app: FastAPI = create_app(settings)
    app.dependency_overrides[get_engine] = lambda: tool_env.engine
    app.dependency_overrides[get_agent] = lambda: agent
    user = next(u.id for u in tool_env.dataset.users if u.role_id == "admin" and u.is_active)
    headers = {"Authorization": f"Bearer {issue_token(tool_env.engine, user, 'ui tests')}"}
    api = TestClient(app)
    METRICS.reset()
    out: dict[str, Any] = {
        "health": {"status": "ok", "version": "test", "checks": {"database": {"status": "ok"}}},
        "me": api.get("/api/me", headers=headers).json(),
        "services": api.get("/api/services", headers=headers).json(),
        "incidents": api.get("/api/incidents", params={"limit": 20}, headers=headers).json(),
        "incident": api.get("/api/incidents/INC-0033", headers=headers).json(),
        "trace": api.get("/api/incidents/INC-0033/trace", headers=headers).json(),
        "ask": api.post(
            "/api/agent/ask",
            json={"question": "Which commit and file caused INC-0033?"},
            headers=headers,
        ).json(),
    }
    out["metrics"] = api.get("/api/metrics", headers=headers).json()
    evaluation = api.get("/api/evaluation", headers=headers)
    out["evaluation"] = evaluation.json() if evaluation.status_code == 200 else None
    return out


@pytest.fixture
def fake(captured: dict[str, Any]) -> FakeClient:
    return FakeClient(captured)
