from __future__ import annotations

import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import __version__
from app.api.dependencies import get_database_probe
from app.config import Settings
from app.schemas.health import ComponentHealth, ComponentStatus


def _ok_probe() -> ComponentHealth:
    return ComponentHealth(status=ComponentStatus.OK, latency_ms=0.5)


def _raising_probe() -> ComponentHealth:
    raise RuntimeError("probe exploded")


def test_health_ok_when_database_reachable(app: FastAPI, client: TestClient) -> None:
    app.dependency_overrides[get_database_probe] = lambda: _ok_probe

    response = client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "opsrag-api"
    assert body["version"] == __version__
    assert body["environment"] == "test"
    assert body["checks"]["database"] == {"status": "ok", "latency_ms": 0.5, "detail": None}
    assert body["timestamp"]


def test_health_degraded_when_database_unreachable(client: TestClient, settings: Settings) -> None:
    # No override: the real engine attempts a real connection to a closed port.
    response = client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    database = body["checks"]["database"]
    assert database["status"] == "down"
    assert database["detail"] == "OperationalError"
    assert database["latency_ms"] is not None
    # Clients get the error class only, never connection details or secrets.
    assert settings.database.password is not None
    assert settings.database.password.get_secret_value() not in response.text
    assert settings.database.host not in response.text


def test_health_degraded_when_probe_raises(app: FastAPI, client: TestClient) -> None:
    app.dependency_overrides[get_database_probe] = lambda: _raising_probe

    response = client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["checks"]["database"]["detail"] == "RuntimeError"
    assert "probe exploded" not in response.text


def test_generates_request_id_when_absent(app: FastAPI, client: TestClient) -> None:
    app.dependency_overrides[get_database_probe] = lambda: _ok_probe

    response = client.get("/api/health")

    request_id = response.headers["X-Request-ID"]
    assert len(request_id) == 32  # uuid4 hex


def test_propagates_valid_client_request_id(app: FastAPI, client: TestClient) -> None:
    app.dependency_overrides[get_database_probe] = lambda: _ok_probe

    response = client.get("/api/health", headers={"X-Request-ID": "incident-42.trace_01"})

    assert response.headers["X-Request-ID"] == "incident-42.trace_01"


@pytest.mark.parametrize("unsafe", ["has spaces", "x" * 65, 'quote"s', "semi;colon"])
def test_replaces_unsafe_client_request_id(app: FastAPI, client: TestClient, unsafe: str) -> None:
    app.dependency_overrides[get_database_probe] = lambda: _ok_probe

    response = client.get("/api/health", headers={"X-Request-ID": unsafe})

    assert response.headers["X-Request-ID"] != unsafe
    assert len(response.headers["X-Request-ID"]) == 32


def test_emits_structured_access_log(
    app: FastAPI, client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    app.dependency_overrides[get_database_probe] = lambda: _ok_probe
    caplog.set_level(logging.INFO, logger="app.access")

    client.get("/api/health?token=should-not-be-logged")

    records = [r for r in caplog.records if r.getMessage() == "request.completed"]
    assert len(records) == 1
    record = records[0]
    assert record.method == "GET"
    assert record.path == "/api/health"
    assert record.status_code == 200
    assert record.total_latency_ms >= 0
    # The test client's own httpx logger records the full URL; the app's must not.
    assert "should-not-be-logged" not in str(vars(record))


def test_unknown_route_returns_404(client: TestClient) -> None:
    assert client.get("/api/does-not-exist").status_code == 404


def test_openapi_documents_health_endpoint(client: TestClient) -> None:
    schema = client.get("/api/openapi.json").json()
    assert "/api/health" in schema["paths"]


def test_ready_when_the_database_answers(app: FastAPI, client: TestClient) -> None:
    app.dependency_overrides[get_database_probe] = lambda: _ok_probe
    response = client.get("/api/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ok" and set(response.json()["checks"]) == {"database"}


def test_not_ready_without_the_database(client: TestClient) -> None:
    # Unlike /api/health (200 while the process is up), readiness fails the request.
    response = client.get("/api/ready")
    assert response.status_code == 503
    assert response.json()["checks"]["database"]["status"] == "down"


def test_readiness_waits_for_a_preloaded_agent(settings: Settings) -> None:
    from app.main import create_app

    preload = settings.model_copy(
        update={"app": settings.app.model_copy(update={"preload_agent": True})}
    )
    application = create_app(preload)
    application.dependency_overrides[get_database_probe] = lambda: _ok_probe
    client = TestClient(application)
    application.state.agent_error = None  # as the lifespan sets it (not run here)
    starting = client.get("/api/ready")
    assert starting.status_code == 503
    assert starting.json()["checks"]["agent"] == {
        "status": "starting",
        "latency_ms": None,
        "detail": "loading models",
    }
    application.state.agent = object()  # built
    assert client.get("/api/ready").status_code == 200
    application.state.agent = None
    application.state.agent_error = "RerankerError"  # the build failed: the class only
    failed = client.get("/api/ready")
    assert failed.status_code == 503
    assert failed.json()["checks"]["agent"]["detail"] == "RerankerError"


def test_preloading_builds_the_agent_once_in_the_background(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    import app.api.routes.agent as agent_route
    from app import main

    built: list[object] = []
    monkeypatch.setattr(agent_route, "build_agent", lambda engine, s: built.append(s) or "agent")
    application = main.create_app(settings)
    application.state.agent_error = None
    main._preload_agent(application, engine=None)  # type: ignore[arg-type]
    assert application.state.agent == "agent" and len(built) == 1
    assert agent_route.ensure_agent(application, None) == "agent"  # type: ignore[arg-type]
    assert len(built) == 1  # shared afterwards, never rebuilt

    def broken(engine: object, s: object) -> object:
        raise RuntimeError("no model")

    monkeypatch.setattr(agent_route, "build_agent", broken)
    other = main.create_app(settings)
    other.state.agent_error = None
    main._preload_agent(other, engine=None)  # type: ignore[arg-type]
    assert other.state.agent_error == "RuntimeError"
    assert getattr(other.state, "agent", None) is None
