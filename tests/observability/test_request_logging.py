"""Every request is logged once, with what it did, and without secrets."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.agents.graph import Agent
from app.agents.synthesis import LLMSynthesizer
from app.api.dependencies import get_engine
from app.api.routes.agent import get_agent
from app.llm.base import ChatMessage, LLMResponse
from app.observability import telemetry
from app.observability.metrics import METRICS
from app.observability.structured_logging import (
    JsonFormatter,
    RedactingFilter,
    RequestContextFilter,
)
from app.security.auth import issue_token
from tests.agents.conftest import make_agent
from tests.tools.conftest import ToolEnv

FIELDS = {
    "request_id",
    "user_id",
    "query_type",
    "tools_called",
    "retrieved_count",
    "evidence_count",
    "retrieval_latency_ms",
    "reranker_latency_ms",
    "llm_latency_ms",
    "verification_latency_ms",
    "total_latency_ms",
    "llm_calls",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "errors",
}


class _JsonLines(logging.Handler):
    """Keeps every line the production handler would write (same formatter and filters)."""

    def __init__(self) -> None:
        super().__init__()
        self.lines: list[dict[str, Any]] = []
        self.setFormatter(JsonFormatter())
        self.addFilter(RequestContextFilter())
        self.addFilter(RedactingFilter())

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(json.loads(self.format(record)))


@pytest.fixture
def log_lines() -> Iterator[list[dict[str, Any]]]:
    handler = _JsonLines()
    root = logging.getLogger()
    previous = root.level
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    yield handler.lines
    root.removeHandler(handler)
    root.setLevel(previous)


def completed(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [line for line in lines if line["message"] == "request.completed"]


class _UsageLLM:
    """A model that reports token usage, as OpenAI-compatible servers do."""

    name = "usage-reporting"

    def complete(self, messages: list[ChatMessage]) -> LLMResponse:
        return LLMResponse(
            text="INC-0406 started on 2026-06-16 at 17:05 UTC after the v2.8.1 release [E1].",
            model="fake",
            input_tokens=812,
            output_tokens=37,
        )


def api_for(app: FastAPI, agent: Agent, env: ToolEnv) -> TestClient:
    app.dependency_overrides[get_engine] = lambda: env.engine
    app.dependency_overrides[get_agent] = lambda: agent
    return TestClient(app)


def bearer(env: ToolEnv, role: str) -> tuple[str, dict[str, str]]:
    user = next(u.id for u in env.dataset.users if u.role_id == role and u.is_active)
    token = issue_token(env.engine, user, "logging tests")
    return token, {"Authorization": f"Bearer {token}"}


def test_an_answer_is_logged_with_everything_it_did(
    app: FastAPI, agent: Agent, tool_env: ToolEnv, log_lines: list[dict[str, Any]]
) -> None:
    METRICS.reset()
    token, headers = bearer(tool_env, "sre")
    response = api_for(app, agent, tool_env).post(
        "/api/agent/ask",
        json={"question": "Why did payment-service fail after deployment v2.8.1?"},
        headers={**headers, "X-Request-ID": "trace-abc-123"},
    )
    assert response.status_code == 200
    body = response.json()
    [event] = completed(log_lines)
    assert set(event) >= FIELDS
    assert event["request_id"] == "trace-abc-123"
    assert event["user_id"] == next(
        u.id for u in tool_env.dataset.users if u.role_id == "sre" and u.is_active
    )
    assert event["query_type"] == body["query_type"]
    assert event["tools_called"] == [t["tool"] for t in body["tools"]]
    assert event["retrieved_count"] == sum(t["results"] for t in body["tools"]) > 0
    assert event["evidence_count"] == len(body["evidence"])
    assert event["route"] == "/api/agent/ask" and event["status_code"] == 200
    assert event["total_latency_ms"] >= body["latency_ms"]["total"] > 0
    assert event["verification_latency_ms"] == pytest.approx(
        body["latency_ms"]["claim_verification"], abs=0.2
    )
    # No LLM and no cross-encoder in the test agent: nothing measured, so null, not 0.
    assert event["llm_latency_ms"] is None and event["llm_calls"] == 0
    assert event["input_tokens"] is None and event["reranker_latency_ms"] is None
    assert event["errors"] == []
    assert token not in json.dumps(log_lines)  # the credential is never logged


def test_token_usage_and_llm_latency_are_logged_when_reported(
    app: FastAPI, tool_env: ToolEnv, log_lines: list[dict[str, Any]]
) -> None:
    agent = make_agent(tool_env)
    agent.synthesizer = LLMSynthesizer(_UsageLLM(), agent.synthesizer)  # type: ignore[arg-type]
    _, headers = bearer(tool_env, "sre")
    response = api_for(app, agent, tool_env).post(
        "/api/agent/ask",
        json={"question": "Why did payment-service fail after deployment v2.8.1?"},
        headers=headers,
    )
    assert response.status_code == 200
    [event] = completed(log_lines)
    assert event["llm_calls"] == 1 and event["llm_latency_ms"] is not None
    assert (event["input_tokens"], event["output_tokens"], event["total_tokens"]) == (
        812,
        37,
        849,
    )


def test_questions_and_credentials_in_them_are_not_logged(
    app: FastAPI, agent: Agent, tool_env: ToolEnv, log_lines: list[dict[str, Any]]
) -> None:
    secret = "sk-" + "live-" + "4f9a8b7c6d5e4f3a2b1c0d9e8f7a6b5c"  # built at run time
    _, headers = bearer(tool_env, "sre")
    api_for(app, agent, tool_env).post(
        "/api/agent/ask",
        json={"question": f"Is the payment key {secret} still valid for payment-service?"},
        headers=headers,
    )
    written = json.dumps(log_lines)
    assert secret not in written and "still valid" not in written
    assert completed(log_lines)


def test_failed_requests_say_why_and_who(
    app: FastAPI, agent: Agent, tool_env: ToolEnv, log_lines: list[dict[str, Any]]
) -> None:
    api = api_for(app, agent, tool_env)
    _, headers = bearer(tool_env, "developer")
    restricted = next(i for i in tool_env.dataset.incidents if i.access_level.value == "sre")
    api.get("/api/me")  # no credentials
    api.get(f"/api/incidents/{restricted.id}", headers=headers)  # not readable: 404
    unauthenticated, hidden = completed(log_lines)
    assert unauthenticated["user_id"] is None and unauthenticated["errors"] == ["http_401"]
    assert hidden["user_id"] is not None
    assert hidden["tools_called"] == ["search_incidents"]
    assert hidden["errors"] == ["http_404"]


def test_tool_failures_are_logged_as_error_codes() -> None:
    collected, token = telemetry.start()
    try:
        telemetry.add_tool("search_logs", "execution_error", 0)
        telemetry.add_tool("search_incidents", "ok", 3)
        with METRICS.timed("retrieval.rerank"):
            pass
        with METRICS.timed("evidence.rerank"):
            pass
        telemetry.add_tokens(10, None)
    finally:
        telemetry.finish(token)
    fields = collected.fields()
    assert fields["tools_called"] == ["search_logs", "search_incidents"]
    assert fields["retrieved_count"] == 3
    assert fields["errors"] == ["tool_execution_error:search_logs"]
    assert fields["reranker_latency_ms"] is not None  # both reranking series, summed
    assert (fields["input_tokens"], fields["output_tokens"], fields["total_tokens"]) == (
        10,
        None,
        10,
    )


def test_outside_a_request_nothing_is_collected() -> None:
    assert telemetry.current() is None
    telemetry.add_tool("search_incidents", "ok", 1)  # no request: a no-op
    telemetry.set_user("someone")
    assert telemetry.current() is None
