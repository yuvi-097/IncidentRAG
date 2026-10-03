"""The in-process metrics registry and the request route labels."""

from __future__ import annotations

import threading

from app.observability.metrics import MetricsRegistry, percentile, summarize
from app.observability.middleware import route_label


def test_nearest_rank_percentiles() -> None:
    values = [float(v) for v in range(1, 101)]  # 1..100
    assert percentile(values, 50) == 50 and percentile(values, 95) == 95
    assert percentile(values, 99) == 99 and percentile(values, 100) == 100
    assert percentile([7.0], 99) == 7.0 and percentile([], 50) is None
    stats = summarize([10.0, 20.0, 30.0, 40.0])
    assert stats == {"count": 4, "mean": 25.0, "p50": 20.0, "p95": 40.0, "p99": 40.0, "max": 40.0}
    assert summarize([])["p95"] is None


def test_window_counters_and_snapshot() -> None:
    registry = MetricsRegistry(window=3)
    for ms in (100, 200, 300, 400):
        registry.observe("retrieval.search", ms)
    snap = registry.snapshot()
    search = snap["retrieval"]["search"]
    assert search["count"] == 3 and search["total"] == 4  # the window keeps the last three
    assert search["p50"] == 300 and search["max"] == 400
    registry.record_request("GET", "/api/health", 200, 5.0)
    registry.record_request("POST", "/api/agent/ask", 500, 50.0)
    snap = registry.snapshot()
    assert snap["http"]["requests"] == 2 and snap["http"]["by_status"] == {"200": 1, "500": 1}
    assert snap["errors"] == {"http_5xx": 1}
    with registry.timed("llm.completion"):
        pass
    assert registry.snapshot()["llm"]["completion"]["count"] == 1
    registry.reset()
    assert registry.snapshot()["http"]["requests"] == 0


def test_recording_is_thread_safe() -> None:
    registry = MetricsRegistry()

    def work() -> None:
        for _ in range(500):
            registry.observe("agent.total", 1.0)
            registry.increment("agent.requests")

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    snap = registry.snapshot()
    assert snap["agent"]["requests"] == 2000 and snap["agent"]["latency_ms"]["total"] == 2000


def test_agent_runs_are_recorded_from_their_state(ask) -> None:
    registry = MetricsRegistry()
    state = ask("What caused INC-0406?", "sre")
    registry.record_agent_run(state)
    snap = registry.snapshot()
    assert snap["agent"]["requests"] == 1
    assert snap["agent"]["by_query_type"] == {state.query_type.value: 1}
    assert set(snap["agent"]["stages"]) >= {"tool_execution", "synthesis", "claim_verification"}
    assert snap["tools"]["search_incidents"]["calls"] >= 1


def test_route_labels_have_one_value_per_endpoint() -> None:
    route = object()
    scope = {
        "path": "/api/incidents/INC-0406",
        "path_params": {"incident_id": "INC-0406"},
        "route": route,
    }
    assert route_label(scope) == "/api/incidents/{incident_id}"
    trace = {**scope, "path": "/api/incidents/INC-0406/trace"}
    assert route_label(trace) == "/api/incidents/{incident_id}/trace"
    assert route_label({"path": "/nowhere", "route": None}) == "other"
