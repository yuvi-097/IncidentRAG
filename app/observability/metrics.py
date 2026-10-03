"""In-process metrics: request counts, latency percentiles, tool calls and errors.

One registry per API process (``METRICS``). Latencies are kept in a rolling window
(the last ``window`` observations of each series), from which p50/p95/p99 are computed
by the nearest-rank method; counters count everything since the process started.

What is recorded, and where:
- ``http.request``: every HTTP request (``RequestContextMiddleware``), by route and status;
- ``agent.total`` and ``stage.<name>``: every agent run, from its per-stage timings
  (``record_agent_run``, called by the API route);
- ``tool.<name>``: every tool call of a run, with its status;
- ``retrieval.search``: every retriever call made by a tool (``TimedRetriever``);
- ``retrieval.first_stage`` / ``retrieval.rerank``: the two stages of the retrieval
  pipeline (candidates, then the cross-encoder);
- ``llm.completion``: every call to the LLM, when one is configured.

Nothing here leaves the process: it is served at ``GET /api/metrics`` for the UI. With
several API workers, each has its own numbers; an exporter to a metrics backend is left
for a later phase.
"""

from __future__ import annotations

import math
import threading
import time
from collections import Counter, deque
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

from app.observability import telemetry

DEFAULT_WINDOW = 2000


def percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile (``q`` in 0..100) of ``values``; None when empty."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return ordered[rank - 1]


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None, "max": None}
    return {
        "count": len(values),
        "mean": round(sum(values) / len(values), 1),
        "p50": round(percentile(values, 50) or 0.0, 1),
        "p95": round(percentile(values, 95) or 0.0, 1),
        "p99": round(percentile(values, 99) or 0.0, 1),
        "max": round(max(values), 1),
    }


class MetricsRegistry:
    def __init__(self, window: int = DEFAULT_WINDOW) -> None:
        self.window = window
        self._lock = threading.Lock()
        self._series: dict[str, deque[float]] = {}
        self._totals: Counter[str] = Counter()  # observations of each series, all time
        self._counters: dict[str, Counter[str]] = {}
        self.started_at = datetime.now(UTC)
        self._started = time.monotonic()

    # --- recording ---------------------------------------------------------------------

    def observe(self, series: str, milliseconds: float) -> None:
        with self._lock:
            values = self._series.get(series)
            if values is None:
                values = self._series[series] = deque(maxlen=self.window)
            values.append(float(milliseconds))
            self._totals[series] += 1

    def increment(self, counter: str, label: str = "total", amount: int = 1) -> None:
        with self._lock:
            self._counters.setdefault(counter, Counter())[label] += amount

    @contextmanager
    def timed(self, series: str) -> Iterator[None]:
        """Time the block into ``series``, and into the current request's telemetry."""
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - started) * 1000
            self.observe(series, elapsed)
            telemetry.add_timing(series, elapsed)

    def record_request(self, method: str, route: str, status: int, milliseconds: float) -> None:
        self.observe("http.request", milliseconds)
        self.increment("http.by_route", f"{method} {route}")
        self.increment("http.by_status", str(status))
        if status >= 500:
            self.increment("errors", "http_5xx")

    def record_agent_run(self, state: Any) -> None:
        """One agent run, from its ``AgentState`` (per-stage timings, tools, errors)."""
        latency: Mapping[str, float] = state.latency_ms
        if "total" in latency:
            self.observe("agent.total", latency["total"])
        for stage, ms in latency.items():
            if stage != "total":
                self.observe(f"stage.{stage}", ms)
        self.increment("agent.requests")
        if state.query_type is not None:
            self.increment("agent.by_query_type", state.query_type.value)
        self.increment("agent.by_confidence", state.confidence.value)
        self.increment("agent.by_plan", state.plan)
        for record in state.tool_results:
            self.observe(f"tool.{record.tool}", record.duration_ms)
            self.increment("tool.calls", record.tool)
            if record.status != "ok":
                self.increment("tool.errors", record.tool)
        for error in state.errors:
            self.increment("errors", error.code)
        if state.security.blocked:
            self.increment("agent.security", "question_refused")
        if state.security.quarantined:
            self.increment("agent.security", "evidence_quarantined")

    # --- reading ------------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            series = {name: list(values) for name, values in self._series.items()}
            totals = dict(self._totals)
            counters = {name: dict(c) for name, c in self._counters.items()}

        def stats(name: str) -> dict[str, float | int | None]:
            out = summarize(series.get(name, []))
            out["total"] = totals.get(name, 0)
            return out

        def prefixed(prefix: str) -> dict[str, dict[str, float | int | None]]:
            return {
                name.removeprefix(prefix): stats(name)
                for name in sorted(series)
                if name.startswith(prefix)
            }

        tool_calls = counters.get("tool.calls", {})
        tool_errors = counters.get("tool.errors", {})
        return {
            "started_at": self.started_at.isoformat(),
            "uptime_seconds": round(time.monotonic() - self._started, 1),
            "window": self.window,
            "http": {
                "requests": sum(counters.get("http.by_status", {}).values()),
                "by_status": counters.get("http.by_status", {}),
                "by_route": counters.get("http.by_route", {}),
                "latency_ms": stats("http.request"),
            },
            "agent": {
                "requests": counters.get("agent.requests", {}).get("total", 0),
                "latency_ms": stats("agent.total"),
                "build_ms": stats("agent.build"),  # one-off: loading the models
                "stages": prefixed("stage."),
                "by_query_type": counters.get("agent.by_query_type", {}),
                "by_confidence": counters.get("agent.by_confidence", {}),
                "by_plan": counters.get("agent.by_plan", {}),
                "security": counters.get("agent.security", {}),
            },
            "retrieval": {
                "search": stats("retrieval.search"),
                "first_stage": stats("retrieval.first_stage"),
                "rerank": stats("retrieval.rerank"),
            },
            "llm": {"completion": stats("llm.completion")},
            "tools": {
                tool: {
                    "calls": calls,
                    "errors": tool_errors.get(tool, 0),
                    "latency_ms": stats(f"tool.{tool}"),
                }
                for tool, calls in sorted(tool_calls.items())
            },
            "errors": counters.get("errors", {}),
        }

    def reset(self) -> None:
        with self._lock:
            self._series.clear()
            self._totals.clear()
            self._counters.clear()
            self.started_at = datetime.now(UTC)
            self._started = time.monotonic()


METRICS = MetricsRegistry()

__all__ = ["DEFAULT_WINDOW", "METRICS", "MetricsRegistry", "percentile", "summarize"]
