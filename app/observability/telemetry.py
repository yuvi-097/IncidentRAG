"""What one request did, collected while it runs and logged once when it ends.

``RequestContextMiddleware`` opens a ``RequestTelemetry`` for every HTTP request (a
context variable, so it follows the request into FastAPI's worker threads) and writes it
into the ``request.completed`` log event. Each layer adds what only it knows:

- the API's authentication: ``user_id``;
- the tool registry: every tool called and how many results it returned;
- timed code (``METRICS.timed``): the time spent in each series, for example
  ``retrieval.rerank`` or ``llm.completion``, summed over the request;
- the LLM synthesizer: token usage, when the provider reports it;
- the agent: the query type, the evidence count and its error codes.

Outside a request (scripts, tests) there is no telemetry and every call is a no-op.

Only identifiers, counts, timings and error codes are kept: never questions, arguments,
evidence, answers or credentials.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Mapping
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any

# Series whose summed time is reported per request (see ``RequestTelemetry.fields``).
RETRIEVAL = "retrieval.search"
RERANK = ("retrieval.rerank", "evidence.rerank")
LLM = "llm.completion"
VERIFY = "stage.claim_verification"


@dataclass
class RequestTelemetry:
    user_id: str | None = None
    query_type: str | None = None
    tools_called: list[str] = field(default_factory=list)
    retrieved_count: int = 0
    evidence_count: int | None = None
    timings: dict[str, float] = field(default_factory=dict)  # series -> summed ms
    llm_calls: int = 0
    tokens: dict[str, int] = field(default_factory=dict)  # input / output / total
    errors: list[str] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add_timing(self, series: str, milliseconds: float) -> None:
        with self._lock:
            self.timings[series] = self.timings.get(series, 0.0) + milliseconds
            if series == LLM:
                self.llm_calls += 1

    def add_tool(self, name: str, status: str, results: int) -> None:
        with self._lock:
            self.tools_called.append(name)
            self.retrieved_count += results
            if status != "ok":
                self.errors.append(f"tool_{status}:{name}")

    def add_tokens(self, input_tokens: int | None, output_tokens: int | None) -> None:
        with self._lock:
            for key, value in (("input", input_tokens), ("output", output_tokens)):
                if value is not None:
                    self.tokens[key] = self.tokens.get(key, 0) + value
            if input_tokens is not None or output_tokens is not None:
                self.tokens["total"] = self.tokens.get("input", 0) + self.tokens.get("output", 0)

    def add_errors(self, codes: Iterable[str]) -> None:
        with self._lock:
            self.errors.extend(codes)

    def fields(self) -> dict[str, Any]:
        """The fields of the ``request.completed`` event (all keys always present)."""
        with self._lock:

            def ms(*series: str) -> float | None:
                values = [self.timings[s] for s in series if s in self.timings]
                return round(sum(values), 1) if values else None

            return {
                "user_id": self.user_id,
                "query_type": self.query_type,
                "tools_called": list(self.tools_called),
                "retrieved_count": self.retrieved_count if self.tools_called else None,
                "evidence_count": self.evidence_count,
                "retrieval_latency_ms": ms(RETRIEVAL),
                "reranker_latency_ms": ms(*RERANK),
                "llm_latency_ms": ms(LLM),
                "verification_latency_ms": ms(VERIFY),
                "llm_calls": self.llm_calls,
                "input_tokens": self.tokens.get("input"),
                "output_tokens": self.tokens.get("output"),
                "total_tokens": self.tokens.get("total"),
                "errors": list(dict.fromkeys(self.errors)),
            }


_current: ContextVar[RequestTelemetry | None] = ContextVar("request_telemetry", default=None)


def start() -> tuple[RequestTelemetry, Token[RequestTelemetry | None]]:
    telemetry = RequestTelemetry()
    return telemetry, _current.set(telemetry)


def finish(token: Token[RequestTelemetry | None]) -> None:
    _current.reset(token)


def current() -> RequestTelemetry | None:
    return _current.get()


def set_user(user_id: str) -> None:
    if (t := current()) is not None:
        t.user_id = user_id


def add_timing(series: str, milliseconds: float) -> None:
    if (t := current()) is not None:
        t.add_timing(series, milliseconds)


def add_tool(name: str, status: str, results: int) -> None:
    if (t := current()) is not None:
        t.add_tool(name, status, results)


def add_tokens(input_tokens: int | None, output_tokens: int | None) -> None:
    if (t := current()) is not None:
        t.add_tokens(input_tokens, output_tokens)


def record_agent(
    query_type: str | None,
    evidence_count: int,
    error_codes: Iterable[str],
    stage_ms: Mapping[str, float] | None = None,
) -> None:
    """The agent's part: its route, the evidence it answered from, its error codes (tool
    failures are already recorded by the registry) and its per-stage times."""
    if (t := current()) is not None:
        t.query_type = query_type
        t.evidence_count = evidence_count
        t.add_errors(error_codes)
        for stage, milliseconds in (stage_ms or {}).items():
            if stage != "total":
                t.add_timing(f"stage.{stage}", milliseconds)


__all__ = [
    "RequestTelemetry",
    "add_timing",
    "add_tokens",
    "add_tool",
    "current",
    "finish",
    "record_agent",
    "set_user",
    "start",
]
