"""ASGI middleware that assigns a request id, emits one access log per request and
records the request in the process metrics (``app/observability/metrics.py``).

The access log, ``request.completed``, describes the whole request: method, path, status,
total latency, and what ``app/observability/telemetry.py`` collected while it ran (user,
query type, tools, result counts, retrieval / reranker / LLM time, tokens, error codes).
"""

from __future__ import annotations

import logging
import re
import time
import uuid

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.observability import telemetry
from app.observability.context import reset_request_id, set_request_id
from app.observability.metrics import METRICS

logger = logging.getLogger("app.access")

REQUEST_ID_HEADER = "X-Request-ID"
# Client-supplied ids are accepted only if they are short and log-safe;
# anything else is replaced to prevent log injection.
_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def resolve_request_id(candidate: str | None) -> str:
    if candidate and _VALID_REQUEST_ID.fullmatch(candidate):
        return candidate
    return uuid.uuid4().hex


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = resolve_request_id(Headers(scope=scope).get(REQUEST_ID_HEADER))
        token = set_request_id(request_id)
        collected, telemetry_token = telemetry.start()
        status_code = 500
        started = time.perf_counter()

        async def send_with_request_id(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                MutableHeaders(scope=message).append(REQUEST_ID_HEADER, request_id)
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        except Exception as exc:
            collected.add_errors([f"unhandled_{type(exc).__name__}"])
            logger.exception(
                "request.failed", extra={"method": scope["method"], "path": scope["path"]}
            )
            raise
        finally:
            duration = (time.perf_counter() - started) * 1000
            if status_code >= 400:
                collected.add_errors([f"http_{status_code}"])
            # Query strings are deliberately not logged: they may carry user data.
            logger.info(
                "request.completed",
                extra={
                    "method": scope["method"],
                    "path": scope["path"],
                    "route": route_label(scope),
                    "status_code": status_code,
                    "total_latency_ms": round(duration, 2),
                    **collected.fields(),
                },
            )
            METRICS.record_request(scope["method"], route_label(scope), status_code, duration)
            telemetry.finish(telemetry_token)
            reset_request_id(token)


def route_label(scope: Scope) -> str:
    """The endpoint's template (``/api/incidents/{incident_id}``), so that metrics have one
    label per endpoint rather than one per URL. It is rebuilt from the request path and
    the matched path parameters (nested routers keep only their own part of the template);
    paths that matched no route are "other"."""
    if scope.get("route") is None:
        return "other"
    segments = scope["path"].split("/")
    for name, value in (scope.get("path_params") or {}).items():
        segments = [f"{{{name}}}" if s == str(value) else s for s in segments]
    return "/".join(segments)
