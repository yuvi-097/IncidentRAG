from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any

from app import __version__
from app.config import Settings
from app.schemas.health import ComponentHealth, ComponentStatus, HealthResponse, OverallStatus

logger = logging.getLogger(__name__)

HealthProbe = Callable[[], ComponentHealth]


def _run_probe(name: str, probe: HealthProbe) -> ComponentHealth:
    try:
        return probe()
    except Exception as exc:  # a broken probe must not break the health endpoint
        logger.exception("health.probe_error", extra={"component": name})
        return ComponentHealth(status=ComponentStatus.DOWN, detail=type(exc).__name__)


def check_agent(state: Any) -> ComponentHealth:
    """The preloaded agent: ok once built, starting while its models load, down if the
    build failed (the error class only)."""
    if getattr(state, "agent", None) is not None:
        return ComponentHealth(status=ComponentStatus.OK)
    error = getattr(state, "agent_error", None)
    if error:
        return ComponentHealth(status=ComponentStatus.DOWN, detail=error)
    return ComponentHealth(status=ComponentStatus.STARTING, detail="loading models")


def build_health_report(settings: Settings, probes: Mapping[str, HealthProbe]) -> HealthResponse:
    checks = {name: _run_probe(name, probe) for name, probe in probes.items()}
    healthy = all(check.status is ComponentStatus.OK for check in checks.values())
    return HealthResponse(
        status=OverallStatus.OK if healthy else OverallStatus.DEGRADED,
        service=settings.app.service_name,
        version=__version__,
        environment=settings.app.environment,
        timestamp=datetime.now(UTC),
        checks=checks,
    )
