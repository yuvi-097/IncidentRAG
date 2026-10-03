from __future__ import annotations

from functools import partial

from fastapi import APIRouter, Request, Response, status

from app.api.dependencies import DatabaseProbeDep, SettingsDep
from app.schemas.health import HealthResponse, OverallStatus
from app.services.health import HealthProbe, build_health_report, check_agent

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health",
    description=(
        "Liveness plus dependency checks. Returns 200 whenever the API process is up; "
        "`status` is `degraded` if any dependency check fails."
    ),
)
def health(settings: SettingsDep, database_probe: DatabaseProbeDep) -> HealthResponse:
    # Sync handler on purpose: the DB probe blocks, so FastAPI runs it in a threadpool.
    return build_health_report(settings, {"database": database_probe})


@router.get(
    "/ready",
    response_model=HealthResponse,
    summary="Readiness",
    description=(
        "200 when the service can answer: the database responds and, with "
        "OPSRAG_PRELOAD_AGENT on, the agent's models are loaded. 503 otherwise, with the "
        "same body. Container health checks and load balancers use this; /api/health "
        "only says the process is up."
    ),
    responses={503: {"model": HealthResponse, "description": "Not ready yet"}},
)
def ready(
    request: Request,
    response: Response,
    settings: SettingsDep,
    database_probe: DatabaseProbeDep,
) -> HealthResponse:
    probes: dict[str, HealthProbe] = {"database": database_probe}
    if settings.app.preload_agent:
        probes["agent"] = partial(check_agent, request.app.state)
    report = build_health_report(settings, probes)
    if report.status is not OverallStatus.OK:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return report
