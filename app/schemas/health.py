from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class ComponentStatus(StrEnum):
    OK = "ok"
    STARTING = "starting"  # e.g. the agent's models are still loading
    DOWN = "down"


class OverallStatus(StrEnum):
    OK = "ok"
    DEGRADED = "degraded"


class ComponentHealth(BaseModel):
    status: ComponentStatus
    latency_ms: float | None = Field(default=None, description="Time taken by the check.")
    detail: str | None = Field(
        default=None,
        description="Non-sensitive failure reason (error class only; details are logged).",
    )


class HealthResponse(BaseModel):
    status: OverallStatus
    service: str
    version: str
    environment: str
    timestamp: datetime
    checks: dict[str, ComponentHealth]
