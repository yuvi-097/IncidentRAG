from __future__ import annotations

import logging
import time

from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from app.schemas.health import ComponentHealth, ComponentStatus

logger = logging.getLogger(__name__)


def check_database(engine: Engine) -> ComponentHealth:
    """Run ``SELECT 1``. Never raises; failures are reported as ``down``."""
    started = time.perf_counter()
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        # The full error goes to server logs only; clients get the class name.
        logger.warning(
            "database.check_failed",
            extra={"error_type": type(exc).__name__, "error": str(exc), "latency_ms": latency_ms},
        )
        return ComponentHealth(
            status=ComponentStatus.DOWN, latency_ms=latency_ms, detail=type(exc).__name__
        )
    return ComponentHealth(
        status=ComponentStatus.OK, latency_ms=round((time.perf_counter() - started) * 1000, 2)
    )
