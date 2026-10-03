"""FastAPI application factory.

Run locally with:  uvicorn app.main:app --reload
"""

from __future__ import annotations

import logging
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy.engine import Engine

from app import __version__
from app.api.router import api_router
from app.config import Settings, get_settings
from app.database.session import create_db_engine, create_session_factory
from app.observability.middleware import RequestContextMiddleware
from app.observability.structured_logging import configure_logging

logger = logging.getLogger(__name__)


def _preload_agent(app: FastAPI, engine: Engine) -> None:
    """Build the agent in the background; GET /api/ready says when it is done."""
    from app.api.routes.agent import ensure_agent

    try:
        ensure_agent(app, engine)
        logger.info("agent.preloaded")
    except Exception as exc:  # reported by GET /api/ready; requests retry the build
        app.state.agent_error = type(exc).__name__
        logger.exception("agent.preload_failed")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    engine = create_db_engine(settings.database, application_name=settings.app.service_name)
    app.state.db_engine = engine
    app.state.session_factory = create_session_factory(engine)
    app.state.agent_error = None
    if settings.app.preload_agent:
        threading.Thread(
            target=_preload_agent, args=(app, engine), name="agent-preload", daemon=True
        ).start()
    logger.info(
        "app.startup",
        extra={
            "version": __version__,
            "environment": settings.app.environment,
            "database_url": settings.database.safe_url,
            "llm_provider": settings.llm.provider,
            "embedding_provider": settings.embedding.provider,
        },
    )
    try:
        yield
    finally:
        engine.dispose()
        logger.info("app.shutdown")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.app.log_level, settings.app.log_format)

    app = FastAPI(
        title="OpsRAG",
        summary="Agentic production incident response copilot",
        version=__version__,
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
    )
    app.state.settings = settings
    app.add_middleware(RequestContextMiddleware)
    app.include_router(api_router)
    return app


app = create_app()
