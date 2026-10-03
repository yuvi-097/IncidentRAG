"""FastAPI dependencies.

Resources are created once in the application lifespan and stored on
``app.state``; dependencies hand them to routes so tests can override them.
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import partial
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from app.config import Settings
from app.database.health import check_database
from app.services.health import HealthProbe


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_engine(request: Request) -> Engine:
    return request.app.state.db_engine


def get_db_session(request: Request) -> Iterator[Session]:
    session: Session = request.app.state.session_factory()
    try:
        yield session
    finally:
        session.close()


def get_database_probe(engine: Annotated[Engine, Depends(get_engine)]) -> HealthProbe:
    return partial(check_database, engine)


SettingsDep = Annotated[Settings, Depends(get_settings)]
DbSessionDep = Annotated[Session, Depends(get_db_session)]
DatabaseProbeDep = Annotated[HealthProbe, Depends(get_database_probe)]
