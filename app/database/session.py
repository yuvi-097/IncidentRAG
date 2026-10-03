"""Engine and session factories.

Creating an engine does not open a connection, so the API can start (and report
``degraded`` health) when PostgreSQL is unavailable.
"""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import DatabaseSettings, Settings


def create_db_engine(settings: DatabaseSettings, application_name: str = "opsrag-api") -> Engine:
    return create_engine(
        settings.url,
        pool_size=settings.pool_size,
        max_overflow=settings.max_overflow,
        pool_timeout=settings.pool_timeout_seconds,
        pool_recycle=settings.pool_recycle_seconds,
        pool_pre_ping=True,
        echo=settings.echo_sql,
        connect_args={
            "connect_timeout": settings.connect_timeout_seconds,
            "application_name": application_name,
        },
    )


def create_sql_reader_engine(settings: Settings) -> Engine | None:
    """The engine query_database runs through: the dedicated read-only role when
    TOOLS_SQL_USER is set, otherwise None (the application's engine is used)."""
    tools = settings.tools
    if not tools.sql_user:
        return None
    reader = settings.database.model_copy(
        update={"user": tools.sql_user, "password": tools.sql_password, "pool_size": 2}
    )
    return create_db_engine(reader, application_name="opsrag-sql-reader")


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
