from __future__ import annotations

from sqlalchemy import create_engine, text

from app.config import Settings
from app.database import check_database, create_db_engine, create_session_factory
from app.schemas.health import ComponentStatus


def test_create_db_engine_applies_settings(settings: Settings) -> None:
    engine = create_db_engine(settings.database)
    try:
        assert engine.url.drivername == "postgresql+psycopg"
        assert engine.url.host == settings.database.host
        assert engine.url.port == settings.database.port
        assert engine.url.database == settings.database.db
        assert engine.pool.size() == settings.database.pool_size
    finally:
        engine.dispose()


def test_check_database_ok_against_live_engine() -> None:
    # SQLite in-memory stands in for a reachable database: the check is dialect-agnostic.
    engine = create_engine("sqlite://")
    try:
        result = check_database(engine)
    finally:
        engine.dispose()

    assert result.status is ComponentStatus.OK
    assert result.latency_ms is not None and result.latency_ms >= 0
    assert result.detail is None


def test_check_database_down_when_unreachable(settings: Settings) -> None:
    engine = create_db_engine(settings.database)
    try:
        result = check_database(engine)
    finally:
        engine.dispose()

    assert result.status is ComponentStatus.DOWN
    assert result.detail == "OperationalError"


def test_session_factory_produces_bound_sessions() -> None:
    engine = create_engine("sqlite://")
    try:
        with create_session_factory(engine)() as session:
            assert session.execute(text("SELECT 1")).scalar_one() == 1
    finally:
        engine.dispose()
