"""seed_db.py --if-empty: what the container bootstrap runs on every start."""

from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy import create_engine

from app.config import Settings
from scripts import seed_db
from tests.tools.conftest import ToolEnv, tool_env  # noqa: F401  (shared fixture)


class _Engine:
    def __init__(self) -> None:
        self.disposed = False
        self.dialect = type("Dialect", (), {"name": "postgresql"})()

    def dispose(self) -> None:
        self.disposed = True


def production(settings: Settings) -> Settings:
    return settings.model_copy(
        update={"app": settings.app.model_copy(update={"environment": "production"})}
    )


def test_has_data_tells_an_empty_database_from_a_loaded_one(tool_env: ToolEnv) -> None:  # noqa: F811
    assert seed_db.has_data(create_engine("sqlite://")) is False  # no tables at all
    assert seed_db.has_data(tool_env.engine) is True


def test_production_refuses_to_replace_data(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(seed_db, "load_settings", lambda: production(settings))
    connect = pytest.fail  # no connection may even be attempted
    monkeypatch.setattr(seed_db, "create_db_engine", lambda *a, **k: connect("connected"))
    assert seed_db.main([]) == 2
    assert seed_db.main(["--if-empty", "--recreate-schema"]) == 2  # would drop tables
    assert "Refusing to replace data" in capsys.readouterr().err


def test_if_empty_changes_nothing_in_a_loaded_database(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    engine = _Engine()
    calls: list[str] = []
    monkeypatch.setattr(seed_db, "load_settings", lambda: production(settings))
    monkeypatch.setattr(seed_db, "create_db_engine", lambda *a, **k: engine)
    monkeypatch.setattr(seed_db, "has_data", lambda e: True)
    for name in ("load_dataset", "prepare_schema", "seed_database"):
        monkeypatch.setattr(seed_db, name, lambda *a, n=name, **k: calls.append(n))
    assert seed_db.main(["--if-empty"]) == 0
    assert calls == [] and engine.disposed
    assert "nothing changed" in capsys.readouterr().out


def test_if_empty_loads_an_empty_database_even_in_production(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = _Engine()
    calls: list[str] = []

    def record(name: str) -> Any:
        return lambda *a, **k: calls.append(name) or []

    monkeypatch.setattr(seed_db, "load_settings", lambda: production(settings))
    monkeypatch.setattr(seed_db, "create_db_engine", lambda *a, **k: engine)
    monkeypatch.setattr(seed_db, "has_data", lambda e: False)
    monkeypatch.setattr(seed_db, "load_dataset", record("load_dataset"))
    monkeypatch.setattr(seed_db, "prepare_schema", record("prepare_schema"))
    monkeypatch.setattr(seed_db, "seed_database", record("seed_database"))
    monkeypatch.setattr(seed_db, "table_counts", lambda e: {"incidents": 540})
    assert seed_db.main(["--if-empty"]) == 0
    assert calls == ["load_dataset", "prepare_schema", "seed_database"]
