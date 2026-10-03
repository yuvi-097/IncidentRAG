from __future__ import annotations

import os
import socket
from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine

from app.config import Settings, load_settings
from app.database.schema import create_schema
from app.main import create_app
from app.synthetic.generator import GenerationConfig, generate_dataset
from app.synthetic.records import SyntheticDataset

CONFIG_ENV_PREFIXES = (
    "OPSRAG_",
    "POSTGRES_",
    "LLM_",
    "EMBEDDING_",
    "CHUNKING_",
    "RETRIEVAL_",
    "BM25_",
    "RERANKER_",
    "TOOLS_",
    "AGENT_",
    "VERIFY_",
    "SECURITY_",
)


def unused_tcp_port() -> int:
    """A localhost port with nothing listening on it (connections are refused)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Strip OpsRAG config variables so a test sees only what it sets itself."""
    for key in list(os.environ):
        if key.upper().startswith(CONFIG_ENV_PREFIXES):
            monkeypatch.delenv(key)
    return monkeypatch


@pytest.fixture
def settings(clean_env: pytest.MonkeyPatch) -> Settings:
    clean_env.setenv("OPSRAG_ENVIRONMENT", "test")
    clean_env.setenv("OPSRAG_LOG_FORMAT", "json")
    # Unit tests never need a database; point at a closed port so any
    # real connection attempt fails fast instead of touching a dev DB.
    clean_env.setenv("POSTGRES_HOST", "127.0.0.1")
    clean_env.setenv("POSTGRES_PORT", str(unused_tcp_port()))
    clean_env.setenv("POSTGRES_CONNECT_TIMEOUT_SECONDS", "2")
    # A password, so tests can check it never appears in responses or logs.
    clean_env.setenv("POSTGRES_PASSWORD", "unit-test-db-password-value")
    return load_settings(env_file=None)


@pytest.fixture
def app(settings: Settings) -> Iterator[FastAPI]:
    application = create_app(settings)
    yield application
    application.dependency_overrides.clear()


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    # Context manager so the lifespan (engine creation/disposal) runs.
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def dataset() -> SyntheticDataset:
    """The full default dataset, generated once per test session (~2s)."""
    return generate_dataset(GenerationConfig())


@pytest.fixture
def sqlite_engine() -> Iterator[Engine]:
    """In-memory SQLite with foreign keys enforced and the OpsRAG schema created."""
    engine = create_engine("sqlite://")

    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection, _record) -> None:
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    create_schema(engine)
    yield engine
    engine.dispose()
