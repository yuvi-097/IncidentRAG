from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool

from app.config import ToolSettings
from app.database.seed import seed_database
from app.rag.chunking import ChunkingConfig
from app.rag.embeddings import EmbeddingPipeline, HashingEmbeddingProvider
from app.rag.ingestion import IngestionPipeline
from app.rag.ingestion.persistence import ensure_chunk_table
from app.rag.retrieval import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    Retriever,
    WeightedRetriever,
)
from app.schemas.enums import AccessLevel, Resource, ToolPermission
from app.security import Principal, principal_for_role
from app.synthetic.records import SyntheticDataset
from app.tools import ToolContext, ToolRegistry, build_registry

NOW = datetime(2026, 9, 1, tzinfo=UTC)  # the dataset's window end

ROLES = ("developer", "sre", "manager", "admin")  # grants: app/security/policy.json


@dataclass
class ToolEnv:
    engine: Engine
    dataset: SyntheticDataset
    retriever: HybridRetriever
    registry: ToolRegistry

    bm25: BM25Retriever

    def context(
        self,
        role: str | Principal = "admin",
        retriever: Retriever | None = None,
        **settings: object,
    ) -> ToolContext:
        return ToolContext(
            engine=self.engine,
            principal=role if isinstance(role, Principal) else principal(role),
            settings=ToolSettings(_env_file=None, **settings),  # type: ignore[call-arg]
            retriever=retriever or self.retriever,
            clock=lambda: NOW,
        )


def shared_sqlite_engine() -> Engine:
    """In-memory SQLite shared by every thread (the API runs handlers in a threadpool),
    with foreign keys enforced."""
    engine = create_engine(
        "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
    )

    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection, _record) -> None:  # type: ignore[no-untyped-def]
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    return engine


def principal(role: str) -> Principal:
    assert role in ROLES or role == "nobody", role
    return principal_for_role(f"test-{role}", role)


def custom_principal(
    grants: dict[Resource, set[AccessLevel]], capabilities: frozenset[ToolPermission] = frozenset()
) -> Principal:
    """A principal outside the packaged policy, to isolate one rule."""
    return Principal(
        user_id="test-custom",
        role="custom",
        grants={r: frozenset(levels) for r, levels in grants.items()},
        capabilities=capabilities,
    )


def build_tool_env(dataset: SyntheticDataset) -> ToolEnv:
    """``dataset`` (including logs) in SQLite, ingested and embedded with the hashing
    provider; text search = dense (hashing) + BM25, fused."""
    engine = shared_sqlite_engine()
    ensure_chunk_table(engine)
    seed_database(engine, dataset)
    IngestionPipeline(ChunkingConfig()).run(engine)
    provider = HashingEmbeddingProvider(dimension=256)
    EmbeddingPipeline(provider, batch_size=256).run(engine)
    bm25 = BM25Retriever(engine)
    retriever = HybridRetriever(
        [WeightedRetriever(DenseRetriever(engine, provider)), WeightedRetriever(bm25)]
    )
    return ToolEnv(engine, dataset, retriever, build_registry(), bm25)


@pytest.fixture(scope="session")
def tool_env(dataset: SyntheticDataset) -> Iterator[ToolEnv]:
    """The full default dataset."""
    env = build_tool_env(dataset)
    yield env
    env.engine.dispose()
