"""Embeddings, dense, BM25 and hybrid retrieval on real PostgreSQL + pgvector.

Opt in with OPSRAG_RUN_INTEGRATION_TESTS=1. Uses the deterministic hashing
provider (fast, no model) to check the database side: vector storage, the
partial HNSW index, filtered search, agreement with exact search, and BM25 /
hybrid search (index built from and filtered by PostgreSQL rows).
Seeds and ingests into the configured database; use a development database only.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass

import pytest
from sqlalchemy import text
from sqlalchemy.dialects import postgresql
from sqlalchemy.engine import Engine

from app.config import load_settings
from app.database.seed import seed_database
from app.database.session import create_db_engine
from app.rag.chunking import ChunkingConfig
from app.rag.embeddings import EmbeddingPipeline, EmbeddingReport, HashingEmbeddingProvider
from app.rag.embeddings.index import vector_index_name
from app.rag.ingestion import IngestionPipeline
from app.rag.ingestion.persistence import ensure_chunk_table
from app.rag.reranking import CrossEncoderReranker
from app.rag.retrieval import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    RetrievalPipeline,
    WeightedRetriever,
)
from app.rag.retrieval.vector_store import InMemoryVectorStore, PgVectorStore
from app.rag.store import ChunkFilter
from app.schemas.enums import AccessLevel, SourceType
from app.security import principal_for_role
from app.synthetic.records import SyntheticDataset
from tests.retrieval.test_bm25 import FILTERS, satisfies
from tests.retrieval.test_reranker import KeywordScorer

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("OPSRAG_RUN_INTEGRATION_TESTS") != "1",
        reason="set OPSRAG_RUN_INTEGRATION_TESTS=1 with PostgreSQL running",
    ),
]


@dataclass
class Setup:
    engine: Engine
    provider: HashingEmbeddingProvider
    report: EmbeddingReport


@pytest.fixture(scope="module")
def setup(dataset: SyntheticDataset) -> Iterator[Setup]:
    settings = load_settings()
    if settings.app.environment == "production":
        pytest.skip("refusing to write synthetic data into production")
    engine = create_db_engine(settings.database)
    ensure_chunk_table(engine)
    seed_database(engine, dataset)
    IngestionPipeline(ChunkingConfig()).run(engine)
    provider = HashingEmbeddingProvider(dimension=256, name="hashing-integration")
    report = EmbeddingPipeline(provider, batch_size=256).run(engine)
    yield Setup(engine, provider, report)
    engine.dispose()


def test_embeddings_are_stored_for_every_chunk(setup: Setup) -> None:
    with setup.engine.connect() as connection:
        chunks = connection.execute(text("SELECT count(*) FROM document_chunks")).scalar_one()
        vectors, dims = connection.execute(
            text(
                "SELECT count(*), max(vector_dims(embedding)) FROM chunk_embeddings "
                "WHERE model = :m"
            ),
            {"m": setup.provider.name},
        ).one()
    assert vectors == chunks == setup.report.total_chunks and dims == 256


def test_partial_hnsw_index_exists_for_the_model(setup: Setup) -> None:
    name = vector_index_name(setup.provider.name, 256)
    with setup.engine.connect() as connection:
        definition = connection.execute(
            text("SELECT indexdef FROM pg_indexes WHERE indexname = :n"), {"n": name}
        ).scalar_one()
    assert "USING hnsw" in definition and "vector_cosine_ops" in definition
    assert "vector(256)" in definition and "hashing-integration" in definition
    assert setup.report.index == name


def test_search_query_can_use_the_hnsw_index(setup: Setup) -> None:
    store = PgVectorStore(setup.engine, setup.provider.name, 256)
    from pgvector.sqlalchemy import Vector
    from sqlalchemy import cast, literal, select

    from app.database.models import ChunkEmbedding

    distance = cast(ChunkEmbedding.embedding, Vector(256)).cosine_distance(
        setup.provider.embed_query("pool")
    )
    query = (
        select(ChunkEmbedding.chunk_id)
        .where(ChunkEmbedding.model == literal(store.model, literal_execute=True))
        .order_by(distance)
        .limit(5)
    )
    sql = str(query.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}))
    with setup.engine.begin() as connection:
        connection.execute(text("SET LOCAL enable_seqscan = off"))
        plan = "\n".join(row[0] for row in connection.execute(text(f"EXPLAIN {sql}")))
    assert vector_index_name(setup.provider.name, 256) in plan


def test_approximate_search_agrees_with_exact_search(setup: Setup) -> None:
    pg = PgVectorStore(setup.engine, setup.provider.name, 256, ef_search=100)
    exact = InMemoryVectorStore(setup.engine, setup.provider.name, 256)
    for query in (
        "database connection pool exhausted",
        "redis memory pressure on carts",
        "jwks signing key rotation 401",
        "kafka consumer lag",
    ):
        vector = setup.provider.embed_query(query)
        approx_ids = {chunk_id for chunk_id, _ in pg.nearest(vector, 10, None)}
        exact_ids = {chunk_id for chunk_id, _ in exact.nearest(vector, 10, None)}
        assert len(approx_ids & exact_ids) >= 8, query  # HNSW is approximate; >= 80% overlap


def test_selective_filters_still_return_every_match(setup: Setup) -> None:
    """HNSW post-filtering can starve a selective filter; the exact-scan fallback must not."""
    chunk_filter = ChunkFilter(
        services=frozenset({"payment-service"}), versions=frozenset({"v2.8.1"})
    )
    retriever = DenseRetriever(setup.engine, setup.provider)
    results = retriever.search("anything at all", top_k=50, filters=chunk_filter)
    with setup.engine.connect() as connection:
        expected = connection.execute(
            text(
                "SELECT count(*) FROM document_chunks "
                "WHERE service_id = 'payment-service' AND version = 'v2.8.1'"
            )
        ).scalar_one()
    assert 0 < len(results) == expected
    assert all(r.version == "v2.8.1" and r.service_id == "payment-service" for r in results)


def test_postgres_and_exact_search_agree_with_filters(setup: Setup) -> None:
    chunk_filter = ChunkFilter(source_types=frozenset({SourceType.RUNBOOK}))
    vector = setup.provider.embed_query("database connection pool")
    pg = PgVectorStore(setup.engine, setup.provider.name, 256).nearest(vector, 5, chunk_filter)
    exact = InMemoryVectorStore(setup.engine, setup.provider.name, 256).nearest(
        vector, 5, chunk_filter
    )
    assert [c for c, _ in pg] == [c for c, _ in exact]


def test_bm25_finds_exact_identifiers_on_postgres(setup: Setup, dataset: SyntheticDataset) -> None:
    bm25 = BM25Retriever(setup.engine)
    assert len(bm25.index) == setup.report.total_chunks
    for incident in dataset.incidents[::20]:
        assert any(r.document_id == incident.id for r in bm25.search(incident.id, 5)), incident.id
    top = bm25.search("PAYMENT_SERVICE_DB_POOL_TIMEOUT_SECONDS", 3)
    assert any("PAYMENT_SERVICE_DB_POOL_TIMEOUT_SECONDS" in r.content for r in top)


@pytest.mark.parametrize("chunk_filter", FILTERS)
def test_bm25_and_hybrid_filters_on_postgres(setup: Setup, chunk_filter: ChunkFilter) -> None:
    dense = DenseRetriever(setup.engine, setup.provider)
    bm25 = BM25Retriever(setup.engine)
    hybrid = HybridRetriever([WeightedRetriever(dense), WeightedRetriever(bm25)], depth=50)
    pipeline = RetrievalPipeline(hybrid, CrossEncoderReranker(KeywordScorer()), candidates=30)
    query = "database connection pool timeout payment v2.8.1 configuration"
    for retriever in (bm25, hybrid, pipeline):
        results = retriever.search(query, top_k=10, filters=chunk_filter)
        assert results and all(satisfies(r, chunk_filter) for r in results), retriever.name


def test_hybrid_fuses_postgres_rankings(setup: Setup) -> None:
    dense = DenseRetriever(setup.engine, setup.provider)
    bm25 = BM25Retriever(setup.engine)
    hybrid = HybridRetriever([WeightedRetriever(dense), WeightedRetriever(bm25)], depth=50)
    query = "payment-service connection pool exhaustion"
    dense_ranks = {r.chunk_id: r.rank for r in dense.search(query, 50)}
    bm25_ranks = {r.chunk_id: r.rank for r in bm25.search(query, 50)}
    results = hybrid.search(query, top_k=10)
    assert len(results) == 10
    for r in results:
        expected = sum(
            1 / (60 + ranks[r.chunk_id])
            for ranks in (dense_ranks, bm25_ranks)
            if r.chunk_id in ranks
        )
        assert r.score == pytest.approx(expected, abs=1e-6)
    developer = hybrid.search(
        query,
        top_k=10,
        filters=ChunkFilter(access=principal_for_role("pg-dev", "developer").chunk_access()),
    )
    assert all(r.access_level in {AccessLevel.PUBLIC, AccessLevel.ENGINEERING} for r in developer)


def test_re_embedding_is_incremental(setup: Setup) -> None:
    again = EmbeddingPipeline(setup.provider).run(setup.engine)
    assert again.embedded == 0 and again.already_embedded == again.total_chunks
