"""DenseRetriever: ranking, filters, provenance, validation, and the Retriever interface.

Uses the hashing provider (lexical, deterministic), so these tests check retrieval
mechanics, not semantic quality; see test_semantic_retrieval.py for the real model."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.rag.embeddings import HashingEmbeddingProvider
from app.rag.retrieval import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    RetrievalError,
    RetrievalPipeline,
    Retriever,
    SparseRetriever,
)
from app.rag.store import ChunkFilter
from app.schemas.enums import AccessLevel, SourceType
from app.security import principal_for_role
from tests.retrieval.conftest import Embedded


@pytest.fixture(scope="module")
def retriever(embedded: Embedded) -> DenseRetriever:
    return DenseRetriever(embedded.engine, embedded.provider)


def test_results_are_ranked_and_bounded(retriever: DenseRetriever) -> None:
    results = retriever.search("database connection pool exhausted timeouts", top_k=7)
    assert len(results) == 7
    assert [r.rank for r in results] == list(range(1, 8))
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert all(r.retriever == "dense" for r in results)


def test_a_chunk_is_its_own_nearest_neighbour(
    retriever: DenseRetriever, embedded: Embedded
) -> None:
    """End-to-end plumbing check: querying with a chunk's own text finds that chunk."""
    target = next(c for c in embedded.chunks if c.document_id == "RB-0001")
    top = retriever.search(target.content, top_k=3)
    assert top[0].chunk_id == target.id


def test_results_carry_provenance(retriever: DenseRetriever, embedded: Embedded) -> None:
    by_id = {c.id: c for c in embedded.chunks}
    for result in retriever.search("payment service HTTP 500 errors", top_k=10):
        stored = by_id[result.chunk_id]
        assert (result.document_id, result.source_type, result.content) == (
            stored.document_id,
            stored.source_type,
            stored.content,
        )
        assert (result.service_id, result.version, result.access_level, result.file_path) == (
            stored.service_id,
            stored.version,
            stored.access_level,
            stored.file_path,
        )
        assert result.metadata["line_start"] == stored.metadata["line_start"]


@pytest.mark.parametrize(
    ("chunk_filter", "check"),
    [
        (
            ChunkFilter(services=frozenset({"payment-service"})),
            lambda r: r.service_id == "payment-service",
        ),
        (
            ChunkFilter(source_types=frozenset({SourceType.RUNBOOK})),
            lambda r: r.source_type is SourceType.RUNBOOK,
        ),
        (
            ChunkFilter(doc_types=frozenset({"architecture"})),
            lambda r: r.doc_type == "architecture",
        ),
        (ChunkFilter(versions=frozenset({"v2.8.1"})), lambda r: r.version == "v2.8.1"),
        (
            ChunkFilter(access=principal_for_role("t", "developer").chunk_access()),
            lambda r: (
                r.access_level in {AccessLevel.PUBLIC, AccessLevel.ENGINEERING}
                and r.source_type is not SourceType.RUNBOOK
            ),
        ),
        (
            ChunkFilter(
                since=datetime(2026, 6, 1, tzinfo=UTC), until=datetime(2026, 7, 1, tzinfo=UTC)
            ),
            lambda r: (
                datetime(2026, 6, 1) <= r.timestamp.replace(tzinfo=None) < datetime(2026, 7, 1)
            ),
        ),
    ],
    ids=["service", "source_type", "doc_type", "version", "access_level", "time_range"],
)
def test_every_filter_is_applied(
    retriever: DenseRetriever, chunk_filter: ChunkFilter, check
) -> None:
    results = retriever.search(
        "payment failures and database errors", top_k=20, filters=chunk_filter
    )
    assert results, "filter matched nothing"
    assert all(check(r) for r in results)


def test_filters_combine(retriever: DenseRetriever) -> None:
    chunk_filter = ChunkFilter(
        services=frozenset({"payment-service"}),
        source_types=frozenset({SourceType.INCIDENT}),
        access=principal_for_role("t", "developer").chunk_access(),
    )
    results = retriever.search("HTTP 500", top_k=15, filters=chunk_filter)
    assert results and all(
        r.service_id == "payment-service"
        and r.source_type is SourceType.INCIDENT
        and r.access_level in {AccessLevel.PUBLIC, AccessLevel.ENGINEERING}
        for r in results
    )


def test_filtered_search_matches_exhaustive_ranking(
    retriever: DenseRetriever, embedded: Embedded
) -> None:
    """The filtered top-k equals ranking every matching chunk by hand."""
    chunk_filter = ChunkFilter(
        services=frozenset({"cart-service"}), source_types=frozenset({SourceType.CODE})
    )
    query = "redis cart expiry"
    got = [r.chunk_id for r in retriever.search(query, top_k=5, filters=chunk_filter)]
    provider = embedded.provider
    q = provider.embed_query(query)
    from app.rag.embeddings.text import embedding_text

    candidates = [
        c
        for c in embedded.chunks
        if c.service_id == "cart-service" and c.source_type is SourceType.CODE
    ]
    scored = sorted(
        candidates,
        key=lambda c: (
            -sum(
                a * b
                for a, b in zip(
                    q, provider.embed_documents([embedding_text(c.model_dump())])[0], strict=True
                )
            ),
            c.id,
        ),
    )
    assert got == [c.id for c in scored[:5]]


@pytest.mark.parametrize(("query", "top_k"), [("", 5), ("   ", 5), ("ok", 0), ("ok", 101)])
def test_invalid_requests_are_rejected(retriever: DenseRetriever, query: str, top_k: int) -> None:
    with pytest.raises(RetrievalError):
        retriever.search(query, top_k=top_k)


def test_model_without_embeddings_returns_nothing(embedded: Embedded) -> None:
    other = DenseRetriever(
        embedded.engine, HashingEmbeddingProvider(dimension=256, name="never-embedded")
    )
    assert other.search("anything", top_k=5) == []


def test_retriever_family_shares_one_interface() -> None:
    for cls in (DenseRetriever, SparseRetriever, HybridRetriever, RetrievalPipeline):
        assert issubclass(cls, Retriever)
    assert issubclass(BM25Retriever, SparseRetriever)
    with pytest.raises(TypeError):  # the sparse interface is abstract
        SparseRetriever()  # type: ignore[abstract]
