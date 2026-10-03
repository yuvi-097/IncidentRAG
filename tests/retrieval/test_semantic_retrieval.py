"""Retrieval quality with the real configured models (embedder and cross-encoder).

Opt in with OPSRAG_RUN_MODEL_TESTS=1 (needs the models downloaded or downloadable;
embedding the full corpus on CPU takes about 13 minutes). Expectations live only in
these tests, as properties of the retrieved chunks' metadata; nothing about them is
given to the retrievers.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from app.config import load_settings
from app.rag.embeddings import EmbeddingPipeline, build_embedding_provider
from app.rag.retrieval import DenseRetriever, RetrievedChunk, Retriever
from app.rag.retrieval.factory import RetrievalComponents
from app.rag.store import ChunkFilter
from app.schemas.enums import RetrievalMode, SourceType
from app.synthetic.records import SyntheticDataset
from tests.retrieval.conftest import build_corpus

pytestmark = [
    pytest.mark.model,
    pytest.mark.skipif(
        os.getenv("OPSRAG_RUN_MODEL_TESTS") != "1",
        reason="set OPSRAG_RUN_MODEL_TESTS=1 to run tests with the real embedding model",
    ),
]


@pytest.fixture(scope="module")
def components(dataset: SyntheticDataset) -> Iterator[RetrievalComponents]:
    corpus = build_corpus(dataset)
    settings = load_settings(env_file=None)
    provider = build_embedding_provider(settings.embedding)
    EmbeddingPipeline(provider, settings.embedding.batch_size).run(corpus.engine)
    yield RetrievalComponents(corpus.engine, settings, embedding_provider=provider)
    corpus.engine.dispose()


@pytest.fixture(scope="module")
def retriever(components: RetrievalComponents) -> DenseRetriever:
    return components.dense


@pytest.fixture(scope="module")
def pipeline(components: RetrievalComponents) -> Retriever:
    """Dense + BM25, fused, reranked by the configured cross-encoder."""
    return components.retriever(RetrievalMode.HYBRID_RERANK)


def _top(retriever: Retriever, query: str, k: int = 5, **filters: object) -> list[RetrievedChunk]:
    return retriever.search(query, top_k=k, filters=ChunkFilter(**filters) if filters else None)


def test_why_did_payment_service_fail(retriever: DenseRetriever) -> None:
    results = _top(retriever, "Why did payment-service fail?")
    assert any(
        r.service_id == "payment-service"
        and r.source_type in {SourceType.INCIDENT, SourceType.POSTMORTEM}
        for r in results
    )


def test_how_do_we_handle_database_connection_exhaustion(retriever: DenseRetriever) -> None:
    results = _top(retriever, "How do we handle database connection exhaustion?")
    assert "Database Connection Pool Exhaustion" in {
        r.title for r in results if r.source_type is SourceType.RUNBOOK
    }


def test_where_is_payment_database_configuration(retriever: DenseRetriever) -> None:
    results = _top(retriever, "Where is payment database configuration?")
    config_sources = (
        "payment_service/config.py",
        "payment_service/db/database.py",
        "deploy/payment-service.yaml",
    )
    assert any(
        (r.file_path or "").endswith(config_sources)
        or r.title == "Payment Service Configuration Reference"
        for r in results
    )


def test_the_v281_incident_is_found(retriever: DenseRetriever) -> None:
    results = _top(
        retriever,
        "Why did payment requests start returning HTTP 500 errors after deployment v2.8.1?",
        k=10,
    )
    assert any(r.source_type is SourceType.INCIDENT and r.version == "v2.8.1" for r in results)


def test_filters_narrow_semantic_results(retriever: DenseRetriever) -> None:
    results = _top(
        retriever,
        "connection pool timeouts",
        k=10,
        services=frozenset({"payment-service"}),
        source_types=frozenset({SourceType.INCIDENT}),
    )
    assert results and all(
        r.service_id == "payment-service" and r.source_type is SourceType.INCIDENT for r in results
    )


# --- the full pipeline (dense + BM25 -> fusion -> cross-encoder) --------------------


def test_pipeline_answers_the_example_questions(pipeline: Retriever) -> None:
    payment = _top(pipeline, "Why did payment-service fail?")
    assert any(
        r.service_id == "payment-service"
        and r.source_type in {SourceType.INCIDENT, SourceType.POSTMORTEM}
        for r in payment
    )
    exhaustion = _top(pipeline, "How do we handle database connection exhaustion?")
    assert "Database Connection Pool Exhaustion" in {r.title for r in exhaustion}
    config = _top(pipeline, "Where is payment database configuration?")
    assert any(
        (r.file_path or "").endswith("payment_service/config.py")
        or r.title == "Payment Service Configuration Reference"
        for r in config
    )


def test_pipeline_semantic_code_queries(pipeline: Retriever) -> None:
    refunds = _top(pipeline, "How are refunds issued against captured payments?")
    assert any((r.file_path or "").endswith("payment_service/refunds.py") for r in refunds)
    templates = _top(pipeline, "What happens when a notification template is unknown?")
    assert any(
        (r.file_path or "").endswith("notification_service/template_renderer.py") for r in templates
    )


def test_pipeline_exact_match_queries(pipeline: Retriever) -> None:
    assert any(r.document_id in {"INC-0078", "PM-0009"} for r in _top(pipeline, "INC-0078"))
    key = _top(pipeline, "What is PAYMENT_SERVICE_HTTP_TIMEOUT_SECONDS set to?")
    assert any("PAYMENT_SERVICE_HTTP_TIMEOUT_SECONDS" in r.content for r in key)


def test_pipeline_filters_and_evidence(pipeline: Retriever) -> None:
    results = _top(
        pipeline,
        "connection pool timeouts",
        k=10,
        services=frozenset({"payment-service"}),
        source_types=frozenset({SourceType.INCIDENT}),
    )
    assert results and all(
        r.service_id == "payment-service" and r.source_type is SourceType.INCIDENT for r in results
    )
    assert all(r.retriever == "hybrid+rerank" for r in results)
    assert all("first_stage_rank" in r.score_details for r in results)
