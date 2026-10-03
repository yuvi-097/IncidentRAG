"""The reranker in isolation: hand-built candidates and a fake pair scorer.

No database, no retriever, no model (except the opt-in test with the real
cross-encoder at the end)."""

from __future__ import annotations

import os
import sys
import types
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, ClassVar

import pytest

from app.config import RerankerSettings
from app.rag.reranking import (
    CrossEncoderReranker,
    RerankerError,
    SentenceTransformerCrossEncoder,
    build_reranker,
    rerank_text,
)
from app.rag.retrieval import RetrievedChunk
from app.schemas.enums import AccessLevel, SourceType


def chunk(chunk_id: str, content: str, rank: int, title: str = "Doc") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=chunk_id.split("#")[0],
        source_type=SourceType.RUNBOOK,
        title=title,
        section=None,
        content=content,
        service_id="payment-service",
        timestamp=datetime(2026, 6, 1, tzinfo=UTC),
        access_level=AccessLevel.ENGINEERING,
        version=None,
        doc_type="runbook",
        file_path=None,
        metadata={"lines": [1, 2]},
        score=1.0 / rank,
        rank=rank,
        retriever="hybrid",
        score_details={"dense_rank": rank},
    )


class KeywordScorer:
    """Scores a passage by how many query words it contains; records its input."""

    name = "keyword"

    def __init__(self) -> None:
        self.pairs: list[tuple[str, str]] = []

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        self.pairs.extend(pairs)
        return [
            float(sum(word in passage.lower() for word in query.lower().split()))
            for query, passage in pairs
        ]


CANDIDATES = [
    chunk("A#000", "kafka consumer lag", 1),
    chunk("B#000", "database connection pool exhausted", 2),
    chunk("C#000", "pool size tuning", 3),
    chunk("D#000", "cache ttl", 4),
]


def test_candidates_are_reordered_by_the_scorer() -> None:
    reranked = CrossEncoderReranker(KeywordScorer()).rerank(
        "connection pool exhausted", CANDIDATES, top_k=3
    )
    assert [r.chunk_id for r in reranked] == ["B#000", "C#000", "A#000"]
    assert [r.rank for r in reranked] == [1, 2, 3]
    assert [r.score for r in reranked] == [3.0, 1.0, 0.0]


def test_reranking_keeps_provenance_and_records_evidence() -> None:
    reranked = CrossEncoderReranker(KeywordScorer()).rerank("pool", CANDIDATES, top_k=4)
    originals = {c.chunk_id: c for c in CANDIDATES}
    for result in reranked:
        original = originals[result.chunk_id]
        unchanged = ("document_id", "content", "access_level", "metadata", "timestamp", "title")
        assert all(getattr(result, f) == getattr(original, f) for f in unchanged)
        assert result.retriever == "hybrid+rerank"
        assert result.score_details["first_stage_rank"] == original.rank
        assert result.score_details["dense_rank"] == original.rank  # earlier evidence kept
        assert result.score_details["rerank_score"] == result.score


def test_only_candidates_are_returned() -> None:
    reranked = CrossEncoderReranker(KeywordScorer()).rerank("anything", CANDIDATES, top_k=10)
    assert {r.chunk_id for r in reranked} == {c.chunk_id for c in CANDIDATES}


def test_ties_keep_the_first_stage_order() -> None:
    reranked = CrossEncoderReranker(KeywordScorer()).rerank("zzz", CANDIDATES, top_k=4)
    assert [r.chunk_id for r in reranked] == ["A#000", "B#000", "C#000", "D#000"]


def test_duplicate_candidates_are_scored_once() -> None:
    scorer = KeywordScorer()
    reranked = CrossEncoderReranker(scorer).rerank("pool", [*CANDIDATES, CANDIDATES[1]], top_k=10)
    assert len(scorer.pairs) == 4 and len(reranked) == 4


def test_the_scorer_sees_the_query_and_the_passage_with_its_header() -> None:
    scorer = KeywordScorer()
    titled = chunk("E#000", "raise the pool size", 1, title="Pool Exhaustion")
    CrossEncoderReranker(scorer).rerank("pool", [titled], top_k=1)
    query, passage = scorer.pairs[0]
    assert query == "pool"
    assert passage == rerank_text(titled)
    assert "Runbook: Pool Exhaustion" in passage and "raise the pool size" in passage


def test_edge_cases() -> None:
    reranker = CrossEncoderReranker(KeywordScorer())
    assert reranker.rerank("pool", [], top_k=5) == []
    with pytest.raises(RerankerError, match="top_k"):
        reranker.rerank("pool", CANDIDATES, top_k=0)

    class Broken:
        name = "broken"

        def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
            return [1.0]

    with pytest.raises(RerankerError, match="1 scores for 4 pairs"):
        CrossEncoderReranker(Broken()).rerank("pool", CANDIDATES, top_k=2)


class _FakeCrossEncoder:
    """Stands in for sentence_transformers.CrossEncoder (no model download)."""

    loads: ClassVar[list[dict[str, Any]]] = []

    def __init__(self, model: str, device: str = "cpu", local_files_only: bool = False) -> None:
        self.loads.append({"model": model, "local_files_only": local_files_only})
        if model == "not-cached" and local_files_only:
            raise OSError("not in the local cache")
        self.max_seq_length = 512

    def predict(self, pairs: list[list[str]], **kwargs: Any) -> list[float]:
        return [float(len(passage)) for _, passage in pairs]


@pytest.fixture
def fake_cross_encoder(monkeypatch: pytest.MonkeyPatch) -> type[_FakeCrossEncoder]:
    module = types.ModuleType("sentence_transformers")
    module.CrossEncoder = _FakeCrossEncoder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    _FakeCrossEncoder.loads = []
    return _FakeCrossEncoder


def test_build_reranker_from_settings(fake_cross_encoder: type[_FakeCrossEncoder]) -> None:
    settings = RerankerSettings(_env_file=None, model="tiny-ce", max_length=128)  # type: ignore[call-arg]
    reranker = build_reranker(settings)
    assert isinstance(reranker, CrossEncoderReranker)
    assert reranker.scorer.name == "tiny-ce"
    assert fake_cross_encoder.loads == [{"model": "tiny-ce", "local_files_only": True}]
    scorer = reranker.scorer
    assert isinstance(scorer, SentenceTransformerCrossEncoder)
    assert scorer._model.max_seq_length == 128
    assert scorer.score([("q", "ab"), ("q", "abcd")]) == [2.0, 4.0]


def test_uncached_models_fall_back_to_a_download(
    fake_cross_encoder: type[_FakeCrossEncoder],
) -> None:
    SentenceTransformerCrossEncoder("not-cached")
    assert [load["local_files_only"] for load in fake_cross_encoder.loads] == [True, False]


def test_reranking_can_be_disabled_and_unknown_providers_fail() -> None:
    assert build_reranker(RerankerSettings(_env_file=None, provider="none")) is None  # type: ignore[call-arg]
    with pytest.raises(RerankerError, match="unknown RERANKER_PROVIDER"):
        build_reranker(RerankerSettings(_env_file=None, provider="magic"))  # type: ignore[call-arg]


@pytest.mark.model
@pytest.mark.skipif(
    os.getenv("OPSRAG_RUN_MODEL_TESTS") != "1",
    reason="set OPSRAG_RUN_MODEL_TESTS=1 to run tests with the real reranker model",
)
def test_real_cross_encoder_prefers_the_relevant_passage() -> None:
    reranker = build_reranker(RerankerSettings(_env_file=None))  # type: ignore[call-arg]
    assert reranker is not None
    candidates = [
        chunk("K#000", "Consumer group lag grows when partitions are rebalanced.", 1, "Kafka Lag"),
        chunk("C#000", "The cart page shows prices in the customer's currency.", 2, "Cart UI"),
        chunk(
            "P#000",
            "When every pooled database connection is in use, requests wait and then fail. "
            "Check the pool size, max overflow and long-running transactions.",
            3,
            "Database Connection Pool Exhaustion",
        ),
    ]
    reranked = reranker.rerank("How do we handle database connection exhaustion?", candidates, 3)
    assert reranked[0].chunk_id == "P#000"
