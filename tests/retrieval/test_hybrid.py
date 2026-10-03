"""Hybrid retrieval, the retrieval pipeline and the factory, on the full corpus.

Dense uses the deterministic hashing embedder, so these tests check fusion and
pipeline mechanics (and filters, exact-match and code queries), not semantic
quality. Semantic behaviour is tested with the real models in
test_semantic_retrieval.py.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from app.config import Settings, load_settings
from app.rag.reranking import CrossEncoderReranker, Reranker
from app.rag.retrieval import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    RetrievalError,
    RetrievalPipeline,
    RetrievedChunk,
    Retriever,
    WeightedRetriever,
)
from app.rag.retrieval.factory import RetrievalComponents, build_retriever
from app.rag.store import ChunkFilter
from app.schemas.enums import FusionMethod, RetrievalMode
from app.synthetic.records import SyntheticDataset
from tests.retrieval.conftest import Embedded
from tests.retrieval.test_bm25 import FILTERS, satisfies
from tests.retrieval.test_reranker import KeywordScorer


@pytest.fixture(scope="module")
def dense(embedded: Embedded) -> DenseRetriever:
    return DenseRetriever(embedded.engine, embedded.provider)


@pytest.fixture(scope="module")
def bm25(embedded: Embedded) -> BM25Retriever:
    return BM25Retriever(embedded.engine)


def hybrid_of(
    dense: Retriever,
    bm25: Retriever,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
    method: FusionMethod = FusionMethod.RRF,
) -> HybridRetriever:
    return HybridRetriever(
        [WeightedRetriever(dense, dense_weight), WeightedRetriever(bm25, sparse_weight)],
        method=method,
        depth=50,
    )


class Counting(Retriever):
    """Wraps a retriever and counts calls."""

    def __init__(self, inner: Retriever) -> None:
        self.inner, self.name, self.calls = inner, inner.name, 0

    def search(
        self, query: str, top_k: int = 10, filters: ChunkFilter | None = None
    ) -> list[RetrievedChunk]:
        self.calls += 1
        return self.inner.search(query, top_k, filters)


QUERY = "payment-service database connection pool timeouts after v2.8.1"


def test_rrf_scores_are_recomputed_from_the_component_rankings(
    dense: DenseRetriever, bm25: BM25Retriever
) -> None:
    results = hybrid_of(dense, bm25).search(QUERY, top_k=10)
    dense_ranks = {r.chunk_id: r.rank for r in dense.search(QUERY, 50)}
    bm25_ranks = {r.chunk_id: r.rank for r in bm25.search(QUERY, 50)}
    assert [r.rank for r in results] == list(range(1, 11))
    for r in results:
        expected = sum(
            1 / (60 + ranks[r.chunk_id])
            for ranks in (dense_ranks, bm25_ranks)
            if r.chunk_id in ranks
        )
        assert r.score == pytest.approx(expected, abs=1e-6)
        assert r.retriever == "hybrid"
        assert r.score_details.get("dense_rank") == dense_ranks.get(r.chunk_id)
        assert r.score_details.get("bm25_rank") == bm25_ranks.get(r.chunk_id)
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)


def test_hybrid_keeps_provenance_and_bm25_evidence(
    dense: DenseRetriever, bm25: BM25Retriever, embedded: Embedded
) -> None:
    chunks = {c.id: c for c in embedded.chunks}
    for r in hybrid_of(dense, bm25).search(QUERY, top_k=10):
        assert (r.content, r.document_id) == (
            chunks[r.chunk_id].content,
            chunks[r.chunk_id].document_id,
        )
        if "bm25_rank" in r.score_details:
            assert r.matched_terms


def test_a_zero_weight_disables_a_retriever(dense: DenseRetriever, bm25: BM25Retriever) -> None:
    counted = Counting(dense)
    only_sparse = hybrid_of(counted, bm25, dense_weight=0.0).search(QUERY, top_k=10)
    assert counted.calls == 0
    assert [r.chunk_id for r in only_sparse] == [r.chunk_id for r in bm25.search(QUERY, 10)]


def test_weights_and_fusion_method_change_the_result(
    dense: DenseRetriever, bm25: BM25Retriever
) -> None:
    orders = {
        tuple(r.chunk_id for r in retriever.search(QUERY, top_k=10))
        for retriever in (
            hybrid_of(dense, bm25, 1.0, 0.05),
            hybrid_of(dense, bm25, 0.05, 1.0),
            hybrid_of(dense, bm25, method=FusionMethod.WEIGHTED),
        )
    }
    assert len(orders) == 3


@pytest.mark.parametrize("chunk_filter", FILTERS)
def test_every_stage_respects_filters(
    dense: DenseRetriever, bm25: BM25Retriever, chunk_filter: ChunkFilter
) -> None:
    seen: list[RetrievedChunk] = []

    class Recording(CrossEncoderReranker):
        def rerank(
            self, query: str, candidates: Sequence[RetrievedChunk], top_k: int
        ) -> list[RetrievedChunk]:
            seen.extend(candidates)
            return super().rerank(query, candidates, top_k)

    pipeline = RetrievalPipeline(hybrid_of(dense, bm25), Recording(KeywordScorer()), candidates=30)
    for retriever in (hybrid_of(dense, bm25), pipeline):
        results = retriever.search(QUERY, top_k=10, filters=chunk_filter)
        assert results and all(satisfies(r, chunk_filter) for r in results)
    assert seen and all(satisfies(c, chunk_filter) for c in seen)  # the reranker saw only allowed


def test_pipeline_reranks_the_fused_candidates(dense: DenseRetriever, bm25: BM25Retriever) -> None:
    first_stage = hybrid_of(dense, bm25)
    pipeline = RetrievalPipeline(first_stage, CrossEncoderReranker(KeywordScorer()), candidates=30)
    candidates = first_stage.search(QUERY, top_k=30)
    results = pipeline.search(QUERY, top_k=5)
    assert pipeline.name == "hybrid+rerank"
    assert len(results) == 5 and {r.chunk_id for r in results} <= {c.chunk_id for c in candidates}
    assert all(r.retriever == "hybrid+rerank" and "fused_score" in r.score_details for r in results)
    without = RetrievalPipeline(first_stage, None).search(QUERY, top_k=5)
    assert [r.chunk_id for r in without] == [c.chunk_id for c in candidates[:5]]


def test_exact_matches_reach_the_rerank_pool(
    dense: DenseRetriever, bm25: BM25Retriever, dataset: SyntheticDataset
) -> None:
    """RRF trade-off: a chunk only BM25 ranks first scores 1/61, while chunks both lists
    rank moderately score more, so a noisy dense list (here: the non-semantic hashing
    embedder) can push exact matches down. They must still reach the 30 candidates the
    reranker sees. Measured with this setup: 472/540 incidents and 264/265 file paths
    in the top 30. A zero dense weight gives exactly the BM25 order (tested above)."""
    hybrid = hybrid_of(dense, bm25)
    incidents = dataset.incidents[::10]
    found = sum(any(r.document_id == i.id for r in hybrid.search(i.id, 30)) for i in incidents)
    assert found / len(incidents) >= 0.8
    python = [c for c in dataset.code_files if c.language == "python"][::10]
    found = sum(any(r.file_path == c.path for r in hybrid.search(c.path, 30)) for c in python)
    assert found / len(python) >= 0.95


def test_sparse_weight_favours_exact_matches(
    dense: DenseRetriever, bm25: BM25Retriever, dataset: SyntheticDataset
) -> None:
    incidents = dataset.incidents[::10]

    def top5(hybrid: HybridRetriever) -> int:
        return sum(any(r.document_id == i.id for r in hybrid.search(i.id, 5)) for i in incidents)

    assert top5(hybrid_of(dense, bm25, 0.1, 1.0)) > top5(hybrid_of(dense, bm25, 1.0, 1.0))


def test_validation(dense: DenseRetriever, bm25: BM25Retriever) -> None:
    with pytest.raises(RetrievalError):
        hybrid_of(dense, bm25).search("", 5)
    with pytest.raises(ValueError, match="unique"):
        HybridRetriever([WeightedRetriever(bm25), WeightedRetriever(bm25)])
    with pytest.raises(ValueError, match="weights"):
        hybrid_of(dense, bm25, 0.0, 0.0)
    with pytest.raises(ValueError, match="at least one"):
        HybridRetriever([])
    with pytest.raises(ValueError, match="candidates"):
        RetrievalPipeline(bm25, None, candidates=0)


# --- the factory -----------------------------------------------------------------


@pytest.fixture
def hashing_settings(clean_env: pytest.MonkeyPatch) -> Settings:
    clean_env.setenv("EMBEDDING_PROVIDER", "hashing")
    clean_env.setenv("EMBEDDING_DIMENSION", "256")  # matches the `embedded` fixture's vectors
    clean_env.setenv("RERANKER_PROVIDER", "none")
    clean_env.setenv("RETRIEVAL_MODE", "hybrid")
    return load_settings(env_file=None)


class FirstStageOrder(Reranker):
    name = "identity"

    def rerank(
        self, query: str, candidates: Sequence[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        return list(candidates)[:top_k]


def test_factory_builds_every_mode_from_settings(
    embedded: Embedded, hashing_settings: Settings
) -> None:
    components = RetrievalComponents(embedded.engine, hashing_settings, reranker=FirstStageOrder())
    names = {mode: components.retriever(mode).name for mode in RetrievalMode}
    assert names == {
        RetrievalMode.DENSE: "dense",
        RetrievalMode.SPARSE: "bm25",
        RetrievalMode.HYBRID: "hybrid",
        RetrievalMode.HYBRID_RERANK: "hybrid+rerank",
    }
    for mode in RetrievalMode:
        assert components.retriever(mode).search(QUERY, top_k=5)
    assert components.retriever(RetrievalMode.HYBRID).components[0].retriever is components.dense
    default = build_retriever(embedded.engine, hashing_settings)
    assert default.name == "hybrid"


def test_factory_applies_weights_and_needs_a_reranker(
    embedded: Embedded, clean_env: pytest.MonkeyPatch
) -> None:
    clean_env.setenv("EMBEDDING_PROVIDER", "hashing")
    clean_env.setenv("EMBEDDING_DIMENSION", "256")
    clean_env.setenv("RERANKER_PROVIDER", "none")
    clean_env.setenv("RETRIEVAL_MODE", "sparse")
    clean_env.setenv("RETRIEVAL_DENSE_WEIGHT", "0")
    clean_env.setenv("RETRIEVAL_FUSION", "weighted")
    components = RetrievalComponents(embedded.engine, load_settings(env_file=None))
    hybrid = components.hybrid()
    assert [c.retriever.name for c in hybrid.components] == ["bm25"]
    assert hybrid.method is FusionMethod.WEIGHTED
    with pytest.raises(ValueError, match="RERANKER_PROVIDER"):
        components.retriever(RetrievalMode.HYBRID_RERANK)
