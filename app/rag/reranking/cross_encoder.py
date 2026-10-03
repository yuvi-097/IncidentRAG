"""Cross-encoder reranking.

A cross-encoder reads the query and a passage together and outputs one relevance
score. That is more accurate than comparing two independently computed embeddings,
and far too slow to run over the whole corpus, so it only reorders the first
stage's top candidates.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from app.config import RerankerSettings
from app.rag.reranking.base import PairScorer, Reranker, RerankerError, rerank_text
from app.rag.retrieval.base import RetrievedChunk


class CrossEncoderReranker(Reranker):
    name = "cross-encoder"

    def __init__(
        self,
        scorer: PairScorer,
        passage: Callable[[RetrievedChunk], str] = rerank_text,
    ) -> None:
        self.scorer = scorer
        self.passage = passage

    def rerank(
        self, query: str, candidates: Sequence[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        if top_k < 1:
            raise RerankerError("top_k must be >= 1")
        first: dict[str, RetrievedChunk] = {}
        for candidate in candidates:
            first.setdefault(candidate.chunk_id, candidate)
        unique = list(first.values())
        if not unique:
            return []
        scores = self.scorer.score([(query, self.passage(c)) for c in unique])
        if len(scores) != len(unique):
            raise RerankerError(f"scorer returned {len(scores)} scores for {len(unique)} pairs")
        order = sorted(
            range(len(unique)),
            key=lambda i: (-float(scores[i]), unique[i].rank, unique[i].chunk_id),
        )[:top_k]
        results = []
        for new_rank, i in enumerate(order, 1):
            candidate = unique[i]
            details = {
                **candidate.score_details,
                "first_stage_rank": candidate.rank,
                "first_stage_score": candidate.score,
                "rerank_score": round(float(scores[i]), 6),
            }
            results.append(
                candidate.model_copy(
                    update={
                        "score": round(float(scores[i]), 6),
                        "rank": new_rank,
                        "retriever": f"{candidate.retriever}+rerank",
                        "score_details": details,
                    }
                )
            )
        return results


class SentenceTransformerCrossEncoder:
    """``PairScorer`` backed by a sentence-transformers ``CrossEncoder`` model."""

    def __init__(
        self, model: str, device: str = "cpu", batch_size: int = 16, max_length: int = 512
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise RerankerError("sentence-transformers is not installed") from exc
        try:
            # Local cache first (no network round-trip); download only if missing.
            self._model: Any = CrossEncoder(model, device=device, local_files_only=True)
        except Exception:
            try:
                self._model = CrossEncoder(model, device=device)
            except Exception as exc:
                raise RerankerError(f"could not load reranker model {model!r}: {exc}") from exc
        if hasattr(self._model, "max_seq_length"):
            self._model.max_seq_length = max_length
        else:  # pragma: no cover - sentence-transformers < 6
            self._model.max_length = max_length
        self.name = model
        self.batch_size = batch_size

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        scores = self._model.predict(
            [list(pair) for pair in pairs],
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [float(s) for s in scores]


RERANKERS: dict[str, Callable[[RerankerSettings], Reranker]] = {
    "cross-encoder": lambda s: CrossEncoderReranker(
        SentenceTransformerCrossEncoder(s.model, s.device, s.batch_size, s.max_length)
    ),
}


def build_reranker(settings: RerankerSettings) -> Reranker | None:
    """The configured reranker, or None when ``RERANKER_PROVIDER=none``."""
    if not settings.enabled:
        return None
    factory = RERANKERS.get(settings.provider)
    if factory is None:
        supported = ", ".join(sorted([*RERANKERS, "none"]))
        raise RerankerError(
            f"unknown RERANKER_PROVIDER {settings.provider!r}; supported: {supported}"
        )
    return factory(settings)
