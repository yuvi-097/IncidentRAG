"""Combining ranked lists from several retrievers. Pure functions, no I/O.

Input: for each retriever name, its results as (chunk id, score), best first.

Reciprocal rank fusion (RRF)
    fused(c) = sum_i  w_i / (k + rank_i(c))        (lists that miss c contribute 0)
    Uses ranks only, so scores on different scales (cosine vs BM25) need no calibration.

Weighted score fusion
    fused(c) = sum_i w_i * norm_i(c) / sum_i w_i,   norm_i = min-max scaled score in list i
    Uses score gaps, not just order; sensitive to each list's score distribution.

Ties are broken by the best rank the chunk reached in any list, then by chunk id,
so fusion is deterministic.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from app.schemas.enums import FusionMethod

Ranking = Sequence[tuple[str, float]]


@dataclass(frozen=True)
class FusedHit:
    chunk_id: str
    score: float
    ranks: dict[str, int] = field(default_factory=dict)  # retriever -> 1-based rank
    scores: dict[str, float] = field(default_factory=dict)  # retriever -> its own score


def _validate(rankings: Mapping[str, Ranking], weights: Mapping[str, float]) -> None:
    missing = set(rankings) - set(weights)
    if missing:
        raise ValueError(f"no weight for: {', '.join(sorted(missing))}")
    if any(weight < 0 for weight in weights.values()):
        raise ValueError("fusion weights must be >= 0")
    if rankings and not any(weights[name] > 0 for name in rankings):
        raise ValueError("at least one fusion weight must be > 0")


def _positions(rankings: Mapping[str, Ranking]) -> dict[str, dict[str, tuple[int, float]]]:
    """retriever -> chunk -> (rank, score); a chunk listed twice keeps its first rank."""
    positions: dict[str, dict[str, tuple[int, float]]] = {}
    for name, ranking in rankings.items():
        seen: dict[str, tuple[int, float]] = {}
        for chunk_id, score in ranking:
            if chunk_id not in seen:
                seen[chunk_id] = (len(seen) + 1, score)
        positions[name] = seen
    return positions


def _combine(
    positions: dict[str, dict[str, tuple[int, float]]], fused: dict[str, float]
) -> list[FusedHit]:
    hits = [
        FusedHit(
            chunk_id=chunk_id,
            score=score,
            ranks={n: p[chunk_id][0] for n, p in positions.items() if chunk_id in p},
            scores={n: p[chunk_id][1] for n, p in positions.items() if chunk_id in p},
        )
        for chunk_id, score in fused.items()
    ]
    return sorted(hits, key=lambda h: (-h.score, min(h.ranks.values()), h.chunk_id))


def reciprocal_rank_fusion(
    rankings: Mapping[str, Ranking], weights: Mapping[str, float], k: int = 60
) -> list[FusedHit]:
    if k < 1:
        raise ValueError("RRF k must be >= 1")
    _validate(rankings, weights)
    positions = _positions(rankings)
    fused: dict[str, float] = {}
    for name, entries in positions.items():
        for chunk_id, (rank, _) in entries.items():
            fused[chunk_id] = fused.get(chunk_id, 0.0) + weights[name] / (k + rank)
    return _combine(positions, fused)


def weighted_score_fusion(
    rankings: Mapping[str, Ranking], weights: Mapping[str, float]
) -> list[FusedHit]:
    _validate(rankings, weights)
    positions = _positions(rankings)
    total = sum(weights[name] for name in rankings) or 1.0
    fused: dict[str, float] = {}
    for name, entries in positions.items():
        if not entries:
            continue
        values = [score for _, score in entries.values()]
        low, high = min(values), max(values)
        for chunk_id, (_, score) in entries.items():
            norm = (score - low) / (high - low) if high > low else 1.0
            fused[chunk_id] = fused.get(chunk_id, 0.0) + weights[name] * norm / total
    return _combine(positions, fused)


def fuse(
    method: FusionMethod,
    rankings: Mapping[str, Ranking],
    weights: Mapping[str, float],
    rrf_k: int = 60,
) -> list[FusedHit]:
    if method == FusionMethod.RRF:
        return reciprocal_rank_fusion(rankings, weights, rrf_k)
    if method == FusionMethod.WEIGHTED:
        return weighted_score_fusion(rankings, weights)
    raise ValueError(f"unknown fusion method: {method!r}")
