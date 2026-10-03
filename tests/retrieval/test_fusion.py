"""Rank fusion arithmetic (pure functions, no retrievers)."""

from __future__ import annotations

import pytest

from app.rag.retrieval import fuse, reciprocal_rank_fusion, weighted_score_fusion
from app.schemas.enums import FusionMethod

DENSE = [("a", 0.9), ("b", 0.8), ("c", 0.1)]
SPARSE = [("c", 12.0), ("a", 6.0), ("d", 3.0)]
EQUAL = {"dense": 1.0, "sparse": 1.0}


def test_rrf_sums_weighted_reciprocal_ranks() -> None:
    fused = reciprocal_rank_fusion({"dense": DENSE, "sparse": SPARSE}, EQUAL, k=60)
    scores = {h.chunk_id: h.score for h in fused}
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["c"] == pytest.approx(1 / 63 + 1 / 61)
    assert scores["b"] == pytest.approx(1 / 62) and scores["d"] == pytest.approx(1 / 63)
    assert [h.chunk_id for h in fused] == ["a", "c", "b", "d"]
    assert fused[0].ranks == {"dense": 1, "sparse": 2}
    assert fused[0].scores == {"dense": 0.9, "sparse": 6.0}


def test_weights_shift_the_ranking() -> None:
    rankings = {"dense": DENSE, "sparse": SPARSE}
    dense_heavy = reciprocal_rank_fusion(rankings, {"dense": 1, "sparse": 0.01})
    sparse_heavy = reciprocal_rank_fusion(rankings, {"dense": 0.01, "sparse": 1})
    # a dominant weight reproduces that list's order; the other list only adds the rest
    assert [h.chunk_id for h in dense_heavy] == ["a", "b", "c", "d"]
    assert [h.chunk_id for h in sparse_heavy] == ["c", "a", "d", "b"]


def test_zero_weight_list_does_not_change_the_order() -> None:
    fused = reciprocal_rank_fusion({"dense": DENSE, "sparse": SPARSE}, {"dense": 0, "sparse": 1})
    assert [h.chunk_id for h in fused if h.score > 0] == ["c", "a", "d"]


def test_weighted_fusion_uses_min_max_scores() -> None:
    fused = weighted_score_fusion({"dense": DENSE, "sparse": SPARSE}, EQUAL)
    scores = {h.chunk_id: h.score for h in fused}
    # dense: a=1, b=0.875, c=0; sparse: c=1, a=(6-3)/9, d=0; each weighted 1/2
    assert scores["a"] == pytest.approx((1 + 3 / 9) / 2)
    assert scores["b"] == pytest.approx(0.875 / 2)
    assert scores["c"] == pytest.approx(0.5) and scores["d"] == pytest.approx(0.0)
    single = weighted_score_fusion({"dense": [("x", 0.3)]}, {"dense": 1.0})
    assert single[0].score == 1.0  # one result (no spread) normalises to 1


def test_ties_break_by_best_rank_then_id() -> None:
    fused = reciprocal_rank_fusion(
        {"one": [("b", 1), ("a", 1)], "two": [("a", 1), ("b", 1)]}, {"one": 1, "two": 1}
    )
    assert fused[0].score == fused[1].score
    assert [h.chunk_id for h in fused] == ["a", "b"]


def test_duplicates_in_a_list_keep_their_first_rank() -> None:
    fused = reciprocal_rank_fusion({"dense": [("a", 0.9), ("a", 0.5), ("b", 0.4)]}, {"dense": 1})
    assert {h.chunk_id: h.ranks["dense"] for h in fused} == {"a": 1, "b": 2}


def test_empty_rankings_fuse_to_nothing() -> None:
    assert reciprocal_rank_fusion({"dense": [], "sparse": []}, EQUAL) == []
    assert weighted_score_fusion({}, {}) == []


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        ({"dense": 1.0}, "no weight"),
        ({"dense": -1.0, "sparse": 1.0}, ">= 0"),
        ({"dense": 0.0, "sparse": 0.0}, "> 0"),
    ],
)
def test_invalid_weights_are_rejected(weights: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        reciprocal_rank_fusion({"dense": DENSE, "sparse": SPARSE}, weights)
    with pytest.raises(ValueError, match=message):
        weighted_score_fusion({"dense": DENSE, "sparse": SPARSE}, weights)


def test_fuse_dispatches_on_the_configured_method() -> None:
    rankings = {"dense": DENSE, "sparse": SPARSE}
    assert fuse(FusionMethod.RRF, rankings, EQUAL, 10) == reciprocal_rank_fusion(
        rankings, EQUAL, 10
    )
    assert fuse(FusionMethod.WEIGHTED, rankings, EQUAL) == weighted_score_fusion(rankings, EQUAL)
    with pytest.raises(ValueError, match="k must be"):
        reciprocal_rank_fusion(rankings, EQUAL, k=0)
