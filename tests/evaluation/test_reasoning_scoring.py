"""The Phase 9 benchmark scorer: what counts as a correct answer."""

from __future__ import annotations

from app.evaluation.reasoning import (
    MultiHopQuestion,
    Scored,
    TemporalQuestion,
    mentioned,
    names_a_cause,
    score_multihop,
    score_temporal,
    summarize,
)


def temporal(
    expected: list[str], match: str = "first", target: str = "deployment"
) -> TemporalQuestion:
    return TemporalQuestion(
        id="T", relation="r", question="Which deployment preceded INC-0001 (DEP-0009)?",
        target=target, expected=expected, match=match,  # type: ignore[arg-type]
    )  # fmt: skip


def test_ids_named_in_the_question_are_ignored() -> None:
    assert mentioned("deployment", "DEP-0009 then DEP-0002", "after DEP-0009?") == ["DEP-0002"]


def test_first_match_needs_the_gold_id_first() -> None:
    assert score_temporal(temporal(["DEP-0002"]), "DEP-0002, then DEP-0003")[0]
    assert not score_temporal(temporal(["DEP-0002"]), "DEP-0003, then DEP-0002")[0]
    assert not score_temporal(temporal(["DEP-0002"]), "no deployment")[0]
    assert score_temporal(temporal(["DEP-0002"]), "DEP-0009 and DEP-0002")[0]  # anchor ignored


def test_set_match_needs_exactly_the_gold_set() -> None:
    q = temporal(["DEP-0002", "DEP-0003"], match="set")
    ok, detail = score_temporal(q, "DEP-0003; DEP-0002")
    assert ok and detail["precision"] == detail["recall"] == 1.0
    ok, detail = score_temporal(q, "DEP-0002; DEP-0003; DEP-0004")
    assert not ok and detail["precision"] < 1 and detail["recall"] == 1.0


def test_versions_are_scored_as_versions() -> None:
    assert score_temporal(temporal(["v1.2.3"], target="version"), "DEP-0002, api v1.2.3")[0]


def multihop(expected: dict[str, object], hops: list[str]) -> MultiHopQuestion:
    return MultiHopQuestion(
        id="M", kind="k", question="What caused INC-0001?", expected=expected, hops=hops
    )


def test_each_hop_is_checked_on_its_own() -> None:
    q = multihop(
        {"deployment": "DEP-0002", "commit": "abc1234", "file": "a/b.py", "change": ["x =  1"]},
        ["deployment", "commit", "file", "change"],
    )
    ok, detail = score_multihop(q, "DEP-0002, commit ABC1234; a/b.py: - x = 1")
    assert ok and all(detail["hops"].values())
    ok, detail = score_multihop(q, "DEP-0003, commit abc1234; a/b.py")
    assert not ok and detail["hops"] == {
        "deployment": False,
        "commit": True,
        "file": True,
        "change": False,
    }


def test_stress_hops() -> None:
    q = multihop(
        {"files": ["a.py", "b.py"], "key": ["X_TIMEOUT"], "author": "kim"},
        ["files", "key", "author"],
    )
    assert score_multihop(q, "a.py and b.py changed X_TIMEOUT, by kim")[0]
    assert not score_multihop(q, "a.py changed X_TIMEOUT_MS, by kim")[0]


def test_no_cause_needs_an_explicit_statement_and_no_named_cause() -> None:
    q = multihop({"no_cause": True}, ["no_cause"])
    assert score_multihop(
        q, "No deployment is recorded as the cause of INC-0001; DEP-0002 was live."
    )[0]
    assert not score_multihop(q, "INC-0001 was caused by DEP-0002.")[0]
    assert not score_multihop(q, "INC-0001: a DNS problem.")[0]  # silent is not enough
    assert names_a_cause("The deployment behind it was DEP-0002.")
    assert not names_a_cause("DEP-0002 did not cause it.")


def test_summaries_count_groups_and_hops() -> None:
    rows = [
        Scored(
            id="1",
            group="a",
            question="q",
            correct=True,
            detail={"hops": {"file": True}},
            answer="",
            tools=[],
        ),
        Scored(
            id="2",
            group="a",
            question="q",
            correct=False,
            detail={"hops": {"file": False}},
            answer="",
            tools=[],
        ),
    ]
    summary = summarize(rows)
    assert summary["correct"] == 1 and summary["accuracy"] == 0.5
    assert summary["by_group"]["a"] == {"questions": 2, "correct": 1}
    assert summary["by_hop"]["file"] == {"checked": 2, "correct": 1}
