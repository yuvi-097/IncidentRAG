"""Phase 9: disagreeing sources are found, compared by time, and all shown."""

from __future__ import annotations

from datetime import UTC, datetime

from app.agents.conflicts import assertions, find_conflicts, relevant
from app.agents.state import AgentState, EvidenceItem, EvidenceKind
from app.schemas.enums import AccessLevel
from tests.agents.conftest import Ask, response
from tests.tools.conftest import ToolEnv


def item(
    source: str,
    text: str,
    when: datetime | None,
    kind: EvidenceKind = EvidenceKind.DOCUMENT,
    service: str | None = "payment-service",
    label: str | None = None,
    **facts: str,
) -> EvidenceItem:
    return EvidenceItem(
        kind=kind,
        source_id=source,
        title=source,
        text=text,
        service_id=service,
        timestamp=when,
        access_level=AccessLevel.ENGINEERING,
        tool="test",
        label=label or source,
        facts=facts,
    )


OLD, NEW = datetime(2025, 10, 1, tzinfo=UTC), datetime(2026, 3, 1, tzinfo=UTC)


def test_values_are_read_from_config_tables_diffs_and_prose() -> None:
    yaml = item(
        "CF-1", 'env:\n  PAYMENT_SERVICE_HTTP_TIMEOUT_SECONDS: "2.5"', NEW, EvidenceKind.CODE
    )
    (a,) = assertions(yaml)
    assert (a.service_id, a.key, a.value, a.in_effect) == (
        "payment-service",
        "HTTP_TIMEOUT_SECONDS",
        "2.5",
        True,
    )
    table = item("DOC-1", "| `CART_SERVICE_HTTP_MAX_RETRIES` | 2 | Retries |", NEW)
    (b,) = assertions(table)
    assert (b.service_id, b.key, b.value) == ("cart-service", "HTTP_MAX_RETRIES", "2")
    diff = item(
        "PR-1",
        '-  X_SERVICE_POOL_SIZE: "25"\n+  X_SERVICE_POOL_SIZE: "20"',
        NEW,
        EvidenceKind.PULL_REQUEST,
    )
    (c,) = assertions(diff)  # the "-" line is the old value, not a statement about now
    assert c.value == "20" and c.context == "diff (+)"
    prose = item(
        "PM-1", "`HTTP_TIMEOUT_SECONDS` dropped from 2.5s to 620ms.", OLD, EvidenceKind.POSTMORTEM
    )
    (d,) = assertions(prose)
    assert d.value == "0.62"  # units normalised for *_SECONDS settings


def test_a_postmortems_root_cause_change_is_not_in_effect() -> None:
    postmortem = item(
        "PM-1",
        "## Root cause\n```diff\n"
        "-JWKS_CACHE_TTL_SECONDS = 300\n+JWKS_CACHE_TTL_SECONDS = 86400\n```",
        NEW,
        EvidenceKind.POSTMORTEM,
        service="api-gateway",
    )
    (a,) = assertions(postmortem)
    assert a.value == "86400" and not a.in_effect


def test_a_conflict_lists_every_value_and_prefers_the_newest_in_effect() -> None:
    items = [
        item(
            "DOC-1", "| `PAYMENT_SERVICE_HTTP_TIMEOUT_SECONDS` | 2.5 | Per call |", OLD, label="E1"
        ),
        item(
            "PM-9",
            "`HTTP_TIMEOUT_SECONDS` dropped from 2.5s to 0.62s.",
            NEW,
            EvidenceKind.POSTMORTEM,
            label="E2",
        ),
        item(
            "CF-7",
            'PAYMENT_SERVICE_HTTP_TIMEOUT_SECONDS: "3.0"',
            datetime(2025, 12, 1, tzinfo=UTC),
            EvidenceKind.CODE,
            label="E3",
        ),
    ]
    items[1] = items[1].model_copy(update={"text": "## Root cause\n" + items[1].text})
    (conflict,) = find_conflicts(items)
    assert conflict.setting == "payment-service HTTP_TIMEOUT_SECONDS"
    assert [v.value for v in conflict.values] == ["0.62", "3", "2.5"]  # newest first
    assert conflict.preferred == "3"  # 0.62 is newer, but describes the change that broke it
    text = conflict.describe()
    for value, label in (("0.62", "[E2]"), ("3", "[E3]"), ("2.5", "[E1]")):
        assert value in text and label in text  # nothing is discarded
    assert "no longer in effect" in text


def test_agreeing_sources_and_single_sources_are_not_conflicts() -> None:
    same = [
        item("DOC-1", "| `PAYMENT_SERVICE_HTTP_MAX_RETRIES` | 2 | x |", OLD),
        item("CF-1", 'PAYMENT_SERVICE_HTTP_MAX_RETRIES: "2"', NEW, EvidenceKind.CODE),
    ]
    assert find_conflicts(same) == []
    one_source = [item("DOC-2", "A_SERVICE_MODE: fast\nA_SERVICE_MODE: slow", NEW)]
    assert find_conflicts(one_source) == []


def test_two_changes_to_one_file_are_two_sources_named_by_path() -> None:
    # The cause and the fix of an incident both edit the same manifest (same file id).
    cause = item(
        "CF-0068",
        'Changed lines:\n+ CART_SERVICE_HTTP_TIMEOUT_SECONDS: "0.25"',
        OLD,
        EvidenceKind.CODE,
        label="E1",
    )
    fix = item(
        "CF-0068",
        'Changed lines:\n+ CART_SERVICE_HTTP_TIMEOUT_SECONDS: "1.0"',
        NEW,
        EvidenceKind.CODE,
        label="E2",
    )
    cause = cause.model_copy(
        update={"chunk_id": "PR-1:CF-0068", "title": "deploy/cart.yaml in PR-1"}
    )
    fix = fix.model_copy(update={"chunk_id": "PR-2:CF-0068", "title": "deploy/cart.yaml in PR-2"})
    (conflict,) = find_conflicts([cause, fix])
    assert conflict.preferred == "1"  # the later change
    text = conflict.describe()
    assert "deploy/cart.yaml in PR-2" in text and "deploy/cart.yaml in PR-1" in text
    assert "CF-0068" not in text and "UTC" in text
    assert conflict.values[0].source_ids == ["CF-0068"]


def test_undated_or_invalid_values_give_no_preference() -> None:
    items = [
        item("DOC-1", "X_SERVICE_LIMIT = 5", None),
        item("PR-2", "+X_SERVICE_LIMIT = 9", NEW, EvidenceKind.PULL_REQUEST, valid="false"),
    ]
    (conflict,) = find_conflicts(items)
    assert conflict.preferred is None and "neither value can be preferred" in conflict.describe()


def test_relevance_to_the_question() -> None:
    items = [
        item("DOC-1", "| `PAYMENT_SERVICE_HTTP_TIMEOUT_SECONDS` | 2.5 | x |", OLD),
        item("CF-1", 'PAYMENT_SERVICE_HTTP_TIMEOUT_SECONDS: "0.62"', NEW, EvidenceKind.CODE),
    ]
    (conflict,) = find_conflicts(items)
    assert relevant(conflict, "What is the payment-service timeout?", "", ["payment-service"])
    assert relevant(conflict, "What changed?", "HTTP_TIMEOUT_SECONDS is 2.5", [])
    assert not relevant(conflict, "What is the cart-service timeout?", "", ["cart-service"])
    assert not relevant(conflict, "How do I restart it?", "", [])


# --- end to end, on the corpus ------------------------------------------------------------


def test_real_disagreement_is_shown_with_dates_and_caps_confidence(ask: Ask) -> None:
    # The corpus disagrees on payment-service's HTTP timeout: PM-0003 describes the
    # change to 0.62s that caused INC-0033; later sources state 2.5.
    state: AgentState = ask("How did the payment-service HTTP timeout change in PM-0003?")
    assert state.conflicts, state.final_answer
    conflict = next(c for c in state.conflicts if c.key == "HTTP_TIMEOUT_SECONDS")
    values = {v.value for v in conflict.values}
    assert {"0.62", "2.5"} <= values
    assert conflict.preferred == "2.5"
    assert "Sources disagree on payment-service HTTP_TIMEOUT_SECONDS" in state.final_answer
    assert state.confidence.value in {"MEDIUM", "LOW"}
    assert any("Sources disagree on" in note for note in state.limitations)
    body = response(state).model_dump(mode="json")
    assert body["conflicts"] and body["conflicts"][0]["setting"].endswith("HTTP_TIMEOUT_SECONDS")
    cited = {c.label for c in state.citations}
    assert all(label in cited for v in conflict.values for label in v.labels)


def test_the_corpus_conflicts_are_real(tool_env: ToolEnv) -> None:
    """The disagreement above is in the generated data, not constructed by the test."""
    docs = {d.id: d for d in tool_env.dataset.documents}
    assert "to 0.62s" in docs["PM-0003"].content
    assert 'PAYMENT_SERVICE_HTTP_TIMEOUT_SECONDS: "2.5"' in docs["PM-0028"].content
