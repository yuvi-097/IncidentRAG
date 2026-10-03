"""Tokenizer: identifiers survive whole, their parts are searchable, words are normalised."""

from __future__ import annotations

import pytest

from app.rag.retrieval import Tokenizer


@pytest.fixture(scope="module")
def tokenize() -> Tokenizer:
    return Tokenizer()


@pytest.mark.parametrize(
    ("text", "whole", "parts"),
    [
        ("INC-0406", "inc-0406", {"inc", "0406"}),
        ("payment-service", "payment-service", {"payment", "servic"}),
        ("PAYMENT_SERVICE_DB_POOL_SIZE", "payment_service_db_pool_size", {"pool", "size", "db"}),
        ("OrderSaga._advance", "ordersaga._advance", {"ordersaga", "order", "saga", "advanc"}),
        ("TokenBucketRateLimiter", "tokenbucketratelimiter", {"token", "bucket", "rate", "limit"}),
        ("payment_service/config.py", "payment_service/config.py", {"config.py", "config"}),
    ],
)
def test_identifiers_are_kept_whole_and_split_into_parts(
    tokenize: Tokenizer, text: str, whole: str, parts: set[str]
) -> None:
    terms = tokenize(text)
    assert terms[0] == whole
    assert parts <= set(terms)


def test_versions_match_with_or_without_v_and_produce_no_fragments(tokenize: Tokenizer) -> None:
    assert tokenize("v2.8.1") == ["v2.8.1", "2.8.1"]
    assert tokenize("2.8.1") == ["2.8.1"]
    assert "v2" not in tokenize("payment-service v2.8.1")  # would match every v2.x release


def test_words_are_stemmed_and_stop_words_dropped(tokenize: Tokenizer) -> None:
    assert tokenize("Which pull requests reduced the pool?") == ["pull", "request", "reduc", "pool"]
    assert tokenize("reduce") == tokenize("reducing") == tokenize("reduced")
    assert "not" in tokenize("OOM command not allowed")  # negations matter in errors


def test_plurals_of_numbers_and_acronyms(tokenize: Tokenizer) -> None:
    assert tokenize("HTTP 500s") == ["http", "500"]
    assert tokenize("PRs") == tokenize("PR") == ["pr"]


def test_analysis_can_be_switched_off() -> None:
    raw = Tokenizer(stemming=False, stopwords=False)
    assert raw("the connections") == ["the", "connections"]


def test_query_terms_are_a_subset_of_the_document_terms(tokenize: Tokenizer) -> None:
    document = "Set PAYMENT_SERVICE_DB_POOL_SIZE in payment_service/config.py for v2.8.1."
    for query in ("PAYMENT_SERVICE_DB_POOL_SIZE", "payment_service/config.py", "v2.8.1"):
        assert set(tokenize(query)) <= set(tokenize(document)), query


@pytest.mark.parametrize("text", ["", "   ", "?!", "a the of", "- _ /"])
def test_text_without_signal_has_no_terms(tokenize: Tokenizer, text: str) -> None:
    assert tokenize(text) == []


def test_identifier_parts_count_less_only_in_a_pure_lookup(tokenize: Tokenizer) -> None:
    lookup = tokenize.query_weights("MailRelayClient", part_weight=0.5)
    assert lookup == {"mailrelayclient": 1.0, "mail": 0.5, "relay": 0.5, "client": 0.5}
    versioned = tokenize.query_weights("IdempotencyStore.claim v2.8.1", part_weight=0.5)
    assert versioned["idempotencystore.claim"] == 1.0 and versioned["v2.8.1"] == 1.0
    assert versioned["2.8.1"] == 1.0 and versioned["claim"] == 0.5
    sentence = tokenize.query_weights("How does consume_refresh_token work?", part_weight=0.5)
    assert set(sentence.values()) == {1.0}  # words present: every term counts fully
    assert tokenize.query_weights("MailRelayClient") == dict.fromkeys(lookup, 1.0)
    assert list(tokenize.query_weights("MailRelayClient")) == tokenize("MailRelayClient")
