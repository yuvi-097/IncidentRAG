"""Text analysis for lexical retrieval.

Operations text is full of identifiers: ``INC-0406``, ``v2.8.1``, ``payment-service``,
``PAYMENT_SERVICE_DB_POOL_SIZE``, ``OrderSaga._advance``, ``payment_service/config.py``.
A plain word tokenizer shreds them into fragments that match everywhere. So every
identifier produces:

- the whole identifier, lower-cased and unstemmed (exact matches score highest,
  because whole identifiers are rare and rare terms get a high IDF);
- its path segments (``payment_service/config.py`` -> ``payment_service``, ``config.py``);
- its parts, split at ``_ . - / :`` and at camelCase boundaries, analysed like words;
- for versions, only the number without the ``v`` (``v2.8.1`` -> ``2.8.1``), not parts.

Plain words are lower-cased, stop words dropped and (optionally) stemmed with
Snowball, so "reduced", "reduces" and "reduce" match. Queries and documents go
through the same analysis.

In a *query*, the parts and segments of an identifier are fallbacks for the whole:
``query_weights`` gives them a lower weight, so a text that only shares fragments
("MailRelay" for ``MailRelayClient``) cannot outrank one that contains the identifier.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from functools import lru_cache

import snowballstemmer

_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[._\-/:]+[A-Za-z0-9]+)*")
_SEGMENT_SEPARATORS = re.compile(r"[/:]+")
_PART_SEPARATORS = re.compile(r"[._\-]+")
_CAMEL_PART = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")
_CAMEL_CASE = re.compile(r"[a-z][A-Z]|[A-Z]{2}[a-z]")
_VERSION = re.compile(r"v?\d+(?:\.\d+)+")
_PLURAL_NUMBER = re.compile(r"(\d+)s")  # "500s" -> "500"
_PLURAL_ACRONYM = re.compile(r"[A-Z]{2,}s")  # "PRs" -> "PR", "APIs" -> "API"

# Function words only. Negations ("not", "no") are kept: they matter in error messages.
_STOP_WORDS_TEXT = """
    a an and are as at be been being but by can could did do does doing for from had
    has have having he her here hers him his how i if in into is it its itself me my of
    on or our ours out over she should so than that the their theirs them then there
    these they this those through to too under until up us very was we were what when
    where which while who whom why will with would you your yours about after again all
    also am any because before between both during each few further just more most
    other own same some such only once off
"""
STOP_WORDS = frozenset(_STOP_WORDS_TEXT.split())


WORD, WHOLE, PART = "word", "whole", "part"  # kinds of term


@lru_cache(maxsize=1)
def _stemmer() -> snowballstemmer.stemmer:
    return snowballstemmer.stemmer("english")


class Tokenizer:
    def __init__(self, stemming: bool = True, stopwords: bool = True) -> None:
        self.stemming = stemming
        self.stopwords = stopwords
        self._cache: dict[str, str | None] = {}

    def __call__(self, text: str) -> list[str]:
        return self.tokenize(text)

    def tokenize(self, text: str) -> list[str]:
        return [term for term, _ in self._terms(text)]

    def query_weights(self, text: str, part_weight: float = 1.0) -> dict[str, float]:
        """Query terms and their weights. In a pure identifier lookup
        ("MailRelayClient", "IdempotencyStore.claim") the parts and segments of each
        identifier get ``part_weight``; in any query with plain words every term gets 1,
        because there the parts carry meaning of their own."""
        terms = list(self._terms(text))
        lookup = bool(terms) and all(kind != WORD for _, kind in terms)
        weights: dict[str, float] = {}
        for term, kind in terms:
            weight = part_weight if kind == PART and lookup else 1.0
            weights[term] = max(weights.get(term, 0.0), weight)
        return weights

    def _terms(self, text: str) -> Iterator[tuple[str, str]]:
        """(term, kind): a plain WORD, a WHOLE identifier or version, or a PART of one."""
        for match in _TOKEN.finditer(text):
            raw = match.group()
            if _PLURAL_ACRONYM.fullmatch(raw):
                raw = raw[:-1]
            compound = _PART_SEPARATORS.search(raw) or _SEGMENT_SEPARATORS.search(raw)
            if not compound and not _CAMEL_CASE.search(raw):
                word = self._word(raw)
                if word:
                    yield word, WORD
                continue
            whole = raw.lower()
            yield whole, WHOLE
            if _VERSION.fullmatch(whole):  # parts of a version ("v2", "8") match everything
                if whole.startswith("v"):
                    yield whole[1:], WHOLE  # the same version, written without the v
                continue
            for segment in _SEGMENT_SEPARATORS.split(raw):
                if segment != raw and _PART_SEPARATORS.search(segment):
                    yield segment.lower(), PART
                for part in _PART_SEPARATORS.split(segment):
                    for term in self._part(part, whole):
                        yield term, PART

    def _part(self, part: str, whole: str) -> Iterator[str]:
        if _CAMEL_CASE.search(part):
            if part.lower() != whole:
                yield part.lower()
            for piece in _CAMEL_PART.findall(part):
                word = self._word(piece)
                if word:
                    yield word
            return
        word = self._word(part)
        if word:
            yield word

    def _word(self, word: str) -> str | None:
        """Normalise one plain word; None if it carries no signal."""
        if word in self._cache:
            return self._cache[word]
        lower = word.lower()
        result: str | None = lower
        plural_number = _PLURAL_NUMBER.fullmatch(lower)
        if plural_number:
            result = plural_number.group(1)
        elif len(lower) < 2 or (self.stopwords and lower in STOP_WORDS):
            result = None
        elif self.stemming and lower.isalpha():
            result = _stemmer().stemWord(lower)
        if len(self._cache) < 200_000:
            self._cache[word] = result
        return result
