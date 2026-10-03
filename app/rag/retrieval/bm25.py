"""Okapi BM25 over an in-memory inverted index. Pure Python, no database.

For a query with unique terms q and a document d of length |d|:

    score(d) = sum over q in d of  idf(q) * tf(q, d) * (k1 + 1)
                                   / (tf(q, d) + k1 * (1 - b + b * |d| / avgdl))
    idf(q)   = ln(1 + (N - df(q) + 0.5) / (df(q) + 0.5))

(the non-negative IDF variant used by Lucene). Query terms are de-duplicated;
repeating a word in the query does not multiply its weight.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class BM25Hit:
    doc_id: str
    score: float
    matched_terms: tuple[str, ...]  # query terms present in the document, in query order


class BM25Index:
    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        if k1 < 0 or not 0 <= b <= 1:
            raise ValueError("BM25 needs k1 >= 0 and 0 <= b <= 1")
        self.k1 = k1
        self.b = b
        self.doc_ids: list[str] = []
        self.doc_lengths: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = {}  # term -> [(doc index, tf)]
        self.avgdl = 0.0

    @classmethod
    def build(
        cls, documents: Iterable[tuple[str, Sequence[str]]], k1: float = 1.2, b: float = 0.75
    ) -> BM25Index:
        """``documents`` are (id, terms) pairs; ids must be unique."""
        index = cls(k1, b)
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        seen: set[str] = set()
        for doc_id, terms in documents:
            if doc_id in seen:
                raise ValueError(f"duplicate document id {doc_id!r}")
            seen.add(doc_id)
            position = len(index.doc_ids)
            index.doc_ids.append(doc_id)
            index.doc_lengths.append(len(terms))
            for term, tf in Counter(terms).items():
                postings[term].append((position, tf))
        index.postings = dict(postings)
        count = len(index.doc_ids)
        index.avgdl = sum(index.doc_lengths) / count if count else 0.0
        return index

    def __len__(self) -> int:
        return len(self.doc_ids)

    def document_frequency(self, term: str) -> int:
        return len(self.postings.get(term, ()))

    def idf(self, term: str) -> float:
        df = self.document_frequency(term)
        return math.log(1 + (len(self.doc_ids) - df + 0.5) / (df + 0.5))

    def search(
        self,
        query_terms: Sequence[str] | Mapping[str, float],
        top_k: int,
        allowed: Collection[str] | None = None,
    ) -> list[BM25Hit]:
        """Best ``top_k`` documents containing at least one query term, restricted to
        ``allowed`` ids when given. Ties are broken by document id. A mapping gives each
        term a weight (1 when a plain sequence is passed)."""
        if top_k < 1 or not self.doc_ids:
            return []
        weights = (
            dict(query_terms)
            if isinstance(query_terms, Mapping)
            else dict.fromkeys(query_terms, 1.0)
        )
        terms = list(weights)
        scores: dict[int, float] = defaultdict(float)
        matched: dict[int, list[str]] = defaultdict(list)
        allowed_set = set(allowed) if allowed is not None else None
        avgdl = self.avgdl or 1.0
        for term in terms:
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self.idf(term) * weights[term]
            for position, tf in postings:
                if allowed_set is not None and self.doc_ids[position] not in allowed_set:
                    continue
                norm = self.k1 * (1 - self.b + self.b * self.doc_lengths[position] / avgdl)
                scores[position] += idf * tf * (self.k1 + 1) / (tf + norm)
                matched[position].append(term)
        best = sorted(scores, key=lambda p: (-scores[p], self.doc_ids[p]))[:top_k]
        return [BM25Hit(self.doc_ids[p], scores[p], tuple(matched[p])) for p in best]
