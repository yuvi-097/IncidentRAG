"""BM25: the scoring formula, and BM25Retriever on the full synthetic corpus.

Corpus-wide expectations are derived from the data at test time (every unique
config key, every incident id, every deployment version...), never from a fixed list
of expected answers.
"""

from __future__ import annotations

import ast
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from sqlalchemy import delete, select, update

from app.database.models import DocumentChunk
from app.rag.ingestion import ChunkRecord
from app.rag.retrieval import (
    BM25Index,
    BM25Retriever,
    RetrievalError,
    RetrievedChunk,
    lexical_text,
)
from app.rag.store import ChunkFilter
from app.schemas.enums import AccessLevel, SourceType
from app.security import principal_for_role
from app.synthetic.records import SyntheticDataset
from tests.retrieval.conftest import Corpus, Embedded, build_corpus

# --- the formula --------------------------------------------------------------------


def _reference_score(
    query: list[str], doc: list[str], docs: list[list[str]], k1: float, b: float
) -> float:
    n, avgdl = len(docs), sum(map(len, docs)) / len(docs)
    total = 0.0
    for term in dict.fromkeys(query):
        tf = doc.count(term)
        if not tf:
            continue
        df = sum(term in d for d in docs)
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        total += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(doc) / avgdl))
    return total


DOCS = {
    "a": ["pool", "exhaust", "payment", "pool"],
    "b": ["pool", "size", "config"],
    "c": ["kafka", "lag", "consum", "partit", "rebalanc", "group"],
    "d": ["payment", "timeout"],
}


@pytest.mark.parametrize(("k1", "b"), [(1.2, 0.75), (0.0, 0.75), (2.0, 0.0), (1.2, 1.0)])
def test_scores_match_the_bm25_formula(k1: float, b: float) -> None:
    index = BM25Index.build(DOCS.items(), k1=k1, b=b)
    query = ["pool", "payment", "missing", "pool"]
    hits = {h.doc_id: h.score for h in index.search(query, top_k=10)}
    for doc_id, terms in DOCS.items():
        expected = _reference_score(query, terms, list(DOCS.values()), k1, b)
        assert hits.get(doc_id, 0.0) == pytest.approx(expected)


def test_rare_terms_weigh_more_and_idf_is_never_negative() -> None:
    index = BM25Index.build(DOCS.items())
    assert index.idf("kafka") > index.idf("pool") > 0
    everywhere = BM25Index.build([("x", ["t"]), ("y", ["t"])])
    assert everywhere.idf("t") > 0


def test_term_frequency_saturates_and_length_is_normalised() -> None:
    index = BM25Index.build([("once", ["x", "y"]), ("many", ["x"] * 20 + ["y"])])
    scores = {h.doc_id: h.score for h in index.search(["x"], 2)}
    assert scores["many"] < 3 * scores["once"]  # 10x the occurrences, far from 10x the score
    no_tf = BM25Index.build([("once", ["x"]), ("many", ["x", "x", "x"])], k1=0.0)
    assert len({h.score for h in no_tf.search(["x"], 2)}) == 1  # k1 = 0 ignores tf


def test_search_limits_ties_matched_terms_and_allowed_ids() -> None:
    index = BM25Index.build([("b", ["x"]), ("a", ["x"]), ("c", ["y"])])
    assert [h.doc_id for h in index.search(["x"], 5)] == ["a", "b"]  # tie: by id
    assert [h.doc_id for h in index.search(["x"], 1)] == ["a"]
    assert [h.doc_id for h in index.search(["x"], 5, allowed={"b", "c"})] == ["b"]
    hits = {h.doc_id: h.matched_terms for h in index.search(["x", "y", "x"], 5)}
    assert hits == {"a": ("x",), "b": ("x",), "c": ("y",)}  # repeated query terms count once
    assert index.search(["nothing"], 5) == [] and index.search(["x"], 0) == []
    assert BM25Index.build([]).search(["x"], 5) == []


def test_invalid_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        BM25Index.build([("a", ["x"]), ("a", ["y"])])
    with pytest.raises(ValueError):
        BM25Index(k1=-1)
    with pytest.raises(ValueError):
        BM25Index(b=1.5)


# --- the retriever on the corpus ----------------------------------------------------


@pytest.fixture(scope="module")
def bm25(embedded: Embedded) -> BM25Retriever:
    return BM25Retriever(embedded.engine)


@pytest.fixture(scope="module")
def chunk_texts(embedded: Embedded) -> dict[str, str]:
    with embedded.engine.connect() as connection:
        rows = connection.execute(select(DocumentChunk.__table__)).mappings()
        return {row["id"]: lexical_text(row) for row in rows}


def _rank(results: list, predicate: object) -> int | None:
    return next((r.rank for r in results if predicate(r)), None)  # type: ignore[operator]


def test_index_covers_every_chunk(bm25: BM25Retriever, embedded: Embedded) -> None:
    assert len(bm25.index) == len(embedded.chunks)


def test_identifiers_unique_to_one_chunk_rank_it_first(
    bm25: BM25Retriever, chunk_texts: dict[str, str]
) -> None:
    where: dict[str, set[str]] = defaultdict(set)
    for chunk_id, text in chunk_texts.items():
        for key in set(re.findall(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b", text)):
            where[key].add(chunk_id)
    unique = {key: ids.pop() for key, ids in where.items() if len(ids) == 1}
    assert len(unique) >= 10
    for key, chunk_id in unique.items():
        assert bm25.search(key, top_k=1)[0].chunk_id == chunk_id, key


def test_every_incident_id_finds_its_incident(
    bm25: BM25Retriever, dataset: SyntheticDataset
) -> None:
    for incident in dataset.incidents:
        results = bm25.search(incident.id, top_k=5)
        assert any(r.document_id == incident.id for r in results), incident.id


def test_service_and_version_find_the_deployment(
    bm25: BM25Retriever, dataset: SyntheticDataset
) -> None:
    releases = Counter((d.service_id, d.version) for d in dataset.deployments)
    for deployment in dataset.deployments:
        if releases[(deployment.service_id, deployment.version)] != 1:
            continue  # redeployed versions have several equally valid answers
        top = bm25.search(f"{deployment.service_id} {deployment.version}", top_k=1)[0]
        assert top.document_id == deployment.id


def test_class_names_return_only_chunks_that_contain_them(
    bm25: BM25Retriever, dataset: SyntheticDataset, chunk_texts: dict[str, str]
) -> None:
    """A class-name lookup returns only chunks containing the name.

    This needs the identifier's parts to count less than the whole identifier in a
    pure lookup (``BM25_PART_WEIGHT``): with equal weights, an incident about the
    MailRelay provider outranked pull requests changing ``MailRelayClient`` (17.58 vs
    17.51), because it shares the parts "mail" and "relay"."""
    names = set()
    for code in dataset.code_files:
        if code.language == "python":
            tree = ast.parse(code.content)
            names |= {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
    camel = sorted(n for n in names if re.fullmatch(r"(?:[A-Z][a-z0-9]+){2,}", n))
    checked = 0
    for name in camel:
        pattern = re.compile(rf"\b{name}\b")
        if sum(bool(pattern.search(t)) for t in chunk_texts.values()) < 5:
            continue
        checked += 1
        for result in bm25.search(name, top_k=5):
            assert pattern.search(chunk_texts[result.chunk_id]), (name, result.chunk_id)
    assert checked >= 20


def test_error_messages_find_the_document_that_quotes_them(
    bm25: BM25Retriever, chunk_texts: dict[str, str], embedded: Embedded
) -> None:
    documents = {c.id: c.document_id for c in embedded.chunks}
    quoted: dict[str, set[str]] = defaultdict(set)
    for chunk_id, text in chunk_texts.items():
        for phrase in re.findall(r"`([A-Za-z][^`\n]{10,60})`", text):
            if len(phrase.split()) >= 3:
                quoted[phrase].add(documents[chunk_id])
    unique = {p: ids.pop() for p, ids in quoted.items() if len(ids) == 1}
    found = sum(
        _rank(bm25.search(p, top_k=3), lambda r, d=doc: r.document_id == d) is not None
        for p, doc in unique.items()
    )
    assert len(unique) >= 50
    assert found / len(unique) >= 0.9  # measured: 111 of 114 in the top 3


def test_code_paths_return_that_file(bm25: BM25Retriever, dataset: SyntheticDataset) -> None:
    matched = sum(bm25.search(c.path, top_k=1)[0].file_path == c.path for c in dataset.code_files)
    assert matched / len(dataset.code_files) >= 0.99


def test_results_carry_provenance_and_evidence(bm25: BM25Retriever, embedded: Embedded) -> None:
    chunks = {c.id: c for c in embedded.chunks}
    results = bm25.search("QueuePool limit reached payment-service", top_k=10)
    assert [r.rank for r in results] == list(range(1, 11))
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)
    for r in results:
        source = chunks[r.chunk_id]
        assert (r.document_id, r.content, r.source_type) == (
            source.document_id,
            source.content,
            source.source_type,
        )
        assert r.retriever == "bm25" and r.matched_terms and "bm25_score" in r.score_details


FILTERS = [
    ChunkFilter(services=frozenset({"payment-service"})),
    ChunkFilter(source_types=frozenset({SourceType.RUNBOOK, SourceType.POSTMORTEM})),
    ChunkFilter(doc_types=frozenset({"configuration"})),
    ChunkFilter(versions=frozenset({"v2.8.1"})),
    ChunkFilter(access=principal_for_role("t", "developer").chunk_access()),
    ChunkFilter(since=datetime(2026, 5, 1, tzinfo=UTC), until=datetime(2026, 7, 1, tzinfo=UTC)),
    ChunkFilter(document_ids=frozenset({"INC-0406", "PM-0039", "RB-0001"})),
]


def satisfies(r: RetrievedChunk | ChunkRecord, f: ChunkFilter) -> bool:
    timestamp = r.timestamp.replace(tzinfo=UTC)  # SQLite returns naive UTC timestamps
    checks = [
        f.services is None or r.service_id in f.services,
        f.source_types is None or r.source_type in f.source_types,
        f.doc_types is None or r.doc_type in f.doc_types,
        f.versions is None or r.version in f.versions,
        f.access is None or (r.source_type, r.access_level) in f.access,
        r.access_level is not AccessLevel.CONFIDENTIAL,  # never, with or without filters
        f.since is None or timestamp >= f.since,
        f.until is None or timestamp < f.until,
        f.document_ids is None or r.document_id in f.document_ids,
    ]
    return all(checks)


@pytest.mark.parametrize("chunk_filter", FILTERS)
def test_filtered_search_is_the_unfiltered_ranking_restricted(
    bm25: BM25Retriever, embedded: Embedded, chunk_filter: ChunkFilter
) -> None:
    query = "database connection pool timeout payment v2.8.1 configuration"
    filtered = bm25.search(query, top_k=10, filters=chunk_filter)
    chunks = {c.id: c for c in embedded.chunks}
    full_ranking = bm25.index.search(bm25.analyze(query), top_k=len(bm25.index))
    allowed = [h.doc_id for h in full_ranking if satisfies(chunks[h.doc_id], chunk_filter)]
    assert filtered and all(satisfies(r, chunk_filter) for r in filtered)
    assert [r.chunk_id for r in filtered] == allowed[:10]


def test_queries_without_terms_and_bad_arguments(bm25: BM25Retriever) -> None:
    assert bm25.search("the of and", top_k=5) == []
    with pytest.raises(RetrievalError):
        bm25.search("  ", top_k=5)
    with pytest.raises(RetrievalError):
        bm25.search("pool", top_k=0)


@pytest.fixture
def private_corpus(dataset: SyntheticDataset) -> Iterator[Corpus]:
    corpus = build_corpus(dataset)
    yield corpus
    corpus.engine.dispose()


def test_filters_use_the_database_not_the_stale_index(private_corpus: Corpus) -> None:
    """Chunks changed or deleted after the index was built are filtered by their
    current rows: a stale index can never leak a chunk the filters exclude."""
    engine, c = private_corpus.engine, DocumentChunk
    retriever = BM25Retriever(engine)
    internal = ChunkFilter(access=principal_for_role("t", "sre").chunk_access())
    top = retriever.search("PaymentServiceDBPoolSaturated", top_k=3, filters=internal)
    assert len(top) == 3
    restricted, removed = top[0].chunk_id, top[1].chunk_id
    with engine.begin() as connection:
        connection.execute(
            update(c).where(c.id == restricted).values(access_level=AccessLevel.CONFIDENTIAL)
        )
        connection.execute(delete(c).where(c.id == removed))
    after = retriever.search("PaymentServiceDBPoolSaturated", top_k=10, filters=internal)
    assert after and restricted not in {r.chunk_id for r in after}
    assert removed not in {r.chunk_id for r in retriever.search("PaymentServiceDBPoolSaturated")}


def test_refresh_picks_up_new_text(private_corpus: Corpus) -> None:
    engine, c = private_corpus.engine, DocumentChunk
    retriever = BM25Retriever(engine)
    assert retriever.search("zyxwvut", top_k=5) == []
    target = private_corpus.chunks[0].id
    with engine.begin() as connection:
        connection.execute(update(c).where(c.id == target).values(content="zyxwvut marker"))
    assert retriever.search("zyxwvut", top_k=5) == []  # index not rebuilt yet
    retriever.refresh()
    assert [r.chunk_id for r in retriever.search("zyxwvut", top_k=5)] == [target]
