"""Retrieval benchmark: labelled questions, relevance resolution, Recall@k.

Relevance is judged at the *source record* level: a retrieved chunk counts as
relevant if its ``document_id`` is one of the question's relevant records. Labels
are selectors over chunk metadata (e.g. "the runbook titled X", "the code file
ending in db/database.py"), resolved against the database at run time, so they
survive dataset regeneration. They are only ever read by this module; the
retriever never sees them.

Metrics, for the top-k retrieved *chunks* (mapped to their source records):
  Recall@k  |relevant ∩ retrieved@k| / min(|relevant|, k)   (capped recall)
  Hit@k     1 if any relevant record is in retrieved@k
  MRR@10    1 / rank of the first relevant chunk (0 if none in the top 10)
  NDCG@k    binary-relevance DCG / ideal DCG. A relevant record gains
            1 / log2(rank + 1) at the rank of its first chunk; further chunks of
            the same record gain nothing, so NDCG stays within [0, 1].
With a single relevant record, Recall@k equals Hit@k.
"""

from __future__ import annotations

import json
import math
import statistics
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import select
from sqlalchemy.engine import Engine

from app.database.models import DocumentChunk
from app.rag.retrieval.base import Retriever
from app.schemas.enums import SourceType


class RelevanceSelector(BaseModel):
    """All given conditions must hold for a chunk's source record to be relevant."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_ids: list[str] | None = None
    source_type: SourceType | None = None
    service_id: str | None = None
    version: str | None = None
    doc_type: str | None = None
    title: str | None = None
    title_contains: str | None = None
    file_path_endswith: str | None = None
    metadata: dict[str, Any] | None = None  # e.g. {"category": "authentication_failure"}

    @model_validator(mode="after")
    def _not_empty(self) -> RelevanceSelector:
        if all(value is None for value in self.model_dump().values()):
            raise ValueError("a relevance selector needs at least one condition")
        return self

    def matches(self, chunk: dict[str, Any]) -> bool:
        checks = [
            self.document_ids is None or chunk["document_id"] in self.document_ids,
            self.source_type is None or chunk["source_type"] == self.source_type,
            self.service_id is None or chunk["service_id"] == self.service_id,
            self.version is None or chunk["version"] == self.version,
            self.doc_type is None or chunk["doc_type"] == self.doc_type,
            self.title is None or chunk["title"] == self.title,
            self.title_contains is None or self.title_contains.lower() in chunk["title"].lower(),
            self.file_path_endswith is None
            or (chunk["file_path"] or "").endswith(self.file_path_endswith),
            self.metadata is None
            or all(chunk["metadata"].get(k) == v for k, v in self.metadata.items()),
        ]
        return all(checks)


class BenchmarkQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    question: str
    category: str
    relevant: list[RelevanceSelector] = Field(min_length=1)
    notes: str = ""


class QuestionResult(BaseModel):
    id: str
    question: str
    category: str
    relevant: list[str]
    retrieved: list[str]  # source record of each retrieved chunk, in rank order
    first_relevant_rank: int | None
    recall: dict[int, float]
    hit: dict[int, float]
    ndcg: dict[int, float]
    latency_ms: float

    @property
    def reciprocal_rank(self) -> float:
        return 1 / self.first_relevant_rank if self.first_relevant_rank else 0.0


class Summary(BaseModel):
    questions: int
    recall: dict[int, float]
    hit: dict[int, float]
    ndcg: dict[int, float]
    mrr: float
    latency_ms_median: float


class BenchmarkReport(Summary):
    retriever: str
    model: str
    ks: list[int]
    by_category: dict[str, Summary]
    results: list[QuestionResult]


def load_benchmark(path: Path) -> list[BenchmarkQuestion]:
    questions = [
        BenchmarkQuestion.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [q.id for q in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate question ids in benchmark")
    return questions


def chunk_catalog(engine: Engine) -> list[dict[str, Any]]:
    c = DocumentChunk
    query = select(
        c.document_id,
        c.source_type,
        c.service_id,
        c.version,
        c.doc_type,
        c.title,
        c.file_path,
        c.chunk_metadata.label("metadata"),
    )
    with engine.connect() as connection:
        return [dict(row) for row in connection.execute(query).mappings()]


def resolve_relevant(question: BenchmarkQuestion, catalog: list[dict[str, Any]]) -> set[str]:
    relevant = {
        row["document_id"] for row in catalog if any(s.matches(row) for s in question.relevant)
    }
    if not relevant:
        raise ValueError(f"{question.id}: relevance selectors match no records")
    return relevant


def ndcg_at(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    seen: set[str] = set()
    dcg = 0.0
    for rank, doc in enumerate(retrieved[:k], 1):
        if doc in relevant and doc not in seen:
            seen.add(doc)
            dcg += 1 / math.log2(rank + 1)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def _mean(values: list[float]) -> float:
    return round(statistics.fmean(values), 4) if values else 0.0


def summarize(results: Sequence[QuestionResult], ks: Sequence[int]) -> Summary:
    return Summary(
        questions=len(results),
        recall={k: _mean([r.recall[k] for r in results]) for k in ks},
        hit={k: _mean([r.hit[k] for r in results]) for k in ks},
        ndcg={k: _mean([r.ndcg[k] for r in results]) for k in ks},
        mrr=_mean([r.reciprocal_rank for r in results]),
        latency_ms_median=round(statistics.median([r.latency_ms for r in results]), 1)
        if results
        else 0.0,
    )


def evaluate(
    retriever: Retriever,
    engine: Engine,
    questions: list[BenchmarkQuestion],
    ks: tuple[int, ...] = (1, 5, 10),
    model: str = "",
) -> BenchmarkReport:
    catalog = chunk_catalog(engine)
    depth = max(ks)
    results: list[QuestionResult] = []
    for question in questions:
        relevant = resolve_relevant(question, catalog)
        started = time.perf_counter()
        retrieved = [
            chunk.document_id for chunk in retriever.search(question.question, top_k=depth)
        ]
        latency = (time.perf_counter() - started) * 1000
        first = next((rank for rank, doc in enumerate(retrieved, 1) if doc in relevant), None)
        recall, hit, ndcg = {}, {}, {}
        for k in ks:
            found = relevant & set(retrieved[:k])
            recall[k] = len(found) / min(len(relevant), k)
            hit[k] = 1.0 if found else 0.0
            ndcg[k] = ndcg_at(retrieved, relevant, k)
        results.append(
            QuestionResult(
                id=question.id,
                question=question.question,
                category=question.category,
                relevant=sorted(relevant),
                retrieved=retrieved,
                first_relevant_rank=first,
                recall=recall,
                hit=hit,
                ndcg=ndcg,
                latency_ms=round(latency, 1),
            )
        )

    by_category: dict[str, list[QuestionResult]] = defaultdict(list)
    for result in results:
        by_category[result.category].append(result)
    return BenchmarkReport(
        **summarize(results, ks).model_dump(),
        retriever=retriever.name,
        model=model,
        ks=list(ks),
        by_category={name: summarize(rs, ks) for name, rs in sorted(by_category.items())},
        results=results,
    )


def save_report(report: BenchmarkReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")
