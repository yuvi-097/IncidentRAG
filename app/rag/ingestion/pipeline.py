"""The ingestion pipeline:

    raw data -> parsing -> cleaning -> metadata extraction -> chunking -> persistence

A source that is empty or malformed is skipped with a reason; one bad record
never aborts the run. Chunk ids are deterministic (``<source id>#<index>``), so
re-ingesting unchanged sources reproduces identical chunks.
"""

from __future__ import annotations

import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy.engine import Engine

from app.rag.chunking import ChunkingConfig, build_chunker, count_tokens
from app.rag.chunking.base import trim_span
from app.rag.ingestion.cleaning import clean_text
from app.rag.ingestion.metadata import chunk_file_path, chunk_metadata
from app.rag.ingestion.models import SOURCE_FOREIGN_KEY, ChunkRecord, RawSource, SkipSource
from app.rag.ingestion.parsing import parse_source
from app.rag.ingestion.persistence import ensure_chunk_table, replace_chunks
from app.rag.ingestion.sources import load_sources
from app.schemas.enums import AccessLevel, SourceType
from app.synthetic.text import sha256

MAX_SECTION = 512


class Skipped(BaseModel):
    source_type: SourceType
    source_id: str
    reason: str


class TokenStats(BaseModel):
    count: int
    min: int
    median: float
    p90: int
    max: int
    mean: float

    @classmethod
    def of(cls, values: list[int]) -> TokenStats:
        ordered = sorted(values)
        return cls(
            count=len(ordered),
            min=ordered[0],
            median=statistics.median(ordered),
            p90=ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))],
            max=ordered[-1],
            mean=round(statistics.fmean(ordered), 1),
        )


class IngestionReport(BaseModel):
    strategy: str
    chunk_size_tokens: int
    chunk_overlap_tokens: int
    sources_read: dict[str, int] = Field(default_factory=dict)
    chunks_by_source_type: dict[str, int] = Field(default_factory=dict)
    tokens_by_source_type: dict[str, TokenStats] = Field(default_factory=dict)
    chunks_per_source: dict[str, TokenStats] = Field(default_factory=dict)
    flags: dict[str, int] = Field(default_factory=dict)
    skipped: list[Skipped] = Field(default_factory=list)
    total_sources: int = 0
    total_chunks: int = 0
    persisted: bool = False
    table_rebuilt: bool = False
    sync: dict[str, int] = Field(default_factory=dict)  # inserted/updated/deleted/unchanged
    seconds: float = 0.0


def source_text(raw: RawSource) -> str:
    """The exact text a chunk's ``char_start``/``char_end`` refer to (parsed, then cleaned).
    Deterministic, so any chunk can be re-derived and checked against ``source_hash``."""
    parsed = parse_source(raw)
    return clean_text(parsed.text, parsed.format)


class IngestionPipeline:
    def __init__(self, config: ChunkingConfig) -> None:
        self.config = config
        self.chunker = build_chunker(config)

    def chunk_source(self, raw: RawSource) -> list[ChunkRecord]:
        parsed = parse_source(raw)  # parsing
        if parsed.access_level is AccessLevel.CONFIDENTIAL:
            # Never indexed, so no retrieval path (or filter bug) can return it.
            raise SkipSource("confidential sources are not indexed")
        text = clean_text(parsed.text, parsed.format)  # cleaning
        if not text.strip():
            raise SkipSource("empty after cleaning")
        source_hash = sha256(text)
        spans = self.chunker.split(text, parsed.format)  # chunking
        records: list[ChunkRecord] = []
        foreign_key = SOURCE_FOREIGN_KEY[parsed.source_type]
        for span in spans:
            bounds = trim_span(text, span.start, span.end)
            if bounds is None:
                continue
            content = text[bounds[0] : bounds[1]]
            tokens = count_tokens(content)
            if tokens == 0:
                continue
            index = len(records)
            records.append(
                ChunkRecord.create(
                    id=f"{parsed.source_id}#{index:03d}",
                    document_id=parsed.source_id,
                    source_type=parsed.source_type,
                    chunk_index=index,
                    title=parsed.title[:512],
                    section=(span.section or None) and span.section[:MAX_SECTION],
                    content=content,
                    service_id=parsed.service_id,
                    timestamp=parsed.timestamp,
                    access_level=parsed.access_level,
                    version=parsed.version,
                    doc_type=parsed.doc_type,
                    file_path=chunk_file_path(parsed, span, content),
                    strategy=self.config.strategy,
                    char_start=bounds[0],
                    char_end=bounds[1],
                    token_count=tokens,
                    source_hash=source_hash,
                    metadata=chunk_metadata(
                        parsed,
                        span.__class__(bounds[0], bounds[1], span.section, span.metadata),
                        text,
                    ),  # metadata extraction
                    **{foreign_key: parsed.source_id},
                )
            )
        if not records:
            raise SkipSource("no chunkable content")
        return records

    def chunk_all(self, sources: Iterable[RawSource]) -> tuple[list[ChunkRecord], IngestionReport]:
        started = time.perf_counter()
        report = IngestionReport(
            strategy=self.config.strategy.value,
            chunk_size_tokens=self.config.chunk_size_tokens,
            chunk_overlap_tokens=self.config.chunk_overlap_tokens,
        )
        chunks: list[ChunkRecord] = []
        read: Counter[str] = Counter()
        per_source: dict[str, list[int]] = defaultdict(list)
        for raw in sources:
            read[raw.source_type.value] += 1
            try:
                records = self.chunk_source(raw)
            except SkipSource as exc:
                report.skipped.append(
                    Skipped(source_type=raw.source_type, source_id=raw.source_id, reason=str(exc))
                )
                continue
            per_source[raw.source_type.value].append(len(records))
            chunks.extend(records)
        report.sources_read = dict(sorted(read.items()))
        report.total_sources = sum(read.values())
        report.total_chunks = len(chunks)
        by_type: dict[str, list[int]] = defaultdict(list)
        flags: Counter[str] = Counter()
        for chunk in chunks:
            by_type[chunk.source_type.value].append(chunk.token_count)
            for flag in ("split_symbol", "split_block", "parse_error", "malformed"):
                if chunk.metadata.get(flag):
                    flags[flag] += 1
        report.chunks_by_source_type = {k: len(v) for k, v in sorted(by_type.items())}
        report.tokens_by_source_type = {k: TokenStats.of(v) for k, v in sorted(by_type.items())}
        report.chunks_per_source = {k: TokenStats.of(v) for k, v in sorted(per_source.items())}
        report.flags = dict(sorted(flags.items()))
        report.seconds = round(time.perf_counter() - started, 2)
        return chunks, report

    def run(
        self,
        engine: Engine,
        source_types: Iterable[SourceType] | None = None,
        dry_run: bool = False,
    ) -> tuple[list[ChunkRecord], IngestionReport]:
        types = set(source_types or SourceType)
        rebuilt = False if dry_run else ensure_chunk_table(engine)
        with engine.connect() as connection:
            sources = list(load_sources(connection, types))  # raw data
        chunks, report = self.chunk_all(sources)
        if not dry_run:
            sync = replace_chunks(engine, chunks, types)  # persistence
            report.sync = {
                "inserted": sync.inserted,
                "updated": sync.updated,
                "deleted": sync.deleted,
                "unchanged": sync.unchanged,
            }
        report.persisted, report.table_rebuilt = not dry_run, rebuilt
        return chunks, report


def summarize(report: IngestionReport) -> dict[str, Any]:
    return report.model_dump(mode="json")
