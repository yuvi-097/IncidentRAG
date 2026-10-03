"""Chunk NovaCart sources from PostgreSQL into ``document_chunks``.

    python scripts/ingest.py [--strategy document_aware|recursive|fixed]
                             [--chunk-size 350] [--overlap 50]
                             [--source-type incident --source-type code ...]
                             [--dry-run] [--show INC-0406 --show PR-1501]

Pipeline: raw data -> parsing -> cleaning -> metadata -> chunking -> persistence.
Idempotent: the chunks of the selected source types are replaced in one
transaction, and chunk ids are deterministic. Defaults come from CHUNKING_* settings.
Run scripts/seed_db.py first. Re-seeding clears chunks, so re-run this script afterwards.
No embeddings are computed here.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import load_settings  # noqa: E402
from app.database.session import create_db_engine  # noqa: E402
from app.observability.structured_logging import configure_logging  # noqa: E402
from app.rag.chunking import ChunkingConfig  # noqa: E402
from app.rag.ingestion import ChunkRecord, IngestionPipeline, IngestionReport  # noqa: E402
from app.schemas.enums import ChunkingStrategy, SourceType  # noqa: E402


def print_report(report: IngestionReport) -> None:
    print(
        f"\nStrategy {report.strategy}: chunk size {report.chunk_size_tokens} tokens, "
        f"overlap {report.chunk_overlap_tokens} ({report.seconds}s, "
        f"{'persisted' if report.persisted else 'dry run, nothing written'})"
    )
    header = f"{'source type':<15}{'sources':>8}{'chunks':>8}   "
    print(f"\n{header}chunks/source (med/max)   tokens/chunk (min/med/p90/max)")
    for source_type, sources in report.sources_read.items():
        chunks = report.chunks_by_source_type.get(source_type, 0)
        per = report.chunks_per_source.get(source_type)
        tok = report.tokens_by_source_type.get(source_type)
        per_s = f"{per.median:g}/{per.max}" if per else "-"
        tok_s = f"{tok.min}/{tok.median:g}/{tok.p90}/{tok.max}" if tok else "-"
        print(f"{source_type:<15}{sources:>8}{chunks:>8}   {per_s:<25} {tok_s}")
    print(f"{'total':<15}{report.total_sources:>8}{report.total_chunks:>8}")
    if report.flags:
        print(f"\nSplit inside a unit (the unit alone exceeded the limit): {report.flags}")
    print(f"Skipped sources: {len(report.skipped)}")
    for skipped in report.skipped[:20]:
        print(f"  {skipped.source_type.value} {skipped.source_id}: {skipped.reason}")


def print_chunk(chunk: ChunkRecord) -> None:
    location = (
        f"{chunk.file_path}:{chunk.metadata['line_start']}-{chunk.metadata['line_end']}"
        if chunk.file_path
        else f"lines {chunk.metadata['line_start']}-{chunk.metadata['line_end']}"
    )
    print(f"\n--- {chunk.id}  [{chunk.source_type.value}] {chunk.title}")
    print(
        f"    section={chunk.section!r}  service={chunk.service_id}  version={chunk.version}  "
        f"access={chunk.access_level.value}  tokens={chunk.token_count}  {location}"
    )
    print(
        f"    timestamp={chunk.timestamp:%Y-%m-%d %H:%M}  mentions={chunk.metadata.get('mentions')}"
    )
    print(
        textwrap.indent(
            chunk.content if len(chunk.content) < 900 else chunk.content[:900] + "\n[...]", "    | "
        )
    )


def main(argv: list[str] | None = None) -> int:
    settings = load_settings()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--strategy",
        choices=[s.value for s in ChunkingStrategy],
        default=settings.chunking.strategy.value,
    )
    parser.add_argument("--chunk-size", type=int, default=settings.chunking.chunk_size_tokens)
    parser.add_argument("--overlap", type=int, default=settings.chunking.chunk_overlap_tokens)
    parser.add_argument(
        "--source-type",
        action="append",
        choices=[s.value for s in SourceType],
        help="limit to these source types (repeatable); default: all",
    )
    parser.add_argument("--dry-run", action="store_true", help="chunk and report without writing")
    parser.add_argument(
        "--show",
        action="append",
        default=[],
        metavar="SOURCE_ID",
        help="print the chunks of a source, e.g. INC-0406 (repeatable)",
    )
    args = parser.parse_args(argv)

    configure_logging(settings.app.log_level, settings.app.log_format)
    config = ChunkingConfig(
        strategy=ChunkingStrategy(args.strategy),
        chunk_size_tokens=args.chunk_size,
        chunk_overlap_tokens=args.overlap,
    )
    source_types = [SourceType(s) for s in args.source_type] if args.source_type else None
    engine = create_db_engine(settings.database, application_name="opsrag-ingest")
    try:
        chunks, report = IngestionPipeline(config).run(engine, source_types, dry_run=args.dry_run)
    finally:
        engine.dispose()
    print_report(report)
    for source_id in args.show:
        for chunk in (c for c in chunks if c.document_id == source_id):
            print_chunk(chunk)
    return 0 if report.total_chunks else 1


if __name__ == "__main__":
    raise SystemExit(main())
