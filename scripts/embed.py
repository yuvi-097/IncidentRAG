"""Embed chunks into pgvector with the configured model.

    python scripts/embed.py [--source-type runbook ...] [--force] [--batch-size 32]

chunks -> batching -> embedding -> pgvector. Incremental: a chunk is embedded
only when it has no vector for the configured model yet, or its text changed;
--force re-embeds everything. The model comes from EMBEDDING_* settings (.env).
Run scripts/ingest.py first.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import load_settings  # noqa: E402
from app.database.session import create_db_engine  # noqa: E402
from app.observability.structured_logging import configure_logging  # noqa: E402
from app.rag.embeddings import EmbeddingPipeline, build_embedding_provider  # noqa: E402
from app.schemas.enums import SourceType  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    settings = load_settings()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source-type", action="append", choices=[s.value for s in SourceType])
    parser.add_argument("--force", action="store_true", help="re-embed even unchanged chunks")
    parser.add_argument("--batch-size", type=int, default=settings.embedding.batch_size)
    args = parser.parse_args(argv)
    configure_logging(settings.app.log_level, settings.app.log_format)

    print(
        f"Loading embedding model {settings.embedding.model!r} ({settings.embedding.provider})..."
    )
    provider = build_embedding_provider(settings.embedding)
    engine = create_db_engine(settings.database, application_name="opsrag-embed")
    try:
        types = [SourceType(s) for s in args.source_type] if args.source_type else None
        report = EmbeddingPipeline(provider, args.batch_size).run(engine, types, force=args.force)
    finally:
        engine.dispose()
    print(f"\nModel {report.model} ({report.dimension} dims, max {report.max_input_tokens} tokens)")
    print(f"  chunks                 {report.total_chunks}")
    print(f"  already embedded       {report.already_embedded}  (skipped)")
    print(f"  embedded now           {report.embedded}  in {report.batches} batches")
    print(f"  identical text reused  {report.reused_identical_text}")
    print(f"  truncated inputs       {report.truncated}")
    print(f"  vector index           {report.index}")
    print(f"  time                   {report.seconds}s ({report.chunks_per_second} chunks/s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
