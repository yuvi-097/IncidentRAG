"""Search from the command line.

python scripts/search.py "Why did payment-service fail?" [--top-k 5]
    [--mode dense|sparse|hybrid|hybrid_rerank]   (default: RETRIEVAL_MODE)
    [--service payment-service] [--source-type incident] [--doc-type runbook]
    [--version v2.8.1] [--since 2026-06-01] [--until 2026-07-01]
    [--max-access-level internal]

Each result shows its provenance and why it ranked there (per-stage ranks and
scores, and the query terms BM25 matched).
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import load_settings  # noqa: E402
from app.database.session import create_db_engine  # noqa: E402
from app.rag.retrieval.factory import build_retriever  # noqa: E402
from app.rag.store import ChunkFilter  # noqa: E402
from app.schemas.enums import RetrievalMode, SourceType  # noqa: E402
from app.security import load_policy, principal_for_role  # noqa: E402


def _date(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _set(values: list[str] | None) -> frozenset[str] | None:
    return frozenset(values) if values else None


def main(argv: list[str] | None = None) -> int:
    settings = load_settings()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=settings.retrieval.top_k)
    parser.add_argument(
        "--mode", choices=[m.value for m in RetrievalMode], default=settings.retrieval.mode.value
    )
    parser.add_argument("--service", action="append")
    parser.add_argument("--source-type", action="append", choices=[s.value for s in SourceType])
    parser.add_argument("--doc-type", action="append")
    parser.add_argument("--version", action="append")
    parser.add_argument("--since", type=_date)
    parser.add_argument("--until", type=_date)
    parser.add_argument("--role", help="only chunks this role may read (access policy)")
    args = parser.parse_args(argv)

    filters = ChunkFilter(
        services=_set(args.service),
        source_types=frozenset(SourceType(s) for s in args.source_type)
        if args.source_type
        else None,
        doc_types=_set(args.doc_type),
        versions=_set(args.version),
        since=args.since,
        until=args.until,
        access=principal_for_role(
            "cli", args.role, load_policy(settings.security.policy_file)
        ).chunk_access()
        if args.role
        else None,
    )
    engine = create_db_engine(settings.database, application_name="opsrag-search")
    try:
        retriever = build_retriever(engine, settings, RetrievalMode(args.mode))
        results = retriever.search(args.query, args.top_k, None if filters.is_empty else filters)
    finally:
        engine.dispose()
    for r in results:
        where = r.file_path or r.section or ""
        print(
            f"{r.rank:>2}. {r.score:.3f}  {r.chunk_id:<16} [{r.source_type.value}] {r.title[:70]}"
        )
        meta = f"{r.service_id or '-'} | {r.version or '-'} | {r.access_level.value}"
        print(f"      {meta} | {where[:90]}")
        ranks = ", ".join(
            f"{k.removesuffix('_rank')} #{int(v)}"
            for k, v in r.score_details.items()
            if k.endswith("_rank")
        )
        evidence = [f"ranks: {ranks}"] if ranks else []
        if r.matched_terms:
            evidence.append("matched: " + " ".join(r.matched_terms[:12]))
        if evidence:
            print("      " + " | ".join(evidence))
    if not results:
        print("No results.")
    print(f"({retriever.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
