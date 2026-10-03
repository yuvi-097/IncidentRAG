"""Compare dense, BM25, hybrid and hybrid + reranker on the retrieval benchmark.

    python scripts/compare_retrieval.py [--benchmark PATH] [--output-dir data/evaluation/results]

All four systems share the same components (one embedding model, one BM25 index,
one reranker) and are evaluated on the same questions against the same database.
Writes <mode>.json per system, comparison.json and comparison.md, and prints the
tables. Configuration comes from the environment / .env (RETRIEVAL_*, BM25_*,
RERANKER_*, EMBEDDING_*).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import load_settings  # noqa: E402
from app.database.session import create_db_engine  # noqa: E402
from app.evaluation.comparison import compare, to_markdown  # noqa: E402
from app.evaluation.retrieval import (  # noqa: E402
    BenchmarkReport,
    evaluate,
    load_benchmark,
    save_report,
)
from app.rag.retrieval.factory import RetrievalComponents  # noqa: E402
from app.schemas.enums import RetrievalMode  # noqa: E402

SYSTEMS = {
    RetrievalMode.DENSE: "Dense only",
    RetrievalMode.SPARSE: "BM25 only",
    RetrievalMode.HYBRID: "Hybrid",
    RetrievalMode.HYBRID_RERANK: "Hybrid + reranker",
}


def main(argv: list[str] | None = None) -> int:
    settings = load_settings()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--benchmark", type=Path, default=ROOT / "data/evaluation/retrieval_benchmark.jsonl"
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/evaluation/results")
    args = parser.parse_args(argv)

    questions = load_benchmark(args.benchmark)
    engine = create_db_engine(settings.database, application_name="opsrag-compare")
    reports: dict[str, BenchmarkReport] = {}
    try:
        components = RetrievalComponents(engine, settings)
        started = time.perf_counter()
        components.bm25.refresh()
        print(f"BM25 index: {len(components.bm25.index)} chunks, ", end="")
        print(f"{len(components.bm25.index.postings)} terms, {time.perf_counter() - started:.2f}s")
        for mode, label in SYSTEMS.items():
            retriever = components.retriever(mode)
            retriever.search("warm-up query", top_k=1)  # model/tensor initialisation
            print(f"Evaluating {label} ({retriever.name})...", flush=True)
            report = evaluate(
                retriever,
                engine,
                questions,
                model=components.embedding_provider.name if components.embedding_provider else "",
            )
            save_report(report, args.output_dir / f"{mode.value}.json")
            reports[mode.value] = report
        dialect = engine.dialect.name
    finally:
        engine.dispose()

    retrieval = settings.retrieval
    config = {
        "database": dialect,
        "embedding_model": settings.embedding.model,
        "reranker_model": settings.reranker.model,
        "bm25": settings.bm25.model_dump(),
        "fusion": retrieval.fusion.value,
        "dense_weight": retrieval.dense_weight,
        "sparse_weight": retrieval.sparse_weight,
        "rrf_k": retrieval.rrf_k,
        "fusion_depth": retrieval.fusion_depth,
        "rerank_candidates": retrieval.rerank_candidates,
        "hnsw_ef_search": retrieval.hnsw_ef_search,
        "top_k": 10,
    }
    comparison = compare(reports, {m.value: label for m, label in SYSTEMS.items()}, config=config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(comparison.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )
    markdown = to_markdown(comparison)
    settings_line = ", ".join(f"{k}={v}" for k, v in config.items())
    (args.output_dir / "comparison.md").write_text(
        f"# Retrieval comparison\n\nConfiguration: {settings_line}\n\n{markdown}", encoding="utf-8"
    )
    print()
    print(markdown.split("### Rank of the first relevant")[0])
    print(f"Written: {args.output_dir / 'comparison.md'} and comparison.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
