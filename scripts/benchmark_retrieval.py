"""Run the retrieval benchmark for one retrieval mode.

    python scripts/benchmark_retrieval.py [--mode dense|sparse|hybrid|hybrid_rerank]
        [--benchmark data/evaluation/retrieval_benchmark.jsonl] [--output PATH]

Uses the configured models and the chunks/embeddings in the database (run
ingest.py and embed.py first). Prints aggregate and per-question results and writes
the full report as JSON (default: data/evaluation/results/<mode>.json). Nothing here
feeds labels back into retrieval. To compare all modes, use compare_retrieval.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import load_settings  # noqa: E402
from app.database.session import create_db_engine  # noqa: E402
from app.evaluation.retrieval import evaluate, load_benchmark, save_report  # noqa: E402
from app.rag.retrieval.factory import RetrievalComponents  # noqa: E402
from app.schemas.enums import RetrievalMode  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    settings = load_settings()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--mode",
        choices=[m.value for m in RetrievalMode],
        default=settings.retrieval.mode.value,
    )
    parser.add_argument(
        "--benchmark", type=Path, default=ROOT / "data/evaluation/retrieval_benchmark.jsonl"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    output = args.output or ROOT / f"data/evaluation/results/{args.mode}.json"

    questions = load_benchmark(args.benchmark)
    engine = create_db_engine(settings.database, application_name="opsrag-benchmark")
    try:
        components = RetrievalComponents(engine, settings)
        retriever = components.retriever(RetrievalMode(args.mode))
        retriever.search("warm-up query", top_k=1)  # load lazily built indexes first
        model = components.embedding_provider.name if components.embedding_provider else ""
        report = evaluate(retriever, engine, questions, model=model)
    finally:
        engine.dispose()
    save_report(report, output)

    print(f"Retriever {report.retriever}, {report.questions} questions")
    print(
        "  "
        + "   ".join(f"Recall@{k} {report.recall[k]:.3f}" for k in report.ks)
        + f"   MRR@10 {report.mrr:.3f}   NDCG@10 {report.ndcg[10]:.3f}"
    )
    print(f"  median latency {report.latency_ms_median} ms/query\n")
    print(f"  {'category':<14}{'n':>3}{'R@5':>8}{'R@10':>8}{'MRR':>8}{'NDCG@10':>9}")
    for name, stats in report.by_category.items():
        print(
            f"  {name:<14}{stats.questions:>3}{stats.recall[5]:>8.3f}{stats.recall[10]:>8.3f}"
            f"{stats.mrr:>8.3f}{stats.ndcg[10]:>9.3f}"
        )
    print("\n  id     rank  relevant  question")
    for r in report.results:
        rank = str(r.first_relevant_rank) if r.first_relevant_rank else "-"
        print(f"  {r.id}  {rank:>4}  {len(r.relevant):>8}  {r.question[:75]}")
    print(f"\nFull report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
