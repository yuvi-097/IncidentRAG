"""Run the Phase 9 benchmarks (temporal, multi-hop) through the agent and score them.

    python scripts/evaluate_reasoning.py --label baseline [--suite benchmark|stress]
        [--backend sqlite|postgres]

The full dataset is loaded into an in-memory SQLite database (or, with
``--backend postgres``, into the PostgreSQL database configured in .env, whose tables are
replaced) and ingested. Text search is BM25 (deterministic, no model download), synthesis
is extractive, and the clock is fixed at 2026-09-01. Questions are asked as an admin, so
the scores measure reasoning, not access control. Writes
data/evaluation/results/reasoning_<label>.json.

Suites: ``benchmark`` (temporal_benchmark.jsonl, multihop_benchmark.jsonl) and
``stress`` (temporal_stress.jsonl, multihop_stress.jsonl: held-out phrasings and
constructs, see scripts/build_reasoning_stress.py).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from app.agents.entities import ServiceCatalog  # noqa: E402
from app.agents.graph import Agent  # noqa: E402
from app.agents.router import RuleBasedRouter  # noqa: E402
from app.config import load_settings  # noqa: E402
from app.database.schema import create_schema, prepare_schema  # noqa: E402
from app.database.seed import seed_database  # noqa: E402
from app.database.session import create_db_engine  # noqa: E402
from app.evaluation.reasoning import (  # noqa: E402
    MultiHopQuestion,
    Scored,
    TemporalQuestion,
    load_questions,
    score_multihop,
    score_temporal,
    summarize,
)
from app.rag.chunking import ChunkingConfig  # noqa: E402
from app.rag.ingestion import IngestionPipeline  # noqa: E402
from app.rag.ingestion.persistence import ensure_chunk_table  # noqa: E402
from app.rag.retrieval import BM25Retriever  # noqa: E402
from app.security import principal_for_role  # noqa: E402
from app.synthetic.storage import load_dataset  # noqa: E402
from app.tools import build_registry  # noqa: E402

NOW = datetime(2026, 9, 1, tzinfo=UTC)
EVALUATION = ROOT / "data" / "evaluation"


def build_agent(backend: str = "sqlite") -> Agent:
    if backend == "postgres":  # the database configured in .env; its tables are replaced
        settings = load_settings()
        if settings.app.environment == "production":
            raise SystemExit("refusing to seed synthetic data into production")
        engine = create_db_engine(settings.database)
        prepare_schema(engine)
        ensure_chunk_table(engine)
    else:
        engine = create_engine(
            "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
        )
        create_schema(engine)
    seed_database(engine, load_dataset(ROOT / "data" / "generated"))
    IngestionPipeline(ChunkingConfig()).run(engine)
    bm25 = BM25Retriever(engine)
    return Agent(
        engine=engine,
        registry=build_registry(),
        router=RuleBasedRouter(ServiceCatalog.from_engine(engine), clock=lambda: NOW),
        retriever=bm25,
        term_stats=bm25.index,
        clock=lambda: NOW,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--label", default="current")
    parser.add_argument("--suite", choices=["benchmark", "stress"], default="benchmark")
    parser.add_argument("--backend", choices=["sqlite", "postgres"], default="sqlite")
    args = parser.parse_args()
    agent = build_agent(args.backend)
    principal = principal_for_role("eval-admin", "admin")
    report: dict[str, object] = {
        "label": args.label,
        "suite": args.suite,
        "backend": args.backend,
    }
    for name, model, scorer, group in (
        ("temporal", TemporalQuestion, score_temporal, "relation"),
        ("multihop", MultiHopQuestion, score_multihop, "kind"),
    ):
        results = []
        started = time.perf_counter()
        for q in load_questions(EVALUATION / f"{name}_{args.suite}.jsonl", model):
            state = agent.run(q.question, principal)
            correct, detail = scorer(q, state.final_answer)  # type: ignore[operator]
            results.append(
                Scored(
                    id=q.id,
                    group=getattr(q, group),
                    question=q.question,
                    correct=correct,
                    detail={
                        **detail,
                        "query_type": state.query_type.value if state.query_type else None,
                        "plan": state.plan,
                    },
                    answer=state.final_answer,
                    tools=[r.tool for r in state.tool_results],
                )
            )
        summary = summarize(results)
        summary["seconds"] = round(time.perf_counter() - started, 1)
        report[name] = {"summary": summary, "results": [r.model_dump() for r in results]}
        print(f"{name}: {summary['correct']}/{summary['questions']} correct")
        for g, s in summary["by_group"].items():
            print(f"  {g:<20} {s['correct']}/{s['questions']}")
        for hop, s in summary.get("by_hop", {}).items():
            print(f"  hop {hop:<16} {s['correct']}/{s['checked']}")
    out = EVALUATION / "results" / f"reasoning_{args.label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
