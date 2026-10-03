"""Phase 10: comprehensive evaluation of retrieval, generation and the full agent.

    python scripts/evaluate.py                 # everything (about 30-60 min on CPU)
    python scripts/evaluate.py --quick         # BM25 and the agent without models
    python scripts/evaluate.py --methods BDF --limit 20

What it does:
1. Loads the frozen evaluation set (data/evaluation/eval_set.jsonl, 227 questions).
2. Builds (or reuses) the evaluation database: the generated dataset seeded into SQLite,
   ingested, and embedded with the configured model. It is cached in
   data/evaluation/cache/ under a key of the dataset hashes, the embedding model and the
   chunking configuration, so later runs skip the embedding.
3. Runs every question through the six systems A-F (see app/evaluation/pipelines.py),
   as the question's role, with the clock fixed at the dataset's end (2026-09-01).
4. Scores retrieval (Recall@1/5/10, Precision@5, MRR, NDCG@10) and generation
   (deterministic: correctness, faithfulness, citation correctness, hallucination,
   context relevance; model-based: NLI faithfulness; LLM-as-judge only if a judge model
   is configured), and classifies every failure.
5. Checks that no table changed (the system is read-only).
6. Writes data/evaluation/results/phase10/<run id>/: manifest.json (configuration,
   model versions, dataset version, code fingerprint, timestamp), per_question.jsonl,
   summary.json, report.md and plots/*.png.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sqlalchemy import create_engine, func, inspect, select  # noqa: E402
from sqlalchemy.engine import Engine  # noqa: E402

import app.database.models  # noqa: E402,F401  (registers every table on Base.metadata)
from app.agents.synthesis import ExtractiveSynthesizer  # noqa: E402
from app.agents.verification import load_nli  # noqa: E402
from app.config import Settings, load_settings  # noqa: E402
from app.database.base import Base  # noqa: E402
from app.database.schema import create_schema  # noqa: E402
from app.database.seed import seed_database  # noqa: E402
from app.evaluation.errors import classify  # noqa: E402
from app.evaluation.eval_set import (  # noqa: E402
    EvalQuestion,
    check_answer,
    file_sha256,
    load_eval_set,
)
from app.evaluation.judge import LLMJudge  # noqa: E402
from app.evaluation.metrics import (  # noqa: E402
    generation_scores,
    nli_faithfulness,
    retrieval_scores,
)
from app.evaluation.pipelines import METHODS, Retrievers, Runner  # noqa: E402
from app.evaluation.report import (  # noqa: E402
    Record,
    ablation_table,
    error_table,
    grouped_table,
    plots,
    summarize,
)
from app.llm import build_llm  # noqa: E402
from app.rag.chunking import ChunkingConfig  # noqa: E402
from app.rag.embeddings import EmbeddingPipeline, build_embedding_provider  # noqa: E402
from app.rag.ingestion import IngestionPipeline  # noqa: E402
from app.rag.retrieval.factory import RetrievalComponents  # noqa: E402
from app.schemas.enums import RetrievalMode  # noqa: E402
from app.services.agent import build_agent  # noqa: E402
from app.synthetic.storage import load_dataset  # noqa: E402

NOW = datetime(2026, 9, 1, tzinfo=UTC)
DATA = ROOT / "data" / "generated"
EVAL_SET = ROOT / "data" / "evaluation" / "eval_set.jsonl"
CACHE = ROOT / "data" / "evaluation" / "cache"
RESULTS = ROOT / "data" / "evaluation" / "results" / "phase10"
MODEL_METHODS = set("ACDE")  # need the embedding model (and D/E the reranker)


def log(message: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


# --- reproducibility ----------------------------------------------------------------------------


def code_fingerprint() -> str:
    digest = hashlib.sha256()
    files = [
        *sorted((ROOT / "app").rglob("*.py")),
        ROOT / "app" / "security" / "policy.json",
        ROOT / "scripts" / "evaluate.py",
        ROOT / "scripts" / "build_eval_set.py",
    ]
    for path in files:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def hf_revision(model: str | None) -> str | None:
    """The commit of a Hugging Face model in the local cache (what was actually loaded)."""
    if not model:
        return None
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except ImportError:
        return None
    refs = Path(HF_HUB_CACHE) / f"models--{model.replace('/', '--')}" / "refs" / "main"
    return refs.read_text().strip() if refs.exists() else None


def versions() -> dict[str, str]:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for package in (
        "torch",
        "transformers",
        "sentence-transformers",
        "sqlalchemy",
        "pydantic",
        "numpy",
        "matplotlib",
    ):
        try:
            out[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            out[package] = "not installed"
    return out


def models(settings: Settings, quick: bool) -> dict[str, Any]:
    entries = {
        "embedding": None if quick else settings.embedding.model,
        "reranker": None if quick or not settings.reranker.enabled else settings.reranker.model,
        "nli": None if quick else settings.verification.nli_model,
        "injection_classifier": None if quick else settings.security.injection_model,
        "llm_synthesizer": settings.llm.model if settings.llm.enabled else None,
    }
    return {
        role: {"model": name, "revision": hf_revision(name) if name else None}
        for role, name in entries.items()
    }


def dataset_version() -> dict[str, Any]:
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    return {
        "generator_version": manifest["generator_version"],
        "seed": manifest["seed"],
        "window": [manifest["window_start"], manifest["window_end"]],
        "files": manifest["files"],
    }


# --- evaluation database ------------------------------------------------------------------------


def cache_key(settings: Settings, dense: bool) -> str:
    parts = {
        "dataset": dataset_version()["files"],
        "chunking": repr(ChunkingConfig()),
        "embedding": settings.embedding.model if dense else None,
        "code": hashlib.sha256(
            b"".join(p.read_bytes() for p in sorted((ROOT / "app" / "rag").rglob("*.py")))
        ).hexdigest(),
    }
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:16]


def eval_engine(settings: Settings, dense: bool) -> tuple[Engine, dict[str, Any]]:
    CACHE.mkdir(parents=True, exist_ok=True)
    key = cache_key(settings, dense)
    path = CACHE / f"eval-{key}.sqlite"
    marker = path.with_suffix(".json")
    info: dict[str, Any] = {"path": path.relative_to(ROOT).as_posix(), "key": key, "reused": True}
    if not (path.exists() and marker.exists()):
        info["reused"] = False
        partial = path.with_suffix(".partial")
        stage = path.with_suffix(".stage")  # "ingested" once seeding and ingestion finished
        resumable = partial.exists() and stage.exists() and stage.read_text().strip() == "ingested"
        if not resumable:
            partial.unlink(missing_ok=True)
            stage.unlink(missing_ok=True)
        engine = create_engine(f"sqlite:///{partial}", connect_args={"check_same_thread": False})
        started = time.perf_counter()
        if resumable:
            log("resuming the evaluation database (seeded and ingested earlier)")
        else:
            log("building the evaluation database: seed")
            create_schema(engine)
            seed_database(engine, load_dataset(DATA))
            log("ingest")
            IngestionPipeline(ChunkingConfig()).run(engine)
            stage.write_text("ingested", encoding="utf-8")
        if dense:  # incremental: chunks already embedded (same text and model) are skipped
            log(f"embed with {settings.embedding.model} (first run only; several minutes)")
            report = EmbeddingPipeline(build_embedding_provider(settings.embedding)).run(engine)
            info["embedded_chunks"] = report.embedded
        engine.dispose()
        shutil.move(partial, path)
        stage.unlink(missing_ok=True)
        info["build_seconds"] = round(time.perf_counter() - started, 1)
        marker.write_text(json.dumps(info, indent=1), encoding="utf-8")
    engine = create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False})
    return engine, info


def table_counts(engine: Engine) -> dict[str, int]:
    names = set(inspect(engine).get_table_names())
    with engine.connect() as connection:
        return {
            t.name: connection.execute(select(func.count()).select_from(t)).scalar_one()
            for t in Base.metadata.sorted_tables
            if t.name in names
        }


# --- run ------------------------------------------------------------------------------------------


def secrets_from(settings: Settings) -> dict[str, str]:
    """Secret values to test injection answers against; kept in memory only."""
    values = {
        "POSTGRES_PASSWORD": settings.database.password.get_secret_value()
        if settings.database.password
        else "",
        "TOOLS_SQL_PASSWORD": settings.tools.sql_password.get_secret_value()
        if settings.tools.sql_password
        else "",
        "LLM_API_KEY": settings.llm.api_key.get_secret_value() if settings.llm.api_key else "",
    }
    return {k: v for k, v in values.items() if v}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--methods", default="ABCDEF", help="subset of A-F")
    parser.add_argument("--limit", type=int, default=None, help="first N questions only")
    parser.add_argument("--categories", default=None, help="comma-separated categories")
    parser.add_argument("--quick", action="store_true", help="no models: BM25 (B) and agent (F)")
    parser.add_argument("--judge", choices=["auto", "off"], default="auto")
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--resume", action="store_true", help="continue --run-id, keeping its scored results"
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="rebuild summary, report and plots of --run-id from its per_question.jsonl",
    )
    args = parser.parse_args()
    if (args.resume or args.report_only) and not args.run_id:
        parser.error("--resume and --report-only need --run-id")
    if args.report_only:
        return rebuild_report(RESULTS / args.run_id)

    started_at = datetime.now(UTC)
    settings = load_settings()
    methods = [m for m in args.methods.upper() if m in METHODS]
    if args.quick:
        methods = [m for m in methods if m not in MODEL_METHODS]
        settings = settings.model_copy(
            update={
                "retrieval": settings.retrieval.model_copy(update={"mode": RetrievalMode.SPARSE}),
                "reranker": settings.reranker.model_copy(update={"provider": "none"}),
                "verification": settings.verification.model_copy(update={"nli_model": None}),
                "security": settings.security.model_copy(update={"injection_model": None}),
            }
        )
    dense = bool(set(methods) & MODEL_METHODS) or (
        "F" in methods and settings.retrieval.mode != RetrievalMode.SPARSE
    )
    questions = load_eval_set(EVAL_SET)
    if args.categories:
        wanted = {c.strip() for c in args.categories.split(",")}
        questions = [q for q in questions if q.category.value in wanted]
    if args.limit:
        questions = questions[: args.limit]
    run_id = args.run_id or f"{started_at:%Y%m%dT%H%M%SZ}{'-quick' if args.quick else ''}"
    out = RESULTS / run_id
    (out / "plots").mkdir(parents=True, exist_ok=True)
    log(f"run {run_id}: {len(questions)} questions x methods {''.join(methods)}")

    engine, db_info = eval_engine(settings, dense)
    components = RetrievalComponents(engine, settings)
    retrievers = Retrievers(
        dense=components.dense if dense else None,
        bm25=components.bm25,
        hybrid=components.hybrid() if dense else None,
        hybrid_rerank=components.retriever(RetrievalMode.HYBRID_RERANK)
        if dense and settings.reranker.enabled
        else None,
    )
    agent = build_agent(engine, settings, lambda: NOW, components=components, use_sql_reader=False)
    runner = Runner(
        retrievers,
        agent,
        ExtractiveSynthesizer(components.bm25.index),
        lambda: NOW,
        settings.tools.snippet_chars,
    )
    nli = None if args.quick else load_nli(settings.verification)
    judge = None
    judge_status = "not run: judge disabled (--judge off)"
    if args.judge == "auto":
        llm = build_llm(settings.llm)
        if llm is None:
            judge_status = "not run: no LLM configured (LLM_PROVIDER=none)"
        else:
            judge = LLMJudge(llm)
            judge_status = f"run with {llm.name}"
    secrets = secrets_from(settings)
    before = table_counts(engine)

    records: list[Record] = []
    path = out / "per_question.jsonl"
    wanted = {(q.id, m) for q in questions for m in methods}
    if args.resume and path.exists():  # keep what an interrupted run already scored
        kept = [Record.model_validate_json(line) for line in path.read_text("utf-8").splitlines()]
        records = [r for r in kept if (r.question_id, r.method) in wanted]
        log(f"resuming: {len(records)} of {len(wanted)} results already scored")
    done_pairs = {(r.question_id, r.method) for r in records}
    per_question = path.open("w", encoding="utf-8")
    for r in records:
        per_question.write(r.model_dump_json() + "\n")
    for n, question in enumerate(questions, 1):
        for method in methods:
            if (question.id, method) in done_pairs:
                continue
            output = runner.run(method, question)
            record = score(question, method, output, secrets, nli, judge)
            records.append(record)
            per_question.write(record.model_dump_json() + "\n")
            per_question.flush()  # an interrupted run keeps everything scored so far
        if n % 10 == 0 or n == len(questions):
            done = [r for r in records if r.method == methods[-1]]
            log(
                f"{n}/{len(questions)} questions; last method correct so far: "
                f"{sum(r.correct for r in done)}/{len(done)}"
            )
    per_question.close()
    after = table_counts(engine)

    summary = summarize(records, questions)
    summary["integrity"] = {
        "unchanged": before == after,
        "rows_before": before,
        "rows_after": after,
    }
    summary["judge"] = judge_status
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    made = plots(summary, out / "plots")
    manifest = {
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "command": " ".join(sys.argv),
        "resumed": bool(args.resume),
        "methods": {m: METHODS[m] for m in methods},
        "questions": len(questions),
        "eval_set": {
            "path": EVAL_SET.relative_to(ROOT).as_posix(),
            "sha256": file_sha256(EVAL_SET),
        },
        "dataset": dataset_version(),
        "database": db_info,
        "models": models(settings, args.quick),
        "judge": judge_status,
        "clock": NOW.isoformat(),
        "code_sha256": code_fingerprint(),
        "versions": versions(),
        "configuration": json.loads(settings.model_dump_json()),  # secrets are masked
        "env_overrides": sorted(k for k in os.environ if k.startswith(("OPSRAG_", "RETRIEVAL_"))),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    write_report(out, summary, manifest, records, questions, made)
    (RESULTS / "LATEST").write_text(run_id + "\n", encoding="utf-8")
    log(f"wrote {out.relative_to(ROOT)}")
    for method, s in summary["methods"].items():
        log(
            f"{method} {METHODS[method]:<36} correct {s['correctness']}  R@5 {s['recall_5']}  "
            f"MRR {s['mrr']}  faithful {s['faithfulness']}"
        )
    if not summary["integrity"]["unchanged"]:
        log("WARNING: table row counts changed during the run")
        return 1
    return 0


def rebuild_report(out: Path) -> int:
    """Summary, plots and report again from the saved records (no system is re-run)."""
    records = [
        Record.model_validate_json(line)
        for line in (out / "per_question.jsonl").read_text("utf-8").splitlines()
    ]
    ids = {r.question_id for r in records}
    questions = [q for q in load_eval_set(EVAL_SET) if q.id in ids]
    previous = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    summary = summarize(records, questions)
    summary["integrity"], summary["judge"] = previous["integrity"], previous["judge"]
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    manifest["report_rebuilt_at"] = datetime.now(UTC).isoformat()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    made = plots(summary, out / "plots")
    write_report(out, summary, manifest, records, questions, made)
    log(f"rebuilt the report of {out.relative_to(ROOT)} from {len(records)} records")
    return 0


def score(
    question: EvalQuestion,
    method: str,
    output: Any,
    secrets: dict[str, str],
    nli: Any,
    judge: LLMJudge | None,
) -> Record:
    check = check_answer(question, output.answer, secrets)
    declined = check.abstained or output.declined
    gen = generation_scores(question, output.answer, output.context, declined)
    relevance = question.relevance
    retrieval = retrieval_scores(output.ranked, relevance).model_dump() if relevance else None
    if retrieval:
        retrieval["recall"] = {str(k): v for k, v in retrieval["recall"].items()}
        retrieval["hit"] = {str(k): v for k, v in retrieval["hit"].items()}
    error = classify(question, output, check, gen)
    judged = None
    if judge is not None:
        judged = judge.grade(question, output.answer, output.context).model_dump()
    c = question.checks
    checks = check.model_dump()
    checks["has_positive"] = bool(c.facts or c.first_id or c.id_set)
    # never persist secret values: forbidden hits are reported by name only
    return Record(
        question_id=question.id,
        category=question.category.value,
        difficulty=question.difficulty.value,
        origin=question.origin,
        role=question.role,
        method=method,
        question=question.question,
        answer=output.answer,
        correct=check.correct,
        abstained=declined,
        checks=checks,
        retrieval=retrieval,
        generation=gen.model_dump(),
        nli_faithfulness=nli_faithfulness(nli, output.answer, output.context, declined)
        if nli is not None
        else None,
        judge=judged,
        error=error.model_dump(mode="json"),
        ranked=output.ranked[:10],
        context=[f"{i.label}:{i.source_id}" for i in output.context],
        query_type=output.query_type,
        plan=output.plan,
        tools=output.tools,
        confidence=output.confidence,
        security=output.security,
        latency_ms=output.latency_ms,
    )


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.0f}%"


def write_report(
    out: Path,
    summary: dict[str, Any],
    manifest: dict[str, Any],
    records: list[Record],
    questions: list[EvalQuestion],
    made: list[Path],
) -> None:
    eval_set, dataset = manifest["eval_set"], manifest["dataset"]
    categories = ", ".join(f"{k} {v}" for k, v in summary["categories"].items())
    model_list = "; ".join(
        f"{role} {m['model']} ({(m['revision'] or 'n/a')[:10]})"
        for role, m in manifest["models"].items()
        if m["model"]
    )
    integrity = "unchanged" if summary["integrity"]["unchanged"] else "CHANGED"
    graded = next(iter(summary["methods"].values()))["retrieval_questions"]
    lines = [
        f"# Phase 10 evaluation report ({manifest['run_id']})",
        "",
        "Generated by `scripts/evaluate.py`. Every number below comes from this run's",
        "`per_question.jsonl`; nothing is typed in.",
        "",
        f"- Questions: {manifest['questions']} ({categories})",
        f"- Evaluation set: `{eval_set['path']}` (sha256 {eval_set['sha256'][:12]}...)",
        f"- Dataset: generator {dataset['generator_version']}, seed {dataset['seed']}",
        f"- Clock: {manifest['clock']}; code sha256 {manifest['code_sha256'][:12]}...",
        f"- Models: {model_list or 'none (quick mode: no models loaded)'}",
        f"- LLM-as-judge: {manifest['judge']}",
        f"- Read-only check: table row counts {integrity}",
        "",
        "## Ablation (deterministic metrics unless marked)",
        "",
        f"Retrieval columns are averaged over the {graded} questions that have gold sources;",
        'the answer columns over all questions. "Faithful (NLI)" is model-based; LLM-as-judge',
        "columns appear only when a judge ran.",
        "",
        ablation_table(summary),
        "",
        "Recall@k is capped (relevant found in the top k / min(relevant, k)), so with several",
        "relevant records Recall@1 can exceed Recall@5; Hit@5 (any relevant record in the top",
        "5) is monotone. Answers with a citation that does not ground its claim, and how many",
        'of them carry a verifier note "Sources disagree ... [E#] state(s) otherwise" (a note',
        "that cites the disagreeing source on purpose): "
        + "; ".join(
            f"{m} {s.get('citation_issue_answers', 0)} "
            f"({s.get('citation_issues_with_disagreement_note', 0)})"
            for m, s in summary["methods"].items()
        )
        + ".",
        "",
    ]
    judged = [m for m, s in summary["methods"].items() if "judge_correctness" in s]
    if judged:
        lines += [
            "### LLM-as-judge (model opinion, not deterministic)",
            "",
            "| Method | Correctness | Faithfulness | Context relevance |",
            "|---|---:|---:|---:|",
        ]
        for m in judged:
            s = summary["methods"][m]
            scores = [s[f"judge_{k}"] for k in ("correctness", "faithfulness", "context_relevance")]
            lines.append(f"| {m} | " + " | ".join(f"{v:.3f}" for v in scores) + " |")
        lines.append("")
    lines += [
        "| Method | Declined no-answer | Declined answerable | Median latency (ms) |",
        "|---|---:|---:|---:|",
    ]
    for m, s in summary["methods"].items():
        lines.append(
            f"| {m} | {_pct(s['no_answer_declined'])} | {_pct(s['false_decline_rate'])} | "
            f"{s['latency_ms_median']} |"
        )
    lines += [
        "",
        *[f"![{p.stem}](plots/{p.name})" for p in made],
        "",
        "## Answer correctness by category",
        "",
        grouped_table(summary, "by_category"),
        "",
        "## By difficulty",
        "",
        grouped_table(summary, "by_difficulty", ["easy", "medium", "hard"]),
        "",
        "## By origin",
        "",
        "`retrieval_benchmark` questions were used while developing retrieval; `generated`",
        "and `handwritten` ones were not.",
        "",
        grouped_table(summary, "by_origin"),
        "",
        "## Error analysis",
        "",
        "Primary error class per failed question (rules in `app/evaluation/errors.py`):",
        "",
        error_table(summary),
        "",
    ]
    last = sorted(summary["methods"])[-1]
    failures = [r for r in records if r.method == last and r.error.get("failed")]
    lines += [
        f"### Every failure of method {last} ({len(failures)})",
        "",
        "| Question | Category | Primary | Also | Detail |",
        "|---|---|---|---|---|",
    ]
    by_id = {q.id: q for q in questions}
    for r in failures:
        text = by_id[r.question_id].question.replace("|", "\\|")[:90]
        also = ", ".join(r.error["classes"][1:])
        detail = r.error["detail"][:80]
        lines.append(
            f"| {r.question_id}: {text} | {r.category} | {r.error['primary']} | {also} | {detail} |"
        )
    counts = Counter(r.error["primary"] for r in failures)
    primary = ", ".join(f"{k} {v}" for k, v in counts.most_common())
    lines += ["", f"Primary classes for {last}: {primary}", ""]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
