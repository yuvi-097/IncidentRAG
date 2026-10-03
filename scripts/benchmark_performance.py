"""Performance benchmarks: ingestion, embedding, retrieval, reranking, LLM and end-to-end.

    python scripts/benchmark_performance.py                       # every section
    python scripts/benchmark_performance.py --sections retrieval,rerank --run-id my-run
    python scripts/benchmark_performance.py --quick               # small samples, minutes

Everything runs in this process against the configured database (``POSTGRES_*``) with the
configured models, on the machine described in the results. Sections:

- ``ingestion``: reading the sources, then parsing + cleaning + chunking them (repeated,
  no writes to the configured database); and a full ingest, writes included, into a fresh
  SQLite file seeded from ``data/generated``;
- ``embedding``: document embedding throughput per batch size on a fixed sample of chunks
  (model only, no writes), and the latency of embedding one query;
- ``retrieval``: first-stage latency of dense, BM25 and hybrid search (top 30, filtered to
  the admin role's labels, as the tools do) over the evaluation questions;
- ``rerank``: the cross-encoder over the hybrid candidates, for several candidate counts,
  sequence lengths and batch sizes, with retrieval quality on the 63 labelled questions of
  ``data/evaluation/retrieval_benchmark.jsonl``, so a faster setting is never judged on
  speed alone;
- ``llm``: completion latency and tokens, when an LLM is configured (otherwise recorded
  as not measured, with the reason);
- ``agent``: building the agent (loading the models), then end-to-end answers to every
  fourth evaluation question, with each stage's share.

Percentiles are nearest-rank, like the live metrics. Results are written after every
section to ``data/benchmarks/performance/<run-id>/results.json``, with ``report.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sqlalchemy import create_engine, select  # noqa: E402
from sqlalchemy.engine import Engine  # noqa: E402

from app.config import Settings, load_settings  # noqa: E402
from app.database.models import DocumentChunk  # noqa: E402
from app.database.session import create_db_engine  # noqa: E402
from app.evaluation.eval_set import load_eval_set  # noqa: E402
from app.evaluation.performance import (  # noqa: E402
    environment,
    keep_awake,
    latency_stats,
    rate,
    timed,
)
from app.evaluation.performance_report import render_report  # noqa: E402
from app.evaluation.retrieval import (  # noqa: E402
    chunk_catalog,
    load_benchmark,
    ndcg_at,
    resolve_relevant,
)
from app.llm import build_llm  # noqa: E402
from app.llm.base import ChatMessage  # noqa: E402
from app.observability.structured_logging import configure_logging  # noqa: E402
from app.rag.chunking import ChunkingConfig  # noqa: E402
from app.rag.embeddings import build_embedding_provider  # noqa: E402
from app.rag.embeddings.text import embedding_text  # noqa: E402
from app.rag.ingestion import IngestionPipeline  # noqa: E402
from app.rag.ingestion.persistence import ensure_chunk_table  # noqa: E402
from app.rag.ingestion.sources import load_sources  # noqa: E402
from app.rag.reranking.cross_encoder import (  # noqa: E402
    CrossEncoderReranker,
    SentenceTransformerCrossEncoder,
)
from app.rag.retrieval.factory import RetrievalComponents  # noqa: E402
from app.rag.store import ChunkFilter  # noqa: E402
from app.schemas.enums import SourceType  # noqa: E402
from app.security.principal import principal_for_role  # noqa: E402
from app.services.agent import build_agent  # noqa: E402
from app.synthetic.storage import load_dataset  # noqa: E402

SECTIONS = (
    "ingestion",
    "embedding",
    "retrieval",
    "rerank",
    "threads",
    "nli",
    "llm",
    "agent",
    "compare",
)
# (candidates, max_length, batch_size).
RERANK_VARIANTS = (
    (30, 512, 4),  # the default since Phase 12
    (30, 512, 16),  # the default before
    (30, 384, 4),
    (30, 256, 4),
    (20, 512, 4),
    (10, 512, 4),
)
EMBED_BATCHES = (4, 8, 16, 32)


def section_ingestion(engine: Engine, repeat: int, data_dir: Path) -> dict[str, Any]:
    config = ChunkingConfig()
    pipeline = IngestionPipeline(config)
    read_ms, chunk_ms, sources_n, chunks_n = [], [], 0, 0
    for _ in range(repeat):
        with engine.connect() as connection:
            sources, ms = timed(lambda c=connection: list(load_sources(c, set(SourceType))))
        read_ms.append(ms)
        (chunks, _report), ms = timed(lambda s=sources: pipeline.chunk_all(s))
        chunk_ms.append(ms)
        sources_n, chunks_n = len(sources), len(chunks)
    read, chunk = latency_stats(read_ms), latency_stats(chunk_ms)
    out: dict[str, Any] = {
        "sources": sources_n,
        "chunks": chunks_n,
        "read_ms": read,
        "chunk_ms": chunk,
        "chunking_sources_per_s": rate(sources_n, (chunk["p50"] or 0) / 1000),
        "chunking_chunks_per_s": rate(chunks_n, (chunk["p50"] or 0) / 1000),
        "repeat": repeat,
        "database": engine.dialect.name,
    }
    # The whole pipeline, writes included, into a fresh database: a scratch SQLite file,
    # so the configured database (and its embeddings) is never touched.
    dataset = load_dataset(data_dir)
    full_ms, seed_ms = [], []
    for _ in range(repeat):
        with tempfile.TemporaryDirectory() as scratch:
            scratch_engine = create_engine(f"sqlite:///{Path(scratch) / 'ingest.db'}")
            try:
                ensure_chunk_table(scratch_engine)
                from app.database.seed import seed_database

                _, ms = timed(lambda e=scratch_engine: seed_database(e, dataset))
                seed_ms.append(ms)
                (_, report), ms = timed(lambda e=scratch_engine: IngestionPipeline(config).run(e))
                full_ms.append(ms)
            finally:
                scratch_engine.dispose()
    full = latency_stats(full_ms)
    out["full_ingest_sqlite"] = {
        "ms": full,
        "seed_ms": latency_stats(seed_ms),
        "sources_per_s": rate(report.total_sources, (full["p50"] or 0) / 1000),
        "chunks_per_s": rate(report.total_chunks, (full["p50"] or 0) / 1000),
    }
    return out


def _chunk_sample(engine: Engine, size: int) -> list[str]:
    c = DocumentChunk
    query = select(c.id, c.source_type, c.title, c.section, c.service_id, c.content).order_by(c.id)
    with engine.connect() as connection:
        rows = [dict(r) for r in connection.execute(query).mappings()]
    step = max(1, len(rows) // size)
    return [embedding_text(r) for r in rows[::step][:size]]


def section_embedding(
    engine: Engine,
    settings: Settings,
    sample: int,
    queries: list[str],
    batch_sizes: tuple[int, ...] = EMBED_BATCHES,
) -> dict[str, Any]:
    """Throughput per batch size. The sample is cut into slices of 64 texts and every batch
    size embeds each slice, in an order that rotates from slice to slice, so a machine whose
    speed drifts (heat, turbo) does not favour whichever batch size runs first."""
    texts = _chunk_sample(engine, sample)
    providers = {
        size: build_embedding_provider(settings.embedding.model_copy(update={"batch_size": size}))
        for size in batch_sizes
    }
    for provider in providers.values():
        provider.embed_documents(texts[:8])  # warm-up
    seconds: dict[int, list[float]] = defaultdict(list)
    slices = [texts[i : i + 64] for i in range(0, len(texts), 64)]
    for index, part in enumerate(slices):
        shift = index % len(batch_sizes)
        for size in batch_sizes[shift:] + batch_sizes[:shift]:
            _, ms = timed(lambda p=providers[size], t=part: p.embed_documents(t))
            seconds[size].append(ms / 1000)
    by_batch = [
        {
            "batch_size": size,
            "chunks": len(texts),
            "seconds": round(sum(seconds[size]), 2),
            "slice_seconds": [round(s, 2) for s in seconds[size]],
            "chunks_per_s": rate(len(texts), sum(seconds[size])),
        }
        for size in batch_sizes
    ]
    provider = providers[
        settings.embedding.batch_size
        if settings.embedding.batch_size in providers
        else batch_sizes[0]
    ]
    tokens = provider.count_tokens(texts) or []
    limit = provider.max_input_tokens or 0
    query_ms = [timed(lambda q=q: provider.embed_query(q))[1] for q in queries]
    return {
        "model": settings.embedding.model,
        "configured_batch_size": settings.embedding.batch_size,
        "slices": len(slices),
        "sample_tokens_mean": round(sum(tokens) / len(tokens), 1) if tokens else None,
        "sample_truncated": sum(t > limit for t in tokens) if limit else None,
        "by_batch_size": by_batch,
        "query_embedding_ms": latency_stats(query_ms),
    }


def section_retrieval(
    components: RetrievalComponents, queries: list[str], top_k: int = 30
) -> dict[str, Any]:
    admin = principal_for_role("benchmark-admin", "admin")
    filters = ChunkFilter(access=admin.chunk_access())
    out: dict[str, Any] = {"queries": len(queries), "top_k": top_k, "filter": "admin labels"}
    retrievers = {"dense": components.dense, "bm25": components.bm25, "hybrid": components.hybrid()}
    for name, retriever in retrievers.items():
        for q in queries[:3]:  # warm-up (model, index, connections)
            retriever.search(q, top_k, filters)
        durations = [
            timed(lambda q=q, r=retriever: r.search(q, top_k, filters))[1] for q in queries
        ]
        out[name] = latency_stats(durations)
    return out


def _ranking(retrieved: list[str], relevant: set[str]) -> dict[str, float]:
    """The retrieval benchmark's definitions (``app.evaluation.retrieval.evaluate``)."""
    first = next((rank for rank, doc in enumerate(retrieved, 1) if doc in relevant), None)
    found = relevant & set(retrieved[:5])
    return {
        "ndcg_10": ndcg_at(retrieved, relevant, 10),
        "recall_5": len(found) / min(len(relevant), 5),
        "hit_5": 1.0 if found else 0.0,
        "mrr": 1 / first if first else 0.0,
    }


def section_rerank(
    engine: Engine,
    settings: Settings,
    components: RetrievalComponents,
    benchmark: Path,
    limit: int,
    variants_to_run: tuple[tuple[int, int, int], ...] = RERANK_VARIANTS,
) -> dict[str, Any]:
    """Every variant reranks the same hybrid candidates of each question, in an order that
    rotates from question to question (so drift in machine speed is spread evenly)."""
    questions = load_benchmark(benchmark)[:limit]
    catalog = chunk_catalog(engine)
    hybrid = components.hybrid()
    rerankers = {
        (length, batch): CrossEncoderReranker(
            SentenceTransformerCrossEncoder(
                settings.reranker.model, settings.reranker.device, batch, length
            )
        )
        for _, length, batch in variants_to_run
    }
    pool_size = max(c for c, _, _ in variants_to_run)
    warm = hybrid.search(questions[0].question, pool_size)
    for reranker in rerankers.values():
        reranker.rerank(questions[0].question, warm, 10)  # warm-up
    first_stage_ms: list[float] = []
    durations: dict[tuple[int, int, int], list[float]] = defaultdict(list)
    quality: dict[tuple[int, int, int], list[dict[str, float]]] = defaultdict(list)
    for index, question in enumerate(questions):
        relevant = resolve_relevant(question, catalog)
        pool, ms = timed(lambda q=question: hybrid.search(q.question, pool_size))
        first_stage_ms.append(ms)
        shift = index % len(variants_to_run)
        for variant in variants_to_run[shift:] + variants_to_run[:shift]:
            candidates, length, batch = variant
            reranked, ms = timed(
                lambda r=rerankers[(length, batch)], q=question, c=pool[:candidates]: r.rerank(
                    q.question, c, 10
                )
            )
            durations[variant].append(ms)
            quality[variant].append(_ranking([c.document_id for c in reranked], relevant))
    configured = (
        settings.retrieval.rerank_candidates,
        settings.reranker.max_length,
        settings.reranker.batch_size,
    )
    variants = []
    for variant in variants_to_run:
        scores = quality[variant]
        variants.append(
            {
                "candidates": variant[0],
                "max_length": variant[1],
                "batch_size": variant[2],
                "configured": variant == configured,
                "rerank_ms": latency_stats(durations[variant]),
                **{
                    name: round(sum(s[name] for s in scores) / len(scores), 4)
                    for name in ("ndcg_10", "recall_5", "hit_5", "mrr")
                },
            }
        )
    return {
        "model": settings.reranker.model,
        "questions": len(questions),
        "benchmark": str(benchmark.relative_to(ROOT)),
        "first_stage_ms": latency_stats(first_stage_ms),
        "variants": variants,
    }


THREAD_COUNTS = (4, 6, 8, 10)


def section_threads(
    settings: Settings, components: RetrievalComponents, benchmark: Path, limit: int
) -> dict[str, Any]:
    """One request's cross-encoder latency per torch thread count (the configured rerank
    settings), rotating the thread count per question. Outputs do not depend on it."""
    import torch

    default = torch.get_num_threads()
    questions = load_benchmark(benchmark)[:limit]
    hybrid = components.hybrid()
    reranker = CrossEncoderReranker(
        SentenceTransformerCrossEncoder(
            settings.reranker.model,
            settings.reranker.device,
            settings.reranker.batch_size,
            settings.reranker.max_length,
        )
    )
    candidates = settings.retrieval.rerank_candidates
    durations: dict[int, list[float]] = defaultdict(list)
    try:
        reranker.rerank(questions[0].question, hybrid.search(questions[0].question, candidates), 10)
        for index, question in enumerate(questions):
            pool = hybrid.search(question.question, candidates)
            shift = index % len(THREAD_COUNTS)
            for threads in THREAD_COUNTS[shift:] + THREAD_COUNTS[:shift]:
                torch.set_num_threads(threads)
                _, ms = timed(lambda q=question, c=pool: reranker.rerank(q.question, c, 10))
                durations[threads].append(ms)
    finally:
        torch.set_num_threads(default)
    return {
        "default_threads": default,
        "questions": len(questions),
        "rerank": f"{candidates} / {settings.reranker.max_length} / {settings.reranker.batch_size}",
        "by_threads": {str(t): latency_stats(durations[t]) for t in THREAD_COUNTS},
    }


class _RecordingNLI:
    """Passes NLI calls through and keeps their pairs, grouped by agent run."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.name = inner.name
        self.runs: list[list[list[tuple[str, str]]]] = []

    def predict(self, pairs: Sequence[tuple[str, str]]) -> list[dict[str, float]]:
        self.runs[-1].append(list(pairs))
        return self.inner.predict(pairs)


def section_nli(engine: Engine, settings: Settings, questions: list[Any]) -> dict[str, Any]:
    """Would batching the NLI calls of one answer help? The agent's real calls are recorded,
    then each answer's pairs are scored one call per claim (as the verifier does) and in
    one call, alternating which goes first."""
    agent = build_agent(engine, settings)
    if agent.verifier.nli is None:
        return {"measured": False, "reason": "no NLI model is configured"}
    recorder = _RecordingNLI(agent.verifier.nli)
    agent.verifier.nli = recorder  # type: ignore[assignment]
    for question in questions:
        recorder.runs.append([])
        agent.run(question.question, principal_for_role("benchmark", question.role))
    runs = [calls for calls in recorder.runs if calls]
    nli = recorder.inner
    nli.predict(runs[0][0])  # warm-up
    per_claim, batched = [], []
    for index, calls in enumerate(runs):
        everything = [pair for call in calls for pair in call]
        modes = [("per_claim", None), ("batched", None)]
        for mode, _ in modes if index % 2 == 0 else reversed(modes):
            if mode == "per_claim":
                _, ms = timed(lambda c=calls: [nli.predict(call) for call in c])
                per_claim.append(ms)
            else:
                _, ms = timed(lambda e=everything: nli.predict(e))
                batched.append(ms)
    calls_per_answer = [len(c) for c in runs]
    pairs_per_answer = [sum(len(call) for call in c) for c in runs]
    return {
        "model": nli.name,
        "answers": len(runs),
        "calls_per_answer_mean": round(sum(calls_per_answer) / len(runs), 2),
        "pairs_per_answer_mean": round(sum(pairs_per_answer) / len(runs), 2),
        "per_claim_ms": latency_stats(per_claim),
        "batched_ms": latency_stats(batched),
        "per_claim_total_s": round(sum(per_claim) / 1000, 2),
        "batched_total_s": round(sum(batched) / 1000, 2),
    }


def section_llm(settings: Settings, questions: list[str]) -> dict[str, Any]:
    llm = build_llm(settings.llm)
    if llm is None:
        return {"measured": False, "reason": "LLM_PROVIDER=none: no LLM is configured"}
    durations, input_tokens, output_tokens, failures = [], [], [], 0
    for question in questions:
        messages = [
            ChatMessage(role="system", content="Answer in one sentence."),
            ChatMessage(role="user", content=question),
        ]
        try:
            reply, ms = timed(lambda m=messages: llm.complete(m))
        except Exception:
            failures += 1
            continue
        durations.append(ms)
        if reply.input_tokens is not None:
            input_tokens.append(reply.input_tokens)
        if reply.output_tokens is not None:
            output_tokens.append(reply.output_tokens)
    return {
        "measured": True,
        "model": llm.name,
        "calls": len(questions),
        "failures": failures,
        "latency_ms": latency_stats(durations),
        "input_tokens_mean": round(sum(input_tokens) / len(input_tokens), 1)
        if input_tokens
        else None,
        "output_tokens_mean": round(sum(output_tokens) / len(output_tokens), 1)
        if output_tokens
        else None,
    }


def section_agent(engine: Engine, settings: Settings, questions: list[Any]) -> dict[str, Any]:
    agent, build_ms = timed(lambda: build_agent(engine, settings))
    first = questions[0]
    _, cold_ms = timed(
        lambda: agent.run(first.question, principal_for_role("benchmark", first.role))
    )
    totals: list[float] = []
    stages: dict[str, list[float]] = defaultdict(list)
    by_category: dict[str, list[float]] = defaultdict(list)
    tool_calls, errors = 0, 0
    for question in questions:
        principal = principal_for_role(f"benchmark-{question.role}", question.role)
        state = agent.run(question.question, principal)
        totals.append(state.latency_ms["total"])
        by_category[question.category].append(state.latency_ms["total"])
        for stage, ms in state.latency_ms.items():
            if stage != "total":
                stages[stage].append(ms)
        tool_calls += len(state.tool_results)
        errors += len(state.errors)
    total_time = sum(totals)
    return {
        "build_ms": round(build_ms, 1),
        "first_question_ms": round(cold_ms, 1),
        "questions": len(questions),
        "total_ms": latency_stats(totals),
        "stages": {
            stage: {**latency_stats(values), "share": round(sum(values) / total_time, 3)}
            for stage, values in sorted(stages.items(), key=lambda kv: -sum(kv[1]))
        },
        "by_category": {c: latency_stats(v) for c, v in sorted(by_category.items())},
        "tool_calls": tool_calls,
        "errors": errors,
        "retrieval_mode": settings.retrieval.mode.value,
        "llm": settings.llm.provider,
    }


def with_overrides(settings: Settings, overrides: str) -> Settings:
    """``settings`` with ``group.field=value`` overrides, e.g. ``reranker.batch_size=16``."""
    updated = settings
    for item in filter(None, (o.strip() for o in overrides.split(","))):
        key, _, raw = item.partition("=")
        group, _, name = key.partition(".")
        section = getattr(updated, group)
        current = getattr(section, name)
        value: Any = type(current)(raw) if current is not None else raw
        updated = updated.model_copy(update={group: section.model_copy(update={name: value})})
    return updated


def _answer_key(state: Any) -> tuple[Any, ...]:
    return (state.final_answer, [c.label for c in state.citations], state.confidence.value)


def section_compare(
    engine: Engine, settings: Settings, before: str, questions: list[Any]
) -> dict[str, Any]:
    """The agent with the earlier settings (``before`` overrides) and with the current ones,
    answering every question in turn, alternating which goes first. Changes made only for
    speed must leave every answer identical; the answers are compared."""
    agents = {
        "before": build_agent(engine, with_overrides(settings, before)),
        "after": build_agent(engine, settings),
    }
    first = questions[0]
    for agent in agents.values():  # warm-up: the first question pays for lazy loading
        agent.run(first.question, principal_for_role("benchmark", first.role))
    totals: dict[str, list[float]] = defaultdict(list)
    stages: dict[str, dict[str, list[float]]] = {name: defaultdict(list) for name in agents}
    different: list[str] = []
    for index, question in enumerate(questions):
        principal = principal_for_role(f"benchmark-{question.role}", question.role)
        order = list(agents) if index % 2 == 0 else list(reversed(agents))
        keys = {}
        for name in order:
            state = agents[name].run(question.question, principal)
            totals[name].append(state.latency_ms["total"])
            for stage, ms in state.latency_ms.items():
                if stage != "total":
                    stages[name][stage].append(ms)
            keys[name] = _answer_key(state)
        if keys["before"] != keys["after"]:
            different.append(question.id)
    return {
        "before_overrides": before,
        "questions": len(questions),
        "answers_different": different,
        "total_ms": {name: latency_stats(values) for name, values in totals.items()},
        "total_seconds": {name: round(sum(v) / 1000, 1) for name, v in totals.items()},
        "stages_p50": {
            name: {stage: latency_stats(v)["p50"] for stage, v in by_stage.items()}
            for name, by_stage in stages.items()
        },
    }


def configuration(settings: Settings) -> dict[str, Any]:
    return {
        "retrieval_mode": settings.retrieval.mode.value,
        "embedding_model": settings.embedding.model,
        "embedding_batch_size": settings.embedding.batch_size,
        "reranker_model": settings.reranker.model if settings.reranker.enabled else None,
        "rerank_candidates": settings.retrieval.rerank_candidates,
        "reranker_max_length": settings.reranker.max_length,
        "reranker_batch_size": settings.reranker.batch_size,
        "nli_model": settings.verification.nli_model,
        "llm": settings.llm.provider,
        "db_pool_size": settings.database.pool_size,
        "db_max_overflow": settings.database.max_overflow,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sections", default=",".join(SECTIONS))
    parser.add_argument("--run-id", default=datetime.now(UTC).strftime("perf-%Y%m%d-%H%M%S"))
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "benchmarks" / "performance")
    parser.add_argument("--quick", action="store_true", help="small samples (a smoke test)")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "generated")
    parser.add_argument(
        "--rerank-variants",
        default="",
        help="candidates/max_length/batch,... (default: the built-in list)",
    )
    parser.add_argument(
        "--rerank-questions", type=int, default=1000, help="labelled questions to rerank"
    )
    parser.add_argument(
        "--threads", type=int, default=0, help="torch intra-op threads (0: torch's default)"
    )
    parser.add_argument(
        "--embed-batches",
        default="",
        help="embedding batch sizes to compare, e.g. 4,8,32 (default: the built-in list)",
    )
    parser.add_argument("--embed-sample", type=int, default=256, help="chunks to embed")
    parser.add_argument(
        "--before",
        default="",
        help="compare: the earlier settings as overrides, e.g. reranker.batch_size=16",
    )
    args = parser.parse_args(argv)
    sections = [s.strip() for s in args.sections.split(",") if s.strip()]
    unknown = sorted(set(sections) - set(SECTIONS))
    if unknown:
        parser.error(f"unknown sections: {', '.join(unknown)}")

    settings = load_settings()
    configure_logging("WARNING", settings.app.log_format)
    results_awake = keep_awake()
    batches = tuple(int(x) for x in args.embed_batches.split(",") if x.strip()) or EMBED_BATCHES
    if args.threads:
        import torch

        torch.set_num_threads(args.threads)
    variants = (
        tuple(
            tuple(int(x) for x in v.split("/"))  # type: ignore[misc]
            for v in args.rerank_variants.split(",")
            if v.strip()
        )
        or RERANK_VARIANTS
    )
    run_dir = args.out / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "results.json"
    results: dict[str, Any] = json.loads(path.read_text("utf-8")) if path.exists() else {}
    engine = create_db_engine(settings.database, application_name="opsrag-benchmark")
    eval_set = load_eval_set(ROOT / "data" / "evaluation" / "eval_set.jsonl")
    queries = [q.question for q in eval_set]
    agent_questions = eval_set[::4]
    if args.quick:
        queries, agent_questions = queries[:12], agent_questions[:6]
    components = RetrievalComponents(engine, settings)

    def save() -> None:
        path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        (run_dir / "report.md").write_text(render_report(results), encoding="utf-8")

    results.setdefault("run_id", args.run_id)
    results["quick"] = args.quick
    results["environment"] = {**environment(engine), "kept_awake": results_awake}
    results["configuration"] = configuration(settings)
    runners: dict[str, Callable[[], dict[str, Any]]] = {
        "ingestion": lambda: section_ingestion(engine, 1 if args.quick else 3, args.data_dir),
        "embedding": lambda: section_embedding(
            engine, settings, 32 if args.quick else args.embed_sample, queries, batches
        ),
        "retrieval": lambda: section_retrieval(components, queries),
        "rerank": lambda: section_rerank(
            engine,
            settings,
            components,
            ROOT / "data" / "evaluation" / "retrieval_benchmark.jsonl",
            4 if args.quick else args.rerank_questions,
            variants,
        ),
        "threads": lambda: section_threads(
            settings,
            components,
            ROOT / "data" / "evaluation" / "retrieval_benchmark.jsonl",
            4 if args.quick else min(args.rerank_questions, 30),
        ),
        "nli": lambda: section_nli(engine, settings, agent_questions),
        "llm": lambda: section_llm(settings, queries[:5] if args.quick else queries[:30]),
        "agent": lambda: section_agent(engine, settings, agent_questions),
        "compare": lambda: section_compare(engine, settings, args.before, agent_questions),
    }
    try:
        for name in sections:
            started = time.perf_counter()
            print(f"== {name}", flush=True)
            results[name] = runners[name]()
            results[name]["section_seconds"] = round(time.perf_counter() - started, 1)
            results.setdefault("finished_sections", {})[name] = datetime.now(UTC).isoformat()
            save()
            print(f"   done in {results[name]['section_seconds']} s", flush=True)
    finally:
        engine.dispose()
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
