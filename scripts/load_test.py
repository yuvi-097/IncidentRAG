"""A lightweight load test: the running API under concurrent questions.

    python scripts/load_test.py --user alex.rivera                        # 1, 2, 4, 8 at a time
    python scripts/load_test.py --concurrency 1,4,16 --requests 32 --token <API token>

The API must be running (``uvicorn app.main:app``). Questions are a fixed spread of the
evaluation set (every category), sent to ``POST /api/agent/ask`` as one user: a demo user
through ``X-OpsRAG-User`` (local API with SECURITY_ALLOW_USER_HEADER on) or an API token.

1. **Baseline.** Each distinct question is asked once, one at a time. This also loads the
   models, so no level pays for start-up. Its answers are the reference.
2. **Levels.** For each concurrency level, ``--requests`` questions (the list, cycled) are
   sent with that many in flight at once. Per level: completed and failed requests,
   throughput, client-side latency (p50 / p95 / p99), the agent's own time reported by
   the server (so queueing shows as the difference), and **answers that changed** from
   the baseline. Without an LLM the answers are deterministic, so a changed answer under
   concurrency would mean requests interfere with each other.

Client and server run on the same machine unless ``--url`` says otherwise; the client's
own CPU use is small (it waits on HTTP). Results go to
``data/benchmarks/load/<run-id>/results.json`` and ``report.md``.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.evaluation.eval_set import load_eval_set  # noqa: E402
from app.evaluation.performance import (  # noqa: E402
    environment,
    keep_awake,
    latency_stats,
    rate,
)
from app.evaluation.performance_report import render_load_report  # noqa: E402


def pick_questions(count: int) -> list[str]:
    """``count`` evaluation questions spread evenly over the set (so every category)."""
    questions = load_eval_set(ROOT / "data" / "evaluation" / "eval_set.jsonl")
    step = max(1, len(questions) // count)
    return [q.question for q in questions[::step][:count]]


def fingerprint(body: dict[str, Any]) -> str:
    """What must not change between identical questions: answer, citations, confidence."""
    key = json.dumps(
        [
            body.get("answer"),
            [c.get("label") for c in body.get("citations", [])],
            body.get("confidence"),
        ],
        sort_keys=True,
    )
    return hashlib.sha256(key.encode()).hexdigest()[:16]


async def ask(client: httpx.AsyncClient, question: str) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        response = await client.post("/api/agent/ask", json={"question": question})
        elapsed = (time.perf_counter() - started) * 1000
        record: dict[str, Any] = {
            "question": question,
            "status": response.status_code,
            "client_ms": round(elapsed, 1),
            "request_id": response.headers.get("X-Request-ID"),
        }
        if response.status_code == 200:
            body = response.json()
            record["server_agent_ms"] = body.get("latency_ms", {}).get("total")
            record["fingerprint"] = fingerprint(body)
        return record
    except httpx.HTTPError as exc:
        return {
            "question": question,
            "status": 0,
            "client_ms": round((time.perf_counter() - started) * 1000, 1),
            "error": type(exc).__name__,
        }


async def run_level(
    client: httpx.AsyncClient, questions: list[str], concurrency: int, requests: int, offset: int
) -> tuple[list[dict[str, Any]], float]:
    queue = [questions[(offset + i) % len(questions)] for i in range(requests)]
    semaphore = asyncio.Semaphore(concurrency)

    async def one(question: str) -> dict[str, Any]:
        async with semaphore:
            return await ask(client, question)

    started = time.perf_counter()
    records = await asyncio.gather(*(one(q) for q in queue))
    return list(records), time.perf_counter() - started


def summarize_level(
    concurrency: int, records: list[dict[str, Any]], seconds: float, baseline: dict[str, str]
) -> dict[str, Any]:
    ok = [r for r in records if r["status"] == 200]
    mismatches = [
        r["question"] for r in ok if baseline.get(r["question"]) not in (None, r["fingerprint"])
    ]
    statuses: dict[str, int] = {}
    for r in records:
        statuses[str(r["status"])] = statuses.get(str(r["status"]), 0) + 1
    return {
        "concurrency": concurrency,
        "requests": len(records),
        "completed": len(ok),
        "errors": len(records) - len(ok),
        "statuses": statuses,
        "wall_seconds": round(seconds, 2),
        "throughput_rps": rate(len(ok), seconds),
        "client_ms": latency_stats([r["client_ms"] for r in ok]),
        "server_agent_ms": latency_stats(
            [r["server_agent_ms"] for r in ok if r.get("server_agent_ms") is not None]
        ),
        "answer_mismatches": len(mismatches),
        "mismatched_questions": sorted(set(mismatches)),
    }


async def main_async(
    args: argparse.Namespace, transport: httpx.AsyncBaseTransport | None = None
) -> dict[str, Any]:
    """Baseline, then every level. ``transport`` lets tests run it against the app in-process."""
    headers = (
        {"Authorization": f"Bearer {args.token}"} if args.token else {"X-OpsRAG-User": args.user}
    )
    levels = [int(c) for c in args.concurrency.split(",")]
    limits = httpx.Limits(max_connections=max(levels), max_keepalive_connections=max(levels))
    async with httpx.AsyncClient(
        base_url=args.url,
        headers=headers,
        timeout=args.timeout,
        limits=limits,
        transport=transport,
    ) as client:
        me = await client.get("/api/me")
        me.raise_for_status()
        questions = pick_questions(args.questions)
        # 1. Baseline: one at a time (also loads the models).
        baseline_records = [await ask(client, q) for q in questions]
        failed = [r for r in baseline_records if r["status"] != 200]
        if failed:
            raise SystemExit(f"baseline failed for {len(failed)} question(s): {failed[:2]}")
        baseline = {r["question"]: r["fingerprint"] for r in baseline_records}
        results: dict[str, Any] = {
            "url": args.url,
            "user": me.json().get("user_id"),
            "role": me.json().get("role"),
            "questions": questions,
            "requests_per_level": args.requests,
            "baseline": {
                "client_ms": latency_stats([r["client_ms"] for r in baseline_records[1:]]),
                "first_request_ms": baseline_records[0]["client_ms"],
                "note": "sequential; the first request may include loading the models",
            },
            "levels": [],
        }
        # 2. Levels.
        for index, concurrency in enumerate(levels):
            records, seconds = await run_level(
                client, questions, concurrency, args.requests, offset=index
            )
            level = summarize_level(concurrency, records, seconds, baseline)
            level["records"] = records
            results["levels"].append(level)
            print(
                f"concurrency {concurrency}: {level['completed']}/{level['requests']} ok, "
                f"{level['throughput_rps']} req/s, p50 {level['client_ms']['p50']} ms, "
                f"p95 {level['client_ms']['p95']} ms, mismatches {level['answer_mismatches']}",
                flush=True,
            )
        metrics = await client.get("/api/metrics")
        if metrics.status_code == 200:
            results["server_metrics"] = metrics.json()
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--url", default=os.environ.get("OPSRAG_API_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--user", default="alex.rivera", help="demo user (X-OpsRAG-User)")
    parser.add_argument("--token", default=os.environ.get("OPSRAG_API_TOKEN"))
    parser.add_argument("--concurrency", default="1,2,4,8")
    parser.add_argument("--requests", type=int, default=24, help="requests per level")
    parser.add_argument("--questions", type=int, default=12, help="distinct questions")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--run-id", default=datetime.now(UTC).strftime("load-%Y%m%d-%H%M%S"))
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "benchmarks" / "load")
    parser.add_argument("--label", default="", help="free text stored with the results")
    args = parser.parse_args(argv)
    awake = keep_awake()
    started = datetime.now(UTC)
    results = asyncio.run(main_async(args))
    results.update(
        run_id=args.run_id,
        label=args.label,
        started_at=started.isoformat(),
        finished_at=datetime.now(UTC).isoformat(),
        environment={**environment(), "kept_awake": awake},
    )
    run_dir = args.out / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (run_dir / "report.md").write_text(render_load_report(results), encoding="utf-8")
    print(f"wrote {run_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
