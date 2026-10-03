"""Performance tooling: statistics, environment capture, reports and the load test."""

from __future__ import annotations

import argparse
import asyncio
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import create_engine

from app.agents.graph import Agent
from app.api.dependencies import get_engine
from app.api.routes.agent import get_agent
from app.evaluation.performance import environment, latency_stats, rate
from app.evaluation.performance_report import render_load_report, render_report
from app.llm.base import ChatMessage, LLMResponse
from scripts import benchmark_performance, load_test
from tests.agents.conftest import agent  # noqa: F401  (shared fixture)
from tests.tools.conftest import ToolEnv


def test_latency_stats_are_nearest_rank_and_carry_their_count() -> None:
    values = [float(v) for v in range(1, 101)]  # 1..100 ms
    stats = latency_stats(values)
    assert (stats["count"], stats["p50"], stats["p95"], stats["p99"], stats["max"]) == (
        100,
        50.0,
        95.0,
        99.0,
        100.0,
    )
    assert latency_stats([7.0])["p99"] == 7.0  # one sample: every percentile is that sample
    assert latency_stats([])["p50"] is None
    assert rate(10, 2.0) == 5.0 and rate(10, 0) is None


def test_environment_describes_machine_software_and_database() -> None:
    engine = create_engine("sqlite://")
    info = environment(engine)
    assert info["logical_cpus"] and info["python"] and info["cpu"]
    assert "torch" in info["packages"] and info["database"]["dialect"] == "sqlite"
    assert info["database"]["server"]
    assert isinstance(info["power"], dict)  # mains or battery, where the OS reports it


def test_settings_overrides_for_the_before_and_after_comparison() -> None:
    settings = benchmark_performance.load_settings(env_file=None)
    before = benchmark_performance.with_overrides(
        settings, "reranker.batch_size=16, retrieval.rerank_candidates=20"
    )
    assert (before.reranker.batch_size, before.retrieval.rerank_candidates) == (16, 20)
    assert settings.reranker.batch_size == 4  # the original is untouched
    assert benchmark_performance.with_overrides(settings, "") is settings


def stats(p50: float) -> dict[str, Any]:
    return {"count": 3, "p50": p50, "p95": p50 * 2, "p99": p50 * 3, "max": p50 * 3}


def test_the_benchmark_report_shows_every_section() -> None:
    results = {
        "run_id": "r1",
        "environment": environment(),
        "configuration": {"retrieval_mode": "hybrid_rerank"},
        "retrieval": {
            "queries": 3,
            "top_k": 30,
            "filter": "admin labels",
            "dense": stats(20),
            "bm25": stats(5),
            "hybrid": stats(30),
        },
        "rerank": {
            "model": "m",
            "questions": 3,
            "benchmark": "b.jsonl",
            "variants": [
                {
                    "candidates": 30,
                    "max_length": 512,
                    "batch_size": 4,
                    "configured": True,
                    "rerank_ms": stats(4000),
                    "retrieval_ms": stats(4100),
                    "ndcg_10": 0.7,
                    "recall_5": 0.6,
                    "hit_5": 0.8,
                    "mrr": 0.65,
                }
            ],
        },
        "llm": {"measured": False, "reason": "LLM_PROVIDER=none: no LLM is configured"},
        "agent": {
            "build_ms": 30000.0,
            "first_question_ms": 9000.0,
            "questions": 3,
            "tool_calls": 4,
            "errors": 0,
            "retrieval_mode": "hybrid_rerank",
            "llm": "none",
            "total_ms": stats(1500),
            "stages": {"claim_verification": {**stats(900), "share": 0.6}},
        },
    }
    results["threads"] = {
        "default_threads": 10,
        "questions": 3,
        "rerank": "30 / 512 / 4",
        "by_threads": {"4": stats(1800), "10": stats(1500)},
    }
    results["nli"] = {
        "model": "nli",
        "answers": 3,
        "calls_per_answer_mean": 3.4,
        "pairs_per_answer_mean": 10.6,
        "per_claim_ms": stats(400),
        "batched_ms": stats(250),
        "per_claim_total_s": 18.8,
        "batched_total_s": 16.5,
    }
    results["compare"] = {
        "before_overrides": "reranker.batch_size=16",
        "questions": 3,
        "answers_different": [],
        "total_ms": {"before": stats(700), "after": stats(650)},
        "total_seconds": {"before": 2.1, "after": 1.9},
    }
    report = render_report(results)
    for heading in (
        "## Environment",
        "## Retrieval",
        "## Reranking",
        "## Torch threads",
        "## Claim verification",
        "## LLM",
        "## Before and after",
        "## End to end",
    ):
        assert heading in report
    assert "answers that differ: 0" in report and "| Power |" in report
    assert "30 / 512 / 4 (configured)" in report and "60.0%" in report
    assert "Not measured: LLM_PROVIDER=none" in report


def test_the_llm_section_says_why_it_was_not_measured(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = benchmark_performance.load_settings(env_file=None)
    monkeypatch.setattr(benchmark_performance, "build_llm", lambda _settings: None)
    assert benchmark_performance.section_llm(settings, ["q"]) == {
        "measured": False,
        "reason": "LLM_PROVIDER=none: no LLM is configured",
    }

    class Fake:
        name = "fake:model"

        def complete(self, messages: list[ChatMessage]) -> LLMResponse:
            return LLMResponse(text="ok", model="m", input_tokens=40, output_tokens=5)

    monkeypatch.setattr(benchmark_performance, "build_llm", lambda _settings: Fake())
    measured = benchmark_performance.section_llm(settings, ["a", "b"])
    assert measured["measured"] and measured["latency_ms"]["count"] == 2
    assert (measured["input_tokens_mean"], measured["output_tokens_mean"]) == (40.0, 5.0)


def test_a_changed_answer_under_load_is_reported() -> None:
    body = {"answer": "A [E1].", "citations": [{"label": "E1"}], "confidence": "HIGH"}
    same = load_test.fingerprint(dict(body))
    assert load_test.fingerprint(dict(body)) == same
    assert load_test.fingerprint({**body, "answer": "B [E1]."}) != same
    records = [
        {
            "question": "q1",
            "status": 200,
            "client_ms": 100.0,
            "server_agent_ms": 90.0,
            "fingerprint": same,
        },
        {
            "question": "q1",
            "status": 200,
            "client_ms": 300.0,
            "server_agent_ms": 95.0,
            "fingerprint": "x",
        },
        {"question": "q2", "status": 0, "client_ms": 50.0, "error": "ReadTimeout"},
    ]
    level = load_test.summarize_level(2, records, 2.0, {"q1": same})
    assert (level["completed"], level["errors"], level["answer_mismatches"]) == (2, 1, 1)
    assert level["throughput_rps"] == 1.0 and level["statuses"] == {"200": 2, "0": 1}
    assert level["mismatched_questions"] == ["q1"]


def test_the_load_test_runs_against_the_api(
    app: FastAPI,
    agent: Agent,  # noqa: F811
    tool_env: ToolEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app.dependency_overrides[get_engine] = lambda: tool_env.engine
    app.dependency_overrides[get_agent] = lambda: agent
    monkeypatch.setattr(
        load_test,
        "pick_questions",
        lambda count: [
            "What caused INC-0406?",
            "How do we handle database connection exhaustion?",
        ][:count],
    )
    args = argparse.Namespace(
        url="http://testserver",
        token=None,
        user="unused",
        concurrency="1,2",
        requests=4,
        questions=2,
        timeout=60.0,
    )
    from app.security.auth import issue_token

    admin = next(u.id for u in tool_env.dataset.users if u.role_id == "admin" and u.is_active)
    args.token = issue_token(tool_env.engine, admin, "load test")
    results = asyncio.run(load_test.main_async(args, httpx.ASGITransport(app=app)))
    assert [level["concurrency"] for level in results["levels"]] == [1, 2]
    for level in results["levels"]:
        assert level["completed"] == 4 and level["errors"] == 0
        assert level["answer_mismatches"] == 0
        assert level["server_agent_ms"]["count"] == 4
    assert results["role"] == "admin"
    assert "| 2 | 4 | 0 |" in render_load_report({**results, "run_id": "t"})
