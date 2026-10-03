"""Markdown reports of the performance benchmark and the load test (from their JSON)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any


def _cell(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:,.2f}" if abs(value) < 100 else f"{value:,.0f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines += ["| " + " | ".join(_cell(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def _stats_row(name: str, stats: dict[str, Any]) -> list[Any]:
    return [
        name,
        stats.get("count"),
        stats.get("p50"),
        stats.get("p95"),
        stats.get("p99"),
        stats.get("max"),
    ]


STATS = ["", "n", "p50 ms", "p95 ms", "p99 ms", "max ms"]


def _power(power: dict[str, Any]) -> str:
    if power.get("ac_power") is None:
        return "not reported"
    text = "mains" if power["ac_power"] else "battery"
    if power.get("battery_percent") is not None:
        text += f" (battery {power['battery_percent']}%)"
    if power.get("battery_saver"):
        text += ", battery saver on"
    return text


def environment_section(env: dict[str, Any]) -> str:
    database = env.get("database", {})
    packages = env.get("packages", {})
    rows = [
        ["CPU", env.get("cpu")],
        ["Logical CPUs", env.get("logical_cpus")],
        ["Memory (GB)", env.get("memory_gb")],
        ["OS", env.get("os")],
        ["Python", env.get("python")],
        ["torch (threads)", f"{packages.get('torch')} ({env.get('torch_threads')} threads)"],
        ["GPU", "yes" if env.get("cuda") else "no (CPU only)"],
        ["Power", _power(env.get("power") or {})],
        ["sentence-transformers", packages.get("sentence-transformers")],
        [
            "Database",
            f"{database.get('dialect')} {database.get('server') or ''}"
            + (f", pgvector {database['pgvector']}" if database.get("pgvector") else ""),
        ],
    ]
    return table(["", ""], rows)


def render_report(results: dict[str, Any]) -> str:
    parts = [f"# Performance benchmark `{results.get('run_id')}`"]
    if results.get("quick"):
        parts.append("**Quick run: small samples, a smoke test only.**")
    if "environment" in results:
        parts += ["## Environment", environment_section(results["environment"])]
    if "configuration" in results:
        config = results["configuration"]
        parts += ["## Configuration", table(["Setting", "Value"], sorted(config.items()))]

    if "ingestion" in results:
        ing = results["ingestion"]
        full = ing.get("full_ingest_sqlite", {})
        parts += [
            "## Ingestion",
            f"{ing['sources']:,} sources -> {ing['chunks']:,} chunks; {ing['repeat']} run(s) each.",
            table(
                ["Step", "p50 ms", "max ms", "Throughput"],
                [
                    [
                        f"Read sources ({ing['database']})",
                        ing["read_ms"]["p50"],
                        ing["read_ms"]["max"],
                        "",
                    ],
                    [
                        "Parse, clean, chunk",
                        ing["chunk_ms"]["p50"],
                        ing["chunk_ms"]["max"],
                        f"{_cell(ing['chunking_chunks_per_s'])} chunks/s",
                    ],
                    [
                        "Full ingest with writes (fresh SQLite)",
                        full.get("ms", {}).get("p50"),
                        full.get("ms", {}).get("max"),
                        f"{_cell(full.get('chunks_per_s'))} chunks/s",
                    ],
                ],
            ),
        ]

    if "embedding" in results:
        emb = results["embedding"]
        slices = emb.get("slices", 1)
        parts += [
            "## Embedding",
            f"Model `{emb['model']}`; sample of {emb['by_batch_size'][0]['chunks']} chunks "
            f"(mean {_cell(emb['sample_tokens_mean'])} tokens, "
            f"{_cell(emb['sample_truncated'])} truncated); configured batch size "
            f"{emb['configured_batch_size']}; batch sizes interleaved over {slices} slice(s).",
            table(
                ["Batch size", "Seconds", "Chunks/s"],
                [
                    [
                        b["batch_size"],
                        ", ".join(_cell(x) for x in b["seconds"])
                        if isinstance(b["seconds"], list)
                        else b["seconds"],
                        b["chunks_per_s"],
                    ]
                    for b in emb["by_batch_size"]
                ],
            ),
            "Embedding one query:",
            table(STATS, [_stats_row("query", emb["query_embedding_ms"])]),
        ]

    if "retrieval" in results:
        ret = results["retrieval"]
        parts += [
            "## Retrieval (first stage)",
            f"{ret['queries']} queries, top {ret['top_k']}, filtered to {ret['filter']}.",
            table(STATS, [_stats_row(m, ret[m]) for m in ("dense", "bm25", "hybrid") if m in ret]),
        ]

    if "rerank" in results:
        rr = results["rerank"]
        rows = [
            [
                f"{v['candidates']} / {v['max_length']} / {v['batch_size']}"
                + (" (configured)" if v["configured"] else ""),
                v["rerank_ms"]["p50"],
                v["rerank_ms"]["p95"],
                v["rerank_ms"]["p99"],
                v["ndcg_10"],
                v["recall_5"],
                v["hit_5"],
                v["mrr"],
            ]
            for v in rr["variants"]
        ]
        first = rr.get("first_stage_ms", {})
        parts += [
            "## Reranking",
            f"`{rr['model']}` over the same hybrid candidates for every variant (first stage "
            f"p50 {_cell(first.get('p50'))} ms); variants interleaved per question; quality "
            f"on {rr['questions']} labelled questions (`{rr['benchmark']}`).",
            table(
                [
                    "Candidates / max tokens / batch",
                    "Rerank p50 ms",
                    "p95",
                    "p99",
                    "NDCG@10",
                    "Recall@5",
                    "Hit@5",
                    "MRR",
                ],
                rows,
            ),
        ]

    if "threads" in results:
        th = results["threads"]
        parts += [
            "## Torch threads (one request)",
            f"Reranking {th['rerank']} (candidates / max tokens / batch) for {th['questions']} "
            f"questions; thread counts rotated per question; torch's default here is "
            f"{th['default_threads']}.",
            table(STATS, [_stats_row(f"{t} threads", v) for t, v in th["by_threads"].items()]),
        ]

    if "nli" in results:
        nli = results["nli"]
        parts.append("## Claim verification (NLI calls)")
        if not nli.get("measured", True):
            parts.append(f"Not measured: {nli.get('reason')}.")
        else:
            parts += [
                f"`{nli['model']}`: the NLI calls of {nli['answers']} real answers "
                f"({nli['calls_per_answer_mean']} calls and {nli['pairs_per_answer_mean']} pairs "
                f"per answer), replayed one call per claim and as one batch, alternating.",
                table(
                    STATS,
                    [
                        _stats_row("one call per claim", nli["per_claim_ms"]),
                        _stats_row("one batched call", nli["batched_ms"]),
                    ],
                ),
                f"Total: {nli['per_claim_total_s']} s per claim vs {nli['batched_total_s']} s "
                "batched.",
            ]

    if "llm" in results:
        llm = results["llm"]
        parts.append("## LLM")
        if not llm.get("measured"):
            parts.append(f"Not measured: {llm.get('reason')}.")
        else:
            parts += [
                f"`{llm['model']}`: {llm['calls']} calls, {llm['failures']} failed; mean "
                f"{_cell(llm['input_tokens_mean'])} input / {_cell(llm['output_tokens_mean'])} "
                "output tokens.",
                table(STATS, [_stats_row("completion", llm["latency_ms"])]),
            ]

    if "compare" in results:
        cmp = results["compare"]
        parts += [
            "## Before and after (same questions, alternating)",
            f"Before: the current settings with `{cmp['before_overrides'] or 'no overrides'}`. "
            f"{cmp['questions']} questions; answers that differ: "
            f"{len(cmp['answers_different'])}"
            + (f" ({', '.join(cmp['answers_different'])})" if cmp["answers_different"] else "")
            + ".",
            table(
                [*STATS, "total s"],
                [
                    [*_stats_row(name, cmp["total_ms"][name]), cmp["total_seconds"][name]]
                    for name in ("before", "after")
                ],
            ),
        ]

    if "agent" in results:
        ag = results["agent"]
        parts += [
            "## End to end (agent)",
            f"Building the agent (loading its models): {_cell(ag['build_ms'])} ms; first "
            f"question after that: {_cell(ag['first_question_ms'])} ms (not in the "
            f"statistics). {ag['questions']} questions, {ag['tool_calls']} tool calls, "
            f"{ag['errors']} errors; retrieval `{ag['retrieval_mode']}`, LLM `{ag['llm']}`.",
            table(STATS, [_stats_row("answer (total)", ag["total_ms"])]),
            "Stages, by share of the total time:",
            table(
                ["Stage", "p50 ms", "p95 ms", "p99 ms", "Share"],
                [
                    [s, v["p50"], v["p95"], v["p99"], f"{v['share'] * 100:.1f}%"]
                    for s, v in ag["stages"].items()
                ],
            ),
        ]
    return "\n\n".join(parts) + "\n"


def render_load_report(results: dict[str, Any]) -> str:
    parts = [f"# Load test `{results.get('run_id')}`"]
    if "environment" in results:
        parts += [
            "## Environment (client and server, same machine)",
            environment_section(results["environment"]),
        ]
    parts.append(
        f"Target `{results['url']}`; {results['requests_per_level']} requests per level; "
        f"{len(results['questions'])} distinct questions, cycled; user `{results['user']}`."
    )
    rows = []
    for level in results["levels"]:
        client, server = level["client_ms"], level["server_agent_ms"]
        rows.append(
            [
                level["concurrency"],
                level["completed"],
                level["errors"],
                level["throughput_rps"],
                client["p50"],
                client["p95"],
                client["p99"],
                server["p50"],
                level["answer_mismatches"],
            ]
        )
    parts.append(
        table(
            [
                "Concurrency",
                "Completed",
                "Errors",
                "Requests/s",
                "Client p50 ms",
                "p95",
                "p99",
                "Server agent p50 ms",
                "Answers changed",
            ],
            rows,
        )
    )
    return "\n\n".join(parts) + "\n"


__all__ = ["render_load_report", "render_report", "table"]
