"""System Metrics: requests, latency percentiles per component, tool calls and errors.

From ``GET /api/metrics``: the API process's own counters and a rolling window of
latencies (since the process started). Numbers appear as the system is used.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from opsrag_ui.charts import hbars
from opsrag_ui.client import ApiError, get_client
from opsrag_ui.theme import ms

STAGES = [
    "query_understanding",
    "routing",
    "tool_selection",
    "tool_execution",
    "evidence_aggregation",
    "security_screening",
    "reranking",
    "evidence_validation",
    "evidence_package",
    "synthesis",
    "claim_verification",
    "output_validation",
    "confidence",
    "final_response",
]

st.title("System Metrics")
top = st.columns([4, 1.2, 1], vertical_alignment="bottom")
top[0].caption(
    "Measured inside the API process since it started (latencies over the last requests of "
    "a rolling window). Ask questions in Chat to generate traffic."
)
auto = top[1].toggle("Auto-refresh", value=False, help="Reload every 10 seconds")
top[2].button("Refresh", icon=":material/refresh:")


def row(name: str, stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "Component": name,
        "Count": stats.get("total", stats.get("count", 0)),
        "p50": stats.get("p50"),
        "p95": stats.get("p95"),
        "p99": stats.get("p99"),
        "Max": stats.get("max"),
    }


@st.fragment(run_every=10 if auto else None)
def dashboard() -> None:
    try:
        data: dict[str, Any] = get_client().metrics()
    except ApiError as exc:
        st.error(exc.detail, icon=":material/error:")
        return
    http, agent = data["http"], data["agent"]
    errors = data.get("errors", {})
    error_count = sum(errors.values())
    tool_calls = sum(t["calls"] for t in data.get("tools", {}).values())
    latency = agent["latency_ms"]

    retrieval, llm = data["retrieval"], data["llm"]
    counts, latencies = st.columns(4), st.columns(4)
    counts[0].metric("HTTP requests", f"{http['requests']:,}")
    counts[1].metric("Questions answered", f"{agent['requests']:,}")
    counts[2].metric("Tool calls", f"{tool_calls:,}")
    counts[3].metric("Errors", f"{error_count:,}")
    latencies[0].metric("Answer latency p50", ms(latency.get("p50")))
    latencies[1].metric("Answer latency p95", ms(latency.get("p95")))
    latencies[2].metric("Answer latency p99", ms(latency.get("p99")))
    latencies[3].metric(
        "Retrieval latency p50",
        ms(retrieval["search"].get("p50")),
        help="One retrieval tool call: first stage plus reranking",
    )

    config = data.get("configuration", {})
    st.caption(
        f"Retrieval `{config.get('retrieval_mode')}` · reranker "
        f"`{config.get('reranker_model') or 'none'}` · LLM `{config.get('llm') or 'none (extractive answers)'}`"
        f" · up {data['uptime_seconds'] / 60:.0f} min · window {data['window']} observations"
    )

    st.subheader("Latency by component")
    stages = agent.get("stages", {})
    components = pd.DataFrame(
        [
            row("HTTP request (all endpoints)", http["latency_ms"]),
            row("Answer (agent, end to end)", latency),
            row("Retrieval (tool searches)", retrieval["search"]),
            row("Retrieval: first stage", retrieval["first_stage"]),
            row("Reranker (cross-encoder)", retrieval["rerank"]),
            row("Evidence reranking (stage)", stages.get("reranking", {})),
            row("Claim verification (stage)", stages.get("claim_verification", {})),
            row("LLM completion", llm["completion"]),
            row("Agent start-up (loads models, once)", agent.get("build_ms", {})),
        ]
    )
    measured = components.dropna(subset=["p95"])
    if measured.empty:
        st.info("No requests measured yet.", icon=":material/hourglass_empty:")
    else:
        st.altair_chart(
            hbars(
                measured,
                "Component",
                "p95",
                "p95 latency (ms)",
                tooltip=["Component", "Count", "p50", "p95", "p99", "Max"],
            ),
            width="stretch",
        )
    table = components.set_index("Component")
    for column in ("p50", "p95", "p99", "Max"):  # shown as text: "—" where nothing was measured
        table[column] = [ms(v) if v == v else ms(None) for v in table[column]]
    st.dataframe(table, width="stretch")
    notes = []
    if agent.get("build_ms", {}).get("count"):
        notes.append(
            "The first question after start-up also loads the models (agent start-up): that "
            "time is part of its HTTP request, so it shows in the HTTP p99 and maximum."
        )
    if not llm["completion"]["count"]:
        notes.append(
            "LLM completion: no calls. No LLM is configured, so answers are extractive and "
            "there is no model latency to measure."
        )
    for note in notes:
        st.caption(note)

    st.subheader("Agent stages")
    stage_rows = pd.DataFrame(
        [
            {
                "Stage": s.replace("_", " "),
                "p50": stages[s]["p50"],
                "p95": stages[s]["p95"],
                "Count": stages[s]["total"],
            }
            for s in STAGES
            if s in stages and stages[s]["p50"] is not None
        ]
    )
    if stage_rows.empty:
        st.caption("No questions answered yet.")
    else:
        st.altair_chart(
            hbars(
                stage_rows,
                "Stage",
                "p50",
                "Median time per stage (ms)",
                fmt=",.1f",
                tooltip=["Stage", "p50", "p95", "Count"],
            ),
            width="stretch",
        )

    st.subheader("Tool calls")
    tools = data.get("tools", {})
    if tools:
        tool_rows = pd.DataFrame(
            [
                {
                    "Tool": name,
                    "Calls": t["calls"],
                    "Errors": t["errors"],
                    "p50": ms(t["latency_ms"]["p50"]),
                    "p95": ms(t["latency_ms"]["p95"]),
                }
                for name, t in tools.items()
            ]
        )
        left, right = st.columns(2, gap="large")
        with left:
            st.altair_chart(hbars(tool_rows, "Tool", "Calls", "Calls per tool"), width="stretch")
        with right:
            st.dataframe(tool_rows.set_index("Tool"), width="stretch")
    else:
        st.caption("No tool calls yet.")

    st.subheader("Errors and outcomes")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Errors by code**")
        if errors:
            st.dataframe(
                pd.DataFrame([{"Code": c, "Count": n} for c, n in errors.items()]),
                hide_index=True,
                width="stretch",
            )
        else:
            st.caption("No errors recorded.")
    with c2:
        st.markdown("**HTTP status codes**")
        st.dataframe(
            pd.DataFrame(
                [{"Status": s, "Requests": n} for s, n in sorted(http["by_status"].items())]
            ),
            hide_index=True,
            width="stretch",
        )
    with c3:
        st.markdown("**Answer confidence**")
        st.dataframe(
            pd.DataFrame(
                [{"Confidence": c, "Answers": n} for c, n in agent.get("by_confidence", {}).items()]
            ),
            hide_index=True,
            width="stretch",
        )


dashboard()
