"""Evaluation dashboard: the measured results of the latest Phase 10 run.

Every number comes from ``GET /api/evaluation`` (the run's summary and per-question
records); nothing is typed into this page.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from opsrag_ui.charts import hbars, heatmap
from opsrag_ui.client import ApiError, get_client
from opsrag_ui.session import memo
from opsrag_ui.theme import pct, when

SHORT = {
    "A": "Dense",
    "B": "BM25",
    "C": "Hybrid",
    "D": "Hybrid+rerank",
    "E": "+verification",
    "F": "Full agent",
}
RETRIEVAL = {
    "hit_5": "Hit@5",
    "recall_1": "Recall@1",
    "recall_5": "Recall@5",
    "recall_10": "Recall@10",
    "precision_5": "Precision@5",
    "mrr": "MRR",
    "ndcg_10": "NDCG@10",
}
ANSWERS = {
    "correctness": "Answer correctness",
    "faithfulness": "Faithfulness",
    "citation_correctness": "Citation accuracy",
    "hallucination_rate": "Hallucination rate",
}
ORIGINS = {
    "retrieval_benchmark": "Natural-language questions",
    "generated": "Identifier-heavy questions",
}

st.title("Evaluation")
try:
    data: dict[str, Any] = memo("evaluation", get_client().evaluation, ttl=600)
except ApiError as exc:
    st.error(exc.detail, icon=":material/error:")
    st.stop()

run, summary = data["run"], data["summary"]
methods = summary["methods"]
order = [SHORT.get(m, m) for m in methods]
st.caption(
    f"Measured on {run['questions']} questions (run `{run['id']}`, finished "
    f"{when(run['finished_at'])}). LLM-as-judge: {run['judge']}."
)

full, baseline = methods.get("F", {}), methods.get("B", {})
k = st.columns(5)
k[0].metric(
    "Answer correctness (full agent)",
    pct(full.get("correctness")),
    f"{(full['correctness'] - baseline['correctness']) * 100:+.1f} pts vs BM25"
    if full and baseline
    else None,
)
k[1].metric("Hit@5 (full agent)", f"{full.get('hit_5', 0):.3f}")
k[2].metric("Faithfulness", pct(full.get("faithfulness")))
k[3].metric("Citation accuracy", pct(full.get("citation_correctness")))
k[4].metric("Hallucination rate", pct(full.get("hallucination_rate")))


def frame(keys: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"System": SHORT.get(m, m), **{label: s.get(key) for key, label in keys.items()}}
            for m, s in methods.items()
        ]
    )


st.subheader("Retrieval approaches")
st.caption(
    f"Over the {next(iter(methods.values()))['retrieval_questions']} questions that have gold "
    "sources. Recall@k is capped at the number of relevant records, so Hit@5 (any relevant "
    "record in the top 5) is the steadier comparison."
)
metric = (
    st.segmented_control(
        "Metric", list(RETRIEVAL), format_func=RETRIEVAL.get, default="hit_5", key="ret_metric"
    )
    or "hit_5"
)
# Small multiples on one scale: every question, then each kind of question.
by_origin = data.get("retrieval_by_origin", {})
panels = [(f"All questions ({next(iter(methods.values()))['retrieval_questions']})", methods)]
for origin, title in ORIGINS.items():
    rows = by_origin.get(origin, {})
    if rows:
        panels.append((f"{title} ({next(iter(rows.values()))['questions']})", rows))
for column, (title, rows) in zip(st.columns(len(panels), gap="medium"), panels, strict=True):
    panel_frame = pd.DataFrame(
        [{"System": SHORT.get(m, m), "value": r.get(metric)} for m, r in rows.items()]
    )
    with column:
        st.altair_chart(
            hbars(panel_frame, "System", "value", title, fmt=".2f", sort=order, domain=(0, 1)),
            width="stretch",
        )
st.caption(
    f"{RETRIEVAL[metric]} per system. Natural-language questions come from the retrieval "
    "benchmark used during development; identifier-heavy ones (ids, setting and class names) "
    "were generated for Phase 10."
)
st.dataframe(frame(RETRIEVAL).set_index("System").style.format("{:.3f}"), width="stretch")

st.subheader("Answer quality")
grid = [c for _ in range(2) for c in st.columns(2, gap="large")]
for col, (key, label) in zip(grid, ANSWERS.items(), strict=True):
    with col:
        st.altair_chart(
            hbars(
                frame({key: "value"}),
                "System",
                "value",
                label,
                fmt=".0%",
                sort=order,
                domain=(0, 1),
            ),
            width="stretch",
        )
st.caption(
    "Deterministic metrics. Faithfulness: claims whose ids, numbers and words occur in the "
    "context. Citation accuracy: cited claims their citations ground. Hallucination: answers "
    "with a record id, number or name found in neither context nor question."
)

st.subheader("Correctness by question category")
categories = sorted({c for m in summary["by_category"].values() for c in m})
cells = pd.DataFrame(
    [
        {
            "Category": c.replace("_", " "),
            "System": SHORT.get(m, m),
            "value": summary["by_category"][m].get(c, {}).get("correctness"),
        }
        for m in methods
        for c in categories
    ]
)
st.altair_chart(heatmap(cells, "Category", "System", "value", column_sort=order), width="stretch")

st.subheader("Why answers failed")
errors = summary.get("errors", {})
classes = list(next(iter(errors.values()))["primary"]) if errors else []
failures = pd.DataFrame(
    [
        {
            "Class": c.replace("_", " "),
            "System": SHORT.get(m, m),
            "value": float(errors[m]["primary"][c]),
        }
        for m in errors
        for c in classes
    ]
)
if not failures.empty:
    st.altair_chart(
        heatmap(
            failures,
            "Class",
            "System",
            "value",
            percent=False,
            row_sort=[c.replace("_", " ") for c in classes],
            column_sort=order,
        ),
        width="stretch",
    )
    st.caption(
        "Primary error class per failed question (count). Rules: permission, hallucination, "
        "routing, retrieval, reranking, temporal, reasoning, citation, in that order."
    )

with st.expander("Read these numbers with care", icon=":material/info:", expanded=True):
    st.markdown(
        f"""
- **No LLM is configured**, so answers are extractive. Hallucination rate 0% and faithfulness
  near 100% hold *by construction*; they do not describe an LLM writer. The LLM-as-judge
  column was **{run["judge"]}**, and no judge number is estimated.
- **Citation shortfalls are notes, not wrong citations:** answers of the verifying systems carry
  "Sources disagree ... [E#] state(s) otherwise" notes, which cite the disagreeing source on
  purpose.
- **The questions and the system have the same author**, and categories hold 15-25 questions:
  one question moves a category by 4-7 points.
"""
    )

with st.expander("How these numbers were produced", icon=":material/science:"):
    models = run.get("models", {})
    st.markdown(
        f"- Run `{run['id']}`: {when(run['started_at'])} to {when(run['finished_at'])}"
        f"{' (resumed after an interruption)' if run.get('resumed') else ''}\n"
        f"- Evaluation set sha256 `{run['eval_set_sha256'][:16]}…`; dataset generator "
        f"{run['dataset']['generator_version']}, seed {run['dataset']['seed']}\n"
        f"- Code sha256 `{run['code_sha256'][:16]}…`\n"
        + "\n".join(
            f"- {role.replace('_', ' ')}: `{m['model']}` (revision {(m.get('revision') or 'n/a')[:10]})"
            for role, m in models.items()
            if m.get("model")
        )
    )
    st.caption("Reproduce with `python scripts/evaluate.py` (see the README).")
