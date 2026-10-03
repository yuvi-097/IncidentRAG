"""Evidence Viewer: every piece of evidence behind an answer, with its provenance."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from opsrag_ui.client import ApiError, get_client
from opsrag_ui.session import history
from opsrag_ui.theme import chip, when

CODE_KINDS = {"code", "pull_request", "logs", "sql_result"}
WIDTHS = {
    "Label": 54,
    "Source": 84,
    "Kind": 96,
    "Title": 220,
    "Relevance": 112,
    "Timestamp": 140,
    "Access level": 100,
    "Trust": 112,
    "Cited": 54,
}

st.title("Evidence Viewer")
st.caption(
    "The sources an answer was written from: what each one is, how relevant it was judged, "
    "when it dates from and which access label it carries. Content is shown as screened "
    "(instructions and credentials removed)."
)

turns = history()
if not turns:
    st.info(
        "No answers yet in this session. Ask a question here or in Chat.", icon=":material/forum:"
    )
    with st.form("ask_here", border=False):
        question = st.text_input(
            "Question",
            value="Why did payment requests start returning HTTP 500 errors after deployment v2.8.1?",
        )
        if st.form_submit_button("Ask", type="primary", icon=":material/send:"):
            try:
                with st.spinner("Investigating…"):
                    response = get_client().ask(question)
                turns.append({"question": question, "response": response})
                st.rerun()
            except ApiError as exc:
                st.error(exc.detail)
    st.stop()

labels = [f"{n}. {t['question']}" for n, t in enumerate(turns, 1)]
choice = st.selectbox("Answer", list(reversed(range(len(turns)))), format_func=lambda i: labels[i])
response: dict[str, Any] = turns[choice]["response"]
evidence: list[dict[str, Any]] = response.get("evidence", [])
if not evidence:
    st.info("This answer used no evidence.", icon=":material/info:")
    st.stop()

kinds = sorted({e["kind"] for e in evidence})
f1, f2, f3 = st.columns([3, 1.3, 2], vertical_alignment="bottom")
wanted = f1.multiselect("Kinds", kinds, default=kinds)
cited_only = f2.toggle("Cited only", value=False)
min_relevance = f3.slider("Minimum relevance", 0.0, 1.0, 0.0, 0.05)
shown = [
    e
    for e in evidence
    if e["kind"] in wanted and (e["cited"] or not cited_only) and e["relevance"] >= min_relevance
]

cited = sum(e["cited"] for e in evidence)
levels = sorted({e.get("access_level", "?") for e in evidence})
m1, m2, m3, m4 = st.columns(4)
m1.metric("Evidence items", len(evidence))
m2.metric("Cited in the answer", cited)
m3.metric("Distinct sources", len({e["source_id"] for e in evidence}))
m4.metric("Access labels", len(levels), help="Labels of the evidence: " + ", ".join(levels))
st.markdown(
    "<span class='ops-muted'>Access labels in this answer:</span> "
    + " ".join(chip(level) for level in levels),
    unsafe_allow_html=True,
)

st.dataframe(
    pd.DataFrame(
        [
            {
                "Label": e["label"],
                "Source": e["source_id"],
                "Kind": e["kind"],
                "Title": e["title"],
                "Relevance": e["relevance"],
                "Timestamp": when(e.get("timestamp")),
                "Access level": e.get("access_level", ""),
                "Trust": e["trust"].replace("_", " "),
                "Cited": "✓" if e["cited"] else "",
            }
            for e in shown
        ]
    ),
    hide_index=True,
    width="stretch",
    column_config={  # fixed widths fit a laptop screen; the title takes the rest
        **{n: st.column_config.Column(width=w) for n, w in WIDTHS.items()},
        "Relevance": st.column_config.ProgressColumn(
            "Relevance", min_value=0.0, max_value=1.0, format="%.2f", width=WIDTHS["Relevance"]
        ),
    },
)

for e in shown:
    header = f"[{e['label']}] {e['source_id']} · {e['title']}"
    with st.expander(header, expanded=len(shown) == 1):
        st.markdown(
            " ".join(
                chip(x)
                for x in (
                    e["kind"],
                    f"relevance {e['relevance']:.2f}",
                    f"access {e.get('access_level', '?')}",
                    f"trust {e['trust']}",
                    when(e.get("timestamp")),
                    "cited" if e["cited"] else "not cited",
                )
            ),
            unsafe_allow_html=True,
        )
        if e.get("location"):
            st.caption(f"Location: {e['location']}")
        content = e.get("content") or e.get("snippet", "")
        st.code(content, language="text" if e["kind"] not in CODE_KINDS else None, wrap_lines=True)
