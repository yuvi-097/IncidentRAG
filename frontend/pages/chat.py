"""Chat: ask an incident question and see the answer with everything behind it."""

from __future__ import annotations

import html
from typing import Any

import pandas as pd
import streamlit as st

from opsrag_ui.client import ApiError, get_client
from opsrag_ui.session import history
from opsrag_ui.theme import answer_html, chip, cite, confidence_badge, ms, when

EXAMPLES = [
    "What caused INC-0406?",
    "Which deployment happened immediately before INC-0421?",
    "Which commit and file caused INC-0033?",
    "How many SEV1 incidents were there in 2026?",
    "How do we handle database connection exhaustion?",
    "What is PAYMENT_SERVICE_HTTP_TIMEOUT_SECONDS set to in production?",
]
KIND_LABEL = {
    "runbook": "runbook",
    "change": "deployment",
    "code": "code change",
    "logs": "logs",
    "conflict": "conflict",
    "gather": "more evidence",
}
# Widths that fit the chat column on a laptop screen; the title takes the rest.
EVIDENCE_WIDTHS = {
    "Label": 54,
    "Cited": 50,
    "Source": 84,
    "Kind": 96,
    "Title": 270,
    "Relevance": 112,
    "Timestamp": 140,
    "Access": 96,
}


def evidence_frame(response: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Label": e["label"],
                "Cited": "✓" if e["cited"] else "",
                "Source": e["source_id"],
                "Kind": e["kind"],
                "Title": e["title"],
                "Relevance": e["relevance"],
                "Timestamp": when(e.get("timestamp")),
                "Access": e.get("access_level", ""),
            }
            for e in response["evidence"]
        ]
    )


def render_response(response: dict[str, Any]) -> None:
    latency = response.get("latency_ms", {}).get("total")
    tools = response.get("tools", [])
    plan = response.get("plan") or "routed"
    header = " ".join(
        [
            confidence_badge(response["confidence"]),
            chip(f"evidence {response.get('evidence_status') or 'none'}"),
            chip(f"{plan} plan · {response.get('query_type') or 'no route'}"),
            chip(f"{len(tools)} tool call{'s' if len(tools) != 1 else ''}"),
            chip(ms(latency)),
        ]
    )
    st.markdown(header, unsafe_allow_html=True)
    st.markdown(answer_html(response["answer"]), unsafe_allow_html=True)

    evidence, citations = response.get("evidence", []), response.get("citations", [])
    recommendations = response.get("recommendations", [])
    tabs = st.tabs(
        [
            f"Evidence ({len(evidence)})",
            f"Citations ({len(citations)})",
            f"Recommendations ({len(recommendations)})",
            f"Tools & steps ({len(tools)})",
            "Limitations & security",
        ]
    )
    with tabs[0]:
        if evidence:
            st.dataframe(
                evidence_frame(response),
                hide_index=True,
                width="stretch",
                column_config={
                    **{n: st.column_config.Column(width=w) for n, w in EVIDENCE_WIDTHS.items()},
                    "Relevance": st.column_config.ProgressColumn(
                        "Relevance",
                        min_value=0.0,
                        max_value=1.0,
                        format="%.2f",
                        width=EVIDENCE_WIDTHS["Relevance"],
                    ),
                },
            )
            st.caption("Full content, trust level and provenance of every item: Evidence Viewer.")
        else:
            st.caption("No evidence was used.")
    with tabs[1]:
        if not citations:
            st.caption("The answer cites nothing.")
        for c in citations:
            where = " · ".join(
                html.escape(x)
                for x in (c.get("section"), c.get("file_path"), when(c.get("timestamp")))
                if x
            )
            st.markdown(  # titles and paths come from evidence: escaped before use
                f"<b>[{html.escape(c['label'])}]</b> <code>{html.escape(c['source_id'])}</code> "
                f"{html.escape(c['title'])}<br><span class='ops-muted'>{c['source_type']} · "
                f"relevance {c['relevance']:.2f} · trust {c.get('trust') or '—'}"
                f"{' · ' + where if where else ''}</span>",
                unsafe_allow_html=True,
            )
    with tabs[2]:
        if not recommendations:
            st.caption(
                "No recommended steps for this answer: steps are only drawn from cited "
                "runbooks, deployments, code changes, logs and conflicts."
            )
        else:
            items = "".join(  # text from the evidence: escaped before use
                f"<li>{chip(KIND_LABEL.get(r['kind'], r['kind']))}{html.escape(r['text'])} "
                + "".join(cite(label) for label in r.get("sources", []))
                + "</li>"
                for r in recommendations
            )
            st.markdown(f"<ol class='ops-recs'>{items}</ol>", unsafe_allow_html=True)
            st.caption(
                "Suggestions drawn from the cited evidence; the agent is read-only and "
                "changes nothing itself."
            )
    with tabs[3]:
        if tools:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Tool": t["tool"],
                            "Purpose": t["purpose"],
                            "Status": t["status"],
                            "Results": t["results"],
                            "Duration": ms(t["duration_ms"]),
                        }
                        for t in tools
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
        with st.expander("What each stage did (summaries, not reasoning)"):
            for line in response.get("reasoning_summary", []):
                stage, _, summary = line.partition(": ")
                st.markdown(f"- **{stage}**: {summary}")
    with tabs[4]:
        for note in response.get("limitations", []) or ["No limitations were reported."]:
            st.markdown(f"- {note}")
        security = response.get("security", {})
        cols = st.columns(5)
        cols[0].metric("Question refused", "yes" if security.get("blocked") else "no")
        cols[1].metric("Sources quarantined", len(security.get("quarantined", [])))
        cols[2].metric("Secrets redacted", security.get("secrets_redacted", 0))
        cols[3].metric("Hidden references", security.get("references_redacted", 0))
        cols[4].metric("Answer lines removed", len(security.get("output_removed", [])))
        reasons = response.get("confidence_reasons", [])
        if reasons:
            st.caption("Confidence: " + "; ".join(reasons))


def ask(question: str) -> None:
    client = get_client()
    try:
        with st.spinner("Investigating: routing, calling tools, verifying claims…"):
            response = client.ask(question)
    except ApiError as exc:
        st.error(f"The API refused the question: {exc.detail}", icon=":material/error:")
        return
    me = st.session_state.get("_memo", {}).get("me")
    who = me[2] if me else {}
    history().append(
        {
            "question": question,
            "response": response,
            "user": who.get("user_id"),
            "role": who.get("role"),
        }
    )


st.title("Ask about an incident")
st.caption(
    "Answers come only from evidence your role may read. Every sentence cites its source; "
    "confidence is computed from the evidence, not claimed by a model."
)

pending = st.session_state.pop("pending_question", None)
typed = st.chat_input("Ask about incidents, deployments, code, logs or runbooks…")
if not history() and not pending and not typed:
    st.markdown("**Try one of these**")
    choice = st.pills("Examples", EXAMPLES, label_visibility="collapsed", key="example")
    if choice:
        pending = choice
question = typed or pending
if question:
    ask(question)

for turn in history():
    with st.chat_message("user", avatar=":material/person:"):
        who = f" · {turn['user']} ({turn['role']})" if turn.get("user") else ""
        st.markdown(turn["question"])
        st.caption(f"asked{who}")
    with st.chat_message("assistant", avatar=":material/emergency_home:"):
        render_response(turn["response"])

if history() and st.button("Clear conversation", icon=":material/delete:", type="tertiary"):
    history().clear()
    st.session_state.pop("example", None)
    st.rerun()
