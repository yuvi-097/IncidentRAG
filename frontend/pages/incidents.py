"""Incident Explorer: search and filter incidents, open one, follow it to its change."""

from __future__ import annotations

import html
from datetime import date
from typing import Any

import pandas as pd
import streamlit as st

from opsrag_ui.client import ApiError, get_client
from opsrag_ui.session import memo
from opsrag_ui.theme import chip, severity_badge, when

SEVERITIES = ["SEV1", "SEV2", "SEV3", "SEV4"]
DATASET_START, DATASET_END = date(2025, 9, 1), date(2026, 8, 31)
MAX_ROWS = 100  # the API caps it at the search tools' limit and says which one it applied

st.title("Incident Explorer")
st.caption("Search incident records by text, or filter by service, severity and date.")
client = get_client()

try:
    services = memo("services", client.services, ttl=600)
except ApiError as exc:
    services = []
    st.warning(f"Service list unavailable: {exc.detail}")

with st.form("filters", border=False):
    c1, c2, c3, c4, c5 = st.columns([2.8, 2.1, 1.5, 2.2, 1.3], vertical_alignment="bottom")
    text = c1.text_input("Search", placeholder="e.g. connection pool timeouts after deploy")
    chosen_services = c2.multiselect(
        "Services", [s["id"] for s in services], placeholder="All services"
    )
    chosen_severities = c3.multiselect("Severity", SEVERITIES, placeholder="All")
    dates = c4.date_input(
        "Started between",
        value=(DATASET_START, DATASET_END),
        min_value=date(2025, 1, 1),
        max_value=date(2027, 12, 31),
    )
    c5.form_submit_button("Search", icon=":material/search:", type="primary", width="stretch")

since, until = dates if isinstance(dates, tuple) and len(dates) == 2 else (None, None)
try:
    result = client.incidents(
        q=text or None,
        service=chosen_services or None,
        severity=chosen_severities or None,
        since=since.isoformat() if since else None,
        until=until.isoformat() if until else None,
        limit=MAX_ROWS,
    )
except ApiError as exc:
    st.error(exc.detail, icon=":material/error:")
    st.stop()

incidents: list[dict[str, Any]] = result["incidents"]
order = "ranked by relevance" if text else "newest first"
capped = result["count"] >= result["limit"]
st.caption(
    f"{result['count']} incident{'s' if result['count'] != 1 else ''} ({order}). "
    + (
        f"A search returns at most {result['limit']}; narrow the filters to see others. "
        if capped
        else ""
    )
    + "Only incidents your role may read are listed."
)
if not incidents:
    st.info("No incident matches these filters.", icon=":material/search_off:")
    st.stop()

table = pd.DataFrame(
    [
        {
            "Incident": i["id"],
            "Severity": i["severity"],
            "Service": i["service_id"],
            "Title": i["title"],
            "Started": when(i["started_at"]),
            "Resolved in": f"{i['resolution_time_minutes']} min",
            "Access": i["access_level"],
        }
        for i in incidents
    ]
)
# Widths fit the content column of a laptop screen; the title takes what is left (and its
# full text, like the category, is in the detail view below).
widths = {
    "Incident": 88,
    "Severity": 66,
    "Service": 160,
    "Title": 290,
    "Started": 140,
    "Resolved in": 92,
    "Access": 96,
}
selection = st.dataframe(
    table,
    hide_index=True,
    width="stretch",
    on_select="rerun",
    selection_mode="single-row",
    key="incident_table",
    column_config={name: st.column_config.Column(width=w) for name, w in widths.items()},
)
rows = selection.selection.rows if selection else []
if not rows:
    st.caption("Select a row to open the incident.")
    st.stop()

incident = incidents[rows[0]]
st.divider()
title = html.escape(incident["title"])
st.markdown(f"### {html.escape(incident['id'])} · {title}", unsafe_allow_html=True)
st.markdown(
    " ".join(
        [
            severity_badge(incident["severity"]),
            chip(incident["service_id"]),
            chip(incident["category"].replace("_", " ")),
            chip(f"status {incident['status']}"),
            chip(f"access {incident['access_level']}"),
        ]
    ),
    unsafe_allow_html=True,
)

facts, story = st.columns([1, 2], gap="large")
with facts:
    linked = [
        ("Started", when(incident["started_at"])),
        ("Resolved", when(incident["resolved_at"])),
        ("Resolution time", f"{incident['resolution_time_minutes']} min"),
        ("Affected version", incident["affected_version"]),
        ("Live deployment", incident["deployment_id"]),
        ("Root-cause deployment", incident.get("root_cause_deployment_id")),
        ("Root-cause PR", incident.get("root_cause_pr_id")),
        ("Fix deployment", incident.get("remediation_deployment_id")),
        ("Upstream incident", incident.get("parent_incident_id")),
        ("Runbook", incident.get("runbook_id")),
        ("Postmortem", incident.get("postmortem_id")),
        ("Alert", incident.get("alert_name")),
    ]
    st.dataframe(
        pd.DataFrame([{"Field": k, "Value": v or "—"} for k, v in linked]),
        hide_index=True,
        width="stretch",
        height=35 * (len(linked) + 1) + 3,  # every row, no inner scrolling
    )
with story:
    for heading, key in (
        ("Symptoms", "symptoms"),
        ("Root cause", "root_cause"),
        ("Resolution", "resolution"),
    ):
        st.markdown(f"**{heading}**")
        # The records are written in Markdown (`code` spans); Streamlit escapes any HTML.
        st.markdown(incident[key])

b1, b2, _ = st.columns([1.3, 1.6, 4])
trace_clicked = b1.button("Trace the change", icon=":material/account_tree:")
if b2.button("Ask about this incident", icon=":material/forum:"):
    st.session_state.pending_question = f"What caused {incident['id']} and how was it resolved?"
    st.switch_page("pages/chat.py")

if trace_clicked:
    try:
        trace = client.trace(incident["id"])
    except ApiError as exc:
        st.error(exc.detail)
        st.stop()
    st.markdown("#### From the incident to the change")
    if trace.get("note"):
        st.info(trace["note"][:1].upper() + trace["note"][1:] + ".", icon=":material/info:")
    if trace.get("parent"):
        p = trace["parent"]
        st.markdown(
            f"<div class='ops-flow-step'><b>Upstream incident</b> {html.escape(p['id'])}: "
            f"{html.escape(p['title'])} ({html.escape(p['service_id'])}, started {when(p['started_at'])})</div>",
            unsafe_allow_html=True,
        )
    d = trace.get("deployment")
    if d:
        st.markdown(
            f"<div class='ops-flow-step'><b>Root-cause deployment</b> {html.escape(d['id'])}: "
            f"{html.escape(d['service_id'])} {html.escape(d['version'])} ({html.escape(d['status'])}), "
            f"deployed {when(d['deployed_at'])}, commit <code>{html.escape(d['commit_sha'][:7])}</code></div>",
            unsafe_allow_html=True,
        )
    for change in trace.get("changes", []):
        st.markdown(
            f"<div class='ops-flow-step'><b>Pull request</b> {html.escape(change['pull_request_id'])}: "
            f"{html.escape(change['title'])} (by {html.escape(change['author'])}, merge commit "
            f"<code>{html.escape((change.get('merge_commit_sha') or '?')[:7])}</code>)</div>",
            unsafe_allow_html=True,
        )
        for f in change["files"]:
            st.markdown(
                f"<div class='ops-flow-step'><b>File</b> <code>{html.escape(f['path'])}</code> "
                f"({html.escape(f['change_type'])}, +{f['additions']} -{f['deletions']})</div>",
                unsafe_allow_html=True,
            )
            if f["changed_lines"]:
                st.code("\n".join(f["changed_lines"]), language="diff")
    if trace.get("remediation"):
        r = trace["remediation"]
        st.markdown(
            f"<div class='ops-flow-step'><b>Fix</b> {html.escape(r['id'])}: {html.escape(r['service_id'])} "
            f"{html.escape(r['version'])}, deployed {when(r['deployed_at'])}, commit "
            f"<code>{html.escape(r['commit_sha'][:7])}</code></div>",
            unsafe_allow_html=True,
        )
    for hop in trace.get("withheld", []):
        st.warning(f"Withheld (your role may not read it): {hop}", icon=":material/lock:")
