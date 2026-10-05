"""Every page renders without errors and shows what the specification asks for."""

from __future__ import annotations

from typing import Any

import pytest
from streamlit.testing.v1 import AppTest

from tests.frontend.conftest import FRONTEND, FakeClient

TIMEOUT = 60


def page(name: str, client: FakeClient, **state: Any) -> AppTest:
    at = AppTest.from_file(str(FRONTEND / "pages" / f"{name}.py"), default_timeout=TIMEOUT)
    at.session_state["client"] = client
    for key, value in state.items():
        at.session_state[key] = value
    return at.run()


def texts(at: AppTest) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption]
    return "\n".join(str(p) for p in parts)


def test_chat_shows_answer_confidence_evidence_citations_and_tools(fake: FakeClient) -> None:
    at = page("chat", fake)
    assert not at.exception
    assert at.title[0].value == "Ask about an incident"
    at.chat_input[0].set_value("Which commit and file caused INC-0033?").run()
    assert not at.exception
    assert ("ask", {"question": "Which commit and file caused INC-0033?"}) in fake.calls
    body = texts(at)
    answer = fake.captured["ask"]
    assert "confidence" in body.lower() and "ops-cite" in body  # badge and citation labels
    assert "DEP-0043" in body
    tab_labels = [t.label for t in at.tabs]
    assert any(label.startswith("Evidence (") for label in tab_labels)
    assert any(label.startswith("Recommendations (") for label in tab_labels)
    assert any(label.startswith("Tools & steps (") for label in tab_labels)
    evidence_table = at.dataframe[0].value
    assert list(evidence_table["Source"]) == [e["source_id"] for e in answer["evidence"]]
    assert {"Relevance", "Timestamp", "Access"} <= set(evidence_table.columns)
    assert len(at.session_state["history"]) == 1


def test_chat_escapes_html_from_evidence(fake: FakeClient) -> None:
    hostile = dict(fake.captured["ask"])
    hostile["answer"] = 'Look <img src=x onerror="alert(1)"> here [E1].'
    fake.captured = {**fake.captured, "ask": hostile}
    at = page("chat", fake)
    at.chat_input[0].set_value("anything").run()
    rendered = texts(at)
    assert "<img" not in rendered and "&lt;img" in rendered


def test_incident_explorer_filters_and_lists(fake: FakeClient) -> None:
    at = page("incidents", fake)
    assert not at.exception
    assert at.title[0].value == "Incident Explorer"
    _, params = next(c for c in fake.calls if c[0] == "incidents")
    assert params["limit"] == 100 and params["since"] == "2025-09-01"  # the API caps it
    labels = {w.label for w in at.multiselect} | {w.label for w in at.text_input}
    assert {"Search", "Services", "Severity"} <= labels
    assert at.date_input[0].label == "Started between"
    table = at.dataframe[0].value
    assert len(table) == fake.captured["incidents"]["count"]
    assert {"Incident", "Severity", "Service", "Started"} <= set(table.columns)


def test_evidence_viewer_shows_provenance(fake: FakeClient) -> None:
    turn = {"question": "Which commit and file caused INC-0033?", "response": fake.captured["ask"]}
    at = page("evidence", fake, history=[turn])
    assert not at.exception
    table = at.dataframe[0].value
    assert {"Source", "Relevance", "Timestamp", "Access level", "Trust"} <= set(table.columns)
    assert len(at.expander) == len(fake.captured["ask"]["evidence"])
    assert any(code.value for code in at.code)  # the full content of each item


def test_evidence_viewer_without_answers_offers_to_ask(fake: FakeClient) -> None:
    at = page("evidence", fake)
    assert not at.exception and at.info  # no history yet


def test_evaluation_dashboard_shows_measured_numbers(fake: FakeClient) -> None:
    if fake.captured["evaluation"] is None:
        pytest.skip("no evaluation run in data/evaluation/results/phase10")
    at = page("evaluation", fake)
    assert not at.exception
    summary = fake.captured["evaluation"]["summary"]["methods"]
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Answer correctness (full agent)"] == f"{summary['F']['correctness'] * 100:.1f}%"
    retrieval = at.dataframe[0].value
    assert {"Hit@5", "Recall@1", "Recall@5", "Recall@10", "MRR", "NDCG@10"} <= set(
        retrieval.columns
    )
    body = texts(at)  # the caveats are shown next to the numbers
    assert "No LLM is configured" in body and "Citation shortfalls are notes" in body


def test_system_metrics_show_latency_percentiles_tools_and_errors(fake: FakeClient) -> None:
    at = page("metrics", fake)
    assert not at.exception
    metrics = {m.label for m in at.metric}
    assert {
        "HTTP requests",
        "Questions answered",
        "Tool calls",
        "Errors",
        "Answer latency p50",
        "Answer latency p95",
        "Answer latency p99",
        "Retrieval latency p50",
    } <= metrics
    components = at.dataframe[0].value
    assert {"p50", "p95", "p99"} <= set(components.columns)
    assert "Reranker (cross-encoder)" in list(components.index)
    assert "LLM completion" in list(components.index)
    assert any("No LLM is configured" in c.value for c in at.caption)


def test_the_signed_in_user_changes_only_on_connect(fake: FakeClient) -> None:
    at = AppTest.from_file(str(FRONTEND / "app.py"), default_timeout=TIMEOUT)
    at.session_state["client"] = fake
    at.run()
    assert not at.exception
    assert at.session_state["connection"]["demo_user"] == "alex.rivera"
    # A widget whose value changes (or is reset by Streamlit) does not switch the user...
    at.selectbox[0].set_value("arjun.mehta").run()
    assert at.session_state["connection"]["demo_user"] == "alex.rivera"
    # ...only Connect does.
    at.button[0].click().run()
    assert at.session_state["connection"]["demo_user"] == "arjun.mehta"


def test_the_client_uses_the_stored_connection() -> None:
    def script() -> None:
        import streamlit as st

        from opsrag_ui.client import get_client

        st.session_state["connection"] = {
            "api_url": "",
            "demo_user": "sarah.miller",
            "api_token": "",
        }
        client = get_client()
        st.session_state["built"] = (client.base_url, client.user, client.token)

    at = AppTest.from_function(script).run()
    base_url, user, token = at.session_state["built"]
    assert base_url.startswith("http") and user == "sarah.miller"  # empty URL: the default
    assert token is None or token  # OPSRAG_API_TOKEN, if set, stays server-side


def test_incident_explorer_opens_an_incident_and_traces_its_change(fake: FakeClient) -> None:
    """The detail view and the trace flow (a row selected, then "Trace the change")."""
    at = AppTest.from_file(str(FRONTEND / "pages" / "incidents.py"), default_timeout=TIMEOUT)
    at.session_state["client"] = fake
    first = fake.captured["incidents"]["incidents"][0]
    at.session_state["incident_table"] = {"selection": {"rows": [0], "columns": [], "cells": []}}
    at.run()
    assert not at.exception
    assert any(first["id"] in m.value for m in at.markdown)  # the detail heading
    facts = at.dataframe[1].value
    assert {"Field", "Value"} <= set(facts.columns) and len(facts) == 12
    trace = next(b for b in at.button if b.label == "Trace the change")
    trace.click().run()
    assert not at.exception
    assert ("trace", {"incident_id": first["id"]}) in fake.calls
    flow = " ".join(m.value for m in at.markdown)
    assert "From the incident to the change" in flow


def test_the_public_demo_offers_only_the_demo_users(
    fake: FakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Shared through a tunnel, the sidebar lets visitors pick a role and nothing else."""
    monkeypatch.setenv("OPSRAG_UI_PUBLIC_DEMO", "true")
    at = AppTest.from_file(str(FRONTEND / "app.py"), default_timeout=TIMEOUT)
    at.session_state["client"] = fake
    at.run()
    assert not at.exception
    assert not [w.label for w in at.text_input if w.label in ("API URL", "API token")]
    picker = next(s for s in at.selectbox if s.label == "Sign in as")
    picker.set_value("arjun.mehta").run()
    assert at.session_state["connection"]["demo_user"] == "alex.rivera"  # not yet
    next(b for b in at.button if b.label == "Switch role").click().run()
    assert at.session_state["connection"]["demo_user"] == "arjun.mehta"


def test_the_public_demo_client_ignores_a_changed_url_and_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even if the session held another address or a token, a public demo would send
    neither: the server must not call out on a visitor's behalf."""
    monkeypatch.setenv("OPSRAG_UI_PUBLIC_DEMO", "true")
    monkeypatch.setenv("OPSRAG_API_TOKEN", "server-side-value")

    def script() -> None:
        import streamlit as st

        from opsrag_ui.client import DEFAULT_URL, get_client

        st.session_state["connection"] = {
            "api_url": "http://attacker.example:9000",
            "demo_user": "noor.hassan",
            "api_token": "pasted-by-a-visitor",
        }
        client = get_client()
        st.session_state["built"] = (client.base_url == DEFAULT_URL, client.token, client.user)

    at = AppTest.from_function(script).run()
    assert at.session_state["built"] == (True, None, "noor.hassan")
