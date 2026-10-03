"""References to records the caller may not read are removed from evidence: ids with
their "(Title)", and titles on their own. The caller learns neither content nor
existence; roles that may read them see the references unchanged."""

from __future__ import annotations

import json

from fastapi import FastAPI

from app.agents.references import HIDDEN, HIDDEN_TITLE, ReferenceFilter
from app.agents.state import EvidenceItem, EvidenceKind
from app.schemas.enums import AccessLevel
from tests.agents.conftest import make_agent
from tests.security.conftest import api_client, ask_api, user_of
from tests.tools.conftest import ToolEnv, principal

FIX_PR, RUNBOOK = "PR-1504", "RB-0049"  # both linked from INC-0406's resolution


def _title(env: ToolEnv, pr: str) -> str:
    return next(p.title for p in env.dataset.pull_requests if p.id == pr)


def test_incident_links_the_caller_may_not_read_are_removed(tool_env: ToolEnv) -> None:
    agent = make_agent(tool_env)
    developer = agent.run("What caused INC-0406?", principal("developer"))
    text = json.dumps([i.model_dump(mode="json") for i in developer.retrieved_documents])
    for hidden in (FIX_PR, RUNBOOK, _title(tool_env, FIX_PR)):
        assert hidden not in text and hidden not in developer.final_answer, hidden
    assert HIDDEN in text and developer.security.references_redacted >= 2
    sre = agent.run("What caused INC-0406?", principal("sre"))
    sre_text = json.dumps([i.model_dump(mode="json") for i in sre.retrieved_documents])
    assert FIX_PR in sre_text and RUNBOOK in sre_text and sre.security.references_redacted == 0


def test_document_cross_references_are_removed(app: FastAPI, tool_env: ToolEnv) -> None:
    """An engineering document mentions DOC-0060 (SRE-only) by id and title."""
    client = api_client(app, tool_env)
    question = "What is in the payment service configuration and its references?"
    body = ask_api(client, user_of(tool_env, "developer"), question)
    body.pop("question")
    text = json.dumps(body)
    assert "DOC-0060" not in text and "Payment Service Configuration Reference" not in text
    sre = json.dumps(ask_api(client, user_of(tool_env, "sre"), question))
    assert "DOC-0060" in sre


def _item(text: str, kind: EvidenceKind = EvidenceKind.SQL_RESULT) -> EvidenceItem:
    return EvidenceItem(
        kind=kind,
        source_id="sql",
        title="query result",
        text=text,
        access_level=AccessLevel.ENGINEERING,
        tool="query_database",
    )


def test_ids_in_sql_rows_are_checked_too(tool_env: ToolEnv) -> None:
    """A manager may count incidents, but may not read runbooks or SRE code."""
    references = ReferenceFilter(tool_env.engine)
    sensitive_file = next(
        c.id for c in tool_env.dataset.code_files if c.access_level.value == "sre"
    )
    rows = _item(f"id=INC-0406, runbook_id={RUNBOOK}, file={sensitive_file}, deployment=DEP-0296")
    (manager,), removed = references.redact(principal("manager"), [rows])
    assert RUNBOOK not in manager.text and sensitive_file not in manager.text
    assert "INC-0406" in manager.text and "DEP-0296" in manager.text  # readable / unlabelled
    assert removed == 2
    (admin,), none = references.redact(principal("admin"), [rows])
    assert admin.text == rows.text and none == 0


def test_titles_are_removed_only_when_hidden(tool_env: ToolEnv) -> None:
    references = ReferenceFilter(tool_env.engine)
    report = next(d for d in tool_env.dataset.documents if d.id == "RPT-0004")
    item = _item(f"See {report.title} for the quarter.", EvidenceKind.DOCUMENT)
    (sre,), _ = references.redact(principal("sre"), [item])
    (manager,), _ = references.redact(principal("manager"), [item])
    assert sre.text == f"See {HIDDEN_TITLE} for the quarter."
    assert manager.text == item.text


def test_an_item_never_hides_itself(tool_env: ToolEnv) -> None:
    references = ReferenceFilter(tool_env.engine)
    own = EvidenceItem(
        kind=EvidenceKind.INCIDENT,
        source_id="INC-0406",
        title="INC-0406",
        text="INC-0406: payment requests failed.",
        access_level=AccessLevel.ENGINEERING,
        tool="search_incidents",
    )
    (kept,), removed = references.redact(principal("developer"), [own])
    assert kept.text == own.text and removed == 0
