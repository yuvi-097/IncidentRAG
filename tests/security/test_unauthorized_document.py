"""Security test 1: unauthorized documents.

A role never receives a document outside its grants, from any tool, retriever, SQL
query or the agent. Rows are filtered inside the database queries, so an
unauthorized document is never loaded, ranked, packaged or shown to a model.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from fastapi import FastAPI
from sqlalchemy import func, select

from app.database.models import DocumentChunk
from app.schemas.enums import AccessLevel
from app.security import SOURCE_RESOURCE
from app.tools import UnsafeSqlError
from app.tools.search import SearchDocumentsTool
from app.tools.sql_tool import QueryDatabaseTool
from tests.agents.conftest import make_agent
from tests.security.conftest import api_client, ask_api, user_of
from tests.tools.conftest import ToolEnv, principal

ROLES = ("developer", "sre", "manager", "admin")
BENCHMARK = Path(__file__).parents[2] / "data" / "evaluation" / "retrieval_benchmark.jsonl"
SRE_DOC, MANAGER_DOC, ADMIN_DOC, CONFIDENTIAL_DOC = "DOC-0060", "RPT-0004", "POL-0002", "DOC-0100"


def _questions() -> list[str]:
    lines = BENCHMARK.read_text(encoding="utf-8").splitlines()
    return [json.loads(line)["question"] for line in lines if line.strip()]


def _search(env: ToolEnv, role: str, query: str, bm25: bool = False) -> set[str]:
    out = SearchDocumentsTool().execute(
        {"query": query, "top_k": 20, "include_postmortems": True},
        env.context(role, retriever=env.bm25 if bm25 else None),
    )
    return {r.document_id for r in out.results}


@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("bm25", [False, True], ids=["hybrid", "bm25"])
def test_no_search_returns_a_chunk_outside_the_grants(
    tool_env: ToolEnv, role: str, bm25: bool
) -> None:
    """All 63 benchmark questions, per role and retriever: every returned chunk is a
    (kind, label) pair the role was granted."""
    context = tool_env.context(role, retriever=tool_env.bm25 if bm25 else None)
    principal = context.principal
    returned = 0
    for question in _questions():
        out = SearchDocumentsTool().execute(
            {"query": question, "top_k": 20, "include_postmortems": True}, context
        )
        for r in out.results:
            returned += 1
            assert principal.may_read(SOURCE_RESOURCE[r.source_type], r.access_level), (
                role,
                question,
                r.chunk_id,
            )
    assert returned > 0


@pytest.mark.parametrize(
    ("query", "document", "allowed"),
    [
        ("Payment Service Configuration Reference", SRE_DOC, {"sre", "admin"}),
        ("Reliability Review 2026-Q2 incidents by service", MANAGER_DOC, {"manager", "admin"}),
        ("Break-glass Production Access Procedure", ADMIN_DOC, {"admin"}),
    ],
    ids=["sre-document", "manager-report", "admin-policy"],
)
def test_labelled_documents_reach_only_their_roles(
    tool_env: ToolEnv, query: str, document: str, allowed: set[str]
) -> None:
    for role in ROLES:
        for bm25 in (False, True):
            found = document in _search(tool_env, role, query, bm25)
            assert found == (role in allowed), (role, document, bm25)


def test_confidential_documents_are_never_indexed_or_retrieved(tool_env: ToolEnv) -> None:
    with tool_env.engine.connect() as connection:
        chunks = connection.execute(
            select(func.count())
            .select_from(DocumentChunk)
            .where(DocumentChunk.access_level == AccessLevel.CONFIDENTIAL)
        ).scalar_one()
    assert chunks == 0  # nothing to retrieve, by any path
    query = "Secrets Management Vault provider API keys rotate quarterly break-glass"
    for role in ROLES:
        assert CONFIDENTIAL_DOC not in _search(tool_env, role, query)
        assert CONFIDENTIAL_DOC not in _search(tool_env, role, query, bm25=True)
    sql = {"sql": f"SELECT id, content FROM documents WHERE id = '{CONFIDENTIAL_DOC}'"}
    for role in ("sre", "manager", "admin"):
        assert QueryDatabaseTool().execute(sql, tool_env.context(role)).rows == []


def test_sql_over_documents_follows_the_grants(tool_env: ToolEnv) -> None:
    ids = f"'{SRE_DOC}', '{MANAGER_DOC}', '{ADMIN_DOC}', '{CONFIDENTIAL_DOC}'"
    sql = {"sql": f"SELECT id FROM documents WHERE id IN ({ids}) ORDER BY id"}

    def visible(role: str) -> list[str]:
        return [row[0] for row in QueryDatabaseTool().execute(sql, tool_env.context(role)).rows]

    assert visible("sre") == [SRE_DOC]
    assert visible("manager") == [MANAGER_DOC]
    assert visible("admin") == [SRE_DOC, ADMIN_DOC, MANAGER_DOC]
    with pytest.raises(Exception, match="sql:read"):
        visible("developer")  # no SQL at all
    with pytest.raises(UnsafeSqlError):
        QueryDatabaseTool().execute(
            {"sql": "SELECT id FROM chunk_embeddings"}, tool_env.context("admin")
        )


def exclusive_passages(env: ToolEnv, doc_id: str, size: int = 8) -> set[str]:
    """Word sequences that occur in ``doc_id`` and in no other record: if one shows up
    in a response, that document's content leaked."""

    def shingles(text: str) -> set[str]:
        words = re.findall(r"[a-z0-9_]+", text.lower())
        return {" ".join(words[i : i + size]) for i in range(len(words) - size + 1)}

    target = next(d.content for d in env.dataset.documents if d.id == doc_id)
    others: set[str] = set()
    for d in env.dataset.documents:
        if d.id != doc_id:
            others |= shingles(d.content)
    for c in env.dataset.code_files:
        others |= shingles(c.content)
    for i in env.dataset.incidents:
        others |= shingles(f"{i.symptoms} {i.root_cause} {i.resolution}")
    exclusive = shingles(target) - others
    assert len(exclusive) > 20  # the document has plenty of content of its own
    return exclusive


def leaked(body: object, passages: set[str]) -> list[str]:
    text = " ".join(re.findall(r"[a-z0-9_]+", json.dumps(body).lower()))
    return [p for p in passages if p in text]


def test_the_agent_never_sees_unauthorized_documents(tool_env: ToolEnv) -> None:
    """The positive control (an SRE retrieves DOC-0060) shows the difference is the
    grants, not the question."""
    agent = make_agent(tool_env)
    question = "What is in the Payment Service Configuration Reference?"
    developer = agent.run(question, principal("developer"))
    sre = agent.run(question, principal("sre"))
    assert SRE_DOC in {i.source_id for i in sre.retrieved_documents}
    assert SRE_DOC in {c.source_id for c in sre.citations}
    assert SRE_DOC not in {i.source_id for i in developer.retrieved_documents}
    assert SRE_DOC not in {c.source_id for c in developer.citations}
    assert developer.security.access_violations == 0  # filtered in the query, not later
    passages = exclusive_passages(tool_env, SRE_DOC)
    assert leaked(developer.final_answer, passages) == []
    assert leaked(sre.final_answer, passages)  # the SRE's answer does quote it


def test_the_api_response_never_contains_unauthorized_content(
    app: FastAPI, tool_env: ToolEnv
) -> None:
    """Neither DOC-0060's content nor its id appears for a developer, not even where a
    readable document cross-references it (the reference is removed before synthesis)."""
    client = api_client(app, tool_env)
    question = "What is in the Payment Service Configuration Reference?"
    passages = exclusive_passages(tool_env, SRE_DOC)
    developer = ask_api(client, user_of(tool_env, "developer"), question)
    assert SRE_DOC not in {c["source_id"] for c in developer["citations"]}
    assert SRE_DOC not in {e["source_id"] for e in developer["evidence"]}
    assert leaked(developer, passages) == []
    developer.pop("question")  # the caller's own words
    assert SRE_DOC not in json.dumps(developer)
    sre = ask_api(client, user_of(tool_env, "sre"), question)
    assert SRE_DOC in {c["source_id"] for c in sre["citations"]}
    assert leaked(sre, passages)
