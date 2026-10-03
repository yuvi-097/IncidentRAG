"""Each tool on its own: typed input (validation), typed output, permissions and
the caller's label grants. Tools are called directly (``Tool.execute``), without the
registry."""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from typing import Any

import pytest
from sqlalchemy import select, update

from app.database.models import Document, Incident
from app.schemas.enums import AccessLevel, DocumentType, Resource, SourceType
from app.tools import (
    ResourceNotFoundError,
    Tool,
    ToolInputError,
    ToolModel,
    ToolPermissionError,
    default_tools,
)
from app.tools.deployments import SearchDeploymentsOutput, SearchDeploymentsTool
from app.tools.logs import SearchLogsOutput, SearchLogsTool
from app.tools.runbooks import GetRunbookOutput, GetRunbookTool
from app.tools.search import (
    SearchCodeTool,
    SearchDocumentsTool,
    SearchIncidentsOutput,
    SearchIncidentsTool,
    SearchResultsOutput,
)
from tests.tools.conftest import ToolEnv, custom_principal

WINDOW = {"since": "2026-06-16T17:00:00Z", "until": "2026-06-16T19:00:00Z"}


def invalid(
    tool: Tool[Any, Any], env: ToolEnv, arguments: dict[str, Any], role: str = "admin"
) -> str:
    with pytest.raises(ToolInputError) as info:
        tool.execute(arguments, env.context(role))
    return str(info.value.details or info.value.message)


# --- every tool -------------------------------------------------------------------


@pytest.mark.parametrize("tool", default_tools(), ids=lambda t: t.name)
def test_inputs_reject_unknown_fields_and_wrong_types(
    tool: Tool[Any, Any], tool_env: ToolEnv
) -> None:
    assert "Extra inputs" in invalid(tool, tool_env, {"max_access_level": "confidential"})
    with pytest.raises(ToolInputError, match="must be an object"):
        tool.execute(["not", "an", "object"], tool_env.context())  # type: ignore[arg-type]


@pytest.mark.parametrize("tool", default_tools(), ids=lambda t: t.name)
def test_specs_describe_typed_inputs_and_outputs(tool: Tool[Any, Any]) -> None:
    spec = tool.spec()
    assert (
        spec.input_schema["type"] == "object"
        and spec.input_schema.get("additionalProperties") is False
    )
    assert spec.output_schema["type"] == "object" and spec.description


# --- search_documents -------------------------------------------------------------


def test_search_documents_returns_documentation_with_provenance(tool_env: ToolEnv) -> None:
    out = SearchDocumentsTool().execute(
        {"query": "payment service configuration reference", "top_k": 5}, tool_env.context("sre")
    )
    assert isinstance(out, SearchResultsOutput) and 0 < len(out.results) <= 5
    assert {r.source_type for r in out.results} <= {SourceType.DOCUMENTATION, SourceType.RUNBOOK}
    assert out.filters.access[Resource.DOCUMENTS] == [
        AccessLevel.ENGINEERING,
        AccessLevel.PUBLIC,
        AccessLevel.SRE,
    ]
    first = out.results[0]
    assert first.chunk_id.startswith(first.document_id) and first.file_path and first.content


def test_search_documents_filters(tool_env: ToolEnv) -> None:
    out = SearchDocumentsTool().execute(
        {
            "query": "alerts and thresholds",
            "services": ["cart-service"],
            "doc_types": ["monitoring"],
        },
        tool_env.context(),
    )
    assert out.results and all(r.service_id == "cart-service" for r in out.results)
    with pytest.raises(ToolInputError, match="unknown service"):
        SearchDocumentsTool().execute(
            {"query": "cache", "services": ["nope-service"]}, tool_env.context()
        )
    assert "at least 2" in invalid(SearchDocumentsTool(), tool_env, {"query": " "})
    assert "less than or equal" in invalid(
        SearchDocumentsTool(), tool_env, {"query": "cache", "top_k": 1000}
    )
    assert "since must be earlier" in invalid(
        SearchDocumentsTool(),
        tool_env,
        {"query": "cache", "since": "2026-02-01T00:00:00Z", "until": "2026-01-01T00:00:00Z"},
    )


def test_documents_follow_the_callers_labels(tool_env: ToolEnv) -> None:
    """Labels are compartments: a developer reads public and engineering documents
    (no runbooks grant), a manager also reads MANAGER reports but not SRE documents,
    an SRE reads SRE documents but not MANAGER reports."""
    query = {"query": "payment service configuration reference reliability review", "top_k": 20}
    developer = SearchDocumentsTool().execute(query, tool_env.context("developer"))
    assert developer.results
    assert {r.access_level for r in developer.results} <= {
        AccessLevel.PUBLIC,
        AccessLevel.ENGINEERING,
    }
    assert all(r.source_type is SourceType.DOCUMENTATION for r in developer.results)
    manager = SearchDocumentsTool().execute(query, tool_env.context("manager"))
    sre = SearchDocumentsTool().execute(query, tool_env.context("sre"))
    assert AccessLevel.MANAGER in {r.access_level for r in manager.results}
    assert AccessLevel.SRE not in {r.access_level for r in manager.results}
    assert AccessLevel.SRE in {r.access_level for r in sre.results}
    assert AccessLevel.MANAGER not in {r.access_level for r in sre.results}


# --- search_code ------------------------------------------------------------------


def test_search_code_finds_identifiers_within_the_callers_labels(tool_env: ToolEnv) -> None:
    """IdempotencyStore lives only in SRE-labelled payment-service files: an admin finds
    the definition, a developer (public and engineering code) gets nothing from them.
    BM25 is the retriever here: this checks the tool, not the fusion (see
    docs/TECHNICAL_REFERENCE.md on how equal-weight RRF dilutes exact identifier matches)."""
    query = {"query": "IdempotencyStore", "top_k": 3}
    sre = SearchCodeTool().execute(query, tool_env.context("admin", retriever=tool_env.bm25))
    assert sre.results and all(r.source_type is SourceType.CODE for r in sre.results)
    paths = [r.file_path for r in sre.results]
    assert "services/payment-service/payment_service/idempotency.py" in paths  # the definition
    assert "idempotencystore" in sre.results[0].matched_terms
    developer = SearchCodeTool().execute(
        query, tool_env.context("developer", retriever=tool_env.bm25)
    )
    assert all(r.access_level is not AccessLevel.SRE for r in developer.results)
    assert not any("IdempotencyStore" in r.content for r in developer.results)
    hybrid = SearchCodeTool().execute(query, tool_env.context("admin"))
    assert all(r.source_type is SourceType.CODE for r in hybrid.results)
    with_prs = SearchCodeTool().execute(
        {"query": "OrderSaga._advance", "top_k": 10, "include_pull_requests": True},
        tool_env.context("developer"),
    )
    assert SourceType.PULL_REQUEST in {r.source_type for r in with_prs.results}


def test_code_needs_permission(tool_env: ToolEnv) -> None:
    for role in ("sre", "manager"):
        with pytest.raises(ToolPermissionError, match="code:read"):
            SearchCodeTool().execute({"query": "pool"}, tool_env.context(role))


# --- search_incidents -------------------------------------------------------------


def test_incidents_by_id_are_structured_and_linked(tool_env: ToolEnv) -> None:
    out = SearchIncidentsTool().execute(
        {"incident_ids": ["INC-0406", "INC-9999"]}, tool_env.context("sre")
    )
    assert isinstance(out, SearchIncidentsOutput) and out.not_found == ["INC-9999"]
    (incident,) = out.incidents
    source = next(i for i in tool_env.dataset.incidents if i.id == "INC-0406")
    assert (incident.deployment_id, incident.postmortem_id, incident.runbook_id) == (
        source.deployment_id,
        source.postmortem_id,
        source.runbook_id,
    )
    assert incident.severity == source.severity and incident.root_cause


def test_incidents_by_text_and_filters(tool_env: ToolEnv) -> None:
    out = SearchIncidentsTool().execute(
        {"query": "redis maxmemory writes failing", "services": ["cart-service"], "top_k": 3},
        tool_env.context(),
    )
    assert out.incidents and all(i.service_id == "cart-service" for i in out.incidents)
    assert {e.document_id for e in out.evidence} <= {i.id for i in out.incidents} | {
        i.postmortem_id for i in out.incidents if i.postmortem_id
    }
    sev1 = SearchIncidentsTool().execute({"severities": ["SEV1"], "top_k": 20}, tool_env.context())
    assert sev1.incidents and all(i.severity.value == "SEV1" for i in sev1.incidents)
    starts = [i.started_at for i in sev1.incidents]
    assert starts == sorted(starts, reverse=True)  # newest first without a query
    assert "give a query" in invalid(SearchIncidentsTool(), tool_env, {})
    assert "INC-" in invalid(SearchIncidentsTool(), tool_env, {"incident_ids": ["421"]})


def test_sensitive_incidents_are_invisible_without_the_label(tool_env: ToolEnv) -> None:
    """Developers read non-sensitive incidents only; SREs and managers read all."""
    with tool_env.engine.connect() as connection:
        sensitive = connection.execute(
            select(Incident.id).where(Incident.access_level == "sre").limit(1)
        ).scalar_one()
    developer = SearchIncidentsTool().execute(
        {"incident_ids": [sensitive]}, tool_env.context("developer")
    )
    assert developer.incidents == [] and developer.not_found == [sensitive]
    for role in ("sre", "manager", "admin"):
        out = SearchIncidentsTool().execute({"incident_ids": [sensitive]}, tool_env.context(role))
        assert [i.id for i in out.incidents] == [sensitive], role


def test_links_to_documents_the_caller_may_not_read_are_hidden(tool_env: ToolEnv) -> None:
    """INC-0406 links postmortem PM-0039. Label PM-0039 sre for this test: a developer
    must no longer see the link, an SRE still does."""
    with tool_env.engine.begin() as connection:
        original = connection.execute(
            select(Document.access_level).where(Document.id == "PM-0039")
        ).scalar_one()
        connection.execute(
            update(Document).where(Document.id == "PM-0039").values(access_level="sre")
        )
    try:
        developer = SearchIncidentsTool().execute(
            {"incident_ids": ["INC-0406"]}, tool_env.context("developer")
        )
        sre = SearchIncidentsTool().execute({"incident_ids": ["INC-0406"]}, tool_env.context("sre"))
    finally:
        with tool_env.engine.begin() as connection:
            connection.execute(
                update(Document).where(Document.id == "PM-0039").values(access_level=original)
            )
    assert developer.incidents[0].postmortem_id is None
    assert sre.incidents[0].postmortem_id == "PM-0039"


# --- search_deployments -----------------------------------------------------------


def test_deployments_by_version_list_prs_and_incidents(tool_env: ToolEnv) -> None:
    out = SearchDeploymentsTool().execute(
        {"services": ["payment-service"], "versions": ["2.8.1"]},
        tool_env.context("admin"),
    )
    assert isinstance(out, SearchDeploymentsOutput)
    (deployment,) = out.deployments
    assert (deployment.id, deployment.version, deployment.status.value) == (
        "DEP-0296",
        "v2.8.1",
        "rolled_back",
    )
    assert [p.id for p in deployment.pull_requests] == ["PR-1501"]
    relations = {(i.id, i.relation) for i in deployment.incidents}
    assert ("INC-0406", "root_cause") in relations


def test_deployment_prs_follow_the_callers_labels(tool_env: ToolEnv) -> None:
    """PR-1501 changes an SRE-labelled file: a caller whose deployments grant lacks the
    sre label does not see it; an SRE does."""
    limited = custom_principal(
        {Resource.DEPLOYMENTS: {AccessLevel.PUBLIC, AccessLevel.ENGINEERING}}
    )
    out = SearchDeploymentsTool().execute(
        {"deployment_ids": ["DEP-0296"]}, tool_env.context(limited)
    )
    assert out.deployments[0].pull_requests == []
    assert out.deployments[0].incidents == []  # no incidents grant: no linked incidents
    sre = SearchDeploymentsTool().execute({"deployment_ids": ["DEP-0296"]}, tool_env.context("sre"))
    assert [p.id for p in sre.deployments[0].pull_requests] == ["PR-1501"]


def test_deployment_listing_filters_and_order(tool_env: ToolEnv) -> None:
    out = SearchDeploymentsTool().execute(
        {"services": ["cart-service"], "rollbacks_only": True, "limit": 5}, tool_env.context()
    )
    assert out.deployments and all(
        d.is_rollback and d.service_id == "cart-service" for d in out.deployments
    )
    times = [d.deployed_at for d in out.deployments]
    assert times == sorted(times, reverse=True)
    ranked = SearchDeploymentsTool().execute(
        {"query": "Reduce idle DB connections per pod", "limit": 3}, tool_env.context()
    )
    assert ranked.deployments and ranked.evidence
    assert "DEP-" in invalid(SearchDeploymentsTool(), tool_env, {"deployment_ids": ["296"]})
    assert "pattern" in invalid(SearchDeploymentsTool(), tool_env, {"versions": ["latest"]})
    for role in ("developer", "manager"):
        with pytest.raises(ToolPermissionError, match="deployments:read"):
            SearchDeploymentsTool().execute({}, tool_env.context(role))


# --- search_logs --------------------------------------------------------------------


def test_logs_in_a_window(tool_env: ToolEnv) -> None:
    out = SearchLogsTool().execute(
        {"services": ["payment-service"], "min_level": "ERROR", "limit": 5, **WINDOW},
        tool_env.context(),
    )
    assert isinstance(out, SearchLogsOutput) and 0 < len(out.entries) <= 5
    assert all(
        e.service_id == "payment-service" and e.level.value in {"ERROR", "CRITICAL"}
        for e in out.entries
    )
    assert out.total_matched == sum(out.level_counts.values()) >= len(out.entries)
    assert out.truncated == (out.total_matched > len(out.entries))
    stamps = [e.timestamp for e in out.entries]
    assert stamps == sorted(stamps, reverse=True)


def test_log_text_is_a_literal_substring(tool_env: ToolEnv) -> None:
    out = SearchLogsTool().execute({"text": "queuepool LIMIT", **WINDOW}, tool_env.context())
    assert out.entries and all("queuepool limit" in e.message.lower() for e in out.entries)
    wildcard = SearchLogsTool().execute({"text": "%_%", **WINDOW}, tool_env.context())
    assert all("%_%" in e.message for e in wildcard.entries)  # not a LIKE pattern


def test_logs_by_deployment_or_trace(tool_env: ToolEnv) -> None:
    by_deployment = SearchLogsTool().execute(
        {"deployment_id": "DEP-0296", "limit": 500}, tool_env.context()
    )
    assert by_deployment.entries and all(
        e.deployment_id == "DEP-0296" for e in by_deployment.entries
    )
    trace = next(e.trace_id for e in by_deployment.entries if e.trace_id)
    by_trace = SearchLogsTool().execute({"trace_id": trace}, tool_env.context())
    assert by_trace.entries and all(e.trace_id == trace for e in by_trace.entries)


def test_log_searches_must_be_bounded(tool_env: ToolEnv) -> None:
    assert "time window" in invalid(SearchLogsTool(), tool_env, {"services": ["payment-service"]})
    assert "either levels or min_level" in invalid(
        SearchLogsTool(), tool_env, {"levels": ["ERROR"], "min_level": "WARNING", **WINDOW}
    )
    with pytest.raises(ToolInputError, match="longer than 31 days"):
        SearchLogsTool().execute(
            {"since": "2026-01-01T00:00:00Z", "until": "2026-06-01T00:00:00Z"}, tool_env.context()
        )
    assert "pattern" in invalid(SearchLogsTool(), tool_env, {"trace_id": "not-a-trace"})


def test_logs_need_permission(tool_env: ToolEnv) -> None:
    for role in ("developer", "manager"):
        with pytest.raises(ToolPermissionError, match="logs:read"):
            SearchLogsTool().execute(WINDOW, tool_env.context(role))


# --- get_runbook --------------------------------------------------------------------


def test_runbook_by_id_title_and_search(tool_env: ToolEnv) -> None:
    by_id = GetRunbookTool().execute({"runbook_id": "RB-0049"}, tool_env.context())
    assert isinstance(by_id, GetRunbookOutput) and by_id.matched_by == "id"
    source = next(d for d in tool_env.dataset.documents if d.id == "RB-0049")
    assert by_id.runbook.content == source.content and "Mitigation" in by_id.runbook.sections
    by_title = GetRunbookTool().execute({"title": source.title.upper()}, tool_env.context())
    assert by_title.runbook.id == "RB-0049" and by_title.matched_by == "title"
    by_search = GetRunbookTool().execute(
        {"query": "redis memory pressure", "service": "cart-service"}, tool_env.context()
    )
    assert by_search.matched_by == "search" and by_search.runbook.service_id == "cart-service"


def test_runbook_input_rules(tool_env: ToolEnv) -> None:
    assert "exactly one" in invalid(GetRunbookTool(), tool_env, {})
    assert "exactly one" in invalid(
        GetRunbookTool(), tool_env, {"runbook_id": "RB-0001", "title": "x y"}
    )
    assert "only applies" in invalid(
        GetRunbookTool(), tool_env, {"runbook_id": "RB-0001", "service": "cart-service"}
    )
    assert "RB-" in invalid(GetRunbookTool(), tool_env, {"runbook_id": "DOC-0001"})


def test_missing_and_forbidden_runbooks_look_the_same(tool_env: ToolEnv) -> None:
    with tool_env.engine.connect() as connection:
        sensitive = connection.execute(
            select(Document.id)
            .where(Document.doc_type == DocumentType.RUNBOOK, Document.access_level == "sre")
            .limit(1)
        ).scalar_one()
    limited = custom_principal({Resource.RUNBOOKS: {AccessLevel.PUBLIC, AccessLevel.ENGINEERING}})
    missing = _not_found(
        lambda: GetRunbookTool().execute({"runbook_id": "RB-9999"}, tool_env.context(limited))
    )
    hidden = _not_found(
        lambda: GetRunbookTool().execute({"runbook_id": sensitive}, tool_env.context(limited))
    )
    assert hidden == missing
    assert (
        GetRunbookTool().execute({"runbook_id": sensitive}, tool_env.context("sre")).runbook.id
        == sensitive
    )
    for role in ("developer", "manager"):
        with pytest.raises(ToolPermissionError, match="runbooks:read"):
            GetRunbookTool().execute({"runbook_id": "RB-0049"}, tool_env.context(role))


def _not_found(call: Callable[[], ToolModel]) -> str:
    with pytest.raises(ResourceNotFoundError) as info:
        call()
    return info.value.message


def test_outputs_serialise_to_json(tool_env: ToolEnv) -> None:
    ctx = tool_env.context("sre")
    outputs = [
        SearchDocumentsTool().execute({"query": "canary analysis"}, ctx),
        SearchIncidentsTool().execute({"incident_ids": ["INC-0406"]}, ctx),
        SearchDeploymentsTool().execute({"deployment_ids": ["DEP-0296"]}, ctx),
        SearchLogsTool().execute({**WINDOW, "limit": 2}, ctx),
        GetRunbookTool().execute({"runbook_id": "RB-0049"}, ctx),
    ]
    for output in outputs:
        assert type(output).model_validate_json(output.model_dump_json()) == output


def test_relative_window_defaults(tool_env: ToolEnv) -> None:
    out = SearchLogsTool().execute({"since": "2026-06-16T17:00:00Z"}, tool_env.context())
    assert out.until is not None and out.since is not None
    assert out.until - out.since == timedelta(days=31)
