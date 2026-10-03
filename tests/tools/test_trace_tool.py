"""Phase 9: trace_change follows recorded links, hop by hop, within the caller's grants."""

from __future__ import annotations

import pytest

from app.schemas.enums import AccessLevel, Resource
from app.tools.trace import TraceChangeOutput, changed_lines
from tests.tools.conftest import ToolEnv, custom_principal


def trace(env: ToolEnv, role: object = "admin", **arguments: str) -> TraceChangeOutput:
    result = env.registry.call("trace_change", arguments, env.context(role))  # type: ignore[arg-type]
    assert result.ok, result.error
    assert isinstance(result.output, TraceChangeOutput)
    return result.output


def test_an_incident_is_followed_to_the_changed_lines(tool_env: ToolEnv) -> None:
    out = trace(tool_env, incident_id="INC-0033")
    incident = next(i for i in tool_env.dataset.incidents if i.id == "INC-0033")
    assert out.incident and out.incident.id == "INC-0033"
    assert out.deployment and out.deployment.id == incident.root_cause_deployment_id
    (change,) = out.changes
    assert change.pull_request_id == incident.root_cause_pr_id
    pr = next(p for p in tool_env.dataset.pull_requests if p.id == change.pull_request_id)
    assert change.merge_commit_sha == pr.merge_commit_sha == out.deployment.commit_sha
    assert change.author == pr.author_id
    (changed,) = change.files
    assert changed.path.endswith("payment-service.yaml")
    assert any(line.startswith("+ ") for line in changed.changed_lines)
    assert out.withheld == [] and out.note is None


def test_the_fix_is_followed_too(tool_env: ToolEnv) -> None:
    incident = next(
        i
        for i in tool_env.dataset.incidents
        if i.remediation_deployment_id and i.root_cause_deployment_id
    )
    out = trace(tool_env, incident_id=incident.id)
    assert out.remediation and out.remediation.id == incident.remediation_deployment_id
    shipped = {
        p.id for p in tool_env.dataset.pull_requests if p.deployment_id == out.remediation.id
    }
    assert {c.pull_request_id for c in out.remediation_changes} == shipped


def test_a_downstream_incident_names_its_upstream_one(tool_env: ToolEnv) -> None:
    out = trace(tool_env, incident_id="INC-0028")
    assert out.parent and out.parent.id == "INC-0027"


def test_an_operational_incident_has_no_recorded_change(tool_env: ToolEnv) -> None:
    incident = next(
        i
        for i in tool_env.dataset.incidents
        if not i.root_cause_deployment_id and not i.parent_incident_id
    )
    out = trace(tool_env, incident_id=incident.id)
    assert out.deployment is None and out.changes == []
    assert out.note and "no deployment is recorded as the cause" in out.note


def test_a_deployment_is_followed_to_the_incidents_it_caused(tool_env: ToolEnv) -> None:
    out = trace(tool_env, deployment_id="DEP-0031")
    caused = sorted(
        i.id for i in tool_env.dataset.incidents if i.root_cause_deployment_id == "DEP-0031"
    )
    assert [i.id for i in out.caused_incidents] == caused and len(caused) >= 2
    assert out.incident is None and out.remediation is None


def test_hops_the_caller_may_not_read_are_withheld_not_dropped(tool_env: ToolEnv) -> None:
    # Developers read incidents and code but have no deployments grant.
    out = trace(tool_env, "developer", incident_id="INC-0033")
    assert out.incident is not None
    assert out.deployment is None
    assert any("deployment DEP-" in hop for hop in out.withheld)
    # Without the code grant, the diff is withheld as well.
    incidents_only = custom_principal({Resource.INCIDENTS: {AccessLevel.ENGINEERING}})
    out = trace(tool_env, incidents_only, incident_id="INC-0033")
    assert out.deployment is None and all(not c.files for c in out.changes)
    assert out.withheld


def test_an_incident_the_caller_may_not_read_looks_missing(tool_env: ToolEnv) -> None:
    restricted = next(
        i for i in tool_env.dataset.incidents if i.access_level is not AccessLevel.ENGINEERING
    )
    nobody = custom_principal({Resource.INCIDENTS: {AccessLevel.ENGINEERING}})
    hidden = tool_env.registry.call(
        "trace_change", {"incident_id": restricted.id}, tool_env.context(nobody)
    )
    missing = tool_env.registry.call(
        "trace_change", {"incident_id": "INC-9999"}, tool_env.context(nobody)
    )
    assert hidden.status == missing.status == "not_found"


def test_exactly_one_anchor_is_required(tool_env: ToolEnv) -> None:
    both = tool_env.registry.call(
        "trace_change", {"incident_id": "INC-0033", "deployment_id": "DEP-0043"}, tool_env.context()
    )
    neither = tool_env.registry.call("trace_change", {}, tool_env.context())
    assert both.status == neither.status == "invalid_input"


@pytest.mark.parametrize(
    ("patch", "expected"),
    [
        ("--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a = 1\n+a = 2\n b = 3\n", ["- a = 1", "+ a = 2"]),
        ('+  KEY: "0.5"\n+\n', ['+ KEY: "0.5"']),
        (None, []),
    ],
)
def test_changed_lines_keep_only_additions_and_removals(
    patch: str | None, expected: list[str]
) -> None:
    assert changed_lines(patch, 10) == expected
