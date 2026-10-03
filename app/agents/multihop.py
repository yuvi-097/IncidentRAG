"""Multi-hop questions: incident -> deployment -> commit -> file -> change, and back.

"Which deployment caused INC-0033 and which code file introduced the change?" needs
several linked records, each found from the previous one. Searching for each hop
separately would return whatever is textually similar; instead the chain is followed
through the recorded links by the ``trace_change`` tool, and every hop becomes its own
evidence item (with its own access label) so each sentence of the answer cites the
record that states it.

Three directions:
- ``cause``: incident -> root-cause deployment -> commit / pull request -> file -> lines
  (and the upstream incident of a downstream one);
- ``fix``: incident -> remediation deployment -> commit / pull request;
- ``impact``: deployment -> the incidents recorded as caused by it (and its changes).

Hops the caller may not read are reported as withheld, never guessed. A question is
treated as multi-hop only when it asks for a hop beyond the deployment (commit, pull
request, file, lines, author, upstream incident), for the fix, or for the incidents a
deployment caused; "What caused INC-0421?" keeps the broader investigation plan.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel

from app.agents.entities import QueryEntities
from app.agents.state import EvidenceItem, EvidenceKind, PlannedCall
from app.schemas.enums import UNLABELLED_LEVEL
from app.tools.trace import TraceChangeOutput, TracedChange, TracedDeployment, TracedIncident

_HOP = re.compile(
    r"\b(?:commits?|hash|sha|files?|code|diffs?|lines?|pull requests?|PRs?|modif\w*|touch\w*|"
    r"upstream|downstream|triggered|chain|trace|walk|wrote|written|authored|authors?)\b",
    re.I,
)
_CAUSE = re.compile(
    r"\b(?:caus\w*|behind|responsib\w*|introduc\w*|led to|lead to|trigger\w*|root cause|"
    r"shipped|regression|trace|walk|follow|explain|upstream|downstream)\b",
    re.I,
)
_FIX = re.compile(
    r"\b(?:fix(?:ed|es|ing)?|hotfix\w*|remediat\w*|resolved by|repair\w*|reverted)\b", re.I
)
_DEPLOY = re.compile(r"\b(?:deploy\w*|releases?|rollouts?|versions?)\b", re.I)
_REVERSE = re.compile(
    r"\bincidents?\b.{0,40}\b(?:caus\w*|trigger\w*|led to|result\w*|by|followed from|"
    r"stemmed from|came from|due to|traced? (?:back )?to)\b|"
    r"\b(?:caus\w*|trigger\w*|led to)\b.{0,40}\bincidents?\b",
    re.I,
)

Direction = Literal["cause", "fix", "impact"]
WITHHELD = "withheld"  # a recorded link the caller may not follow


class ChainQuery(BaseModel):
    incident_id: str | None = None
    deployment_id: str | None = None
    direction: Direction = "cause"

    @property
    def anchor(self) -> str:
        return self.incident_id or self.deployment_id or ""


def parse_chain(question: str, entities: QueryEntities) -> ChainQuery | None:
    """The chain the question asks to follow, or None."""
    if entities.incident_ids:
        incident = entities.incident_ids[0]
        if _FIX.search(question) and (_HOP.search(question) or _DEPLOY.search(question)):
            return ChainQuery(incident_id=incident, direction="fix")
        if _HOP.search(question) and _CAUSE.search(question):
            return ChainQuery(incident_id=incident)
        return None
    if entities.deployment_ids and _REVERSE.search(question):
        return ChainQuery(deployment_id=entities.deployment_ids[0], direction="impact")
    return None


def chain_call(query: ChainQuery) -> PlannedCall:
    if query.incident_id:
        what = "its fix"
        if query.direction != "fix":
            what = "its deployment, commit, files and changes"
        return PlannedCall(
            tool="trace_change",
            arguments={"incident_id": query.incident_id},
            purpose=f"follow {query.incident_id} to {what}",
        )
    return PlannedCall(
        tool="trace_change",
        arguments={"deployment_id": query.deployment_id},
        purpose=f"follow {query.deployment_id} to its changes and the incidents it caused",
    )


# --- evidence -----------------------------------------------------------------------------


def _when(value: datetime) -> str:
    utc = value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    return f"{utc:%Y-%m-%d %H:%M} UTC"


def _incident_item(
    incident: TracedIncident, hop: str, note: str | None, fix: str | None
) -> EvidenceItem:
    upstream = incident.parent_incident_id
    parent = f" Upstream incident: {upstream}." if upstream else ""
    text = (
        f"{incident.id}: {incident.title}. Service {incident.service_id}, category "
        f"{incident.category}, started {_when(incident.started_at)}.{parent}\n"
        f"Root cause: {incident.root_cause}"
    )
    if note:
        text += f"\n{note[:1].upper()}{note[1:]}."
    if hop == "incident":
        if fix == WITHHELD:  # recorded, but not readable with the caller's role
            text += "\nThe remediation deployment is withheld: your role may not read deployments."
        elif fix:
            text += f"\nRemediation deployment: {fix}."
        else:
            text += f"\nNo remediation deployment is recorded for {incident.id}."
    return EvidenceItem(
        kind=EvidenceKind.INCIDENT,
        source_id=incident.id,
        title=incident.title,
        text=text,
        service_id=incident.service_id,
        timestamp=incident.started_at,
        access_level=incident.access_level,
        tool="trace_change",
        pinned=True,
        facts={
            "hop": hop,
            "id": incident.id,
            "title": incident.title,
            "service": incident.service_id,
            "started": _when(incident.started_at),
            "parent": incident.parent_incident_id or "",
            "note": note or "",
            "fix": fix or "",
        },
    )


def _shipped(changes: list[TracedChange]) -> str:
    return "; ".join(
        f"{c.pull_request_id} ({c.title}, by {c.author}), merge commit "
        f"{(c.merge_commit_sha or '?')[:7]}"
        for c in changes
    )


def _deployment_item(
    deployment: TracedDeployment, hop: str, changes: list[TracedChange], role: str
) -> EvidenceItem:
    commit = deployment.commit_sha
    text = (
        f"{deployment.id}: {deployment.service_id} {deployment.version}, status "
        f"{deployment.status}, deployed {_when(deployment.deployed_at)}, commit "
        f"{commit[:7]} ({commit}).{role}\nPull requests shipped: "
        f"{_shipped(changes) or 'none visible'}."
    )
    return EvidenceItem(
        kind=EvidenceKind.DEPLOYMENT,
        source_id=deployment.id,
        title=f"{deployment.service_id} {deployment.version}",
        text=text,
        service_id=deployment.service_id,
        timestamp=deployment.deployed_at,
        access_level=UNLABELLED_LEVEL,
        tool="trace_change",
        pinned=True,
        facts={
            "hop": hop,
            "id": deployment.id,
            "service": deployment.service_id,
            "version": deployment.version,
            "status": deployment.status,
            "deployed": _when(deployment.deployed_at),
            "commit": commit[:7],
            "shipped": _shipped(changes),
        },
    )


def _file_items(
    changes: list[TracedChange], deployment: TracedDeployment | None, hop: str
) -> list[EvidenceItem]:
    in_effect = deployment is None or deployment.status == "succeeded"
    items = []
    for change in changes:
        sha = (change.merge_commit_sha or "")[:7] or "?"
        for f in change.files:
            lines = "\n".join(f.changed_lines) or "(no line changes shown)"
            items.append(
                EvidenceItem(
                    kind=EvidenceKind.CODE,
                    source_id=f.code_file_id,
                    chunk_id=f"{change.pull_request_id}:{f.code_file_id}",
                    title=f"{f.path} in {change.pull_request_id}",
                    text=(
                        f"{change.pull_request_id} ({change.title}, by {change.author}, commit "
                        f"{sha}) {f.change_type} {f.path} ({f.kind}), +{f.additions} "
                        f"-{f.deletions}.\nChanged lines:\n{lines}"
                    ),
                    service_id=deployment.service_id if deployment else None,
                    timestamp=change.merged_at,
                    access_level=f.access_level,
                    location=f.path,
                    tool="trace_change",
                    pinned=True,
                    facts={
                        "hop": hop,
                        "pull_request": change.pull_request_id,
                        "author": change.author,
                        "commit": sha,
                        "path": f.path,
                        "change_type": f.change_type,
                        "lines": "\n".join(f.changed_lines),
                        "matched": ", ".join(f.matched_terms),
                        # A change shipped by a deployment that was rolled back or failed
                        # is not what runs now (used when sources disagree).
                        "valid": "true" if in_effect else "false",
                    },
                )
            )
    return items


def chain_items(output: TraceChangeOutput) -> list[EvidenceItem]:
    """One evidence item per hop of the traced chain."""
    items: list[EvidenceItem] = []
    fix = output.remediation.id if output.remediation else None
    if fix is None and any(h.startswith("remediation deployment") for h in output.withheld):
        fix = WITHHELD  # recorded, not absent: never report it as missing
    if output.incident:
        items.append(_incident_item(output.incident, "incident", output.note, fix))
    if output.parent:
        items.append(_incident_item(output.parent, "parent", None, None))
    deployment = output.deployment
    if deployment:
        if output.incident:
            role = f" It is the recorded root cause of {output.incident.id}."
        else:
            caused = "; ".join(
                f"{i.id} ({i.title}, started {_when(i.started_at)})"
                for i in output.caused_incidents
            )
            role = f"\nIncidents recorded as caused by {deployment.id}: {caused or 'none'}."
        item = _deployment_item(deployment, "deployment", output.changes, role)
        if output.incident is None:
            item = item.model_copy(update={"facts": {**item.facts, "caused": caused}})
        else:  # the recorded root cause of the traced incident
            item = item.model_copy(update={"facts": {**item.facts, "cause_of": output.incident.id}})
        items.append(item)
    if output.remediation and output.incident:
        role = f" It is the recorded fix of {output.incident.id}."
        items.append(
            _deployment_item(output.remediation, "remediation", output.remediation_changes, role)
        )
    items += _file_items(output.changes, deployment, "file")
    items += _file_items(output.remediation_changes, output.remediation, "fix_file")
    return items


# --- answer ---------------------------------------------------------------------------------


def _hop(items: list[EvidenceItem], hop: str) -> list[EvidenceItem]:
    return [i for i in items if i.facts.get("hop") == hop]


def _file_parts(files: list[EvidenceItem]) -> list[tuple[str, list[EvidenceItem]]]:
    parts: list[tuple[str, list[EvidenceItem]]] = []
    for f in files[:3]:
        facts = f.facts
        parts.append(
            (
                f"{facts['pull_request']} (commit {facts['commit']}, by {facts['author']}) "
                f"{facts['change_type']} {facts['path']}",
                [f],
            )
        )
        if facts["lines"]:
            changed = "; ".join(facts["lines"].splitlines()[:6])
            parts.append((f"Changed lines in {facts['path']}: {changed}", [f]))
    return parts


def answer_parts(
    items: list[EvidenceItem], query: ChainQuery | None = None
) -> list[tuple[str, list[EvidenceItem]]]:
    """The chain as (sentence, items it cites): the hops the question asks for first."""
    parts: list[tuple[str, list[EvidenceItem]]] = []
    incident = next(iter(_hop(items, "incident")), None)
    parent = next(iter(_hop(items, "parent")), None)
    deployment = next(iter(_hop(items, "deployment")), None)
    remediation = next(iter(_hop(items, "remediation")), None)
    if query is not None and query.direction == "fix" and incident:
        if remediation:
            r = remediation.facts
            parts.append(
                (
                    f"{incident.facts['id']} was fixed by {r['id']}: {r['service']} "
                    f"{r['version']} ({r['status']}), deployed {r['deployed']}, commit "
                    f"{r['commit']}",
                    [remediation],
                )
            )
            if r["shipped"]:
                parts.append((f"{r['id']} shipped {r['shipped']}", [remediation]))
            parts += _file_parts(_hop(items, "fix_file"))[:2]
        elif incident.facts.get("fix") == WITHHELD:
            parts.append(
                (
                    "The remediation deployment is withheld: your role may not read deployments",
                    [incident],
                )
            )
        else:
            parts.append(
                (f"No remediation deployment is recorded for {incident.facts['id']}", [incident])
            )
    if incident and parent:
        p = parent.facts
        parts.append(
            (
                f"{incident.facts['id']} was downstream impact of {p['id']} ({p['title']}, "
                f"{p['service']}, started {p['started']})",
                [incident, parent],
            )
        )
    elif incident and incident.facts.get("parent"):
        upstream = incident.facts["parent"]
        parts.append((f"{incident.facts['id']} was downstream impact of {upstream}", [incident]))
    if incident and incident.facts.get("note"):
        note = incident.facts["note"]
        parts.append((note[:1].upper() + note[1:], [incident]))
    if deployment:
        d = deployment.facts
        if incident:
            parts.append(
                (
                    f"The deployment recorded as the cause of {incident.facts['id']} is "
                    f"{d['id']}: {d['service']} {d['version']} ({d['status']}), deployed "
                    f"{d['deployed']}, commit {d['commit']}",
                    [deployment],
                )
            )
        else:
            parts.append(
                (
                    f"{d['id']} ({d['service']} {d['version']}, {d['status']}, deployed "
                    f"{d['deployed']}) is recorded as the cause of: {d['caused'] or 'no incident'}",
                    [deployment],
                )
            )
    parts += _file_parts(_hop(items, "file"))
    return parts


def withheld_notes(output: TraceChangeOutput) -> list[str]:
    return [
        f"Part of the chain was withheld because your role may not read it: {hop}."
        for hop in output.withheld
    ]


__all__ = [
    "ChainQuery",
    "answer_parts",
    "chain_call",
    "chain_items",
    "parse_chain",
    "withheld_notes",
]
