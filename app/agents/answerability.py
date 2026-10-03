"""What to say when the evidence is not enough, and what evidence would help."""

from __future__ import annotations

import re

from app.agents.state import AgentState
from app.schemas.enums import QueryType

ROOT_CAUSE_NO_ANSWER = (
    "I don't have sufficient evidence in the available knowledge base to determine the root cause."
)
GENERIC_NO_ANSWER = (
    "I don't have sufficient evidence in the available knowledge base to answer this question."
)
OUT_OF_SCOPE = (
    "This question is outside what OpsRAG can answer from NovaCart's operational data "
    "(incidents, deployments, logs, code and documentation)."
)
_ROOT_CAUSE = re.compile(r"\b(?:why|cause[ds]?|root cause|fail\w*|broke|outage|went down)\b", re.I)


def no_answer_text(state: AgentState) -> str:
    if state.query_type is QueryType.UNKNOWN:
        return OUT_OF_SCOPE
    causal = state.query_type in {QueryType.INCIDENT_SEARCH, QueryType.MULTI_SOURCE}
    return ROOT_CAUSE_NO_ANSWER if causal or _ROOT_CAUSE.search(state.query) else GENERIC_NO_ANSWER


def suggest_evidence(state: AgentState) -> list[str]:
    """Concrete additional evidence that would let the question be answered."""
    entities = state.entities
    services = (
        ", ".join(entities.services) if entities and entities.services else "the affected service"
    )
    when = f" ({entities.time_range.expression})" if entities and entities.time_range else ""
    ideas: list[str] = []
    if state.query_type is QueryType.UNKNOWN:
        return ["A question about NovaCart incidents, deployments, logs, code or documentation."]
    per_goal = {
        "incident": f"Incident records or postmortems for {services}{when}.",
        "change": (
            f"Deployment history for {services} before the incident (versions, pull requests)."
        ),
        "corroboration": (
            f"Application logs for {services} during the incident window, "
            "or the code change behind it."
        ),
        "documents": "Documentation or a runbook that covers this topic.",
        "runbook": f"A runbook for this situation on {services}.",
        "code": "Source files or pull requests that implement this behaviour.",
        "deployment": f"Deployment records for {services}{when}.",
        "logs": "A time window, trace id or deployment id to scope the log search.",
        "sql_result": (
            "An aggregate question the SQL templates support (counts, average resolution time, "
            "top-N, per-service or per-month breakdowns), or an analyst query."
        ),
    }
    for goal, met in state.goals.items():
        if not met and goal in per_goal:
            ideas.append(per_goal[goal])
    if state.unknown_terms:
        ideas.append(
            f"Sources that mention {', '.join(state.unknown_terms)}; the knowledge base has none."
        )
    for note in state.evidence_notes:
        if note.endswith("was not found or is not accessible"):
            ideas.append(
                f"The record {note.split()[0]} (it does not exist or your role cannot see it)."
            )
    for record in state.tool_results:
        if record.status != "ok":
            ideas.append(
                f"Results from {record.tool}, which "
                + ("was not permitted." if record.status == "permission_denied" else "failed.")
            )
    for limitation in state.limitations:
        if "not permitted for role" in limitation:
            tool = limitation.split()[0]
            ideas.append(
                f"Access to {tool} (your role cannot call it); ask someone whose role can."
            )
    if not ideas:
        ideas.append("More specific identifiers: an incident id, service, version or time window.")
    return list(dict.fromkeys(ideas))
