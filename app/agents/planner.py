"""Tool planning: which calls to make, in what order, and when to stop.

The planner is rule-based and explainable, like the router. It works in two steps:

- ``initial`` turns the routing decision and the extracted entities into the first
  calls. A simple question gets one call. Structured anchors are used first when the
  question names them: an incident id or a deployment version is fetched directly
  instead of searched for.
- ``follow_ups`` reads each result and plans what it makes necessary. An incident's
  root-cause deployment gets fetched; its time window scopes the log search; a
  root-cause pull request is looked up in the code. Later calls are parameterised
  by earlier results, so the agent does not re-run one retriever for everything.

``goals`` names the evidence a question needs (for a causal question: the incident,
the change, and corroboration from logs or code). Execution stops as soon as every
goal is met, or when the call budget runs out.

Two structured plans take precedence when the question asks for them:

- temporal (``temporal.py``): fetch the anchor record, then query the target records
  by time relative to the anchor's timestamps (goals: anchor, timeline);
- chain (``multihop.py``): follow the recorded links from an incident to the change
  behind it, or from a deployment to the incidents it caused (goal: chain).
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any

from app.agents import temporal
from app.agents.multihop import chain_call
from app.agents.sql_templates import build_sql
from app.agents.state import AgentState, PlannedCall
from app.config import AgentSettings
from app.schemas.enums import LogLevel, QueryType
from app.tools.base import ToolModel
from app.tools.deployments import SearchDeploymentsOutput
from app.tools.logs import SearchLogsOutput
from app.tools.runbooks import GetRunbookOutput
from app.tools.search import SearchIncidentsOutput, SearchResultsOutput
from app.tools.sql_tool import QueryDatabaseOutput
from app.tools.trace import TraceChangeOutput

Q = QueryType
_FIX = re.compile(
    r"\b(?:fix|resolve|mitigat\w*|recover|remediat\w*|what should we do|runbook)\b", re.I
)


def _iso(value: datetime) -> str:
    return value.isoformat()


class ToolPlanner:
    def __init__(self, settings: AgentSettings, dialect: str) -> None:
        self.settings = settings
        self.dialect = dialect

    # --- goals ------------------------------------------------------------------------

    def goals(self, state: AgentState) -> dict[str, bool]:
        """The evidence this question needs; all False until satisfied."""
        if state.temporal:
            return {"anchor": False, "timeline": False}
        if state.chain:
            return {"chain": False}
        query_type = state.query_type
        tools = state.routing.tools if state.routing else []
        needed = {
            Q.DOCUMENT_SEARCH: ["documents"],
            Q.INCIDENT_SEARCH: ["incident"],
            Q.CODE_SEARCH: ["code"],
            Q.SQL_QUERY: ["sql_result"],
            Q.DEPLOYMENT_SEARCH: ["deployment"],
            Q.LOG_SEARCH: ["logs"],
            Q.MULTI_SOURCE: ["incident", "change", "corroboration"],
        }.get(query_type, [])  # type: ignore[arg-type]
        if query_type is Q.DOCUMENT_SEARCH and "get_runbook" in tools:
            needed = [*needed, "runbook"]
        return dict.fromkeys(needed, False)

    def satisfied(self, state: AgentState) -> bool:
        return bool(state.goals) and all(state.goals.values())

    def update_goals(self, state: AgentState, call: PlannedCall, output: ToolModel) -> None:
        goals = state.goals
        if state.temporal:
            if call.purpose.startswith(temporal.ANCHOR) and state.temporal_anchor is None:
                goals["anchor"] = temporal.anchor_from(state.temporal, output) is not None
            elif call.purpose.startswith(temporal.TARGET):
                goals["timeline"] = True  # no matching record is also an answer
            return
        if isinstance(output, TraceChangeOutput):
            if "chain" in goals:
                goals["chain"] = True
            return
        if isinstance(output, SearchResultsOutput) and output.results:
            if call.tool == "search_documents" and "documents" in goals:
                goals["documents"] = True
            if call.tool == "search_code":
                for goal in ("code", "corroboration"):
                    if goal in goals:
                        goals[goal] = True
        elif isinstance(output, SearchIncidentsOutput) and output.incidents:
            if "incident" in goals:
                goals["incident"] = True
            if "corroboration" in goals and any(
                e.source_type.value == "postmortem" for e in output.evidence
            ):
                goals["corroboration"] = True
        elif isinstance(output, SearchDeploymentsOutput) and output.deployments:
            for goal in ("deployment", "change"):
                if goal in goals:
                    goals[goal] = True
        elif isinstance(output, SearchLogsOutput):
            if "logs" in goals:
                goals["logs"] = True  # zero matching lines is also an answer
            if "corroboration" in goals and output.entries:
                goals["corroboration"] = True
        elif isinstance(output, GetRunbookOutput) and "runbook" in goals:
            goals["runbook"] = True
        elif isinstance(output, QueryDatabaseOutput) and "sql_result" in goals:
            goals["sql_result"] = True

    # --- initial calls ----------------------------------------------------------------

    def initial(self, state: AgentState) -> list[PlannedCall]:
        entities, routing = state.entities, state.routing
        assert entities is not None and routing is not None
        query, k = state.query, self.settings.results_per_tool
        services = entities.services or None
        window: dict[str, Any] = {}
        if entities.time_range:
            window = {
                "since": _iso(entities.time_range.since),
                "until": _iso(entities.time_range.until),
            }
        query_type = state.query_type
        calls: list[PlannedCall] = []

        if state.temporal:
            anchor = temporal.anchor_call(state.temporal)
            if anchor is not None:
                return [anchor]
            state.temporal_anchor = temporal.fixed_anchor(state.temporal, state.now)
            state.goals["anchor"] = True
            return [temporal.target_call(state.temporal, state.temporal_anchor)]
        if state.chain:
            return [chain_call(state.chain)]
        if query_type is Q.DOCUMENT_SEARCH:
            calls.append(
                PlannedCall(
                    tool="search_documents",
                    arguments={"query": query, "top_k": k, "services": services},
                    purpose="search documentation and runbooks",
                )
            )
            if "get_runbook" in routing.tools:
                calls.append(self._runbook_call(state))
        elif query_type is Q.INCIDENT_SEARCH:
            calls.append(self._incident_call(state, window))
        elif query_type is Q.CODE_SEARCH:
            calls.append(
                PlannedCall(
                    tool="search_code",
                    arguments={"query": query, "top_k": k, "services": services},
                    purpose="search the source code",
                )
            )
        elif query_type is Q.SQL_QUERY:
            plan = build_sql(query, entities, self.dialect)
            if plan is None:
                state.limit(
                    "This aggregate question could not be translated into SQL by the "
                    "built-in templates; no SQL was run."
                )
            else:
                calls.append(
                    PlannedCall(
                        tool="query_database",
                        arguments={"sql": plan.sql, "max_rows": 50},
                        purpose=plan.description,
                    )
                )
        elif query_type is Q.DEPLOYMENT_SEARCH:
            calls.append(self._deployment_call(state, window))
        elif query_type is Q.LOG_SEARCH:
            if entities.incident_ids and not window:
                calls.append(self._incident_call(state, window))  # its window scopes the logs
            else:
                calls.append(self._log_call(state, window))
        elif query_type is Q.MULTI_SOURCE:
            # routing.tools holds only tools the caller may use: a role without the
            # deployments grant anchors on the failure instead of the change.
            if (entities.deployment_ids or entities.versions) and (
                "search_deployments" in routing.tools
            ):
                calls.append(self._deployment_call(state, window))  # anchor on the change
            else:
                calls.append(self._incident_call(state, window))  # anchor on the failure
        return [c for c in calls if c is not None]

    def _incident_call(self, state: AgentState, window: dict[str, Any]) -> PlannedCall:
        entities = state.entities
        assert entities is not None
        if entities.incident_ids:
            return PlannedCall(
                tool="search_incidents",
                arguments={"incident_ids": entities.incident_ids[:20]},
                purpose=f"fetch {', '.join(entities.incident_ids[:3])}",
            )
        arguments: dict[str, Any] = {
            "query": state.query,
            "top_k": 3 if state.query_type is Q.MULTI_SOURCE else self.settings.results_per_tool,
            "services": entities.services or None,
            "severities": [s.value for s in entities.severities] or None,
            **window,
        }
        return PlannedCall(
            tool="search_incidents", arguments=arguments, purpose="search past incidents"
        )

    def _deployment_call(self, state: AgentState, window: dict[str, Any]) -> PlannedCall:
        entities = state.entities
        assert entities is not None
        arguments: dict[str, Any] = {
            "deployment_ids": entities.deployment_ids or None,
            "versions": entities.versions or None,
            "services": entities.services or None,
            **window,
        }
        if not (entities.deployment_ids or entities.versions or entities.services or window):
            arguments["query"] = state.query
        purpose = "fetch " + ", ".join(
            [*entities.deployment_ids, *entities.versions][:3] or ["deployments"]
        )
        return PlannedCall(tool="search_deployments", arguments=arguments, purpose=purpose)

    def _log_call(self, state: AgentState, window: dict[str, Any]) -> PlannedCall:
        entities = state.entities
        assert entities is not None
        arguments: dict[str, Any] = {
            "services": entities.services or None,
            "text": entities.quoted[0] if entities.quoted else None,
            "trace_id": entities.trace_ids[0] if entities.trace_ids else None,
            "deployment_id": entities.deployment_ids[0] if entities.deployment_ids else None,
            "limit": 50,
        }
        if entities.log_levels:
            order = list(LogLevel)
            arguments["min_level"] = min(entities.log_levels, key=order.index).value
        if window:
            arguments.update(window)
        elif not (arguments["trace_id"] or arguments["deployment_id"]):
            hours = self.settings.default_log_window_hours
            arguments["since"] = _iso(state.now - timedelta(hours=hours))
            arguments["until"] = _iso(state.now)
            state.limit(f"No time window was given; logs were searched for the last {hours} hours.")
        return PlannedCall(tool="search_logs", arguments=arguments, purpose="search the logs")

    def _runbook_call(self, state: AgentState) -> PlannedCall:
        entities = state.entities
        assert entities is not None
        runbooks = [d for d in entities.document_ids if d.startswith("RB-")]
        if runbooks:
            return PlannedCall(
                tool="get_runbook",
                arguments={"runbook_id": runbooks[0]},
                purpose=f"fetch {runbooks[0]}",
            )
        service = entities.services[0] if len(entities.services) == 1 else None
        return PlannedCall(
            tool="get_runbook",
            arguments={"query": state.query, "service": service},
            purpose="find the matching runbook",
        )

    # --- follow-ups -------------------------------------------------------------------

    def follow_ups(
        self, state: AgentState, call: PlannedCall, output: ToolModel
    ) -> list[PlannedCall]:
        query_type = state.query_type
        wants_fix = bool(_FIX.search(state.query))
        calls: list[PlannedCall] = []
        if state.temporal:
            if call.purpose.startswith(temporal.ANCHOR) and state.temporal_anchor is None:
                anchor = temporal.anchor_from(state.temporal, output)
                if anchor is not None:
                    state.temporal_anchor = anchor
                    calls.append(temporal.target_call(state.temporal, anchor))
            return calls
        if state.chain:
            return calls
        if isinstance(output, SearchDeploymentsOutput) and query_type is Q.MULTI_SOURCE:
            linked: list[str] = []
            for deployment in output.deployments:
                ordered = sorted(deployment.incidents, key=lambda i: i.relation != "root_cause")
                linked += [i.id for i in ordered if i.id not in linked]
            if linked:
                calls.append(
                    PlannedCall(
                        tool="search_incidents",
                        arguments={"incident_ids": linked[:3]},
                        purpose=f"fetch incidents linked to {output.deployments[0].id}",
                    )
                )
            elif output.deployments:
                first = output.deployments[0]
                calls.append(
                    PlannedCall(
                        tool="search_incidents",
                        arguments={
                            "query": state.query,
                            "top_k": 3,
                            "services": [first.service_id],
                            "since": _iso(first.deployed_at),
                        },
                        purpose=f"search incidents after {first.id}",
                    )
                )
        elif isinstance(output, SearchIncidentsOutput) and output.incidents:
            incident = output.incidents[0]
            if query_type is Q.MULTI_SOURCE:
                deployment = incident.root_cause_deployment_id or incident.deployment_id
                if not state.goals.get("change"):
                    calls.append(
                        PlannedCall(
                            tool="search_deployments",
                            arguments={"deployment_ids": [deployment]},
                            purpose=f"fetch {deployment}, the change linked to {incident.id}",
                        )
                    )
                pad = timedelta(minutes=self.settings.log_window_padding_minutes)
                calls.append(
                    PlannedCall(
                        tool="search_logs",
                        arguments={
                            "services": [incident.service_id],
                            "min_level": "WARNING",
                            "since": _iso(incident.started_at - pad),
                            "until": _iso(incident.resolved_at),
                            "limit": 50,
                        },
                        purpose=f"logs of {incident.service_id} during {incident.id}",
                    )
                )
                if incident.root_cause_pr_id:
                    calls.append(
                        PlannedCall(
                            tool="search_code",
                            arguments={
                                "query": incident.root_cause_pr_id,
                                "top_k": 3,
                                "services": [incident.service_id],
                                "include_pull_requests": True,
                            },
                            purpose=f"code change {incident.root_cause_pr_id}",
                        )
                    )
            if wants_fix and incident.runbook_id:
                calls.append(
                    PlannedCall(
                        tool="get_runbook",
                        arguments={"runbook_id": incident.runbook_id},
                        purpose=f"runbook {incident.runbook_id} for {incident.id}",
                    )
                )
                # A fix was asked for and a runbook exists: do not stop before reading it.
                state.goals.setdefault("runbook", False)
            if query_type is Q.LOG_SEARCH:
                pad = timedelta(minutes=self.settings.log_window_padding_minutes)
                calls.append(
                    PlannedCall(
                        tool="search_logs",
                        arguments={
                            "services": [incident.service_id],
                            "since": _iso(incident.started_at - pad),
                            "until": _iso(incident.resolved_at),
                            "limit": 50,
                        },
                        purpose=f"logs during {incident.id}",
                    )
                )
        return calls
