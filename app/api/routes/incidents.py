"""Incident explorer endpoints: search and filter incidents, open one, trace its change.

Every endpoint calls the same read-only tools the agent uses (``search_incidents``,
``trace_change``) through the tool registry, as the authenticated caller: the caller's
grants apply exactly as they do to the agent, and nothing here queries data directly
except the service catalog, which checks the catalog grant.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import select

from app.agents.graph import Agent
from app.api.routes.agent import get_agent
from app.api.security import PrincipalDep
from app.database.models import Service
from app.schemas.enums import Resource, Severity
from app.security.principal import Principal
from app.tools.base import ToolContext, ToolModel
from app.tools.search import IncidentSummary, SearchIncidentsOutput
from app.tools.trace import TraceChangeOutput

router = APIRouter(tags=["explorer"])
AgentDep = Annotated[Agent, Depends(get_agent)]
_STATUS = {
    "invalid_input": status.HTTP_422_UNPROCESSABLE_CONTENT,
    "permission_denied": status.HTTP_403_FORBIDDEN,
    "not_found": status.HTTP_404_NOT_FOUND,
}


class ServiceView(BaseModel):
    id: str
    name: str
    tier: str
    owner_team: str


class IncidentList(BaseModel):
    incidents: list[IncidentSummary]
    count: int
    limit: int  # the limit applied: the request's, capped at TOOLS_MAX_TOP_K


class IncidentDetail(BaseModel):
    incident: IncidentSummary


def _context(agent: Agent, principal: Principal) -> ToolContext:
    return ToolContext(
        engine=agent.engine,
        principal=principal,
        settings=agent.tool_settings,
        retriever=agent.retriever,
        sql_engine=agent.sql_engine,
    )


def _call(agent: Agent, principal: Principal, tool: str, arguments: dict[str, Any]) -> ToolModel:
    result = agent.registry.call(
        tool, {k: v for k, v in arguments.items() if v is not None}, _context(agent, principal)
    )
    if not result.ok or result.output is None:
        code = _STATUS.get(result.status, status.HTTP_500_INTERNAL_SERVER_ERROR)
        detail = result.error.message if result.error else result.status
        raise HTTPException(code, detail)
    return result.output


def _start(day: date | None) -> str | None:
    return datetime.combine(day, time.min, UTC).isoformat() if day else None


@router.get("/services", response_model=list[ServiceView], summary="The service catalog")
def services(principal: PrincipalDep, agent: AgentDep) -> list[ServiceView]:
    if not principal.may_read_unlabelled(Resource.CATALOG):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "no access to the service catalog")
    columns = (Service.id, Service.display_name, Service.tier, Service.owner_team)
    with agent.engine.connect() as connection:
        rows = connection.execute(select(*columns).order_by(Service.id)).all()
    return [
        ServiceView(id=r.id, name=r.display_name, tier=str(r.tier), owner_team=r.owner_team)
        for r in rows
    ]


@router.get(
    "/incidents",
    response_model=IncidentList,
    summary="Search and filter incidents",
    description="Text search (ranked by relevance) or filters only (newest first). Dates are "
    "inclusive UTC days. Only incidents the caller may read are returned. The search tool "
    "returns at most TOOLS_MAX_TOP_K results; `limit` in the response is the limit applied.",
)
def list_incidents(
    principal: PrincipalDep,
    agent: AgentDep,
    q: Annotated[str | None, Query(max_length=500)] = None,
    service: Annotated[list[str] | None, Query()] = None,
    severity: Annotated[list[Severity] | None, Query()] = None,
    since: date | None = None,
    until: date | None = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> IncidentList:
    query = q.strip() if q and q.strip() else None
    end = _start(until + timedelta(days=1)) if until else None
    if not (query or service or severity or since or end):
        end = datetime.now(UTC).isoformat()  # browsing: everything that started so far
    limit = min(limit, agent.tool_settings.max_top_k)
    output = _call(
        agent,
        principal,
        "search_incidents",
        {
            "query": query,
            "services": service or None,
            "severities": [s.value for s in severity] if severity else None,
            "since": _start(since),
            "until": end,
            "top_k": limit,
            "include_postmortems": False,
        },
    )
    assert isinstance(output, SearchIncidentsOutput)
    return IncidentList(incidents=output.incidents, count=len(output.incidents), limit=limit)


@router.get("/incidents/{incident_id}", response_model=IncidentDetail, summary="One incident")
def get_incident(incident_id: str, principal: PrincipalDep, agent: AgentDep) -> IncidentDetail:
    output = _call(
        agent,
        principal,
        "search_incidents",
        {"incident_ids": [incident_id], "include_postmortems": False},
    )
    assert isinstance(output, SearchIncidentsOutput)
    if not output.incidents:  # missing and not permitted look the same
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"{incident_id} was not found")
    return IncidentDetail(incident=output.incidents[0])


@router.get(
    "/incidents/{incident_id}/trace",
    response_model=TraceChangeOutput,
    summary="Follow an incident to the change behind it",
    description="Root-cause deployment, commit, pull request, files and changed lines, the "
    "upstream incident and the fix. Hops the caller may not read are listed as withheld.",
)
def trace_incident(incident_id: str, principal: PrincipalDep, agent: AgentDep) -> TraceChangeOutput:
    output = _call(agent, principal, "trace_change", {"incident_id": incident_id})
    assert isinstance(output, TraceChangeOutput)
    return output
