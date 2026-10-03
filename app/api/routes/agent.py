"""POST /api/agent/ask: ask the incident-response agent a question.

The caller authenticates with an API token (see ``app/api/security.py``). The user's
role decides which tools may run and what data is visible.
"""

from __future__ import annotations

import threading
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, Request
from sqlalchemy.engine import Engine

from app.agents.graph import Agent
from app.agents.response import AgentResponse, AskRequest
from app.api.dependencies import get_engine
from app.api.security import PrincipalDep
from app.observability.metrics import METRICS
from app.services.agent import build_agent

router = APIRouter(prefix="/agent", tags=["agent"])
_lock = threading.Lock()


def ensure_agent(app: FastAPI, engine: Engine) -> Agent:
    """The app's agent: built once (loads the models), then shared. Called by the first
    request, or at startup when OPSRAG_PRELOAD_AGENT is on."""
    agent = getattr(app.state, "agent", None)
    if agent is None:
        with _lock:
            agent = getattr(app.state, "agent", None)
            if agent is None:
                # Timed apart: this one-off model loading lands in the first request's
                # HTTP latency (unless preloaded), and the dashboard shows it on its own.
                with METRICS.timed("agent.build"):
                    agent = build_agent(engine, app.state.settings)
                app.state.agent = agent
    return agent


def get_agent(request: Request, engine: Annotated[Engine, Depends(get_engine)]) -> Agent:
    return ensure_agent(request.app, engine)


@router.post(
    "/ask",
    response_model=AgentResponse,
    summary="Ask a question",
    description=(
        "Routes the question, calls read-only tools within the user's permissions, validates "
        "the evidence and returns a cited answer with a short per-stage summary. No model "
        "reasoning is returned."
    ),
)
def ask(
    body: AskRequest, principal: PrincipalDep, agent: Annotated[Agent, Depends(get_agent)]
) -> AgentResponse:
    state = agent.run(body.question, principal)
    METRICS.record_agent_run(state)
    return AgentResponse.from_state(state)
