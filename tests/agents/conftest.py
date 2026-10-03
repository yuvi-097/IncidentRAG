from __future__ import annotations

from collections.abc import Callable

import pytest

from app.agents.entities import ServiceCatalog
from app.agents.graph import Agent
from app.agents.response import AgentResponse
from app.agents.router import RuleBasedRouter
from app.agents.state import AgentState
from app.tools import build_registry
from tests.tools.conftest import NOW, ToolEnv, principal, tool_env  # noqa: F401  (shared fixture)

Ask = Callable[..., AgentState]


def make_agent(env: ToolEnv, **overrides: object) -> Agent:
    """The agent over the test corpus. BM25 is the retriever (deterministic stand-in for
    the model pipeline); no cross-encoder, no LLM, a fixed clock."""
    options: dict[str, object] = {
        "engine": env.engine,
        "registry": build_registry(),
        "router": RuleBasedRouter(ServiceCatalog.from_engine(env.engine), clock=lambda: NOW),
        "retriever": env.bm25,
        "term_stats": env.bm25.index,
        "clock": lambda: NOW,
    }
    options.update(overrides)
    return Agent(**options)  # type: ignore[arg-type]


@pytest.fixture(scope="session")
def agent(tool_env: ToolEnv) -> Agent:  # noqa: F811
    return make_agent(tool_env)


@pytest.fixture(scope="session")
def ask(agent: Agent) -> Ask:
    def run(question: str, role: str = "admin") -> AgentState:
        return agent.run(question, principal(role))

    return run


def response(state: AgentState) -> AgentResponse:
    return AgentResponse.from_state(state)
