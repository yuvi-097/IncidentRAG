"""The central tool registry: the only way the (future) agent can act.

- Only ``Tool`` instances can be registered, each under its fixed name. Plain
  functions, lambdas or arbitrary callables are refused, and the registry can be
  frozen so nothing is added after start-up.
- ``call(name, arguments, context)`` looks the tool up by name, then runs
  ``Tool.execute``, which validates the arguments against the input model, checks
  the caller's permission, and type-checks the output. There is no path to eval,
  exec, imports or shell commands.
- Every call returns a ``ToolResult`` envelope (status, output or error) and is
  logged as a structured event without the argument values (they can contain
  queries or SQL); a hash of them is logged for correlation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, SerializeAsAny

from app.observability import telemetry
from app.security.principal import Principal
from app.tools.base import (
    Tool,
    ToolContext,
    ToolError,
    ToolModel,
    ToolNotFoundError,
    ToolSpec,
)

logger = logging.getLogger(__name__)
_NAME = re.compile(r"^[a-z][a-z0-9_]{2,63}$")

Status = Literal[
    "ok",
    "invalid_input",
    "unsafe_sql",
    "permission_denied",
    "not_found",
    "unknown_tool",
    "execution_error",
]


class ToolErrorInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    message: str
    details: list[dict[str, Any]] = []


class ToolResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    tool: str
    status: Status
    output: SerializeAsAny[ToolModel] | None = None
    error: ToolErrorInfo | None = None
    duration_ms: float

    @property
    def ok(self) -> bool:
        return self.status == "ok"


class RegistryError(RuntimeError):
    pass


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool[Any, Any]] = ()) -> None:
        self._tools: dict[str, Tool[Any, Any]] = {}
        self._frozen = False
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool[Any, Any]) -> None:
        if self._frozen:
            raise RegistryError("the registry is frozen")
        if not isinstance(tool, Tool):
            raise RegistryError(f"only Tool instances can be registered, not {type(tool).__name__}")
        if not _NAME.fullmatch(tool.name):
            raise RegistryError(f"invalid tool name {tool.name!r}")
        if tool.name in self._tools:
            raise RegistryError(f"tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool

    def freeze(self) -> ToolRegistry:
        self._frozen = True
        return self

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def get(self, name: str) -> Tool[Any, Any]:
        tool = self._tools.get(name) if isinstance(name, str) else None
        if tool is None:
            raise ToolNotFoundError(f"no tool named {name!r}; available: {', '.join(self.names)}")
        return tool

    def specs(self, context: ToolContext | None = None) -> list[ToolSpec]:
        """Tool descriptions for the agent; with a context, only those it may call."""
        tools = [self._tools[n] for n in self.names]
        if context is not None:
            tools = [t for t in tools if context.principal.can(t.permission)]
        return [t.spec() for t in tools]

    def permitted(self, principal: Principal) -> Callable[[str], bool]:
        """For the router: may ``principal`` call the tool with this name?"""
        return lambda name: name in self._tools and principal.can(self._tools[name].permission)

    def invoke(self, name: str, arguments: Mapping[str, Any], context: ToolContext) -> ToolModel:
        """Run a tool and return its output; raises ``ToolError`` subclasses."""
        return self.get(name).execute(arguments, context)

    def call(self, name: str, arguments: Mapping[str, Any], context: ToolContext) -> ToolResult:
        """Run a tool and wrap the outcome; tool errors become a status, never raise."""
        started = time.perf_counter()
        try:
            output = self.invoke(name, arguments, context)
            result = ToolResult(tool=name, status="ok", output=output, duration_ms=0.0)
        except ToolError as exc:
            result = ToolResult(
                tool=str(name),
                status=exc.code,  # type: ignore[arg-type]
                error=ToolErrorInfo(message=exc.message, details=exc.details),
                duration_ms=0.0,
            )
        except Exception as exc:  # a bug or an outage inside a tool: report, never raise
            logger.exception("tool.crashed", extra={"tool": str(name)[:64]})
            result = ToolResult(
                tool=str(name),
                status="execution_error",
                error=ToolErrorInfo(message=f"{name} failed unexpectedly ({type(exc).__name__})"),
                duration_ms=0.0,
            )
        duration = round((time.perf_counter() - started) * 1000, 1)
        result = result.model_copy(update={"duration_ms": duration})
        telemetry.add_tool(
            str(name)[:64],
            result.status,
            result.output.result_count() if result.ok and result.output is not None else 0,
        )
        logger.info(
            "tool.called",
            extra={
                "tool": str(name)[:64],
                "status": result.status,
                "user": context.principal.user_id,
                "role": context.principal.role,
                "duration_ms": duration,
                "argument_keys": sorted(str(k) for k in arguments)[:20]
                if isinstance(arguments, Mapping)
                else [],
                "arguments_sha256": _digest(arguments),
            },
        )
        return result


def _digest(arguments: object) -> str:
    try:
        payload = json.dumps(arguments, sort_keys=True, default=str)
    except (TypeError, ValueError):
        payload = repr(arguments)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def default_tools() -> list[Tool[Any, Any]]:
    from app.tools.deployments import SearchDeploymentsTool
    from app.tools.logs import SearchLogsTool
    from app.tools.runbooks import GetRunbookTool
    from app.tools.search import SearchCodeTool, SearchDocumentsTool, SearchIncidentsTool
    from app.tools.sql_tool import QueryDatabaseTool
    from app.tools.trace import TraceChangeTool

    return [
        SearchDocumentsTool(),
        SearchIncidentsTool(),
        SearchCodeTool(),
        SearchDeploymentsTool(),
        SearchLogsTool(),
        GetRunbookTool(),
        QueryDatabaseTool(),
        TraceChangeTool(),
    ]


def build_registry() -> ToolRegistry:
    """The eight read-only tools, frozen."""
    return ToolRegistry(default_tools()).freeze()
