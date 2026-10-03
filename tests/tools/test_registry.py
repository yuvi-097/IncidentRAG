"""The registry: the only way to call tools, and nothing but registered tools."""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Any

import pytest

from app.schemas.enums import ToolPermission
from app.tools import (
    RegistryError,
    Tool,
    ToolContext,
    ToolModel,
    ToolNotFoundError,
    ToolRegistry,
    build_registry,
    default_tools,
)
from tests.tools.conftest import ToolEnv

EXPECTED = {
    "search_documents",
    "search_incidents",
    "search_code",
    "search_deployments",
    "search_logs",
    "get_runbook",
    "query_database",
    "trace_change",  # Phase 9: follows recorded links, read-only like the others
}


class _EchoInput(ToolModel):
    text: str


class _Echo(Tool[_EchoInput, _EchoInput]):
    name = "echo_tool"
    description = "test double"
    permission = ToolPermission.DOCUMENTS_READ
    input_model = _EchoInput
    output_model = _EchoInput

    def _run(self, arguments: _EchoInput, context: ToolContext) -> _EchoInput:
        return arguments


def test_the_registry_holds_exactly_the_eight_read_only_tools() -> None:
    registry = build_registry()
    assert set(registry.names) == EXPECTED
    assert {spec.name for spec in registry.specs()} == EXPECTED
    with pytest.raises(RegistryError, match="frozen"):
        registry.register(_Echo())


def test_only_tool_instances_can_be_registered() -> None:
    registry = ToolRegistry()
    for thing in (print, lambda **kwargs: eval("1"), "search_documents", _Echo):
        with pytest.raises(RegistryError, match="only Tool instances"):
            registry.register(thing)  # type: ignore[arg-type]
    registry.register(_Echo())
    with pytest.raises(RegistryError, match="already registered"):
        registry.register(_Echo())


@pytest.mark.parametrize(
    "name", ["eval", "exec", "__import__", "os.system", "", "search_documents; drop"]
)
def test_unregistered_names_cannot_be_called(tool_env: ToolEnv, name: str) -> None:
    registry = build_registry()
    result = registry.call(
        name, {"code": "__import__('os').system('echo hi')"}, tool_env.context("sre")
    )
    assert result.status == "unknown_tool" and result.output is None
    with pytest.raises(ToolNotFoundError):
        registry.invoke(name, {}, tool_env.context("sre"))


def test_call_returns_an_envelope_for_every_outcome(tool_env: ToolEnv) -> None:
    registry = build_registry()
    ok = registry.call("search_incidents", {"incident_ids": ["INC-0406"]}, tool_env.context("sre"))
    assert ok.ok and ok.output is not None and ok.error is None and ok.duration_ms >= 0
    assert ok.model_dump(mode="json")["output"]["incidents"][0]["id"] == "INC-0406"
    outcomes = {
        "invalid_input": registry.call(
            "search_incidents", {"incident_ids": ["bad"]}, tool_env.context()
        ),
        "unsafe_sql": registry.call(
            "query_database", {"sql": "DROP TABLE incidents"}, tool_env.context()
        ),
        "permission_denied": registry.call(
            "search_logs", {"trace_id": "0" * 32}, tool_env.context("developer")
        ),
        "not_found": registry.call("get_runbook", {"runbook_id": "RB-9999"}, tool_env.context()),
        "execution_error": registry.call(
            "query_database", {"sql": "SELECT nope FROM incidents"}, tool_env.context()
        ),
    }
    for status, result in outcomes.items():
        assert result.status == status and result.error is not None and result.output is None


def test_specs_are_filtered_by_permission(tool_env: ToolEnv) -> None:
    registry = build_registry()

    def names(role: str) -> set[str]:
        return {s.name for s in registry.specs(tool_env.context(role))}

    assert names("developer") == {
        "search_documents",
        "search_code",
        "search_incidents",
        "trace_change",
    }
    assert names("sre") == EXPECTED - {"search_code"}
    assert names("manager") == {
        "search_documents",
        "search_incidents",
        "query_database",
        "trace_change",
    }
    assert names("admin") == EXPECTED
    permitted = registry.permitted(tool_env.context("developer").principal)
    assert (
        permitted("search_incidents") and not permitted("query_database") and not permitted("eval")
    )


def test_calls_are_logged_without_argument_values(
    tool_env: ToolEnv, caplog: pytest.LogCaptureFixture
) -> None:
    secret = "SELECT count(*) FROM incidents WHERE title = 'very-secret-marker'"
    with caplog.at_level(logging.INFO, logger="app.tools.registry"):
        build_registry().call("query_database", {"sql": secret}, tool_env.context())
    (record,) = [r for r in caplog.records if r.getMessage() == "tool.called"]
    fields: dict[str, Any] = record.__dict__
    assert fields["tool"] == "query_database" and fields["status"] == "ok"
    assert fields["argument_keys"] == ["sql"] and len(fields["arguments_sha256"]) == 16
    assert "very-secret-marker" not in repr(fields)


def test_the_tool_layer_cannot_execute_code() -> None:
    """No eval/exec/compile/__import__, no subprocess/os.system, no dynamic imports
    anywhere in app/tools."""
    builtins = {"eval", "exec", "compile", "__import__", "open", "getattr", "setattr"}
    methods = {"system", "popen", "import_module", "eval", "exec", "__import__"}
    forbidden_modules = {"subprocess", "os", "importlib", "pickle", "marshal", "ctypes", "builtins"}
    for path in Path("app/tools").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name):
                    assert func.id not in builtins, f"{path}: {func.id}()"
                elif isinstance(func, ast.Attribute):
                    assert func.attr not in methods, f"{path}: .{func.attr}()"
            if isinstance(node, ast.Import | ast.ImportFrom):
                modules = (
                    [a.name for a in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module or ""]
                )
                assert not {m.split(".")[0] for m in modules} & forbidden_modules, (
                    f"{path}: {modules}"
                )


def test_default_tools_are_all_read_only_tool_classes() -> None:
    for tool in default_tools():
        assert isinstance(tool, Tool)
        assert tool.permission in set(ToolPermission)
        assert tool.name in EXPECTED
