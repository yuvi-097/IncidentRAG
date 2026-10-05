"""The container definitions: start order, health checks, configuration and hardening.

These read docker-compose.yml and the Dockerfiles; they do not need Docker. Starting the
stack (docker compose up --build) is the real test, documented in the README.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from app.config import load_settings

ROOT = Path(__file__).resolve().parents[1]
APP_SERVICES = ("bootstrap", "backend")

# Inside the test image the container definitions are not copied (.dockerignore).
pytestmark = pytest.mark.skipif(
    not (ROOT / "docker-compose.yml").exists(),
    reason="the container definitions are not part of the test image",
)


class ComposeLoader(yaml.SafeLoader):
    """YAML plus Compose's merge tags (``!reset``, ``!override``), read as plain values."""


def _merge_tag(loader: yaml.SafeLoader, node: yaml.Node) -> Any:
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    return loader.construct_scalar(node)  # type: ignore[arg-type]


for _tag in ("!reset", "!override"):
    ComposeLoader.add_constructor(_tag, _merge_tag)


def load(path: Path) -> dict[str, Any]:
    # ComposeLoader is a SafeLoader subclass: no arbitrary Python objects.
    return yaml.load(path.read_text(encoding="utf-8"), Loader=ComposeLoader)


@pytest.fixture(scope="module")
def compose() -> dict[str, Any]:
    return load(ROOT / "docker-compose.yml")


def stages(dockerfile: Path) -> dict[str, list[str]]:
    """Instructions per build stage (continuation lines joined)."""
    text = re.sub(r"\\\n", " ", dockerfile.read_text(encoding="utf-8"))
    out: dict[str, list[str]] = {}
    current, count = "_", 0  # "_": global ARGs before the first FROM
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"FROM\s+\S+(?:\s+AS\s+(\S+))?", line, re.I)
        if match:
            current = match.group(1) or f"stage{count}"
            count += 1
        out.setdefault(current, []).append(line)
    return out


def test_the_spec_services_exist_and_start_in_order(compose: dict[str, Any]) -> None:
    services = compose["services"]
    assert {"postgres", "backend", "frontend"} <= set(services)
    depends = {name: s.get("depends_on", {}) for name, s in services.items()}
    assert depends["bootstrap"]["postgres"]["condition"] == "service_healthy"
    assert depends["backend"]["postgres"]["condition"] == "service_healthy"
    assert depends["backend"]["bootstrap"]["condition"] == "service_completed_successfully"
    assert depends["frontend"]["backend"]["condition"] == "service_healthy"
    assert services["tests"]["profiles"] == ["test"]  # never started by `up`


def test_long_running_services_have_health_checks(compose: dict[str, Any]) -> None:
    services = compose["services"]
    assert "pg_isready" in " ".join(services["postgres"]["healthcheck"]["test"])
    assert "/api/ready" in " ".join(services["backend"]["healthcheck"]["test"])
    assert "/_stcore/health" in " ".join(services["frontend"]["healthcheck"]["test"])
    for name in ("postgres", "backend", "frontend"):
        assert services[name]["restart"] == "unless-stopped"
    assert services["bootstrap"]["restart"] == "no"  # a one-shot job
    # It runs the backend image, whose HEALTHCHECK is the API's: disabled for the job
    # (found by the Phase 14 Docker run: the job showed as "unhealthy").
    assert services["bootstrap"]["healthcheck"] == {"disable": True}


def test_the_database_persists_and_is_not_exposed(compose: dict[str, Any]) -> None:
    postgres = compose["services"]["postgres"]
    assert "pgdata:/var/lib/postgresql/data" in postgres["volumes"]
    assert "pgdata" in compose["volumes"]
    assert all(port.startswith("127.0.0.1:") for port in postgres["ports"])


def test_no_secret_is_written_into_the_compose_files(compose: dict[str, Any]) -> None:
    for path in ROOT.glob("docker-compose*.yml"):
        data = load(path)
        for service in data["services"].values():
            for key, value in (service.get("environment") or {}).items():
                if re.search(r"PASSWORD|TOKEN|KEY|SECRET", key):
                    assert str(value).startswith("${"), f"{path.name}: {key} is inline"


def test_app_containers_run_in_production_mode(compose: dict[str, Any]) -> None:
    for name in APP_SERVICES:
        env = compose["services"][name]["environment"]
        assert env["OPSRAG_ENVIRONMENT"] == "production"
        assert env["SECURITY_ALLOW_USER_HEADER"] == "false"
        assert env["POSTGRES_HOST"] == "postgres"
        assert "TOOLS_SQL_PASSWORD" in env
    assert compose["services"]["backend"]["environment"]["OPSRAG_PRELOAD_AGENT"] == "true"
    assert compose["services"]["tests"]["environment"]["POSTGRES_DB"] == "opsrag_test"
    demo = load(ROOT / "docker-compose.demo.yml")
    assert demo["services"]["backend"]["environment"]["OPSRAG_ENVIRONMENT"] == "local"


def test_every_configured_variable_is_a_real_setting(
    compose: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A typo in a variable name would be silently ignored by the app: load them."""
    env = {
        k: str(v)
        for k, v in compose["services"]["backend"]["environment"].items()
        if not str(v).startswith("${")
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("POSTGRES_PASSWORD", "a-strong-password-for-tests")
    monkeypatch.setenv("TOOLS_SQL_PASSWORD", "another-strong-password-1")
    monkeypatch.setenv("TOOLS_SQL_USER", "opsrag_sql_reader")
    settings = load_settings(env_file=None)  # production checks pass with these
    assert settings.app.environment == "production" and settings.app.preload_agent
    assert settings.database.host == "postgres" and not settings.security.allow_user_header


def test_the_backend_image_is_non_root_offline_and_checks_readiness() -> None:
    by_stage = stages(ROOT / "Dockerfile")
    backend = " ".join(by_stage["backend"])
    assert "USER 10001:10001" in backend and "useradd" in backend
    assert "HF_HUB_OFFLINE=1" in backend  # models only from the image
    assert "/api/ready" in backend and "HEALTHCHECK" in backend
    assert "OPSRAG_ENVIRONMENT=production" in backend
    assert "download.pytorch.org/whl/cpu" in " ".join(by_stage["deps"])  # no CUDA wheels
    assert "download_models.py" in " ".join(by_stage["models"])
    copied = [line.split()[1] for line in by_stage["backend"] if line.startswith("COPY ")]
    for path in copied:
        if not path.startswith("--"):
            assert (ROOT / path).exists(), path
    test = " ".join(by_stage["test"])
    assert "requirements-dev.txt" in test and "COPY tests" in test


def test_the_frontend_image_is_non_root_and_checks_health() -> None:
    lines = " ".join(stages(ROOT / "frontend" / "Dockerfile")["stage0"])
    assert "USER 10002:10002" in lines and "/_stcore/health" in lines
    assert "COPY frontend/" in lines  # built from the repository root


def test_the_build_context_keeps_secrets_out() -> None:
    ignored = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in ignored and ".venv" in ignored
    assert "data/evaluation/cache" in ignored


def test_the_test_stage_copies_every_requirements_file_it_includes() -> None:
    """requirements-dev.txt pulls in others with `-r`; each must be in the stage (the
    Phase 14 Docker run failed on this before)."""
    test = " ".join(stages(ROOT / "Dockerfile")["test"])
    included = re.findall(r"^-r\s+(\S+)", (ROOT / "requirements-dev.txt").read_text(), re.M)
    assert included
    for name in ["requirements-dev.txt", *included]:
        assert name in test, name


def test_the_public_demo_exposes_only_the_ui() -> None:
    """docker-compose.public.yml shares the UI through a tunnel. Its API trusts the demo-user
    header, so the API must get no host port and the tunnel must point at the UI."""
    path = ROOT / "docker-compose.public.yml"
    services = load(path)["services"]
    assert "ports: !reset []" in path.read_text(encoding="utf-8")  # a plain [] would merge
    assert services["backend"]["ports"] == []
    assert services["backend"]["environment"]["SECURITY_ALLOW_USER_HEADER"] == "true"
    assert services["frontend"]["environment"]["OPSRAG_UI_PUBLIC_DEMO"] == "true"
    tunnel = services["tunnel"]
    assert "http://frontend:8501" in tunnel["command"]
    assert not any("backend" in part for part in tunnel["command"])
    assert re.search(r":\d{4}\.\d+\.\d+$", tunnel["image"]), "pin a cloudflared release"
    assert tunnel["depends_on"]["frontend"]["condition"] == "service_healthy"
