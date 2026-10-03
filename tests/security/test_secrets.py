"""Secret management: no credentials in source, configuration from the environment
or ``.env`` (never committed), secrets masked in settings, and never logged."""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path

import pytest

from app.config import load_settings
from app.observability.structured_logging import configure_logging
from app.security.secrets import REDACTED, contains_secret, redact_secrets
from tests.security.conftest import fake_key, fake_password

ROOT = Path(__file__).parents[2]
SCANNED = [
    "app/**/*",
    "scripts/**/*",
    "tests/**/*",
    "docs/**/*",
    "*.md",
    "*.toml",
    "*.yml",
    "*.yaml",
    "Dockerfile",
    ".env.example",
    ".gitignore",
    "data/sample/*",
    "data/evaluation/*.jsonl",
]


def repository_files() -> list[Path]:
    files: dict[Path, None] = {}
    for pattern in SCANNED:
        for path in ROOT.glob(pattern):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                files[path] = None
    return list(files)


def test_no_credentials_in_the_repository() -> None:
    """Every source, test, doc and config file is scanned with the same detector that
    redacts evidence and logs. Test credentials are built at run time instead."""
    files = repository_files()
    assert len(files) > 150
    findings = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if contains_secret(line):
                findings.append(f"{path.relative_to(ROOT)}:{number}: {line.strip()[:80]}")
    assert findings == []


def test_env_files_are_never_committed_and_the_example_has_no_values() -> None:
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in ignored and ".env.*" in ignored and "!.env.example" in ignored
    example = dict(
        line.split("=", 1)
        for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#") and "=" in line
    )
    for name in ("POSTGRES_PASSWORD", "LLM_API_KEY", "EMBEDDING_API_KEY", "TOOLS_SQL_PASSWORD"):
        assert example[name] == "", name


def test_settings_read_secrets_from_the_environment_and_mask_them(
    clean_env: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(f"POSTGRES_PASSWORD={fake_password()}\n", encoding="utf-8")
    clean_env.setenv("LLM_API_KEY", fake_key())
    settings = load_settings(env_file=env_file)
    assert settings.database.password is not None
    assert settings.database.password.get_secret_value() == fake_password()  # from .env
    assert (
        settings.llm.api_key is not None and settings.llm.api_key.get_secret_value() == fake_key()
    )
    rendered = repr(settings) + str(settings.model_dump()) + settings.database.safe_url
    assert fake_password() not in rendered and fake_key() not in rendered


@pytest.mark.parametrize("fmt", ["json", "console"])
def test_secrets_are_never_logged(fmt: str) -> None:
    configure_logging("INFO", fmt)
    handler = next(h for h in logging.getLogger().handlers if getattr(h, "_opsrag_handler", False))
    stream = io.StringIO()
    handler.setStream(stream)  # type: ignore[attr-defined]
    logger = logging.getLogger("test.secrets")
    try:
        logger.info(
            "calling provider with Bearer %s",
            fake_key(),
            extra={
                "api_key": fake_key(),
                "password": fake_password(),
                "authorization": "Bearer " + fake_key(),
                "headers": {"Authorization": "Basic abcdefghijklmnop", "x-trace": "ok"},
                "url": "postgresql://opsrag:" + fake_password() + "@db:5432/opsrag",
                "tokens": 42,  # a count, not a secret
            },
        )
        try:
            raise RuntimeError("connect failed: postgresql://opsrag:" + fake_password() + "@db/x")
        except RuntimeError:
            logger.exception("database.failed")
    finally:
        configure_logging("INFO", "console")
    output = stream.getvalue()
    assert fake_key() not in output and fake_password() not in output
    assert "abcdefghijklmnop" not in output
    assert REDACTED in output and "x-trace" in output
    if fmt == "json":
        first = json.loads(output.splitlines()[0])
        assert first["tokens"] == 42 and first["api_key"] == REDACTED


def test_the_redactor_on_known_formats() -> None:
    samples = {
        "openai": "key " + fake_key(),
        "opsrag_token": "token opsrag_" + "0a1b2c3d4e5f" + "_" + "k" * 43,
        "aws": "AKIA" + "ABCDEFGHIJKLMNOP",
        "github": "ghp_" + "a" * 36,
        "slack": "xoxb-" + "1234567890-abcdef",
        "jwt": "eyJ" + "a" * 10 + ".eyJ" + "b" * 10 + "." + "c" * 10,
        "pem": "-----BEGIN RSA " + "PRIVATE KEY-----\nMIIE\n-----END RSA " + "PRIVATE KEY-----",
        "env": "DB_PASSWORD=" + fake_password(),
        "json": '{"client_secret": "' + fake_password() + '"}',
        "url": "redis://cache:" + fake_password() + "@redis:6379",
        "bearer": "Authorization: Bearer " + "z" * 30,
    }
    for name, text in samples.items():
        redacted, count = redact_secrets(text)
        assert count >= 1 and REDACTED in redacted, name
    not_secrets = [
        "password_hash: Mapped[str] = mapped_column(String(256))",
        'return {"access_token": access, "token_type": "Bearer"}',
        "api_key: SecretStr | None = None",
        "settings(api_key=TEST_KEY, temperature=0.2)",
        "password=self.password.get_secret_value() if self.password else None",
        'db_password: "${VAULT_DB_PASSWORD}"',
        "client_secret=vault://payments/client",
        "# scheme://user:PASSWORD@host",
        "Rotate the signing key weekly; never paste secrets into tickets.",
    ]
    for text in not_secrets:
        assert redact_secrets(text) == (text, 0), text
