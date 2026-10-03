from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import MIN_PRODUCTION_PASSWORD_LENGTH, load_settings


@pytest.mark.usefixtures("clean_env")
def test_defaults() -> None:
    settings = load_settings(env_file=None)

    assert settings.app.environment == "local"
    assert settings.app.log_level == "INFO"
    assert settings.database.host == "127.0.0.1"
    assert settings.database.connect_timeout_seconds == 2
    assert settings.database.port == 5432
    assert settings.database.url.drivername == "postgresql+psycopg"
    assert settings.llm.provider == "none"
    assert settings.llm.enabled is False
    assert settings.embedding.dimension == 384
    assert settings.retrieval.mode == "hybrid_rerank"
    assert settings.retrieval.fusion == "rrf" and settings.retrieval.rrf_k == 60
    assert settings.retrieval.dense_weight == settings.retrieval.sparse_weight == 1.0
    assert settings.retrieval.rerank_candidates == 30
    assert (settings.bm25.k1, settings.bm25.b) == (1.2, 0.75)
    assert settings.reranker.enabled


def test_environment_variables_override_defaults(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("OPSRAG_ENVIRONMENT", "staging")
    clean_env.setenv("OPSRAG_LOG_LEVEL", "debug")
    clean_env.setenv("POSTGRES_HOST", "db.internal")
    clean_env.setenv("POSTGRES_PORT", "6543")
    clean_env.setenv("LLM_PROVIDER", "  Some-Provider ")
    clean_env.setenv("LLM_MODEL", "some-model")
    clean_env.setenv("EMBEDDING_DIMENSION", "768")

    settings = load_settings(env_file=None)

    assert settings.app.environment == "staging"
    assert settings.app.log_level == "DEBUG"
    assert settings.database.host == "db.internal"
    assert settings.database.port == 6543
    assert settings.llm.provider == "some-provider"
    assert settings.llm.enabled is True
    assert settings.embedding.dimension == 768


def test_env_file_is_read_and_environment_wins(
    clean_env: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("POSTGRES_DB=from_file\nPOSTGRES_USER=file_user\n", encoding="utf-8")
    clean_env.setenv("POSTGRES_USER", "env_user")

    settings = load_settings(env_file=env_file)

    assert settings.database.db == "from_file"
    assert settings.database.user == "env_user"


@pytest.mark.usefixtures("clean_env")
def test_empty_values_are_treated_as_unset(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_MODEL=\nLLM_API_KEY=\nLLM_BASE_URL=\n", encoding="utf-8")

    settings = load_settings(env_file=env_file)

    assert settings.llm.model is None
    assert settings.llm.api_key is None
    assert settings.llm.base_url is None


def test_database_url_escapes_password_and_masks_it(clean_env: pytest.MonkeyPatch) -> None:
    password = "p@ss:" + "w/rd#1"  # built at run time
    clean_env.setenv("POSTGRES_PASSWORD", password)

    settings = load_settings(env_file=None)

    assert settings.database.url.password == password
    assert password not in settings.database.safe_url
    assert "***" in settings.database.safe_url


def test_secrets_are_not_exposed_in_repr(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("POSTGRES_PASSWORD", "db-secret-value")
    clean_env.setenv("LLM_PROVIDER", "some-provider")
    clean_env.setenv("LLM_MODEL", "some-model")
    clean_env.setenv("LLM_API_KEY", "llm-secret-value")

    settings = load_settings(env_file=None)
    rendered = repr(settings) + str(settings.model_dump())

    assert "db-secret-value" not in rendered
    assert "llm-secret-value" not in rendered


def test_no_credential_has_a_default(clean_env: pytest.MonkeyPatch) -> None:
    settings = load_settings(env_file=None)

    assert settings.database.password is None and settings.database.url.password is None
    assert settings.llm.api_key is None and settings.embedding.api_key is None


@pytest.mark.parametrize(
    "password",
    [
        None,  # unset
        "short-secret",  # too short
        "opsrag_" + "local_dev",  # the old local-development password (checked by hash)
    ],
)
def test_production_rejects_weak_or_development_passwords(
    clean_env: pytest.MonkeyPatch, password: str | None
) -> None:
    clean_env.setenv("OPSRAG_ENVIRONMENT", "production")
    if password is not None:
        clean_env.setenv("POSTGRES_PASSWORD", password)

    with pytest.raises(ValidationError, match="POSTGRES_PASSWORD"):
        load_settings(env_file=None)


def test_production_accepts_strong_passwords_and_a_sql_reader(
    clean_env: pytest.MonkeyPatch,
) -> None:
    clean_env.setenv("OPSRAG_ENVIRONMENT", "production")
    clean_env.setenv("POSTGRES_PASSWORD", "x" * MIN_PRODUCTION_PASSWORD_LENGTH)
    clean_env.setenv("TOOLS_SQL_USER", "opsrag_sql_reader")
    clean_env.setenv("TOOLS_SQL_PASSWORD", "y" * MIN_PRODUCTION_PASSWORD_LENGTH)

    assert load_settings(env_file=None).app.environment == "production"


@pytest.mark.parametrize(
    ("user", "password", "message"),
    [
        (None, None, "TOOLS_SQL_USER"),  # no dedicated role
        ("opsrag_sql_reader", "short", "TOOLS_SQL_USER"),  # weak password
        ("opsrag", "y" * 20, "must differ"),  # the application's own role
    ],
)
def test_production_requires_a_dedicated_sql_reader(
    clean_env: pytest.MonkeyPatch, user: str | None, password: str | None, message: str
) -> None:
    clean_env.setenv("OPSRAG_ENVIRONMENT", "production")
    clean_env.setenv("POSTGRES_PASSWORD", "x" * MIN_PRODUCTION_PASSWORD_LENGTH)
    if user:
        clean_env.setenv("TOOLS_SQL_USER", user)
    if password:
        clean_env.setenv("TOOLS_SQL_PASSWORD", password)

    with pytest.raises(ValidationError, match=message):
        load_settings(env_file=None)


def test_bm25_part_weight_setting(clean_env: pytest.MonkeyPatch) -> None:
    assert load_settings(env_file=None).bm25.part_weight == 0.75
    clean_env.setenv("BM25_PART_WEIGHT", "1")
    assert load_settings(env_file=None).bm25.part_weight == 1.0
    clean_env.setenv("BM25_PART_WEIGHT", "1.5")
    with pytest.raises(ValidationError):
        load_settings(env_file=None)


def test_enabled_llm_requires_model(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("LLM_PROVIDER", "some-provider")

    with pytest.raises(ValidationError, match="LLM_MODEL"):
        load_settings(env_file=None)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("OPSRAG_ENVIRONMENT", "prod-ish"),
        ("OPSRAG_LOG_FORMAT", "xml"),
        ("POSTGRES_PORT", "70000"),
        ("LLM_TEMPERATURE", "5"),
        ("EMBEDDING_DIMENSION", "0"),
        ("RETRIEVAL_MODE", "keyword"),
        ("RETRIEVAL_FUSION", "max"),
        ("RETRIEVAL_DENSE_WEIGHT", "-1"),
        ("RETRIEVAL_RERANK_CANDIDATES", "500"),
        ("BM25_B", "1.5"),
        ("RERANKER_MAX_LENGTH", "4"),
        ("TOOLS_SQL_MAX_ROWS", "0"),
        ("TOOLS_SQL_TIMEOUT_SECONDS", "0"),
        ("TOOLS_LOGS_MAX_WINDOW_DAYS", "1000"),
        ("AGENT_MAX_TOOL_CALLS", "0"),
        ("AGENT_MIN_QUERY_COVERAGE", "1.5"),
        ("VERIFY_UNSUPPORTED_POLICY", "ignore"),
        ("VERIFY_SUPPORTED_THRESHOLD", "2"),
    ],
)
def test_invalid_values_are_rejected(clean_env: pytest.MonkeyPatch, name: str, value: str) -> None:
    clean_env.setenv(name, value)

    with pytest.raises(ValidationError):
        load_settings(env_file=None)


def test_retrieval_weights_and_reranking_are_configurable(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("RETRIEVAL_MODE", "hybrid")
    clean_env.setenv("RETRIEVAL_FUSION", "weighted")
    clean_env.setenv("RETRIEVAL_DENSE_WEIGHT", "0.3")
    clean_env.setenv("RETRIEVAL_SPARSE_WEIGHT", "0.7")
    clean_env.setenv("BM25_STEMMING", "false")
    clean_env.setenv("RERANKER_PROVIDER", " None ")

    settings = load_settings(env_file=None)

    assert settings.retrieval.fusion == "weighted"
    assert (settings.retrieval.dense_weight, settings.retrieval.sparse_weight) == (0.3, 0.7)
    assert settings.bm25.stemming is False
    assert not settings.reranker.enabled


def test_both_fusion_weights_cannot_be_zero(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("RETRIEVAL_DENSE_WEIGHT", "0")
    clean_env.setenv("RETRIEVAL_SPARSE_WEIGHT", "0")

    with pytest.raises(ValidationError, match="cannot both be 0"):
        load_settings(env_file=None)


def test_reranking_mode_needs_a_reranker(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("RERANKER_PROVIDER", "none")

    with pytest.raises(ValidationError, match="needs a reranker"):
        load_settings(env_file=None)  # default mode is hybrid_rerank


def test_tool_limits_are_configurable(clean_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
    clean_env.setenv("TOOLS_SQL_MAX_ROWS", "25")
    clean_env.setenv("TOOLS_SQL_TIMEOUT_SECONDS", "1.5")

    settings = load_settings(env_file=None)

    assert settings.tools.sql_max_rows == 25 and settings.tools.sql_timeout_seconds == 1.5
    assert load_settings(env_file=None).tools.logs_max_window_days == 31


def test_security_settings(clean_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
    defaults = load_settings(env_file=None).security
    assert defaults.policy_file is None
    assert (defaults.injection_action, defaults.query_injection_action) == ("quarantine", "refuse")
    clean_env.setenv("SECURITY_POLICY_FILE", str(tmp_path / "policy.json"))
    clean_env.setenv("SECURITY_INJECTION_ACTION", "redact")
    clean_env.setenv("SECURITY_QUERY_INJECTION_ACTION", "flag")
    settings = load_settings(env_file=None).security
    assert settings.policy_file == tmp_path / "policy.json"
    assert (settings.injection_action, settings.query_injection_action) == ("redact", "flag")
    clean_env.setenv("SECURITY_INJECTION_ACTION", "ignore")
    with pytest.raises(ValidationError):
        load_settings(env_file=None)


def test_agent_settings(clean_env: pytest.MonkeyPatch) -> None:
    assert load_settings(env_file=None).agent.max_tool_calls == 6
    clean_env.setenv("AGENT_MAX_TOOL_CALLS", "3")
    clean_env.setenv("AGENT_RERANK_EVIDENCE", "false")
    settings = load_settings(env_file=None)
    assert settings.agent.max_tool_calls == 3 and settings.agent.rerank_evidence is False


def test_verification_settings(clean_env: pytest.MonkeyPatch) -> None:
    defaults = load_settings(env_file=None).verification
    assert defaults.unsupported_policy == "remove" and defaults.partial_policy == "label"
    assert defaults.nli_model == "cross-encoder/nli-deberta-v3-xsmall"
    clean_env.setenv("VERIFY_NLI_MODEL", "none")
    clean_env.setenv("VERIFY_UNSUPPORTED_POLICY", "hedge")
    settings = load_settings(env_file=None).verification
    assert settings.nli_model is None and settings.unsupported_policy == "hedge"
    clean_env.setenv("VERIFY_PARTIAL_THRESHOLD", "0.9")
    clean_env.setenv("VERIFY_SUPPORTED_THRESHOLD", "0.8")
    with pytest.raises(ValidationError, match="PARTIAL_THRESHOLD"):
        load_settings(env_file=None)
