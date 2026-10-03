"""Application configuration.

Configuration is split into independent groups, each read from environment
variables (and an optional ``.env`` file) with its own prefix:

    OPSRAG_*     application / runtime settings (e.g. OPSRAG_ENVIRONMENT)
    POSTGRES_*   database connection (shared with docker-compose's Postgres)
    LLM_*        LLM provider selection
    EMBEDDING_*  embedding provider selection
    CHUNKING_*   ingestion chunking strategy and sizes
    RETRIEVAL_*  search defaults

Business logic must depend on the ``Settings`` object, never on ``os.environ``.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import URL

from app.schemas.enums import ChunkingStrategy, FusionMethod, RetrievalMode

Environment = Literal["local", "test", "staging", "production"]
LogFormat = Literal["json", "console"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

DEFAULT_ENV_FILE = ".env"
# No credential has a default in code: they come from the environment or .env.
# In production the database password must be set, long enough, and not the value
# the local-development setup used (compared by fingerprint, not stored here).
MIN_PRODUCTION_PASSWORD_LENGTH = 16
_KNOWN_DEV_PASSWORD_SHA256 = frozenset(
    {"e755a944b670e0e26789cf7fcecdce950b23044b128830b7497eb1972e57523d"}
)


def _group_config(prefix: str) -> SettingsConfigDict:
    return SettingsConfigDict(
        env_prefix=prefix,
        env_file=DEFAULT_ENV_FILE,
        env_file_encoding="utf-8",
        env_ignore_empty=True,  # `LLM_MODEL=` in .env means "unset", not ""
        extra="ignore",  # the .env file is shared by all groups
        case_sensitive=False,
    )


class AppSettings(BaseSettings):
    """Runtime settings for the API process."""

    model_config = _group_config("OPSRAG_")

    service_name: str = "opsrag-api"
    environment: Environment = "local"
    log_level: LogLevel = "INFO"
    log_format: LogFormat = "json"
    # Phase 10 evaluation runs (``LATEST`` names the one GET /api/evaluation serves).
    evaluation_dir: Path = Path("data/evaluation/results/phase10")
    # Build the agent (load its models) in the background at startup, so the first
    # question does not pay for it; GET /api/ready reports 503 until it is built.
    preload_agent: bool = False

    @field_validator("log_level", mode="before")
    @classmethod
    def _upper_log_level(cls, value: object) -> object:
        return value.upper() if isinstance(value, str) else value


class DatabaseSettings(BaseSettings):
    """PostgreSQL connection and pool settings."""

    model_config = _group_config("POSTGRES_")

    # 127.0.0.1 rather than "localhost": the latter resolves to ::1 and 127.0.0.1,
    # and libpq waits out connect_timeout on each, doubling time-to-fail.
    host: str = "127.0.0.1"
    port: int = Field(default=5432, ge=1, le=65535)
    user: str = "opsrag"
    password: SecretStr | None = None  # from POSTGRES_PASSWORD / .env only
    db: str = "opsrag"
    driver: str = "postgresql+psycopg"

    pool_size: int = Field(default=5, ge=1)
    max_overflow: int = Field(default=5, ge=0)
    pool_timeout_seconds: float = Field(default=10.0, gt=0)
    pool_recycle_seconds: int = Field(default=1800, ge=-1)
    # Bounds how long /api/health blocks when the DB is down. libpq minimum is 2.
    connect_timeout_seconds: int = Field(default=2, ge=2)
    echo_sql: bool = False

    @property
    def url(self) -> URL:
        """SQLAlchemy URL. Built structurally so special characters in the
        password are escaped correctly."""
        return URL.create(
            drivername=self.driver,
            username=self.user,
            password=self.password.get_secret_value() if self.password else None,
            host=self.host,
            port=self.port,
            database=self.db,
        )

    @property
    def safe_url(self) -> str:
        """URL with the password masked, safe for logs."""
        return self.url.render_as_string(hide_password=True)


class LLMSettings(BaseSettings):
    """LLM provider selection. Provider clients are resolved by name in a later
    phase; nothing outside the provider layer should branch on these values."""

    model_config = _group_config("LLM_")

    provider: str = "none"  # "none" = no LLM configured
    model: str | None = None
    api_key: SecretStr | None = None
    base_url: str | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=1024, ge=1)
    timeout_seconds: float = Field(default=60.0, gt=0)

    @field_validator("provider", mode="before")
    @classmethod
    def _normalize_provider(cls, value: object) -> object:
        return value.strip().lower() if isinstance(value, str) else value

    @model_validator(mode="after")
    def _model_required_when_enabled(self) -> LLMSettings:
        if self.enabled and not self.model:
            raise ValueError(f"LLM_MODEL must be set when LLM_PROVIDER={self.provider!r}")
        return self

    @property
    def enabled(self) -> bool:
        return self.provider != "none"


class EmbeddingSettings(BaseSettings):
    """Embedding provider selection. Nothing outside the provider layer depends on
    which model this is; the defaults below are only the out-of-the-box choice."""

    model_config = _group_config("EMBEDDING_")

    provider: str = "sentence-transformers"
    model: str = "BAAI/bge-small-en-v1.5"
    # Expected output size; checked against the loaded model (fail fast on a mismatch).
    dimension: int = Field(default=384, ge=1)
    device: str = "cpu"
    # Texts per encode call. Small batches were measured faster on CPU (Phase 12: batch 8
    # embedded 17-18% more chunks/s than 32, in two interleaved runs); on a GPU, raise it.
    batch_size: int = Field(default=8, ge=1)
    normalize: bool = True  # unit-length vectors, so cosine similarity = dot product
    # Some models expect instructions/prefixes (bge: query instruction; e5: "query: "/"passage: ").
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    document_prefix: str = ""
    max_seq_length: int | None = Field(default=None, ge=16)  # None = the model's own limit
    api_key: SecretStr | None = None
    base_url: str | None = None

    @field_validator("provider", mode="before")
    @classmethod
    def _normalize_provider(cls, value: object) -> object:
        return value.strip().lower() if isinstance(value, str) else value


class ChunkingSettings(BaseSettings):
    """How sources are split into retrievable chunks (sizes in approximate tokens)."""

    model_config = _group_config("CHUNKING_")

    strategy: ChunkingStrategy = ChunkingStrategy.DOCUMENT_AWARE
    chunk_size_tokens: int = Field(default=350, ge=32, le=2048)
    chunk_overlap_tokens: int = Field(default=50, ge=0)

    @model_validator(mode="after")
    def _overlap_smaller_than_size(self) -> ChunkingSettings:
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError(
                "CHUNKING_CHUNK_OVERLAP_TOKENS must be smaller than CHUNKING_CHUNK_SIZE_TOKENS"
            )
        return self


class RetrievalSettings(BaseSettings):
    """Search defaults and how dense and sparse results are combined."""

    model_config = _group_config("RETRIEVAL_")

    mode: RetrievalMode = RetrievalMode.HYBRID_RERANK
    top_k: int = Field(default=10, ge=1, le=100)
    # pgvector HNSW candidate list size; raised automatically to at least 4 x top_k.
    hnsw_ef_search: int = Field(default=100, ge=10, le=1000)
    fusion: FusionMethod = FusionMethod.RRF
    dense_weight: float = Field(default=1.0, ge=0.0)
    sparse_weight: float = Field(default=1.0, ge=0.0)
    rrf_k: int = Field(default=60, ge=1)  # RRF damping constant: score = weight / (rrf_k + rank)
    fusion_depth: int = Field(default=50, ge=1, le=100)  # results taken from each retriever
    rerank_candidates: int = Field(default=30, ge=1, le=100)  # fused results the reranker sees

    @model_validator(mode="after")
    def _some_weight(self) -> RetrievalSettings:
        if self.dense_weight == 0 and self.sparse_weight == 0:
            raise ValueError("RETRIEVAL_DENSE_WEIGHT and RETRIEVAL_SPARSE_WEIGHT cannot both be 0")
        return self


class BM25Settings(BaseSettings):
    """Okapi BM25 parameters and text analysis. Defaults are the textbook values."""

    model_config = _group_config("BM25_")

    k1: float = Field(default=1.2, ge=0.0, le=10.0)  # term-frequency saturation
    b: float = Field(default=0.75, ge=0.0, le=1.0)  # document-length normalisation
    stemming: bool = True  # Snowball (Porter2) stemming of plain words
    stopwords: bool = True  # drop common English function words
    # In a pure identifier lookup ("MailRelayClient"), the query weight of the
    # identifier's parts and segments (mail, relay, client) relative to the whole one.
    # Questions with plain words always weight every term 1. 1 = no difference.
    part_weight: float = Field(default=0.75, ge=0.0, le=1.0)


class RerankerSettings(BaseSettings):
    """Second-stage reranker. ``provider=none`` disables reranking."""

    model_config = _group_config("RERANKER_")

    provider: str = "cross-encoder"
    model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    device: str = "cpu"
    # Pairs per forward pass. On CPU, 4 was measured faster than 16 (15%) and 8 (5%) with
    # identical scores (Phase 12, interleaved); on a GPU, raise it.
    batch_size: int = Field(default=4, ge=1)
    max_length: int = Field(default=512, ge=16)  # tokens of (query, passage) per pair

    @field_validator("provider", mode="before")
    @classmethod
    def _normalize_provider(cls, value: object) -> object:
        return value.strip().lower() if isinstance(value, str) else value

    @property
    def enabled(self) -> bool:
        return self.provider != "none"


class ToolSettings(BaseSettings):
    """Limits of the agent tool layer (app/tools)."""

    model_config = _group_config("TOOLS_")

    max_top_k: int = Field(default=20, ge=1, le=100)  # most results any search tool returns
    snippet_chars: int = Field(default=1500, ge=100, le=20000)  # content per result
    runbook_max_chars: int = Field(default=20000, ge=1000, le=200000)
    sql_max_rows: int = Field(default=200, ge=1, le=5000)
    sql_timeout_seconds: float = Field(default=5.0, gt=0, le=120)
    logs_max_results: int = Field(default=200, ge=1, le=5000)
    logs_max_window_days: int = Field(default=31, ge=1, le=366)
    # query_database connects as this read-only role (PostgreSQL) when set: it may only
    # SELECT the tables the tool exposes. Create it with scripts/create_sql_reader.py.
    # Required in production. Empty: the tool uses the application's connection.
    sql_user: str | None = Field(default=None, pattern=r"^[a-z_][a-z0-9_]{0,62}$")
    sql_password: SecretStr | None = None


class AgentSettings(BaseSettings):
    """The agent loop (app/agents): tool budget, evidence limits, answerability."""

    model_config = _group_config("AGENT_")

    max_tool_calls: int = Field(default=6, ge=1, le=20)
    results_per_tool: int = Field(default=5, ge=1, le=20)
    max_evidence: int = Field(default=8, ge=1, le=30)  # evidence items given to synthesis
    # Share of the question's (IDF-weighted) key terms the evidence must contain.
    min_query_coverage: float = Field(default=0.5, ge=0.0, le=1.0)
    log_window_padding_minutes: int = Field(default=30, ge=0, le=720)
    default_log_window_hours: int = Field(default=24, ge=1, le=744)
    time_budget_seconds: float = Field(default=60.0, gt=0, le=600)
    rerank_evidence: bool = True  # cross-encoder over aggregated evidence (if configured)


class VerificationSettings(BaseSettings):
    """Claim verification and confidence (app/agents/verification.py, confidence.py)."""

    model_config = _group_config("VERIFY_")

    # 3-way NLI cross-encoder for semantic support; "none" = lexical checks only.
    nli_model: str | None = "cross-encoder/nli-deberta-v3-xsmall"
    device: str = "cpu"
    # Share of a claim's content terms an item must contain (its values must all match).
    supported_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    partial_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    entailment_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    partial_entailment_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    contradiction_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    # Term overlap at which a sentence with different values counts as conflicting.
    conflict_overlap: float = Field(default=0.6, ge=0.0, le=1.0)
    unsupported_policy: Literal["remove", "hedge", "label"] = "remove"
    partial_policy: Literal["label", "hedge", "keep"] = "label"
    high_confidence: float = Field(default=0.75, ge=0.0, le=1.0)
    medium_confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("nli_model", mode="before")
    @classmethod
    def _none_disables(cls, value: object) -> object:
        if isinstance(value, str) and value.strip().lower() in {"", "none"}:
            return None
        return value

    @model_validator(mode="after")
    def _ordered(self) -> VerificationSettings:
        if self.partial_threshold > self.supported_threshold:
            raise ValueError("VERIFY_PARTIAL_THRESHOLD must not exceed VERIFY_SUPPORTED_THRESHOLD")
        if self.medium_confidence > self.high_confidence:
            raise ValueError("VERIFY_MEDIUM_CONFIDENCE must not exceed VERIFY_HIGH_CONFIDENCE")
        return self


class SecuritySettings(BaseSettings):
    """Access policy and prompt-injection handling (app/security, app/agents/guard.py)."""

    model_config = _group_config("SECURITY_")

    # Role -> data grants (JSON). None: the packaged app/security/policy.json.
    policy_file: Path | None = None
    # Retrieved content containing instructions aimed at the assistant:
    # quarantine = drop the whole source; redact = drop the flagged sentences and keep
    # the rest, marked suspicious.
    injection_action: Literal["quarantine", "redact"] = "quarantine"
    # A question containing instructions aimed at the assistant:
    # refuse = answer nothing, run no tool; flag = answer, with a limitation noted.
    query_injection_action: Literal["refuse", "flag"] = "refuse"
    # Development convenience: accept the X-OpsRAG-User header as identity without a
    # token. Only allowed when OPSRAG_ENVIRONMENT is local or test; off by default.
    allow_user_header: bool = False
    # Second injection detector for retrieved content: a classifier run on sentences
    # addressed to an AI. none = rules only. The rules always run.
    injection_model: str | None = "protectai/deberta-v3-base-prompt-injection-v2"
    injection_threshold: float = Field(default=0.9, gt=0.0, le=1.0)
    injection_device: str = "cpu"

    @field_validator("injection_model", mode="before")
    @classmethod
    def _model_none_disables(cls, value: object) -> object:
        if isinstance(value, str) and value.strip().lower() in {"", "none"}:
            return None
        return value


class Settings(BaseModel):
    """Aggregate of all configuration groups."""

    model_config = ConfigDict(frozen=True)

    app: AppSettings
    database: DatabaseSettings
    llm: LLMSettings
    embedding: EmbeddingSettings
    chunking: ChunkingSettings
    retrieval: RetrievalSettings
    bm25: BM25Settings
    reranker: RerankerSettings
    tools: ToolSettings
    agent: AgentSettings
    verification: VerificationSettings
    security: SecuritySettings

    @model_validator(mode="after")
    def _reranker_available(self) -> Settings:
        if self.retrieval.mode == RetrievalMode.HYBRID_RERANK and not self.reranker.enabled:
            raise ValueError(
                "RETRIEVAL_MODE=hybrid_rerank needs a reranker; set RERANKER_PROVIDER "
                "or choose another RETRIEVAL_MODE"
            )
        return self

    @model_validator(mode="after")
    def _identity_header_only_locally(self) -> Settings:
        if self.security.allow_user_header and self.app.environment not in {"local", "test"}:
            raise ValueError(
                "SECURITY_ALLOW_USER_HEADER trusts an unauthenticated header; it is only "
                "allowed when OPSRAG_ENVIRONMENT is local or test"
            )
        return self

    @model_validator(mode="after")
    def _production_guards(self) -> Settings:
        if self.app.environment != "production":
            return self
        password = self.database.password.get_secret_value() if self.database.password else ""
        if len(password) < MIN_PRODUCTION_PASSWORD_LENGTH:
            raise ValueError(
                f"POSTGRES_PASSWORD must be set to at least {MIN_PRODUCTION_PASSWORD_LENGTH} "
                "characters in production"
            )
        if hashlib.sha256(password.encode()).hexdigest() in _KNOWN_DEV_PASSWORD_SHA256:
            raise ValueError("POSTGRES_PASSWORD must not be the local-development password")
        reader = self.tools.sql_password.get_secret_value() if self.tools.sql_password else ""
        if not self.tools.sql_user or len(reader) < MIN_PRODUCTION_PASSWORD_LENGTH:
            raise ValueError(
                "TOOLS_SQL_USER and TOOLS_SQL_PASSWORD (at least "
                f"{MIN_PRODUCTION_PASSWORD_LENGTH} characters) are required in production: "
                "query_database must run as a dedicated read-only database role"
            )
        if self.tools.sql_user == self.database.user:
            raise ValueError("TOOLS_SQL_USER must differ from POSTGRES_USER")
        return self


def load_settings(env_file: str | Path | None = DEFAULT_ENV_FILE) -> Settings:
    """Build settings from the process environment and ``env_file``.

    Pass ``env_file=None`` to ignore any ``.env`` file (used by tests).
    Environment variables always take precedence over the file.
    """
    return Settings(
        app=AppSettings(_env_file=env_file),  # type: ignore[call-arg]
        database=DatabaseSettings(_env_file=env_file),  # type: ignore[call-arg]
        llm=LLMSettings(_env_file=env_file),  # type: ignore[call-arg]
        embedding=EmbeddingSettings(_env_file=env_file),  # type: ignore[call-arg]
        chunking=ChunkingSettings(_env_file=env_file),  # type: ignore[call-arg]
        retrieval=RetrievalSettings(_env_file=env_file),  # type: ignore[call-arg]
        bm25=BM25Settings(_env_file=env_file),  # type: ignore[call-arg]
        reranker=RerankerSettings(_env_file=env_file),  # type: ignore[call-arg]
        tools=ToolSettings(_env_file=env_file),  # type: ignore[call-arg]
        agent=AgentSettings(_env_file=env_file),  # type: ignore[call-arg]
        verification=VerificationSettings(_env_file=env_file),  # type: ignore[call-arg]
        security=SecuritySettings(_env_file=env_file),  # type: ignore[call-arg]
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings, loaded once."""
    return load_settings()
