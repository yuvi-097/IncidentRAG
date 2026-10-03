"""Domain vocabulary shared by the database models, the synthetic data generator
and (later) the API. Values are stored as-is in the database."""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum


class AccessLevel(StrEnum):
    """Sensitivity labels on every stored record, enforced before content reaches an LLM.

    They are compartments, not a ladder: the access policy (``app/security/policy.json``)
    grants each role a *set* of labels per kind of data, so a manager can read
    MANAGER-labelled reports without reading SRE-labelled runbooks. CONFIDENTIAL can
    never be granted: confidential sources are not indexed and never retrieved.
    """

    PUBLIC = "public"  # anyone, including customers (published API reference)
    ENGINEERING = "engineering"  # internal engineering material, not sensitive
    SRE = "sre"  # operationally sensitive: auth and payment internals, security incidents
    MANAGER = "manager"  # management reports
    ADMIN = "admin"  # administrative material: production access, break-glass
    CONFIDENTIAL = "confidential"  # never retrieved by the assistant (secrets)


# Records without a label of their own (deployments, logs, the service catalog) are
# engineering material; a role needs a grant for this label on their kind of data.
UNLABELLED_LEVEL = AccessLevel.ENGINEERING

# Partial order used only to label *composite* content (a pull request changing files
# with different labels): the result must be at least as restrictive as every part.
# SRE and MANAGER are incomparable; their combination needs ADMIN.
_BELOW: dict[AccessLevel, frozenset[AccessLevel]] = {
    AccessLevel.PUBLIC: frozenset(),
    AccessLevel.ENGINEERING: frozenset({AccessLevel.PUBLIC}),
    AccessLevel.SRE: frozenset({AccessLevel.PUBLIC, AccessLevel.ENGINEERING}),
    AccessLevel.MANAGER: frozenset({AccessLevel.PUBLIC, AccessLevel.ENGINEERING}),
    AccessLevel.ADMIN: frozenset(
        {AccessLevel.PUBLIC, AccessLevel.ENGINEERING, AccessLevel.SRE, AccessLevel.MANAGER}
    ),
    AccessLevel.CONFIDENTIAL: frozenset(set(AccessLevel) - {AccessLevel.CONFIDENTIAL}),
}


def most_restrictive(levels: Iterable[AccessLevel]) -> AccessLevel:
    """The least label at least as restrictive as all of ``levels`` (their join)."""
    wanted = set(levels) or {AccessLevel.PUBLIC}
    for candidate in AccessLevel:  # declared from least to most restrictive
        if all(level == candidate or level in _BELOW[candidate] for level in wanted):
            return candidate
    return AccessLevel.CONFIDENTIAL


class Resource(StrEnum):
    """Kinds of data an access policy grants labels for."""

    DOCUMENTS = "documents"  # technical documentation, operational reports, policies
    RUNBOOKS = "runbooks"
    INCIDENTS = "incidents"  # incident records and their postmortems
    DEPLOYMENTS = "deployments"  # deployments and pull-request metadata (id, title)
    CODE = "code"  # source files, pull-request descriptions and diffs
    LOGS = "logs"
    CATALOG = "catalog"  # services and their dependencies


class SourceTrust(StrEnum):
    """How far retrieved content can be trusted. All of it is data, never instructions."""

    SYSTEM_RECORD = "system_record"  # structured fields written by operational systems
    CURATED = "curated"  # reviewed internal documentation (docs, runbooks, postmortems)
    USER_CONTENT = "user_content"  # free text anyone can write (code, PRs, log messages)
    SUSPICIOUS = "suspicious"  # instruction-like text was found and removed


class ServiceTier(StrEnum):
    TIER_0 = "tier_0"  # revenue-critical path (checkout, payments, auth)
    TIER_1 = "tier_1"
    TIER_2 = "tier_2"


class DependencyProtocol(StrEnum):
    HTTP = "http"
    KAFKA = "kafka"


class DependencyCriticality(StrEnum):
    HARD = "hard"  # caller fails without it
    SOFT = "soft"  # caller degrades gracefully


class DeploymentStatus(StrEnum):
    SUCCEEDED = "succeeded"
    ROLLED_BACK = "rolled_back"  # went live, later reverted by a rollback deployment
    FAILED = "failed"  # aborted during rollout, never served traffic


class DeploymentStrategy(StrEnum):
    ROLLING = "rolling"
    CANARY = "canary"
    BLUE_GREEN = "blue_green"


class PullRequestState(StrEnum):
    OPEN = "open"
    MERGED = "merged"
    CLOSED = "closed"


class FileChangeType(StrEnum):
    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"


class CodeFileKind(StrEnum):
    SOURCE = "source"
    TEST = "test"
    CONFIG = "config"
    MIGRATION = "migration"
    DOCS = "docs"
    BUILD = "build"


class Severity(StrEnum):
    SEV1 = "SEV1"  # critical customer impact
    SEV2 = "SEV2"
    SEV3 = "SEV3"
    SEV4 = "SEV4"  # minor


class IncidentStatus(StrEnum):
    INVESTIGATING = "investigating"
    MITIGATED = "mitigated"
    RESOLVED = "resolved"


class IncidentCategory(StrEnum):
    DB_CONNECTION_EXHAUSTION = "db_connection_exhaustion"
    REDIS_MEMORY_PRESSURE = "redis_memory_pressure"
    KAFKA_CONSUMER_LAG = "kafka_consumer_lag"
    API_TIMEOUT = "api_timeout"
    AUTHENTICATION_FAILURE = "authentication_failure"
    DEPLOYMENT_REGRESSION = "deployment_regression"
    CONFIGURATION_ERROR = "configuration_error"
    DATABASE_DEADLOCK = "database_deadlock"
    MEMORY_LEAK = "memory_leak"
    CPU_SATURATION = "cpu_saturation"
    DEPENDENCY_FAILURE = "dependency_failure"
    NETWORK_TIMEOUT = "network_timeout"
    RATE_LIMITING = "rate_limiting"
    CACHE_INVALIDATION = "cache_invalidation"
    SCHEMA_MISMATCH = "schema_mismatch"


class DocumentType(StrEnum):
    RUNBOOK = "runbook"
    ARCHITECTURE = "architecture"
    API_REFERENCE = "api_reference"
    SERVICE_BEHAVIOR = "service_behavior"
    CONFIGURATION = "configuration"
    DEPLOYMENT = "deployment"
    DATABASE = "database"
    TROUBLESHOOTING = "troubleshooting"
    MONITORING = "monitoring"
    POSTMORTEM = "postmortem"
    REPORT = "report"  # operational reports for management (reliability reviews)
    POLICY = "policy"  # administrative policies (production access, break-glass)


class SourceType(StrEnum):
    """Where a retrievable chunk came from."""

    INCIDENT = "incident"
    RUNBOOK = "runbook"
    DOCUMENTATION = "documentation"  # technical docs (architecture, API, config, ...)
    POSTMORTEM = "postmortem"
    DEPLOYMENT = "deployment"
    CODE = "code"
    PULL_REQUEST = "pull_request"


class ChunkingStrategy(StrEnum):
    FIXED = "fixed"
    RECURSIVE = "recursive"
    DOCUMENT_AWARE = "document_aware"


class RetrievalMode(StrEnum):
    DENSE = "dense"
    SPARSE = "sparse"  # BM25
    HYBRID = "hybrid"  # dense + BM25, fused
    HYBRID_RERANK = "hybrid_rerank"  # hybrid candidates, reordered by a cross-encoder


class FusionMethod(StrEnum):
    RRF = "rrf"  # reciprocal rank fusion (ranks only)
    WEIGHTED = "weighted"  # min-max normalised scores, weighted sum


class QueryType(StrEnum):
    """What kind of question a user asked; decides which tools are appropriate."""

    DOCUMENT_SEARCH = "DOCUMENT_SEARCH"  # docs, runbooks, architecture, configuration
    INCIDENT_SEARCH = "INCIDENT_SEARCH"  # past incidents and postmortems
    CODE_SEARCH = "CODE_SEARCH"  # source code and where things are implemented
    SQL_QUERY = "SQL_QUERY"  # counts, aggregates, rankings over structured records
    DEPLOYMENT_SEARCH = "DEPLOYMENT_SEARCH"  # releases, versions, rollbacks, shipped PRs
    LOG_SEARCH = "LOG_SEARCH"  # application log lines
    MULTI_SOURCE = "MULTI_SOURCE"  # needs several sources (e.g. what changed before an outage)
    UNKNOWN = "UNKNOWN"  # outside the operations domain, or unintelligible


class ToolPermission(StrEnum):
    """Capabilities a role can hold; every tool requires exactly one."""

    DOCUMENTS_READ = "documents:read"
    RUNBOOKS_READ = "runbooks:read"
    INCIDENTS_READ = "incidents:read"
    DEPLOYMENTS_READ = "deployments:read"
    CODE_READ = "code:read"
    LOGS_READ = "logs:read"
    SQL_READ = "sql:read"


class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
