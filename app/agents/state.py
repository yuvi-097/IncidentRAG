"""Agent state: everything one run knows, from the question to the final answer.

``AgentState`` is internal and mutable; it is filled in stage by stage. The API
never returns it. ``app/agents/response.py`` projects the parts meant for users:
answer, citations, evidence, a short summary per stage, limitations, errors and
timings. Raw tool outputs and prompts stay inside.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from app.agents.entities import QueryEntities
from app.agents.router import RoutingDecision
from app.schemas.enums import AccessLevel, QueryType, SourceTrust
from app.security.principal import Principal
from app.tools.base import ToolModel

if TYPE_CHECKING:
    from app.agents.conflicts import Conflict
    from app.agents.multihop import ChainQuery
    from app.agents.recommendations import Recommendation
    from app.agents.temporal import Anchor, TemporalQuery


class Stage(StrEnum):
    UNDERSTAND = "query_understanding"
    ROUTE = "routing"
    SELECT_TOOLS = "tool_selection"
    EXECUTE = "tool_execution"
    AGGREGATE = "evidence_aggregation"
    SCREEN = "security_screening"
    RERANK = "reranking"
    VALIDATE_EVIDENCE = "evidence_validation"
    PACKAGE = "evidence_package"
    SYNTHESIZE = "synthesis"
    VERIFY = "claim_verification"
    GUARD_OUTPUT = "output_validation"
    CONFIDENCE = "confidence"
    RESPOND = "final_response"
    DONE = "done"


class EvidenceKind(StrEnum):
    DOCUMENT = "document"
    RUNBOOK = "runbook"
    POSTMORTEM = "postmortem"
    INCIDENT = "incident"
    DEPLOYMENT = "deployment"
    PULL_REQUEST = "pull_request"
    CODE = "code"
    LOGS = "logs"
    SQL_RESULT = "sql_result"
    TIMELINE = "timeline"  # derived: records ordered by their timestamps (temporal questions)


class EvidenceStatus(StrEnum):
    SUFFICIENT = "sufficient"
    PARTIAL = "partial"
    INSUFFICIENT = "insufficient"


class AnswerConfidence(StrEnum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"  # no answer given


class ClaimLabel(StrEnum):
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"


class EvidenceItem(BaseModel):
    """One piece of evidence, from any tool, in a common shape."""

    model_config = ConfigDict(frozen=True)

    kind: EvidenceKind
    source_id: str  # INC-0406, DEP-0296, CF-0164, RB-0049, "sql", "logs:payment-service"
    chunk_id: str | None = None
    title: str
    text: str  # what synthesis reads (already within the caller's clearance)
    service_id: str | None = None
    timestamp: datetime | None = None
    access_level: AccessLevel
    location: str | None = None  # file or source path
    section: str | None = None  # heading path or code symbol
    tool: str
    score: float = 0.0  # ranking score (retriever or reranker scale)
    relevance: float = 0.0  # normalised to [0, 1] during reranking
    pinned: bool = False  # asked for directly (by id, or a SQL result): never ranked away
    label: str | None = None  # "E1"... assigned after reranking, used in citations
    facts: dict[str, str] = Field(default_factory=dict)  # structured fields, for templates
    trust: SourceTrust = SourceTrust.USER_CONTENT  # set by security screening
    security_flags: tuple[str, ...] = ()  # what screening removed (injection categories)

    @property
    def key(self) -> tuple[str, str, str | None]:
        return (self.kind.value, self.source_id, self.chunk_id)


class PlannedCall(BaseModel):
    model_config = ConfigDict(frozen=True)

    tool: str
    arguments: dict[str, Any]
    purpose: str  # one line, shown in the reasoning summary

    @property
    def signature(self) -> str:
        return f"{self.tool}:{sorted(self.arguments.items())!r}"


class ToolCallRecord(BaseModel):
    tool: str
    purpose: str
    arguments: dict[str, Any]
    status: str
    duration_ms: float
    results: int = 0
    error: str | None = None


class AgentError(BaseModel):
    stage: Stage
    code: str
    message: str
    tool: str | None = None


class Citation(BaseModel):
    """A citation always points at an item of the evidence package; none are invented."""

    model_config = ConfigDict(frozen=True)

    label: str
    source_id: str
    title: str
    source_type: EvidenceKind
    timestamp: datetime | None = None
    section: str | None = None
    file_path: str | None = None
    relevance: float
    chunk_id: str | None = None
    trust: SourceTrust | None = None


class PackagedEvidence(BaseModel):
    """One entry of the evidence package the answer is generated from."""

    model_config = ConfigDict(frozen=True)

    label: str  # how the answer cites it: [E1]
    source_id: str
    source_type: EvidenceKind
    title: str
    content: str
    relevance_score: float  # [0, 1]
    timestamp: datetime | None = None
    section: str | None = None
    file_path: str | None = None
    chunk_id: str | None = None
    trust: SourceTrust = SourceTrust.USER_CONTENT
    security_flags: tuple[str, ...] = ()


class EvidencePackage(BaseModel):
    model_config = ConfigDict(frozen=True)

    query: str
    evidence: list[PackagedEvidence]

    def by_label(self) -> dict[str, PackagedEvidence]:
        return {e.label: e for e in self.evidence}


class ClaimVerdict(BaseModel):
    """One factual claim from the answer and what the evidence says about it."""

    text: str  # the claim as generated, without citation labels
    label: ClaimLabel
    cited: list[str]  # labels the answer gave it
    supporting: list[str]  # labels whose evidence supports it (the citations kept)
    conflicting: list[str] = Field(default_factory=list)  # evidence stating otherwise
    support_score: float  # [0, 1]
    action: str  # kept | removed | hedged | labelled
    reason: str = ""


class ConfidenceBreakdown(BaseModel):
    """Confidence is computed from these components; never taken from a model."""

    retrieval_quality: float
    source_agreement: float
    evidence_coverage: float
    temporal_consistency: float
    verification: float
    overall: float
    level: AnswerConfidence
    reasons: list[str] = Field(default_factory=list)


class ScreenedSource(BaseModel):
    """A retrieved source the security layer acted on (never its content)."""

    source_id: str
    kind: EvidenceKind
    categories: list[str]  # injection categories found


class SecurityReport(BaseModel):
    """What the security layer did for one question. Never contains flagged text."""

    query_flags: list[str] = Field(default_factory=list)  # injection categories in the question
    blocked: bool = False  # the question was refused before any tool ran
    quarantined: list[ScreenedSource] = Field(default_factory=list)  # removed entirely
    sanitized: list[ScreenedSource] = Field(default_factory=list)  # flagged text removed
    secrets_redacted: int = 0  # credentials removed from the question, evidence or answer
    references_redacted: int = 0  # ids/titles of records the caller may not read, removed
    access_violations: int = 0  # items tools returned that the caller may not read (bug if > 0)
    output_removed: list[str] = Field(default_factory=list)  # why answer sentences were removed


class StepSummary(BaseModel):
    """What a stage did, in one line. A summary, never a reasoning trace."""

    stage: Stage
    summary: str = Field(max_length=300)


@dataclass
class AgentState:
    query: str
    principal: Principal
    now: datetime
    entities: QueryEntities | None = None
    routing: RoutingDecision | None = None
    query_type: QueryType | None = None
    # Structured plans, when the question asks for one (Phase 9):
    temporal: TemporalQuery | None = None  # records selected by their position in time
    temporal_anchor: Anchor | None = None  # the record (or moment) they are relative to
    chain: ChainQuery | None = None  # incident -> deployment -> commit -> file -> change
    conflicts: list[Conflict] = field(default_factory=list)  # sources disagreeing
    recommendations: list[Recommendation] = field(default_factory=list)  # next steps
    pending: list[PlannedCall] = field(default_factory=list)
    selected_tools: list[str] = field(default_factory=list)
    tool_results: list[ToolCallRecord] = field(default_factory=list)
    outputs: list[tuple[PlannedCall, ToolModel]] = field(default_factory=list)
    goals: dict[str, bool] = field(default_factory=dict)
    retrieved_documents: list[EvidenceItem] = field(default_factory=list)
    reranked_evidence: list[EvidenceItem] = field(default_factory=list)
    evidence_status: EvidenceStatus | None = None
    evidence_notes: list[str] = field(default_factory=list)
    query_coverage: float | None = None  # share of the question's key terms in the evidence
    unknown_terms: list[str] = field(default_factory=list)  # question words absent from the corpus
    evidence_package: EvidencePackage | None = None
    claims: list[ClaimVerdict] = field(default_factory=list)
    confidence_breakdown: ConfidenceBreakdown | None = None
    suggested_evidence: list[str] = field(default_factory=list)
    final_answer: str = ""
    synthesis_method: str = ""
    citations: list[Citation] = field(default_factory=list)
    confidence: AnswerConfidence = AnswerConfidence.INSUFFICIENT_EVIDENCE
    confidence_reasons: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    latency_ms: dict[str, float] = field(default_factory=dict)
    errors: list[AgentError] = field(default_factory=list)
    steps: list[StepSummary] = field(default_factory=list)
    security: SecurityReport = field(default_factory=SecurityReport)
    # Set once the evidence passed security screening; no model is called before.
    evidence_screened: bool = False
    # Text of quarantined sources, kept only to detect it echoed in the answer.
    quarantined_text: list[str] = field(default_factory=list, repr=False)

    @property
    def user_id(self) -> str:
        return self.principal.user_id

    @property
    def role(self) -> str:
        return self.principal.role

    def step(self, stage: Stage, summary: str) -> None:
        self.steps.append(StepSummary(stage=stage, summary=summary[:300]))

    @property
    def plan(self) -> str:
        """Which plan answers the question: temporal, chain or the routed default."""
        return "temporal" if self.temporal else "chain" if self.chain else "routed"

    def limit(self, text: str) -> None:
        if text not in self.limitations:
            self.limitations.append(text)
