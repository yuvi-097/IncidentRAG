"""What the agent returns to users: the API boundary.

Only the answer, its citations and evidence, a one-line summary per stage, tool
calls, limitations, errors and timings. Prompts, raw tool outputs, internal scores
and model reasoning are never included.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.agents.conflicts import ConflictValue
from app.agents.recommendations import Recommendation
from app.agents.state import (
    AgentState,
    AnswerConfidence,
    Citation,
    ClaimVerdict,
    ConfidenceBreakdown,
    EvidenceKind,
    EvidenceStatus,
    SecurityReport,
)
from app.schemas.enums import AccessLevel, QueryType, SourceTrust

SNIPPET = 320


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=2, max_length=1000)


class EvidenceView(BaseModel):
    label: str
    kind: EvidenceKind
    source_id: str
    title: str
    location: str | None
    service_id: str | None
    timestamp: datetime | None
    snippet: str
    content: str  # the full text the answer was written from (screened, redacted)
    relevance: float
    cited: bool
    trust: SourceTrust
    access_level: AccessLevel  # the item's label (the caller's grants include it)


class ToolCallView(BaseModel):
    tool: str
    purpose: str
    status: str
    results: int
    duration_ms: float
    error: str | None = None


class ConflictView(BaseModel):
    """Sources that state different values for one setting; every side is listed."""

    setting: str
    values: list[ConflictValue]  # newest first, with their sources and evidence labels
    preferred: str | None  # the newest value still in effect, if there is one
    summary: str


class ErrorView(BaseModel):
    stage: str
    code: str
    message: str
    tool: str | None = None


class AgentResponse(BaseModel):
    question: str
    answer: str
    query_type: QueryType | None
    confidence: AnswerConfidence
    confidence_reasons: list[str]
    confidence_breakdown: ConfidenceBreakdown | None
    evidence_status: EvidenceStatus | None
    claims: list[ClaimVerdict]  # every factual claim, its label and what was done with it
    citations: list[Citation]
    suggested_evidence: list[str]  # when the evidence is insufficient
    recommendations: list[Recommendation]  # next steps, each from cited evidence
    evidence: list[EvidenceView]
    conflicts: list[ConflictView]  # disagreeing sources relevant to the question
    plan: str  # temporal, chain or routed
    tools: list[ToolCallView]
    reasoning_summary: list[str]  # one line per stage: what was done, not how it was thought
    limitations: list[str]
    errors: list[ErrorView]
    security: SecurityReport  # what screening did; never the flagged text itself
    synthesis: str
    latency_ms: dict[str, float]

    @classmethod
    def from_state(cls, state: AgentState) -> AgentResponse:
        cited = {c.label for c in state.citations}
        return cls(
            question=state.query,
            answer=state.final_answer,
            query_type=state.query_type,
            confidence=state.confidence,
            confidence_reasons=state.confidence_reasons,
            confidence_breakdown=state.confidence_breakdown,
            evidence_status=state.evidence_status,
            claims=state.claims,
            citations=state.citations,
            suggested_evidence=state.suggested_evidence,
            recommendations=state.recommendations,
            evidence=[
                EvidenceView(
                    label=item.label or "",
                    kind=item.kind,
                    source_id=item.source_id,
                    title=item.title,
                    location=item.location,
                    service_id=item.service_id,
                    timestamp=item.timestamp,
                    snippet=" ".join(item.text.split())[:SNIPPET],
                    content=item.text,
                    relevance=item.relevance,
                    cited=item.label in cited,
                    trust=item.trust,
                    access_level=item.access_level,
                )
                for item in state.reranked_evidence
            ],
            conflicts=[
                ConflictView(
                    setting=c.setting, values=c.values, preferred=c.preferred, summary=c.describe()
                )
                for c in state.conflicts
            ],
            plan=state.plan,
            tools=[
                ToolCallView(
                    tool=r.tool,
                    purpose=r.purpose,
                    status=r.status,
                    results=r.results,
                    duration_ms=r.duration_ms,
                    error=r.error,
                )
                for r in state.tool_results
            ],
            reasoning_summary=[f"{s.stage.value}: {s.summary}" for s in state.steps],
            limitations=state.limitations,
            errors=[
                ErrorView(stage=e.stage.value, code=e.code, message=e.message, tool=e.tool)
                for e in state.errors
            ],
            security=state.security,
            synthesis=state.synthesis_method,
            latency_ms=state.latency_ms,
        )
