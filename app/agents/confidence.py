"""Answer confidence, computed from the evidence and the verification.

Five components, each in [0, 1]:

- retrieval quality: mean relevance of the evidence the answer cites;
- source agreement: independent sources behind the kept claims, and no conflicts;
- evidence coverage: the planner's evidence goals met, and the share of the
  question's key terms the evidence contains;
- temporal consistency: evidence inside the question's time range, and causes
  before effects (a root-cause deployment before its incident);
- verification: the share of claims supported (partially supported counts half).

The overall score is a weighted mean, mapped to HIGH / MEDIUM / LOW with the
thresholds in ``VERIFY_HIGH_CONFIDENCE`` and ``VERIFY_MEDIUM_CONFIDENCE``, and then
capped:
- conflicting sources, unsupported claims, partial evidence or a failed tool: at most MEDIUM;
- a temporal contradiction: at most LOW.
No supported claim, or insufficient evidence, gives INSUFFICIENT_EVIDENCE. A model
never sets or influences this value.
"""

from __future__ import annotations

from datetime import UTC, datetime

from app.agents.state import (
    AgentState,
    AnswerConfidence,
    ClaimLabel,
    ConfidenceBreakdown,
    EvidenceKind,
    EvidenceStatus,
)
from app.config import VerificationSettings

WEIGHTS = {
    "retrieval_quality": 0.2,
    "source_agreement": 0.2,
    "evidence_coverage": 0.2,
    "temporal_consistency": 0.15,
    "verification": 0.25,
}
_ORDER = [AnswerConfidence.LOW, AnswerConfidence.MEDIUM, AnswerConfidence.HIGH]


def _utc(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=UTC)


def _cap(level: AnswerConfidence, ceiling: AnswerConfidence) -> AnswerConfidence:
    return min(level, ceiling, key=_ORDER.index)


def temporal_checks(state: AgentState) -> tuple[int, int, list[str]]:
    """(checks passed, checks run, problems)."""
    passed = total = 0
    problems: list[str] = []
    items = state.reranked_evidence
    window = state.entities.time_range if state.entities else None
    if window:
        for item in items:
            if (
                item.kind in {EvidenceKind.INCIDENT, EvidenceKind.DEPLOYMENT, EvidenceKind.LOGS}
                and item.timestamp
            ):
                total += 1
                when = _utc(item.timestamp)
                if window.since <= when < window.until:
                    passed += 1
                else:
                    problems.append(f"{item.source_id} is outside {window.expression}")
    incidents = {i.source_id: i for i in items if i.kind is EvidenceKind.INCIDENT and i.timestamp}
    for deployment in (i for i in items if i.kind is EvidenceKind.DEPLOYMENT and i.timestamp):
        for part in deployment.facts.get("incidents", "").split(";"):
            incident_id, _, relation = part.strip().partition(" ")
            incident = incidents.get(incident_id)
            if incident is None or "root_cause" not in relation:
                continue
            total += 1
            if _utc(deployment.timestamp) <= _utc(incident.timestamp):  # type: ignore[arg-type]
                passed += 1
            else:
                problems.append(f"{deployment.source_id} was deployed after {incident_id} began")
    return passed, total, problems


def compute_confidence(state: AgentState, settings: VerificationSettings) -> ConfidenceBreakdown:
    claims = state.claims
    kept = [c for c in claims if c.action != "removed" and c.label is not ClaimLabel.UNSUPPORTED]
    reasons: list[str] = []
    package = state.evidence_package
    relevance = {e.label: e.relevance_score for e in package.evidence} if package else {}
    cited = [label for c in kept for label in c.supporting]
    scored = [relevance[label] for label in dict.fromkeys(cited) if label in relevance]
    retrieval = sum(scored) / len(scored) if scored else 0.0

    sources = {e.source_id for e in (package.evidence if package else []) if e.label in set(cited)}
    conflicts = [c for c in kept if c.conflicting]
    if conflicts or state.conflicts:
        agreement = 0.3
        if conflicts:
            reasons.append(f"sources disagree on {len(conflicts)} claim(s)")
        if state.conflicts:
            reasons.append(
                "sources state different values for "
                + ", ".join(c.setting for c in state.conflicts[:3])
            )
    else:
        agreement = 1.0 if len(sources) >= 2 else 0.75 if sources else 0.0
        reasons.append(f"{len(sources)} independent source(s), no conflicts")

    goals = state.goals
    goal_share = sum(goals.values()) / len(goals) if goals else 0.0
    coverage = state.query_coverage if state.query_coverage is not None else goal_share
    anchored = any(i.pinned for i in state.reranked_evidence)
    evidence_coverage = goal_share if anchored else (goal_share + coverage) / 2
    reasons.append(f"{sum(goals.values())}/{len(goals)} evidence goals met")
    failed = [r for r in state.tool_results if r.status != "ok"]
    if failed:  # a source that could not be read: the view of the evidence is partial
        evidence_coverage *= max(0.5, 1 - 0.15 * len(failed))
        reasons.append(f"{len(failed)} tool call(s) failed")

    passed, total, problems = temporal_checks(state)
    temporal = passed / total if total else 1.0
    reasons += problems or ([f"{total} temporal check(s) passed"] if total else [])

    weights = {
        ClaimLabel.SUPPORTED: 1.0,
        ClaimLabel.PARTIALLY_SUPPORTED: 0.5,
        ClaimLabel.UNSUPPORTED: 0.0,
    }
    verification = sum(weights[c.label] for c in claims) / len(claims) if claims else 0.0
    counts = {label: sum(c.label is label for c in claims) for label in ClaimLabel}
    reasons.append(
        f"claims: {counts[ClaimLabel.SUPPORTED]} supported, "
        f"{counts[ClaimLabel.PARTIALLY_SUPPORTED]} partial, "
        f"{counts[ClaimLabel.UNSUPPORTED]} unsupported"
    )

    components = {
        "retrieval_quality": retrieval,
        "source_agreement": agreement,
        "evidence_coverage": evidence_coverage,
        "temporal_consistency": temporal,
        "verification": verification,
    }
    overall = sum(WEIGHTS[k] * v for k, v in components.items())
    if state.evidence_status is EvidenceStatus.INSUFFICIENT or not kept:
        level = AnswerConfidence.INSUFFICIENT_EVIDENCE
    else:
        level = (
            AnswerConfidence.HIGH
            if overall >= settings.high_confidence
            else AnswerConfidence.MEDIUM
            if overall >= settings.medium_confidence
            else AnswerConfidence.LOW
        )
        partial_view = failed or state.evidence_status is EvidenceStatus.PARTIAL
        unapplied = bool(state.temporal and state.temporal.unhandled)
        if unapplied:
            reasons.append("the question has a qualifier the temporal plan does not apply")
        if (
            conflicts
            or state.conflicts
            or counts[ClaimLabel.UNSUPPORTED]
            or partial_view
            or unapplied
        ):
            level = _cap(level, AnswerConfidence.MEDIUM)
        security = state.security
        if security.output_removed or security.sanitized or security.quarantined:
            # Something in the evidence or the answer was untrustworthy.
            level = _cap(level, AnswerConfidence.MEDIUM)
            reasons.append("security screening removed content from the evidence or the answer")
        if problems:
            level = _cap(level, AnswerConfidence.LOW)
    return ConfidenceBreakdown(
        **{k: round(v, 3) for k, v in components.items()},
        overall=round(overall, 3),
        level=level,
        reasons=reasons,
    )
