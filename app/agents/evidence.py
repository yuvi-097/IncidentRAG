"""Evidence: aggregation, reranking and validation.

1. **Aggregation.** Each tool returns its own typed output: passages, incident
   summaries, deployments, log lines, a runbook, SQL rows. ``aggregate`` converts
   all of them into ``EvidenceItem``s with a readable text, the source id and the
   access level. Log lines are summarised (counts per message) rather than copied.
2. **Reranking.** Items from different tools are ranked against the question. The
   cross-encoder is used when one is configured; otherwise IDF-weighted term
   coverage. Items asked for directly (an incident by id, a SQL result) are pinned
   first. At most two passages per source are kept, then the top ``max_evidence``.
3. **Validation.** Items are checked again against the caller's clearance (defence
   in depth). The planner's evidence goals are checked, and so is answerability: how
   much of the question's key vocabulary, weighted by rarity, the evidence contains.
   Terms that appear nowhere in the corpus count fully, so a question about
   something the corpus does not know ("the Mars colony warehouse") is reported as
   insufficient instead of being answered with loosely related passages.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from typing import Protocol

from app.agents.guard import may_read_item
from app.agents.multihop import chain_items
from app.agents.state import (
    AgentState,
    EvidenceItem,
    EvidenceKind,
    EvidenceStatus,
    PlannedCall,
)
from app.observability.metrics import METRICS
from app.rag.reranking.base import PairScorer
from app.rag.retrieval.tokenizer import Tokenizer
from app.schemas.enums import UNLABELLED_LEVEL, QueryType, SourceType
from app.tools.base import Evidence, ToolModel
from app.tools.deployments import DeploymentRecord, SearchDeploymentsOutput
from app.tools.logs import SearchLogsOutput
from app.tools.runbooks import GetRunbookOutput
from app.tools.search import IncidentSummary, SearchIncidentsOutput, SearchResultsOutput
from app.tools.sql_tool import QueryDatabaseOutput
from app.tools.trace import TraceChangeOutput

MAX_TEXT = 2400
_KIND = {
    SourceType.DOCUMENTATION: EvidenceKind.DOCUMENT,
    SourceType.RUNBOOK: EvidenceKind.RUNBOOK,
    SourceType.POSTMORTEM: EvidenceKind.POSTMORTEM,
    SourceType.INCIDENT: EvidenceKind.INCIDENT,
    SourceType.DEPLOYMENT: EvidenceKind.DEPLOYMENT,
    SourceType.CODE: EvidenceKind.CODE,
    SourceType.PULL_REQUEST: EvidenceKind.PULL_REQUEST,
}
_ID_ARGUMENTS = ("incident_ids", "deployment_ids", "versions", "runbook_id")


def _utc(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=UTC)


def _when(value: datetime) -> str:
    return f"{_utc(value):%Y-%m-%d %H:%M} UTC"


# --- rendering ---------------------------------------------------------------------


def render_incident(incident: IncidentSummary) -> str:
    links = [
        ("deployment", incident.deployment_id),
        ("root-cause deployment", incident.root_cause_deployment_id),
        ("remediation deployment", incident.remediation_deployment_id),
        ("root-cause pull request", incident.root_cause_pr_id),
        ("runbook", incident.runbook_id),
        ("postmortem", incident.postmortem_id),
        ("parent incident", incident.parent_incident_id),
    ]
    linked = "; ".join(f"{name} {value}" for name, value in links if value)
    return (
        f"{incident.id}: {incident.title}. Severity {incident.severity.value}, service "
        f"{incident.service_id}, category {incident.category.value}, status "
        f"{incident.status.value}. Started {_when(incident.started_at)}, resolved "
        f"{_when(incident.resolved_at)} ({incident.resolution_time_minutes} min). Affected "
        f"version {incident.affected_version}. Linked: {linked or 'none'}.\n"
        f"Symptoms: {incident.symptoms}\nRoot cause: {incident.root_cause}\n"
        f"Resolution: {incident.resolution}"
    )


def render_deployment(deployment: DeploymentRecord) -> str:
    prs = "; ".join(f"{p.id} {p.title}" for p in deployment.pull_requests) or "none visible"
    incidents = "; ".join(f"{i.id} ({i.relation}): {i.title}" for i in deployment.incidents)
    previous = f" (previous {deployment.previous_version})" if deployment.previous_version else ""
    rollback = f", rollback of {deployment.rollback_of_id}" if deployment.rollback_of_id else ""
    return (
        f"{deployment.id}: {deployment.service_id} {deployment.version}{previous}, status "
        f"{deployment.status.value}, {deployment.strategy.value} deployment at "
        f"{_when(deployment.deployed_at)}{rollback}.\nChanges: {deployment.changes}\n"
        f"Pull requests: {prs}\nLinked incidents: {incidents or 'none'}"
    )


_VARIABLE = re.compile(r"\b[0-9a-f]{12,}\b|\b\d+(?:\.\d+)?\b")


def render_logs(output: SearchLogsOutput) -> tuple[str, str]:
    services = sorted({e.service_id for e in output.entries})
    window = ""
    if output.since and output.until:
        window = f" between {_when(output.since)} and {_when(output.until)}"
    counts = ", ".join(f"{level} {n}" for level, n in output.level_counts.items())
    head = (
        f"Logs of {', '.join(services) or 'the selected services'}{window}: "
        f"{output.total_matched} matching lines ({counts or 'none'})."
    )
    groups: dict[str, list] = {}
    for entry in output.entries:
        groups.setdefault(_VARIABLE.sub("#", entry.message), []).append(entry)
    lines = []
    for _, entries in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:5]:
        first, last = min(e.timestamp for e in entries), max(e.timestamp for e in entries)
        sample = entries[0]
        lines.append(
            f'- {sample.level.value} {sample.service_id}: "{sample.message}" '
            f"({len(entries)} of the returned lines; first {_when(first)}, last {_when(last)})"
        )
    title = f"Logs: {', '.join(services) or 'no matching lines'}"
    return title, head + ("\n" + "\n".join(lines) if lines else "")


def render_sql(call: PlannedCall, output: QueryDatabaseOutput) -> str:
    rows = output.rows[:20]
    if output.row_count == 1 and len(output.columns) == 1:
        table = f"{output.columns[0]} = {rows[0][0]}"
    else:
        table = "\n".join(
            ", ".join(f"{c}={v}" for c, v in zip(output.columns, row, strict=False)) for row in rows
        )
    more = f"\n(first {len(rows)} rows; more exist)" if output.truncated else ""
    result = table or "(no rows)"
    return f"Query: {call.purpose}.\nSQL: {call.arguments.get('sql')}\nResult:\n{result}{more}"


def _top_log_line(output: SearchLogsOutput) -> str:
    if not output.entries:
        return ""
    groups = Counter(_VARIABLE.sub("#", e.message) for e in output.entries)
    pattern, count = groups.most_common(1)[0]
    sample = next(e for e in output.entries if _VARIABLE.sub("#", e.message) == pattern)
    return f'{sample.level.value} "{sample.message}" ({count} of the returned lines)'


def _section(markdown: str, heading: str) -> str:
    match = re.search(rf"^#+\s+{heading}\s*$(.*?)(?=^#+\s|\Z)", markdown, re.M | re.S)
    if not match:
        return ""
    steps = []
    for line in match.group(1).splitlines():
        text = re.sub(r"^\s*(?:\d+[.)]|[-*])\s*", "", line).strip().rstrip(".")
        if text:
            steps.append(" ".join(text.split()))
    return "; ".join(steps)[:800]


def _sql_facts(call: PlannedCall, output: QueryDatabaseOutput) -> dict[str, str]:
    facts = {"description": call.purpose, "rows": str(output.row_count)}
    if output.row_count == 1 and len(output.columns) == 1:
        facts["value"] = str(output.rows[0][0])
    else:
        facts["table"] = "; ".join(
            ", ".join(f"{c}={v}" for c, v in zip(output.columns, row, strict=False))
            for row in output.rows[:10]
        )
    return facts


# --- aggregation -------------------------------------------------------------------


def _from_passage(evidence: Evidence, tool: str) -> EvidenceItem:
    section = (
        f" / {evidence.section}" if evidence.section and evidence.section != evidence.title else ""
    )
    return EvidenceItem(
        kind=_KIND.get(evidence.source_type, EvidenceKind.DOCUMENT),
        source_id=evidence.document_id,
        chunk_id=evidence.chunk_id,
        title=evidence.title,
        text=(f"{evidence.title}{section}\n{evidence.content}")[:MAX_TEXT],
        service_id=evidence.service_id,
        timestamp=evidence.timestamp,
        access_level=evidence.access_level,
        location=evidence.file_path,
        section=evidence.section,
        tool=tool,
        score=evidence.score,
        facts={"section": evidence.section or "", "content": evidence.content},
    )


def aggregate(outputs: Sequence[tuple[PlannedCall, ToolModel]]) -> list[EvidenceItem]:
    items: list[EvidenceItem] = []
    for call, output in outputs:
        pinned = any(call.arguments.get(name) for name in _ID_ARGUMENTS)
        if isinstance(output, SearchResultsOutput):
            items += [_from_passage(e, call.tool) for e in output.results]
        elif isinstance(output, SearchIncidentsOutput):
            for incident in output.incidents:
                items.append(
                    EvidenceItem(
                        kind=EvidenceKind.INCIDENT,
                        source_id=incident.id,
                        title=incident.title,
                        text=render_incident(incident)[:MAX_TEXT],
                        service_id=incident.service_id,
                        timestamp=incident.started_at,
                        access_level=incident.access_level,
                        tool=call.tool,
                        pinned=pinned,
                        facts={
                            "id": incident.id,
                            "title": incident.title,
                            "severity": incident.severity.value,
                            "service": incident.service_id,
                            "started": _when(incident.started_at),
                            "resolved": _when(incident.resolved_at),
                            "root_cause": incident.root_cause,
                            "resolution": incident.resolution,
                            "symptoms": incident.symptoms,
                        },
                    )
                )
            items += [_from_passage(e, call.tool) for e in output.evidence]
        elif isinstance(output, SearchDeploymentsOutput):
            for deployment in output.deployments:
                items.append(
                    EvidenceItem(
                        kind=EvidenceKind.DEPLOYMENT,
                        source_id=deployment.id,
                        title=f"{deployment.service_id} {deployment.version}",
                        text=render_deployment(deployment)[:MAX_TEXT],
                        service_id=deployment.service_id,
                        timestamp=deployment.deployed_at,
                        access_level=UNLABELLED_LEVEL,
                        tool=call.tool,
                        pinned=pinned,
                        facts={
                            "id": deployment.id,
                            "service": deployment.service_id,
                            "version": deployment.version,
                            "status": deployment.status.value,
                            "deployed": _when(deployment.deployed_at),
                            "changes": " ".join(deployment.changes.split()),
                            "pull_requests": "; ".join(
                                f"{p.id} {p.title}" for p in deployment.pull_requests
                            ),
                            "incidents": "; ".join(
                                f"{i.id} ({i.relation})" for i in deployment.incidents
                            ),
                        },
                    )
                )
            items += [_from_passage(e, call.tool) for e in output.evidence]
        elif isinstance(output, SearchLogsOutput) and output.total_matched:
            title, text = render_logs(output)
            services = sorted({e.service_id for e in output.entries})
            items.append(
                EvidenceItem(
                    kind=EvidenceKind.LOGS,
                    source_id=f"logs:{','.join(services)}:{output.since:%Y%m%dT%H%M}"
                    if output.since
                    else f"logs:{','.join(services)}",
                    title=title,
                    text=text[:MAX_TEXT],
                    service_id=services[0] if len(services) == 1 else None,
                    timestamp=output.since,
                    access_level=UNLABELLED_LEVEL,
                    tool=call.tool,
                    pinned=bool(
                        call.arguments.get("trace_id") or call.arguments.get("deployment_id")
                    ),
                    facts={"summary": text.splitlines()[0], "top": _top_log_line(output)},
                )
            )
        elif isinstance(output, GetRunbookOutput):
            runbook = output.runbook
            items.append(
                EvidenceItem(
                    kind=EvidenceKind.RUNBOOK,
                    source_id=runbook.id,
                    title=runbook.title,
                    text=f"Runbook {runbook.id}: {runbook.title}\n{runbook.content}"[:MAX_TEXT],
                    service_id=runbook.service_id,
                    timestamp=runbook.updated_at,
                    access_level=runbook.access_level,
                    location=runbook.source_path,
                    tool=call.tool,
                    pinned=True,  # fetched deliberately (by id, title or best match)
                    facts={
                        "id": runbook.id,
                        "title": runbook.title,
                        "mitigation": _section(runbook.content, "Mitigation"),
                    },
                )
            )
        elif isinstance(output, TraceChangeOutput):
            items += chain_items(output)  # one item per hop, each with its own label
        elif isinstance(output, QueryDatabaseOutput):
            items.append(
                EvidenceItem(
                    kind=EvidenceKind.SQL_RESULT,
                    source_id="sql",
                    title=call.purpose,
                    text=render_sql(call, output)[:MAX_TEXT],
                    access_level=UNLABELLED_LEVEL,
                    tool=call.tool,
                    pinned=True,
                    facts=_sql_facts(call, output),
                )
            )
    # A complete runbook subsumes passages of the same document.
    full = {i.source_id for i in items if i.kind is EvidenceKind.RUNBOOK and i.chunk_id is None}
    items = [i for i in items if i.chunk_id is None or i.source_id not in full]
    unique: dict[tuple[str, str, str | None], EvidenceItem] = {}
    for item in items:
        existing = unique.get(item.key)
        if existing is None or (item.pinned and not existing.pinned):
            unique[item.key] = item
    return list(unique.values())


# --- term statistics and coverage ----------------------------------------------------


class TermStatistics(Protocol):
    def idf(self, term: str) -> float: ...

    def document_frequency(self, term: str) -> int: ...

    def __len__(self) -> int: ...


def term_weights(
    query: str, tokenizer: Tokenizer, stats: TermStatistics | None
) -> dict[str, float]:
    """Query terms and their weights: IDF, with unseen terms at the maximum IDF."""
    terms = dict.fromkeys(t for t in tokenizer(query) if len(t) > 1)
    if stats is None:
        return dict.fromkeys(terms, 1.0)
    unseen = math.log(1 + (len(stats) + 0.5) / 0.5)
    return {t: stats.idf(t) if stats.document_frequency(t) else unseen for t in terms}


def coverage(weights: dict[str, float], texts: Iterable[str], tokenizer: Tokenizer) -> float:
    if not weights:
        return 0.0
    present: set[str] = set()
    for text in texts:
        present.update(tokenizer(text))
    total = sum(weights.values())
    return sum(w for t, w in weights.items() if t in present) / total if total else 0.0


# --- reranking ---------------------------------------------------------------------


class EvidenceRanker:
    def __init__(
        self,
        scorer: PairScorer | None = None,
        stats: TermStatistics | None = None,
        tokenizer: Tokenizer | None = None,
        per_source: int = 2,
    ) -> None:
        self.scorer = scorer
        self.stats = stats
        self.tokenizer = tokenizer or Tokenizer()
        self.per_source = per_source

    @property
    def method(self) -> str:
        return f"cross-encoder ({self.scorer.name})" if self.scorer else "term coverage"

    def rank(self, query: str, items: Sequence[EvidenceItem], limit: int) -> list[EvidenceItem]:
        pinned = [i for i in items if i.pinned]
        others = [i for i in items if not i.pinned]
        if others:
            if self.scorer is not None:
                with METRICS.timed("evidence.rerank"):
                    scores = self.scorer.score([(query, i.text[:MAX_TEXT]) for i in others])
                relevance = [1 / (1 + math.exp(-float(x))) for x in scores]  # logits -> [0, 1]
            else:
                weights = term_weights(query, self.tokenizer, self.stats)
                scores = [coverage(weights, [i.text], self.tokenizer) for i in others]
                relevance = list(scores)
            others = [
                item.model_copy(
                    update={"score": round(float(score), 4), "relevance": round(rel, 4)}
                )
                for item, score, rel in sorted(
                    zip(others, scores, relevance, strict=True), key=lambda row: -float(row[1])
                )
            ]
        pinned = [i.model_copy(update={"relevance": 1.0}) for i in pinned]  # asked for directly
        per_source: Counter[str] = Counter()
        kept: list[EvidenceItem] = []
        for item in [*pinned, *others]:
            if per_source[item.source_id] >= self.per_source:
                continue
            per_source[item.source_id] += 1
            kept.append(item)
        return [
            item.model_copy(update={"label": f"E{n}"}) for n, item in enumerate(kept[:limit], 1)
        ]


# --- validation ----------------------------------------------------------------------

_TEXT_SEARCH = set(QueryType) - {QueryType.SQL_QUERY, QueryType.UNKNOWN}


class EvidenceValidator:
    def __init__(
        self,
        min_coverage: float,
        stats: TermStatistics | None = None,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        self.min_coverage = min_coverage
        self.stats = stats
        self.tokenizer = tokenizer or Tokenizer()

    def validate(self, state: AgentState) -> None:
        principal = state.principal
        allowed = [i for i in state.reranked_evidence if may_read_item(principal, i)]
        if len(allowed) != len(state.reranked_evidence):  # screened already; never expected
            state.evidence_notes.append("evidence the caller may not read was removed")
        state.reranked_evidence = allowed
        notes = state.evidence_notes
        for _, output in state.outputs:
            for missing in getattr(output, "not_found", []) or []:
                notes.append(f"{missing} was not found or is not accessible")
        goals = state.goals
        if not goals or not allowed:
            state.evidence_status = EvidenceStatus.INSUFFICIENT
            notes.append("no evidence was retrieved" if goals else "no tool fits this question")
            return
        met = [g for g, ok in goals.items() if ok]
        missing_goals = [g for g, ok in goals.items() if not ok]
        if missing_goals:
            notes.append("missing evidence: " + ", ".join(missing_goals))
        status = (
            EvidenceStatus.SUFFICIENT
            if not missing_goals
            else EvidenceStatus.PARTIAL
            if met
            else EvidenceStatus.INSUFFICIENT
        )
        anchored = any(i.pinned for i in allowed)
        weights = term_weights(state.query, self.tokenizer, self.stats)
        score = coverage(weights, [i.text for i in allowed], self.tokenizer)
        state.query_coverage = round(score, 3)
        unseen_terms = {
            t for t in weights if self.stats is not None and not self.stats.document_frequency(t)
        }
        state.unknown_terms = list(
            dict.fromkeys(
                w
                for w in re.findall(r"[A-Za-z][\w-]*", state.query)
                if unseen_terms & set(self.tokenizer(w))
            )
        )
        if (
            state.query_type in _TEXT_SEARCH
            and not anchored
            and status is not EvidenceStatus.INSUFFICIENT
            and score < self.min_coverage
        ):
            # The evidence does not talk about what was asked.
            unseen = {
                t
                for t in weights
                if self.stats is not None and not self.stats.document_frequency(t)
            }
            words = [
                w
                for w in re.findall(r"[A-Za-z][\w-]*", state.query)
                if unseen & set(self.tokenizer(w))
            ]
            detail = f"; not in the corpus: {', '.join(dict.fromkeys(words))}" if words else ""
            notes.append(
                f"the evidence covers {score:.0%} of the question's key terms "
                f"(minimum {self.min_coverage:.0%}){detail}"
            )
            status = EvidenceStatus.INSUFFICIENT
        state.evidence_status = status
