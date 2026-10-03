"""Query understanding and routing: which kind of question is this, and which tools
fit it?

``RuleBasedRouter`` is deterministic and explainable. Every decision lists the
signals that produced it (``RoutingDecision.signals``), a one-line summary, and a
confidence. It never calls an LLM, so routing is testable and cannot be talked into
anything. A model-based router can replace it behind ``QueryRouter``, as long as it
returns the same ``RoutingDecision``.

Signals come in three kinds:
- **intent:** what the user wants done ("how many" -> SQL, "where is ... implemented"
  -> code, "what caused" -> incident). Intent decides the type.
- **entity:** a record the query names (INC-0406 -> incident, v2.8.1 -> deployment,
  a class name -> code).
- **topic:** words that indicate a subject but not the task ("deployment", "logs").
  A question *about* deployments ("what is the rollback procedure?") is still a
  document question.

Decision:
1. No domain vocabulary and no signals -> UNKNOWN.
2. An aggregation intent -> SQL_QUERY (counting incidents is SQL, not incident search).
3. A causal or temporal link between changes and failures ("what changed before the
   outage", "after the deploy", "correlate", "timeline") -> MULTI_SOURCE.
4. Two or more types with intent signals -> MULTI_SOURCE.
5. Otherwise the type with an intent wins; failing that, the highest score. Domain
   words without any signal fall back to DOCUMENT_SEARCH with low confidence.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from app.agents.entities import QueryEntities, ServiceCatalog, extract_entities
from app.schemas.enums import QueryType

Q = QueryType


class Confidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Signal(BaseModel):
    model_config = ConfigDict(frozen=True)

    query_type: QueryType
    kind: str  # intent | entity | topic | causal
    label: str  # which rule fired
    evidence: str  # the matched text
    weight: float


class RoutingDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    query: str
    query_type: QueryType
    tools: list[str]  # in the order the agent should consider them
    confidence: Confidence
    summary: str  # one line: why this route (no chain of thought)
    signals: list[Signal]
    scores: dict[str, float]
    entities: QueryEntities
    denied_tools: list[str] = []  # appropriate but not permitted for the caller


# --- rules -------------------------------------------------------------------------


@dataclass(frozen=True)
class Rule:
    query_type: QueryType
    kind: str
    label: str
    pattern: re.Pattern[str]
    weight: float


def _r(query_type: QueryType, kind: str, label: str, weight: float, *alternatives: str) -> Rule:
    """A rule matching any of ``alternatives`` (regexes, case-insensitive)."""
    return Rule(query_type, kind, label, re.compile("|".join(alternatives), re.IGNORECASE), weight)


INTENT, TOPIC = "intent", "topic"
RULES: tuple[Rule, ...] = (
    # SQL: aggregation over records (numbers, rankings, trends).
    _r(
        Q.SQL_QUERY,
        INTENT,
        "count",
        3,
        r"\bhow many\b",
        r"\bcount\b",
        r"\bnumber of\b",
        r"\btally\b",
    ),
    _r(
        Q.SQL_QUERY,
        INTENT,
        "aggregate",
        3,
        r"\b(?:average|avg|mean|median|total|sum of|percentage|percent|ratio|proportion)\b",
        r"\bMTT[RDA]\b",
    ),
    _r(
        Q.SQL_QUERY,
        INTENT,
        "ranking",
        3,
        r"\btop\s+\d+\b",
        r"\b(?:the\s+)?(?:most|fewest|least)\s+"
        r"(?!recent|recently|likely|important|common\s+cause)\w+",
        r"\bhighest\b",
        r"\blowest\b",
    ),
    _r(
        Q.SQL_QUERY,
        INTENT,
        "grouping",
        3,
        r"\b(?:per|by)\s+(?:service|month|week|day|severity|category|team|quarter|year)\b",
        r"\bbreakdown\b",
        r"\btrend\b",
        r"\bhow often\b",
        r"\bfrequency\b",
        r"\bstatistics\b",
    ),
    # Documents: how things work, procedures, configuration, policies.
    _r(
        Q.DOCUMENT_SEARCH,
        INTENT,
        "how it works",
        3,
        r"\bhow (?:does|do|is|are)\b.+"
        r"\b(?:work|works|handled|handle|configured|set up|validated|validate)\b",
    ),
    _r(
        Q.DOCUMENT_SEARCH,
        INTENT,
        "how-to",
        3,
        r"\bhow (?:do|should|can) (?:we|i|you)\b",
        r"\bhow to\b",
        r"\bwhat should (?:we|i) (?:check|do)\b",
    ),
    _r(
        Q.DOCUMENT_SEARCH,
        INTENT,
        "procedure",
        3,
        r"\b(?:procedure|runbook|playbook|steps? (?:to|for)|checklist)\b",
        r"\b(?:best practices?|guidelines?)\b",
    ),
    _r(
        Q.DOCUMENT_SEARCH,
        INTENT,
        "explain",
        3,
        r"\b(?:explain|describe|overview of|documentation|docs)\b",
    ),
    _r(
        Q.DOCUMENT_SEARCH,
        INTENT,
        "reference",
        3,
        r"\bwhat (?:is|are) the\b.*\b(?:architecture|design|policy|process|data model|schema"
        r"|options|settings|thresholds?|limits?|slos?|sla)\b",
    ),
    _r(
        Q.DOCUMENT_SEARCH,
        TOPIC,
        "doc topic",
        1,
        r"\b(?:architecture|configuration|config reference|alerts?|thresholds?|rate limits?"
        r"|limits?|slos?|policy|api|endpoints?|data model)\b",
    ),
    # Incidents: what went wrong, why, and past occurrences.
    _r(
        Q.INCIDENT_SEARCH,
        INTENT,
        "cause",
        3,
        r"\bwhat caused\b",
        r"\broot cause\b",
        r"\bwhy (?:did|does|do|is|are|was|were)\b.+\b(?:fail|failing|break|broken|crash"
        r"|go down|down|return|returning|error|erroring|stop|degrade|time out|timing out|slow)\w*",
    ),
    _r(Q.INCIDENT_SEARCH, INTENT, "what happened", 3, r"\bwhat happened\b", r"\bpost-?mortems?\b"),
    _r(
        Q.INCIDENT_SEARCH,
        INTENT,
        "past incidents",
        3,
        r"\b(?:have we had|were there|was there|any|past|previous|similar|which)\s+"
        r"(?:an?\s+)?(?:incidents?|outages?)\b",
    ),
    _r(
        Q.INCIDENT_SEARCH,
        TOPIC,
        "incident topic",
        1,
        r"\b(?:incidents?|outages?|downtime|sev\s?[1-4]|degradation|page[ds]?)\b",
    ),
    # Code: where and how something is implemented.
    _r(
        Q.CODE_SEARCH,
        INTENT,
        "implementation",
        3,
        r"\b(?:where|how) (?:is|are)\b.+\b(?:implemented|defined|coded|written)\b",
        r"\bimplementation of\b",
    ),
    _r(
        Q.CODE_SEARCH,
        INTENT,
        "code object",
        3,
        r"\b(?:which|what) (?:function|class|method|module|file)\b",
        r"\bshow me the\b.+\b(?:class|function|method|code|module)\b",
    ),
    _r(
        Q.CODE_SEARCH,
        INTENT,
        "find code",
        3,
        r"\b(?:find|show|search)(?: me)? the code\b",
        r"\bthe code (?:that|which|where|for)\b",
        r"\bin (?:the )?(?:[\w-]+ )?(?:code|codebase|source|repo(?:sitory)?)\b",
        r"\bsource code\b",
    ),
    _r(
        Q.CODE_SEARCH,
        INTENT,
        "what does it do",
        3,
        r"\bwhat does\s+[A-Za-z_][\w.]*(?:\(\))?\s+do\b",
    ),
    _r(
        Q.CODE_SEARCH,
        TOPIC,
        "code topic",
        1,
        r"\b(?:code|function|class|method|module|library|unit tests?)\b",
    ),
    # Deployments: releases, versions, rollbacks, what shipped.
    _r(
        Q.DEPLOYMENT_SEARCH,
        INTENT,
        "what changed in",
        3,
        r"\bwhat (?:changed|was (?:released|deployed|shipped)|went out)\b.*\b(?:in|with)\b",
    ),
    _r(
        Q.DEPLOYMENT_SEARCH,
        INTENT,
        "shipped",
        3,
        r"\bshipped in\b",
        r"\bwhich (?:version|release)\b",
        r"\bcurrently deployed\b",
        r"\b(?:recent|latest|last)\s+(?:deployments?|deploys?|releases?)\b",
    ),
    _r(
        Q.DEPLOYMENT_SEARCH,
        INTENT,
        "rollback",
        3,
        r"\bwhen was\b.+\b(?:deployed|released|rolled back)\b",
        r"\blast rolled back\b",
        r"\bwas rolled back\b",
    ),
    _r(
        Q.DEPLOYMENT_SEARCH,
        TOPIC,
        "deployment topic",
        1,
        r"\b(?:deploy(?:s|ed|ment|ments)?|releases?|released|rollbacks?|rolled back|roll back"
        r"|canary|changelog|versions?|shipped)\b",
    ),
    # Logs: log lines, traces, error messages as logged.
    _r(
        Q.LOG_SEARCH,
        INTENT,
        "logs",
        3,
        r"\b(?:show|find|search|grep|get|fetch|list|any|are there)\b.*"
        r"\blog(?:s| lines| entries| messages)?\b",
        r"\bwhat did the logs\b",
    ),
    _r(
        Q.LOG_SEARCH,
        INTENT,
        "log lines",
        3,
        r"\blog (?:lines|entries|messages)\b",
        r"\b(?:debug|info|warn|warning|error|critical)\s+logs\b",
        r"\bstack ?traces?\b",
        r"\btraceback\b",
    ),
    _r(
        Q.LOG_SEARCH,
        TOPIC,
        "log topic",
        1,
        r"\blogs?\b",
        r"\blogged\b",
        r"\btrace\b",
        r"\bexceptions?\b",
    ),
)

# Changes connected to failures in time -> several sources are needed.
_CHANGE = r"(?:release|deploy(?:ment)?|change|rollout|v\d+(?:\.\d+)+)"
CAUSAL = re.compile(
    "|".join(
        [
            r"\bwhat changed (?:before|after|prior to|leading up to|around)\b",
            r"\b(?:right |just |shortly )?(?:before|after|prior to|leading up to|since)\s+"
            r"(?:the\s+)?(?:[\w.-]+[\s-]+){0,2}"
            r"(?:outage|incident|deploy(?:ment)?|release|rollout|INC-\d+|DEP-\d+|v\d+(?:\.\d+)+)\b",
            r"\bcorrelat\w*\b",
            r"\btimeline\b",
            rf"\bdid\b.+\b{_CHANGE}\b.+\bcause\b",
            rf"\b{_CHANGE}\b.+\bcaus(?:e|ed)\b.+\b(?:outage|incident|errors?|spike)\b",
        ]
    ),
    re.IGNORECASE,
)

# Operations vocabulary: its presence means "in domain" even without a specific signal.
DOMAIN = re.compile(
    r"\b(?:incident|outage|error|errors|latency|timeouts?|database|db|redis|kafka|cache|pods?|"
    r"cpu|memory|oom\w*|api|requests?|deploy\w*|release\w*|versions?|logs?|code|services?|"
    r"payments?|cart|orders?|inventory|checkout|products?|prices?|search|notifications?|emails?|"
    r"auth\w*|tokens?|users?|gateway|queue|consumer|http|5\d\d|4\d\d|sql|stock|warehouse|"
    r"alerts?|slo|runbook|postmortem|config\w*|jwt|jwks|novacart|rollback|canary|"
    r"reliability|reports?)\b",
    re.IGNORECASE,
)

TOOLS: dict[QueryType, list[str]] = {
    Q.DOCUMENT_SEARCH: ["search_documents"],
    Q.INCIDENT_SEARCH: ["search_incidents"],
    Q.CODE_SEARCH: ["search_code"],
    Q.SQL_QUERY: ["query_database"],
    Q.DEPLOYMENT_SEARCH: ["search_deployments"],
    Q.LOG_SEARCH: ["search_logs"],
    Q.UNKNOWN: [],
}
# For causal questions: what changed (deployments), what broke (incidents), and the
# evidence in between (logs).
CAUSAL_TOOLS = ["search_incidents", "search_deployments", "search_logs"]
TIE_ORDER = [Q.INCIDENT_SEARCH, Q.DEPLOYMENT_SEARCH, Q.CODE_SEARCH, Q.LOG_SEARCH, Q.DOCUMENT_SEARCH]
RUNBOOK_HINT = re.compile(
    r"\b(?:runbook|playbook|procedure|mitigat\w*"
    r"|how (?:do|should) we (?:handle|respond|fix|recover))\b",
    re.IGNORECASE,
)


def entity_signals(entities: QueryEntities) -> list[Signal]:
    rows: list[tuple[QueryType, str, Iterable[str], float]] = [
        (Q.INCIDENT_SEARCH, "incident id", entities.incident_ids, 3),
        (
            Q.INCIDENT_SEARCH,
            "postmortem id",
            [d for d in entities.document_ids if d.startswith("PM-")],
            2,
        ),
        (Q.DEPLOYMENT_SEARCH, "deployment id", entities.deployment_ids, 3),
        (Q.DEPLOYMENT_SEARCH, "version", entities.versions, 2),
        (Q.DEPLOYMENT_SEARCH, "pull request id", entities.pull_request_ids, 1.5),
        (Q.CODE_SEARCH, "pull request id", entities.pull_request_ids, 1.5),
        (
            Q.DOCUMENT_SEARCH,
            "document id",
            [d for d in entities.document_ids if not d.startswith("PM-")],
            3,
        ),
        (Q.CODE_SEARCH, "code file id", entities.code_file_ids, 3),
        (Q.CODE_SEARCH, "file path", entities.file_paths, 3),
        (Q.CODE_SEARCH, "code identifier", entities.code_identifiers, 2),
        (Q.CODE_SEARCH, "config key", entities.config_keys, 1.5),
        (Q.LOG_SEARCH, "trace id", entities.trace_ids, 3),
        (Q.LOG_SEARCH, "log level", [level.value for level in entities.log_levels], 1.5),
    ]
    return [
        Signal(query_type=qt, kind="entity", label=label, evidence=value, weight=weight)
        for qt, label, values, weight in rows
        for value in values
    ]


class QueryRouter(ABC):
    @abstractmethod
    def route(self, query: str, permitted: Callable[[str], bool] | None = None) -> RoutingDecision:
        """Classify ``query``; ``permitted(tool)`` drops tools the caller may not use."""


class RuleBasedRouter(QueryRouter):
    def __init__(
        self,
        services: ServiceCatalog | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        rules: tuple[Rule, ...] = RULES,
    ) -> None:
        self.services = services
        self.clock = clock
        self.rules = rules

    def route(self, query: str, permitted: Callable[[str], bool] | None = None) -> RoutingDecision:
        text = " ".join(query.split())[:1000]
        entities = extract_entities(text, self.services, self.clock())
        signals = entity_signals(entities)
        for rule in self.rules:
            match = rule.pattern.search(text)
            if match:
                signals.append(
                    Signal(
                        query_type=rule.query_type,
                        kind=rule.kind,
                        label=rule.label,
                        evidence=match.group()[:80],
                        weight=rule.weight,
                    )
                )
        causal = CAUSAL.search(text)
        if causal:
            signals.append(
                Signal(
                    query_type=Q.MULTI_SOURCE,
                    kind="causal",
                    label="change linked to failure",
                    evidence=causal.group()[:80],
                    weight=3,
                )
            )
        scores: dict[QueryType, float] = defaultdict(float)
        intents: dict[QueryType, int] = defaultdict(int)
        for signal in signals:
            scores[signal.query_type] += signal.weight
            if signal.kind == INTENT:
                intents[signal.query_type] += 1

        query_type, tools, confidence, summary = self._decide(
            text, entities, signals, scores, intents, bool(causal)
        )
        denied: list[str] = []
        if permitted is not None:
            denied = [t for t in tools if not permitted(t)]
            tools = [t for t in tools if permitted(t)]
        return RoutingDecision(
            query=text,
            query_type=query_type,
            tools=tools,
            confidence=confidence,
            summary=summary,
            signals=signals,
            scores={k.value: round(v, 2) for k, v in sorted(scores.items())},
            entities=entities,
            denied_tools=denied,
        )

    def _decide(
        self,
        text: str,
        entities: QueryEntities,
        signals: list[Signal],
        scores: dict[QueryType, float],
        intents: dict[QueryType, int],
        causal: bool,
    ) -> tuple[QueryType, list[str], Confidence, str]:
        def evidence(query_type: QueryType) -> str:
            found = [s.evidence for s in signals if s.query_type == query_type][:3]
            return ", ".join(repr(e) for e in found)

        in_domain = bool(DOMAIN.search(text)) or bool(entities.services) or bool(signals)
        if not in_domain:
            return Q.UNKNOWN, [], Confidence.HIGH, "no operations vocabulary or record ids"
        if intents.get(Q.SQL_QUERY):
            return (
                Q.SQL_QUERY,
                TOOLS[Q.SQL_QUERY],
                Confidence.HIGH,
                f"aggregation requested ({evidence(Q.SQL_QUERY)})",
            )
        typed = [t for t in TIE_ORDER if scores.get(t, 0) > 0]
        with_intent = [t for t in TIE_ORDER if intents.get(t)]
        if causal or len(with_intent) >= 2:
            tools = list(CAUSAL_TOOLS) if causal else []
            for t in sorted(typed, key=lambda t: (-scores[t], TIE_ORDER.index(t))):
                if intents.get(t) or scores[t] >= 2 or not causal:
                    for tool in TOOLS[t]:
                        if tool not in tools:
                            tools.append(tool)
            reason = (
                f"links a change to a failure ({evidence(Q.MULTI_SOURCE)})"
                if causal
                else "asks for several things: " + ", ".join(t.value for t in with_intent)
            )
            return (
                Q.MULTI_SOURCE,
                tools,
                Confidence.MEDIUM if not causal else Confidence.HIGH,
                reason,
            )
        if with_intent:
            best = (
                with_intent[0]
                if len(with_intent) == 1
                else max(with_intent, key=scores.__getitem__)
            )
            confidence = Confidence.HIGH
        elif typed:
            best = max(typed, key=lambda t: (scores[t], -TIE_ORDER.index(t)))
            confidence = Confidence.MEDIUM if scores[best] >= 2 else Confidence.LOW
        else:
            return (
                Q.DOCUMENT_SEARCH,
                TOOLS[Q.DOCUMENT_SEARCH],
                Confidence.LOW,
                "operations question without a specific signal; general search",
            )
        tools = list(TOOLS[best])
        if best == Q.DOCUMENT_SEARCH and RUNBOOK_HINT.search(text):
            tools.append("get_runbook")
        if best == Q.INCIDENT_SEARCH and RUNBOOK_HINT.search(text):
            tools.append("get_runbook")
        return best, tools, confidence, f"{best.value} ({evidence(best)})"
