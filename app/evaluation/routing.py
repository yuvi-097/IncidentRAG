"""Router evaluation against labelled queries (``data/evaluation/routing_benchmark.jsonl``).

Each case has an expected query type and, for multi-source questions, the tools
that must be among those selected. Reports accuracy, per-type precision and recall,
a confusion matrix and every miss with the router's own explanation.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from app.agents.router import QueryRouter
from app.schemas.enums import QueryType


class RoutingCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    query: str
    expected_type: QueryType
    expected_tools: list[str] = []  # must all be selected (multi-source cases)
    source: str = "written"  # "spec" = example from the project specification


class RoutingOutcome(BaseModel):
    id: str
    query: str
    expected_type: QueryType
    predicted_type: QueryType
    tools: list[str]
    missing_tools: list[str]
    confidence: str
    summary: str

    @property
    def correct(self) -> bool:
        return self.expected_type == self.predicted_type and not self.missing_tools


class RoutingReport(BaseModel):
    cases: int
    accuracy: float  # type and required tools both right
    type_accuracy: float
    per_type: dict[str, dict[str, float]]  # precision / recall / support
    confusion: dict[str, dict[str, int]]  # expected -> predicted -> count
    outcomes: list[RoutingOutcome]


def load_cases(path: Path) -> list[RoutingCase]:
    cases = [
        RoutingCase.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len({c.id for c in cases}) != len(cases):
        raise ValueError("duplicate case ids")
    return cases


def evaluate_router(router: QueryRouter, cases: list[RoutingCase]) -> RoutingReport:
    outcomes = []
    for case in cases:
        decision = router.route(case.query)
        outcomes.append(
            RoutingOutcome(
                id=case.id,
                query=case.query,
                expected_type=case.expected_type,
                predicted_type=decision.query_type,
                tools=decision.tools,
                missing_tools=[t for t in case.expected_tools if t not in decision.tools],
                confidence=decision.confidence.value,
                summary=decision.summary,
            )
        )
    n = len(outcomes) or 1
    confusion: dict[str, Counter[str]] = {}
    for o in outcomes:
        confusion.setdefault(o.expected_type.value, Counter())[o.predicted_type.value] += 1
    per_type = {}
    for query_type in QueryType:
        tp = sum(o.expected_type == o.predicted_type == query_type for o in outcomes)
        predicted = sum(o.predicted_type == query_type for o in outcomes)
        support = sum(o.expected_type == query_type for o in outcomes)
        per_type[query_type.value] = {
            "precision": round(tp / predicted, 3) if predicted else 0.0,
            "recall": round(tp / support, 3) if support else 0.0,
            "support": float(support),
        }
    return RoutingReport(
        cases=len(outcomes),
        accuracy=round(sum(o.correct for o in outcomes) / n, 4),
        type_accuracy=round(sum(o.expected_type == o.predicted_type for o in outcomes) / n, 4),
        per_type=per_type,
        confusion={k: dict(v) for k, v in sorted(confusion.items())},
        outcomes=outcomes,
    )


# Cross-check on questions written for the retrieval benchmark (Phases 3-4), before
# the router existed. Acceptable types derive from each question's existing category
# (or, for exact-match questions, the kind in its notes); the mapping was fixed before
# the router was run on them.
CATEGORY_TYPES: dict[str, set[QueryType]] = {
    "runbook": {QueryType.DOCUMENT_SEARCH},
    "documentation": {QueryType.DOCUMENT_SEARCH},
    "code": {QueryType.CODE_SEARCH},
    "incident": {QueryType.INCIDENT_SEARCH, QueryType.MULTI_SOURCE},
    "change": {QueryType.DEPLOYMENT_SEARCH, QueryType.MULTI_SOURCE},
}
EXACT_KIND_TYPES: dict[str, set[QueryType]] = {
    "error message": {QueryType.DOCUMENT_SEARCH, QueryType.LOG_SEARCH, QueryType.INCIDENT_SEARCH},
    "incident id": {QueryType.INCIDENT_SEARCH},
    "deployment id": {QueryType.DEPLOYMENT_SEARCH},
    "version number": {QueryType.DEPLOYMENT_SEARCH, QueryType.MULTI_SOURCE},
    "config key": {QueryType.CODE_SEARCH, QueryType.DOCUMENT_SEARCH},
    "class name": {QueryType.CODE_SEARCH},
    "function name": {QueryType.CODE_SEARCH},
    "service name": {QueryType.INCIDENT_SEARCH},
    "alert name": {QueryType.DOCUMENT_SEARCH},
}


class CrossCheckOutcome(BaseModel):
    id: str
    question: str
    category: str
    acceptable: list[QueryType]
    predicted_type: QueryType
    agrees: bool
    summary: str


def acceptable_types(category: str, notes: str) -> set[QueryType]:
    if category == "exact-match":
        kind = notes.split(".")[0].strip()
        return EXACT_KIND_TYPES[kind]
    return CATEGORY_TYPES[category]


def cross_check(
    router: QueryRouter, questions: list[tuple[str, str, str, str]]
) -> list[CrossCheckOutcome]:
    """``questions``: (id, question, category, notes) from the retrieval benchmark."""
    outcomes = []
    for qid, question, category, notes in questions:
        allowed = acceptable_types(category, notes)
        decision = router.route(question)
        outcomes.append(
            CrossCheckOutcome(
                id=qid,
                question=question,
                category=category,
                acceptable=sorted(allowed),
                predicted_type=decision.query_type,
                agrees=decision.query_type in allowed,
                summary=decision.summary,
            )
        )
    return outcomes
