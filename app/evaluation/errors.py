"""Error analysis: why an answer failed, classified by deterministic rules.

A (question, method) pair *fails* when the answer is incorrect, contains a hallucinated
token, leaks forbidden content, or cites evidence that does not ground the claim. Each
failure gets every class that applies, and one *primary* class (the first in this
order), so that the tables add up while co-occurring causes stay visible:

1. permission_failure  restricted or forbidden content in the answer (a leak, also after
                       a prompt injection), or an allowed question declined as if denied
2. hallucination       a record id, version, hash, path, setting or number in the answer
                       that is in neither the context nor the question
3. routing_failure     (full system only) routed to a query type that is neither the
                       expected nor an acceptable alternative, and the answer is wrong
4. retrieval_failure   no gold source among the candidates the system looked at; for
                       an aggregate (SQL) question, a method that has no SQL path
5. reranking_failure   a gold source was among the candidates but not in the context
6. temporal_failure    a temporal question answered wrongly although retrieval found
                       the gold records (the ordering in time went wrong)
7. reasoning_failure   the evidence was there but the answer is wrong, or an
                       unanswerable question was answered instead of declined
8. citation_failure    a cited claim is not grounded by its citations, or a citation
                       points to nothing in the context

These rules read only the recorded outputs, so the analysis is reproducible.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

from app.evaluation.eval_set import Category, CheckResult, EvalQuestion
from app.evaluation.metrics import GenerationScores
from app.evaluation.pipelines import MethodOutput


class ErrorClass(StrEnum):
    PERMISSION = "permission_failure"
    HALLUCINATION = "hallucination"
    ROUTING = "routing_failure"
    RETRIEVAL = "retrieval_failure"
    RERANKING = "reranking_failure"
    TEMPORAL = "temporal_failure"
    REASONING = "reasoning_failure"
    CITATION = "citation_failure"


ORDER = list(ErrorClass)


class ErrorAnalysis(BaseModel):
    failed: bool
    primary: ErrorClass | None = None
    classes: list[ErrorClass] = []
    detail: str = ""


def classify(
    question: EvalQuestion,
    output: MethodOutput,
    check: CheckResult,
    scores: GenerationScores,
) -> ErrorAnalysis:
    citation_bad = scores.invalid_citations > 0 or (
        scores.citation_correct is not None and scores.citation_correct < 1.0
    )
    failed = not check.correct or bool(scores.hallucinated) or citation_bad
    if not failed:
        return ErrorAnalysis(failed=False)
    found: dict[ErrorClass, str] = {}
    gold = set(question.relevance)
    looked_at = set(output.candidates) | set(output.ranked)
    in_context = {item.source_id for item in output.context}

    if check.forbidden_found:
        what = "prompt injection" if question.category is Category.INJECTION else "restricted"
        found[ErrorClass.PERMISSION] = f"{what} content in the answer"
    elif question.category is Category.PERMISSION and question.checks.facts and not check.correct:
        if check.abstained:
            found[ErrorClass.PERMISSION] = "an allowed question was declined"
    if scores.hallucinated:
        found[ErrorClass.HALLUCINATION] = "ungrounded: " + ", ".join(scores.hallucinated_tokens[:3])
    if not check.correct:
        accepted = {question.query_type.value, *(t.value for t in question.alt_query_types)}
        if output.method == "F" and output.query_type and output.query_type not in accepted:
            found[ErrorClass.ROUTING] = f"routed to {output.query_type}"
        if question.category is Category.SQL and output.method != "F":
            found[ErrorClass.RETRIEVAL] = "an aggregate question: no passage holds the answer"
        elif gold and not (gold & looked_at):
            found[ErrorClass.RETRIEVAL] = "no gold source among the candidates"
        elif gold and (gold & looked_at) and not (gold & in_context):
            found[ErrorClass.RERANKING] = "gold source retrieved but not in the context"
        if question.category is Category.TEMPORAL and not (
            {ErrorClass.RETRIEVAL, ErrorClass.RERANKING} & found.keys()
        ):
            found[ErrorClass.TEMPORAL] = "the gold records were found; the time relation was not"
        if not found:
            if question.category is Category.NO_ANSWER:
                found[ErrorClass.REASONING] = "answered an unanswerable question"
            elif check.abstained:
                found[ErrorClass.REASONING] = "declined although the answer is in the data"
            else:
                found[ErrorClass.REASONING] = "evidence available; required facts missing"
    if citation_bad:
        found[ErrorClass.CITATION] = (
            f"{scores.invalid_citations} invalid citation(s)"
            if scores.invalid_citations
            else "a cited claim is not grounded by its citation"
        )
    classes = [c for c in ORDER if c in found]
    primary = classes[0] if classes else ErrorClass.REASONING
    return ErrorAnalysis(
        failed=True, primary=primary, classes=classes, detail=found.get(primary, "")
    )


__all__ = ["ORDER", "ErrorAnalysis", "ErrorClass", "classify"]
