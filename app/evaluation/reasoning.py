"""Scoring for the Phase 9 benchmarks (temporal and multi-hop questions).

The benchmarks (``data/evaluation/{temporal,multihop}_benchmark.jsonl``) carry gold
answers computed from the raw dataset by ``scripts/build_reasoning_benchmark.py``. Here
only the agent's final answer text is scored, against those frozen answers:

- identifiers named in the question are ignored (an answer may repeat its anchor);
- ``first``: the first identifier of the target kind in the answer must be the gold one
  (so listing many deployments does not count as finding "the" one);
- ``set``: the identifiers of the target kind must equal the gold set (precision and
  recall are reported too);
- multi-hop: each hop is checked on its own (deployment, PR and parent incident must be
  the first of their kind; the commit by its 7-character prefix; the file by its full
  path; the change by one of the changed lines, whitespace-normalised); a question is
  correct when every hop is.
- stress-set hops: ``files`` (every path named), ``key`` (one of the changed setting
  names), ``author`` (the author's login), ``no_cause`` (the answer says no deployment
  or change is recorded as the cause, and names none as the cause).
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

PATTERNS = {
    "deployment": re.compile(r"\bDEP-\d{4}\b"),
    "incident": re.compile(r"\bINC-\d{4}\b"),
    "pull_request": re.compile(r"\bPR-\d{3,5}\b"),
    "parent_incident": re.compile(r"\bINC-\d{4}\b"),
    "version": re.compile(r"\bv\d+(?:\.\d+){1,3}\b"),
}


class TemporalQuestion(BaseModel):
    id: str
    relation: str
    question: str
    target: Literal["deployment", "incident", "version"]
    expected: list[str]
    match: Literal["first", "set"]


class MultiHopQuestion(BaseModel):
    id: str
    kind: str
    question: str
    expected: dict[str, Any]
    hops: list[str]


class Scored(BaseModel):
    id: str
    group: str  # relation or kind
    question: str
    correct: bool
    detail: dict[str, Any]
    answer: str
    tools: list[str]


def load_questions(path: Path, model: type[BaseModel]) -> list[Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [model.model_validate(json.loads(line)) for line in lines if line.strip()]


def mentioned(kind: str, text: str, question: str) -> list[str]:
    """Identifiers of ``kind`` in ``text``, in order, without those the question names."""
    pattern = PATTERNS[kind]
    asked = set(pattern.findall(question))
    return [m for m in dict.fromkeys(pattern.findall(text)) if m not in asked]


def _normal(text: str) -> str:
    return " ".join(text.split())


def score_temporal(q: TemporalQuestion, answer: str) -> tuple[bool, dict[str, Any]]:
    found = mentioned(q.target, answer, q.question)
    if q.match == "first":
        return bool(found) and found[0] == q.expected[0], {"predicted": found[:3]}
    predicted, expected = set(found), set(q.expected)
    hit = len(predicted & expected)
    precision = hit / len(predicted) if predicted else 0.0
    recall = hit / len(expected) if expected else 1.0
    return predicted == expected, {
        "predicted": sorted(predicted),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
    }


_CAUSAL = re.compile(r"caus|responsib|behind|root cause|introduc", re.I)
_NEGATION = re.compile(r"\b(?:no|not|none|without)\b", re.I)
_NO_CAUSE = re.compile(
    r"\bno (?:deployment|change|code change|commit)\b.{0,60}\b(?:recorded|found|known|caus)", re.I
)


def names_a_cause(answer: str) -> bool:
    """Whether a sentence of ``answer`` names a deployment as a cause, without negation."""
    for sentence in re.split(r"(?<=[.!?])\s+|\n", answer):
        if (
            PATTERNS["deployment"].search(sentence)
            and _CAUSAL.search(sentence)
            and not _NEGATION.search(sentence)
        ):
            return True
    return False


def score_multihop(q: MultiHopQuestion, answer: str) -> tuple[bool, dict[str, Any]]:
    hops: dict[str, bool] = {}
    text = _normal(answer)
    for hop in q.hops:
        gold = q.expected[hop]
        if hop in {"deployment", "pull_request", "parent_incident"}:
            found = mentioned(hop, answer, q.question)
            hops[hop] = bool(found) and found[0] == gold
        elif hop == "commit":
            hops[hop] = gold.lower() in answer.lower()
        elif hop == "file":
            hops[hop] = gold in answer
        elif hop == "change":
            hops[hop] = any(_normal(line) in text for line in gold)
        elif hop == "incidents":
            hops[hop] = set(mentioned("incident", answer, q.question)) == set(gold)
        elif hop == "files":
            hops[hop] = all(path in answer for path in gold)
        elif hop == "key":
            hops[hop] = any(re.search(rf"\b{re.escape(key)}\b", answer) for key in gold)
        elif hop == "author":
            hops[hop] = gold in answer
        elif hop == "no_cause":
            hops[hop] = bool(_NO_CAUSE.search(answer)) and not names_a_cause(answer)
    return all(hops.values()), {"hops": hops}


def summarize(results: list[Scored]) -> dict[str, Any]:
    groups: dict[str, list[Scored]] = defaultdict(list)
    for r in results:
        groups[r.group].append(r)
    summary: dict[str, Any] = {
        "questions": len(results),
        "correct": sum(r.correct for r in results),
        "accuracy": round(sum(r.correct for r in results) / len(results), 3) if results else 0.0,
        "by_group": {
            g: {"questions": len(rs), "correct": sum(r.correct for r in rs)}
            for g, rs in sorted(groups.items())
        },
    }
    hop_totals: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        for hop, ok in r.detail.get("hops", {}).items():
            hop_totals[hop].append(ok)
    if hop_totals:
        summary["by_hop"] = {
            hop: {"checked": len(v), "correct": sum(v)} for hop, v in sorted(hop_totals.items())
        }
    return summary


__all__ = [
    "MultiHopQuestion",
    "Scored",
    "TemporalQuestion",
    "load_questions",
    "mentioned",
    "names_a_cause",
    "score_multihop",
    "score_temporal",
    "summarize",
]
