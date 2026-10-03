"""Recommended next steps for an investigation, derived from the evidence.

Each recommendation is built from one screened evidence item, and cites it, by rules:

- a runbook in the evidence: its first mitigation steps;
- a deployment recorded as the cause of an incident: review it (and, if it still
  succeeded, whether a rollback applies);
- a traced code change: inspect the file and the pull request;
- log evidence: look at the most frequent line;
- sources that disagree on a setting: confirm the value in effect;
- no answer: the evidence that would help (from ``answerability``).

Only items the answer cites are used when the answer cites any, so the steps follow what
the answer rests on. Nothing is generated: a step says what to look at, never what the
cause is. At most ``LIMIT`` steps.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from app.agents.state import AgentState, EvidenceItem, EvidenceKind

LIMIT = 5
Kind = Literal["runbook", "change", "code", "logs", "conflict", "gather"]


class Recommendation(BaseModel):
    kind: Kind
    text: str
    sources: list[str] = []  # evidence labels, as in the answer's citations


def _label(item: EvidenceItem) -> list[str]:
    return [item.label] if item.label else []


def recommend(state: AgentState) -> list[Recommendation]:
    if state.synthesis_method == "none" or not state.citations:
        return [Recommendation(kind="gather", text=idea) for idea in state.suggested_evidence][
            :LIMIT
        ]
    cited = {c.label for c in state.citations}
    items = [i for i in state.reranked_evidence if i.label in cited] or state.reranked_evidence
    out: list[Recommendation] = []
    for item in items:
        f = item.facts
        if item.kind is EvidenceKind.RUNBOOK and f.get("mitigation"):
            steps = "; ".join(s.strip() for s in f["mitigation"].split(";")[:3] if s.strip())
            out.append(
                Recommendation(
                    kind="runbook",
                    text=f"Follow runbook {f['id']} ({f['title']}): {steps}.",
                    sources=_label(item),
                )
            )
        elif item.kind is EvidenceKind.DEPLOYMENT and "id" in f:
            causes = [
                part.split(" ")[0]
                for part in f.get("incidents", "").split("; ")
                if "root_cause" in part
            ]
            if f.get("cause_of"):  # a traced chain: the item names the incident it caused
                causes = [f["cause_of"]]
            if not causes:
                continue
            status = f.get("status", "")
            advice = (
                "; if it is still live, check whether a rollback applies"
                if status == "succeeded"
                else f" (status {status})"
            )
            out.append(
                Recommendation(
                    kind="change",
                    text=f"Review {f['id']} ({f['service']} {f['version']}), recorded as the "
                    f"cause of {', '.join(causes)}{advice}.",
                    sources=_label(item),
                )
            )
        elif item.kind is EvidenceKind.CODE and f.get("hop") in {"file", "fix_file"}:
            out.append(
                Recommendation(
                    kind="code",
                    text=f"Inspect the change to {f['path']} in {f['pull_request']} "
                    f"(commit {f['commit']}).",
                    sources=_label(item),
                )
            )
        elif item.kind is EvidenceKind.LOGS and f.get("top"):
            out.append(
                Recommendation(
                    kind="logs",
                    text=f"Check the most frequent log line: {f['top']}.",
                    sources=_label(item),
                )
            )
    for conflict in state.conflicts:
        labels = [label for v in conflict.values for label in v.labels]
        preferred = (
            f"; the newest statement in effect gives {conflict.preferred}"
            if conflict.preferred is not None
            else ""
        )
        out.append(
            Recommendation(
                kind="conflict",
                text=f"Confirm the production value of {conflict.setting}: sources "
                f"disagree{preferred}.",
                sources=list(dict.fromkeys(labels)),
            )
        )
    unique: dict[str, Recommendation] = {}
    for r in out:
        unique.setdefault(r.text, r)
    return list(unique.values())[:LIMIT]


__all__ = ["LIMIT", "Recommendation", "recommend"]
