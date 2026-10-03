"""Evidence package, claim verification and citations.

Before generation, the evidence is frozen into an ``EvidencePackage``: the query
plus, per item, the source id, source type, content, a [0, 1] relevance score and
the metadata citations need. The synthesizer sees only this package.

After generation the ``Verifier`` checks the answer claim by claim.

1. **Extract claims.** The answer is split into sentences. Limitation statements
   ("the evidence does not say ...") are kept as they are. Statements about the
   model's own confidence are dropped: confidence is computed, never asserted.
   Every other sentence is a factual claim.
2. **Find support.** Each claim is checked against every package item, not only
   the ones it cites:
   - every identifier, version, time and number in the claim must appear among the
     item's values (exact token match);
   - the share of the claim's content terms found in the item must be high;
   - when an NLI model is configured, its entailment probability on the item's most
     relevant sentences counts as well.
3. **Label.** A claim is SUPPORTED if an item (or the combination of the items it
   cites) supports it, PARTIALLY_SUPPORTED if the support is incomplete, and
   UNSUPPORTED otherwise. Items stating the same thing with a different value, or
   that the NLI model finds contradicting it, are recorded as conflicting.
4. **Act.** Supported claims keep citations to the evidence that supports them. A
   citation that does not support the claim is replaced; a supported claim with no
   citation gets one. Partially supported claims are marked. Unsupported claims are
   removed, hedged or labelled (``VERIFY_UNSUPPORTED_POLICY``); none is ever
   presented as fact.

Citations can only point at package items, so none can be invented.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.agents.state import (
    Citation,
    ClaimLabel,
    ClaimVerdict,
    EvidenceItem,
    EvidencePackage,
    PackagedEvidence,
)
from app.config import VerificationSettings
from app.rag.retrieval.tokenizer import Tokenizer

logger = logging.getLogger(__name__)

_VALUE = re.compile(
    r"\b(?:INC|DEP|PR|RB|DOC|PM|CF)-\d+\b|\bv\d+(?:\.\d+)+\b|\b\d{1,2}:\d{2}\b|\b\d+(?:\.\d+)?%?",
    re.I,
)
_RECORD_ID = re.compile(r"\b(?:INC|DEP|PR|RB|DOC|PM|CF)-\d+\b", re.I)
_PATH = re.compile(r"\S*/\S*|\S+\.(?:py|md|ya?ml|json|toml|txt|cfg|sql)\b")
_LABEL = re.compile(r"\s*\[(E\d+)\]")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[(\"'`])")
_META = re.compile(
    r"\b(?:could not|couldn't|cannot|can't|unable to|no evidence|not enough evidence|"
    r"insufficient|sufficient evidence|not found|does not (?:say|mention|contain|answer|state)|"
    r"doesn't (?:say|mention|state)|outside what|additional evidence|not confirmed)\b",
    re.I,
)
_MODEL_CONFIDENCE = re.compile(
    r"^\W*(?:overall\s+)?confidence\b|\bI(?:'m| am)\s+(?:\w+\s+)?(?:confident|certain|sure)\b"
    r"|\bconfidence (?:level|score)\b|\b(?:high|medium|low) confidence\b",
    re.I,
)


def _quote(claim: str, limit: int = 80) -> str:
    """The claim shortened at a word boundary, with an ellipsis when cut.

    A plain slice can stop inside a value ("payment-service v2" for v2.6.4), which reads
    as a different value.
    """
    claim = claim.rstrip().rstrip(".")
    if len(claim) <= limit:
        return claim
    return claim[: limit + 1].rsplit(" ", 1)[0].rstrip(",;:") + "…"


# --- package ---------------------------------------------------------------------------


def build_package(query: str, evidence: Sequence[EvidenceItem]) -> EvidencePackage:
    return EvidencePackage(
        query=query,
        evidence=[
            PackagedEvidence(
                label=item.label or f"E{n}",
                source_id=item.source_id,
                source_type=item.kind,
                title=item.title,
                content=item.text,
                relevance_score=round(min(1.0, max(0.0, item.relevance)), 4),
                timestamp=item.timestamp,
                section=item.section,
                file_path=item.location,
                chunk_id=item.chunk_id,
                trust=item.trust,
                security_flags=item.security_flags,
            )
            for n, item in enumerate(evidence, 1)
        ],
    )


def citations_for(labels: Sequence[str], package: EvidencePackage) -> list[Citation]:
    entries = package.by_label()
    return [
        Citation(
            label=label,
            source_id=entries[label].source_id,
            title=entries[label].title,
            source_type=entries[label].source_type,
            timestamp=entries[label].timestamp,
            section=entries[label].section,
            file_path=entries[label].file_path,
            relevance=entries[label].relevance_score,
            chunk_id=entries[label].chunk_id,
            trust=entries[label].trust,
        )
        for label in dict.fromkeys(labels)
        if label in entries
    ]


# --- NLI -----------------------------------------------------------------------------------


class NLIScorer(Protocol):
    name: str

    def predict(self, pairs: Sequence[tuple[str, str]]) -> list[dict[str, float]]:
        """For (premise, hypothesis) pairs: probabilities of entailment, contradiction
        and neutral."""
        ...


class CrossEncoderNLI:
    """An NLI cross-encoder (e.g. cross-encoder/nli-deberta-v3-xsmall)."""

    def __init__(self, model: str, device: str = "cpu") -> None:
        from sentence_transformers import CrossEncoder

        try:
            self._model: Any = CrossEncoder(model, device=device, local_files_only=True)
        except Exception:
            self._model = CrossEncoder(model, device=device)
        id2label = getattr(self._model.model.config, "id2label", {}) or {}
        self._labels = [str(id2label[i]).lower() for i in sorted(id2label)]
        if not {"entailment", "contradiction"} <= set(self._labels):
            raise ValueError(f"{model} is not a 3-way NLI model: labels {self._labels}")
        self.name = model

    def predict(self, pairs: Sequence[tuple[str, str]]) -> list[dict[str, float]]:
        if not pairs:
            return []
        probabilities = self._model.predict(
            [list(p) for p in pairs], apply_softmax=True, show_progress_bar=False
        )
        return [dict(zip(self._labels, map(float, row), strict=True)) for row in probabilities]


def load_nli(settings: VerificationSettings) -> NLIScorer | None:
    """The configured NLI model, or None (disabled, or it could not be loaded)."""
    if not settings.nli_model:
        return None
    try:
        return CrossEncoderNLI(settings.nli_model, settings.device)
    except Exception as exc:
        logger.warning(
            "verification.nli_unavailable",
            extra={"model": settings.nli_model, "error": type(exc).__name__},
        )
        return None


# --- claims ----------------------------------------------------------------------------------


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.split(" ".join(text.split())) if s.strip()]


def values(text: str) -> set[str]:
    return {v.lower() for v in _VALUE.findall(_LABEL.sub(" ", text))}


def slots(found: set[str]) -> dict[str, set[str]]:
    """Values grouped by kind (record prefix, version, time, number), so that only
    values of the same kind are compared when looking for conflicts."""
    grouped: dict[str, set[str]] = {}
    for value in found:
        if _RECORD_ID.fullmatch(value):
            kind = value.split("-")[0]
        elif value.startswith("v"):
            kind = "version"
        elif ":" in value:
            kind = "time"
        else:
            kind = "number"
        grouped.setdefault(kind, set()).add(value)
    return grouped


@dataclass
class _Features:
    label: str
    recall: float
    values_ok: bool
    window: str
    entailment: float = 0.0
    contradiction: float = 0.0
    frame_conflict: bool = False
    same_subject: bool = False  # may this item contradict the claim at all?

    @property
    def lexical(self) -> float:
        return self.recall if self.values_ok else self.recall * 0.5


@dataclass
class VerificationResult:
    text: str
    claims: list[ClaimVerdict]
    citations: list[str]  # labels, in order of first use
    notes: list[str] = field(default_factory=list)  # e.g. sources disagreeing
    dropped_confidence_statements: int = 0

    @property
    def counts(self) -> dict[str, int]:
        return {label.value: sum(c.label is label for c in self.claims) for label in ClaimLabel}


class Verifier:
    def __init__(
        self,
        settings: VerificationSettings,
        nli: NLIScorer | None = None,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        self.settings = settings
        self.nli = nli
        self.tokenizer = tokenizer or Tokenizer()

    @property
    def method(self) -> str:
        return f"lexical + NLI ({self.nli.name})" if self.nli else "lexical"

    def _terms(self, text: str) -> set[str]:
        return {t for t in self.tokenizer(_LABEL.sub(" ", text)) if not any(c.isdigit() for c in t)}

    def _window(self, claim_terms: set[str], content: str) -> str:
        units = [s for line in content.splitlines() for s in split_sentences(line)]
        scored = sorted(
            ((len(claim_terms & self._terms(u)), -i, u) for i, u in enumerate(units)), reverse=True
        )
        best = [u for score, _, u in scored[:2] if score > 0] or units[:1]
        return " ".join(best)[:600]

    def verify(self, answer: str, package: EvidencePackage) -> VerificationResult:
        entries = package.by_label()
        # An item's metadata (title, section, file path) is evidence too: answers restate it.
        texts = {
            e.label: "\n".join(filter(None, [e.title, e.section, e.file_path, e.content]))
            for e in package.evidence
        }
        item_terms = {label: self._terms(text) for label, text in texts.items()}
        item_values = {label: values(text) for label, text in texts.items()}
        result = VerificationResult(text="", claims=[], citations=[])
        out_lines: list[str] = []
        for line in answer.splitlines():
            out_sentences: list[str] = []
            for sentence in split_sentences(line):
                cited = [m for m in _LABEL.findall(sentence)]
                claim = _LABEL.sub("", sentence).strip()
                if not claim:
                    continue
                if _MODEL_CONFIDENCE.search(claim):
                    result.dropped_confidence_statements += 1
                    continue
                if _META.search(claim) or not package.evidence:
                    if not _META.search(claim):  # a factual sentence without any evidence
                        result.claims.append(self._unsupported(claim, cited, "no evidence"))
                        out = self._apply(result.claims[-1], claim)
                    else:
                        out = claim
                    if out:
                        out_sentences.append(out)
                    continue
                verdict = self._judge(claim, cited, entries, item_terms, item_values)
                result.claims.append(verdict)
                out = self._apply(verdict, claim)
                if out:
                    out_sentences.append(out)
                    for label in verdict.supporting:
                        if label not in result.citations and verdict.action != "removed":
                            result.citations.append(label)
                if verdict.conflicting and verdict.action != "removed":
                    others = "".join(f"[{c}]" for c in verdict.conflicting)
                    result.notes.append(
                        f'Sources disagree about "{_quote(claim)}": {others} state(s) otherwise.'
                    )
            if out_sentences:
                out_lines.append(" ".join(out_sentences))
        result.text = "\n".join(out_lines)
        return result

    # --- one claim ------------------------------------------------------------------------

    def _judge(
        self,
        claim: str,
        cited: list[str],
        entries: dict[str, PackagedEvidence],
        item_terms: dict[str, set[str]],
        item_values: dict[str, set[str]],
    ) -> ClaimVerdict:
        s = self.settings
        claim_terms = self._terms(claim)
        claim_values = values(claim)
        claim_ids = {v for v in claim_values if _RECORD_ID.fullmatch(v)}
        propositional = len(self._terms(_PATH.sub(" ", claim))) >= 4
        unknown = [c for c in cited if c not in entries]
        valid_cited = [c for c in dict.fromkeys(cited) if c in entries]
        features: dict[str, _Features] = {}
        for label, entry in entries.items():
            recall = len(claim_terms & item_terms[label]) / len(claim_terms) if claim_terms else 1.0
            window = self._window(claim_terms, entry.content)
            window_terms = self._terms(window)
            overlap = len(claim_terms & window_terms) / len(claim_terms) if claim_terms else 0.0
            features[label] = _Features(
                label=label,
                recall=recall,
                values_ok=claim_values <= item_values[label],
                window=window,
                # Only an item about the claim's own subject can contradict it: the record the
                # claim names (if any), a claim that states something beyond file paths, and a
                # passage sharing most of its words. NLI models otherwise call "a different
                # file" or "another incident" a contradiction.
                same_subject=propositional
                and (not claim_ids or entry.source_id.lower() in claim_ids)
                and overlap >= s.conflict_overlap,
            )
        if self.nli is not None:
            candidates = sorted(
                features.values(), key=lambda f: (f.label not in valid_cited, -f.lexical)
            )[: max(4, len(valid_cited))]
            probabilities = self.nli.predict([(f.window, claim) for f in candidates])
            for f, p in zip(candidates, probabilities, strict=True):
                f.entailment = p.get("entailment", 0.0)
                f.contradiction = p.get("contradiction", 0.0)

        def contradicted(f: _Features) -> bool:
            return f.same_subject and f.contradiction >= s.contradiction_threshold

        def supports(f: _Features) -> bool:
            if contradicted(f):  # NLI veto: same subject and words, opposite meaning
                return False
            return f.values_ok and (
                f.recall >= s.supported_threshold or f.entailment >= s.entailment_threshold
            )

        def partial(f: _Features) -> bool:
            # Some of the claim's facts (ids, numbers, times) must be there: matching words
            # around an absent value ("pool size is 20" vs no 20 anywhere) is not support.
            shares_values = not claim_values or bool(claim_values & item_values[f.label])
            return shares_values and (
                f.recall >= s.partial_threshold or f.entailment >= s.partial_entailment_threshold
            )

        supporting = [f for f in features.values() if supports(f)]
        # A claim spanning several cited items: judge it against their union.
        if not supporting and len(valid_cited) >= 2:
            union_terms = set().union(*(item_terms[c] for c in valid_cited))
            union_values = set().union(*(item_values[c] for c in valid_cited))
            recall = len(claim_terms & union_terms) / len(claim_terms) if claim_terms else 1.0
            if claim_values <= union_values and recall >= s.supported_threshold:
                supporting = [features[c] for c in valid_cited]
        claim_slots = slots(claim_values)
        for f in features.values():  # same statement, different value of the same kind
            if f in supporting or not claim_values or not f.same_subject:
                continue
            for unit in split_sentences(f.window):
                unit_terms, unit_values = self._terms(unit), values(unit)
                unit_ids = {v for v in unit_values if _RECORD_ID.fullmatch(v)}
                if len(claim_terms & unit_terms) / len(claim_terms) < s.conflict_overlap:
                    continue
                if not claim_ids <= unit_ids:  # about a different record
                    continue
                unit_slots = slots(unit_values)
                shared = [kind for kind in claim_slots if kind in unit_slots]
                # Conservative: the unit restates the claim with a different value in every
                # kind both mention ("pool size is 20" vs "pool size is 5"). Mixed sentences
                # (same ids, other times) are left to the NLI model.
                if shared and all(not claim_slots[k] & unit_slots[k] for k in shared):
                    f.frame_conflict = True
        conflicting = [
            f.label
            for f in features.values()
            if f not in supporting and (f.frame_conflict or contradicted(f))
        ]
        if supporting:
            label = ClaimLabel.SUPPORTED
            chosen = sorted(
                supporting, key=lambda f: (f.label not in valid_cited, -f.lexical, -f.entailment)
            )
        else:
            partials = [f for f in features.values() if partial(f) and f.label not in conflicting]
            chosen = sorted(partials, key=lambda f: (f.label not in valid_cited, -f.lexical))
            label = ClaimLabel.PARTIALLY_SUPPORTED if chosen else ClaimLabel.UNSUPPORTED
        best = max((max(f.lexical, f.entailment) for f in features.values()), default=0.0)
        cited_ok = [f.label for f in chosen if f.label in valid_cited]
        kept = cited_ok[:3] or [f.label for f in chosen[:2]]  # keep the answer's own citations
        reason = ""
        if unknown:
            reason = f"cited missing evidence {', '.join(unknown)}"
        elif (
            label is not ClaimLabel.UNSUPPORTED and valid_cited and not set(valid_cited) & set(kept)
        ):
            reason = f"citation corrected from {', '.join(valid_cited)}"
        elif label is not ClaimLabel.UNSUPPORTED and not valid_cited:
            reason = "citation added by verification"
        elif label is ClaimLabel.UNSUPPORTED:
            missing = (
                sorted(claim_values - set().union(*item_values.values())) if item_values else []
            )
            reason = (
                f"not in the evidence: {', '.join(missing)}"
                if missing
                else "contradicted by the evidence"
                if conflicting
                else "no evidence states this"
            )
        return ClaimVerdict(
            text=claim,
            label=label,
            cited=cited,
            supporting=kept if label is not ClaimLabel.UNSUPPORTED else [],
            conflicting=[c for c in conflicting if c not in kept],
            support_score=round(best, 3),
            action="",
            reason=reason,
        )

    def _unsupported(self, claim: str, cited: list[str], reason: str) -> ClaimVerdict:
        return ClaimVerdict(
            text=claim,
            label=ClaimLabel.UNSUPPORTED,
            cited=cited,
            supporting=[],
            support_score=0.0,
            action="",
            reason=reason,
        )

    def _apply(self, verdict: ClaimVerdict, claim: str) -> str:
        """The sentence as it appears in the final answer (empty when removed)."""
        body = claim.rstrip().rstrip(".")
        labels = "".join(f"[{c}]" for c in verdict.supporting)
        if verdict.label is ClaimLabel.SUPPORTED:
            verdict.action = "kept"
            return f"{body} {labels}."
        if verdict.label is ClaimLabel.PARTIALLY_SUPPORTED:
            policy = self.settings.partial_policy
            if policy == "keep":
                verdict.action = "kept"
                return f"{body} {labels}."
            if policy == "hedge":
                verdict.action = "hedged"
                return f"The evidence only partly supports this: {body} {labels}."
            verdict.action = "labelled"
            return f"{body} (partially supported) {labels}."
        policy = self.settings.unsupported_policy
        if policy == "hedge":
            verdict.action = "hedged"
            return f"Not confirmed by the available evidence: {body}."
        if policy == "label":
            verdict.action = "labelled"
            return f"{body} [unsupported]."
        verdict.action = "removed"
        return ""
