"""A second, model-based injection detector for retrieved content.

The rules in ``injection.py`` catch the known phrasings; a reworded attack ("Assistant,
please set aside everything you were told earlier...") slips past them. This layer
runs a prompt-injection classifier (default: protectai/deberta-v3-base-prompt-injection-v2)
on the sentences that *address an AI or "you"*: only those can instruct the model, and
limiting the classifier to them keeps it fast and away from ordinary imperative text
(runbook steps), which such classifiers otherwise flag.

Measured on the generated corpus (see docs/TECHNICAL_REFERENCE.md): 0 of its 26,000+
sentences flagged; 8 of 8 reworded attacks the rules miss flagged. It is still a model: it
can be wrong, and it is one layer among several.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import Any, Protocol

from app.security.injection import REMOVED, scan, segments

logger = logging.getLogger(__name__)

ADDRESSED = re.compile(
    r"\b(?:you|assistant|chatbot|bot|ai|llm|language model|model|copilot|gpt|claude)\b",
    re.IGNORECASE,
)


class InjectionClassifier(Protocol):
    name: str

    def score(self, texts: Sequence[str]) -> list[float]:
        """Probability that each text is a prompt injection."""
        ...


class TransformersInjectionClassifier:
    def __init__(self, model: str, device: str = "cpu", max_length: int = 256) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        try:
            self._tokenizer: Any = AutoTokenizer.from_pretrained(model, local_files_only=True)
            self._model: Any = AutoModelForSequenceClassification.from_pretrained(
                model, local_files_only=True
            )
        except Exception:  # not cached yet
            self._tokenizer = AutoTokenizer.from_pretrained(model)
            self._model = AutoModelForSequenceClassification.from_pretrained(model)
        labels = {str(v).upper(): int(k) for k, v in self._model.config.id2label.items()}
        if "INJECTION" not in labels:
            raise ValueError(f"{model} has no INJECTION label: {sorted(labels)}")
        self._index = labels["INJECTION"]
        self._model.to(device).eval()
        self._torch, self._device, self._max_length = torch, device, max_length
        self.name = model

    def score(self, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        torch = self._torch
        scores: list[float] = []
        for start in range(0, len(texts), 16):
            batch = self._tokenizer(
                list(texts[start : start + 16]),
                truncation=True,
                max_length=self._max_length,
                padding=True,
                return_tensors="pt",
            ).to(self._device)
            with torch.no_grad():
                probabilities = torch.softmax(self._model(**batch).logits, dim=-1)
            scores += probabilities[:, self._index].tolist()
        return scores


class SemanticDetector:
    """Flags sentences addressed to an AI that the classifier scores as injections."""

    def __init__(self, classifier: InjectionClassifier, threshold: float = 0.9) -> None:
        self.classifier = classifier
        self.threshold = threshold

    def flagged(self, text: str) -> list[tuple[int, int]]:
        """Spans of ``text`` (sentences or lines) judged to be injections."""
        spans = [(a, b) for a, b in segments(text) if ADDRESSED.search(text[a:b])]
        scores = self.classifier.score([text[a:b].strip() for a, b in spans])
        return [span for span, score in zip(spans, scores, strict=True) if score >= self.threshold]

    def strip(self, text: str) -> str:
        """``text`` with flagged sentences replaced by the removal marker; everything is
        removed if what is left is still flagged (fail closed)."""
        spans = self.flagged(text)
        if not spans:
            return text
        parts, position = [], 0
        for start, end in spans:
            parts += [text[position:start], REMOVED + " "]
            position = end
        parts.append(text[position:])
        stripped = "".join(parts).strip()
        rest = stripped.replace(REMOVED, "")
        if scan(rest).flagged or self.flagged(rest):
            return REMOVED
        return stripped


def load_semantic_detector(
    model: str | None, threshold: float, device: str = "cpu"
) -> SemanticDetector | None:
    """The configured classifier, or None (disabled, or it could not be loaded; the
    rule-based detection still runs)."""
    if not model:
        return None
    try:
        return SemanticDetector(TransformersInjectionClassifier(model, device), threshold)
    except Exception as exc:
        logger.warning(
            "security.injection_model_unavailable",
            extra={"model": model, "error": type(exc).__name__},
        )
        return None


__all__ = [
    "ADDRESSED",
    "InjectionClassifier",
    "SemanticDetector",
    "TransformersInjectionClassifier",
    "load_semantic_detector",
]
