"""Phase 10 metrics: retrieval, and deterministic / model-based generation metrics.

Three families, reported separately and never mixed:

1. **Deterministic** (string and identifier matching; reproducible bit for bit)
   - retrieval: Recall@k, Precision@5, MRR, NDCG@10 over the ranked source records;
   - answer correctness: the question's checks (``eval_set.check_answer``);
   - faithfulness: share of answer claims *grounded* in the context: every checkable
     token of the claim (record ids, versions, commit hashes, file paths, setting
     names, numbers) occurs in the context, and at least half of its content words do;
   - hallucination rate: share of answers with at least one checkable token that is
     in neither the context nor the question;
   - citation correctness: share of cited claims whose citations all exist in the
     context and whose cited items ground the claim (as above);
   - context relevance: share of context items that are relevant (a gold source, or
     containing a gold fact).
2. **Model-based** (an NLI cross-encoder): NLI faithfulness, the share of claims
   entailed (p >= 0.5) by a cited item (or, uncited, by one of the first three items).
   The same model family is used inside the verifier of methods E and F, so this
   number is not independent for them; it is reported for comparison only.
3. **LLM-as-judge** (``judge.py``): runs only when a judge model is configured.

The deterministic measures are strict and literal: a correct paraphrase fails them, and
an extractive answer (copied from the context) scores high on faithfulness by
construction. The report says so next to the numbers.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from datetime import UTC, datetime

from pydantic import BaseModel

from app.agents.verification import NLIScorer
from app.evaluation.eval_set import EvalQuestion, contains

# --- retrieval --------------------------------------------------------------------------------

KS = (1, 5, 10)


class RetrievalScores(BaseModel):
    recall: dict[int, float]
    hit: dict[int, float]
    precision_5: float
    reciprocal_rank: float
    ndcg_10: float
    first_relevant_rank: int | None


def retrieval_scores(ranked: Sequence[str], relevance: dict[str, int]) -> RetrievalScores:
    """Scores for one ranked list of source records (one entry per retrieved chunk)."""
    relevant = set(relevance)
    recall, hit = {}, {}
    for k in KS:
        found = relevant & set(ranked[:k])
        recall[k] = len(found) / min(len(relevant), k)
        hit[k] = 1.0 if found else 0.0
    top5 = list(ranked[:5])
    precision = sum(doc in relevant for doc in top5) / 5
    first = next((r for r, doc in enumerate(ranked[:10], 1) if doc in relevant), None)
    seen: set[str] = set()
    dcg = 0.0
    for rank, doc in enumerate(ranked[:10], 1):
        if doc in relevant and doc not in seen:
            seen.add(doc)
            dcg += (2 ** relevance[doc] - 1) / math.log2(rank + 1)
    ideal_grades = sorted(relevance.values(), reverse=True)[:10]
    idcg = sum((2**g - 1) / math.log2(r + 1) for r, g in enumerate(ideal_grades, 1))
    return RetrievalScores(
        recall=recall,
        hit=hit,
        precision_5=precision,
        reciprocal_rank=1 / first if first else 0.0,
        ndcg_10=dcg / idcg if idcg else 0.0,
        first_relevant_rank=first,
    )


# --- claims and grounding ---------------------------------------------------------------------

CHECKABLE = re.compile(
    r"\b(?:INC|DEP|PR|CF|DOC|RB|PM|RPT|POL)-\d{3,5}\b"  # record ids
    r"|\bv\d+(?:\.\d+){1,3}\b"  # versions
    r"|\b(?=[0-9a-f]*[a-f])(?=[0-9a-f]*\d)[0-9a-f]{7,40}\b"  # commit hashes
    r"|[\w./-]+\.(?:py|ya?ml|md|txt|toml|json|sql|cfg|ini)\b"  # file paths
    r"|\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b"  # setting / constant names
    r"|(?<![\w.])\d+(?:\.\d+)?(?![\w.])"  # numbers
)
_LABEL = re.compile(r"\[(E\d+)\]")
_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[(\"'`])|\n+")
_WORD = re.compile(r"[a-z][a-z0-9_-]{3,}")
STOPWORDS = frozenset(
    [
        "that",
        "this",
        "with",
        "from",
        "have",
        "were",
        "been",
        "which",
        "when",
        "where",
        "what",
        "there",
        "their",
        "they",
        "them",
        "then",
        "than",
        "into",
        "also",
        "only",
        "after",
        "before",
        "about",
        "more",
        "most",
        "some",
        "such",
        "very",
        "will",
        "would",
        "should",
        "could",
        "because",
        "while",
        "over",
        "under",
        "each",
        "other",
        "these",
        "those",
        "does",
        "done",
        "being",
    ]
)


class ContextItem(BaseModel):
    label: str
    source_id: str
    text: str  # content plus title, id, location and timestamp (what the reader saw)


def context_item(
    label: str,
    source_id: str,
    title: str,
    content: str,
    location: str | None = None,
    timestamp: datetime | None = None,
) -> ContextItem:
    when = ""
    if timestamp is not None:
        utc = timestamp.replace(tzinfo=UTC) if timestamp.tzinfo is None else timestamp
        when = f"{utc:%Y-%m-%d %H:%M} UTC {utc:%Y-%m-%d}"
    text = " ".join(part for part in (source_id, title, location or "", when, content) if part)
    return ContextItem(label=label, source_id=source_id, text=text)


class Claim(BaseModel):
    text: str
    labels: list[str]


def claims(answer: str) -> list[Claim]:
    out = []
    for part in _SPLIT.split(answer):
        labels = _LABEL.findall(part)
        text = _LABEL.sub("", part).strip(" .-")
        if len(text) >= 3:
            out.append(Claim(text=text, labels=labels))
    return out


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in STOPWORDS}


def ungrounded_tokens(text: str, context: str, question: str) -> list[str]:
    haystack = f"{context}\n{question}".lower()
    return [t for t in dict.fromkeys(CHECKABLE.findall(text)) if t.lower() not in haystack]


def grounded(text: str, context: str, question: str) -> bool:
    if ungrounded_tokens(text, context, question):
        return False
    words = _content_words(text)
    if not words:
        return True
    return len(words & _content_words(context)) / len(words) >= 0.5


# --- generation metrics ------------------------------------------------------------------------


class GenerationScores(BaseModel):
    claims: int
    faithfulness: float | None  # grounded claims / claims (None: no claims)
    hallucinated: bool | None  # None: nothing to check (abstention)
    hallucinated_tokens: list[str]
    cited_claims: int
    citation_correct: float | None  # correctly cited claims / cited claims
    invalid_citations: int  # labels not in the context
    context_relevance: float | None  # relevant context items / context items


def generation_scores(
    question: EvalQuestion, answer: str, context: Sequence[ContextItem], abstained: bool
) -> GenerationScores:
    joined = "\n".join(item.text for item in context)
    by_label = {item.label: item for item in context}
    items = claims(answer) if not abstained else []
    grounded_flags = [grounded(c.text, joined, question.question) for c in items]
    tokens = list(
        dict.fromkeys(
            t for c in items for t in ungrounded_tokens(c.text, joined, question.question)
        )
    )
    cited = [c for c in items if c.labels]
    invalid = sum(1 for c in cited for label in c.labels if label not in by_label)
    correct = 0
    for c in cited:
        if all(label in by_label for label in c.labels):
            cited_text = "\n".join(by_label[label].text for label in c.labels)
            correct += grounded(c.text, cited_text, question.question)
    relevance = None
    gold = set(question.relevance)
    facts = [alt for group in question.checks.facts for alt in group]
    if context and (gold or facts):
        relevant = [
            item.source_id in gold or any(contains(item.text, f) for f in facts) for item in context
        ]
        relevance = sum(relevant) / len(relevant)
    return GenerationScores(
        claims=len(items),
        faithfulness=sum(grounded_flags) / len(items) if items else None,
        hallucinated=bool(tokens) if items else None,
        hallucinated_tokens=tokens[:10],
        cited_claims=len(cited),
        citation_correct=correct / len(cited) if cited else None,
        invalid_citations=invalid,
        context_relevance=relevance,
    )


def nli_faithfulness(
    nli: NLIScorer,
    answer: str,
    context: Sequence[ContextItem],
    abstained: bool,
    max_chars: int = 1500,
) -> float | None:
    """Share of claims entailed (p >= 0.5) by a cited item, or by one of the first three
    context items when the claim cites nothing. None when there is nothing to score."""
    if abstained or not context:
        return None
    by_label = {item.label: item for item in context}
    pairs: list[tuple[int, str, str]] = []
    items = claims(answer)
    for n, c in enumerate(items):
        premises = [by_label[label] for label in c.labels if label in by_label] or list(context[:3])
        pairs += [(n, p.text[:max_chars], c.text) for p in premises]
    if not pairs:
        return None
    predictions = nli.predict([(premise, hypothesis) for _, premise, hypothesis in pairs])
    entailed = [False] * len(items)
    for (n, _, _), p in zip(pairs, predictions, strict=True):
        entailed[n] = entailed[n] or p.get("entailment", 0.0) >= 0.5
    return sum(entailed) / len(items) if items else None


__all__ = [
    "KS",
    "Claim",
    "ContextItem",
    "GenerationScores",
    "RetrievalScores",
    "claims",
    "context_item",
    "generation_scores",
    "grounded",
    "nli_faithfulness",
    "retrieval_scores",
    "ungrounded_tokens",
]
