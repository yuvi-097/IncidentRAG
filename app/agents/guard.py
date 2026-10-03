"""The agent's security stages: question screening, evidence screening, output validation.

Order in the pipeline (see ``graph.py``):

    question  -> screen_question   (injection check, secrets removed from the question)
    tools     -> (tools filter by the caller's grants inside their database queries)
    evidence  -> EvidenceScreen    (access re-check, injection quarantine, secret
                                    redaction, trust labels) -- before reranking,
                                    packaging or any model call
    answer    -> OutputGuard       (after claim verification: secrets, instruction-like
                                    or quarantined text, foreign links, unknown ids)

Every stage fails closed: if screening raises, the evidence is dropped, not passed on
unscreened; if output validation raises, the answer is withheld.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import Literal

from app.agents.references import ReferenceFilter
from app.agents.state import (
    AgentState,
    EvidenceItem,
    EvidenceKind,
    ScreenedSource,
)
from app.schemas.enums import AccessLevel, Resource, SourceTrust, ToolPermission
from app.security.injection import REMOVED, InjectionCategory, scan, strip_instructions
from app.security.injection_model import SemanticDetector
from app.security.policy import AccessPolicy
from app.security.principal import Principal
from app.security.secrets import REDACTED, redact_secrets

logger = logging.getLogger(__name__)

InjectionAction = Literal["quarantine", "redact"]
QueryInjectionAction = Literal["refuse", "flag"]

# The kind of data each evidence item is; SQL results need the SQL capability instead
# (the query itself only saw rows the caller may read).
EVIDENCE_RESOURCE: dict[EvidenceKind, Resource] = {
    EvidenceKind.DOCUMENT: Resource.DOCUMENTS,
    EvidenceKind.RUNBOOK: Resource.RUNBOOKS,
    EvidenceKind.POSTMORTEM: Resource.INCIDENTS,
    EvidenceKind.INCIDENT: Resource.INCIDENTS,
    EvidenceKind.DEPLOYMENT: Resource.DEPLOYMENTS,
    EvidenceKind.PULL_REQUEST: Resource.CODE,
    EvidenceKind.CODE: Resource.CODE,
    EvidenceKind.LOGS: Resource.LOGS,
}

REFUSAL = (
    "This question contains instructions aimed at the assistant ({categories}), so it "
    "was not processed. Ask the question on its own; access to data is decided by your "
    "role, not by the question."
)
WITHHELD = "The answer was withheld because it could not be validated."


def may_read_item(principal: Principal, item: EvidenceItem) -> bool:
    if item.kind is EvidenceKind.SQL_RESULT:
        return principal.can(ToolPermission.SQL_READ)
    if item.kind is EvidenceKind.TIMELINE:
        # Derived from several records: readable only if every one of them is.
        required = [part.split(":") for part in item.facts.get("requires", "").split(",") if part]
        return bool(required) and all(
            principal.may_read(Resource(resource), AccessLevel(level))
            for resource, level in required
        )
    resource = EVIDENCE_RESOURCE.get(item.kind)
    return resource is not None and principal.may_read(resource, item.access_level)


# --- question -----------------------------------------------------------------------


def screen_question(state: AgentState, action: QueryInjectionAction) -> bool:
    """Remove credentials from the question and check it for injection. Returns
    False when the question must not be processed."""
    query, secrets = redact_secrets(state.query)
    if secrets:
        state.query = query
        state.security.secrets_redacted += secrets
        state.limit("Credentials in the question were removed before processing.")
    result = scan(state.query)
    if not result.flagged:
        return True
    state.security.query_flags = result.categories
    logger.warning(
        "security.query_injection",
        extra={
            "user": state.user_id,
            "role": state.role,
            "categories": result.categories,
            "action": action,
        },
    )
    if action == "refuse":
        state.security.blocked = True
        state.final_answer = REFUSAL.format(categories=", ".join(result.categories))
        state.limit("The question was refused: it contains instruction-like text.")
        return False
    state.limit(
        "The question contains instruction-like text ("
        + ", ".join(result.categories)
        + "); it was treated as a question only, and access was not changed."
    )
    return True


# --- evidence -----------------------------------------------------------------------


class EvidenceScreen:
    """Screens every retrieved item before it can be ranked, packaged or sent to a model."""

    def __init__(
        self,
        policy: AccessPolicy,
        action: InjectionAction = "quarantine",
        references: ReferenceFilter | None = None,
        semantic: SemanticDetector | None = None,
    ) -> None:
        self.policy = policy
        self.action = action
        self.references = references
        self.semantic = semantic  # the model-based second detector, if configured

    def screen(self, state: AgentState, items: Sequence[EvidenceItem]) -> list[EvidenceItem]:
        kept: list[EvidenceItem] = []
        for item in items:
            if not may_read_item(state.principal, item):
                # Tools filter in their queries, so this should never happen.
                state.security.access_violations += 1
                logger.error(
                    "security.access_violation",
                    extra={
                        "user": state.user_id,
                        "role": state.role,
                        "source_id": item.source_id,
                        "kind": item.kind.value,
                        "access_level": item.access_level.value,
                    },
                )
                continue
            screened = self._screen_item(state, item)
            if screened is not None:
                kept.append(screened)
        if self.references is not None and kept:
            kept, removed = self.references.redact(state.principal, kept)
            state.security.references_redacted += removed
            if removed:
                state.limit(
                    f"{removed} reference(s) to records your role may not read were removed "
                    "from the evidence."
                )
        if state.security.access_violations:
            state.limit("Evidence the caller may not read was removed.")
        if state.security.quarantined:
            state.limit(
                f"{len(state.security.quarantined)} source(s) contained instructions aimed at "
                "the assistant and were excluded from the evidence."
            )
        if state.security.sanitized:
            state.limit(
                "Instruction-like text was removed from "
                f"{len(state.security.sanitized)} source(s); they are marked suspicious."
            )
        state.evidence_screened = True
        return kept

    def _screen_item(self, state: AgentState, item: EvidenceItem) -> EvidenceItem | None:
        texts = [item.title, item.text, *item.facts.values()]
        categories = scan("\n".join(texts)).categories
        semantic = self.semantic
        if not categories and semantic and any(semantic.flagged(text) for text in texts):
            categories = [InjectionCategory.SEMANTIC.value]
        trust = self.policy.trust_of(item.kind.value)
        update: dict[str, object] = {}
        if categories:
            record = ScreenedSource(source_id=item.source_id, kind=item.kind, categories=categories)
            logger.warning(
                "security.evidence_injection",
                extra={
                    "user": state.user_id,
                    "source_id": item.source_id,
                    "kind": item.kind.value,
                    "categories": categories,
                    "action": self.action,
                },
            )
            if self.action == "quarantine":
                state.security.quarantined.append(record)
                state.quarantined_text.append(item.text)
                return None
            state.security.sanitized.append(record)
            state.quarantined_text.append(item.text)
            trust = SourceTrust.SUSPICIOUS
            update["title"] = self._strip(item.title)
            update["text"] = self._strip(item.text)
            update["facts"] = {k: self._strip(v) for k, v in item.facts.items()}
            update["security_flags"] = tuple(categories)
        update["trust"] = trust
        item = item.model_copy(update=update)
        return self._redact(state, item)

    def _strip(self, text: str) -> str:
        text = strip_instructions(text)[0]
        return self.semantic.strip(text) if self.semantic is not None else text

    @staticmethod
    def _redact(state: AgentState, item: EvidenceItem) -> EvidenceItem:
        title, n_title = redact_secrets(item.title)
        text, n_text = redact_secrets(item.text)
        facts: dict[str, str] = {}
        n_facts = 0
        for key, value in item.facts.items():
            facts[key], n = redact_secrets(value)
            n_facts += n
        total = n_title + n_text + n_facts
        if not total:
            return item
        state.security.secrets_redacted += total
        logger.warning(
            "security.secret_redacted",
            extra={"source_id": item.source_id, "kind": item.kind.value, "count": total},
        )
        return item.model_copy(update={"title": title, "text": text, "facts": facts})


# --- answer -------------------------------------------------------------------------

_SENTENCE = re.compile(r"[^\n]*?(?:[.!?](?=\s|$)|\n|$)")
_URL = re.compile(r"https?://[^\s)\]>\"'`<]+")
_URL_TRAILING = ".,;:!?"
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_IDENTIFIER = re.compile(r"\b(?:INC|DEP|RB|DOC|PM|RPT|POL|CF)-\d{4}\b|\bPR-\d{3,5}\b")
_WORD = re.compile(r"[a-z0-9]+")
SHINGLE = 6


def _urls(text: str) -> set[str]:
    return {u.rstrip(_URL_TRAILING) for u in _URL.findall(text)}


_LABEL = re.compile(r"\s*\[E\d+\]")


def plain(text: str) -> str:
    """Text without citation labels, links or punctuation at the ends, for comparing a
    claim with the answer it came from."""
    text = _LABEL.sub("", _URL.sub("", text.replace("[link removed]", "")))
    return " ".join(text.split()).strip(" .")


def _shingles(text: str, size: int = SHINGLE) -> set[tuple[str, ...]]:
    words = _WORD.findall(text.lower())
    return {tuple(words[i : i + size]) for i in range(len(words) - size + 1)}


class OutputGuard:
    """Validates the final answer before it leaves the agent."""

    def __init__(self, system_prompt: str, canary: str | None = None) -> None:
        self.prompt_shingles = _shingles(system_prompt)
        self.canary = canary

    def check(self, state: AgentState) -> None:
        answer = state.final_answer
        if not answer:
            return
        package = state.evidence_package
        entries = package.evidence if package else []
        # An item's own id counts as evidence: it is in the (screened) package.
        allowed_text = " ".join(
            [state.query, *(f"{e.source_id} {e.title} {e.content}" for e in entries)]
        )
        allowed_urls = _urls(allowed_text)
        allowed_ids = set(_IDENTIFIER.findall(allowed_text))
        quarantined = set().union(*(_shingles(t) for t in state.quarantined_text))
        reasons: list[str] = []

        answer, images = _IMAGE.subn("", answer)
        if images:
            reasons.append("embedded image links")
        answer, secrets = redact_secrets(answer)
        if secrets:
            state.security.secrets_redacted += secrets
            reasons.append("credentials")

        kept: list[str] = []
        for sentence in _SENTENCE.findall(answer):
            reason = self._problem(sentence, allowed_ids, quarantined)
            if reason:
                reasons.append(reason)
                continue
            kept.append(
                _URL.sub(
                    lambda m: (
                        m.group(0)
                        if m.group(0).rstrip(_URL_TRAILING) in allowed_urls
                        else "[link removed]"
                    ),
                    sentence,
                )
            )
            if kept[-1] != sentence:
                reasons.append("links not found in the evidence")
        text = "".join(kept).strip()
        if reasons:
            unique = list(dict.fromkeys(reasons))
            state.security.output_removed.extend(unique)
            state.limit(
                "Parts of the answer failed output validation and were removed: "
                + ", ".join(unique)
                + "."
            )
            logger.warning(
                "security.output_blocked",
                extra={"user": state.user_id, "reasons": unique},
            )
        state.final_answer = text or WITHHELD

    def _problem(
        self, sentence: str, allowed_ids: set[str], quarantined: set[tuple[str, ...]]
    ) -> str | None:
        if not sentence.strip():
            return None
        if REMOVED in sentence:
            return "text removed by screening"
        if scan(sentence).flagged:
            return "instruction-like text"
        if self.canary and self.canary in sentence:
            return "system prompt text"
        shingles = _shingles(sentence)
        if shingles & quarantined:
            return "text from a quarantined source"
        if len(shingles & self.prompt_shingles) >= 2:
            return "system prompt text"
        unknown = set(_IDENTIFIER.findall(sentence)) - allowed_ids
        if unknown:
            return "identifiers not in the evidence"
        return None


__all__ = [
    "EVIDENCE_RESOURCE",
    "REDACTED",
    "REFUSAL",
    "WITHHELD",
    "EvidenceScreen",
    "InjectionAction",
    "OutputGuard",
    "QueryInjectionAction",
    "may_read_item",
    "plain",
    "screen_question",
]
