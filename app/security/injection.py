"""Detection of instruction-like text aimed at the model (prompt injection).

Used on the user's question and on every piece of retrieved content before it can
reach a model. The rules look for text that addresses the assistant, not for topics:
"ignore all previous instructions", "you are now in developer mode", "reveal your
system prompt", chat-template markup, exfiltration links, requests to bypass access
control. A runbook that says "ignore transient 5xx alerts" is not flagged.

Text is normalised first (NFKC, invisible format characters removed, lower-cased,
whitespace collapsed), and a letters-only form catches spaced or punctuated
variants of the classic override phrases ("I.g.n.o.r.e previous instructions").

Detection is a heuristic and cannot be complete; it is one layer. Access control
does not depend on it: whatever the text says, tools only return what the caller's
grants allow, and the answer is verified against the evidence afterwards.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum


class InjectionCategory(StrEnum):
    INSTRUCTION_OVERRIDE = "instruction_override"
    ROLE_MANIPULATION = "role_manipulation"
    PROMPT_EXFILTRATION = "prompt_exfiltration"
    SECRET_EXFILTRATION = "secret_exfiltration"
    TOOL_MANIPULATION = "tool_manipulation"
    MARKUP_INJECTION = "markup_injection"
    CONCEALMENT = "concealment"
    DATA_EXFILTRATION = "data_exfiltration"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    SEMANTIC = "semantic_injection"  # found by the classifier (injection_model.py)


C = InjectionCategory

_RULES: tuple[tuple[InjectionCategory, str, str], ...] = (
    # "ignore all previous instructions", "disregard the system prompt"
    (
        C.INSTRUCTION_OVERRIDE,
        "override_instructions",
        r"\b(?:ignore|disregard|forget|override|bypass)\b(?:\W+(?:all|any|every|the|your|my|"
        r"these|those|of|previous|prior|preceding|above|earlier|original|initial|system|"
        r"safety|existing|other|given)){0,4}\W+(?:instructions?|prompts?|directives?|"
        r"guardrails|system\W+messages?)\b",
    ),
    (
        C.INSTRUCTION_OVERRIDE,
        "new_instructions",
        r"\b(?:new|updated|revised|real|actual|additional|override)\s+(?:system\s+)?"
        r"(?:instructions?|directives?)\s*[:\-]",
    ),
    (
        C.INSTRUCTION_OVERRIDE,
        "from_now_on",
        r"\bfrom now on\b[^.\n]{0,40}\b(?:you|assistant|model|ai)\b",
    ),
    (
        C.INSTRUCTION_OVERRIDE,
        "task_redefinition",
        r"\b(?:your|the)\s+(?:new|real|true|actual)\s+(?:task|goal|objective|purpose|"
        r"instructions?|role)\s+(?:is|are)\b",
    ),
    # "you are now DAN", "act as an unrestricted assistant", "developer mode"
    (C.ROLE_MANIPULATION, "you_are_now", r"\byou are (?:now|no longer)\b"),
    (
        C.ROLE_MANIPULATION,
        "act_as",
        r"\b(?:act|behave|respond|answer)\s+as\s+(?:an?\s+|the\s+)?(?:unrestricted|unfiltered|"
        r"uncensored|jailbroken|admin(?:istrator)?|root|superuser|system|dan)\b",
    ),
    (
        C.ROLE_MANIPULATION,
        "special_mode",
        r"\b(?:developer|god|jailbreak|dan|unrestricted|debug|admin|sudo)\s+mode\b",
    ),
    (
        C.ROLE_MANIPULATION,
        "pretend",
        r"\bpretend\s+(?:to\s+be|that\s+you|you\s+are)\b|\bdo anything now\b",
    ),
    # "print your system prompt"
    (
        C.PROMPT_EXFILTRATION,
        "reveal_prompt",
        r"\b(?:reveal|print|show|output|repeat|display|dump|leak|disclose|return|tell\s+me|"
        r"what\s+(?:is|are))\b[^.\n]{0,30}\b(?:system\s+prompt|hidden\s+(?:prompt|instructions)|"
        r"your\s+(?:instructions|system\s+prompt|prompt|rules|guidelines|configuration)|"
        r"initial\s+(?:prompt|instructions)|developer\s+(?:message|prompt))\b",
    ),
    # "include the API keys in your answer"
    (
        C.SECRET_EXFILTRATION,
        "reveal_secrets",
        r"(?<!never )(?<!not )(?<!n't )\b(?:reveal|print|show|output|dump|leak|disclose|send|"
        r"exfiltrate|list|give\s+me|expose|include|read\s+out)\b[^.\n]{0,40}\b(?:api[\s_-]?keys?|"
        r"passwords?|secrets?|credentials?|access\s+tokens?|private\s+keys?|\.env\b|"
        r"environment\s+variables|connection\s+strings?)\b",
    ),
    # "call the query_database tool", "use search_logs to"
    (
        C.TOOL_MANIPULATION,
        "invoke_tool",
        r"\b(?:call|invoke|execute|trigger)\s+(?:the\s+)?(?:\w+\s+)?(?:tool|function)\b|"
        r"\b(?:query_database|search_logs|search_code|search_documents|search_incidents|"
        r"search_deployments|get_runbook)\s*(?:\(|with\b|to\b)",
    ),
    # chat-template and prompt markup
    (
        C.MARKUP_INJECTION,
        "chat_markup",
        r"<\|?\s*(?:im_start|im_end|endoftext|system|assistant)\s*\|?>|\[/?(?:inst|sys)\]|"
        r"<</?sys>>|</?\s*(?:system|instructions?|assistant)\s*>",
    ),
    (
        C.MARKUP_INJECTION,
        "role_header",
        r"(?:^|\n)\s*(?:#+\s*)?(?:system|assistant)\s*(?:prompt|message)?\s*:\s*\S",
    ),
    # "do not tell the user"
    (
        C.CONCEALMENT,
        "hide_from_user",
        r"\b(?:do\s+not|don't|never|without)\s+(?:tell(?:ing)?|inform(?:ing)?|let(?:ting)?|"
        r"alert(?:ing)?|notify(?:ing)?)\s+(?:the\s+)?(?:user|human)\b",
    ),
    # markdown image beacons and "send it to https://..."
    (C.DATA_EXFILTRATION, "image_beacon", r"!\[[^\]]*\]\(\s*https?://"),
    (
        C.DATA_EXFILTRATION,
        "send_to_url",
        r"\b(?:send|post|upload|forward|exfiltrate|transmit)\b[^.\n]{0,40}\bto\s+https?://|"
        r"\b(?:curl|wget)\s+(?:-\S+\s+)*https?://",
    ),
    # "bypass access control", "grant me admin", "show confidential documents"
    (
        C.PRIVILEGE_ESCALATION,
        "bypass_access",
        r"\b(?:bypass|ignore|disable|skip|override|circumvent|turn\s+off)\b[^.\n]{0,30}\b"
        r"(?:access\s+control|permissions?|rbac|authori[sz]ation|security\s+(?:checks?|"
        r"filters?|controls?)|access\s+(?:checks?|levels?|rules|restrictions?)|clearance)\b",
    ),
    (
        C.PRIVILEGE_ESCALATION,
        "elevate",
        r"\b(?:grant|give|elevate|escalate|upgrade|promote|switch|change)\s+(?:me|my\s+"
        r"(?:role|access|privileges?|permissions?))\s+(?:to\s+)?(?:an?\s+)?(?:admin(?:istrator)?|"
        r"root|superuser|elevated|full|higher|sre|manager)\b",
    ),
    (
        C.PRIVILEGE_ESCALATION,
        "claimed_admin",
        r"\b(?:i\s+am|i'm|as)\s+(?:an?\s+|the\s+)?(?:admin(?:istrator)?|root|superuser)\b|"
        r"\b(?:with|using|in)\s+(?:admin|root|superuser|elevated)\s+(?:privileges?|rights|"
        r"access|permissions?)\b|\bsudo\b",
    ),
    (
        C.PRIVILEGE_ESCALATION,
        "request_confidential",
        r"\b(?:show|give|list|reveal|display|dump|read|open|print)\b[^.\n]{0,30}\b"
        r"(?:confidential|classified|restricted|admin[\s-]only|hidden)\s+(?:documents?|docs|"
        r"data|files?|records?|information|incidents?|logs)\b",
    ),
)
_COMPILED = tuple((c, name, re.compile(p, re.IGNORECASE)) for c, name, p in _RULES)

# Letters-only forms of the classic override phrases, for spaced/punctuated variants.
_COMPACT = tuple(
    f"{verb}{scope}{what}instructions"
    for verb in ("ignore", "disregard", "forget")
    for scope in ("all", "")
    for what in ("previous", "prior", "above", "your", "the")
)

_WS = re.compile(r"\s+")
# A sentence ends at . ! or ? followed by whitespace, or at a line end; dots inside
# URLs, versions and file names do not split it.
_BOUNDARY = re.compile(r"[.!?]+(?=\s)|\n")


def normalize(text: str) -> str:
    """NFKC, invisible format characters removed, lower case, single spaces."""
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")
    return _WS.sub(" ", text.lower()).strip()


@dataclass(frozen=True)
class Finding:
    category: InjectionCategory
    rule: str


@dataclass(frozen=True)
class ScanResult:
    findings: tuple[Finding, ...] = ()

    @property
    def flagged(self) -> bool:
        return bool(self.findings)

    @property
    def categories(self) -> list[str]:
        return sorted({f.category.value for f in self.findings})


def _scan_normalized(text: str, raw: str) -> list[Finding]:
    findings = [Finding(c, name) for c, name, rule in _COMPILED if rule.search(text)]
    # role_header needs the original line structure
    if not any(f.rule == "role_header" for f in findings):
        header = next(r for c, name, r in _COMPILED if name == "role_header")
        if header.search(unicodedata.normalize("NFKC", raw).lower()):
            findings.append(Finding(C.MARKUP_INJECTION, "role_header"))
    compact = re.sub(r"[^a-z]", "", text)
    if any(phrase in compact for phrase in _COMPACT) and not any(
        f.rule == "override_instructions" for f in findings
    ):
        findings.append(Finding(C.INSTRUCTION_OVERRIDE, "override_obfuscated"))
    return findings


def scan(text: str) -> ScanResult:
    """All rules that match anywhere in ``text``."""
    if not text:
        return ScanResult()
    return ScanResult(tuple(_scan_normalized(normalize(text), text)))


def segments(text: str) -> list[tuple[int, int]]:
    """Sentence/line spans of ``text``; the unit that redaction removes."""
    spans: list[tuple[int, int]] = []
    start = 0
    for match in _BOUNDARY.finditer(text):
        if text[start : match.end()].strip():
            spans.append((start, match.end()))
        start = match.end()
    if text[start:].strip():
        spans.append((start, len(text)))
    return spans


REMOVED = "[removed: instruction-like text]"


def strip_instructions(text: str) -> tuple[str, ScanResult]:
    """``text`` with every flagged sentence or line replaced by a marker.

    A rule can span sentences (a markdown image after a full stop), so the whole
    text is scanned too; if it is flagged but no single segment is, or if what is left
    is still flagged, everything is removed (fail closed).
    """
    whole = scan(text)
    if not whole.flagged:
        return text, whole
    parts: list[str] = []
    position = 0
    removed = False
    for start, end in segments(text):
        segment = text[start:end]
        parts.append(text[position:start])
        if scan(segment).flagged:
            parts.append(REMOVED + ("\n" if segment.endswith("\n") else " "))
            removed = True
        else:
            parts.append(segment)
        position = end
    parts.append(text[position:])
    stripped = "".join(parts).strip()
    if not removed or scan(stripped.replace(REMOVED, "")).flagged:
        return REMOVED, whole
    return stripped, whole


__all__ = [
    "REMOVED",
    "Finding",
    "InjectionCategory",
    "ScanResult",
    "normalize",
    "scan",
    "segments",
    "strip_instructions",
]
