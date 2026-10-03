"""Source conflicts: when two pieces of evidence state different values for one setting.

The corpus states configuration values in several places: configuration reference
tables in the documentation, deploy manifests and source files, pull-request diffs, and
postmortems ("`HTTP_TIMEOUT_SECONDS` dropped from 2.5s to 0.62s"). They do not always
agree: a postmortem describes the value at the time of its incident, a document may be
older than the last change. This module

1. extracts ``setting = value`` assertions from every evidence item, with the item's
   timestamp and whether the assertion describes the value in effect: a ``+`` line of a
   change whose deployment was rolled back, or the change a postmortem names as the
   cause of its incident, is not what runs now (``-`` lines are old values and are not
   assertions at all);
2. groups them by (service, setting) and reports every group whose sources disagree,
   with all values, sources and timestamps: nothing is discarded;
3. prefers the newest assertion still in effect, and says so, or says that no
   preference can be given.

It is deterministic and cites evidence labels; it does not decide which source is
"right" beyond the stated rule, and the answer shows both sides.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime

from pydantic import BaseModel

from app.agents.state import EvidenceItem, EvidenceKind

_KEY = r"`?([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)`?"
_ASSIGN = re.compile(rf"^([+-]?)\s*{_KEY}\s*[:=]\s*(.+?)\s*$")
_TABLE = re.compile(rf"^\|\s*{_KEY}\s*\|\s*`?([^|`]+?)`?\s*\|")
_TRANSITION = re.compile(
    rf"{_KEY}\s+(?:was\s+)?(?:dropped|rose|raised|lowered|increased|decreased|reduced|"
    r"changed|went|set)\s+from\s+([\w.\-]+)\s+to\s+([\w.\-]+)",
    re.I,
)
_SERVICE_PREFIX = re.compile(r"^([A-Z]+_(?:SERVICE|GATEWAY))_(.+)$")
_NUMBER = re.compile(r"^(-?\d+(?:\.\d+)?)(ms|s|m|h)?$")
_GENERIC = {"seconds", "service", "http", "max", "min", "ms", "enabled", "count", "size"}
_VALUE_END = re.compile(r"\s+#.*$|[,;.]$")  # a trailing comment or sentence punctuation


class Assertion(BaseModel):
    """One source stating a value for a setting."""

    service_id: str | None
    key: str  # without the service prefix
    value: str  # normalised, for comparison
    raw: str  # as written
    source_id: str
    source: tuple[str, str | None]  # (source id, chunk id): two changes to one file differ
    name: str  # how the answer names the source
    label: str | None
    kind: EvidenceKind
    timestamp: datetime | None
    in_effect: bool  # False: describes a change that was undone, or a failed one
    context: str  # how it was stated, e.g. "diff (+)", "postmortem root cause"


class ConflictValue(BaseModel):
    value: str
    in_effect: bool
    latest: datetime | None  # newest source stating it
    sources: list[str]  # source names, as the answer shows them
    source_ids: list[str]
    labels: list[str]  # evidence labels, for citations


class Conflict(BaseModel):
    service_id: str | None
    key: str
    values: list[ConflictValue]  # newest first
    preferred: str | None
    reason: str

    @property
    def setting(self) -> str:
        return f"{self.service_id} {self.key}" if self.service_id else self.key

    def describe(self) -> str:
        """One sentence naming every value, where it is stated and when."""
        sides = []
        for v in self.values:
            when = _moment(v.latest) if v.latest else "undated"
            state = "" if v.in_effect else ", no longer in effect"
            cites = "".join(f"[{label}]" for label in v.labels)
            sides.append(f"{v.value} ({', '.join(v.sources[:3])}, {when}{state}) {cites}".strip())
        text = f"Sources disagree on {self.setting}: " + " vs ".join(sides) + "."
        if self.preferred is not None:
            text += f" The newest statement still in effect gives {self.preferred}."
        else:
            text += f" {self.reason[:1].upper()}{self.reason[1:]}."
        return text


def _moment(value: datetime) -> str:
    return f"{_utc(value):%Y-%m-%d %H:%M} UTC"


def _name(item: EvidenceItem) -> str:
    """Code is named by its path (and pull request); records and documents by their id."""
    if item.kind in {EvidenceKind.CODE, EvidenceKind.PULL_REQUEST}:
        name = item.title or item.location or item.source_id
        return name if len(name) <= 90 else name[:89].rsplit(" ", 1)[0] + "…"
    return item.source_id


def _normal_value(raw: str, key: str) -> str:
    text = _VALUE_END.sub("", raw.strip()).strip().strip("\"'`").strip()
    match = _NUMBER.match(text)
    if not match:
        return text.upper() if text.isalpha() else text
    number, unit = float(match.group(1)), match.group(2)
    if key.endswith("_SECONDS") and unit in {"ms", "m", "h"}:
        number = number * {"ms": 0.001, "m": 60, "h": 3600}[unit]
    return f"{number:g}"


def _split_key(key: str, service_id: str | None) -> tuple[str | None, str]:
    match = _SERVICE_PREFIX.match(key)
    if match:
        return match.group(1).lower().replace("_", "-"), match.group(2)
    return service_id, key


def _root_cause_postmortem(item: EvidenceItem) -> bool:
    if item.kind is not EvidenceKind.POSTMORTEM:
        return False
    section = (item.section or item.facts.get("section") or "").lower()
    return "root cause" in section or "## root cause" in item.text.lower()


def assertions(item: EvidenceItem) -> list[Assertion]:
    """The setting values ``item`` states."""
    found: list[Assertion] = []
    cause = _root_cause_postmortem(item)
    in_effect_item = item.facts.get("valid", "true") != "false"

    def add(key: str, raw: str, in_effect: bool, context: str) -> None:
        service, short = _split_key(key, item.service_id)
        value = _normal_value(raw, short)
        if not value or len(value) > 60:
            return
        found.append(
            Assertion(
                service_id=service,
                key=short,
                value=value,
                raw=raw.strip(),
                source_id=item.source_id,
                source=(item.source_id, item.chunk_id),
                name=_name(item),
                label=item.label,
                kind=item.kind,
                timestamp=item.timestamp,
                in_effect=in_effect,
                context=context,
            )
        )

    for line in item.text.splitlines():
        if line.startswith(("+++", "---")):
            continue
        assign = _ASSIGN.match(line.strip() if not line.startswith(("+", "-")) else line)
        if assign:
            sign, key, raw = assign.groups()
            if sign == "-":
                continue  # the old value of a change: not a statement about now
            if sign == "+":
                add(key, raw, in_effect_item and not cause, "diff (+)")
            else:
                add(key, raw, in_effect_item, "stated")
            continue
        table = _TABLE.match(line.strip())
        if table:
            add(table.group(1), table.group(2), in_effect_item, "table")
            continue
        for transition in _TRANSITION.finditer(line):
            key, _, new = transition.groups()
            add(key, new, in_effect_item and not cause, "change")
    # A postmortem's diff also lists the new value with the "+": one statement per value.
    unique: dict[tuple[str | None, str, str], Assertion] = {}
    for a in found:
        unique.setdefault((a.service_id, a.key, a.value), a)
    return list(unique.values())


def _newest(values: Iterable[Assertion]) -> datetime | None:
    stamps = [a.timestamp for a in values if a.timestamp is not None]
    return max(stamps, key=_utc) if stamps else None


def _utc(value: datetime | None) -> datetime:
    if value is None:
        return datetime.min.replace(tzinfo=UTC)
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def find_conflicts(items: Sequence[EvidenceItem]) -> list[Conflict]:
    """Every (service, setting) on which the evidence states more than one value."""
    groups: dict[tuple[str | None, str], list[Assertion]] = defaultdict(list)
    for item in items:
        for a in assertions(item):
            groups[(a.service_id, a.key)].append(a)
    conflicts = []
    for (service, key), stated in groups.items():
        by_value: dict[str, list[Assertion]] = defaultdict(list)
        for a in stated:
            by_value[a.value].append(a)
        sources = {a.source for a in stated}
        if len(by_value) < 2 or len(sources) < 2:
            continue  # one value, or one source listing several (e.g. a diff)
        values = [
            ConflictValue(
                value=value,
                in_effect=any(a.in_effect for a in group),
                latest=_newest(group),
                sources=list(dict.fromkeys(a.name for a in group)),
                source_ids=list(dict.fromkeys(a.source_id for a in group)),
                labels=list(dict.fromkeys(a.label for a in group if a.label)),
            )
            for value, group in by_value.items()
        ]
        values.sort(key=lambda v: _utc(v.latest), reverse=True)
        current = [
            (_utc(a.timestamp), a.value) for a in stated if a.in_effect and a.timestamp is not None
        ]
        if current:
            preferred = max(current)[1]
            reason = "the newest statement still in effect is preferred"
        else:
            preferred = None
            reason = "no dated statement still in effect, so neither value can be preferred"
        conflicts.append(
            Conflict(service_id=service, key=key, values=values, preferred=preferred, reason=reason)
        )
    conflicts.sort(key=lambda c: (c.service_id or "", c.key))
    return conflicts


def _terms(text: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", text.lower()) if len(t) > 2}


def relevant(conflict: Conflict, question: str, answer: str, services: Sequence[str]) -> bool:
    """Whether the conflict bears on this question: the setting is named in the question
    or the answer, or its distinctive words are in the question (for the named services)."""
    full = f"{(conflict.service_id or '').upper().replace('-', '_')}_{conflict.key}".lstrip("_")
    for text in (question, answer):
        if conflict.key in text or full in text:
            return True
    if services and conflict.service_id and conflict.service_id not in services:
        return False
    words = _terms(conflict.key.replace("_", " ")) - _GENERIC
    return bool(words & _terms(question))


__all__ = ["Assertion", "Conflict", "ConflictValue", "assertions", "find_conflicts", "relevant"]
