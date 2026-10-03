"""Temporal questions: relations in time, answered from timestamps.

"Which deployment happened immediately before INC-0421?" is not a similarity question:
the answer is the latest deployment whose timestamp precedes the incident's start. This
module turns such a question into a ``TemporalQuery`` (relation, target, anchor), plans
two calls (fetch the anchor, then query the target ordered by time relative to the
anchor's timestamps) and states the result in a *timeline* evidence item: a sentence
built only from the records' timestamps and ids, citable and checkable like any other
evidence.

Relations: before / after (within a window), during, latest, previous / next (same kind
as the anchor), immediately before / after (a different kind), at the time of.

Scope: deployments are looked up for the anchor's own service unless the question names
services; incidents across all services, except "previous/next incident", which follows
the anchor's service.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel

from app.agents.entities import QueryEntities
from app.agents.state import EvidenceItem, EvidenceKind, PlannedCall
from app.schemas.enums import UNLABELLED_LEVEL, AccessLevel, Resource, most_restrictive
from app.tools.deployments import DeploymentRecord, SearchDeploymentsOutput
from app.tools.search import IncidentSummary, SearchIncidentsOutput

ANCHOR, TARGET = "temporal anchor", "temporal target"
SECOND = timedelta(seconds=1)
LIVE_STATUSES = ["succeeded", "rolled_back"]  # failed deployments never served traffic
DEFAULT_WINDOW = timedelta(hours=24)


class Relation(StrEnum):
    IMMEDIATELY_BEFORE = "immediately_before"  # nearest earlier record of another kind
    IMMEDIATELY_AFTER = "immediately_after"  # nearest later record of another kind
    PREVIOUS = "previous"  # nearest earlier record of the same kind
    NEXT = "next"  # nearest later record of the same kind
    LATEST = "latest"  # the newest, as of the clock
    AT_TIME_OF = "at_time_of"  # the deployment live (or incident open) at that moment
    DURING = "during"  # everything within the anchor's span
    BEFORE = "before"  # everything within a window before the anchor
    AFTER = "after"  # everything within a window after the anchor


SINGLE = {
    Relation.IMMEDIATELY_BEFORE,
    Relation.IMMEDIATELY_AFTER,
    Relation.PREVIOUS,
    Relation.NEXT,
    Relation.LATEST,
    Relation.AT_TIME_OF,
}

Target = Literal["deployment", "incident"]


AnchorKind = Literal["incident", "deployment", "version", "date", "now"]


class TemporalQuery(BaseModel):
    relation: Relation
    target: Target
    anchor_kind: AnchorKind
    anchor_id: str | None = None  # INC-/DEP- id, the version, or the ISO date
    services: list[str] = []  # named in the question
    all_services: bool = False  # "on any service": do not scope to the anchor's service
    statuses: list[str] = []  # deployment statuses asked for ("only successful ones")
    window: timedelta | None = None
    unhandled: list[str] = []  # qualifiers in the question this plan does not apply


class Anchor(BaseModel):
    kind: Literal["incident", "deployment", "date", "now"]
    id: str | None
    name: str  # what the question called it: the id, a version, or "now"
    service_id: str | None
    start: datetime
    end: datetime
    label: str  # how answers refer to it
    resource: Resource | None = None
    access_level: AccessLevel = UNLABELLED_LEVEL


# --- parsing ---------------------------------------------------------------------------

_DEPLOY = re.compile(
    r"\b(?:deployments?|deploys?|deployed|releases?|rollouts?|versions?|went out|shipped|"
    r"running|live|in production)\b",
    re.I,
)
_INCIDENT = re.compile(r"\bincidents?\b", re.I)
_PLURAL = re.compile(r"\b(?:deployments|releases|rollouts|incidents|versions|deploys)\b", re.I)
_AT_TIME = re.compile(
    r"\bat the time of\b|\b(?:running|live|in production|serving traffic|serving|deployed|"
    r"open|ongoing|active|in progress)\b.{0,40}\b(?:when|at the time|as of)\b",
    re.I,
)
_AMOUNT = re.compile(r"\b(\d+|a|an|one|the)\s+(hours?|days?|weeks?)\b", re.I)
_BEFORE_WORD = re.compile(r"\b(?:before|preced\w*|prior to|leading up to)\b", re.I)
_AFTER_WORD = re.compile(r"\b(?:after|follow\w*|since)\b", re.I)
_DURING = re.compile(
    r"\bduring\b|\bwhile\b.{0,60}\b(?:ongoing|open|active|happening|in progress)\b|\boverlap",
    re.I,
)
_IMMEDIATELY_BEFORE = re.compile(
    r"\b(?:immediately|right|just)\s+(?:before|preced\w*)|\blast\b.{0,60}\bbefore\b|"
    r"\b(?:latest|most recent)\b.{0,60}\bbefore\b|"
    r"\bbefore\b.{0,60}\b(?:latest|most recent|last)\b",  # "Before INC-1, what was the last ..."
    re.I,
)
_IMMEDIATELY_AFTER = re.compile(
    r"\b(?:immediately|right|just)\s+after\b|\bfirst\b.{0,60}\bafter\b|\bnext\b.{0,60}\bafter\b|"
    r"\bfollow(?:ed|ing|s)\b|\bcame after\b|"
    r"\bafter\b.{0,60}\b(?:first|next)\b",  # "After INC-1, which deployment was the first ..."
    re.I,
)
_PREVIOUS = re.compile(r"\bprevious\b|\bpreced(?:ed|ing|es)\b|\bprior to\b|\bbefore\b", re.I)
_LATEST = re.compile(r"\blatest\b|\bmost recent(?:ly)?\b|\bnewest\b|\blast\b", re.I)
_SELECTION = re.compile(r"\b(?:which|what|name|list|show|when)\b", re.I)
_UNITS = {"hour": timedelta(hours=1), "day": timedelta(days=1), "week": timedelta(weeks=1)}
_STATUS_WORDS = {
    "successful": "succeeded",
    "succeeded": "succeeded",
    "failed": "failed",
    "rolled back": "rolled_back",
    "rolled-back": "rolled_back",
}
_STATUS = re.compile(
    r"\b(?:only\s+)?(successful|succeeded|failed|rolled[ -]back)\s+"
    r"(?:deployments?|deploys?|releases?|rollouts?|ones)\b|"
    r"\bonly\s+(?:the\s+)?(successful|succeeded|failed|rolled[ -]back)\b",
    re.I,
)
_ANY_SERVICE = re.compile(
    r"\b(?:on|of|to|for|across)\s+(?:any|all|every)\s+(?:the\s+)?services?\b|"
    r"\bacross (?:all )?services\b|\b(?:platform|service)[- ]wide\b",
    re.I,
)
_DATE = re.compile(r"\b(20\d\d-\d\d-\d\d)(?:[ T](\d\d:\d\d))?(?:\s*UTC|Z)?\b")
_ON_DATE = re.compile(r"\bon\s+20\d\d-\d\d-\d\d\b", re.I)
# Words that restrict the answer in ways the plan below does not apply.
_QUALIFIER = re.compile(
    r"\b(?:only|except|excluding|other than|not counting|apart from|besides|ignoring|unless|"
    r"without)\b",
    re.I,
)


def _window(text: str) -> timedelta | None:
    match = _AMOUNT.search(text)
    if not match:
        return None
    amount = match.group(1).lower()
    count = int(amount) if amount.isdigit() else 1
    return _UNITS[match.group(2).lower().rstrip("s")] * count


def _statuses(text: str) -> tuple[list[str], str]:
    """The deployment statuses asked for, and the text without the phrase."""
    found: list[str] = []
    for match in _STATUS.finditer(text):
        word = (match.group(1) or match.group(2)).lower().replace("-", " ")
        found.append(_STATUS_WORDS[word])
    return list(dict.fromkeys(found)), _STATUS.sub(" ", text)


def _target(text: str) -> Target | None:
    deploy, incident = _DEPLOY.search(text), _INCIDENT.search(text)
    if deploy and (not incident or deploy.start() < incident.start()):
        return "deployment"
    return "incident" if incident else None


def parse_temporal(question: str, entities: QueryEntities) -> TemporalQuery | None:
    """The temporal reading of ``question``, or None when it is not a question that
    selects records by their position in time."""
    if not _SELECTION.search(question):
        return None
    ids = sorted(
        [(question.find(i), "incident", i) for i in entities.incident_ids]
        + [(question.find(d), "deployment", d) for d in entities.deployment_ids]
    )
    date = _DATE.search(question) if not ids else None
    # The target is usually named before the anchor ("which deployment ... before
    # INC-0421"), but may follow it ("INC-0421: which deployment preceded it?").
    first = ids[0][0] if ids else date.start() if date else len(question)
    head, tail = question[:first], question[first:]
    target = _target(head) or _target(tail)
    if target is None:
        return None
    anchor_kind: AnchorKind
    anchor_id: str | None = None
    if ids:
        _, anchor_kind, anchor_id = ids[0]  # type: ignore[assignment]
    elif date:
        anchor_kind = "date"
        anchor_id = date.group(1) + (f"T{date.group(2)}" if date.group(2) else "")
    elif entities.versions and entities.services:
        anchor_kind, anchor_id = "version", entities.versions[0]
    else:
        anchor_kind = "now"
    statuses, rest = _statuses(question) if target == "deployment" else ([], question)
    window = _window(question)
    relation: Relation | None = None
    if _AT_TIME.search(question):
        relation = Relation.AT_TIME_OF
    elif window and (_BEFORE_WORD.search(question) or _AFTER_WORD.search(question)):
        relation = Relation.BEFORE if _BEFORE_WORD.search(question) else Relation.AFTER
    elif _DURING.search(question) or (anchor_kind == "date" and _ON_DATE.search(question)):
        relation = Relation.DURING
    elif _IMMEDIATELY_BEFORE.search(question):
        relation = Relation.IMMEDIATELY_BEFORE
    elif _IMMEDIATELY_AFTER.search(question):
        relation = Relation.IMMEDIATELY_AFTER
    elif anchor_kind != "now" and _PREVIOUS.search(question):
        relation = Relation.IMMEDIATELY_BEFORE
    elif anchor_kind != "now" and _AFTER_WORD.search(question):
        relation = Relation.IMMEDIATELY_AFTER
    elif anchor_kind == "now" and _LATEST.search(question):
        relation = Relation.LATEST
    if relation is None or (anchor_kind == "now" and relation is not Relation.LATEST):
        return None
    same_kind = anchor_kind == target or (anchor_kind == "version" and target == "deployment")
    if relation is Relation.IMMEDIATELY_BEFORE and same_kind:
        relation = Relation.PREVIOUS
    if relation is Relation.IMMEDIATELY_AFTER and same_kind:
        relation = Relation.NEXT
    plural = bool(_PLURAL.search(head if _target(head) else tail))
    if plural and relation in {Relation.IMMEDIATELY_BEFORE, Relation.PREVIOUS}:
        relation, window = Relation.BEFORE, window or DEFAULT_WINDOW
    if plural and relation in {Relation.IMMEDIATELY_AFTER, Relation.NEXT}:
        relation, window = Relation.AFTER, window or DEFAULT_WINDOW
    if relation is Relation.DURING and anchor_kind == "date":
        window = timedelta(days=1)
    return TemporalQuery(
        relation=relation,
        target=target,
        anchor_kind=anchor_kind,
        anchor_id=anchor_id,
        services=list(entities.services),
        all_services=bool(_ANY_SERVICE.search(question)),
        statuses=statuses,
        window=window,
        unhandled=list(dict.fromkeys(m.group(0).lower() for m in _QUALIFIER.finditer(rest))),
    )


# --- planning ----------------------------------------------------------------------------


def _iso(value: datetime) -> str:
    return value.isoformat()


def tools_for(query: TemporalQuery) -> list[str]:
    """The tools the plan will call: the anchor's (if it is a record) and the target's."""
    anchor = {
        "incident": ["search_incidents"],
        "deployment": ["search_deployments"],
        "version": ["search_deployments"],
    }.get(query.anchor_kind, [])
    target = "search_deployments" if query.target == "deployment" else "search_incidents"
    return list(dict.fromkeys([*anchor, target]))


def anchor_call(query: TemporalQuery) -> PlannedCall | None:
    if query.anchor_kind == "incident":
        return PlannedCall(
            tool="search_incidents",
            arguments={"incident_ids": [query.anchor_id], "include_postmortems": False},
            purpose=f"{ANCHOR}: fetch {query.anchor_id}",
        )
    if query.anchor_kind == "deployment":
        return PlannedCall(
            tool="search_deployments",
            arguments={"deployment_ids": [query.anchor_id]},
            purpose=f"{ANCHOR}: fetch {query.anchor_id}",
        )
    if query.anchor_kind == "version":
        return PlannedCall(
            tool="search_deployments",
            arguments={"services": query.services[:1], "versions": [query.anchor_id], "limit": 5},
            purpose=f"{ANCHOR}: fetch {query.services[0]} {query.anchor_id}",
        )
    return None


def fixed_anchor(query: TemporalQuery, now: datetime) -> Anchor:
    """The anchor of a question relative to the clock or to a date: no record to fetch."""
    if query.anchor_kind == "date" and query.anchor_id:
        moment = datetime.fromisoformat(query.anchor_id).replace(tzinfo=UTC)
        end = moment
        if query.relation is Relation.DURING:
            end = moment + (query.window or timedelta(days=1))
        return Anchor(
            kind="date",
            id=None,
            name=query.anchor_id,
            service_id=None,
            start=moment,
            end=end,
            label=_when(moment),
        )
    return Anchor(
        kind="now", id=None, name="now", service_id=None, start=now, end=now, label=_when(now)
    )


def anchor_from(query: TemporalQuery, output: Any) -> Anchor | None:
    if isinstance(output, SearchIncidentsOutput) and output.incidents:
        i = output.incidents[0]
        return Anchor(
            kind="incident",
            id=i.id,
            name=i.id,
            service_id=i.service_id,
            start=i.started_at,
            end=i.resolved_at,
            label=f"{i.id} ({i.service_id}, started {_when(i.started_at)})",
            resource=Resource.INCIDENTS,
            access_level=i.access_level,
        )
    if isinstance(output, SearchDeploymentsOutput) and output.deployments:
        d = min(output.deployments, key=lambda d: d.deployed_at)  # first release of a version
        if query.anchor_kind == "version":  # named by version: no other deployment id
            name = d.version
            label = f"{d.service_id} {d.version} (first deployed {_when(d.deployed_at)})"
        else:
            name = d.id
            label = f"{d.id} ({d.service_id} {d.version}, deployed {_when(d.deployed_at)})"
        return Anchor(
            kind="deployment",
            id=d.id,
            name=name,
            service_id=d.service_id,
            start=d.deployed_at,
            end=d.deployed_at,
            label=label,
            resource=Resource.DEPLOYMENTS,
        )
    return None


def _services(query: TemporalQuery, anchor: Anchor) -> list[str] | None:
    if query.services:
        return query.services
    if query.all_services:
        return None
    if query.target == "deployment" or query.relation in {Relation.PREVIOUS, Relation.NEXT}:
        return [anchor.service_id] if anchor.service_id else None
    return None  # incidents: across services


def target_call(query: TemporalQuery, anchor: Anchor) -> PlannedCall:
    """The query for the target records, from the anchor's timestamps.

    Single answers take the nearest record (newest before, oldest after); windows list
    every record in them, oldest first. Windows are [since, until): one second is added
    where the anchor's own moment must be included."""
    r, start, end = query.relation, anchor.start, anchor.end
    window = query.window or DEFAULT_WINDOW
    since: datetime | None = None
    until: datetime | None = None
    one, order, overlap = single(query), "oldest", False
    if r in {Relation.IMMEDIATELY_BEFORE, Relation.PREVIOUS, Relation.LATEST}:
        until, order = start, "newest"
    elif r in {Relation.IMMEDIATELY_AFTER, Relation.NEXT}:
        since = start + SECOND
    elif r is Relation.AT_TIME_OF and query.target == "deployment":
        until, order = start + SECOND, "newest"
    elif r in {Relation.DURING, Relation.AT_TIME_OF}:  # incidents open at that moment
        since, until = start, (end if r is Relation.DURING else start) + SECOND
        if anchor.kind == "date" and r is Relation.DURING:
            until = end  # the whole day, end exclusive
        overlap = query.target == "incident"
    elif r is Relation.BEFORE:
        since, until = start - window, start
    else:  # AFTER
        since, until = start, start + window
    args: dict[str, Any] = {
        "services": _services(query, anchor),
        "since": _iso(since) if since else None,
        "until": _iso(until) if until else None,
        "order": order,
    }
    if query.target == "deployment":
        tool = "search_deployments"
        args["limit"] = 1 if one else 50
        if r is Relation.AT_TIME_OF:
            live = [s for s in query.statuses if s in LIVE_STATUSES]
            args["statuses"] = live or LIVE_STATUSES
        elif query.statuses:
            args["statuses"] = query.statuses
    else:
        tool = "search_incidents"
        args |= {"top_k": 1 if one else 20, "include_postmortems": False}
        if overlap:
            args["overlap"] = True
    what = f"{r.value.replace('_', ' ')} {anchor.name}"
    return PlannedCall(tool=tool, arguments=args, purpose=f"{TARGET}: {query.target}s {what}")


# --- the timeline evidence ----------------------------------------------------------------


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _when(value: datetime) -> str:
    return f"{_utc(value):%Y-%m-%d %H:%M} UTC"


def _gap(earlier: datetime, later: datetime) -> str:
    seconds = int((_utc(later) - _utc(earlier)).total_seconds())
    days, rest = divmod(abs(seconds), 86400)
    hours, rest = divmod(rest, 3600)
    minutes = rest // 60
    parts = [f"{days} d"] if days else []
    if hours:
        parts.append(f"{hours} h")
    if minutes and not days:
        parts.append(f"{minutes} min")
    return " ".join(parts) or "0 min"


def _deployment(d: DeploymentRecord) -> str:
    return f"{d.id}, {d.service_id} {d.version} ({d.status.value}), deployed {_when(d.deployed_at)}"


def _incident(i: IncidentSummary) -> str:
    return f"{i.id} ({i.service_id}, {i.severity.value}), {i.title}, started {_when(i.started_at)}"


_PHRASE = {
    Relation.IMMEDIATELY_BEFORE: "{kind} immediately before {anchor}",
    Relation.IMMEDIATELY_AFTER: "first {kind} after {anchor}",
    Relation.PREVIOUS: "previous {kind} before {anchor}",
    Relation.NEXT: "next {kind} after {anchor}",
    Relation.LATEST: "latest {kind} as of {anchor}",
    Relation.AT_TIME_OF: "{kind} {live} at the time of {anchor}",
    Relation.DURING: "{kind} during {anchor}",
    Relation.BEFORE: "{kind} in the {window} before {anchor}",
    Relation.AFTER: "{kind} in the {window} after {anchor}",
}


def single(query: TemporalQuery) -> bool:
    """Whether the question asks for one record (the nearest) rather than all in a span.
    Several incidents can be open at one moment; only one deployment is live."""
    if query.relation is Relation.AT_TIME_OF:
        return query.target == "deployment"
    return query.relation in SINGLE


def timeline_item(query: TemporalQuery, anchor: Anchor, output: Any) -> EvidenceItem:
    """The answer to the temporal question, stated from the records' timestamps.

    The item is derived from the anchor and the target records, so reading it needs
    every grant they need: ``facts["requires"]`` lists them for the access checks."""
    scope = _services(query, anchor)
    window = _gap(anchor.start - (query.window or DEFAULT_WINDOW), anchor.start)
    kind = query.target if single(query) else f"{query.target}s"
    if query.statuses and query.relation is not Relation.AT_TIME_OF:
        kind = f"{' or '.join(s.replace('_', ' ') for s in query.statuses)} {kind}"
    if query.target == "deployment" and scope:
        kind += f" of {', '.join(scope)}"
    elif query.all_services:
        kind += " on any service"
    live = "live" if query.target == "deployment" else "open"
    phrase = _PHRASE[query.relation].format(
        kind=kind, live=live, anchor=anchor.label, window=window
    )
    records: list[Any]
    if query.target == "deployment":
        records = list(getattr(output, "deployments", []))
        stamps = [d.deployed_at for d in records]
        described = [_deployment(d) for d in records]
        resource, levels = Resource.DEPLOYMENTS, [UNLABELLED_LEVEL] * len(records)
    else:
        records = [i for i in getattr(output, "incidents", []) if i.id != anchor.id]
        stamps = [i.started_at for i in records]
        described = [_incident(i) for i in records]
        resource, levels = Resource.INCIDENTS, [i.access_level for i in records]
    heading = phrase[0].upper() + phrase[1:]
    if not records:
        text = f"{heading}: none found."
    elif single(query):
        relative = ""
        if anchor.kind != "now":
            direction = "before" if _utc(stamps[0]) <= _utc(anchor.start) else "after"
            relative = f", {_gap(stamps[0], anchor.start)} {direction} {anchor.name}"
        text = f"{heading}: {described[0]}{relative}."
    else:
        text = f"{heading}, in time order: " + "; ".join(described) + "."
    if query.relation is Relation.AT_TIME_OF and query.target == "deployment":
        text += " Failed deployments never served traffic and are not counted."
    requires = {(resource, level) for level in levels} or {(resource, UNLABELLED_LEVEL)}
    if anchor.resource is not None:
        requires.add((anchor.resource, anchor.access_level))
    return EvidenceItem(
        kind=EvidenceKind.TIMELINE,
        source_id=f"timeline:{query.relation.value}:{anchor.name}",
        title=f"Timeline: {phrase}",
        text=text,
        service_id=scope[0] if scope and len(scope) == 1 else None,
        timestamp=anchor.start,
        access_level=most_restrictive(level for _, level in requires),
        tool="temporal",
        pinned=True,
        facts={
            "answer": text,
            "records": ", ".join(r.id for r in records),
            "requires": ",".join(sorted(f"{r.value}:{level.value}" for r, level in requires)),
        },
    )


__all__ = [
    "ANCHOR",
    "TARGET",
    "Anchor",
    "Relation",
    "TemporalQuery",
    "anchor_call",
    "anchor_from",
    "fixed_anchor",
    "parse_temporal",
    "single",
    "target_call",
    "timeline_item",
    "tools_for",
]
