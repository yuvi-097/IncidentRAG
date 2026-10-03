"""References to records the caller may not read, removed from evidence text.

A document the caller may read can mention one they may not: "see DOC-0060 (Payment
Service Configuration Reference)", an incident naming the pull request behind it, an
SQL row carrying a runbook id. Access control keeps the *content* of those records away,
but their ids and titles would still leak. ``ReferenceFilter`` finds every such
reference in the evidence and replaces it before ranking, packaging or any model call:

- ids of documents, runbooks, postmortems, reports and policies (DOC-, RB-, PM-, RPT-,
  POL-), incidents (INC-), code files (CF-) and pull requests (PR-), checked against
  the caller's grants in the database; with a following "(Title)", that goes too;
- exact titles of documents the caller may not read, when no document they may read
  has the same title.

Deployments and log lines carry no sensitivity label (they are engineering material),
so references to them stay: not having the deployments grant limits the tools a role
may call, not what it may be told exists.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Any

from sqlalchemy import not_, select
from sqlalchemy.engine import Engine

from app.agents.state import EvidenceItem
from app.database.models import CodeFile, Document, Incident, PullRequest
from app.schemas.enums import Resource
from app.security.principal import Principal
from app.tools.common import document_access, visible_pull_requests

HIDDEN = "[restricted reference]"
HIDDEN_TITLE = "[restricted document]"
_ID = re.compile(r"\b(?:DOC|RB|PM|RPT|POL|INC|CF)-\d{4}\b|\bPR-\d{3,5}\b")
MIN_TITLE = 12  # shorter titles are too generic to replace safely
# An optional "(Title)" after an id, allowing one level of nested parentheses.
_PARENTHETICAL = r"(?:\s*\((?:[^()\n]|\([^()\n]*\)){1,200}\))?"
_LEFTOVER = re.compile(re.escape(HIDDEN) + r"\s*\(\W*" + re.escape(HIDDEN_TITLE) + r"\W*\)")


class ReferenceFilter:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def hidden(self, principal: Principal, texts: Iterable[str]) -> tuple[set[str], set[str]]:
        """(ids, titles) mentioned in ``texts`` or hidden anywhere, that the caller may
        not read."""
        ids = {m for text in texts for m in _ID.findall(text)}
        by_prefix: dict[str, set[str]] = {}
        for identifier in ids:
            by_prefix.setdefault(identifier.split("-")[0], set()).add(identifier)
        documents = set().union(
            *(by_prefix.get(p, set()) for p in ("DOC", "RB", "PM", "RPT", "POL"))
        )
        hidden: set[str] = set()
        with self.engine.connect() as connection:
            if documents:
                existing = set(
                    connection.execute(
                        select(Document.id).where(Document.id.in_(sorted(documents)))
                    ).scalars()
                )
                readable = set(
                    connection.execute(
                        select(Document.id).where(
                            Document.id.in_(sorted(documents)), document_access(principal)
                        )
                    ).scalars()
                )
                hidden |= existing - readable
            incidents = by_prefix.get("INC", set())
            if incidents:
                labels = principal.visible_levels(Resource.INCIDENTS)
                hidden |= set(
                    connection.execute(
                        select(Incident.id).where(
                            Incident.id.in_(sorted(incidents)),
                            not_(Incident.access_level.in_(labels))
                            if labels
                            else Incident.id.isnot(None),
                        )
                    ).scalars()
                )
            files = by_prefix.get("CF", set())
            if files:
                labels = principal.visible_levels(Resource.CODE)
                hidden |= set(
                    connection.execute(
                        select(CodeFile.id).where(
                            CodeFile.id.in_(sorted(files)),
                            not_(CodeFile.access_level.in_(labels))
                            if labels
                            else CodeFile.id.isnot(None),
                        )
                    ).scalars()
                )
            prs = by_prefix.get("PR", set())
            if prs:
                hidden |= prs - visible_pull_requests(connection, prs, principal)
            titles = self._hidden_titles(connection, principal)
            titles |= self._titles_of_hidden(connection, principal, hidden)
        return hidden, titles

    @staticmethod
    def _titles_of_hidden(connection: Any, principal: Principal, hidden: set[str]) -> set[str]:
        """Titles of the hidden incidents and pull requests that were mentioned, unless a
        record of the same kind the caller may read has the same title."""
        titles: set[str] = set()
        incidents = sorted(i for i in hidden if i.startswith("INC-"))
        if incidents:
            candidates = set(
                connection.execute(
                    select(Incident.title).where(Incident.id.in_(incidents))
                ).scalars()
            )
            labels = principal.visible_levels(Resource.INCIDENTS)
            shared = (
                set(
                    connection.execute(
                        select(Incident.title).where(
                            Incident.title.in_(sorted(candidates)),
                            Incident.access_level.in_(labels),
                        )
                    ).scalars()
                )
                if labels and candidates
                else set()
            )
            titles |= candidates - shared
        prs = sorted(i for i in hidden if i.startswith("PR-"))
        if prs:
            candidates = set(
                connection.execute(
                    select(PullRequest.title).where(PullRequest.id.in_(prs))
                ).scalars()
            )
            same_title = set(
                connection.execute(
                    select(PullRequest.id).where(PullRequest.title.in_(sorted(candidates)))
                ).scalars()
            )
            visible = visible_pull_requests(connection, same_title, principal)
            shared = set(
                connection.execute(
                    select(PullRequest.title).where(PullRequest.id.in_(sorted(visible)))
                ).scalars()
            )
            titles |= candidates - shared
        return {t for t in titles if len(t) >= MIN_TITLE}

    @staticmethod
    def _hidden_titles(connection: object, principal: Principal) -> set[str]:
        rows = list(connection.execute(select(Document.title, document_access(principal))))  # type: ignore[attr-defined]
        readable = {title for title, ok in rows if ok}
        return {
            title
            for title, ok in rows
            if not ok and title not in readable and len(title) >= MIN_TITLE
        }

    def redact(
        self, principal: Principal, items: Sequence[EvidenceItem]
    ) -> tuple[list[EvidenceItem], int]:
        """``items`` with hidden references replaced, and how many were replaced."""
        texts = [t for item in items for t in (item.title, item.text, *item.facts.values())]
        ids, titles = self.hidden(principal, texts)
        own = {item.source_id for item in items}
        ids -= own  # an item never hides itself (it passed the access check)
        if not ids and not titles:
            return list(items), 0
        titles_by_length = sorted(titles, key=len, reverse=True)
        pattern = (
            re.compile(
                r"\b(?:" + "|".join(re.escape(i) for i in sorted(ids)) + r")\b" + _PARENTHETICAL
            )
            if ids
            else None
        )
        count = 0

        def clean(text: str) -> str:
            nonlocal count
            if pattern is not None:
                text, n = pattern.subn(HIDDEN, text)
                count += n
            for title in titles_by_length:
                if title in text:
                    count += text.count(title)
                    text = text.replace(title, HIDDEN_TITLE)
            # "[restricted reference] ("[restricted document]")" reads as one reference
            return _LEFTOVER.sub(HIDDEN, text)

        cleaned = []
        for item in items:
            title, text = clean(item.title), clean(item.text)
            facts = {key: clean(value) for key, value in item.facts.items()}
            if (title, text, facts) != (item.title, item.text, item.facts):
                item = item.model_copy(update={"title": title, "text": text, "facts": facts})
            cleaned.append(item)
        return cleaned, count


__all__ = ["HIDDEN", "HIDDEN_TITLE", "ReferenceFilter"]
