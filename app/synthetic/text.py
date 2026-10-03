"""Small text helpers shared by the generators."""

from __future__ import annotations

import hashlib
import re
import textwrap

_TOKEN = re.compile(r"\{\{(\w+)\}\}")


def render(template: str, **values: object) -> str:
    """Dedent ``template`` and substitute ``{{name}}`` tokens.

    ``{{...}}`` never collides with Python/YAML/Markdown syntax used in templates.
    A missing value raises instead of silently leaving a token behind.
    """

    def substitute(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in values:
            raise KeyError(f"template value missing: {key}")
        return str(values[key])

    return _TOKEN.sub(substitute, textwrap.dedent(template).lstrip("\n"))


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def camel(service_id: str) -> str:
    """ "payment-service" -> "PaymentService"."""
    return "".join(part.capitalize() for part in service_id.split("-"))


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def bullet_list(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)
