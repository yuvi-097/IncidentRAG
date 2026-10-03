"""The text that is embedded for a chunk.

A chunk alone often lacks context (a method body does not say which file or
service it belongs to), so the embedded text starts with a short header built
from the chunk's own metadata, followed by the chunk content. The header uses only
facts stored on the chunk, so nothing about specific questions is baked in.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.schemas.enums import SourceType

SOURCE_LABELS = {
    SourceType.INCIDENT: "Incident",
    SourceType.RUNBOOK: "Runbook",
    SourceType.DOCUMENTATION: "Documentation",
    SourceType.POSTMORTEM: "Postmortem",
    SourceType.DEPLOYMENT: "Deployment",
    SourceType.CODE: "Code",
    SourceType.PULL_REQUEST: "Pull request",
}


def embedding_text(chunk: Mapping[str, Any]) -> str:
    """``chunk`` provides source_type, title, section, service_id and content."""
    source_type = SourceType(chunk["source_type"])
    lines = [f"{SOURCE_LABELS[source_type]}: {chunk['title']}"]
    section = chunk.get("section")
    if section and section != chunk["title"] and not str(chunk["title"]).endswith(section):
        lines.append(f"Section: {section}")
    if chunk.get("service_id"):
        lines.append(f"Service: {chunk['service_id']}")
    return "\n".join(lines) + "\n\n" + str(chunk["content"])
