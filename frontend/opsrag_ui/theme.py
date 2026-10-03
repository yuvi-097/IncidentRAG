"""Colours, badges and small formatting helpers shared by every page.

Colour roles follow one rule set: a single series uses one colour; magnitude uses one
blue ramp; state (confidence, severity) uses the reserved status colours, and always
with an icon and a text label, never colour alone. Access labels are compartments, not a
scale, so they are drawn as neutral chips.

Everything that comes from the API is HTML-escaped before it is placed into markup.
"""

from __future__ import annotations

import html
import re
from datetime import datetime

import streamlit as st

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = "#2a78d6"
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}

CONFIDENCE = {
    "HIGH": ("good", "✓", "High confidence"),
    "MEDIUM": ("warning", "!", "Medium confidence"),
    "LOW": ("serious", "▲", "Low confidence"),
    "INSUFFICIENT_EVIDENCE": ("critical", "✕", "Insufficient evidence"),
}
SEVERITY = {
    "SEV1": ("critical", "✕"),
    "SEV2": ("serious", "▲"),
    "SEV3": ("warning", "!"),
    "SEV4": (None, "•"),
}

CSS = f"""
<style>
.block-container {{ padding-top: 2.2rem; max-width: 1280px; }}
h1, h2, h3 {{ letter-spacing: -0.01em; }}
.ops-badge {{
  display: inline-flex; align-items: center; gap: .35rem; padding: .12rem .55rem;
  border-radius: 999px; font-size: .82rem; font-weight: 600; color: {INK};
  border: 1px solid rgba(11,11,11,.10); white-space: nowrap;
}}
.ops-chip {{
  display: inline-block; padding: .05rem .45rem; border-radius: 6px; font-size: .78rem;
  color: {INK_2}; border: 1px solid {GRID}; background: #f4f3ef; margin-right: .25rem;
}}
.ops-cite {{
  display: inline-block; padding: 0 .3rem; margin: 0 .05rem; border-radius: 4px;
  font-size: .72rem; font-weight: 700; color: {SERIES}; background: #e8f1fc;
  vertical-align: 1px;
}}
.ops-answer {{ font-size: 1.0rem; line-height: 1.6; color: {INK}; }}
.ops-answer p {{ margin: 0 0 .45rem 0; }}
.ops-muted {{ color: {MUTED}; font-size: .85rem; }}
.ops-card {{
  border: 1px solid rgba(11,11,11,.08); border-radius: 12px; padding: .9rem 1rem;
  background: #ffffff;
}}
.ops-flow-step {{ border-left: 3px solid {SERIES}; padding: .2rem 0 .6rem .8rem; }}
.ops-recs {{ margin: .2rem 0 .4rem 1.1rem; padding: 0; color: {INK}; line-height: 1.55; }}
.ops-recs li {{ margin-bottom: .55rem; padding-left: .2rem; }}
.ops-recs .ops-chip {{ margin-right: .4rem; }}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def badge(text: str, role: str | None, icon: str) -> str:
    color = STATUS.get(role or "", MUTED)
    return (
        f'<span class="ops-badge" style="background:{color}22;border-color:{color}66">'
        f'<span style="color:{color}">{icon}</span>{html.escape(text)}</span>'
    )


def confidence_badge(level: str) -> str:
    role, icon, label = CONFIDENCE.get(level, (None, "•", level))
    return badge(label, role, icon)


def severity_badge(severity: str) -> str:
    role, icon = SEVERITY.get(severity, (None, "•"))
    return badge(severity, role, icon)


def chip(text: str) -> str:
    return f'<span class="ops-chip">{html.escape(str(text))}</span>'


def cite(label: str) -> str:
    """An evidence label ("E2") as the citation badge used in answers."""
    return f'<span class="ops-cite">{html.escape(label)}</span>'


_LABEL = re.compile(r"\[(E\d+)\]")


def answer_html(text: str) -> str:
    """The answer as safe HTML: escaped, one paragraph per line, citations as badges."""
    paragraphs = []
    for line in text.splitlines():
        if not line.strip():
            continue
        escaped = html.escape(line)
        paragraphs.append("<p>" + _LABEL.sub(r'<span class="ops-cite">\1</span>', escaped) + "</p>")
    return '<div class="ops-answer">' + "".join(paragraphs) + "</div>"


def when(value: str | None) -> str:
    if not value:
        return "—"
    try:
        moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    return f"{moment:%Y-%m-%d %H:%M} UTC"


def pct(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value * 100:.{digits}f}%"


def ms(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value / 1000:.2f} s" if value >= 1000 else f"{value:.0f} ms"


__all__ = [
    "BASELINE",
    "CONFIDENCE",
    "GRID",
    "INK",
    "INK_2",
    "MUTED",
    "RAMP",
    "SERIES",
    "STATUS",
    "SURFACE",
    "answer_html",
    "badge",
    "chip",
    "cite",
    "confidence_badge",
    "inject_css",
    "ms",
    "pct",
    "severity_badge",
    "when",
]
