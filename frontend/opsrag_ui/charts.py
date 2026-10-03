"""Altair charts in one visual system.

- One series -> one colour (the methods on an axis are categories, not series).
- Magnitude over a grid -> one blue ramp, light (low) to dark (high); labels switch to
  white on dark cells.
- Thin bars with rounded data ends, a hairline grid, no dual axes; every mark has a
  tooltip, and a table view of the same numbers sits next to each chart on its page.
"""

from __future__ import annotations

import altair as alt
import pandas as pd

from opsrag_ui.theme import BASELINE, GRID, INK, INK_2, MUTED, RAMP, SERIES, SURFACE

TITLE_AND_AXIS = 64  # px: Streamlit sizes charts to fit, so the height includes these


def _style(chart: alt.Chart, height: int) -> alt.Chart:
    """``height`` is the whole chart, title and axes included (Streamlit's autosize)."""
    return (
        chart.properties(height=height, background=SURFACE)
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelColor=INK_2,
            titleColor=MUTED,
            gridColor=GRID,
            domainColor=BASELINE,
            tickColor=BASELINE,
            labelFontSize=11,
            titleFontSize=11,
            titleFontWeight="normal",
        )
        .configure_title(color=INK, fontSize=13, anchor="start", fontWeight=600)
    )


def bars(
    data: pd.DataFrame,
    category: str,
    value: str,
    title: str = "",
    percent: bool = False,
    domain: tuple[float, float] | None = None,
    height: int = 300,
    sort: list[str] | None = None,
    tooltip: list[str] | None = None,
) -> alt.Chart:
    """Vertical bars for one measure across categories (e.g. methods A-F)."""
    fmt = ".0%" if percent else ".2f"
    scale = alt.Scale(domain=list(domain)) if domain else alt.Undefined
    base = alt.Chart(data, title=title).encode(
        x=alt.X(f"{category}:N", sort=sort or None, axis=alt.Axis(labelAngle=0, title=None)),
        y=alt.Y(f"{value}:Q", scale=scale, axis=alt.Axis(format=fmt, title=None, tickCount=5)),
    )
    marks = base.mark_bar(color=SERIES, size=26, cornerRadiusEnd=4).encode(
        tooltip=tooltip or [alt.Tooltip(f"{category}:N"), alt.Tooltip(f"{value}:Q", format=fmt)]
    )
    labels = base.mark_text(dy=-7, color=INK_2, fontSize=10).encode(
        text=alt.Text(f"{value}:Q", format=fmt)
    )
    return _style(marks + labels, height)


def hbars(
    data: pd.DataFrame,
    category: str,
    value: str,
    title: str = "",
    fmt: str = ",.0f",
    height: int | None = None,
    tooltip: list[str] | None = None,
    sort: list[str] | None = None,
    domain: tuple[float, float] | None = None,
) -> alt.Chart:
    """Horizontal bars for long category names (stages, systems in narrow panels).

    Longest first, unless ``sort`` fixes the order (small multiples keep one order and,
    with ``domain``, one scale, so panels can be compared).
    """
    scale = alt.Scale(domain=list(domain)) if domain else alt.Undefined
    base = alt.Chart(data, title=title).encode(
        y=alt.Y(f"{category}:N", sort=sort or "-x", axis=alt.Axis(title=None, labelLimit=260)),
        x=alt.X(f"{value}:Q", scale=scale, axis=alt.Axis(title=None, format=fmt, tickCount=4)),
    )
    marks = base.mark_bar(color=SERIES, size=14, cornerRadiusEnd=4).encode(
        tooltip=tooltip or [alt.Tooltip(f"{category}:N"), alt.Tooltip(f"{value}:Q", format=fmt)]
    )
    labels = base.mark_text(align="left", dx=4, color=INK_2, fontSize=10).encode(
        text=alt.Text(f"{value}:Q", format=fmt)
    )
    return _style(marks + labels, height or 28 * len(data) + TITLE_AND_AXIS)


def heatmap(
    data: pd.DataFrame,
    row: str,
    column: str,
    value: str,
    title: str = "",
    percent: bool = True,
    row_sort: list[str] | None = None,
    column_sort: list[str] | None = None,
) -> alt.Chart:
    """A grid of magnitudes on the blue ramp, with the value in each cell."""
    fmt = ".0%" if percent else ",.0f"
    top = float(data[value].max() or 1.0) if not percent else 1.0
    base = alt.Chart(data, title=title).encode(
        x=alt.X(
            f"{column}:N",
            sort=column_sort or None,
            axis=alt.Axis(orient="top", labelAngle=0, title=None),
        ),
        y=alt.Y(f"{row}:N", sort=row_sort or None, axis=alt.Axis(title=None, labelLimit=220)),
    )
    cells = base.mark_rect(stroke=SURFACE, strokeWidth=2).encode(
        color=alt.Color(
            f"{value}:Q",
            scale=alt.Scale(range=RAMP, domain=[0, top]),
            legend=alt.Legend(format=fmt, title=None, orient="bottom", gradientLength=180),
        ),
        tooltip=[
            alt.Tooltip(f"{row}:N"),
            alt.Tooltip(f"{column}:N"),
            alt.Tooltip(f"{value}:Q", format=fmt),
        ],
    )
    text = base.mark_text(fontSize=11).encode(
        text=alt.Text(f"{value}:Q", format=fmt),
        color=alt.condition(f"datum.{value} > {top * 0.55}", alt.value("#ffffff"), alt.value(INK)),
    )
    # Rows, plus the title, the column labels on top and the legend below.
    return _style(cells + text, 32 * data[row].nunique() + TITLE_AND_AXIS + 56)


__all__ = ["bars", "hbars", "heatmap"]
