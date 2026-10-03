"""Phase 10 aggregation, tables, plots and the written report.

Everything here is computed from the per-question records of one run; no number is
typed in. The plots follow one rule set: one colour per series (the methods are
categories on an axis, not series), a sequential single-hue ramp for the heatmaps,
thin bars with value labels, recessive grid, no dual axes.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from app.evaluation.errors import ORDER, ErrorClass
from app.evaluation.eval_set import Category, EvalQuestion
from app.evaluation.pipelines import METHODS


class Record(BaseModel):
    """One (question, method) result, as saved to per_question.jsonl."""

    question_id: str
    category: str
    difficulty: str
    origin: str
    role: str
    method: str
    question: str
    answer: str
    correct: bool
    abstained: bool
    checks: dict[str, Any]
    retrieval: dict[str, Any] | None  # None: the question has no gold sources
    generation: dict[str, Any]
    nli_faithfulness: float | None
    judge: dict[str, Any] | None
    error: dict[str, Any]
    ranked: list[str]
    context: list[str]
    query_type: str | None
    plan: str | None
    tools: list[str]
    confidence: str | None
    security: dict[str, Any]
    latency_ms: float


def _mean(values: Sequence[float | None]) -> float | None:
    kept = [v for v in values if v is not None]
    return round(statistics.fmean(kept), 4) if kept else None


def method_summary(records: Sequence[Record]) -> dict[str, Any]:
    retrieval = [r.retrieval for r in records if r.retrieval]
    gen = [r.generation for r in records]
    answered = [r for r in records if not r.abstained]
    no_answer = [r for r in records if r.category == Category.NO_ANSWER.value]
    answerable = [
        r
        for r in records
        if r.category not in {Category.NO_ANSWER.value, Category.INJECTION.value}
        and r.checks.get("has_positive")
    ]
    summary: dict[str, Any] = {
        "questions": len(records),
        "correctness": _mean([float(r.correct) for r in records]),
        "retrieval_questions": len(retrieval),
        "recall_1": _mean([x["recall"]["1"] for x in retrieval]),
        "recall_5": _mean([x["recall"]["5"] for x in retrieval]),
        "recall_10": _mean([x["recall"]["10"] for x in retrieval]),
        # Recall@k is capped (found / min(relevant, k)), so with several relevant records
        # Recall@1 can exceed Recall@5; Hit@k (any relevant record in the top k) is monotone.
        "hit_1": _mean([x["hit"]["1"] for x in retrieval]),
        "hit_5": _mean([x["hit"]["5"] for x in retrieval]),
        "hit_10": _mean([x["hit"]["10"] for x in retrieval]),
        "precision_5": _mean([x["precision_5"] for x in retrieval]),
        "mrr": _mean([x["reciprocal_rank"] for x in retrieval]),
        "ndcg_10": _mean([x["ndcg_10"] for x in retrieval]),
        "faithfulness": _mean([g["faithfulness"] for g in gen]),
        "nli_faithfulness": _mean([r.nli_faithfulness for r in records]),
        "citation_correctness": _mean([g["citation_correct"] for g in gen]),
        "hallucination_rate": _mean(
            [float(g["hallucinated"]) for g in gen if g["hallucinated"] is not None]
        ),
        "context_relevance": _mean([g["context_relevance"] for g in gen]),
        # Answers with a cited claim its citation does not ground, and how many of them
        # carry the verifier's "Sources disagree ... [E#] state(s) otherwise" note: that
        # note cites the *disagreeing* source on purpose, which this metric counts as a miss.
        "citation_issue_answers": sum(
            1 for g in gen if g["citation_correct"] is not None and g["citation_correct"] < 1
        ),
        "citation_issues_with_disagreement_note": sum(
            1
            for r in records
            if r.generation["citation_correct"] is not None
            and r.generation["citation_correct"] < 1
            and "Sources disagree" in r.answer
        ),
        "answered": len(answered),
        "no_answer_declined": _mean([float(r.abstained) for r in no_answer]),
        "false_decline_rate": _mean([float(r.abstained) for r in answerable]),
        "latency_ms_median": round(statistics.median([r.latency_ms for r in records]), 1)
        if records
        else None,
    }
    judged = [r.judge for r in records if r.judge and r.judge.get("scores")]
    if judged:
        for name in ("correctness", "faithfulness", "context_relevance"):
            summary[f"judge_{name}"] = _mean([j["scores"][name] for j in judged])
    return summary


def summarize(records: Sequence[Record], questions: Sequence[EvalQuestion]) -> dict[str, Any]:
    by_method: dict[str, list[Record]] = defaultdict(list)
    for r in records:
        by_method[r.method].append(r)
    out: dict[str, Any] = {"methods": {}, "by_category": {}, "by_difficulty": {}, "by_origin": {}}
    for method, rs in sorted(by_method.items()):
        out["methods"][method] = method_summary(rs)
        for field, key in (
            ("by_category", "category"),
            ("by_difficulty", "difficulty"),
            ("by_origin", "origin"),
        ):
            groups: dict[str, list[Record]] = defaultdict(list)
            for r in rs:
                value = getattr(r, key)
                groups[value.split(":")[0] if key == "origin" else value].append(r)
            out[field][method] = {
                g: {"questions": len(v), "correctness": _mean([float(x.correct) for x in v])}
                for g, v in sorted(groups.items())
            }
        errors = Counter(r.error["primary"] for r in rs if r.error.get("failed"))
        classes = Counter(c for r in rs if r.error.get("failed") for c in r.error["classes"])
        out.setdefault("errors", {})[method] = {
            "failed": sum(1 for r in rs if r.error.get("failed")),
            "primary": {c.value: errors.get(c.value, 0) for c in ORDER},
            "any": {c.value: classes.get(c.value, 0) for c in ORDER},
        }
    out["categories"] = dict(Counter(q.category.value for q in questions))
    return out


# --- tables ---------------------------------------------------------------------------------


def _fmt(value: float | None, pct: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%" if pct else f"{value:.3f}"


def ablation_table(summary: dict[str, Any]) -> str:
    head = (
        "| Method | R@1 | R@5 | R@10 | Hit@5 | P@5 | MRR | NDCG@10 | Correct | Faithful | "
        "Faithful (NLI) | Citation correct | Hallucination | Context relevance |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    rows = []
    for method, s in summary["methods"].items():
        rows.append(
            f"| {method}. {METHODS[method]} | {_fmt(s['recall_1'])} | {_fmt(s['recall_5'])} | "
            f"{_fmt(s['recall_10'])} | {_fmt(s.get('hit_5'))} | {_fmt(s['precision_5'])} | "
            f"{_fmt(s['mrr'])} | "
            f"{_fmt(s['ndcg_10'])} | {_fmt(s['correctness'], True)} | "
            f"{_fmt(s['faithfulness'], True)} | {_fmt(s['nli_faithfulness'], True)} | "
            f"{_fmt(s['citation_correctness'], True)} | {_fmt(s['hallucination_rate'], True)} | "
            f"{_fmt(s['context_relevance'], True)} |"
        )
    return head + "\n".join(rows)


def grouped_table(summary: dict[str, Any], field: str, order: Sequence[str] | None = None) -> str:
    methods = list(summary[field])
    groups = order or sorted({g for m in methods for g in summary[field][m]})
    head = "| Group | n | " + " | ".join(f"{m}" for m in methods) + " |\n"
    head += "|---|---:|" + "---:|" * len(methods) + "\n"
    rows = []
    for g in groups:
        cells = [summary[field][m].get(g, {}) for m in methods]
        n = next((c["questions"] for c in cells if c), 0)
        values = " | ".join(_fmt(c.get("correctness"), True) if c else "n/a" for c in cells)
        rows.append(f"| {g} | {n} | {values} |")
    return head + "\n".join(rows)


def error_table(summary: dict[str, Any]) -> str:
    methods = list(summary["errors"])
    head = "| Primary class | " + " | ".join(methods) + " |\n|---|" + "---:|" * len(methods) + "\n"
    rows = [
        f"| {c.value} | "
        + " | ".join(str(summary["errors"][m]["primary"][c.value]) for m in methods)
        + " |"
        for c in ORDER
    ]
    rows.append(
        "| **failed (any class)** | "
        + " | ".join(f"**{summary['errors'][m]['failed']}**" for m in methods)
        + " |"
    )
    return head + "\n".join(rows)


# --- plots ------------------------------------------------------------------------------------

INK, INK_2, MUTED, GRID, BASE, SURFACE = (
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
    "#fcfcfb",
)
SERIES_1 = "#2a78d6"
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _axes_style(ax: Any) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASE)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _small_multiples(
    summary: dict[str, Any], metrics: Sequence[tuple[str, str, bool]], title: str, path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = list(summary["methods"])
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.3 * len(metrics), 3.4), facecolor=SURFACE)
    for ax, (key, label, pct) in zip(axes, metrics, strict=True):
        values = [summary["methods"][m].get(key) for m in methods]
        heights = [v if v is not None else 0.0 for v in values]
        bars = ax.bar(methods, heights, width=0.55, color=SERIES_1, edgecolor=SURFACE, linewidth=2)
        _axes_style(ax)
        ax.set_ylim(0, 1.08)
        ax.set_title(label, color=INK, fontsize=10, loc="left")
        for bar, v in zip(bars, values, strict=True):
            text = "n/a" if v is None else (f"{v * 100:.0f}%" if pct else f"{v:.2f}")
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                text,
                ha="center",
                va="bottom",
                fontsize=7,
                color=INK_2,
            )
    fig.suptitle(title, color=INK, fontsize=11, x=0.01, ha="left")
    fig.text(
        0.01,
        0.005,
        "  ".join(f"{m} = {METHODS[m]}" for m in methods),
        fontsize=7,
        color=MUTED,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _heatmap(
    rows: Sequence[str],
    cols: Sequence[str],
    values: list[list[float | None]],
    title: str,
    path: Path,
    fmt: str = "pct",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("blue", SEQUENTIAL)
    finite = [v for row in values for v in row if v is not None]
    top = max(finite) if finite and fmt == "count" else 1.0
    grid = [[(v if v is not None else float("nan")) for v in row] for row in values]
    fig, ax = plt.subplots(
        figsize=(1.0 + 0.9 * len(cols), 0.9 + 0.36 * len(rows)), facecolor=SURFACE
    )
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=top or 1, aspect="auto")
    ax.set_xticks(range(len(cols)), cols, fontsize=8, color=INK_2)
    ax.set_yticks(range(len(rows)), rows, fontsize=8, color=INK_2)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    for i, row in enumerate(values):
        for j, v in enumerate(row):
            if v is None:
                text, dark = "n/a", False
            else:
                text = f"{v * 100:.0f}%" if fmt == "pct" else f"{v:.0f}"
                dark = (v / (top or 1)) > 0.55
            ax.text(
                j, i, text, ha="center", va="center", fontsize=7, color="#ffffff" if dark else INK
            )
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plots(summary: dict[str, Any], directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    made = []
    path = directory / "retrieval_ablation.png"
    _small_multiples(
        summary,
        [
            ("recall_5", "Recall@5 (capped)", False),
            ("hit_5", "Hit@5", False),
            ("mrr", "MRR", False),
            ("ndcg_10", "NDCG@10", False),
        ],
        "Retrieval quality by method (questions with gold sources)",
        path,
    )
    made.append(path)
    path = directory / "generation_ablation.png"
    _small_multiples(
        summary,
        [
            ("correctness", "Answer correctness", True),
            ("faithfulness", "Faithfulness (deterministic)", True),
            ("citation_correctness", "Citation correctness", True),
            ("hallucination_rate", "Hallucination rate (lower is better)", True),
        ],
        "Answer quality by method (deterministic metrics)",
        path,
    )
    made.append(path)
    methods = list(summary["methods"])
    categories = [c.value for c in Category]
    path = directory / "correctness_by_category.png"
    _heatmap(
        categories,
        methods,
        [
            [summary["by_category"][m].get(c, {}).get("correctness") for m in methods]
            for c in categories
        ],
        "Answer correctness by category and method",
        path,
    )
    made.append(path)
    path = directory / "error_classes.png"
    _heatmap(
        [c.value for c in ORDER],
        methods,
        [[float(summary["errors"][m]["primary"][c.value]) for m in methods] for c in ORDER],
        "Failures by primary error class (count of questions)",
        path,
        fmt="count",
    )
    made.append(path)
    return made


__all__ = [
    "ErrorClass",
    "Record",
    "ablation_table",
    "error_table",
    "grouped_table",
    "method_summary",
    "plots",
    "summarize",
]
