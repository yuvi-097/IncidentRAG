"""Side-by-side comparison of several retrievers on the same benchmark.

Pure functions over ``BenchmarkReport`` objects: subset summaries, per-question
rank matrices and a Markdown rendering. Running the retrievers is the caller's job.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from pydantic import BaseModel

from app.evaluation.retrieval import BenchmarkReport, QuestionResult, Summary, summarize

Subset = Callable[[QuestionResult], bool]

DEFAULT_SUBSETS: dict[str, Subset] = {
    "all": lambda r: True,
    "natural-language": lambda r: r.category != "exact-match",
    "exact-match": lambda r: r.category == "exact-match",
}


class Comparison(BaseModel):
    systems: list[str]
    labels: dict[str, str]  # system -> display name
    subsets: dict[str, dict[str, Summary]]  # subset -> system -> summary
    categories: dict[str, dict[str, Summary]]  # category -> system -> summary
    first_relevant_rank: dict[str, dict[str, int | None]]  # question -> system -> rank
    questions: dict[str, str]  # question id -> text
    config: dict[str, object]


def compare(
    reports: Mapping[str, BenchmarkReport],
    labels: Mapping[str, str] | None = None,
    subsets: Mapping[str, Subset] | None = None,
    config: Mapping[str, object] | None = None,
) -> Comparison:
    systems = list(reports)
    if not systems:
        raise ValueError("nothing to compare")
    ids = [r.id for r in reports[systems[0]].results]
    for name, report in reports.items():
        if [r.id for r in report.results] != ids:
            raise ValueError(f"{name} was evaluated on different questions")
    ks = reports[systems[0]].ks
    subset_rules = subsets or DEFAULT_SUBSETS
    categories = sorted({r.category for r in reports[systems[0]].results})
    return Comparison(
        systems=systems,
        labels={s: (labels or {}).get(s, s) for s in systems},
        subsets={
            subset: {s: summarize([r for r in reports[s].results if rule(r)], ks) for s in systems}
            for subset, rule in subset_rules.items()
        },
        categories={
            category: {
                s: summarize([r for r in reports[s].results if r.category == category], ks)
                for s in systems
            }
            for category in categories
        },
        first_relevant_rank={
            qid: {s: reports[s].results[i].first_relevant_rank for s in systems}
            for i, qid in enumerate(ids)
        },
        questions={r.id: r.question for r in reports[systems[0]].results},
        config=dict(config or {}),
    )


def _row(cells: Sequence[object]) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def to_markdown(comparison: Comparison) -> str:
    systems, labels = comparison.systems, comparison.labels
    lines: list[str] = []
    header = ["System", "Recall@5", "Recall@10", "MRR@10", "NDCG@10", "Median latency"]
    for subset, summaries in comparison.subsets.items():
        count = next(iter(summaries.values())).questions
        lines += [f"### {subset} ({count} questions)", "", _row(header)]
        lines.append(_row(["---"] * len(header)))
        for s in systems:
            m = summaries[s]
            lines.append(
                _row(
                    [
                        labels[s],
                        f"{m.recall[5]:.3f}",
                        f"{m.recall[10]:.3f}",
                        f"{m.mrr:.3f}",
                        f"{m.ndcg[10]:.3f}",
                        f"{m.latency_ms_median} ms",
                    ]
                )
            )
        lines.append("")

    lines += ["### NDCG@10 by category", ""]
    lines.append(_row(["Category", "n", *[labels[s] for s in systems]]))
    lines.append(_row(["---"] * (len(systems) + 2)))
    for category, summaries in comparison.categories.items():
        count = next(iter(summaries.values())).questions
        lines.append(_row([category, count, *[f"{summaries[s].ndcg[10]:.3f}" for s in systems]]))
    lines.append("")

    lines += ["### Rank of the first relevant result (- = not in the top 10)", ""]
    lines.append(_row(["Question", *[labels[s] for s in systems], "Text"]))
    lines.append(_row(["---"] * (len(systems) + 2)))
    for qid, ranks in comparison.first_relevant_rank.items():
        cells = [ranks[s] if ranks[s] else "-" for s in systems]
        lines.append(_row([qid, *cells, comparison.questions[qid][:80].replace("|", "/")]))
    lines.append("")
    return "\n".join(lines)
