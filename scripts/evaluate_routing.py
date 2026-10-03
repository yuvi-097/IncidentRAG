"""Evaluate the query router on the labelled routing set.

    python scripts/evaluate_routing.py [--cases data/evaluation/routing_benchmark.jsonl]
        [--now 2026-09-01T00:00:00+00:00] [--output data/evaluation/results/routing.json]

Needs no database: services come from the generated dataset. ``--now`` fixes the
clock for relative time expressions (default: the dataset's window end).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.agents.entities import ServiceCatalog  # noqa: E402
from app.agents.router import RuleBasedRouter  # noqa: E402
from app.evaluation.retrieval import load_benchmark  # noqa: E402
from app.evaluation.routing import cross_check, evaluate_router, load_cases  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--cases", type=Path, default=ROOT / "data/evaluation/routing_benchmark.jsonl"
    )
    parser.add_argument("--data", type=Path, default=ROOT / "data/generated")
    parser.add_argument("--now", type=datetime.fromisoformat)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/evaluation/results/routing.json"
    )
    args = parser.parse_args(argv)

    manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    now = args.now or datetime.fromisoformat(manifest["window_end"].replace("Z", "+00:00"))
    services = [
        json.loads(line)["id"]
        for line in (args.data / "services.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    router = RuleBasedRouter(ServiceCatalog(services), clock=lambda: now)
    report = evaluate_router(router, load_cases(args.cases))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )

    print(
        f"Routing: {report.cases} cases, accuracy {report.accuracy:.3f} "
        f"(type only {report.type_accuracy:.3f}); clock {now.isoformat()}\n"
    )
    print(f"  {'type':<18}{'precision':>10}{'recall':>8}{'n':>4}")
    for name, m in report.per_type.items():
        print(f"  {name:<18}{m['precision']:>10.3f}{m['recall']:>8.3f}{int(m['support']):>4}")
    misses = [o for o in report.outcomes if not o.correct]
    print(f"\n  misses ({len(misses)}):")
    for o in misses:
        extra = f" missing {o.missing_tools}" if o.missing_tools else ""
        print(f"  {o.id} expected {o.expected_type.value}, got {o.predicted_type.value}{extra}")
        print(f"         {o.query}\n         -> {o.summary}")

    retrieval = load_benchmark(ROOT / "data/evaluation/retrieval_benchmark.jsonl")
    checks = cross_check(router, [(q.id, q.question, q.category, q.notes) for q in retrieval])
    agree = sum(c.agrees for c in checks)
    print(
        f"\nCross-check on {len(checks)} retrieval-benchmark questions (written before the "
        f"router): {agree}/{len(checks)} = {agree / len(checks):.3f} agree with their category"
    )
    for c in checks:
        if not c.agrees:
            allowed = "/".join(t.value for t in c.acceptable)
            print(f"  {c.id} [{c.category}] expected {allowed}, got {c.predicted_type.value}")
            print(f"         {c.question}\n         -> {c.summary}")
    payload = report.model_dump(mode="json")
    payload["cross_check"] = [c.model_dump(mode="json") for c in checks]
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nFull report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
