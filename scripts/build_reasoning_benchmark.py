"""Build the Phase 9 benchmarks: temporal and multi-hop questions with gold answers.

    python scripts/build_reasoning_benchmark.py [--data-dir data/generated]

Gold answers are computed here, from the raw dataset records (data/generated/*.jsonl),
by a reference implementation that shares no code with the agent: sorting timestamps
and following ids. Anchors are sampled with a fixed seed. The output is written once and
then frozen (data/evaluation/temporal_benchmark.jsonl, multihop_benchmark.jsonl); the
evaluation never recomputes gold answers with the system under test.

Definitions used for the gold answers (documented in docs/TECHNICAL_REFERENCE.md):
- a deployment "happened" whatever its status (a failed canary happened too);
- "running at the time of" an incident = the latest deployment of the incident's service
  at or before its start that went live (status succeeded or rolled_back);
- the scope of "before/after <incident>" is the incident's own service;
- "latest"/"most recent" is relative to the evaluation clock, 2026-09-01T00:00:00Z;
- "during" an incident = between its start and its resolution;
- the commit of a change = the root-cause deployment's commit (= its PR's merge commit).
"""

# ruff: noqa: E501  (benchmark question texts are data)
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime.fromisoformat("2026-09-01T00:00:00+00:00")
LIVE = {"succeeded", "rolled_back"}


def load(directory: Path, name: str) -> list[dict[str, Any]]:
    rows = [
        json.loads(line) for line in (directory / f"{name}.jsonl").read_text("utf-8").splitlines()
    ]
    for row in rows:
        for key in ("started_at", "resolved_at", "deployed_at", "merged_at"):
            if row.get(key):
                row[key] = datetime.fromisoformat(row[key].replace("Z", "+00:00"))
    return rows


def changed_lines(patch: str) -> list[str]:
    lines = []
    for line in patch.splitlines():
        if line.startswith(("+++", "---")) or not line.startswith(("+", "-")):
            continue
        text = " ".join(line[1:].split())
        if len(text) >= 6:
            lines.append(text)
    return lines


def build(directory: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    incidents = load(directory, "incidents")
    deployments = load(directory, "deployments")
    prs = {p["id"]: p for p in load(directory, "pull_requests")}
    files = {f["id"]: f for f in load(directory, "code_files")}
    pr_files = defaultdict(list)
    for f in load(directory, "pull_request_files"):
        pr_files[f["pull_request_id"]].append(f)
    by_service: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for d in sorted(deployments, key=lambda d: (d["deployed_at"], d["id"])):
        by_service[d["service_id"]].append(d)
    incidents_by_service: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for i in sorted(incidents, key=lambda i: (i["started_at"], i["id"])):
        incidents_by_service[i["service_id"]].append(i)
    rng = random.Random(9)

    def before(service: str, moment: datetime) -> list[dict[str, Any]]:
        return [d for d in by_service[service] if d["deployed_at"] < moment]

    def after(service: str, moment: datetime) -> list[dict[str, Any]]:
        return [d for d in by_service[service] if d["deployed_at"] > moment]

    temporal: list[dict[str, Any]] = []

    def add_t(relation: str, question: str, target: str, expected: list[str], match: str) -> None:
        temporal.append(
            {
                "id": f"TQ-{len(temporal) + 1:02d}",
                "relation": relation,
                "question": question,
                "target": target,
                "expected": expected,
                "match": match,
            }
        )

    candidates = [i for i in incidents if before(i["service_id"], i["started_at"])]
    sample = rng.sample(candidates, 40)
    # immediately before (spec example first)
    spec = next(i for i in incidents if i["id"] == "INC-0421")
    phrasings = [
        "Which deployment happened immediately before {inc}?",
        "What was the last {svc} deployment before {inc} started?",
        "Which release of {svc} went out right before {inc}?",
        "What was deployed to {svc} just before {inc}?",
        "Which deployment happened immediately before {inc}?",
        "Name the {svc} deployment that immediately preceded {inc}.",
    ]
    for phrasing, inc in zip(phrasings, [spec, *sample[:5]], strict=True):
        gold = before(inc["service_id"], inc["started_at"])[-1]
        add_t(
            "immediately_before",
            phrasing.format(inc=inc["id"], svc=inc["service_id"]),
            "deployment",
            [gold["id"]],
            "first",
        )
    # immediately after
    after_candidates = [i for i in sample[5:] if after(i["service_id"], i["started_at"])]
    for phrasing, inc in zip(
        [
            "Which deployment came right after {inc} started?",
            "What was the first {svc} deployment after {inc}?",
            "Which deployment followed {inc} on {svc}?",
            "What was deployed to {svc} next after {inc} began?",
        ],
        after_candidates[:4],
        strict=True,
    ):
        gold = after(inc["service_id"], inc["started_at"])[0]
        add_t(
            "immediately_after",
            phrasing.format(inc=inc["id"], svc=inc["service_id"]),
            "deployment",
            [gold["id"]],
            "first",
        )
    # latest (relative to the clock)
    services = sorted(by_service)
    for phrasing, service in zip(
        ["What is the latest deployment of {svc}?", "Which {svc} deployment is the most recent?"],
        rng.sample(services, 2),
        strict=True,
    ):
        gold = [d for d in by_service[service] if d["deployed_at"] < NOW][-1]
        add_t("latest", phrasing.format(svc=service), "deployment", [gold["id"]], "first")
    for phrasing, service in zip(
        ["What was the most recent incident on {svc}?", "Which {svc} incident happened last?"],
        rng.sample(services, 2),
        strict=True,
    ):
        gold = [i for i in incidents_by_service[service] if i["started_at"] < NOW][-1]
        add_t("latest", phrasing.format(svc=service), "incident", [gold["id"]], "first")
    # previous
    deps_with_previous = [d for d in deployments if before(d["service_id"], d["deployed_at"])]
    for phrasing, dep in zip(
        [
            "Which deployment came before {dep}?",
            "What was the previous {svc} deployment before {dep}?",
        ],
        rng.sample(deps_with_previous, 2),
        strict=True,
    ):
        gold = before(dep["service_id"], dep["deployed_at"])[-1]
        add_t(
            "previous",
            phrasing.format(dep=dep["id"], svc=dep["service_id"]),
            "deployment",
            [gold["id"]],
            "first",
        )
    with_previous_incident = [
        i
        for i in sample[10:]
        if [p for p in incidents_by_service[i["service_id"]] if p["started_at"] < i["started_at"]]
    ]
    for phrasing, inc in zip(
        [
            "What was the previous incident on {svc} before {inc}?",
            "Which {svc} incident preceded {inc}?",
        ],
        with_previous_incident[:2],
        strict=True,
    ):
        gold = [
            p
            for p in incidents_by_service[inc["service_id"]]
            if p["started_at"] < inc["started_at"]
        ][-1]
        add_t(
            "previous",
            phrasing.format(inc=inc["id"], svc=inc["service_id"]),
            "incident",
            [gold["id"]],
            "first",
        )
    # at the time of
    for phrasing, inc in zip(
        [
            "Which version of {svc} was running at the time of {inc}?",
            "What {svc} version was live when {inc} started?",
            "Which deployment was live at the time of {inc}?",
            "What version of {svc} was in production at the time of the incident {inc}?",
            "Which {svc} release was serving traffic when {inc} began?",
        ],
        sample[14:19],
        strict=True,
    ):
        live = [
            d
            for d in by_service[inc["service_id"]]
            if d["deployed_at"] <= inc["started_at"] and d["status"] in LIVE
        ]
        if "deployment was live" in phrasing:
            add_t(
                "at_time_of",
                phrasing.format(inc=inc["id"], svc=inc["service_id"]),
                "deployment",
                [live[-1]["id"]],
                "first",
            )
        else:
            add_t(
                "at_time_of",
                phrasing.format(inc=inc["id"], svc=inc["service_id"]),
                "version",
                [live[-1]["version"]],
                "first",
            )

    # during
    def during(inc: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            d
            for d in by_service[inc["service_id"]]
            if inc["started_at"] <= d["deployed_at"] <= inc["resolved_at"]
        ]

    with_deploys_during = [i for i in incidents if not i["parent_incident_id"] and during(i)]
    for phrasing, inc in zip(
        [
            "Which deployments happened during {inc}?",
            "Which {svc} deployments were made while {inc} was ongoing?",
            "What was deployed to {svc} during {inc}?",
        ],
        rng.sample(with_deploys_during, 3),
        strict=True,
    ):
        add_t(
            "during",
            phrasing.format(inc=inc["id"], svc=inc["service_id"]),
            "deployment",
            [d["id"] for d in during(inc)],
            "set",
        )

    def overlapping(inc: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            o
            for o in incidents
            if o["id"] != inc["id"]
            and o["started_at"] <= inc["resolved_at"]
            and o["resolved_at"] >= inc["started_at"]
        ]

    few_overlaps = [i for i in incidents if 1 <= len(overlapping(i)) <= 3]
    for phrasing, inc in zip(
        [
            "Which other incidents were open during {inc}?",
            "What other incidents overlapped with {inc}?",
        ],
        rng.sample(few_overlaps, 2),
        strict=True,
    ):
        add_t(
            "during",
            phrasing.format(inc=inc["id"]),
            "incident",
            [o["id"] for o in overlapping(inc)],
            "set",
        )

    # after (window)
    def incidents_after(dep: dict[str, Any], hours: int) -> list[dict[str, Any]]:
        end = dep["deployed_at"] + timedelta(hours=hours)
        return [i for i in incidents if dep["deployed_at"] <= i["started_at"] < end]

    with_followers = [d for d in deployments if 1 <= len(incidents_after(d, 24)) <= 4]
    for phrasing, dep in zip(
        [
            "Which incidents started within 24 hours after {dep}?",
            "What incidents began in the 24 hours after deployment {dep}?",
            "Which incidents followed {dep} within a day?",
        ],
        rng.sample(with_followers, 3),
        strict=True,
    ):
        add_t(
            "after",
            phrasing.format(dep=dep["id"]),
            "incident",
            [i["id"] for i in incidents_after(dep, 24)],
            "set",
        )

    # before (window)
    def deploys_before(inc: dict[str, Any], days: int) -> list[dict[str, Any]]:
        start = inc["started_at"] - timedelta(days=days)
        return [
            d
            for d in by_service[inc["service_id"]]
            if start <= d["deployed_at"] < inc["started_at"]
        ]

    with_recent = [i for i in incidents if 1 <= len(deploys_before(i, 7)) <= 4]
    for phrasing, inc in zip(
        [
            "Which {svc} deployments happened in the 7 days before {inc}?",
            "What was deployed to {svc} during the week before {inc}?",
            "Which deployments of {svc} preceded {inc} within 7 days?",
        ],
        rng.sample(with_recent, 3),
        strict=True,
    ):
        add_t(
            "before",
            phrasing.format(inc=inc["id"], svc=inc["service_id"]),
            "deployment",
            [d["id"] for d in deploys_before(inc, 7)],
            "set",
        )

    # --- multi-hop --------------------------------------------------------------------
    multihop: list[dict[str, Any]] = []
    dep_by_id = {d["id"]: d for d in deployments}

    def chain(inc: dict[str, Any]) -> dict[str, Any]:
        dep = dep_by_id[inc["root_cause_deployment_id"]]
        pr = prs[inc["root_cause_pr_id"]]
        (f,) = pr_files[pr["id"]]
        return {
            "deployment": dep["id"],
            "pull_request": pr["id"],
            "commit": dep["commit_sha"][:7],
            "file": files[f["code_file_id"]]["path"],
            "change": changed_lines(f["patch"]),
        }

    def add_m(kind: str, question: str, expected: dict[str, Any], hops: list[str]) -> None:
        multihop.append(
            {
                "id": f"MH-{len(multihop) + 1:02d}",
                "kind": kind,
                "question": question,
                "expected": {k: v for k, v in expected.items() if k in hops},
                "hops": hops,
            }
        )

    primary = [i for i in incidents if i["root_cause_pr_id"] and not i["parent_incident_id"]]
    cascades = [i for i in incidents if i["root_cause_pr_id"] and i["parent_incident_id"]]
    picks = rng.sample(primary, 26)
    config_like = {
        "configuration_error",
        "api_timeout",
        "db_connection_exhaustion",
        "rate_limiting",
    }
    for inc in picks[:8]:
        what = "configuration change" if inc["category"] in config_like else "change"
        add_m(
            "cause_and_file",
            f"Which deployment caused {inc['id']} and which code file introduced the relevant {what}?",
            chain(inc),
            ["deployment", "file"],
        )
    for phrasing, inc in zip(
        [
            "Which commit introduced the change that caused {inc}?",
            "What is the commit behind {inc}?",
            "Which commit shipped the regression behind {inc}?",
            "Give the commit hash of the change that caused {inc}.",
            "Which commit and deployment are responsible for {inc}?",
        ],
        picks[8:13],
        strict=True,
    ):
        hops = ["commit", "deployment"] if "deployment" in phrasing else ["commit"]
        add_m("commit", phrasing.format(inc=inc["id"]), chain(inc), hops)
    for phrasing, inc in zip(
        [
            "Trace {inc} to the code: which deployment, commit, file and change were responsible?",
            "For {inc}, follow the chain from the incident to the deployment, the commit, the file and the change.",
            "What deployment, commit and file caused {inc}, and what exactly changed?",
            "Walk from {inc} to the code change that caused it: deployment, commit, file, diff.",
            "Which deployment caused {inc}, which commit did it contain, which file did that commit change, and how?",
            "Explain the change behind {inc}: the deployment, its commit, the file and the modified lines.",
        ],
        picks[13:19],
        strict=True,
    ):
        add_m(
            "full_chain",
            phrasing.format(inc=inc["id"]),
            chain(inc),
            ["deployment", "commit", "file", "change"],
        )
    for phrasing, inc in zip(
        [
            "Which pull request caused {inc}, and what file did it change?",
            "Which PR is behind {inc} and which file did it modify?",
            "Name the pull request that introduced {inc} and the file it touched.",
            "What PR caused {inc}, and in which file?",
        ],
        picks[19:23],
        strict=True,
    ):
        add_m("pr_and_file", phrasing.format(inc=inc["id"]), chain(inc), ["pull_request", "file"])
    for phrasing, inc in zip(
        [
            "What exactly changed in the code that caused {inc}?",
            "Show the code change behind {inc}.",
            "Which lines changed in the change that caused {inc}?",
        ],
        picks[23:26],
        strict=True,
    ):
        add_m("what_changed", phrasing.format(inc=inc["id"]), chain(inc), ["file", "change"])
    for phrasing, inc in zip(
        [
            "{inc} was caused by an upstream problem. Which incident was it, which deployment was behind it, and which file changed?",
            "What upstream incident led to {inc}, and which deployment and file caused that?",
            "{inc} was downstream impact. Trace it to the deployment and code file responsible.",
            "Which incident triggered {inc}, and which deployment and file were the root cause?",
        ],
        rng.sample(cascades, 4),
        strict=True,
    ):
        expected = {**chain(inc), "parent_incident": inc["parent_incident_id"]}
        add_m(
            "cascade",
            phrasing.format(inc=inc["id"]),
            expected,
            ["parent_incident", "deployment", "file"],
        )
    caused_by: dict[str, list[str]] = defaultdict(list)
    for i in incidents:
        if i["root_cause_deployment_id"]:
            caused_by[i["root_cause_deployment_id"]].append(i["id"])
    multi = [d for d, ids in caused_by.items() if len(ids) >= 2]
    for phrasing, dep in zip(
        ["Which incidents did deployment {dep} cause?", "Which incidents were caused by {dep}?"],
        rng.sample(sorted(multi), 2),
        strict=True,
    ):
        add_m(
            "reverse",
            phrasing.format(dep=dep),
            {"incidents": sorted(caused_by[dep])},
            ["incidents"],
        )
    return temporal, multihop


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "generated")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "evaluation")
    args = parser.parse_args()
    temporal, multihop = build(args.data_dir)
    for name, rows in (
        ("temporal_benchmark.jsonl", temporal),
        ("multihop_benchmark.jsonl", multihop),
    ):
        (args.out_dir / name).write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
        )
        print(f"{name}: {len(rows)} questions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
