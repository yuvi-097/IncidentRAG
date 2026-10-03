"""Build the Phase 9 *stress* sets: held-out temporal and multi-hop questions.

    python scripts/build_reasoning_stress.py

The main benchmarks (build_reasoning_benchmark.py) were written together with the
parser, so a high score on them shows that those phrasings are understood, not that the
system generalises. These sets were written afterwards and measured before any change
was made for them. They use other anchors (none of the main benchmarks' ids), other
phrasings, and constructs outside the main set: the anchor named before the target,
status filters, explicit dates, "any service", incidents with no recorded cause, the
fix instead of the cause, authors. Gold answers come from the raw records, with the same
definitions as the main benchmark (see its docstring).

Writes data/evaluation/temporal_stress.jsonl and multihop_stress.jsonl.
"""

# ruff: noqa: E501  (question texts are data)
from __future__ import annotations

import json
import random
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_reasoning_benchmark import LIVE, NOW, changed_lines, load  # noqa: E402

# A setting assigned on a changed line ("KEY: value", "KEY = value"), not a name used in code.
KEY = re.compile(r"^\s*([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)\s*[:=]")


def build(directory: Path, used: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    incidents = load(directory, "incidents")
    for i in incidents:
        if i.get("detected_at"):
            i["detected_at"] = datetime.fromisoformat(i["detected_at"].replace("Z", "+00:00"))
    deployments = load(directory, "deployments")
    prs = {p["id"]: p for p in load(directory, "pull_requests")}
    files = {f["id"]: f for f in load(directory, "code_files")}
    pr_files: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for f in load(directory, "pull_request_files"):
        pr_files[f["pull_request_id"]].append(f)
    dep_by_id = {d["id"]: d for d in deployments}
    ordered = sorted(deployments, key=lambda d: (d["deployed_at"], d["id"]))
    by_service: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for d in ordered:
        by_service[d["service_id"]].append(d)
    inc_by_service: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for i in sorted(incidents, key=lambda i: (i["started_at"], i["id"])):
        inc_by_service[i["service_id"]].append(i)
    fresh_inc = [i for i in incidents if i["id"] not in used]
    fresh_dep = [d for d in deployments if d["id"] not in used]
    rng = random.Random(91)

    temporal: list[dict[str, Any]] = []

    def add_t(relation: str, question: str, target: str, expected: list[str], match: str) -> None:
        temporal.append(
            {
                "id": f"TS-{len(temporal) + 1:02d}",
                "relation": relation,
                "question": question,
                "target": target,
                "expected": expected,
                "match": match,
            }
        )

    def before(service: str, t: datetime) -> list[dict[str, Any]]:
        return [d for d in by_service[service] if d["deployed_at"] < t]

    def after(service: str, t: datetime) -> list[dict[str, Any]]:
        return [d for d in by_service[service] if d["deployed_at"] > t]

    def live_at(service: str, t: datetime) -> list[dict[str, Any]]:
        return [d for d in by_service[service] if d["deployed_at"] <= t and d["status"] in LIVE]

    def pick(pool: list[Any], n: int) -> list[Any]:
        chosen = rng.sample([x for x in pool if x["id"] not in used], n)
        used.update(x["id"] for x in chosen)
        return chosen

    # --- paraphrases of supported relations ---------------------------------------------
    pool = [i for i in fresh_inc if before(i["service_id"], i["started_at"])]
    for inc in pick(pool, 2):
        add_t(
            "immediately_before",
            f"Which release was the last one to go out before {inc['id']}?",
            "deployment",
            [before(inc["service_id"], inc["started_at"])[-1]["id"]],
            "first",
        )
    pool = [d for d in fresh_dep if after(d["service_id"], d["deployed_at"])]
    for dep in pick(pool, 2):
        add_t(
            "next",
            f"Which {dep['service_id']} deployment came next after {dep['id']}?",
            "deployment",
            [after(dep["service_id"], dep["deployed_at"])[0]["id"]],
            "first",
        )

    def next_incident(inc: dict[str, Any]) -> list[dict[str, Any]]:
        return [o for o in inc_by_service[inc["service_id"]] if o["started_at"] > inc["started_at"]]

    for inc in pick([i for i in fresh_inc if next_incident(i)], 2):
        add_t(
            "next",
            f"Which incident on {inc['service_id']} came after {inc['id']}?",
            "incident",
            [next_incident(inc)[0]["id"]],
            "first",
        )
    for service in rng.sample(sorted(by_service), 2):
        gold = [d for d in by_service[service] if d["deployed_at"] < NOW][-1]
        add_t(
            "latest",
            f"What was the newest version of {service}?",
            "version",
            [gold["version"]],
            "first",
        )

    def incidents_before(inc: dict[str, Any], days: int) -> list[dict[str, Any]]:
        start = inc["started_at"] - timedelta(days=days)
        return [
            o
            for o in inc_by_service[inc["service_id"]]
            if start <= o["started_at"] < inc["started_at"]
        ]

    for inc in pick([i for i in fresh_inc if 1 <= len(incidents_before(i, 2)) <= 4], 2):
        add_t(
            "before",
            f"Which {inc['service_id']} incidents happened in the 2 days before {inc['id']}?",
            "incident",
            [o["id"] for o in incidents_before(inc, 2)],
            "set",
        )

    def during(inc: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            d
            for d in by_service[inc["service_id"]]
            if inc["started_at"] <= d["deployed_at"] <= inc["resolved_at"]
        ]

    for inc in pick([i for i in fresh_inc if during(i)], 2):
        add_t(
            "during",
            f"What deployments did {inc['service_id']} get while {inc['id']} was open?",
            "deployment",
            [d["id"] for d in during(inc)],
            "set",
        )

    def incidents_after(dep: dict[str, Any], hours: int) -> list[dict[str, Any]]:
        end = dep["deployed_at"] + timedelta(hours=hours)
        return [i for i in incidents if dep["deployed_at"] <= i["started_at"] < end]

    for dep in pick([d for d in fresh_dep if 1 <= len(incidents_after(d, 12)) <= 4], 2):
        add_t(
            "after",
            f"List the incidents that started within 12 hours after {dep['id']}.",
            "incident",
            [i["id"] for i in incidents_after(dep, 12)],
            "set",
        )
    pool = [d for d in fresh_dep if before(d["service_id"], d["deployed_at"])]
    for dep in pick(pool, 2):
        add_t(
            "previous",
            f"Which deployment of {dep['service_id']} preceded {dep['id']}?",
            "deployment",
            [before(dep["service_id"], dep["deployed_at"])[-1]["id"]],
            "first",
        )
    pool = [
        i for i in fresh_inc if i.get("detected_at") and live_at(i["service_id"], i["detected_at"])
    ]
    for inc in pick(pool, 2):
        add_t(
            "at_time_of",
            f"What was running on {inc['service_id']} at the time {inc['id']} was detected?",
            "deployment",
            [live_at(inc["service_id"], inc["detected_at"])[-1]["id"]],
            "first",
        )

    def deploys_after(inc: dict[str, Any], hours: int) -> list[dict[str, Any]]:
        end = inc["started_at"] + timedelta(hours=hours)
        return [
            d for d in by_service[inc["service_id"]] if inc["started_at"] <= d["deployed_at"] < end
        ]

    for inc in pick([i for i in fresh_inc if 1 <= len(deploys_after(i, 48)) <= 4], 2):
        add_t(
            "after",
            f"Which deployments went out to {inc['service_id']} in the 48 hours after {inc['id']} started?",
            "deployment",
            [d["id"] for d in deploys_after(inc, 48)],
            "set",
        )

    # --- other constructs -----------------------------------------------------------------
    pool = [i for i in fresh_inc if before(i["service_id"], i["started_at"])]
    for inc in pick(pool, 2):
        add_t(
            "immediately_before",
            f"Before {inc['id']} began, what was the most recent deployment to {inc['service_id']}?",
            "deployment",
            [before(inc["service_id"], inc["started_at"])[-1]["id"]],
            "first",
        )
    for inc in pick(pool, 2):
        add_t(
            "immediately_before",
            f"{inc['id']}: which deployment preceded it?",
            "deployment",
            [before(inc["service_id"], inc["started_at"])[-1]["id"]],
            "first",
        )
    pool = [i for i in fresh_inc if after(i["service_id"], i["started_at"])]
    for inc in pick(pool, 2):
        add_t(
            "immediately_after",
            f"After {inc['id']}, which deployment was the first to reach {inc['service_id']}?",
            "deployment",
            [after(inc["service_id"], inc["started_at"])[0]["id"]],
            "first",
        )
    pool = [i for i in fresh_inc if live_at(i["service_id"], i["started_at"])]
    for inc in pick(pool, 2):
        add_t(
            "at_time_of",
            f"Which version was deployed to {inc['service_id']} when {inc['id']} occurred?",
            "version",
            [live_at(inc["service_id"], inc["started_at"])[-1]["version"]],
            "first",
        )

    def last_succeeded(inc: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            d for d in before(inc["service_id"], inc["started_at"]) if d["status"] == "succeeded"
        ]

    pool = [
        i
        for i in fresh_inc
        if last_succeeded(i)
        and before(i["service_id"], i["started_at"])[-1]["status"] != "succeeded"
    ]
    for inc in pick(pool, 2):
        add_t(
            "immediately_before",
            f"Which deployment happened immediately before {inc['id']}, counting only successful deployments?",
            "deployment",
            [last_succeeded(inc)[-1]["id"]],
            "first",
        )

    def any_before(inc: dict[str, Any]) -> list[dict[str, Any]]:
        return [d for d in ordered if d["deployed_at"] < inc["started_at"]]

    pool = [
        i
        for i in fresh_inc
        if before(i["service_id"], i["started_at"])
        and any_before(i)[-1]["service_id"] != i["service_id"]
    ]
    for inc in pick(pool, 2):
        add_t(
            "immediately_before",
            f"Which deployment on any service happened immediately before {inc['id']}?",
            "deployment",
            [any_before(inc)[-1]["id"]],
            "first",
        )
    for service, day in zip(
        rng.sample(sorted(by_service), 2), ["2026-03-01", "2025-12-15"], strict=True
    ):
        moment = datetime.fromisoformat(f"{day}T00:00:00+00:00")
        gold = before(service, moment)[-1]
        add_t(
            "latest",
            f"What was the latest deployment of {service} before {day}?",
            "deployment",
            [gold["id"]],
            "first",
        )

    def open_at(dep: dict[str, Any]) -> list[dict[str, Any]]:
        t = dep["deployed_at"]
        return [i for i in incidents if i["started_at"] <= t <= i["resolved_at"]]

    for dep in pick([d for d in fresh_dep if 1 <= len(open_at(d)) <= 3], 2):
        add_t(
            "at_time_of",
            f"Which incidents were already open when {dep['id']} went out?",
            "incident",
            [i["id"] for i in open_at(dep)],
            "set",
        )

    # --- multi-hop ------------------------------------------------------------------------
    multihop: list[dict[str, Any]] = []
    rng = random.Random(92)

    def add_m(kind: str, question: str, expected: dict[str, Any], hops: list[str]) -> None:
        multihop.append(
            {
                "id": f"MS-{len(multihop) + 1:02d}",
                "kind": kind,
                "question": question,
                "expected": {k: v for k, v in expected.items() if k in hops},
                "hops": hops,
            }
        )

    def chain(inc: dict[str, Any]) -> dict[str, Any]:
        dep = dep_by_id[inc["root_cause_deployment_id"]]
        pr = prs[inc["root_cause_pr_id"]]
        (f,) = pr_files[pr["id"]]
        lines = changed_lines(f["patch"])
        return {
            "deployment": dep["id"],
            "pull_request": pr["id"],
            "commit": dep["commit_sha"][:7],
            "file": files[f["code_file_id"]]["path"],
            "change": lines,
            "key": sorted({m.group(1) for line in lines if (m := KEY.match(line))}),
            "author": pr["author_id"],
        }

    primary = [i for i in fresh_inc if i["root_cause_pr_id"] and not i["parent_incident_id"]]
    cascades = [i for i in fresh_inc if i["root_cause_pr_id"] and i["parent_incident_id"]]
    for phrasing, hops in [
        ("{inc}: which file was changed by the deployment that caused it?", ["file"]),
        ("Name the source file whose change led to {inc}.", ["file"]),
        ("Which merge commit is linked to the root cause of {inc}?", ["commit"]),
        ("What did the diff behind {inc} change?", ["change"]),
        (
            "Which pull request introduced the bug in {inc}, and what did it touch?",
            ["pull_request", "file"],
        ),
        ("Which deployment broke things in {inc}?", ["deployment"]),
        ("Who wrote the change that caused {inc}?", ["author"]),
    ]:
        for inc in pick(primary, 2):
            add_m(
                "paraphrase" if "Who" not in phrasing else "author",
                phrasing.format(inc=inc["id"]),
                chain(inc),
                hops,
            )
    for inc in pick(cascades, 2):
        add_m(
            "cascade",
            f"What upstream failure was {inc['id']} a symptom of?",
            {"parent_incident": inc["parent_incident_id"]},
            ["parent_incident"],
        )
    with_keys = [i for i in primary if chain(i)["key"]]
    for inc in pick(with_keys, 2):
        add_m(
            "config_key",
            f"Which configuration key did the change behind {inc['id']} modify?",
            chain(inc),
            ["key"],
        )
    caused_by: dict[str, list[str]] = defaultdict(list)
    for i in incidents:
        if i["root_cause_deployment_id"]:
            caused_by[i["root_cause_deployment_id"]].append(i["id"])
    causing = sorted(d for d in caused_by if d not in used)
    for dep in rng.sample(causing, 2):
        used.add(dep)
        add_m(
            "reverse",
            f"Which incidents followed from {dep}?",
            {"incidents": sorted(caused_by[dep])},
            ["incidents"],
        )
    for dep in rng.sample([d for d in causing if d not in used], 2):
        used.add(dep)
        paths = sorted(
            files[f["code_file_id"]]["path"]
            for pr in prs.values()
            if pr.get("deployment_id") == dep
            for f in pr_files[pr["id"]]
        )
        add_m(
            "reverse",
            f"Which files did {dep} change, and which incidents did it cause?",
            {"incidents": sorted(caused_by[dep]), "files": paths},
            ["incidents", "files"],
        )
    operational = [
        i for i in fresh_inc if not i["root_cause_deployment_id"] and not i["parent_incident_id"]
    ]
    for phrasing in [
        "Which deployment and commit caused {inc}?",
        "Trace {inc} to the code change responsible.",
    ]:
        for inc in pick(operational, 1):
            add_m("no_cause", phrasing.format(inc=inc["id"]), {"no_cause": True}, ["no_cause"])
    fixed = [
        i
        for i in primary
        if i.get("remediation_deployment_id")
        and dep_by_id[i["remediation_deployment_id"]]["commit_sha"]
        != dep_by_id[i["root_cause_deployment_id"]]["commit_sha"]
    ]
    for inc in pick(fixed, 2):
        fix = dep_by_id[inc["remediation_deployment_id"]]
        add_m(
            "remediation",
            f"In which commit was the regression behind {inc['id']} fixed?",
            {"commit": fix["commit_sha"][:7]},
            ["commit"],
        )
    return temporal, multihop


def main() -> int:
    out = ROOT / "data" / "evaluation"
    used: set[str] = set()
    for name in ("temporal_benchmark.jsonl", "multihop_benchmark.jsonl"):
        for line in (out / name).read_text(encoding="utf-8").splitlines():
            used.update(re.findall(r"\b(?:INC|DEP)-\d{4}\b", json.loads(line)["question"]))
    temporal, multihop = build(ROOT / "data" / "generated", used)
    for name, rows in (("temporal_stress.jsonl", temporal), ("multihop_stress.jsonl", multihop)):
        (out / name).write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
        )
        print(f"{name}: {len(rows)} questions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
