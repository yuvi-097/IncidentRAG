# Phase 9: temporal and multi-hop evaluation, failure analysis

All numbers below were produced by `scripts/evaluate_reasoning.py`. The raw per-question results
(question, answer, tools, score detail) are in the `reasoning_*.json` files next to this one.

## Setup

- **Data:** the full generated NovaCart dataset in in-memory SQLite. Text search is BM25, so the
  run needs no model download and is deterministic. Synthesis is extractive (no LLM). The clock
  is fixed at 2026-09-01T00:00Z. Questions are asked as `admin`, so the scores measure reasoning,
  not access control (access control is covered by the tests).
- **Gold answers** come from the raw JSONL records via reference code that shares nothing with
  the agent (`scripts/build_reasoning_benchmark.py`, `scripts/build_reasoning_stress.py`). They
  are frozen files and are never recomputed by the system under test.
- **Scoring** (`app/evaluation/reasoning.py`) reads only the final answer text:
  - Identifiers named in the question are ignored.
  - "first" questions: the first identifier of the target kind in the answer must be the gold one.
  - "set" questions: the identifiers must equal the gold set exactly.
  - Multi-hop questions: every hop is checked separately, and a question counts as correct only
    when all its hops are.

| Suite | File | Questions | Written |
|---|---|---|---|
| Benchmark (temporal) | `temporal_benchmark.jsonl` | 34 | before the Phase 9 code; baseline measured on the Phase 8 system |
| Benchmark (multi-hop) | `multihop_benchmark.jsonl` | 32 | same |
| Stress (temporal) | `temporal_stress.jsonl` | 36 | after the Phase 9 code, measured before any change for it |
| Stress (multi-hop) | `multihop_stress.jsonl` | 26 | same |

The two benchmark files were built before any Phase 9 code existed, and are unchanged since
(sha256 `ea0df5f7…` and `339e2ccf…`). The same developer later wrote the parser, knowing the
benchmark's phrasings. A high benchmark score therefore shows those phrasings are handled; it
does not show that the system generalises. For that, the stress sets were written afterwards:

- other anchors (no id from the benchmarks);
- other phrasings;
- constructs the benchmarks lack:
  - the anchor named before the target;
  - status filters;
  - explicit dates;
  - "on any service";
  - the fix instead of the cause;
  - authors;
  - incidents with no recorded cause.

They were measured once, before anything was changed for them. That first run is the held-out
result.

## Results

| Run | Temporal | Multi-hop | File |
|---|---|---|---|
| Phase 8 system (baseline) | 9/34 | 7/32 | `reasoning_baseline.json` |
| Phase 9, benchmark, first run | 34/34 | 32/32 | `reasoning_phase9_first.json` |
| Phase 9, **held-out stress set, first run** | **23/36** | **20/26** | `reasoning_stress_first.json` |
| Phase 9, benchmark, final code | 34/34 | 32/32 | `reasoning_phase9.json` (SQLite), `reasoning_phase9_pg.json` (PostgreSQL) |
| Phase 9, stress set after fixes (not held-out) | 36/36 | 26/26 | `reasoning_stress_after_fixes.json`, `reasoning_stress_after_fixes_pg.json` |

The final-code runs on SQLite and PostgreSQL produced identical answer text for all 128
questions.

## Why the Phase 8 system failed (baseline)

**Temporal, 9/34.** The Phase 8 agent had no notion of order in time. A question naming an
incident was routed as multi-source. That plan fetches the incident, then the deployment its
record links to: the one live when it started, or its root-cause deployment. That explains every
hit:

- **"Immediately before" (5/6):** the linked live deployment usually *is* the last one before the
  incident, so these were right by coincidence of the data, not by comparing timestamps.
- **"After", "during", "next", "first after" (0/12):** nothing linked points forward in time, so
  the answer listed the live deployment again, or text-search matches.
- **"Latest" (1/4):** without an anchor, deployment search returned the newest by default only
  when the question named a service. Incident text search ranked by similarity, not time.
- **Windows (1/6):** no time window was derived from "in the 7 days before".

**Multi-hop, 7/32.** The multi-source plan reached the root-cause deployment (deployment hop
14/19). It then searched the code with the pull request id as a text query. That found the PR's
description, but not reliably the file it changed (file hop 5/25), and never its changed lines
(change 0/9). The pull request id was never stated as a hop of the chain (0/4). Cascades
reached the upstream incident through the incident record (4/4), but not beyond it.

## What Phase 9 changed

- **Temporal plan** (`app/agents/temporal.py`):
  - Fetch the anchor record, then query the target records by timestamp relative to it (nearest
    before or after, windows, overlap, live status).
  - State the order in a derived *timeline* evidence item, citing the records it lists.
- **Chain plan** (`app/agents/multihop.py`, tool `trace_change`):
  - Follow recorded links: incident → root-cause deployment → commit / PR → file → changed lines.
    Also the upstream incident, the remediation, and the reverse direction.
  - One evidence item per hop, each with its own access label.
  - Hops the caller may not read are withheld, never guessed.
- **Conflicts** (`app/agents/conflicts.py`):
  - Settings stated with different values by different sources are listed with their dates.
  - The newest statement still in effect is preferred, and nothing is dropped.

## Held-out stress run: every failure (first run, before any change for the stress set)

Temporal 23/36, multi-hop 20/26. All 19 failures, grouped by cause. **Unsafe** marks a wrong
answer given with HIGH confidence. **Safe** marks a failure where the system declined, or
answered a different but stated question.

| Cause | Questions | What happened | Safety |
|---|---|---|---|
| Anchor named before the target ("INC-0442: which deployment preceded it?", "After INC-0453, which deployment was the first…") | TS-23, TS-24, TS-25, TS-26 | The parser looked for the target noun only before the anchor, so the temporal plan was not used. The routed plan answered with the incident's own record (TS-23/24) or with the live deployment (TS-25/26, wrong: that deployment was *before* the incident). | TS-25/26 unsafe |
| "Deployed to X when INC occurred" | TS-27 | "deployed … when" was not an at-the-time-of pattern. Routed deployment search found nothing relevant, and the answer said evidence was insufficient. | safe |
| Status filter ignored ("counting only successful deployments") | TS-29, TS-30 | The temporal plan ran without the filter and returned the nearest deployment, which had failed or been rolled back. | **unsafe** |
| Scope override ignored ("on any service") | TS-31, TS-32 | The plan scoped to the incident's own service, as the definition says for unqualified questions, and ignored the override. | **unsafe** |
| Explicit date as the anchor ("before 2026-03-01") | TS-33, TS-34 | Dates were not anchors, so the routed search answered "insufficient evidence". | safe |
| "Incidents already open when DEP went out" | TS-35, TS-36 | "open when" was not an at-the-time-of pattern. Routed incident text search listed unrelated incidents. | **unsafe** |
| Author hop ("Who wrote the change…") | MS-13, MS-14 | No hop reached the PR author. The answer described the incident but named no author. | safe (no invented author) |
| "Incidents that followed from DEP" | MS-19, MS-20 | Read as temporal ("incidents in the day after the deployment") instead of causal ("caused by"). MS-19 listed an unrelated incident. | **unsafe** |
| Fix instead of cause ("In which commit was the regression fixed?") | MS-25, MS-26 | Treated as a cause chain. The answer stated the *cause* commit, correctly labelled as the cause, but did not answer the question. | safe-ish (stated, but not what was asked) |

**Fixes made after this analysis** (general rules, not per question):

- Parser:
  - the target noun may follow the anchor;
  - reversed orderings ("Before X, what was the most recent…", "After X, … first");
  - "deployed / open / ongoing … when" as at-the-time-of;
  - status filters;
  - "on any / all services";
  - ISO dates as anchors ("before / after / on <date>").
- Qualifiers the plan still cannot apply ("excluding", "except", "only" without a known status,
  …) are no longer ignored silently. They become a limitation, and confidence is capped at MEDIUM.
- Chain:
  - "followed from / resulted from / stemmed from / due to" are causal;
  - a *fix* direction follows the incident's remediation deployment;
  - authors come from the PR record.
- Precedence: an explicit relation in time wins over a "fix" read from a noun ("… before INC-1,
  excluding hotfixes").
- A structured plan starts only when every tool it will call is permitted for the role.
  Otherwise it falls back to the routed plan, with a limitation.
- A bug the stress run exposed: comparing a timezone-aware date anchor with SQLite's naive
  timestamps raised a `TypeError`. The stage failure was contained (the answer said "insufficient
  evidence"), and the comparison now normalises both to UTC.
- A bug the final benchmark run exposed: in MH-25 ("Show the code change behind INC-0338"),
  output validation removed half of a conflict note. The note named a code file by its internal
  id (CF-0068), which appears in no evidence text, so the guard rightly treated it as an
  unknown identifier. The score was unaffected, but the conflict was shown incompletely. Now:
  - code sources are named by path and pull request;
  - times are shown, not just dates (the cause and the fix fell on the same day);
  - two changes to one file count as two sources;
  - a packaged item's own id counts as evidence for the guard.

The stress set has now been used to guide fixes, so the second run is **no longer held-out**.
The first stress run is the best estimate of how the system handles questions phrased
differently from what its author anticipated.

## Remaining weaknesses and threats to validity

- **Same author.** The parser, the benchmarks and the stress sets were all written by the same
  developer. Real users' phrasings will differ more. Treat the held-out numbers as an upper
  bound on real-world accuracy for these question types.
- **Small samples.** With 26–36 questions per suite, one question moves the accuracy by about
  3 points. With the extractive synthesizer the runs are deterministic, so repeating a run
  gives the same number; that shows reproducibility, not precision.
- **Rule-based parsing.** Constructs not covered by a rule fall back to the routed plan. That
  plan does not reason about time, and can still return confident but unrelated records
  (the TS-35/36 pattern) for phrasings no rule recognises.
- **Definitions are choices.**
  - "Before / after an incident" is scoped to the incident's service, and plural "right before"
    means the 24 hours before.
  - "Running at the time" counts only deployments that went live (succeeded or rolled back).
  - "At the time the incident was *detected*" uses the start time: the incident tool does not
    return `detected_at`.
  - The answers state their scope and window, but a user with a different definition in mind
    may get a different answer than expected.
- **Conflict detection covers settings only.** Only `KEY: value` / `KEY = value` assignments,
  config-table rows and "changed from X to Y" sentences are compared. Disagreements in prose
  (for example two postmortems naming different root causes) are left to the claim verifier's
  contradiction check. Conflicts are also found only among the evidence that was retrieved.
- **Validity of a value** is judged by rules:
  - a `+` line of a change whose deployment did not succeed is not in effect;
  - the change a postmortem names as a root cause is not in effect.
  A value can also have been superseded by a change that was not retrieved.
- **The `no_cause` scorer is a heuristic.** It needs an explicit "no deployment … recorded"
  statement and no sentence naming a deployment as the cause. It would reject a correct answer
  worded differently.
- **Not measured here:** an LLM synthesizer (the extractive one was used), dense or hybrid
  retrieval, and roles other than admin. Role behaviour is covered by the tests, not by these
  scores.
