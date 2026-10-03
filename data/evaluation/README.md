# Evaluation data

## Retrieval benchmark (`retrieval_benchmark.jsonl`)

63 questions over the synthetic NovaCart dataset, one JSON object per line:

```json
{"id": "RQ-43", "category": "exact-match", "question": "What happened in INC-0406?",
 "relevant": [{"document_ids": ["INC-0406", "PM-0039"]}], "notes": "incident id. ..."}
```

- **Question sets:**
  - RQ-01 to RQ-37 are natural-language questions (categories runbook, documentation, code,
    incident, change), written in Phase 3.
  - RQ-38 to RQ-63 (`exact-match`) test error messages, ids, versions, config keys, class and
    function names, written in Phase 4.
- **Selectors:** `relevant` holds selectors over chunk metadata (`document_ids`, `source_type`,
  `title`, `title_contains`, `service_id`, `version`, `doc_type`, `file_path_endswith`,
  `metadata`). A source record is relevant if any selector matches it, so labels survive data
  regeneration and chunking changes.
- **When labels are written:** each set was labelled from the data before any retriever was run
  on it, and labels are never changed after seeing results. A test checks that every selector
  resolves.
- **What the retriever sees:** only the question text.

## Routing set (`routing_benchmark.jsonl`)

58 questions, each labelled with the query type the router should choose (all eight types) and,
for multi-source questions, the tools that must be selected. Five are the examples from the
project specification (`"source": "spec"`).

```json
{"id": "RT-05", "query": "What changed before the payment outage?", "expected_type": "MULTI_SOURCE",
 "expected_tools": ["search_incidents", "search_deployments"], "source": "spec"}
```

The router's rules were written with this set in view, so it is a development set. The held-out
estimate is a cross-check on the retrieval benchmark's questions, which were written before the
router existed; see `scripts/evaluate_routing.py`.

## Results (`results/`)

Written by `scripts/compare_retrieval.py` (all four systems), `scripts/benchmark_retrieval.py`
(one mode) and `scripts/evaluate_routing.py` (the router):

| File | Contents |
|---|---|
| `comparison.md` | Tables per subset and category, and the first relevant rank of every question per system |
| `comparison.json` | The same, machine-readable, with the configuration used |
| `dense.json`, `sparse.json`, `hybrid.json`, `hybrid_rerank.json` | Full per-system reports: retrieved documents and metrics per question |
| `routing.json` | Router accuracy per type, confusion matrix, every decision, and the held-out cross-check |

Results are only reported after actually running the evaluation. An answer-level evaluation set
(expected facts and evidence per question) is planned for a later phase.

## Phase 10 evaluation set (`eval_set.jsonl`)

227 questions in 11 categories (direct retrieval, semantic retrieval, incident investigation,
code search, SQL, temporal reasoning, multi-hop, conflicting evidence, no-answer, prompt
injection, permission restricted). Written by `scripts/build_eval_set.py`, deterministically;
frozen once built.

- **Each question has:**
  - `expected_answer`;
  - `expected_sources` (graded 2) and `supporting_sources` (graded 1);
  - `query_type` and `alt_query_types`;
  - `difficulty`;
  - `role`;
  - `checks` (facts, first identifier, identifier set, forbidden text, decline);
  - `origin`.
- **Origins:**
  - the 63 retrieval-benchmark questions, with their labels resolved to record ids and expected
    answers added from the records;
  - 124 generated from the records (temporal and multi-hop anchors are ones Phase 9 did not
    use);
  - 40 hand-written no-answer and injection questions.
- **Forbidden entries:**
  - `secret:NAME` is resolved from the configuration at run time and never written;
  - `exact:` and `regex:` are described in `app/evaluation/eval_set.py`.

Runs of `scripts/evaluate.py` are written to `results/phase10/<run id>/`; see the main README.
