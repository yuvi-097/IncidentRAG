# OpsRAG quality audit (Phase 14)

An audit of the whole system: functional behaviour, security, retrieval, the agent, the
API, the containers, performance and the full regression suite. Nothing was added
except tests, a QA script and fixes for what the audit found. Every number below comes
from a run made during this audit (2026-10-02/03); the raw outputs are in `data/qa/`,
`data/benchmarks/` and the logs listed under [How to reproduce](#how-to-reproduce).

**Verdict.** The audit found **4 defects, and all are fixed**. One of them was in a test the
audit itself had added. For each defect, a test failed before the fix. After the fixes,
every test passes. The audit does **not** show the
system is correct: on the 227-question evaluation set it answers **77.1% correctly**
(175/227). The weak categories are known and listed under
[Remaining limitations](#remaining-limitations).

## Environment

| | |
|---|---|
| Host | Laptop, Intel Core i7-12650H (10 cores, 16 threads), 31.7 GB RAM, Windows 11, Python 3.11.9, CPU only |
| Containers | Docker Desktop 29.8.1 (WSL2, 16 CPUs, 15.5 GB for Docker), Compose v5.5.1; images built from a clean copy of the repository |
| Database | `pgvector/pgvector:pg17` in Docker; PostgreSQL 16.2 + pgvector 0.6.2 (pgserver) on the host |
| Models | `BAAI/bge-small-en-v1.5`, `cross-encoder/ms-marco-MiniLM-L6-v2`, `cross-encoder/nli-deberta-v3-xsmall`, `protectai/deberta-v3-base-prompt-injection-v2`; **no LLM** (extractive answers) |
| Data | Synthetic NovaCart dataset (seed 42): 540 incidents, 2,033 sources, 2,619 chunks |

## Summary

| Suite | Where | Total | Passed | Failed | Skipped |
|---|---|---|---|---|---|
| **Unit tests, final regression (after all fixes)** | Host (Windows), with coverage | **1,199** | **1,091** | **0** | **108** |
| Unit tests, first audit run (before the fixes; the failure is defect 4) | Host (Windows), with coverage | 1,196 | 1,087 | 1 | 108 |
| Unit tests | Docker, test image (Linux) | 1,199 | 1,081 | 0 | 118 |
| Lint and format (`ruff check`, `ruff format --check`) | Host | 287 files | clean | 0 | — |
| Integration tests (PostgreSQL + pgvector) | Docker, `opsrag_test` database | 87 | 87 | 0 | — |
| Model tests (real embedding, reranker, NLI, injection models) | Host | 21 | 21 | 0 | — |
| Live functional audit (evaluation questions through the API) | Docker, production mode | 227 | 175 correct | 52 incorrect | — |
| Live security probes (access matrix, injection payloads, secrets) | Docker, production mode | 4 roles, 6 payloads, 7 secrets | all passed | 0 | — |
| Reasoning benchmarks (temporal + multi-hop) | Host | 66 | 66 | 0 | — |
| Reasoning stress sets (see the note below) | Host | 62 | 62 | 0 | — |
| Routing | Host | 58 | 58 | 0 | — |

Notes:

- **Skipped tests** are the opt-in suites, which ran separately: integration (87) and model
  tests (21); the final run's 108 skips are exactly these. In the Docker image, the 10 tests of the container definitions also skip,
  because those files are not in the image.
- **"Incorrect" in the functional audit** means an answer failed the evaluation's
  deterministic checks: a missing fact, a wrong id, forbidden text, or a decline where an
  answer was expected (or the reverse). It is not a crash: there were 0 HTTP errors.
- **The reasoning stress sets are no longer held out.** Phase 9 used them to fix bugs, so
  their 100% is not a generalisation result. Their first, honest run scored 23/36 and
  20/26.
- **Code coverage** of the final unit run: **95.0% of lines, 93.6% including branches**
  (12,562 statements in `app/` and `frontend/`; 94.7% and 93.2% before the audit's new
  tests). The new Incident Explorer test raised that page from 35% to 79%. The lowest files
  are mostly those that only the opt-in suites reach: the PostgreSQL schema code (26%), the
  real injection model (50%) and the vector store (68%). Also low are the UI's HTTP client
  (53%), whose page tests use a fake client, and the benchmark helpers (58%).

## Defects found and fixed

| # | Defect | How it was found | Fix | Guard |
|---|---|---|---|---|
| 1 | In 13 of 227 answers, the verifier's "Sources disagree about … [E#] state(s) otherwise" note cited an evidence label that was **missing from the response's `citations`**. A client (the UI's Citations tab) could not resolve it. The label always pointed at returned evidence, so nothing was invented | Live audit: citation integrity check on every answer | `app/agents/graph.py`: the labels a kept claim's note cites join the citations. After the fix: 0 of 227 | `test_every_label_in_the_answer_is_a_listed_citation` (fails without the fix) |
| 2 | The `bootstrap` job ran the backend image and **inherited its health check** (`/api/ready`), so `docker compose ps` showed the job as "unhealthy" while it worked. It did not block the start order | First real `docker compose up --build` | `healthcheck: disable: true` for `bootstrap` | `test_long_running_services_have_health_checks` |
| 3 | **The test image did not build**: the `test` stage copied `requirements-dev.txt`, which includes `requirements.txt` (`-r`), without copying it | `docker compose --profile test run tests` | `requirements.txt` copied into the stage | `test_the_test_stage_copies_every_requirements_file_it_includes` |
| 4 | A new test file contained a token-shaped literal, which the repository's secret scanner rejected (the scanner worked as intended) | Regression run | The test builds the string at run time | The existing secret-scan test |

Defects 2 and 3 could not have been found in Phase 13, when Docker was not available.

Gaps closed with new tests:

- **Every API endpoint** (`tests/api/test_all_endpoints.py`, 21 tests): the endpoint
  table must equal the OpenAPI schema, every protected endpoint refuses anonymous, forged
  and header-only callers, and every endpoint answers an allowed caller.
- **The Incident Explorer's detail view and trace flow** (`tests/frontend`).

## Functional results

From the live audit: all 227 evaluation questions, asked through the containerized API in
production mode, each as a user of the question's role (tokens for 4 roles), scored with
the evaluation's deterministic checks.

| Spec item | Category | Questions | Correct | Rate |
|---|---|---|---|---|
| Document search | direct retrieval | 25 | 9 | 36.0% |
| Document search (paraphrased) | semantic retrieval | 22 | 16 | 72.7% |
| Incident search | incident investigation | 25 | 21 | 84.0% |
| Code search | code search | 20 | 13 | 65.0% |
| SQL | sql | 20 | 18 | 90.0% |
| Multi-source investigation | multi-hop | 20 | 19 | 95.0% |
| Temporal reasoning | temporal reasoning | 20 | 19 | 95.0% |
| Conflicting sources | conflicting evidence | 15 | 2 | 13.3% |
| No-answer behaviour | no answer | 20 | 18 | 90.0% |
| Prompt injection | prompt injection | 20 | 20 | 100% |
| Denied permission | permission restricted | 20 | 20 | 100% |
| **All** | | **227** | **175** | **77.1%** |

- **Unchanged since Phase 10.** The overall figure and every per-category rate are the
  same as in Phase 10, before the changes of Phases 11–13.
- **Repeatable.** The audit ran three times: before fix 1, after it, and after a restart.
  The runs before and after the fix have the same result for every question: verdict,
  decline, confidence level, tools and refusal. Only the citation lists changed, as
  intended.
- **Citations.** Every `[E#]` in an answer is a citation, and every citation points at a
  returned evidence item with the same source. After fix 1, 0 problems in 227 answers.
- **Confidence.** Always one of the four levels: 145 HIGH, 18 MEDIUM, 64
  INSUFFICIENT_EVIDENCE. No INSUFFICIENT_EVIDENCE answer cites anything.
- **No-answer behaviour.**
  - 18 of 20 unanswerable questions were declined.
  - 14 of 167 answerable questions were also declined (false declines, 8.4%).
- **Reasoning benchmarks** (`scripts/evaluate_reasoning.py`): temporal 34/34, multi-hop
  32/32; stress sets 36/36 and 26/26 (see the note above).
- **Latency in the containers:** answers took p50 0.72 s, p95 3.07 s, p99 4.49 s.

## Security findings

| Area | Test | Result |
|---|---|---|
| RBAC | Access matrix for 4 roles through the API. Each role's incident list compared with its grants (`/api/me`); hidden incidents requested directly | **Pass.** No role listed an incident it may not read. 10 incidents hidden from the developer (a sample of the 59 `sre`-labelled ones), requested directly: 404 on detail and on trace |
| Unauthorized retrieval | 20 "permission restricted" questions (a role asking about records it may not read) | **Pass.** 20/20 leaked none of the restricted text |
| Authentication | Every protected endpoint, called anonymously, with a forged token, and with the demo header (off in production) | **Pass.** 401 everywhere (`test_all_endpoints.py`) |
| Prompt injection | 20 injection questions (instruction overrides, system-prompt and canary extraction) | **Pass.** None complied: no answer contained a question's forbidden compliance text. 6 were refused outright, and 14 were answered or declined without complying |
| SQL injection | 6 payloads in questions (`DROP`, `UNION SELECT password_hash`, `DELETE`, `UPDATE`, `TRUNCATE`, `INSERT`) | **Pass.** No server errors. The agent builds SQL from fixed templates and ignored the payloads. Afterwards the database was checked directly: 540 incidents, 40 users, no inserted row, 4 API tokens, 2,619 chunks |
| Destructive SQL blocking | The SQL guard and read-only role suites, unit and on PostgreSQL | **Pass** (part of the regression and integration runs) |
| Secret protection | 7 secret values (2 passwords, the UI token, 4 API tokens) searched for in every answer, error body and all container logs (963 log lines, 329 requests) | **Pass.** 0 occurrences. No question text appears in the logs |
| Malicious documents | Planted injection documents (quarantine), unit and on PostgreSQL | **Pass** in the test suites. The live dataset has no planted documents on the evaluation questions' paths (0 quarantined in the live run), so this is verified by tests only |
| Dependencies | `pip-audit` over the 96 installed packages | **0 known vulnerabilities** |
| Static analysis | `bandit` over `app/`, `scripts/`, `frontend/` (32k lines) | **0 high, 18 medium, 48 low; all triaged** (below) |

Static-analysis triage:

- **B608, possible SQL injection (14, medium): false positives.**
  - The SQL templates interpolate only fixed table and column names, service ids matched
    against the catalog, enum values, formatted timestamps and clamped integers, with
    literals quote-escaped.
  - The shadow views interpolate schema table names and policy labels.
  - Everything also passes the SQL guard, a read-only transaction and the read-only role.
  - The live payloads confirm it.
- **B615, Hugging Face downloads without a pinned revision (4, medium): open finding.**
  The image fetches each model's current revision at build time, so builds are not
  reproducible and trust upstream changes.
  - Mitigations: models load without remote code, and at runtime they come only from the
    image (`HF_HUB_OFFLINE=1`).
  - Recommendation: pin revisions in the build arguments.
- **Low (48): informational.**
  - `assert` statements are invariants that fail closed if stripped.
  - `random` is used only by the synthetic data generator.
  - "Hardcoded passwords" are constant names.
  - `subprocess` calls run fixed argument lists without a shell.

Deployment findings (open, by design of the demo stack, documented here):

- **No TLS.** The compose stack serves the API and UI over plain HTTP, so tokens travel in
  clear text. A production deployment needs a TLS-terminating reverse proxy in front.
- **No rate limiting.** One worker answers about 1–1.3 questions per second (Phase 12), so
  a burst of expensive questions saturates the CPU. A proxy or gateway should limit
  requests per client.
- **API docs are public.** `/api/docs` and `/api/openapi.json` are served without
  authentication in production. They contain no data, but some deployments hide them.

## RAG results

| Check | Result |
|---|---|
| Dense, BM25, hybrid, reranking (`scripts/compare_retrieval.py`, 63 labelled questions, PostgreSQL) | NDCG@10: dense 0.497, BM25 0.652, hybrid 0.617, hybrid + reranker 0.695; Recall@10 0.643 / 0.766 / 0.773 / 0.829. This reproduces Phase 4 (0.497 / 0.652 / 0.614 / 0.699); the small differences come from near-ties after re-embedding the corpus. By subset: on the 37 natural-language questions the reranker is best (NDCG@10 0.641); on the 26 exact-match questions BM25 alone is best (0.791, against 0.771 with the reranker) |
| Real-model tests (dense search, reranker, NLI, injection classifier) | 21/21 passed |
| Evidence validation | The verification and evidence suites pass (in the regression runs); citation integrity in 227 live answers: 0 problems after fix 1 |

## Agent results

| Check | Result |
|---|---|
| Routing | 58/58 on the development set; held-out cross-check 52/63 = 82.5% (unchanged since Phase 5) |
| Tool selection | Agent suites pass. Live: 182 answers used one tool, 23 used two and 1 used three. 21 used none, and all 21 were refusals (6 injection questions) or declines (15) |
| Tool failure handling | `test_a_failing_tool_is_reported_not_papered_over` and the crash suites pass; a crashing tool becomes an error code, never a 500 (0 HTTP errors live) |
| Unnecessary tool avoidance | `test_simple_questions_use_one_tool`; the planner stops when its evidence goals are met |
| Multi-tool workflows | Multi-source and chain questions: 19/20 correct live; reasoning benchmarks 100% |

## API results

All 10 operations: health, ready, ask, services, incidents, incident, trace, me,
evaluation, metrics. They are covered by `tests/api/test_all_endpoints.py`:

- The table equals the OpenAPI schema.
- Anonymous, forged and header-only callers get 401.
- An allowed caller gets 200, with a JSON body and a request id.
- Wrong methods get 405, and malformed bodies (empty, 100,000 characters, wrong types)
  get 422.

Live, through Docker:

- `/api/health` returned 200 in production mode.
- `/api/ready` returned 200 once the models had loaded.
- The 227 questions produced 0 HTTP errors.

## Docker results

Started from a clean environment: a fresh copy of the repository without `.env`,
virtualenv or generated data, a new `.env` from `.env.example` with two generated
passwords, and an empty Docker engine (no images or volumes).

| Step | Result |
|---|---|
| `docker compose up --build` | All services healthy **11 min 17 s** after the command, including building the images (CPU torch; the 4 models downloaded in 132 s) and the first start's embedding |
| Start order | postgres healthy, then bootstrap, then backend healthy, then frontend healthy |
| Bootstrap (first start) | Seeded the empty database in production mode, created the read-only role, chunked and embedded 2,619 chunks in 290 s (9.0 chunks/s); exit 0, after 304 s |
| Database connection | By service name (`postgres:5432`) |
| Health / readiness | 200 / 200 (the agent's models loaded) |
| Chat endpoint | 200 with tokens issued in the container (`create_token.py`); the demo header and bad tokens got 401 |
| Frontend | Healthy. Signed in through `OPSRAG_API_TOKEN` (set in `.env`, then `docker compose up -d frontend`): a question was answered with citations in 282 ms |
| Tests in the containers | Unit 1,081 passed / 0 failed; integration 87 passed on `opsrag_test` (the application database was untouched: 540 incidents) |
| Restart (`docker compose down`, then `up --build -d`) | 3.5 min (only the changed layer rebuilt); the bootstrap found the data in the volume (13.7 s, 0 chunks re-embedded); the tokens issued before the restart still worked |
| Images | backend 4.54 GB, frontend 798 MB, `pgvector/pgvector:pg17` 637 MB |

Notes:

- **Changing `.env` recreates the backend and the bootstrap job.** Both read the whole file
  (`env_file`), so `docker compose up -d frontend` after setting `OPSRAG_API_TOKEN`
  recreated them. The bootstrap found nothing to do, but the backend's earlier logs were
  gone, so the live audit was run again before the log scan. The technical reference's troubleshooting
  section now describes this.
- **The real-model tests did not run inside the container.** That run was interrupted, and
  Docker Desktop was later found stopped and was not restarted, so the model tests ran on
  the host instead (21/21).

## Performance findings

The full benchmark (`scripts/benchmark_performance.py --run-id qa-phase14 --before
reranker.batch_size=16,embedding.batch_size=32`) ran on the host against PostgreSQL +
pgvector, on mains power, with the machine kept awake. It used the same script and
settings as Phase 12's run `final`. Nothing else ran during it; the regression run started
afterwards. Report: `data/benchmarks/performance/qa-phase14/report.md`.

| Stage | n | p50 | p95 | p99 | Phase 12 `final` (p50 / p95 / p99) |
|---|---|---|---|---|---|
| Ingestion: parse, clean, chunk (2,619 chunks) | 3 | 0.86 s (3,037 chunks/s) | 0.93 s | 0.93 s | 1.63 s / 1.70 s / 1.70 s |
| Ingestion: full, with writes (fresh SQLite) | 3 | 1.09 s (2,413 chunks/s) | 1.19 s | 1.19 s | 2.07 s / 2.10 s / 2.10 s |
| Embedding documents (batch 8) | 256 chunks | 12.4 chunks/s | | | 4.3 chunks/s |
| Embedding one query | 227 | 16 ms | 19 ms | 22 ms | 73 / 108 / 136 ms |
| Retrieval: dense / BM25 / hybrid | 227 | 25 / 12 / 44 ms | 29 / 15 / 50 ms | 31 / 16 / 53 ms | 86 / 24 / 112 ms (p50) |
| Reranking: 30 candidates, 512 tokens, batch 4 | 63 | 1,294 ms | 1,445 ms | 1,572 ms | 1,457 / 4,332 / 4,485 ms |
| Claim verification (agent stage) | 57 | 290 ms | 753 ms | 1,952 ms | 318 / 835 / 2,044 ms |
| LLM | — | not measured: no LLM is configured | | | |
| **Answer, end to end (agent)** | 57 | **621 ms** | **2,241 ms** | **2,608 ms** | 656 / 2,190 / 2,717 ms |

Findings:

1. **No large regression since Phase 12.** End-to-end latency is within about 5% of run
   `final` at every percentile, after Phase 13's start-up changes and the citation fix.
   The machine's speed varies between sessions (finding 2), so this rules out a large
   regression, not a small one.
2. **The laptop was in a fast phase.** Embedding was 2.9× faster than in `final`, query
   embedding 4.6× faster and ingestion 1.9× faster, with the same code and settings.
   Phase 12 documented this variance (up to 3×). Only comparisons within a run, with
   interleaved variants, are meaningful.
3. **The bottlenecks are unchanged.** Shares of the agent's total time:
   - tool execution, mostly the cross-encoder inside the search tools: 52.2%;
   - claim verification (NLI): 31.5%;
   - evidence reranking (the same cross-encoder): 12.2%;
   - security screening: 3.3%;
   - everything else: under 1%.

   The database is not a bottleneck: hybrid first-stage retrieval takes 44 ms at p50.
4. **Phase 12's two optimizations reproduce.** In 57 questions alternating between the old
   and new batch sizes, total time fell 10.6% and p95 10.3%, with **0 answers changed**.
   - **Reranker batch 4 vs 16:** p50 −17.6%, identical quality.
   - **Embedding batch 8 vs 32:** +19% throughput.
   - **Batch 4 vs 8 for embedding:** 3% ahead. Phase 12 measured +6% and level, which is too
     small and inconsistent to justify a change.
5. **The rejected options remain rejected.**
   - **Reranker input of 384 tokens:** p50 −14.5%, but Hit@5 falls from 0.89 to 0.87.
   - **Batched NLI:** total −7.5%, but p99 +12%.
   - **Torch threads:** the default 10 was again the fastest.
6. **In the containers** (the live audit, measured by the client through HTTP), answers took
   p50 0.72 s, p95 3.07 s and p99 4.49 s. This is not directly comparable with the agent's
   own time above: it covers all 227 questions rather than every fourth, includes HTTP and
   ran in a Linux VM. The containers' first start embedded the corpus at 9.0 chunks/s.
7. **Throughput under load was not re-measured.** Phase 12's load test found about 1–1.3
   answers per second with 1–8 requests in flight, CPU-bound, with p95 about 12 s at 8 in
   flight. The rate-limiting finding above relies on that measurement.

## Remaining limitations

- **Answer quality** (from Phase 10, unchanged):
  - **Exact lookups (36%).** Answers often list the right record without quoting the asked
    value.
  - **Code (65%).** Code answers name files rather than quoting the values that changed.
  - **Conflicting sources (13%).** Conflicts are rarely surfaced for configuration
    questions.
  - **Wrong citations.** The small NLI model sometimes treats a supporting record as
    contradicting a claim, which produces "Sources disagree" notes on correct answers.
- **False declines:** 8.4% of answerable questions are declined.
- **No real LLM was tested.** All answers are extractive. LLM synthesis, token usage and
  LLM latency are covered only with fake providers, and the LLM judge was not run.
- **The evaluation questions and the system have the same author.** Categories hold 15–25
  questions, so one question moves a category by 4–7 points.
- **Model revisions are unpinned** (see the security findings).
- **The deployment has no TLS and no rate limiting** (see the security findings).
- **Performance was measured on one laptop.** Its speed varied about 3× between sessions:
  embedding ran at 2.9 to 12.4 chunks/s on the host across Phases 12–14, and at 9.0 in the
  containers. Absolute numbers are indicative only.
- **The backend image is 4.5 GB.** CPU torch and four models are baked in.
- **The live dataset contains no malicious documents** on the evaluation questions'
  paths. Quarantine is verified by the test suites only.

## How to reproduce

```bash
python -m coverage run -m pytest -p no:cacheprovider && python -m coverage report   # unit + coverage
OPSRAG_RUN_INTEGRATION_TESTS=1 pytest -m integration       # needs PostgreSQL + pgvector
OPSRAG_RUN_MODEL_TESTS=1 pytest -m model                   # real models (about 11 min on CPU)
docker compose up --build                                  # the stack, from a fresh .env
docker compose --profile test run --rm tests               # unit tests in the test image
docker compose --profile test run --rm -e OPSRAG_RUN_INTEGRATION_TESTS=1 tests pytest -m integration -p no:cacheprovider
docker compose exec backend python scripts/create_token.py issue <user>   # one per role -> tokens.json
python scripts/audit_live.py --tokens tokens.json --env-file .env --out data/qa/live
python scripts/evaluate_routing.py --output data/qa/routing.json
python scripts/evaluate_reasoning.py --label qa --suite benchmark   # and --suite stress
python scripts/compare_retrieval.py --output-dir data/qa/retrieval
python scripts/benchmark_performance.py --run-id qa --before reranker.batch_size=16
```

Static analysis ran from a separate virtual environment, so that the tools' own
dependencies were not audited:

- `bandit -r app scripts frontend`;
- `pip-audit -r freeze.txt --no-deps`, where `freeze.txt` is the project environment's
  `pip freeze`.
