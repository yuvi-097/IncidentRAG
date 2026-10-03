# Why OpsRAG is built this way

Eight design decisions, each with the problem it addresses, what was built, the measurement
that supports it, and what it costs. Every number was measured on this project; the
[technical reference](TECHNICAL_REFERENCE.md) has the full experiments and the
[QA report](QA_REPORT.md) the final audit.

The guiding rule behind all eight: **the LLM is never the source of truth.** Tools and
retrieval produce evidence with provenance. Access control filters that evidence before
any model sees it. Answers are checked against it afterwards.

Contents:

1. [Why hybrid retrieval](#1-why-hybrid-retrieval)
2. [Why reranking](#2-why-reranking)
3. [Why an agent](#3-why-an-agent)
4. [Why SQL and RAG coexist](#4-why-sql-and-rag-coexist)
5. [Why evidence verification](#5-why-evidence-verification)
6. [Why RBAC happens before retrieval](#6-why-rbac-happens-before-retrieval)
7. [Why temporal metadata matters](#7-why-temporal-metadata-matters)
8. [Why evaluation is necessary](#8-why-evaluation-is-necessary)

---

## 1. Why hybrid retrieval

**The problem.** Incident questions mix two kinds of language:

- **Descriptions:** "why do customers get signed out", "how do we handle Kafka consumer lag".
  These need meaning, not exact words.
- **Identifiers:** `INC-0406`, `v2.8.1`, `PAYMENT_SERVICE_DB_POOL_SIZE`,
  `OrderSaga._advance` and quoted error strings. These must match exactly.

Embeddings blur identifiers: `v2.8.1` and `v2.6.16` look alike to a sentence encoder.
Keyword search misses paraphrases.

**Measured** on 63 labelled questions (NDCG@10):

| Questions | Dense (bge-small) | BM25 |
|---|---:|---:|
| 37 natural-language | **0.593** | 0.555 |
| 26 exact-match | 0.360 | **0.791** |

Neither retriever wins both kinds of question, so OpsRAG runs both:

- **Dense search** in pgvector.
- **Identifier-aware BM25.** Identifiers are indexed whole *and* split into parts, so
  `PAYMENT_SERVICE_DB_POOL_SIZE` matches itself exactly (a rare term, so it gets a high IDF)
  and still matches "pool size".
- **Fusion** of the two lists by reciprocal rank (RRF, k = 60).

**The honest part.** Equal-weight RRF on its own was *worse* than BM25 alone: NDCG@10 0.614
against 0.652.

- **Why:** RRF uses ranks only. A document that only BM25 finds, at rank 1, scores 1/61, while
  documents that both lists rank moderately score more.
- **What fusion did improve was recall:** a relevant document appeared in the top 10 for 56 of
  63 questions, against 54 for BM25 and 47 for dense.

So hybrid search is not the final ranking. It is a **high-recall candidate generator** for
the reranker (decision 2), which is where the quality comes from.

**Cost.** Two first-stage searches take tens of milliseconds (hybrid p50 112 ms, including
the query embedding). That is small next to the reranker.

**What would change it.** In the Phase 10 evaluation, BM25 alone was best on identifier-heavy
questions (Hit@5 0.978 against 0.809 for reranked hybrid). The router already detects
identifier lookups, so sending those straight to BM25 is the next step.

## 2. Why reranking

**The problem.** A bi-encoder embeds the question and each passage separately, then
compares two vectors. It cannot check whether *this* passage answers *this* question. A
cross-encoder reads the pair together and scores relevance directly. It is too slow to run
over the whole corpus, but cheap enough over 30 candidates.

**What was built.** `cross-encoder/ms-marco-MiniLM-L6-v2` reranks the top 30 fused
candidates, each pair limited to 512 tokens. It only reorders: it never adds a chunk, so it
cannot bring back anything that access filtering removed.

**Measured** (63 questions):

| | Recall@10 | MRR@10 | NDCG@10 | Hit@10 |
|---|---:|---:|---:|---:|
| Hybrid | 0.768 | 0.621 | 0.614 | 56/63 |
| **Hybrid + reranker** | **0.829** | **0.751** | **0.699** | **62/63** |

- **Exact-match questions:** it recovers most of what fusion lost, to 0.782 against BM25's
  0.791.
- **Natural-language questions:** MRR rises from 0.596 to 0.714.
- **Reproduced:** the Phase 14 audit measured 0.695 again.

**It is not uniformly better.** Compared with hybrid, 22 questions improve and 13 get
worse. The model was trained on web search (MS MARCO) and knows nothing about document types
or version identity:

- For "how do I troubleshoot Kafka consumer lag", it ranks incidents that echo the wording
  above the runbook.
- It ranks the similar v2.6.16 incident above the v2.8.1 one.

**Cost.** The reranker is the dominant cost: 1.46 s at p50 for 30 candidates on CPU, and
about half of all answer time. Cheaper settings were measured and rejected:

| Setting | Reranker p50 | Quality | Decision |
|---|---|---|---|
| Shorter input (384 tokens) | −19% | Hit@5 0.87 vs 0.89 | Not the default: it changes answers |
| Fewer candidates (20) | −33% | NDCG@10 0.64 vs 0.69 | Rejected |
| Smaller batches (16 → 4) | −14.5% | identical | **Adopted:** less padding per batch on CPU, 0 answers changed |

## 3. Why an agent

**The problem.** Many incident questions are not "find similar text":

| Question | What it needs |
|---|---|
| "How many SEV1 incidents did payment-service have?" | A count over records |
| "Which deployment happened immediately before INC-0406?" | Ordering by timestamp |
| "Which commit and file caused INC-0033?" | Following links: incident → deployment → pull request → file → diff |
| "Why did payment-service fail after v2.8.1?" | Several sources, each query parameterised by the previous result |

A retrieve-then-read pipeline sends every question to the same retriever.

**Measured.** The Phase 10 ablation asked 227 questions of six systems:

| Category | Best retrieve-then-read (A–D) | Full agent (F) |
|---|---:|---:|
| SQL | 10% | **90%** |
| Temporal | 5% | **95%** |
| Multi-hop | 30% | **95%** |
| **All 227 questions** | 54.2% (BM25) | **77.1%** |

**What kind of agent.** It is a deterministic state machine with a rule-based router and
planner, not an LLM deciding what to do next. The reasons:

- **Explainable:** every routing decision lists the signals that fired.
- **Testable:** the same question gives the same plan.
- **Cheap:** no model calls to plan.
- **Safe:** it can only call 8 typed, read-only tools through a frozen registry.

The planner anchors on the most specific entity, parameterises later calls with earlier
results, and **stops when its evidence goals are met**. A simple question uses one tool.

**Costs, measured.**

- **Rules generalise imperfectly.** Routing is 58/58 on its development set but 82.5% on a
  held-out set, and routing errors caused 12 of the agent's 60 failures.
- **The agent is worse than plain BM25 on three categories:**
  - exact lookups: 36% against 88%;
  - code: 65% against 95%;
  - conflicting values: 13% against 80%.

  Its code answers name the right file but do not quote the value asked for.
- **Next step:** learned routing, validated against the same `RoutingDecision` schema.

## 4. Why SQL and RAG coexist

**The problem.** The knowledge comes in two shapes:

- **Facts in tables:** counts, averages, which pull requests shipped in a deployment, when
  each deployment happened. Text retrieval can only *approximate* these, by finding a
  passage that mentions them. A database *computes* them.
- **Knowledge in text:** runbooks, postmortems, design documents, code. Only retrieval can
  find these.

**Measured.** On the 20 SQL questions, every retrieve-then-read system scored 0–10%. The
agent scored 90%. On runbook and documentation questions, retrieval does the work: 72.7%
for the agent, 77.3% for hybrid + reranker.

**How SQL is kept safe.**

- **Questions become SQL through templates:** counts, averages, top-N, and per-service or
  per-month breakdowns, filtered by the services, time range and severities in the question.
  No model writes SQL. A question no template fits is reported as untranslatable, not guessed.
- **The guarded `query_database` tool**, for analysts, has five layers:
  1. static checks: one SELECT, no comments, function and table allowlists;
  2. a per-caller view of every table, filtered to the caller's grants;
  3. a read-only transaction, locked;
  4. a single prepared statement;
  5. a database role that can only SELECT nine tables.
- **SQL results are evidence like any other:** the query is shown, the rows are cited, and
  the claims are verified.

## 5. Why evidence verification

**The problem.** Retrieving a relevant passage does not make an answer correct:

- an extractive answer can stitch sentences together out of context;
- an LLM can paraphrase a value wrongly or fill a gap with something plausible;
- "I don't know" is the right answer to some questions, and a pipeline that always answers
  cannot give it.

During an outage, a confident wrong cause is worse than no answer.

**What was built.**

- **The evidence package is frozen before generation.** The synthesizer, extractive or LLM,
  sees exactly that and nothing else.
- **Every claim is checked against every item:**
  - each identifier, version and number must match exactly, and at least 75% of the claim's
    terms must be present;
  - an NLI model can confirm a paraphrase (entailment ≥ 0.6) or veto support (contradiction
    ≥ 0.7);
  - unsupported claims are removed, and wrong citations are replaced.
- **Confidence is computed, never chosen by the model:** five components (retrieval quality,
  source agreement, coverage, temporal consistency, verification) with caps. Conflicting
  sources or a failed tool cap it at MEDIUM.
- **The system can decline.** If the evidence covers less than half of the question's key
  terms (weighted by rarity), the answer is "insufficient evidence", with what would help.

**Measured** (Phase 10): adding verification to the same retrieval (D → E) gave these
results:

| | Without verification (D) | With verification (E) |
|---|---:|---:|
| Correct | 48.9% | 53.3% |
| Unanswerable questions declined | 0% | 85% |
| Permission questions handled | 55% | 85% |

The full agent declines 90% of the unanswerable questions.

**Costs, measured.**

- **False declines:** 21% of answerable questions for E and 8.4% for the agent.
- **A small NLI model misreads semi-structured records.** In Phase 10, 8 of the agent's 19
  chain answers carried a spurious "Sources disagree" note.
- **Time:** verification is about a third of answer time.
- **A bug in the verifier's own notes:** the Phase 14 audit found that these notes cited
  labels missing from the citation list, in 13 of 227 answers. It is fixed and tested.

**Not proven.** The measured hallucination rate is 0% because the default reader is
extractive and copies text. That says nothing about an LLM reader, which was never tested
with a real model.

## 6. Why RBAC happens before retrieval

**The problem.** If restricted text reaches the model's context, the model is the only
barrier left. Models follow instructions embedded in text, summarise what they see, and
leak through side channels:

- a citation or a title;
- a ranking;
- a count;
- a reference such as "see DOC-0060".

Asking the LLM to ignore what it was given is not access control.

**What was built.**

- **Roles come from the database,** never from the request.
- **A policy file grants labels per kind of data** (documents, runbooks, incidents, code,
  logs, …). Four roles, six labels; labels are compartments, not a ladder.
- **Every query that loads data filters to those grants:**
  - dense search, BM25 and chunk hydration;
  - every tool;
  - SQL, through per-caller views.
- **Confidential documents are never indexed at all.**
- **Defence in depth:**
  - a screening stage re-checks every evidence item before reranking, and a test plants a
    deliberately leaky tool to prove it is caught;
  - IDs and titles of hidden records are redacted from records the caller *may* read;
  - "not found" and "not permitted" look identical (404), so existence is not revealed.

**Measured.**

- **Unit tests:** 63 benchmark questions × 4 roles × 2 retrievers return only granted
  chunks.
- **In the live audit through the API:**
  - 20/20 permission-restricted questions leaked nothing;
  - the access matrix found 0 violations;
  - 10 incidents hidden from developers, requested directly by id, returned 404 on both
    detail and trace.
- **In SQL:** a developer counts 481 incidents where an SRE counts 540.

**Cost.** Answers differ by role. A developer asking about the v2.8.1 failure gets MEDIUM
confidence, a list of the sources they could not search, and `[restricted reference]` in
place of the fix's pull request. That is the intended behaviour.

## 7. Why temporal metadata matters

**The problem.** Incident investigation is about order:

- what changed *before* the failure;
- what was live *when* it started;
- what is the *latest* setting.

Similarity search has no notion of "before". "Which deployment happened immediately before
INC-0406?" shares no words with its answer, DEP-0296. Configuration values change over time,
so two documents can both be "relevant" and disagree.

**What was built.**

- **Every record and chunk keeps its timestamp.**
- **Temporal questions are parsed into a relation, a target and an anchor:** immediately
  before, after, during, at the time of, latest, or within a window. They are answered by
  time-ordered queries, and a *timeline* evidence item states the gap ("3 h 25 min before
  INC-0406").
- **Disagreeing sources are shown with their dates:** the newest statement still in effect
  is preferred, and confidence is capped.
- **Temporal consistency is a confidence component:** a root-cause deployment must precede
  its incident.

**Measured.**

| | Temporal questions |
|---|---|
| Phase 8 system, before this work | 9/34 |
| After | 34/34 |
| Held-out stress set, first run (the honest figure) | 23/36 |
| Phase 10, retrieve-then-read systems | 0–5% |
| Phase 10, the agent | 95% |

## 8. Why evaluation is necessary

Measurement contradicted the obvious choice several times in this project:

| Assumption | Measured |
|---|---|
| Hybrid search beats BM25 | Not without reranking: 0.614 against 0.652 |
| A reranker always helps | 13 of 63 questions got worse; it pushes exact matches down on identifier questions |
| The full agent beats simple retrieval everywhere | It loses to BM25 on lookups, code and conflicts |
| 34/34 on the temporal benchmark means it works | The held-out stress set's first run scored 23/36 |
| Batching NLI calls is faster | Total −8%, but p99 +27%; not adopted |
| Every cited label resolves to a source | 13 of 227 answers did not (found and fixed in the audit) |

How the evaluation is kept honest:

- **Frozen sets.** Labels and gold answers are fixed before results are seen.
- **Labels are metadata selectors** ("the runbook titled …"), resolved at evaluation time.
- **Separate gold code.** Gold answers are computed from the raw records by code that shares
  nothing with the agent.
- **Held-out sets** are reported separately from development sets.
- **An ablation of six systems** isolates what each layer adds.
- **Deterministic metrics, with their caveats stated.** No LLM-judge numbers are reported,
  because no judge model was configured. None were estimated.

Its limits: the same author wrote the system and the questions, and categories hold 15–25
questions, so one question moves a category by 4–7 points.
