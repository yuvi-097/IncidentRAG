# NovaCart synthetic dataset

Everything here is **fictional**: the company, the people, the vendors (PayFlux, RiskShield, StockSync,
MailRelay, TextBridge), the code, and every incident. The data is built by simulating one year of
NovaCart operations, so the records are causally linked rather than randomly generated.

| Directory | Contents | In Git |
|---|---|---|
| `generated/` | Full dataset (~18 MB) plus the materialised code repository | no, regenerate it |
| `sample/` | One complete incident chain (~160 KB) for browsing and tests | yes |
| `evaluation/` | Retrieval benchmark (63 questions), routing set (58 questions) and measured results | yes |

```bash
python scripts/generate_data.py      # deterministic: seed 42 -> byte-identical files
python scripts/seed_db.py            # load into PostgreSQL (replaces table contents)
```

## Contents (seed 42)

| Table | Rows | Notes |
|---|---:|---|
| services | 11 | api-gateway, auth, user, product, inventory, cart, order, payment, notification, recommendation, search |
| service_dependencies | 23 | HTTP (hard/soft) and Kafka edges |
| users / roles | 40 / 4 | 11 teams; developer, sre, manager, admin (grants are in `app/security/policy.json`); one deactivated account |
| code_files | 265 | 161 source, 39 test, 24 config, 14 migrations, 14 build, 13 docs; all Python compiles |
| deployments | 376 | 282 succeeded, 42 rolled back (+42 rollback deployments), 10 failed canaries |
| pull_requests | 627 | 772 file changes, each a real unified diff against the repository |
| documents | 225 | 62 runbooks, 104 technical docs, 52 postmortems, 5 reliability reports, 2 admin policies |
| incidents | 540 | 15 categories; SEV1 43 / SEV2 183 / SEV3 190 / SEV4 124 |
| logs | 30,587 | 15.9k INFO, 5.9k WARNING, 8.8k ERROR |
| document_chunks | - | Derived: written by `scripts/ingest.py` (2,619 with defaults; the confidential document is not indexed), not part of the JSONL dataset |

The incidents fall into three groups:

- **74 deployment-caused.** A faulty PR shipped in a deployment and was fixed by a rollback or hotfix.
- **84 cascades.** An incident in a dependent service, linked through `parent_incident_id`.
- **The rest are operational.** Causes include traffic peaks (Black Friday, sales), infrastructure (node drains, NTP, DNS, conntrack), provider outages, and manual changes such as feature flags or ConfigMaps.

## How records link

```
incident ──deployment_id──────────────► deployment live in the affected service (affected_version)
    │  ──root_cause_deployment_id─────► deployment that introduced the fault (may be another service)
    │  ──root_cause_pr_id─────────────► pull request ──files──► code_files (unified diff in patch)
    │  ──remediation_deployment_id────► rollback (rollback_of_id) or hotfix deployment
    │  ──parent_incident_id───────────► upstream incident (cascades)
    │  ──runbook_id / postmortem_id───► documents
logs ──deployment_id/version─────────► deployment live at that moment; pod names embed the commit
pull_request.merge_commit_sha == deployment.commit_sha for the last PR in a release
```

Rules the generator enforces (`app/synthetic/validation.py` checks every one of them):

- `affected_version` is the version that was live in the incident's service when it started.
- A root-cause PR was shipped by the root-cause deployment, and it changed at least one file.
- A rollback redeploys exactly the previous version; the deployment it reverts has status `rolled_back`.
- Release versions only increase for each service. Rollbacks are the only deployments that go back.
- Fix and maintenance PRs end at the repository's HEAD. Faulty PRs start from HEAD, because their fix
  restored it.
- Every timestamp is inside the window (2025-09-01 to 2026-09-01), and `resolution_time_minutes`
  equals `resolved_at - started_at`.

Incident text deliberately varies in how much it reveals. Some root causes name the PR, commit and file.
Others give only the version ("Regression introduced by payment-service v2.8.1 (DEP-0296)..."). The
second kind can only be answered by following the links above, which is the multi-hop reasoning later
phases must perform.

## Anchor incidents

Five storylines are pinned, so their details are stable regardless of other generator changes:

| Incident | Story |
|---|---|
| INC-0406 | payment-service **v2.8.1** hard-codes `pool_size=5` in `db/database.py` → HTTP 500s at the evening peak → rollback to v2.8.0 → fix in v2.8.2; cascades into order-service and api-gateway |
| INC-0078 | inventory-service drops WMS delta de-duplication → stock drift (found by reconciliation, not alerts) |
| INC-0215 | cart-service stops refreshing cart TTLs → `redis-carts` (volatile-lru) fills up days later |
| INC-0266 | api-gateway caches JWKS for 24h and stops refetching unknown `kid`s → 401s after key rotation |
| INC-0336 | product-service renames an event field → search and recommendation consumers fail |

`data/sample/` contains INC-0406 with every record it links to.

## Access levels

Documents, code files and incidents carry an `access_level`, one of six labels: `public`,
`engineering`, `sre`, `manager`, `admin`, `confidential`. They are compartments, not a ladder (see the
main README, *Security*).
- **Examples:** auth-service and payment-service code, their configuration docs and authentication
  incidents are `sre`; the quarterly *Reliability Review* reports (computed from the incident and
  deployment records) are `manager`; the production access register and break-glass procedure are
  `admin`; the Secrets Management document is `confidential`.
- **Chunks:** ingestion copies each label onto the chunks. A PR chunk takes the most restrictive
  label among the files it changes. Confidential sources are never chunked.
- **Enforcement:** the access policy decides which roles read which labels, per kind of data,
  inside every query.

## Known simplifications

- Maintenance PRs are diffs that end at HEAD, so two PRs changing the same line may show the same `+` line.
- Logs are sampled (about 4 healthy lines per service per day, plus 10-40 lines per incident), not full
  request volume.
- Every PR is merged. There are no abandoned or open PRs.
