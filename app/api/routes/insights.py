"""Read-only views for the dashboard: the caller, evaluation results and process metrics.

- ``GET /api/me``: the authenticated user, their role, grants and permitted tools.

- ``GET /api/evaluation``: the latest Phase 10 run (``<evaluation_dir>/LATEST``): its
  summary, provenance (run id, dates, models, dataset and evaluation-set versions, judge
  status) and retrieval metrics split by question origin, computed from the saved
  per-question records. It serves measured numbers only; with no run it returns 404.
- ``GET /api/metrics``: the process metrics (``app/observability/metrics.py``) and the
  configuration they were measured under (retrieval mode and models, LLM or none).

Both need an authenticated caller; neither exposes answers, evidence or configuration
secrets (the evaluation summary holds aggregate numbers, not per-question answers).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

from fastapi import APIRouter, HTTPException, status

from app.api.dependencies import SettingsDep
from app.api.security import PrincipalDep
from app.evaluation.pipelines import METHODS
from app.observability.metrics import METRICS
from app.tools import build_registry

router = APIRouter(tags=["dashboard"])


@router.get("/me", summary="Who the caller is and what their role may read")
def me(principal: PrincipalDep) -> dict[str, Any]:
    """The UI shows this next to every answer: the same question can get different
    answers for different roles."""
    tools = build_registry()
    permitted = tools.permitted(principal)
    return {
        "user_id": principal.user_id,
        "role": principal.role,
        "grants": {
            resource.value: sorted(level.value for level in levels)
            for resource, levels in principal.grants.items()
        },
        "tools": [name for name in tools.names if permitted(name)],
    }


def latest_run(directory: Path) -> Path:
    pointer = directory / "LATEST"
    if not pointer.exists():
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, "no evaluation run yet: run python scripts/evaluate.py"
        )
    run = directory / pointer.read_text(encoding="utf-8").strip()
    if not (run / "summary.json").exists() or not (run / "manifest.json").exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"evaluation run {run.name} is incomplete")
    return run


def retrieval_by_origin(per_question: Path) -> dict[str, dict[str, dict[str, float]]]:
    """Mean Recall@5, Hit@5, MRR and NDCG@10 per question origin and method."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for line in per_question.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("retrieval"):
            groups[(record["origin"].split(":")[0], record["method"])].append(record["retrieval"])
    out: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for (origin, method), rows in sorted(groups.items()):
        out[origin][method] = {
            "questions": len(rows),
            "recall_5": round(fmean(r["recall"]["5"] for r in rows), 4),
            "hit_5": round(fmean(r["hit"]["5"] for r in rows), 4),
            "mrr": round(fmean(r["reciprocal_rank"] for r in rows), 4),
            "ndcg_10": round(fmean(r["ndcg_10"] for r in rows), 4),
        }
    return dict(out)


@router.get("/evaluation", summary="The latest evaluation run (measured numbers only)")
def evaluation(principal: PrincipalDep, settings: SettingsDep) -> dict[str, Any]:
    run = latest_run(settings.app.evaluation_dir)
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    per_question = run / "per_question.jsonl"
    return {
        "run": {
            "id": manifest["run_id"],
            "started_at": manifest["started_at"],
            "finished_at": manifest["finished_at"],
            "resumed": manifest.get("resumed", False),
            "questions": manifest["questions"],
            "eval_set_sha256": manifest["eval_set"]["sha256"],
            "dataset": {k: manifest["dataset"][k] for k in ("generator_version", "seed", "window")},
            "models": manifest["models"],
            "judge": manifest["judge"],
            "code_sha256": manifest["code_sha256"],
        },
        "methods": {m: METHODS.get(m, m) for m in summary["methods"]},
        "summary": summary,
        "retrieval_by_origin": retrieval_by_origin(per_question) if per_question.exists() else {},
    }


@router.get("/metrics", summary="Request, latency, tool and error metrics of this process")
def metrics(principal: PrincipalDep, settings: SettingsDep) -> dict[str, Any]:
    snapshot = METRICS.snapshot()
    snapshot["configuration"] = {
        "retrieval_mode": settings.retrieval.mode.value,
        "embedding_model": settings.embedding.model,
        "reranker_model": settings.reranker.model if settings.reranker.enabled else None,
        "llm": f"{settings.llm.provider}/{settings.llm.model}" if settings.llm.enabled else None,
        "nli_model": settings.verification.nli_model,
    }
    return snapshot
