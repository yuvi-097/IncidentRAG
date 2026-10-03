"""Download the configured models into the Hugging Face cache (``HF_HOME``).

    python scripts/download_models.py [--attempts 5]

The backend image runs this at build time, so its containers start without network
access (they run with ``HF_HUB_OFFLINE=1``). Each model is loaded exactly as the API
loads it, so every file the API needs ends up in the cache:

- the embedding model (``EMBEDDING_MODEL``),
- the cross-encoder reranker (``RERANKER_MODEL``),
- the NLI model for claim verification (``VERIFY_NLI_MODEL``),
- the prompt-injection classifier (``SECURITY_INJECTION_MODEL``).

Models configured as ``none`` are skipped. Unlike the API, which falls back to weaker
checks when the NLI or injection model cannot load, this fails loudly: an image must not
be built without a model it is configured to use. Downloads are retried, because the
Hugging Face hub sometimes resets connections.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.agents.verification import CrossEncoderNLI  # noqa: E402
from app.config import load_settings  # noqa: E402
from app.rag.embeddings import build_embedding_provider  # noqa: E402
from app.rag.reranking import build_reranker  # noqa: E402
from app.security.injection_model import TransformersInjectionClassifier  # noqa: E402


def with_retries(name: str, load: Callable[[], Any], attempts: int) -> None:
    for attempt in range(1, attempts + 1):
        started = time.monotonic()
        try:
            load()
            print(f"{name}: ready ({time.monotonic() - started:.1f}s)", flush=True)
            return
        except Exception as exc:
            print(f"{name}: attempt {attempt} failed: {type(exc).__name__}: {exc}", flush=True)
            if attempt == attempts:
                raise
            time.sleep(min(30, 5 * attempt))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempts", type=int, default=5)
    args = parser.parse_args(argv)
    settings = load_settings(env_file=None)  # build arguments arrive as environment variables
    jobs: list[tuple[str, Callable[[], Any]]] = [
        (
            f"embedding {settings.embedding.model}",
            lambda: build_embedding_provider(settings.embedding),
        ),
    ]
    if settings.reranker.enabled:
        jobs.append(
            (f"reranker {settings.reranker.model}", lambda: build_reranker(settings.reranker))
        )
    if settings.verification.nli_model:
        model = settings.verification.nli_model
        jobs.append((f"NLI {model}", lambda: CrossEncoderNLI(model, settings.verification.device)))
    if settings.security.injection_model:
        classifier = settings.security.injection_model
        jobs.append(
            (
                f"injection classifier {classifier}",
                lambda: TransformersInjectionClassifier(
                    classifier, settings.security.injection_device
                ),
            )
        )
    for name, load in jobs:
        with_retries(name, load, args.attempts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
