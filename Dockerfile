# syntax=docker/dockerfile:1.7
# OpsRAG backend image: the API (default command), the bootstrap job
# (python scripts/bootstrap.py) and, as the `test` target, the test suite.
#
#   docker build -t opsrag-backend .                  # the runtime image
#   docker build --target test -t opsrag-tests .      # + test dependencies and tests
#
# Stages: deps (CPU-only torch + requirements) -> models (downloaded at build time, so
# containers need no network) -> backend (non-root runtime) -> test.

ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim AS python-base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH
RUN python -m venv /opt/venv

# --- dependencies --------------------------------------------------------------------
FROM python-base AS deps
COPY requirements.txt /tmp/requirements.txt
# The default torch wheel on Linux bundles CUDA (several GB) that this CPU image never
# uses: install the CPU build first, so requirements.txt finds torch already satisfied.
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.14.0" \
  && pip install -r /tmp/requirements.txt

# --- models --------------------------------------------------------------------------
FROM deps AS models
# Override to bake other models (the runtime reads the same variables).
ARG EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
ARG RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L6-v2
ARG VERIFY_NLI_MODEL=cross-encoder/nli-deberta-v3-xsmall
ARG SECURITY_INJECTION_MODEL=protectai/deberta-v3-base-prompt-injection-v2
ENV HF_HOME=/opt/hf \
    EMBEDDING_MODEL=${EMBEDDING_MODEL} \
    RERANKER_MODEL=${RERANKER_MODEL} \
    VERIFY_NLI_MODEL=${VERIFY_NLI_MODEL} \
    SECURITY_INJECTION_MODEL=${SECURITY_INJECTION_MODEL}
WORKDIR /build
COPY app ./app
COPY scripts/download_models.py ./scripts/
RUN python scripts/download_models.py

# --- runtime -------------------------------------------------------------------------
FROM python-base AS backend
ARG EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
ARG RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L6-v2
ARG VERIFY_NLI_MODEL=cross-encoder/nli-deberta-v3-xsmall
ARG SECURITY_INJECTION_MODEL=protectai/deberta-v3-base-prompt-injection-v2
# Models come from the image only: no download at runtime, ever.
ENV HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    EMBEDDING_MODEL=${EMBEDDING_MODEL} \
    RERANKER_MODEL=${RERANKER_MODEL} \
    VERIFY_NLI_MODEL=${VERIFY_NLI_MODEL} \
    SECURITY_INJECTION_MODEL=${SECURITY_INJECTION_MODEL}
WORKDIR /srv/opsrag
COPY --from=deps /opt/venv /opt/venv
COPY --from=models /opt/hf /opt/hf
COPY app ./app
COPY scripts ./scripts
# Evaluation results (served by GET /api/evaluation) and the benchmark question sets.
COPY data/evaluation ./data/evaluation
# The synthetic dataset is deterministic (seed 42): generated here, loaded by bootstrap.
RUN python scripts/generate_data.py \
  && groupadd --system --gid 10001 opsrag \
  && useradd --system --uid 10001 --gid 10001 --create-home --home-dir /home/opsrag opsrag
# Production checks on by default: strong database passwords and a read-only SQL role are
# required, and the unauthenticated demo header is refused (see app/config.py).
ENV OPSRAG_ENVIRONMENT=production \
    OPSRAG_LOG_FORMAT=json
USER 10001:10001
EXPOSE 8000
# Readiness, not just liveness: 200 only when the database answers and the agent's
# models are loaded (OPSRAG_PRELOAD_AGENT). python:slim has no curl; use the stdlib.
HEALTHCHECK --interval=15s --timeout=5s --start-period=180s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/ready', timeout=4)"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-server-header", "--timeout-keep-alive", "5"]

# --- bootstrap -----------------------------------------------------------------------
FROM backend AS bootstrap
CMD ["python", "scripts/bootstrap.py"]

# --- tests ---------------------------------------------------------------------------
FROM backend AS test
USER 0:0
# requirements-dev.txt includes the other two with -r (found by the Phase 14 Docker run).
COPY requirements.txt requirements-dev.txt requirements-frontend.txt ./
RUN pip install -r requirements-dev.txt
# pyproject.toml: pytest's configuration; .gitignore and .env.example: checked by the
# secret-management tests (neither holds a secret).
COPY pyproject.toml .gitignore .env.example ./
COPY tests ./tests
COPY frontend ./frontend
RUN chown -R 10001:10001 /srv/opsrag
USER 10001:10001
# Unit tests by default; integration: OPSRAG_RUN_INTEGRATION_TESTS=1 pytest -m integration
ENV OPSRAG_ENVIRONMENT=test
HEALTHCHECK NONE
CMD ["pytest", "-q", "-p", "no:cacheprovider"]
