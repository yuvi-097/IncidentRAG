# syntax=docker/dockerfile:1.7
# OpsRAG frontend image (Streamlit). Build from the repository root:
#
#   docker build -f frontend/Dockerfile -t opsrag-frontend .
#
# The UI talks to the API over HTTP only (OPSRAG_API_URL) and imports nothing from app/.

ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    OPSRAG_API_URL=http://backend:8000

WORKDIR /srv/frontend
COPY requirements-frontend.txt /tmp/requirements-frontend.txt
RUN pip install -r /tmp/requirements-frontend.txt \
 && groupadd --system --gid 10002 opsrag-ui \
 && useradd --system --uid 10002 --gid 10002 --create-home --home-dir /home/opsrag-ui opsrag-ui
# Run from here: Streamlit reads the theme from ./.streamlit/config.toml.
COPY frontend/ ./
USER 10002:10002
EXPOSE 8501
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4)"]
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
