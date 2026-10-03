# OpsRAG developer commands. Requires GNU make and a POSIX shell
# (Linux/macOS, WSL, or Git Bash on Windows). See README for plain commands.

ifeq ($(OS),Windows_NT)
VENV_PYTHON := .venv/Scripts/python.exe
else
VENV_PYTHON := .venv/bin/python
endif
PYTHON ?= $(VENV_PYTHON)
HOST ?= 127.0.0.1
PORT ?= 8000

.DEFAULT_GOAL := help
.PHONY: help venv install env run ui screenshots perf load-test data seed ingest embed benchmark compare route-eval ask test test-integration test-model lint format up up-demo down db-up docker-test docker-test-integration logs health clean

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-18s %s\n", $$1, $$2}'

venv: ## Create the virtualenv in .venv
	python -m venv .venv

install: ## Install runtime + dev dependencies into the virtualenv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements-dev.txt

env: ## Create .env from .env.example (does not overwrite)
	@test -f .env && echo ".env already exists" || (cp .env.example .env && echo "created .env")

run: ## Run the API locally with auto-reload
	$(PYTHON) -m uvicorn app.main:app --reload --host $(HOST) --port $(PORT)

# Run from frontend/ so Streamlit reads frontend/.streamlit/config.toml (the theme).
ui: ## Run the Streamlit frontend (needs the API running; see README > Frontend)
	cd frontend && $(if $(filter $(VENV_PYTHON),$(PYTHON)),../)$(PYTHON) -m streamlit run app.py

screenshots: ## Save a screenshot of every frontend page into docs/screenshots (needs API + UI)
	$(PYTHON) scripts/screenshot_ui.py

perf: ## Performance benchmarks (needs a seeded, embedded database): make perf ARGS="--quick"
	$(PYTHON) scripts/benchmark_performance.py $(ARGS)

load-test: ## Load test the running API (local demo users): make load-test ARGS="--concurrency 1,4"
	$(PYTHON) scripts/load_test.py $(ARGS)

data: ## Generate the synthetic NovaCart dataset into data/generated (deterministic)
	$(PYTHON) scripts/generate_data.py

seed: ## Load data/generated into PostgreSQL (replaces OpsRAG table contents)
	$(PYTHON) scripts/seed_db.py

ingest: ## Chunk all sources from PostgreSQL into document_chunks (idempotent)
	$(PYTHON) scripts/ingest.py

embed: ## Embed new/changed chunks with the configured model (EMBEDDING_*)
	$(PYTHON) scripts/embed.py

benchmark: ## Run the retrieval benchmark for RETRIEVAL_MODE
	$(PYTHON) scripts/benchmark_retrieval.py

compare: ## Compare dense, BM25, hybrid and hybrid + reranker on the benchmark
	$(PYTHON) scripts/compare_retrieval.py

route-eval: ## Evaluate the query router on the labelled routing set
	$(PYTHON) scripts/evaluate_routing.py

ask: ## Ask the agent: make ask Q="What caused INC-0406?" USER=arjun.mehta
	$(PYTHON) scripts/ask.py "$(Q)" --user $(USER)

test: ## Run the unit test suite
	$(PYTHON) -m pytest

test-integration: ## Run integration tests (needs a running PostgreSQL, e.g. `make db-up`)
	OPSRAG_RUN_INTEGRATION_TESTS=1 $(PYTHON) -m pytest -m integration

test-model: ## Run retrieval tests with the real embedding and reranker models (slow on CPU)
	OPSRAG_RUN_MODEL_TESTS=1 $(PYTHON) -m pytest -m model

lint: ## Lint and check formatting
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format: ## Auto-format and apply safe lint fixes
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check --fix .

up: ## Build and start the whole stack (postgres, bootstrap, backend, frontend) in Docker
	docker compose up --build -d

up-demo: ## The stack with the demo users (local demo only; no API tokens needed)
	docker compose -f docker-compose.yml -f docker-compose.demo.yml up --build -d

down: ## Stop the Docker stack (the database volume is kept; add -v by hand to delete it)
	docker compose down

db-up: ## Start only PostgreSQL + pgvector in Docker
	docker compose up -d postgres

docker-test: ## Run the unit tests in the test image (needs the postgres service)
	docker compose --profile test run --rm tests

docker-test-integration: ## Run the integration tests in the test image, on the opsrag_test database
	docker compose --profile test run --rm -e OPSRAG_RUN_INTEGRATION_TESTS=1 tests pytest -m integration -p no:cacheprovider

logs: ## Tail Docker logs
	docker compose logs -f

health: ## Query the health endpoint
	curl -s http://$(HOST):$(PORT)/api/health

clean: ## Remove caches
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -not -path "./.venv/*" -prune -exec rm -rf {} +
