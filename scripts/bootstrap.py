"""Prepare a deployment's database: the one-shot ``bootstrap`` service of docker compose.

    python scripts/bootstrap.py

Steps, each a script of its own (so every step can also be run by hand):

1. Wait until the database accepts connections (``BOOTSTRAP_WAIT_SECONDS``, default 120).
2. Generate the synthetic dataset if ``data/generated`` is missing (the image has it).
3. Load it **only into an empty database** (``seed_db.py --if-empty``). Existing data is
   never replaced, so restarting the stack keeps everything, embeddings included.
4. Create or update the read-only SQL role, when ``TOOLS_SQL_USER`` is set
   (``create_sql_reader.py``).
5. Chunk new or changed sources (``ingest.py``; unchanged chunks are left alone).
6. Embed new or changed chunks (``embed.py``). The first start embeds the whole corpus,
   which takes minutes on a CPU; later starts find nothing to do.

Exits non-zero at the first failing step, so the services that depend on it (the
backend) do not start on a half-prepared database.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sqlalchemy import text  # noqa: E402

from app.config import load_settings  # noqa: E402
from app.database.session import create_db_engine  # noqa: E402

SCRIPTS = ROOT / "scripts"


def log(message: str) -> None:
    print(f"[bootstrap] {message}", flush=True)


def wait_for_database(seconds: float) -> bool:
    settings = load_settings()
    engine = create_db_engine(settings.database, application_name="opsrag-bootstrap")
    deadline = time.monotonic() + seconds
    try:
        while True:
            try:
                with engine.connect() as connection:
                    connection.execute(text("SELECT 1"))
                log(f"database {settings.database.safe_url} is up")
                return True
            except Exception as exc:  # not up yet: retry until the deadline
                if time.monotonic() > deadline:
                    log(f"database not reachable after {seconds:.0f}s: {type(exc).__name__}")
                    return False
                time.sleep(2)
    finally:
        engine.dispose()


def run(script: str, *args: str) -> bool:
    started = time.monotonic()
    log(f"{script} {' '.join(args)}".strip())
    result = subprocess.run([sys.executable, str(SCRIPTS / script), *args], cwd=ROOT)
    elapsed = time.monotonic() - started
    if result.returncode != 0:
        log(f"{script} failed (exit {result.returncode}) after {elapsed:.1f}s")
        return False
    log(f"{script} done in {elapsed:.1f}s")
    return True


def main() -> int:
    started = time.monotonic()
    if not wait_for_database(float(os.environ.get("BOOTSTRAP_WAIT_SECONDS", "120"))):
        return 1
    steps: list[tuple[str, ...]] = []
    if not (ROOT / "data" / "generated" / "manifest.json").exists():
        steps.append(("generate_data.py",))
    steps.append(("seed_db.py", "--if-empty"))
    settings = load_settings()
    if settings.tools.sql_user and settings.tools.sql_password:
        steps.append(("create_sql_reader.py",))
    steps += [("ingest.py",), ("embed.py",)]
    for step in steps:
        if not run(*step):
            return 1
    log(f"ready in {time.monotonic() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
