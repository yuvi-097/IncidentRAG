"""Create (or update) the read-only database role that query_database connects as.

    TOOLS_SQL_USER=opsrag_sql_reader TOOLS_SQL_PASSWORD=... python scripts/create_sql_reader.py

The role can log in, is read-only by default, and may SELECT exactly the tables the
tool exposes: nothing on users, roles, chunks or embeddings, and no write privilege
anywhere. The script connects with the POSTGRES_* credentials, which must be allowed to
create roles (the compose database user is). seed_db.py re-applies the grants whenever
it recreates tables. Idempotent.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import load_settings  # noqa: E402
from app.database.session import create_db_engine  # noqa: E402
from app.tools.sql_tool import READABLE_TABLES, ensure_sql_reader  # noqa: E402


def main() -> int:
    settings = load_settings()
    tools = settings.tools
    if not tools.sql_user or not tools.sql_password:
        print("Set TOOLS_SQL_USER and TOOLS_SQL_PASSWORD (in .env) first.", file=sys.stderr)
        return 2
    if tools.sql_user == settings.database.user:
        print("TOOLS_SQL_USER must differ from POSTGRES_USER.", file=sys.stderr)
        return 2
    engine = create_db_engine(settings.database, application_name="opsrag-admin")
    try:
        ensure_sql_reader(engine, tools.sql_user, tools.sql_password.get_secret_value())
    finally:
        engine.dispose()
    print(f"role {tools.sql_user}: read-only, SELECT on {', '.join(READABLE_TABLES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
