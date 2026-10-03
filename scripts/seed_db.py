"""Load the generated dataset into PostgreSQL.

    python scripts/seed_db.py [--data-dir data/generated] [--recreate-schema]

Behaviour (idempotent):
  * enables the pgvector extension and creates missing tables;
  * REPLACES the contents of every OpsRAG table with the dataset, in one
    transaction (rows added by hand are removed; a failure changes nothing);
  * with --recreate-schema, DROPS and recreates all OpsRAG tables first
    (needed after model changes; destroys document_chunks embeddings too);
  * with --if-empty, loads only into a database without OpsRAG data and otherwise
    changes nothing (the container bootstrap uses this on every start).

Refuses to replace data when OPSRAG_ENVIRONMENT=production: there, only --if-empty runs,
which never overwrites anything.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sqlalchemy import func, inspect, select  # noqa: E402
from sqlalchemy.engine import Engine  # noqa: E402

from app.config import load_settings  # noqa: E402
from app.database.models import Incident  # noqa: E402
from app.database.schema import prepare_schema  # noqa: E402
from app.database.seed import seed_database, table_counts  # noqa: E402
from app.database.session import create_db_engine  # noqa: E402
from app.observability.structured_logging import configure_logging  # noqa: E402
from app.synthetic.storage import load_dataset  # noqa: E402
from app.tools.sql_tool import ensure_sql_reader  # noqa: E402


def has_data(engine: Engine) -> bool:
    """Whether the database already holds OpsRAG records (any incident)."""
    if not inspect(engine).has_table(Incident.__tablename__):
        return False
    with engine.connect() as connection:
        return bool(connection.execute(select(func.count()).select_from(Incident)).scalar())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "generated")
    parser.add_argument(
        "--recreate-schema",
        action="store_true",
        help="drop and recreate the OpsRAG tables first (done automatically when the "
        "schema is from an earlier version)",
    )
    parser.add_argument(
        "--if-empty",
        action="store_true",
        help="load only into a database without OpsRAG data; otherwise change nothing",
    )
    args = parser.parse_args(argv)

    settings = load_settings()
    configure_logging(settings.app.log_level, settings.app.log_format)
    if settings.app.environment == "production" and not (
        args.if_empty and not args.recreate_schema
    ):
        print(
            "Refusing to replace data in a production environment "
            "(--if-empty loads into an empty database only).",
            file=sys.stderr,
        )
        return 2

    engine = create_db_engine(settings.database, application_name="opsrag-seed")
    if args.if_empty and has_data(engine):
        engine.dispose()
        print(f"{settings.database.safe_url} already holds OpsRAG data: nothing changed.")
        return 0
    dataset = load_dataset(args.data_dir)
    try:
        started = time.perf_counter()
        drift = prepare_schema(engine, recreate=args.recreate_schema)
        if drift:
            print("The schema was from an earlier version and has been recreated:")
            for problem in drift[:10]:
                print(f"  - {problem}")
        seed_database(engine, dataset)
        tools = settings.tools
        if tools.sql_user and tools.sql_password:
            try:  # create or update the read-only role and (re)grant its tables
                ensure_sql_reader(engine, tools.sql_user, tools.sql_password.get_secret_value())
                print(f"SQL reader role {tools.sql_user}: ready (read-only)")
            except Exception as exc:  # e.g. the database user may not create roles
                print(
                    f"Could not set up {tools.sql_user} (see scripts/create_sql_reader.py): "
                    f"{str(exc).splitlines()[0]}",
                    file=sys.stderr,
                )
        counts = table_counts(engine)
    finally:
        engine.dispose()

    print(f"Seeded {settings.database.safe_url} in {time.perf_counter() - started:.1f}s")
    for table, count in counts.items():
        print(f"  {table:<22} {count:>7}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
