"""Call one registered tool as a given user (for manual testing).

    python scripts/call_tool.py --list --user arjun.mehta
    python scripts/call_tool.py search_incidents --user arjun.mehta
        --args '{"incident_ids": ["INC-0406"]}'
    python scripts/call_tool.py query_database --user arjun.mehta
        --args '{"sql": "SELECT count(*) FROM incidents"}'

Permissions and clearance come from the user's role. Text-search tools use the
configured retrieval pipeline (RETRIEVAL_MODE); pass --mode to override.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import load_settings  # noqa: E402
from app.database.session import create_db_engine, create_sql_reader_engine  # noqa: E402
from app.rag.retrieval.factory import RetrievalComponents  # noqa: E402
from app.schemas.enums import RetrievalMode  # noqa: E402
from app.security import PrincipalError, load_policy, load_principal  # noqa: E402
from app.tools import ToolContext, build_registry  # noqa: E402

TEXT_SEARCH = {
    "search_documents",
    "search_code",
    "search_incidents",
    "search_deployments",
    "get_runbook",
}


def main(argv: list[str] | None = None) -> int:
    settings = load_settings()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("tool", nargs="?")
    parser.add_argument("--user", required=True)
    parser.add_argument("--args", default="{}", help="tool arguments as a JSON object")
    parser.add_argument("--list", action="store_true", help="list the tools this user may call")
    parser.add_argument("--mode", choices=[m.value for m in RetrievalMode])
    args = parser.parse_args(argv)

    registry = build_registry()
    engine = create_db_engine(settings.database, application_name="opsrag-tools")
    try:
        try:
            principal = load_principal(
                engine, args.user, load_policy(settings.security.policy_file)
            )
        except PrincipalError as exc:
            print(exc, file=sys.stderr)
            return 2
        retriever = None
        if args.tool in TEXT_SEARCH:
            mode = RetrievalMode(args.mode) if args.mode else None
            retriever = RetrievalComponents(engine, settings).retriever(mode)
        context = ToolContext(
            engine=engine,
            principal=principal,
            settings=settings.tools,
            retriever=retriever,
            sql_engine=create_sql_reader_engine(settings),
        )
        if args.list or not args.tool:
            grants = "; ".join(
                f"{r.value}: {', '.join(principal.visible_levels(r))}" for r in principal.grants
            )
            print(f"{principal.user_id} ({principal.role}) may read {grants}")
            for spec in registry.specs(context):
                print(f"  {spec.name:<20} {spec.description[:90]}")
            return 0
        result = registry.call(args.tool, json.loads(args.args), context)
    finally:
        engine.dispose()
    print(json.dumps(result.model_dump(mode="json"), indent=2, default=str))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
