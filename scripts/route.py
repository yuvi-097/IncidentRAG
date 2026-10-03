"""Show how the router classifies a question and which tools it would use.

    python scripts/route.py "What caused INC-0421?" [--user arjun.mehta] [--json]

With --user, tools the user may not call are listed separately (needs the
database); without it, services come from the database if reachable, else from
the generated dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.agents.entities import ServiceCatalog  # noqa: E402
from app.agents.router import RuleBasedRouter  # noqa: E402
from app.config import load_settings  # noqa: E402
from app.database.session import create_db_engine  # noqa: E402
from app.security import PrincipalError, load_policy, load_principal  # noqa: E402
from app.tools import build_registry  # noqa: E402


def _dataset_services() -> list[str]:
    path = ROOT / "data/generated/services.jsonl"
    return [
        json.loads(line)["id"] for line in path.read_text(encoding="utf-8").splitlines() if line
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("query")
    parser.add_argument("--user")
    parser.add_argument("--json", action="store_true", help="print the full decision")
    args = parser.parse_args(argv)

    permitted = None
    if args.user:
        settings = load_settings()
        engine = create_db_engine(settings.database, application_name="opsrag-route")
        try:
            catalog = ServiceCatalog.from_engine(engine)
            principal = load_principal(
                engine, args.user, load_policy(settings.security.policy_file)
            )
        except PrincipalError as exc:
            print(exc, file=sys.stderr)
            return 2
        finally:
            engine.dispose()
        permitted = build_registry().permitted(principal)
    else:
        catalog = ServiceCatalog(_dataset_services())
    decision = RuleBasedRouter(catalog).route(args.query, permitted=permitted)
    if args.json:
        print(json.dumps(decision.model_dump(mode="json"), indent=2))
        return 0
    print(f"{decision.query_type.value} ({decision.confidence.value}): {decision.summary}")
    print(f"tools: {', '.join(decision.tools) or '-'}")
    if decision.denied_tools:
        print(f"not permitted: {', '.join(decision.denied_tools)}")
    entities = {k: v for k, v in decision.entities.model_dump(mode="json").items() if v}
    if entities:
        print(f"entities: {json.dumps(entities)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
