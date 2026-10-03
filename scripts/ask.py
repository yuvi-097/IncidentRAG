"""Ask the agent a question as a given user.

    python scripts/ask.py "Why did payment-service fail after deployment v2.8.1?" --user arjun.mehta
    python scripts/ask.py "How many payment incidents happened last month?" --user arjun.mehta
    python scripts/ask.py "What caused INC-0406?" --user arjun.mehta --json

Uses the configured retrieval pipeline, reranker and LLM (LLM_PROVIDER=none gives an
extractive answer). Prints the answer, its citations, and one line per stage.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.agents.response import AgentResponse  # noqa: E402
from app.config import load_settings  # noqa: E402
from app.database.session import create_db_engine  # noqa: E402
from app.security import PrincipalError, load_policy, load_principal  # noqa: E402
from app.services.agent import build_agent  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    settings = load_settings()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("question")
    parser.add_argument("--user", required=True)
    parser.add_argument("--json", action="store_true", help="print the full API response")
    args = parser.parse_args(argv)

    engine = create_db_engine(settings.database, application_name="opsrag-ask")
    try:
        try:
            principal = load_principal(
                engine, args.user, load_policy(settings.security.policy_file)
            )
        except PrincipalError as exc:
            print(exc, file=sys.stderr)
            return 2
        response = AgentResponse.from_state(
            build_agent(engine, settings).run(args.question, principal)
        )
    finally:
        engine.dispose()
    if args.json:
        print(json.dumps(response.model_dump(mode="json"), indent=2))
        return 0
    print(response.answer)
    print(f"\nconfidence: {response.confidence.value} ({'; '.join(response.confidence_reasons)})")
    for citation in response.citations:
        where = f" - {citation.location}" if citation.location else ""
        source = f"{citation.kind.value} {citation.source_id}"
        print(f"  [{citation.label}] {source}: {citation.title}{where}")
    print("\nsteps:")
    for line in response.reasoning_summary:
        print(f"  {line}")
    for limitation in response.limitations:
        print(f"limitation: {limitation}")
    print(f"\n{response.latency_ms.get('total', 0):.0f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
