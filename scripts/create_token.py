"""Issue, list or revoke API tokens.

    python scripts/create_token.py issue alex.rivera --name laptop [--days 90]
    python scripts/create_token.py list [--user alex.rivera]
    python scripts/create_token.py revoke <token-id>

An issued token is printed once; only a hash of it is stored. Send it as
``Authorization: Bearer <token>`` to /api/agent/ask.
"""

from __future__ import annotations

import argparse
import sys
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import load_settings  # noqa: E402
from app.database.schema import create_schema  # noqa: E402
from app.database.session import create_db_engine  # noqa: E402
from app.security.auth import issue_token, list_tokens, revoke_token  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    issue = commands.add_parser("issue", help="create a token for a user")
    issue.add_argument("user")
    issue.add_argument("--name", required=True, help="what the token is for")
    issue.add_argument("--days", type=int, default=90, help="lifetime; 0 = no expiry")
    listing = commands.add_parser("list", help="show tokens (never their secrets)")
    listing.add_argument("--user")
    revoke = commands.add_parser("revoke", help="revoke a token by its id")
    revoke.add_argument("token_id")
    args = parser.parse_args(argv)

    settings = load_settings()
    engine = create_db_engine(settings.database, application_name="opsrag-tokens")
    try:
        create_schema(engine)  # the tokens table may not exist yet
        if args.command == "issue":
            if args.days < 0:
                parser.error("--days must be 0 or more")
            ttl = timedelta(days=args.days) if args.days else None
            try:
                token = issue_token(engine, args.user, args.name, ttl)
            except ValueError as exc:
                print(exc, file=sys.stderr)
                return 2
            print(token)
            print("Store it now: it is not shown again.", file=sys.stderr)
        elif args.command == "list":
            for row in list_tokens(engine, args.user):
                state = "revoked" if row["revoked_at"] else "active"
                print(
                    f"{row['id']}  {row['user_id']:<18} {row['name']:<16} {state:<8} "
                    f"expires {row['expires_at'] or 'never'}  last used {row['last_used_at']}"
                )
        elif not revoke_token(engine, args.token_id):
            print(f"no active token {args.token_id}", file=sys.stderr)
            return 1
    finally:
        engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
