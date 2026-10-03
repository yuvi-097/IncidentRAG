"""API tokens: format, what is stored, expiry, revocation, and surviving re-seeding."""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine, select

from app.database.models import ApiToken
from app.database.schema import create_schema
from app.database.seed import seed_database
from app.security.auth import (
    AuthenticationError,
    authenticate,
    issue_token,
    list_tokens,
    revoke_token,
)
from app.synthetic.records import SyntheticDataset
from tests.tools.conftest import ToolEnv

USER = "alex.rivera"


def test_tokens_are_random_and_only_their_hash_is_stored(tool_env: ToolEnv) -> None:
    first, second = (issue_token(tool_env.engine, USER, "t") for _ in range(2))
    assert first != second
    assert re.fullmatch(r"opsrag_[0-9a-f]{12}_[A-Za-z0-9_-]{43}", first)
    token_id, secret = first.split("_")[1], first.split("_", 2)[2]
    with tool_env.engine.connect() as connection:
        row = connection.execute(select(ApiToken.__table__).where(ApiToken.id == token_id)).one()
    assert secret not in str(tuple(row)) and len(row.secret_sha256) == 64
    assert authenticate(tool_env.engine, first) == USER
    listed = list_tokens(tool_env.engine, USER)
    assert all("secret_sha256" not in r and secret not in str(r) for r in listed)


def test_expiry_and_revocation(tool_env: ToolEnv) -> None:
    now = datetime(2026, 9, 1, tzinfo=UTC)
    token = issue_token(tool_env.engine, USER, "t", timedelta(hours=1), clock=lambda: now)
    assert authenticate(tool_env.engine, token, clock=lambda: now) == USER
    later = now + timedelta(hours=2)
    with pytest.raises(AuthenticationError):
        authenticate(tool_env.engine, token, clock=lambda: later)
    forever = issue_token(tool_env.engine, USER, "t", ttl=None)
    token_id = forever.split("_")[1]
    assert revoke_token(tool_env.engine, token_id) and not revoke_token(tool_env.engine, token_id)
    with pytest.raises(AuthenticationError):
        authenticate(tool_env.engine, forever)


def test_no_token_for_unknown_or_inactive_users(tool_env: ToolEnv) -> None:
    for user in ("nobody", "contractor.docs"):
        with pytest.raises(ValueError, match="unknown or inactive"):
            issue_token(tool_env.engine, user, "t")


def test_tokens_survive_reseeding(dataset: SyntheticDataset) -> None:
    engine = create_engine("sqlite://")
    create_schema(engine)
    seed_database(engine, dataset)
    token = issue_token(engine, USER, "t")
    seed_database(engine, dataset)  # replaces every dataset table, users included
    assert authenticate(engine, token) == USER
    engine.dispose()
