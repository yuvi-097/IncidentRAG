"""Authentication with API tokens.

A token is ``opsrag_<id>_<secret>``: ``id`` (12 hex characters) finds the row, and the
``secret`` (43 URL-safe characters, 256 bits from ``secrets``) proves possession. Only
the SHA-256 of the secret is stored, so a copy of the database does not reveal usable
tokens; a slow hash is unnecessary because the secret is random, not a password. The
comparison is constant-time, and an unknown id costs the same as a wrong secret.

A token is valid while it is neither revoked nor expired, and while its user exists and
is active (checked on every request, by ``load_principal``). Tokens are shown once, when
issued, and never logged.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import re
import secrets
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from sqlalchemy import select, update
from sqlalchemy.engine import Engine

from app.database.models import ApiToken, User

logger = logging.getLogger(__name__)

PREFIX = "opsrag"
_TOKEN = re.compile(rf"^{PREFIX}_([0-9a-f]{{12}})_([A-Za-z0-9_-]{{43}})$")
_DUMMY_HASH = hashlib.sha256(b"no such token").hexdigest()


class AuthenticationError(PermissionError):
    """The token is malformed, unknown, wrong, expired or revoked (never says which to
    the caller)."""


def _hash(secret: str) -> str:
    return hashlib.sha256(secret.encode()).hexdigest()


def _utcnow() -> datetime:
    return datetime.now(UTC)


def issue_token(
    engine: Engine,
    user_id: str,
    name: str,
    ttl: timedelta | None = timedelta(days=90),
    clock: Callable[[], datetime] = _utcnow,
) -> str:
    """Create a token for an active user and return it; this is the only time the full
    token exists outside the caller's hands."""
    with engine.begin() as connection:
        active = connection.execute(
            select(User.is_active).where(User.id == user_id)
        ).scalar_one_or_none()
        if not active:
            raise ValueError(f"unknown or inactive user: {user_id!r}")
        token_id, secret = secrets.token_hex(6), secrets.token_urlsafe(32)
        now = clock()
        connection.execute(
            ApiToken.__table__.insert().values(
                id=token_id,
                user_id=user_id,
                name=name[:128],
                secret_sha256=_hash(secret),
                created_at=now,
                expires_at=now + ttl if ttl else None,
                revoked_at=None,
                last_used_at=None,
            )
        )
    logger.info("auth.token_issued", extra={"token_id": token_id, "user": user_id})
    return f"{PREFIX}_{token_id}_{secret}"


def authenticate(engine: Engine, token: str, clock: Callable[[], datetime] = _utcnow) -> str:
    """The user id a valid token belongs to; ``AuthenticationError`` otherwise."""
    match = _TOKEN.fullmatch(token.strip())
    if match is None:
        logger.warning("auth.failed", extra={"reason": "malformed"})
        raise AuthenticationError("invalid token")
    token_id, secret = match.groups()
    with engine.connect() as connection:
        row = connection.execute(select(ApiToken.__table__).where(ApiToken.id == token_id)).first()
    expected = row.secret_sha256 if row is not None else _DUMMY_HASH
    if not hmac.compare_digest(_hash(secret), expected) or row is None:
        logger.warning("auth.failed", extra={"reason": "unknown", "token_id": token_id})
        raise AuthenticationError("invalid token")
    now = clock()
    if row.revoked_at is not None:
        logger.warning("auth.failed", extra={"reason": "revoked", "token_id": token_id})
        raise AuthenticationError("invalid token")
    expires = row.expires_at
    if expires is not None and (expires if expires.tzinfo else expires.replace(tzinfo=UTC)) <= now:
        logger.warning("auth.failed", extra={"reason": "expired", "token_id": token_id})
        raise AuthenticationError("invalid token")
    with engine.begin() as connection:
        connection.execute(update(ApiToken).where(ApiToken.id == token_id).values(last_used_at=now))
    return str(row.user_id)


def revoke_token(engine: Engine, token_id: str, clock: Callable[[], datetime] = _utcnow) -> bool:
    with engine.begin() as connection:
        result = connection.execute(
            update(ApiToken)
            .where(ApiToken.id == token_id, ApiToken.revoked_at.is_(None))
            .values(revoked_at=clock())
        )
    revoked = bool(result.rowcount)
    if revoked:
        logger.info("auth.token_revoked", extra={"token_id": token_id})
    return revoked


def list_tokens(engine: Engine, user_id: str | None = None) -> list[dict[str, object]]:
    """Token metadata (never the secret or its hash)."""
    query = select(
        ApiToken.id,
        ApiToken.user_id,
        ApiToken.name,
        ApiToken.created_at,
        ApiToken.expires_at,
        ApiToken.revoked_at,
        ApiToken.last_used_at,
    ).order_by(ApiToken.created_at)
    if user_id:
        query = query.where(ApiToken.user_id == user_id)
    with engine.connect() as connection:
        return [dict(row._mapping) for row in connection.execute(query)]


__all__ = [
    "AuthenticationError",
    "authenticate",
    "issue_token",
    "list_tokens",
    "revoke_token",
]
