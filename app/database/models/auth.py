"""API tokens: how callers of the HTTP API prove who they are.

Only a SHA-256 hash of each token's secret is stored. ``user_id`` is deliberately not a
foreign key: seeding replaces the users table, and tokens must survive that. Whether
the user still exists and is active is checked on every request instead.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base


class ApiToken(Base):
    __tablename__ = "api_tokens"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)  # public part, for lookup
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    name: Mapped[str] = mapped_column(String(128))  # what the token is for ("laptop", "ci")
    secret_sha256: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime]
    expires_at: Mapped[datetime | None]
    revoked_at: Mapped[datetime | None]
    last_used_at: Mapped[datetime | None]
