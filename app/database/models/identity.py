from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.database.models.delivery import Deployment, PullRequest


class Role(TimestampMixin, Base):
    __tablename__ = "roles"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128), unique=True)
    description: Mapped[str] = mapped_column(Text)
    # What a role may read is defined by the access policy (app/security/policy.json).

    users: Mapped[list[User]] = relationship(back_populates="role")


class User(TimestampMixin, Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # username
    email: Mapped[str] = mapped_column(String(256), unique=True)
    full_name: Mapped[str] = mapped_column(String(128))
    team: Mapped[str] = mapped_column(String(64), index=True)
    role_id: Mapped[str] = mapped_column(ForeignKey("roles.id"), index=True)
    is_active: Mapped[bool] = mapped_column(default=True)

    role: Mapped[Role] = relationship(back_populates="users")
    deployments: Mapped[list[Deployment]] = relationship(back_populates="author")
    pull_requests: Mapped[list[PullRequest]] = relationship(back_populates="author")
