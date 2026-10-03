"""Declarative base, naming conventions and shared column helpers."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import JSON, DateTime, MetaData, func
from sqlalchemy import Enum as SAEnum
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Deterministic constraint names (stable across environments and migrations).
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

# JSONB on PostgreSQL, plain JSON elsewhere (SQLite in unit tests).
JSONType = JSON().with_variant(JSONB(), "postgresql")


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)
    # SQLAlchemy's documented class-level configuration hook, not instance state.
    type_annotation_map = {  # noqa: RUF012
        dict[str, Any]: JSONType,
        list[str]: JSONType,
        list[dict[str, Any]]: JSONType,
        datetime: DateTime(timezone=True),
    }


def enum_column(enum_cls: type[StrEnum], name: str) -> SAEnum:
    """Store enum *values* as VARCHAR + CHECK constraint (no native PG enum types,
    which are awkward to migrate)."""
    return SAEnum(
        enum_cls,
        name=name,
        native_enum=False,
        create_constraint=True,
        validate_strings=True,
        values_callable=lambda members: [member.value for member in members],
        length=max(len(member.value) for member in enum_cls),
    )


class TimestampMixin:
    """When the record was created / last changed in its source system.

    The generator supplies historical values; the server default applies when a
    caller does not provide one.
    """

    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())
