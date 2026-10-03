from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, TimestampMixin, enum_column
from app.schemas.enums import DependencyCriticality, DependencyProtocol, ServiceTier

if TYPE_CHECKING:
    from app.database.models.delivery import CodeFile, Deployment
    from app.database.models.operations import Incident


class Service(TimestampMixin, Base):
    __tablename__ = "services"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # e.g. "payment-service"
    display_name: Mapped[str] = mapped_column(String(128))
    description: Mapped[str] = mapped_column(Text)
    owner_team: Mapped[str] = mapped_column(String(64), index=True)
    tier: Mapped[ServiceTier] = mapped_column(enum_column(ServiceTier, "tier"))
    language: Mapped[str] = mapped_column(String(32))
    repository_path: Mapped[str] = mapped_column(String(256))
    port: Mapped[int] = mapped_column(Integer)
    oncall_channel: Mapped[str] = mapped_column(String(64))
    # e.g. ["postgres:payments", "redis:redis-payments", "kafka"]
    datastores: Mapped[list[str]]

    dependencies: Mapped[list[ServiceDependency]] = relationship(
        foreign_keys="ServiceDependency.service_id", back_populates="service"
    )
    dependents: Mapped[list[ServiceDependency]] = relationship(
        foreign_keys="ServiceDependency.depends_on_id", back_populates="depends_on"
    )
    deployments: Mapped[list[Deployment]] = relationship(back_populates="service")
    incidents: Mapped[list[Incident]] = relationship(
        foreign_keys="Incident.service_id", back_populates="service"
    )
    code_files: Mapped[list[CodeFile]] = relationship(back_populates="service")


class ServiceDependency(Base):
    """Directed edge: ``service_id`` calls / consumes from ``depends_on_id``."""

    __tablename__ = "service_dependencies"
    __table_args__ = (CheckConstraint("service_id <> depends_on_id", name="no_self_dependency"),)

    service_id: Mapped[str] = mapped_column(ForeignKey("services.id"), primary_key=True)
    depends_on_id: Mapped[str] = mapped_column(
        ForeignKey("services.id"), primary_key=True, index=True
    )
    protocol: Mapped[DependencyProtocol] = mapped_column(
        enum_column(DependencyProtocol, "protocol")
    )
    criticality: Mapped[DependencyCriticality] = mapped_column(
        enum_column(DependencyCriticality, "criticality")
    )
    description: Mapped[str] = mapped_column(Text)

    service: Mapped[Service] = relationship(
        foreign_keys=[service_id], back_populates="dependencies"
    )
    depends_on: Mapped[Service] = relationship(
        foreign_keys=[depends_on_id], back_populates="dependents"
    )
