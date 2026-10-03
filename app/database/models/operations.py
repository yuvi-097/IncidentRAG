"""Incidents and logs: what went wrong, and the telemetry around it."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import BigInteger, CheckConstraint, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, TimestampMixin, enum_column
from app.database.models.delivery import Deployment, PullRequest
from app.database.models.identity import User
from app.database.models.knowledge import Document
from app.database.models.services import Service
from app.schemas.enums import AccessLevel, IncidentCategory, IncidentStatus, LogLevel, Severity


class Incident(TimestampMixin, Base):
    """A production incident.

    Deployment links have distinct meanings:
      deployment_id              what ``service_id`` was running when the incident began
                                 (``affected_version`` is its version); always set
      root_cause_deployment_id   the deployment that introduced the fault, possibly in
                                 another service; NULL when not deployment-related
      remediation_deployment_id  the rollback / hotfix that mitigated it, if any
    """

    __tablename__ = "incidents"
    __table_args__ = (
        Index("ix_incidents_service_id_started_at", "service_id", "started_at"),
        CheckConstraint("detected_at >= started_at", name="detected_after_start"),
        CheckConstraint("resolved_at >= detected_at", name="resolved_after_detected"),
        CheckConstraint("resolution_time_minutes >= 0", name="resolution_time_non_negative"),
    )

    id: Mapped[str] = mapped_column(String(16), primary_key=True)  # "INC-0001"
    title: Mapped[str] = mapped_column(String(256))
    service_id: Mapped[str] = mapped_column(ForeignKey("services.id"))
    root_cause_service_id: Mapped[str | None] = mapped_column(ForeignKey("services.id"), index=True)
    parent_incident_id: Mapped[str | None] = mapped_column(ForeignKey("incidents.id"), index=True)
    category: Mapped[IncidentCategory] = mapped_column(
        enum_column(IncidentCategory, "category"), index=True
    )
    severity: Mapped[Severity] = mapped_column(enum_column(Severity, "severity"), index=True)
    status: Mapped[IncidentStatus] = mapped_column(enum_column(IncidentStatus, "status"))
    started_at: Mapped[datetime]
    detected_at: Mapped[datetime]
    resolved_at: Mapped[datetime]
    resolution_time_minutes: Mapped[int] = mapped_column(Integer)
    affected_version: Mapped[str] = mapped_column(String(32))
    deployment_id: Mapped[str] = mapped_column(ForeignKey("deployments.id"), index=True)
    root_cause_deployment_id: Mapped[str | None] = mapped_column(
        ForeignKey("deployments.id"), index=True
    )
    root_cause_pr_id: Mapped[str | None] = mapped_column(ForeignKey("pull_requests.id"), index=True)
    remediation_deployment_id: Mapped[str | None] = mapped_column(ForeignKey("deployments.id"))
    runbook_id: Mapped[str | None] = mapped_column(ForeignKey("documents.id"), index=True)
    postmortem_id: Mapped[str | None] = mapped_column(ForeignKey("documents.id"))
    commander_id: Mapped[str] = mapped_column(ForeignKey("users.id"))
    alert_name: Mapped[str | None] = mapped_column(String(128), index=True)
    symptoms: Mapped[str] = mapped_column(Text)
    root_cause: Mapped[str] = mapped_column(Text)
    resolution: Mapped[str] = mapped_column(Text)
    # e.g. {"peak_error_rate_pct": 38.2, "p99_latency_ms": 5400, "failed_requests": 41210}
    metrics: Mapped[dict[str, Any]]
    tags: Mapped[list[str]]
    access_level: Mapped[AccessLevel] = mapped_column(enum_column(AccessLevel, "access_level"))

    service: Mapped[Service] = relationship(foreign_keys=[service_id], back_populates="incidents")
    root_cause_service: Mapped[Service | None] = relationship(foreign_keys=[root_cause_service_id])
    parent_incident: Mapped[Incident | None] = relationship(
        remote_side=[id], back_populates="child_incidents"
    )
    child_incidents: Mapped[list[Incident]] = relationship(back_populates="parent_incident")
    deployment: Mapped[Deployment] = relationship(foreign_keys=[deployment_id])
    root_cause_deployment: Mapped[Deployment | None] = relationship(
        foreign_keys=[root_cause_deployment_id]
    )
    remediation_deployment: Mapped[Deployment | None] = relationship(
        foreign_keys=[remediation_deployment_id]
    )
    root_cause_pull_request: Mapped[PullRequest | None] = relationship()
    runbook: Mapped[Document | None] = relationship(foreign_keys=[runbook_id])
    postmortem: Mapped[Document | None] = relationship(foreign_keys=[postmortem_id])
    commander: Mapped[User] = relationship()


class LogEntry(Base):
    """A structured application log line (append-only, high volume)."""

    __tablename__ = "logs"
    __table_args__ = (Index("ix_logs_service_id_timestamp", "service_id", "timestamp"),)

    # BIGINT on PostgreSQL; SQLite only auto-increments INTEGER primary keys.
    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True
    )
    timestamp: Mapped[datetime] = mapped_column(index=True)
    service_id: Mapped[str] = mapped_column(ForeignKey("services.id"))
    level: Mapped[LogLevel] = mapped_column(enum_column(LogLevel, "level"), index=True)
    logger: Mapped[str] = mapped_column(String(128))
    message: Mapped[str] = mapped_column(Text)
    trace_id: Mapped[str | None] = mapped_column(String(32), index=True)
    span_id: Mapped[str | None] = mapped_column(String(16))
    deployment_id: Mapped[str | None] = mapped_column(ForeignKey("deployments.id"), index=True)
    version: Mapped[str | None] = mapped_column(String(32))
    host: Mapped[str] = mapped_column(String(128))
    # e.g. {"http.route": "/v1/payments", "http.status_code": 500, "duration_ms": 3012}
    attributes: Mapped[dict[str, Any]]

    service: Mapped[Service] = relationship()
    deployment: Mapped[Deployment | None] = relationship()
