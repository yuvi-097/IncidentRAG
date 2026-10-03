"""Who is asking, and what they may read.

A ``Principal`` carries the caller's grants from the access policy: for each kind of
data (``Resource``), the set of sensitivity labels they may read. Everything else
derives from that:
- tool permissions: a tool that reads one kind of data needs a grant for it;
- row filters: tools and the chunk store only select rows whose (kind, label) pair
  is granted, *in the database query*, so nothing else is ever loaded, let alone
  passed to a model.

The role comes from the ``users`` table, never from the request. Authentication
(proving who the caller is) is not part of this module.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator
from sqlalchemy import select
from sqlalchemy.engine import Engine

from app.database.models import Role, User
from app.schemas.enums import (
    UNLABELLED_LEVEL,
    AccessLevel,
    DocumentType,
    Resource,
    SourceType,
    ToolPermission,
)
from app.security.policy import RESOURCE_PERMISSION, AccessPolicy, load_policy

# The kind of data each stored chunk belongs to.
SOURCE_RESOURCE: dict[SourceType, Resource] = {
    SourceType.DOCUMENTATION: Resource.DOCUMENTS,
    SourceType.RUNBOOK: Resource.RUNBOOKS,
    SourceType.POSTMORTEM: Resource.INCIDENTS,
    SourceType.INCIDENT: Resource.INCIDENTS,
    SourceType.DEPLOYMENT: Resource.DEPLOYMENTS,
    SourceType.CODE: Resource.CODE,
    SourceType.PULL_REQUEST: Resource.CODE,
}


def resource_for_document(doc_type: DocumentType | str) -> Resource:
    """Rows of the ``documents`` table: runbooks and postmortems have their own grants."""
    value = DocumentType(doc_type)
    if value is DocumentType.RUNBOOK:
        return Resource.RUNBOOKS
    if value is DocumentType.POSTMORTEM:
        return Resource.INCIDENTS
    return Resource.DOCUMENTS


class PrincipalError(PermissionError):
    pass


class Principal(BaseModel):
    model_config = ConfigDict(frozen=True)

    user_id: str = Field(min_length=1)
    role: str = Field(min_length=1)
    # Read-only (a mapping proxy): a principal cannot be widened after it is built.
    grants: Mapping[Resource, frozenset[AccessLevel]] = Field(
        default_factory=lambda: MappingProxyType({})
    )
    capabilities: frozenset[ToolPermission] = frozenset()

    @field_validator("grants")
    @classmethod
    def _never_confidential(
        cls, value: Mapping[Resource, frozenset[AccessLevel]]
    ) -> Mapping[Resource, frozenset[AccessLevel]]:
        # Also checked when the policy loads; repeated so no code path can build one.
        if any(AccessLevel.CONFIDENTIAL in labels for labels in value.values()):
            raise ValueError("confidential data can never be granted")
        return MappingProxyType(
            {resource: frozenset(labels) for resource, labels in value.items() if labels}
        )

    @field_serializer("grants")
    def _serialize_grants(
        self, value: Mapping[Resource, frozenset[AccessLevel]]
    ) -> dict[str, list[str]]:
        return {r.value: sorted(level.value for level in labels) for r, labels in value.items()}

    @field_validator("capabilities", mode="before")
    @classmethod
    def _as_frozenset(cls, value: object) -> object:
        return frozenset(value) if isinstance(value, list | set | tuple) else value

    @property
    def permissions(self) -> frozenset[ToolPermission]:
        derived = {RESOURCE_PERMISSION[r] for r in self.grants if r in RESOURCE_PERMISSION}
        return frozenset(derived) | self.capabilities

    def can(self, permission: ToolPermission) -> bool:
        return permission in self.permissions

    def labels(self, resource: Resource) -> frozenset[AccessLevel]:
        return self.grants.get(resource, frozenset())

    def visible_levels(self, resource: Resource) -> list[AccessLevel]:
        """Sorted, for ``IN (...)`` clauses."""
        return sorted(self.labels(resource))

    def may_read(self, resource: Resource, level: AccessLevel) -> bool:
        return level is not AccessLevel.CONFIDENTIAL and level in self.labels(resource)

    def may_read_unlabelled(self, resource: Resource) -> bool:
        """Records without their own label (deployments, logs, the catalog)."""
        return self.may_read(resource, UNLABELLED_LEVEL)

    def chunk_access(self) -> frozenset[tuple[SourceType, AccessLevel]]:
        """The (source type, label) pairs of stored chunks this caller may retrieve."""
        return frozenset(
            (source_type, level)
            for source_type, resource in SOURCE_RESOURCE.items()
            for level in self.labels(resource)
        )


def principal_for_role(user_id: str, role: str, policy: AccessPolicy | None = None) -> Principal:
    policy = policy or load_policy()
    grants = policy.role(role)
    return Principal(
        user_id=user_id,
        role=role,
        grants=dict(grants.data),
        capabilities=grants.capabilities,
    )


def load_principal(engine: Engine, user_id: str, policy: AccessPolicy | None = None) -> Principal:
    """The principal for an active user; the role is read from ``users``/``roles``."""
    query = (
        select(User.id, User.is_active, Role.id.label("role"))
        .join(Role, Role.id == User.role_id)
        .where(User.id == user_id)
    )
    with engine.connect() as connection:
        row = connection.execute(query).first()
    if row is None or not row.is_active:
        raise PrincipalError(f"unknown or inactive user: {user_id!r}")
    return principal_for_role(row.id, row.role, policy)
