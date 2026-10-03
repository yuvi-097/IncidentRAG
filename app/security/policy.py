"""The access policy: which kinds of data each role may read, and at which labels.

Loaded from JSON (``policy.json`` next to this module, or ``SECURITY_POLICY_FILE``) so
access rules are configuration, not code. Validation is strict and fails closed:
- a label or kind of data the code does not know is an error, not ignored;
- ``confidential`` can never be granted, to any role;
- a role missing from the policy gets nothing (deny by default).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.schemas.enums import AccessLevel, Resource, SourceTrust, ToolPermission

DEFAULT_POLICY_FILE = Path(__file__).with_name("policy.json")

# Tools that read one kind of data need that kind's permission.
RESOURCE_PERMISSION: dict[Resource, ToolPermission] = {
    Resource.DOCUMENTS: ToolPermission.DOCUMENTS_READ,
    Resource.RUNBOOKS: ToolPermission.RUNBOOKS_READ,
    Resource.INCIDENTS: ToolPermission.INCIDENTS_READ,
    Resource.DEPLOYMENTS: ToolPermission.DEPLOYMENTS_READ,
    Resource.CODE: ToolPermission.CODE_READ,
    Resource.LOGS: ToolPermission.LOGS_READ,
}
# Interfaces over several kinds of data; granted explicitly as capabilities.
CAPABILITIES = frozenset({ToolPermission.SQL_READ})


class PolicyError(ValueError):
    pass


class RolePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    description: str = ""
    data: dict[Resource, frozenset[AccessLevel]] = Field(default_factory=dict)
    capabilities: frozenset[ToolPermission] = frozenset()

    @field_validator("data")
    @classmethod
    def _never_confidential(
        cls, value: dict[Resource, frozenset[AccessLevel]]
    ) -> dict[Resource, frozenset[AccessLevel]]:
        for resource, labels in value.items():
            if AccessLevel.CONFIDENTIAL in labels:
                raise ValueError(
                    f"{resource.value}: 'confidential' can never be granted; confidential "
                    "sources are not indexed and never reach the assistant"
                )
        return value

    @field_validator("capabilities")
    @classmethod
    def _known_capabilities(cls, value: frozenset[ToolPermission]) -> frozenset[ToolPermission]:
        unknown = value - CAPABILITIES
        if unknown:
            raise ValueError(
                f"not capabilities: {sorted(p.value for p in unknown)}; permissions for one "
                "kind of data follow from 'data'"
            )
        return value


class AccessPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    description: str = ""
    roles: dict[str, RolePolicy]
    # Trust of each evidence kind (see SourceTrust); unknown kinds are user content.
    source_trust: dict[str, SourceTrust] = Field(default_factory=dict)

    def role(self, name: str) -> RolePolicy:
        """The role's grants; an unknown role gets none."""
        return self.roles.get(name, RolePolicy())

    def trust_of(self, kind: str) -> SourceTrust:
        return self.source_trust.get(kind, SourceTrust.USER_CONTENT)


@lru_cache(maxsize=8)
def load_policy(path: Path | None = None) -> AccessPolicy:
    source = path or DEFAULT_POLICY_FILE
    try:
        return AccessPolicy.model_validate(json.loads(source.read_text(encoding="utf-8")))
    except (OSError, ValueError) as exc:
        raise PolicyError(f"invalid access policy {source}: {exc}") from exc
