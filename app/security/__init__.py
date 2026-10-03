"""Security: access control, prompt-injection defences and secret handling.

- ``policy`` / ``principal``: which kinds of data a role may read, at which labels;
  enforced in the database queries of every tool, before anything reaches a model.
- ``injection``: detection of instruction-like text in questions and retrieved content.
- ``secrets``: detection and redaction of credentials (evidence, answers, logs).

Authentication (proving who the caller is) is not implemented yet.
"""

from app.security.policy import (
    DEFAULT_POLICY_FILE,
    AccessPolicy,
    PolicyError,
    RolePolicy,
    load_policy,
)
from app.security.principal import (
    SOURCE_RESOURCE,
    Principal,
    PrincipalError,
    load_principal,
    principal_for_role,
    resource_for_document,
)

__all__ = [
    "DEFAULT_POLICY_FILE",
    "SOURCE_RESOURCE",
    "AccessPolicy",
    "PolicyError",
    "Principal",
    "PrincipalError",
    "RolePolicy",
    "load_policy",
    "load_principal",
    "principal_for_role",
    "resource_for_document",
]
