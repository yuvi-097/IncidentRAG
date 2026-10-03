"""The access policy and principals: grants per kind of data, labels as compartments."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy import update

from app.database.models import User
from app.schemas.enums import AccessLevel, Resource, SourceType, ToolPermission, most_restrictive
from app.security import (
    PolicyError,
    Principal,
    PrincipalError,
    load_policy,
    load_principal,
    principal_for_role,
)
from tests.tools.conftest import ToolEnv

A = AccessLevel


def test_the_packaged_policy_covers_every_role(tool_env: ToolEnv) -> None:
    policy = load_policy()
    assert (
        set(policy.roles)
        == {r.id for r in tool_env.dataset.roles}
        == {
            "developer",
            "sre",
            "manager",
            "admin",
        }
    )


@pytest.mark.parametrize(
    ("role", "permissions"),
    [
        ("developer", {"documents:read", "code:read", "incidents:read"}),
        (
            "sre",
            {
                "documents:read",
                "runbooks:read",
                "incidents:read",
                "deployments:read",
                "logs:read",
                "sql:read",
            },
        ),
        ("manager", {"documents:read", "incidents:read", "sql:read"}),
        ("admin", {p.value for p in ToolPermission}),
    ],
)
def test_role_permissions_follow_the_spec(role: str, permissions: set[str]) -> None:
    assert {p.value for p in principal_for_role("u", role).permissions} == permissions


def test_role_labels_follow_the_spec() -> None:
    developer, sre, manager, admin = (
        principal_for_role("u", r) for r in ("developer", "sre", "manager", "admin")
    )
    # developer: non-sensitive incidents only
    assert developer.labels(Resource.INCIDENTS) == {A.PUBLIC, A.ENGINEERING}
    # SRE and manager: all incidents; manager: non-sensitive documents plus reports
    assert A.SRE in sre.labels(Resource.INCIDENTS) and A.SRE in manager.labels(Resource.INCIDENTS)
    assert manager.labels(Resource.DOCUMENTS) == {A.PUBLIC, A.ENGINEERING, A.MANAGER}
    assert not sre.may_read(Resource.DOCUMENTS, A.MANAGER)
    # admin: everything the assistant may use, never confidential
    assert all(A.ADMIN in admin.labels(r) or r is Resource.CATALOG for r in Resource)
    for principal in (developer, sre, manager, admin):
        assert not any(principal.may_read(r, A.CONFIDENTIAL) for r in Resource)


def test_unknown_roles_get_nothing() -> None:
    nobody = principal_for_role("u", "no-such-role")
    assert nobody.grants == {} and nobody.permissions == frozenset()
    assert nobody.chunk_access() == frozenset()


def test_chunk_access_maps_source_types_to_grants() -> None:
    manager = principal_for_role("u", "manager")
    access = manager.chunk_access()
    assert (SourceType.POSTMORTEM, A.SRE) in access  # postmortems follow incidents
    assert (SourceType.DOCUMENTATION, A.SRE) not in access
    assert not any(st in {SourceType.CODE, SourceType.RUNBOOK} for st, _ in access)


def test_a_custom_policy_file(tmp_path: Path) -> None:
    path = tmp_path / "policy.json"
    path.write_text(
        json.dumps(
            {"roles": {"auditor": {"data": {"incidents": ["engineering"]}, "capabilities": []}}}
        ),
        encoding="utf-8",
    )
    auditor = principal_for_role("u", "auditor", load_policy(path))
    assert auditor.permissions == {ToolPermission.INCIDENTS_READ}


@pytest.mark.parametrize(
    "roles",
    [
        {"x": {"data": {"documents": ["confidential"]}}},  # never grantable
        {"x": {"data": {"documents": ["top-secret"]}}},  # unknown label
        {"x": {"data": {"secrets": ["engineering"]}}},  # unknown kind of data
        {"x": {"capabilities": ["logs:read"]}},  # data permissions come from grants
        {"x": {"data": {}, "role": "admin"}},  # unknown field
    ],
)
def test_invalid_policies_fail_closed(tmp_path: Path, roles: dict[str, object]) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"roles": roles}), encoding="utf-8")
    with pytest.raises(PolicyError):
        load_policy(path)


def test_principals_are_loaded_from_users_and_roles(tool_env: ToolEnv) -> None:
    user = next(u for u in tool_env.dataset.users if u.role_id == "sre" and u.is_active)
    principal = load_principal(tool_env.engine, user.id)
    assert (principal.user_id, principal.role) == (user.id, "sre")
    assert principal.can(ToolPermission.LOGS_READ) and not principal.can(ToolPermission.CODE_READ)
    assert principal.visible_levels(Resource.INCIDENTS) == [A.ENGINEERING, A.PUBLIC, A.SRE]


def test_unknown_and_inactive_users_have_no_principal(tool_env: ToolEnv) -> None:
    with pytest.raises(PrincipalError, match="unknown or inactive"):
        load_principal(tool_env.engine, "nobody")
    inactive = next(u for u in tool_env.dataset.users if not u.is_active)
    assert inactive.id == "contractor.docs"
    with pytest.raises(PrincipalError):
        load_principal(tool_env.engine, inactive.id)
    user = next(u for u in tool_env.dataset.users if u.is_active)
    with tool_env.engine.begin() as connection:
        connection.execute(update(User).where(User.id == user.id).values(is_active=False))
    try:
        with pytest.raises(PrincipalError):
            load_principal(tool_env.engine, user.id)
    finally:
        with tool_env.engine.begin() as connection:
            connection.execute(update(User).where(User.id == user.id).values(is_active=True))


def test_principals_are_immutable_and_never_confidential() -> None:
    principal = principal_for_role("u", "developer")
    with pytest.raises(ValueError):
        principal.role = "admin"  # type: ignore[misc]
    with pytest.raises(ValueError):
        principal.grants = {}  # type: ignore[misc]
    with pytest.raises(ValueError, match="confidential"):
        Principal(user_id="x", role="r", grants={Resource.DOCUMENTS: frozenset({A.CONFIDENTIAL})})
    assert not principal.may_read(Resource.INCIDENTS, A.SRE)
    assert principal.may_read(Resource.INCIDENTS, A.PUBLIC)


def test_most_restrictive_label_of_composite_content() -> None:
    assert most_restrictive([A.PUBLIC, A.ENGINEERING]) is A.ENGINEERING
    assert most_restrictive([A.ENGINEERING, A.SRE]) is A.SRE
    assert most_restrictive([A.SRE, A.MANAGER]) is A.ADMIN  # incomparable: needs both
    assert most_restrictive([A.ADMIN, A.CONFIDENTIAL]) is A.CONFIDENTIAL
    assert most_restrictive([]) is A.PUBLIC
