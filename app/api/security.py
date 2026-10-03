"""Who is calling the API.

``Authorization: Bearer opsrag_<id>_<secret>`` (issued by ``scripts/create_token.py``)
identifies the user. The role, and so everything the caller may read, comes from the
``users`` table; nothing in the request can change it.

``X-OpsRAG-User`` (a bare user name) is accepted only when SECURITY_ALLOW_USER_HEADER is
on, which the configuration refuses outside the local and test environments. A token
always takes precedence over the header.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.engine import Engine

from app.api.dependencies import SettingsDep, get_engine
from app.observability import telemetry
from app.security.auth import AuthenticationError, authenticate
from app.security.principal import Principal, PrincipalError, load_policy, load_principal

logger = logging.getLogger(__name__)
bearer = HTTPBearer(auto_error=False, description="An OpsRAG API token")
LOCAL_ENVIRONMENTS = {"local", "test"}


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status.HTTP_401_UNAUTHORIZED, detail, headers={"WWW-Authenticate": "Bearer"}
    )


def get_principal(
    settings: SettingsDep,
    engine: Annotated[Engine, Depends(get_engine)],
    credentials: Annotated[HTTPAuthorizationCredentials | None, Security(bearer)] = None,
    x_opsrag_user: Annotated[str | None, Header()] = None,
) -> Principal:
    if credentials is not None:
        try:
            user_id = authenticate(engine, credentials.credentials)
        except AuthenticationError as exc:
            raise _unauthorized("invalid, expired or revoked token") from exc
    elif (
        x_opsrag_user
        and settings.security.allow_user_header
        and settings.app.environment in LOCAL_ENVIRONMENTS
    ):
        user_id = x_opsrag_user
    else:
        raise _unauthorized("authentication required: send Authorization: Bearer <token>")
    try:
        principal = load_principal(engine, user_id, load_policy(settings.security.policy_file))
    except PrincipalError as exc:
        logger.warning("auth.inactive_user", extra={"user": user_id})
        raise HTTPException(status.HTTP_403_FORBIDDEN, "unknown or inactive user") from exc
    telemetry.set_user(principal.user_id)  # for the request's log event
    return principal


PrincipalDep = Annotated[Principal, Depends(get_principal)]
