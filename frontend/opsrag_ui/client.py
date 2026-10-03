"""HTTP client for the OpsRAG API. The frontend talks to the backend only through this.

Authentication, in order of precedence:
- an API token (``Authorization: Bearer ...``), from the sidebar or ``OPSRAG_API_TOKEN``;
- a demo user name (``X-OpsRAG-User``), which the API accepts only in the local and test
  environments with SECURITY_ALLOW_USER_HEADER on. It exists so a local demo can switch
  roles and show that answers depend on them.

The client never logs or stores the token beyond the Streamlit session.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx
import streamlit as st

DEFAULT_URL = os.environ.get("OPSRAG_API_URL", "http://127.0.0.1:8000")
DEMO_USERS = {
    "arjun.mehta": "developer",
    "alex.rivera": "sre",
    "sarah.miller": "manager",
    "noor.hassan": "admin",
}


class ApiError(Exception):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"{status}: {detail}")
        self.status = status
        self.detail = detail


@dataclass
class ApiClient:
    base_url: str = DEFAULT_URL
    token: str | None = None
    user: str | None = None
    timeout: float = 180.0  # the first question loads the models

    def _headers(self) -> dict[str, str]:
        if self.token:
            return {"Authorization": f"Bearer {self.token}"}
        if self.user:
            return {"X-OpsRAG-User": self.user}
        return {}

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = httpx.request(
                method,
                f"{self.base_url.rstrip('/')}/api{path}",
                headers=self._headers(),
                timeout=self.timeout,
                **kwargs,
            )
        except httpx.HTTPError as exc:
            raise ApiError(0, f"the API at {self.base_url} is not reachable") from exc
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise ApiError(response.status_code, str(detail))
        return response.json()

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def me(self) -> dict[str, Any]:
        return self._request("GET", "/me")

    def ask(self, question: str) -> dict[str, Any]:
        return self._request("POST", "/agent/ask", json={"question": question})

    def services(self) -> list[dict[str, Any]]:
        return self._request("GET", "/services")

    def incidents(self, **params: Any) -> dict[str, Any]:
        query = {k: v for k, v in params.items() if v not in (None, "", [])}
        return self._request("GET", "/incidents", params=query)

    def incident(self, incident_id: str) -> dict[str, Any]:
        return self._request("GET", f"/incidents/{incident_id}")

    def trace(self, incident_id: str) -> dict[str, Any]:
        return self._request("GET", f"/incidents/{incident_id}/trace")

    def evaluation(self) -> dict[str, Any]:
        return self._request("GET", "/evaluation")

    def metrics(self) -> dict[str, Any]:
        return self._request("GET", "/metrics")


def connection() -> dict[str, str]:
    """Where the session connects and as whom: ``api_url``, ``demo_user``, ``api_token``.

    Kept apart from the sidebar's widgets and changed only by its Connect button: widget
    state can be reset by Streamlit (a run interrupted on a cold server hands back
    default values), and that must never point the session at another URL or user.
    """
    default_user = os.environ.get("OPSRAG_DEMO_USER", "alex.rivera")
    return st.session_state.setdefault(  # type: ignore[no-any-return]
        "connection",
        {
            "api_url": DEFAULT_URL,
            "demo_user": default_user if default_user in DEMO_USERS else "alex.rivera",
            "api_token": "",  # typed in the sidebar; OPSRAG_API_TOKEN stays server-side
        },
    )


def get_client() -> ApiClient:
    """The session's client (tests put a fake one in ``st.session_state.client``)."""
    injected = st.session_state.get("client")
    if injected is not None:
        return injected  # type: ignore[no-any-return]
    conn = connection()
    return ApiClient(
        base_url=conn["api_url"] or DEFAULT_URL,
        token=conn["api_token"] or os.environ.get("OPSRAG_API_TOKEN") or None,
        user=conn["demo_user"],
    )


__all__ = ["DEFAULT_URL", "DEMO_USERS", "ApiClient", "ApiError", "connection", "get_client"]
