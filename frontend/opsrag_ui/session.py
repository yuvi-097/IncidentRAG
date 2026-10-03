"""Session helpers: a small time-limited memo, and the chat history shared by pages."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, TypeVar

import streamlit as st

T = TypeVar("T")


def identity() -> tuple[str, str, str]:
    """What the cached values depend on: the API and who is signed in."""
    conn = st.session_state.get("connection", {})
    return (
        str(conn.get("api_url", "")),
        str(conn.get("api_token", "")),
        str(conn.get("demo_user", "")),
    )


def memo(key: str, load: Callable[[], T], ttl: float = 60.0) -> T:
    """``load()`` once per ``ttl`` seconds per identity, kept in the session."""
    store: dict[str, tuple[float, tuple[str, str, str], Any]] = st.session_state.setdefault(
        "_memo", {}
    )
    cached = store.get(key)
    now = time.monotonic()
    if cached and cached[1] == identity() and now - cached[0] < ttl:
        return cached[2]  # type: ignore[no-any-return]
    value = load()
    store[key] = (now, identity(), value)
    return value


def forget(key: str | None = None) -> None:
    store = st.session_state.get("_memo", {})
    if key is None:
        store.clear()
    else:
        store.pop(key, None)


def history() -> list[dict[str, Any]]:
    """Questions asked in this session, oldest first: {question, response, user, role}."""
    return st.session_state.setdefault("history", [])  # type: ignore[no-any-return]


__all__ = ["forget", "history", "identity", "memo"]
