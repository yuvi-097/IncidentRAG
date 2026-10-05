"""IncidentRAG frontend: run with ``streamlit run app.py`` from this directory.

The API must be running (``uvicorn app.main:app`` in the project root). Configure it with
``OPSRAG_API_URL`` and sign in with an API token (``OPSRAG_API_TOKEN`` or the sidebar);
against a local API with SECURITY_ALLOW_USER_HEADER on, a demo user can be picked instead
(``OPSRAG_DEMO_USER`` or the sidebar), to see how answers change with the role. As a public
demo (``OPSRAG_UI_PUBLIC_DEMO=true``) the sidebar offers only the demo users.
"""

from __future__ import annotations

import streamlit as st

from opsrag_ui.client import (
    DEFAULT_URL,
    DEMO_USERS,
    ApiError,
    connection,
    get_client,
    public_demo,
)
from opsrag_ui.session import forget, memo
from opsrag_ui.theme import inject_css

st.set_page_config(
    page_title="IncidentRAG · Incident copilot",
    page_icon=":material/emergency_home:",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()


def connection_form() -> None:
    """Change the API URL or who is signed in; nothing changes until Connect is pressed."""
    conn = connection()
    users = list(DEMO_USERS)
    with st.form("connection_form", border=False):
        url = st.text_input("API URL", value=conn["api_url"])
        user = st.selectbox(
            "Demo user",
            users,
            index=users.index(conn["demo_user"]),
            format_func=lambda u: f"{u} ({DEMO_USERS[u]})",
            help="Accepted only by a local API with SECURITY_ALLOW_USER_HEADER on.",
        )
        token = st.text_input(  # never echoed back to the page
            "API token",
            type="password",
            placeholder="Set; leave empty to keep it" if conn["api_token"] else "Optional",
            help="An API token takes precedence over the demo user.",
        )
        if st.form_submit_button("Connect", icon=":material/login:", width="stretch"):
            conn.update(
                api_url=url.strip() or DEFAULT_URL,
                demo_user=user,
                api_token=token.strip() or conn["api_token"],
            )
            forget()
            st.rerun()
    if conn["api_token"] and st.button("Sign out of the token", type="tertiary"):
        conn["api_token"] = ""
        forget()
        st.rerun()


def role_form() -> None:
    """Public demo: pick a demo user (and so a role); the API and credentials are fixed."""
    conn = connection()
    users = list(DEMO_USERS)
    with st.form("role_form", border=False):
        user = st.selectbox(
            "Sign in as",
            users,
            index=users.index(conn["demo_user"]),
            format_func=lambda u: f"{u} ({DEMO_USERS[u]})",
            help="Each role may read different data, so answers change with it (RBAC).",
        )
        if st.form_submit_button("Switch role", icon=":material/swap_horiz:", width="stretch"):
            conn["demo_user"] = user
            forget()
            st.rerun()


def sidebar() -> None:
    with st.sidebar:
        st.markdown("### IncidentRAG")
        st.caption("Incident response copilot for NovaCart (synthetic data)")
        if public_demo():
            role_form()
        else:
            with st.expander("Connection", icon=":material/key:"):
                connection_form()
        client = get_client()
        try:
            me = memo("me", client.me, ttl=300)
            st.markdown(f"**{me['user_id']}** · role `{me['role']}`")
            st.caption(f"{len(me['tools'])} tools available to this role")
        except ApiError as exc:
            st.warning(f"Not signed in: {exc.detail}", icon=":material/lock:")
        try:
            health = memo("health", client.health, ttl=15)
            database = health.get("checks", {}).get("database", {}).get("status", "unknown")
            ok = health.get("status") == "ok"
            st.caption(
                f"{':material/check_circle:' if ok else ':material/error:'} API {health['status']}"
                f" · database {database} · v{health.get('version', '?')}"
            )
        except ApiError as exc:
            st.error(exc.detail, icon=":material/cloud_off:")


sidebar()
pages = {
    "Investigate": [
        st.Page("pages/chat.py", title="Chat", icon=":material/forum:", default=True),
        st.Page("pages/incidents.py", title="Incident Explorer", icon=":material/emergency_home:"),
        st.Page("pages/evidence.py", title="Evidence Viewer", icon=":material/fact_check:"),
    ],
    "Quality": [
        st.Page("pages/evaluation.py", title="Evaluation", icon=":material/analytics:"),
        st.Page("pages/metrics.py", title="System Metrics", icon=":material/speed:"),
    ],
}
st.navigation(pages).run()
