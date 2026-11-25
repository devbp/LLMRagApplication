"""Running cost totals of one app session, shared by the chat and test case pages.

The totals live in st.session_state, so every page adds to the same session. After a
change, save_session_cost() sends them to the auth API (session_costs table) for the
logged-in user.
"""

import os
import uuid

import httpx
import streamlit as st


def init_session_costs() -> None:
    """Create the session's cost totals once (no-op on later reruns and pages)."""
    defaults = {
        # Identifies this session's row in the session_costs table; a new login gets a new one.
        "session_id": lambda: str(uuid.uuid4()),
        # One entry per embedded file: {"file", "collection", "chunks", "tokens", "cost_usd"}
        "upload_costs": list,
        # Chat answers; kept when the chat is cleared.
        "chat_costs": lambda: {"answers": 0, "tokens": 0, "agent_usd": 0.0, "search_usd": 0.0},
        # Test case generation and evaluation runs (test_data_gui.py).
        "testgen_costs": lambda: {"runs": 0, "cases": 0, "tokens": 0, "usd": 0.0},
        "eval_costs": lambda: {"runs": 0, "cases": 0, "tokens": 0, "usd": 0.0},
    }
    for key, make in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = make()


def add_run_cost(kind: str, cases: int, tokens: int, usd: float) -> None:
    """Add one test generation ("testgen") or evaluation ("eval") run, then save the session."""
    totals = st.session_state[f"{kind}_costs"]
    totals["runs"] += 1
    totals["cases"] += cases
    totals["tokens"] += tokens
    totals["usd"] += usd
    save_session_cost()


def session_total_usd() -> float:
    ss = st.session_state
    return (
        ss.chat_costs["agent_usd"]
        + ss.chat_costs["search_usd"]
        + sum(u["cost_usd"] for u in ss.upload_costs)
        + ss.testgen_costs["usd"]
        + ss.eval_costs["usd"]
    )


def save_session_cost() -> None:
    """Send this session's running cost totals to the auth API (session_costs table).

    Only for logged-in users (the username comes from main_app.py); when a page runs on
    its own there is no user, so nothing is saved. A failure is kept in
    st.session_state.cost_save_error for show_cost_save_error(): the app keeps working.
    """
    username = st.session_state.get("username")
    if not username:
        return
    ss = st.session_state
    payload = {
        "username": username,
        "answers": ss.chat_costs["answers"],
        "tokens": ss.chat_costs["tokens"],
        "agent_usd": ss.chat_costs["agent_usd"],
        "search_usd": ss.chat_costs["search_usd"],
        "upload_files": len(ss.upload_costs),
        "upload_tokens": sum(u["tokens"] for u in ss.upload_costs),
        "upload_usd": sum(u["cost_usd"] for u in ss.upload_costs),
        "testgen_runs": ss.testgen_costs["runs"],
        "testgen_tokens": ss.testgen_costs["tokens"],
        "testgen_usd": ss.testgen_costs["usd"],
        "eval_runs": ss.eval_costs["runs"],
        "eval_tokens": ss.eval_costs["tokens"],
        "eval_usd": ss.eval_costs["usd"],
    }
    api_url = os.getenv("AUTH_API_URL", "http://127.0.0.1:8000").rstrip("/")
    try:
        response = httpx.put(f"{api_url}/sessions/{ss.session_id}/cost", json=payload, timeout=5)
        response.raise_for_status()
    except httpx.HTTPError as e:
        # Kept in the session so it survives a st.rerun() before it can be shown.
        ss.cost_save_error = f"Could not save the session cost: {e}"


def show_cost_save_error() -> None:
    """Show (once) a toast if the last save failed."""
    if error := st.session_state.pop("cost_save_error", None):
        st.toast(error, icon=":material/cloud_off:")


def money(usd: float) -> str:
    """$0.0123 for normal amounts; tiny ones keep their digits ($0.00000039, not $0.0000)."""
    if usd == 0 or usd >= 0.001:
        return f"${usd:.4f}"
    return "$" + f"{usd:.8f}".rstrip("0")
