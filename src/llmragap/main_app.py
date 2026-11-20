import os
from pathlib import Path

import httpx
import streamlit as st
from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "openai/gpt-4o-mini"

load_dotenv(PROJECT_ROOT / ".env")

AUTH_API_URL = os.getenv("AUTH_API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(page_title="LLM Q&A", page_icon=":material/chat:")


def call_auth_api(path: str, payload: dict) -> httpx.Response | None:
    """POST to the auth API; show an error and return None if it can't be reached."""
    try:
        return httpx.post(f"{AUTH_API_URL}{path}", json=payload, timeout=10)
    except httpx.HTTPError:
        st.error(f"Could not reach the login API at {AUTH_API_URL}. Is it running?")
        return None


def api_error_message(response: httpx.Response) -> str:
    """Turn the API's error response into a readable message."""
    try:
        detail = response.json().get("detail")
    except ValueError:
        detail = None
    if isinstance(detail, list):  # validation errors (422): one entry per invalid field
        return " ".join(
            f"{err['loc'][-1].capitalize()}: {err['msg'].removeprefix('Value error, ')}."
            for err in detail
        )
    return detail or f"The API returned status {response.status_code}."


def log_in(username: str) -> None:
    st.session_state.authenticated = True
    st.session_state.username = username
    st.rerun()


def login_form() -> None:
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in", width="stretch")

    if submitted:
        response = call_auth_api("/auth/login", {"username": username, "password": password})
        if response is None:
            return
        if response.status_code == 200:
            log_in(response.json()["username"])
        elif response.status_code == 401:
            st.error("Incorrect username or password.")
        else:
            st.error(f"Login failed: {api_error_message(response)}")


def create_account_form() -> None:
    with st.form("create_account_form"):
        username = st.text_input(
            "Username", help="3–50 characters: letters, digits, dot, dash or underscore."
        )
        password = st.text_input("Password", type="password", help="At least 8 characters.")
        confirm = st.text_input("Repeat password", type="password")
        submitted = st.form_submit_button("Create account", width="stretch")

    if submitted:
        if password != confirm:
            st.error("The passwords do not match.")
            return
        response = call_auth_api("/auth/users", {"username": username.strip(), "password": password})
        if response is None:
            return
        if response.status_code == 201:
            st.success(f"Account **{response.json()['username']}** created. Logging you in...")
            log_in(response.json()["username"])
        elif response.status_code == 409:
            st.error("That username is already taken. Choose another one.")
        else:
            st.error(f"Could not create the account. {api_error_message(response)}")


def login_page() -> None:
    st.title("Welcome")
    login_tab, create_tab = st.tabs([":material/login: Log in", ":material/person_add: Create account"])
    with login_tab:
        login_form()
    with create_tab:
        create_account_form()
    st.stop()


if not st.session_state.get("authenticated"):
    login_page()

with st.sidebar:
    st.write(f"Signed in as **{st.session_state.get('username', '')}**")
    if st.button("Log out"):
        st.session_state.clear()
        st.rerun()

st.title("LLM Q&A")
st.caption(f"Answers powered by {MODEL_NAME} through OpenRouter.")

with st.form("question_form"):
    question = st.text_input("Your question")
    submitted = st.form_submit_button("Get answer")

if submitted:
    if not question.strip():
        st.warning("Enter a question to get an answer.")
    elif not os.getenv("OPENROUTER_API_KEY"):
        st.error("Add OPENROUTER_API_KEY to the project .env file before asking a question.")
    else:
        with st.spinner("Generating an answer..."):
            model = ChatOpenRouter(model=MODEL_NAME, temperature=0)
            response = model.invoke(question.strip())

        st.subheader("Answer")
        st.write(response.content)
