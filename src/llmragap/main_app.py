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


def login_page() -> None:
    st.title("Log in")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")

    if submitted:
        try:
            response = httpx.post(
                f"{AUTH_API_URL}/auth/login",
                json={"username": username, "password": password},
                timeout=10,
            )
        except httpx.HTTPError:
            st.error(f"Could not reach the login API at {AUTH_API_URL}. Is it running?")
            st.stop()

        if response.status_code == 200:
            st.session_state.authenticated = True
            st.session_state.username = response.json()["username"]
            st.rerun()
        elif response.status_code == 401:
            st.error("Incorrect username or password.")
        else:
            st.error(f"Login failed: the API returned status {response.status_code}.")

    st.stop()


#if not st.session_state.get("authenticated"):
#    login_page()

with st.sidebar:
    #st.write(f"Signed in as **{st.session_state.username}**")
    st.write(f"Signed in as ")
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
