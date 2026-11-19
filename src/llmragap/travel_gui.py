"""Simple Streamlit chat GUI for the travel assistant in mcp_exercise.py.

Ask about weather, flights and hotel prices. The assistant uses the Xweather,
AirTreks and trivago MCP servers, and each answer shows which tools it used.

Run:
    uv run streamlit run src/llmragap/travel_gui.py
"""

import asyncio
import json
import uuid

import streamlit as st

from llmragap.mcp_exercise import TravelAgent


class TravelApp:
    """Streamlit chat page around a TravelAgent.

    Streamlit reruns this script on every interaction, so the agent (which holds
    this user's conversation memory) is kept per browser session in st.session_state.
    """

    TITLE = "Travel assistant"
    SUGGESTIONS = {
        ":blue[:material/partly_cloudy_day:] Weather": "What's the weather in Munich for the next 3 days?",
        ":green[:material/flight:] Flights": "Find the cheapest flight from Munich to Frankfurt next Friday.",
        ":orange[:material/hotel:] Hotels": "Find a good hotel in Frankfurt for 1 night next Friday.",
        ":violet[:material/luggage:] Plan a trip": (
            "Which of the next 5 days is warmest in Munich? Find a flight from Frankfurt "
            "to Munich on that day and a hotel in Munich for that night."
        ),
    }
    MAX_TOOL_OUTPUT_CHARS = 2000

    def __init__(self):
        if "travel_messages" not in st.session_state:
            st.session_state.travel_messages = []  # what is shown in the chat
        if "travel_agent" not in st.session_state:
            st.session_state.travel_agent = TravelAgent(thread_id=str(uuid.uuid4()))

    @property
    def messages(self) -> list[dict]:
        """Chat shown on the page: {"role", "content", "steps"?} dicts."""
        return st.session_state.travel_messages

    @property
    def agent(self) -> TravelAgent:
        return st.session_state.travel_agent

    def reset(self):
        st.session_state.travel_messages = []
        st.session_state.travel_agent = TravelAgent(thread_id=str(uuid.uuid4()))

    # ---------- rendering ----------

    def render_steps(self, steps: list[dict]):
        """Show the tool calls behind an answer as a collapsed step timeline."""
        if not steps:
            return
        label = f"Used {len(steps)} tool{'s' if len(steps) != 1 else ''}"
        with st.expander(label, type="compact", icon=":material/build:"):
            for step in steps:
                with st.expander(step["name"], type="step"):
                    st.caption("Input")
                    st.code(json.dumps(step["args"], indent=2, ensure_ascii=False), language="json")
                    st.caption("Output")
                    output = str(step["output"])
                    if len(output) > self.MAX_TOOL_OUTPUT_CHARS:
                        output = output[: self.MAX_TOOL_OUTPUT_CHARS] + " …"
                    st.text(output)

    def render_history(self):
        for msg in self.messages:
            with st.chat_message(msg["role"]):
                if msg["role"] == "assistant":
                    self.render_steps(msg.get("steps", []))
                st.markdown(msg["content"])

    def render_sidebar(self):
        with st.sidebar:
            st.header("About")
            st.write("Weather: Xweather · Flights: AirTreks · Hotels: trivago")
            st.caption("Prices are live and can change. trivago compares prices; it can't book.")
            if self.agent.tools:
                st.caption(f"{len(self.agent.tools)} tools loaded")
            for name, error in self.agent.unavailable.items():
                st.warning(f"{name.capitalize()} unavailable: {error}", icon=":material/cloud_off:")
            if st.button("New chat", icon=":material/refresh:", width="stretch"):
                self.reset()
                st.rerun()

    def pick_suggestion(self) -> str | None:
        """Suggestion chips, shown only before the first message."""
        if self.messages:
            return None
        selected = st.pills(
            "Try asking", list(self.SUGGESTIONS), label_visibility="collapsed"
        )
        return self.SUGGESTIONS[selected] if selected else None

    # ---------- chat ----------

    def answer(self, prompt: str):
        self.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.status(":shimmer[Searching weather, flights and hotels]", type="compact") as status:
                try:
                    # Streamlit runs scripts synchronously; run the async agent here.
                    answer, steps = asyncio.run(self.agent.chat(prompt))
                    status.update(label="Done", state="complete")
                except Exception as exc:  # e.g. MCP server or network error
                    answer, steps = f"Sorry, something went wrong: {exc}", []
                    status.update(label="Failed", state="error")
            self.render_steps(steps)
            st.markdown(answer)

        self.messages.append({"role": "assistant", "content": answer, "steps": steps})

    def run(self):
        st.set_page_config(page_title=self.TITLE, page_icon=":material/travel_explore:")
        st.title(f":material/travel_explore: {self.TITLE}")
        st.caption("Ask about weather, flights and hotel prices.")

        self.render_sidebar()
        self.render_history()

        suggestion = self.pick_suggestion()
        prompt = st.chat_input("Ask about weather, flights or hotels", submit_mode="disable")
        if prompt or suggestion:
            self.answer(prompt or suggestion)
            st.rerun()  # hide suggestion chips and refresh the sidebar


TravelApp().run()
