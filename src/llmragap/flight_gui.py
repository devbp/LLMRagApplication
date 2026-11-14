"""Simple Streamlit chat GUI for the flight assistant in supabase_flights.py.

Ask about flights, airports and bookings, or book a flight: the assistant lists
available flights, asks which one you want, shows a summary and books only after
you confirm in your next message. Each answer shows which tools were used.

Run:
    uv run streamlit run src/llmragap/flight_gui.py
"""

import streamlit as st
from langchain_core.messages import AIMessage, ToolMessage

from llmragap.supabase_flights import FlightDatabase, FlightToolChain


class FlightApp:
    """Streamlit chat page around a FlightToolChain.

    Streamlit reruns this script on every interaction. The database connection is
    shared (st.cache_resource); the tool chain is stored per browser session in
    st.session_state, because it holds that user's conversation and pending bookings.
    """

    TITLE = "Flight assistant"
    SUGGESTIONS = {
        ":blue[:material/flight:] Flight details": "What are the details of flight LH410?",
        ":green[:material/search:] Flights from Munich": "Which flights leave from Munich?",
        ":orange[:material/airplane_ticket:] Book a flight": "I want to book a flight from Munich.",
        ":violet[:material/confirmation_number:] Check a booking": "Show me booking MUC99A.",
    }
    MAX_TOOL_OUTPUT_CHARS = 2000

    def __init__(self):
        if "flight_messages" not in st.session_state:
            st.session_state.flight_messages = []  # what is shown in the chat
        if "flight_chain" not in st.session_state:
            st.session_state.flight_chain = FlightToolChain(self.get_flight_db())

    @property
    def messages(self) -> list[dict]:
        """Chat shown on the page: {"role", "content", "steps"?} dicts."""
        return st.session_state.flight_messages

    @property
    def chain(self) -> FlightToolChain:
        return st.session_state.flight_chain

    @staticmethod
    @st.cache_resource
    def get_flight_db() -> FlightDatabase:
        """One Supabase connection shared by all sessions."""
        return FlightDatabase()

    # --- Assistant ---

    def run_chain(self, prompt: str) -> tuple[str, list[dict]]:
        """Send the message to the tool chain; return (answer, tool steps of this turn)."""
        start = len(self.chain.messages)
        answer = self.chain.chat(prompt)
        new_messages = self.chain.messages[start:]
        outputs = {m.tool_call_id: m.text for m in new_messages if isinstance(m, ToolMessage)}
        steps = [
            {"tool": call["name"], "input": call["args"], "output": outputs.get(call["id"], "")}
            for m in new_messages
            if isinstance(m, AIMessage)
            for call in m.tool_calls
        ]
        return answer, steps

    # --- UI parts ---

    def render_header(self) -> None:
        st.set_page_config(page_title=self.TITLE, page_icon=":material/flight:")
        st.title(f":material/flight: {self.TITLE}")
        st.caption("Ask about flights, airports and bookings, or book a flight. Times are in UTC.")

    def render_sidebar(self) -> None:
        with st.sidebar:
            st.subheader("Conversation")
            if st.button("New conversation", icon=":material/refresh:", width="stretch"):
                self.new_conversation()
            st.caption(
                "Bookings need your confirmation: the assistant shows a summary first "
                "and books only after you reply **yes**."
            )
            pending = len(self.chain.pending_bookings)
            if pending:
                st.info(f"{pending} booking(s) waiting for your confirmation.", icon=":material/pending:")

    def new_conversation(self) -> None:
        self.chain.reset()
        st.session_state.flight_messages = []
        st.session_state.pop("flight_suggestion", None)  # so the old suggestion isn't sent again
        st.rerun()

    def show_steps(self, steps: list[dict]) -> None:
        """Show the tool calls as a collapsed timeline."""
        if not steps:
            return
        with st.expander(f"Used {len(steps)} tool call(s)", type="compact"):
            for step in steps:
                icon = ":material/check_circle:" if step["tool"] == "confirm_booking" else None
                with st.expander(step["tool"], type="step", icon=icon):
                    st.json(step["input"])
                    output = step["output"]
                    if len(output) > self.MAX_TOOL_OUTPUT_CHARS:
                        output = output[: self.MAX_TOOL_OUTPUT_CHARS] + "\n..."
                    st.text(output)

    def render_history(self) -> None:
        for msg in self.messages:
            with st.chat_message(msg["role"]):
                if msg["role"] == "assistant":
                    self.show_steps(msg.get("steps", []))
                st.markdown(msg["content"])

    def get_prompt(self) -> str | None:
        """Return the typed message, or a picked suggestion while the chat is empty."""
        prompt = st.chat_input("Ask about a flight or say what you want to book", submit_mode="disable")
        if not self.messages and not prompt:
            picked = st.pills(
                "Try asking:", list(self.SUGGESTIONS), label_visibility="collapsed", key="flight_suggestion"
            )
            if picked:
                prompt = self.SUGGESTIONS[picked]
        return prompt

    def handle_prompt(self, prompt: str) -> None:
        """Show the message, run the tool chain, show and store the answer."""
        self.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking..."):
                    answer, steps = self.run_chain(prompt)
            except Exception as e:
                answer, steps = f":red[Error: {e}]", []
            self.show_steps(steps)
            st.markdown(answer)

        self.messages.append({"role": "assistant", "content": answer, "steps": steps})
        st.rerun()

    def run(self) -> None:
        """Draw the page; called on every Streamlit rerun."""
        self.render_header()
        self.render_sidebar()
        self.render_history()
        if prompt := self.get_prompt():
            self.handle_prompt(prompt)


FlightApp().run()
