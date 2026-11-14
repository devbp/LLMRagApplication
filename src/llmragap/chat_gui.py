"""Simple Streamlit chatbot for the RagAgent in RAG_Pdf.py.

The agent answers law questions (VVG / BGB search), music store data questions
(SQL) and general questions. Each answer shows which tools the agent used.

Run:
    uv run streamlit run src/llmragap/chat_gui.py
"""

import streamlit as st
from langchain_core.messages import AIMessage, ToolMessage

from llmragap.RAG_Pdf import DEFAULT_SELECTION_MODE, SEARCH_MODES, FilesEmbeddings, RagAgent, SqlDatabase


class ChatApp:
    """Streamlit chat page around a RagAgent.

    Streamlit reruns this whole script on every interaction, so the chat history
    lives in st.session_state and the slow-to-build objects (Chroma collections,
    agent) are cached with st.cache_resource.
    """

    TITLE = "Law & data chatbot"
    SUGGESTIONS = {
        ":blue[:material/gavel:] Limitation period": "What is the standard limitation period under the Civil Code?",
        ":green[:material/database:] Top artist": "Which artist has the most albums?",
        ":orange[:material/public:] General question": "What is the capital of Nepal?",
    }
    MAX_TOOL_OUTPUT_CHARS = 2000  # tool outputs longer than this are cut off in the UI

    def __init__(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []
        self.selection_mode = DEFAULT_SELECTION_MODE

    @property
    def messages(self) -> list[dict]:
        """Chat history for this browser session: {"role", "content", "steps"?} dicts."""
        return st.session_state.messages

    # --- Cached resources (shared across reruns and sessions) ---

    @staticmethod
    @st.cache_resource
    def get_files_embeddings() -> FilesEmbeddings:
        """Open the Chroma collections once."""
        return FilesEmbeddings()

    @staticmethod
    @st.cache_resource
    def get_agent(selection_mode: str) -> RagAgent:
        """Build one agent per search mode."""
        sql_db = SqlDatabase()
        sql_db.ensure_downloaded()
        return RagAgent(ChatApp.get_files_embeddings(), selection_mode, sql_db=sql_db)

    # --- Agent ---

    def run_agent(self) -> tuple[str, list[dict]]:
        """Send the whole conversation to the agent; return (answer, tool steps)."""
        history = [{"role": m["role"], "content": m["content"]} for m in self.messages]
        result_messages = self.get_agent(self.selection_mode).agent.invoke({"messages": history})["messages"]

        # Only look at messages from this turn (after the conversation we sent).
        new_messages = result_messages[len(history):]
        outputs = {m.tool_call_id: m.text for m in new_messages if isinstance(m, ToolMessage)}
        steps = [
            {"tool": call["name"], "input": call["args"], "output": outputs.get(call["id"], "")}
            for m in new_messages
            if isinstance(m, AIMessage)
            for call in m.tool_calls
        ]
        return result_messages[-1].text, steps

    # --- UI parts ---

    def render_header(self) -> None:
        st.set_page_config(page_title=self.TITLE, page_icon=":material/smart_toy:")
        st.title(f":material/smart_toy: {self.TITLE}")
        st.caption("Ask about German insurance or civil law, the music store database, or anything else.")

    def render_sidebar(self) -> None:
        with st.sidebar:
            self.selection_mode = st.segmented_control(
                "Document search mode",
                SEARCH_MODES,
                default=DEFAULT_SELECTION_MODE,
                required=True,
            )
            if st.button("Clear chat", icon=":material/delete:", width="stretch"):
                self.clear_chat()

    def clear_chat(self) -> None:
        st.session_state.messages = []
        st.session_state.pop("suggestion", None)  # so the old suggestion isn't asked again
        st.rerun()

    def show_steps(self, steps: list[dict]) -> None:
        """Show the agent's tool calls as a collapsed timeline."""
        if not steps:
            return
        with st.expander(f"Used {len(steps)} tool call(s)", type="compact"):
            for step in steps:
                with st.expander(step["tool"], type="step"):
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
        """Return the typed question, or a picked suggestion while the chat is empty."""
        prompt = st.chat_input("Ask a question", submit_mode="disable")
        if not self.messages and not prompt:
            picked = st.pills(
                "Try asking:", list(self.SUGGESTIONS), label_visibility="collapsed", key="suggestion"
            )
            if picked:
                prompt = self.SUGGESTIONS[picked]
        return prompt

    def handle_prompt(self, prompt: str) -> None:
        """Show the question, ask the agent, show and store the answer."""
        self.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking..."):
                    answer, steps = self.run_agent()
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


ChatApp().run()
