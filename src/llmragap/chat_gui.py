"""Simple Streamlit chatbot for the RagAgent in RAG_Pdf.py.

The agent answers law questions (VVG / BGB search), music store data questions
(SQL) and general questions. Each answer shows which tools the agent used.
The sidebar also lets you upload PDFs and add them to a Chroma collection.

Run:
    uv run streamlit run src/llmragap/chat_gui.py
"""

import os

import streamlit as st
from langchain_core.messages import AIMessage, ToolMessage

from llmragap.RAG_Pdf import (
    CIVIL_CODE_COLLECTION,
    DEFAULT_SELECTION_MODE,
    INSURANCE_COLLECTION,
    SEARCH_MODES,
    FilesEmbeddings,
    RagAgent,
    SqlDatabase,
)


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
    # Collections you can upload into (label shown in the UI -> Chroma collection name).
    COLLECTIONS = {
        "Insurance law (VVG)": INSURANCE_COLLECTION,
        "Civil code (BGB)": CIVIL_CODE_COLLECTION,
    }

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
            st.divider()
            self.render_upload_form()

    # --- Adding files to the vector database ---

    def render_upload_form(self) -> None:
        """Sidebar form: upload PDFs and add them to the chosen collection."""
        st.subheader("Add documents")
        with st.form("upload_form", clear_on_submit=True, border=False):
            uploads = st.file_uploader("PDF files", type="pdf", accept_multiple_files=True)
            collection_label = st.selectbox("Collection", list(self.COLLECTIONS))
            replace = st.checkbox(
                "Replace if already stored",
                help="Off: files already in the collection are skipped. "
                "On: their chunks are re-embedded and updated in place.",
            )
            submitted = st.form_submit_button("Add to database", icon=":material/upload:", width="stretch")

        if submitted:
            if not uploads:
                st.warning("Choose at least one PDF first.")
            else:
                self.add_files(uploads, self.COLLECTIONS[collection_label], replace)

    def save_upload(self, upload) -> str:
        """Save an uploaded file into the data folder and return its full path."""
        files = self.get_files_embeddings()
        os.makedirs(files.data_dir, exist_ok=True)
        # basename() keeps a crafted file name from writing outside the data folder.
        file_path = os.path.join(files.data_dir, os.path.basename(upload.name))
        with open(file_path, "wb") as f:
            f.write(upload.getbuffer())
        return file_path

    def add_files(self, uploads, collection_name: str, replace: bool) -> None:
        """Embed each uploaded PDF into the collection and report what happened."""
        files = self.get_files_embeddings()
        for upload in uploads:
            name = os.path.basename(upload.name)
            try:
                with st.spinner(f"Embedding {name}..."):
                    file_path = self.save_upload(upload)
                    if replace:
                        count = files.update_document(collection_name, file_path)
                        st.success(f"{name}: {count} chunks stored in '{collection_name}'.")
                    else:
                        count = files.add_document(collection_name, file_path)
                        if count:
                            st.success(f"{name}: added {count} chunks to '{collection_name}'.")
                        else:
                            st.info(f"{name} is already in '{collection_name}', skipped.")
            except Exception as e:
                st.error(f"{name}: {e}")

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
