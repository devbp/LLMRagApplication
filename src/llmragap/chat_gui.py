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
        if "upload_costs" not in st.session_state:
            # One entry per embedded file: {"file", "collection", "chunks", "tokens", "cost_usd"}
            st.session_state.upload_costs = []
        self.selection_mode = DEFAULT_SELECTION_MODE

    @property
    def messages(self) -> list[dict]:
        """Chat history for this browser session: {"role", "content", "steps"?} dicts."""
        return st.session_state.messages

    # --- Cached resources (shared across reruns and sessions) ---

    @staticmethod
    @st.cache_resource
    def get_vectors_embeddings() -> FilesEmbeddings:
        """Open the Chroma vector collections once."""
        return FilesEmbeddings()

    @staticmethod
    @st.cache_resource
    def get_agent(selection_mode: str) -> RagAgent:
        """Build one agent per search mode."""
        sql_db = SqlDatabase()
        sql_db.ensure_downloaded()
        return RagAgent(ChatApp.get_vectors_embeddings(), selection_mode, sql_db=sql_db)

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
            self.render_embedding_cost()
            st.divider()
            self.render_reset_form()

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
        vectors = self.get_vectors_embeddings()
        os.makedirs(vectors.data_dir, exist_ok=True)
        # basename() keeps a crafted file name from writing outside the data folder.
        file_path = os.path.join(vectors.data_dir, os.path.basename(upload.name))
        with open(file_path, "wb") as f:
            f.write(upload.getbuffer())
        return file_path

    @staticmethod
    def format_cost(usage: dict) -> str:
        return f"{usage['tokens']:,} tokens, ${usage['cost_usd']:.4f}"

    def add_files(self, uploads, collection_name: str, replace: bool) -> None:
        """Embed each uploaded PDF into the collection and report what happened."""
        vectors = self.get_vectors_embeddings()
        for upload in uploads:
            name = os.path.basename(upload.name)
            try:
                with st.spinner(f"Embedding {name}..."):
                    file_path = self.save_upload(upload)
                    if replace:
                        count = vectors.update_document(collection_name, file_path)
                    else:
                        count = vectors.add_document(collection_name, file_path)
                if not count:
                    st.info(f"{name} is already in '{collection_name}', skipped (no cost).")
                    continue
                usage = vectors.last_upload_usage
                action = "stored" if replace else "added"
                st.success(f"{name}: {count} chunks {action} in '{collection_name}' ({self.format_cost(usage)}).")
                st.session_state.upload_costs.append(
                    {"file": name, "collection": collection_name, "chunks": count, **usage}
                )
            except Exception as e:
                st.error(f"{name}: {e}")

    def render_embedding_cost(self) -> None:
        """Sidebar: what embedding the uploaded files cost, as billed by OpenRouter."""
        uploads = st.session_state.upload_costs
        with st.container(border=True):
            st.markdown(":material/payments: **Embedding cost**")
            total = sum(u["cost_usd"] for u in uploads)
            tokens = sum(u["tokens"] for u in uploads)
            st.metric("Uploads this session", f"${total:.4f}", help=f"{tokens:,} tokens", border=False)
            for u in reversed(uploads[-5:]):  # latest first
                st.caption(f"{u['file']}: {u['chunks']} chunks · {self.format_cost(u)}")
            all_calls = self.get_vectors_embeddings().total_usage
            st.caption(
                f"All embedding calls since app start (incl. search queries): "
                f"{self.format_cost(all_calls)}"
            )

    # --- Resetting a collection ---

    def render_reset_form(self) -> None:
        """Sidebar form: empty a collection. Needs a ticked confirmation box."""
        st.subheader("Reset collection")
        with st.form("reset_form", clear_on_submit=True, border=False):
            collection_label = st.selectbox("Collection to reset", list(self.COLLECTIONS))
            confirmed = st.checkbox("I understand this deletes all chunks in the collection")
            submitted = st.form_submit_button(
                "Reset collection", icon=":material/delete_forever:", type="primary", width="stretch"
            )

        if submitted:
            if not confirmed:
                st.warning("Tick the confirmation box to reset the collection.")
            else:
                self.reset_collection(self.COLLECTIONS[collection_label])

    def reset_collection(self, collection_name: str) -> None:
        """Empty the collection and report how many chunks were deleted."""
        try:
            with st.spinner(f"Resetting {collection_name}..."):
                deleted = self.get_vectors_embeddings().reset_collection(collection_name)
            st.success(f"'{collection_name}' is now empty ({deleted} chunks deleted).")
        except Exception as e:
            st.error(f"Could not reset '{collection_name}': {e}")

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
