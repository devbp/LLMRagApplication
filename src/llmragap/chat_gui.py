"""Simple Streamlit chatbot for the RagAgent in RAG_Pdf.py.

The agent answers law questions (VVG / BGB search), medicine questions (search of the
uploaded medicine documents) and general questions. Each answer shows which tools the agent used.
Adding documents and resetting collections is on the settings page (settings_gui.py).

Run:
    uv run streamlit run src/llmragap/chat_gui.py
"""

import streamlit as st
from langchain_core.messages import AIMessage, ToolMessage

from llmragap.app_resources import get_files_embeddings
from llmragap.RAG_Pdf import (
    DEFAULT_SELECTION_MODE,
    SEARCH_MODES,
    ChatCostCallback,
    FilesEmbeddings,
    RagAgent,
)
from llmragap.session_cost import (
    init_session_costs,
    money,
    save_session_cost,
    session_total_usd,
    show_cost_save_error,
)


class ChatApp:
    """Streamlit chat page around a RagAgent.

    Streamlit reruns this whole script on every interaction, so the chat history
    lives in st.session_state and the slow-to-build objects (Chroma collections,
    agent) are cached with st.cache_resource.
    """

    TITLE = "Law & medicine chatbot"
    SUGGESTIONS = {
        ":blue[:material/gavel:] Limitation period": "What is the standard limitation period under the Civil Code?",
        ":green[:material/policy:] Cancel insurance": "When can a policyholder cancel an insurance contract?",
        ":orange[:material/public:] General question": "What is the capital of Nepal?",
    }
    MAX_TOOL_OUTPUT_CHARS = 2000  # tool outputs longer than this are cut off in the UI

    def __init__(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []
        init_session_costs()  # upload_costs, chat_costs, session_id, ... (shared with other pages)
        self.selection_mode = DEFAULT_SELECTION_MODE

    @property
    def messages(self) -> list[dict]:
        """Chat history for this browser session: {"role", "content", "steps"?} dicts."""
        return st.session_state.messages

    # --- Cached resources (shared across reruns and sessions) ---

    @staticmethod
    def get_vectors_embeddings() -> FilesEmbeddings:
        """The Chroma collections, opened once and shared with the settings page."""
        return get_files_embeddings()

    @staticmethod
    @st.cache_resource
    def get_agent(selection_mode: str) -> RagAgent:
        """Build one agent per search mode."""
        return RagAgent(ChatApp.get_vectors_embeddings(), selection_mode)

    # --- Agent ---

    def run_agent(self) -> tuple[str, list[dict], dict]:
        """Send the whole conversation to the agent; return (answer, tool steps, cost)."""
        history = [{"role": m["role"], "content": m["content"]} for m in self.messages]
        cost = ChatCostCallback()  # counts every model call in this run, including inside tools
        # Document searches embed the query; measure that as the change in the embedding total.
        # (The embeddings object is shared, so a simultaneous search in another tab could add to it.)
        embeddings_before = self.get_vectors_embeddings().total_usage["cost_usd"]
        result_messages = self.get_agent(self.selection_mode).agent.invoke(
            {"messages": history}, config={"callbacks": [cost]}
        )["messages"]
        search_cost = self.get_vectors_embeddings().total_usage["cost_usd"] - embeddings_before

        # Only look at messages from this turn (after the conversation we sent).
        new_messages = result_messages[len(history):]
        outputs = {m.tool_call_id: m.text for m in new_messages if isinstance(m, ToolMessage)}
        steps = [
            {"tool": call["name"], "input": call["args"], "output": outputs.get(call["id"], "")}
            for m in new_messages
            if isinstance(m, AIMessage)
            for call in m.tool_calls
        ]
        return result_messages[-1].text, steps, {**cost.as_dict(), "search_embedding_usd": search_cost}

    # --- UI parts ---

    def render_header(self) -> None:
        st.set_page_config(page_title=self.TITLE, page_icon=":material/smart_toy:")
        st.title(f":material/smart_toy: {self.TITLE}")
        st.caption("Ask about German insurance or civil law, your medicine documents, or anything else.")

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
            self.render_cost_summary()

    money = staticmethod(money)

    @staticmethod
    def plural(n: int, word: str) -> str:
        return f"{n:,} {word}{'' if n == 1 else 's'}"

    @classmethod
    def format_cost(cls, usage: dict) -> str:
        return f"{usage['tokens']:,} tokens, {cls.money(usage['cost_usd'])}"

    @staticmethod
    def save_session_cost() -> None:
        """Send this session's running cost totals to the auth API (see session_cost.py)."""
        save_session_cost()

    def render_cost_summary(self) -> None:
        """Sidebar: everything this session cost (as billed by OpenRouter), with a breakdown."""
        uploads = st.session_state.upload_costs
        chat = st.session_state.chat_costs
        testgen = st.session_state.testgen_costs
        evals = st.session_state.eval_costs
        upload_usd = sum(u["cost_usd"] for u in uploads)
        upload_tokens = sum(u["tokens"] for u in uploads)

        with st.container(border=True):
            st.metric(":material/payments: Total cost this session", self.money(session_total_usd()), border=False)
            st.table(
                {
                    "Part": ["Agent model", "Search embeddings", "Upload embeddings", "Test generation", "Evaluation"],
                    "Cost": [
                        self.money(chat["agent_usd"]),
                        self.money(chat["search_usd"]),
                        self.money(upload_usd),
                        self.money(testgen["usd"]),
                        self.money(evals["usd"]),
                    ],
                    "Details": [
                        f"{self.plural(chat['answers'], 'answer')}, {chat['tokens']:,} tokens",
                        "questions embedded for document search",
                        f"{self.plural(len(uploads), 'file')}, {upload_tokens:,} tokens",
                        f"{self.plural(testgen['runs'], 'run')}, {self.plural(testgen['cases'], 'test case')}",
                        f"{self.plural(evals['runs'], 'run')}, {self.plural(evals['cases'], 'test case')}",
                    ],
                },
                hide_index=True,
            )
            for u in reversed(uploads[-5:]):  # latest uploads first
                st.caption(f"{u['file']}: {u['chunks']} chunks · {self.format_cost(u)}")

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

    @staticmethod
    def show_message_cost(cost: dict | None) -> None:
        """Small line under an answer: what the model calls for it cost."""
        if not cost:
            return
        total = cost["cost_usd"] + cost.get("search_embedding_usd", 0.0)
        st.caption(
            f":material/payments: {ChatApp.money(total)} · "
            f"{cost['input_tokens']:,} in / {cost['output_tokens']:,} out tokens · "
            f"{ChatApp.plural(cost['calls'], 'model call')}"
        )

    def render_history(self) -> None:
        for msg in self.messages:
            with st.chat_message(msg["role"]):
                if msg["role"] == "assistant":
                    self.show_steps(msg.get("steps", []))
                st.markdown(msg["content"])
                if msg["role"] == "assistant":
                    self.show_message_cost(msg.get("cost"))

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
                    answer, steps, cost = self.run_agent()
            except Exception as e:
                answer, steps, cost = f":red[Error: {e}]", [], None
            self.show_steps(steps)
            st.markdown(answer)
            self.show_message_cost(cost)

        if cost:
            totals = st.session_state.chat_costs
            totals["answers"] += 1
            totals["tokens"] += cost["input_tokens"] + cost["output_tokens"]
            totals["agent_usd"] += cost["cost_usd"]
            totals["search_usd"] += cost["search_embedding_usd"]
            self.save_session_cost()

        self.messages.append({"role": "assistant", "content": answer, "steps": steps, "cost": cost})
        st.rerun()

    def run(self) -> None:
        """Draw the page; called on every Streamlit rerun."""
        self.render_header()
        show_cost_save_error()
        self.render_sidebar()
        self.render_history()
        if prompt := self.get_prompt():
            self.handle_prompt(prompt)


ChatApp().run()
