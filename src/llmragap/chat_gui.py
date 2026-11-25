"""Simple Streamlit chatbot for the RagAgent in RAG_Pdf.py.

The agent answers law questions (VVG / BGB search), medicine questions (search of the
uploaded medicine documents) and general questions. Each answer shows which tools the agent used.
The sidebar also lets you upload PDFs and add them to a Chroma collection.

Run:
    uv run streamlit run src/llmragap/chat_gui.py
"""

import os
import re

import streamlit as st
from langchain_core.messages import AIMessage, ToolMessage

from llmragap.bm25_search import query_terms
from llmragap.RAG_Pdf import (
    CIVIL_CODE_COLLECTION,
    DEFAULT_SELECTION_MODE,
    INSURANCE_COLLECTION,
    MEDICINE_COLLECTION,
    RRF_K,
    SEARCH_MODES,
    ChatCostCallback,
    FilesEmbeddings,
    RagAgent,
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
    # Collections you can upload into (label shown in the UI -> Chroma collection name).
    COLLECTIONS = {
        "Insurance law (VVG)": INSURANCE_COLLECTION,
        "Civil code (BGB)": CIVIL_CODE_COLLECTION,
        "Medicine": MEDICINE_COLLECTION,
    }

    def __init__(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "upload_costs" not in st.session_state:
            # One entry per embedded file: {"file", "collection", "chunks", "tokens", "cost_usd"}
            st.session_state.upload_costs = []
        if "chat_costs" not in st.session_state:
            # Running totals for agent answers; kept when the chat is cleared.
            st.session_state.chat_costs = {"answers": 0, "tokens": 0, "agent_usd": 0.0, "search_usd": 0.0}
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
        tool_messages = {m.tool_call_id: m for m in new_messages if isinstance(m, ToolMessage)}
        steps = [
            {
                "tool": call["name"],
                "input": call["args"],
                "output": tool_messages[call["id"]].text if call["id"] in tool_messages else "",
                # search trace (query, collection, mode, ranked chunks with scores) for the UI
                "trace": getattr(tool_messages.get(call["id"]), "artifact", None),
            }
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
            self.render_upload_form()
            self.render_cost_summary()
            st.divider()
            self.render_reset_form()

    # --- Adding files to the vector database ---

    def render_upload_form(self) -> None:
        """Sidebar form: upload PDFs and add them to the chosen collection."""
        st.subheader("Add documents")
        with st.form("upload_form", clear_on_submit=True, border=False):
            uploads = st.file_uploader(
                "PDF files",
                type="pdf",
                accept_multiple_files=True,
                help="Select or drop several PDFs at once; they are embedded one after another.",
            )
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
    def money(usd: float) -> str:
        """$0.0123 for normal amounts; tiny ones keep their digits ($0.00000039, not $0.0000)."""
        if usd == 0 or usd >= 0.001:
            return f"${usd:.4f}"
        return "$" + f"{usd:.8f}".rstrip("0")

    @staticmethod
    def plural(n: int, word: str) -> str:
        return f"{n:,} {word}{'' if n == 1 else 's'}"

    @classmethod
    def format_cost(cls, usage: dict) -> str:
        return f"{usage['tokens']:,} tokens, {cls.money(usage['cost_usd'])}"

    def add_files(self, uploads, collection_name: str, replace: bool) -> None:
        """Embed each uploaded PDF into the collection and report what happened."""
        vectors = self.get_vectors_embeddings()
        n_files = len(uploads)
        multiple = n_files > 1
        # Overall bar: each file is an equal share; inside a file it moves with its chunks.
        overall = st.progress(0, text=f"0/{n_files} files done") if multiple else None
        file_bar = st.progress(0)
        results = st.container()  # one result line per file, in upload order
        totals = {"added": 0, "skipped": 0, "failed": 0, "chunks": 0, "tokens": 0, "cost_usd": 0.0}

        for index, upload in enumerate(uploads):
            name = os.path.basename(upload.name)
            label = f"{name} ({index + 1}/{n_files})" if multiple else name
            file_bar.progress(0, text=f"{label}: reading and splitting PDF...")

            def show_progress(done: int, total: int, index=index, label=label) -> None:
                fraction = done / total if total else 1.0
                file_bar.progress(fraction, text=f"{label}: embedding {done}/{total} chunks")
                if overall:
                    overall.progress(
                        (index + fraction) / n_files,
                        text=f"{index}/{n_files} files done · embedding {name}",
                    )

            try:
                file_path = self.save_upload(upload)
                if replace:
                    count = vectors.update_document(collection_name, file_path, progress=show_progress)
                else:
                    count = vectors.add_document(collection_name, file_path, progress=show_progress)
            except Exception as e:
                totals["failed"] += 1
                results.error(f"{name}: {e}")
                count = None
            else:
                if not count:
                    totals["skipped"] += 1
                    results.info(f"{name} is already in '{collection_name}', skipped (no cost).")
                else:
                    usage = vectors.last_upload_usage
                    totals["added"] += 1
                    totals["chunks"] += count
                    totals["tokens"] += usage["tokens"]
                    totals["cost_usd"] += usage["cost_usd"]
                    action = "stored" if replace else "added"
                    results.success(
                        f"{name}: {count} chunks {action} in '{collection_name}' ({self.format_cost(usage)})."
                    )
                    st.session_state.upload_costs.append(
                        {"file": name, "collection": collection_name, "chunks": count, **usage}
                    )
            if overall:
                overall.progress((index + 1) / n_files, text=f"{index + 1}/{n_files} files done")

        file_bar.empty()
        if overall:
            overall.empty()
            parts = [f"{totals['added']} embedded"]
            if totals["skipped"]:
                parts.append(f"{totals['skipped']} skipped")
            if totals["failed"]:
                parts.append(f"{totals['failed']} failed")
            st.markdown(
                f"**{n_files} files:** {', '.join(parts)} · {totals['chunks']:,} chunks · "
                f"{self.format_cost(totals)}"
            )

    def render_cost_summary(self) -> None:
        """Sidebar: everything this session cost (as billed by OpenRouter), with a breakdown."""
        uploads = st.session_state.upload_costs
        chat = st.session_state.chat_costs
        upload_usd = sum(u["cost_usd"] for u in uploads)
        upload_tokens = sum(u["tokens"] for u in uploads)
        total = upload_usd + chat["agent_usd"] + chat["search_usd"]

        with st.container(border=True):
            st.metric(":material/payments: Total cost this session", self.money(total), border=False)
            st.table(
                {
                    "Part": ["Agent model", "Search embeddings", "Upload embeddings"],
                    "Cost": [self.money(chat["agent_usd"]), self.money(chat["search_usd"]), self.money(upload_usd)],
                    "Details": [
                        f"{self.plural(chat['answers'], 'answer')}, {chat['tokens']:,} tokens",
                        "questions embedded for document search",
                        f"{self.plural(len(uploads), 'file')}, {upload_tokens:,} tokens",
                    ],
                },
                hide_index=True,
            )
            for u in reversed(uploads[-5:]):  # latest uploads first
                st.caption(f"{u['file']}: {u['chunks']} chunks · {self.format_cost(u)}")

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

    # --- Visualising the retrieval ---

    # Characters that would be read as Markdown/LaTeX in chunk text.
    MARKDOWN_SPECIAL = re.compile(r"([\\`*_{}\[\]<>#|$~])")

    @staticmethod
    def cited_pages(answer: str) -> set[int]:
        """Page numbers the answer mentions, e.g. "page 12", "pages 12-13", "p. 7"."""
        pages = set()
        for match in re.finditer(r"\b(?:pages?|p\.)\s*(\d+(?:\s*(?:[-–,&]|and)\s*\d+)*)", answer, re.IGNORECASE):
            pages.update(int(n) for n in re.findall(r"\d+", match.group(1)))
        return pages

    @classmethod
    def highlight(cls, text: str, query: str) -> str:
        """Chunk text as Markdown, with the query's search words highlighted."""
        text = cls.MARKDOWN_SPECIAL.sub(r"\\\1", text)
        terms, _ = query_terms(query, "text")  # the same words BM25 searches for
        if not terms:
            return text
        pattern = re.compile(r"\b(" + "|".join(map(re.escape, sorted(set(terms), key=len, reverse=True))) + r")\b", re.I)
        return pattern.sub(r":orange-background[\1]", text)

    def show_retrieval(self, steps: list[dict], answer: str) -> None:
        """'How this answer was found': one tab per document search the agent made."""
        traces = [s["trace"] for s in steps if s.get("trace")]
        if not traces:
            return
        cited = self.cited_pages(answer)
        with st.expander("How this answer was found", icon=":material/manage_search:"):
            labels = [f"{i}. {t['collection']}" for i, t in enumerate(traces, start=1)]
            for tab, trace in zip(st.tabs(labels), traces):
                with tab:
                    self.show_trace(trace, cited)

    def show_trace(self, trace: dict, cited: set[int]) -> None:
        """One search: query, where it searched, ranked chunks, score chart and context."""
        results = trace["results"]
        mode = trace["mode"]

        st.markdown("**1 · Search query** written by the agent")
        st.code(trace["query"], language=None, wrap_lines=True)
        st.markdown(
            f"**2 · Searched** `{trace['collection']}` with **{mode}** search "
            f"→ {self.plural(len(results), 'chunk')} sent to the model"
        )
        if not results:
            st.info("No matching passages found.", icon=":material/search_off:")
            return

        # 3. Ranked chunks with the score of each search method
        st.markdown("**3 · Retrieved chunks**")
        rows = []
        for r in results:
            row = {"#": r["rank"], "File": r["file"], "Page": r["page"], "Cited": r["page"] in cited}
            if mode in ("semantic", "hybrid"):
                row["Semantic rank"] = r["semantic_rank"]
                row["Distance"] = r["semantic_distance"]
            if mode in ("keyword", "hybrid"):
                row["Keyword rank"] = r["keyword_rank"]
                row["BM25"] = r["keyword_score"]
            if mode == "hybrid":
                row["Fused score"] = r["fused_score"]
            rows.append(row)
        bm25_max = max((r["keyword_score"] or 0) for r in results) or 1.0
        st.dataframe(
            rows,
            hide_index=True,
            column_config={
                "#": st.column_config.NumberColumn(width="small"),
                "Page": st.column_config.NumberColumn(width="small"),
                "Cited": st.column_config.CheckboxColumn(help="The answer mentions this page number"),
                "Semantic rank": st.column_config.NumberColumn(
                    help="Position among the semantic search candidates (empty = not found by it)"
                ),
                "Distance": st.column_config.NumberColumn(format="%.3f", help="Embedding distance; lower = closer in meaning"),
                "Keyword rank": st.column_config.NumberColumn(
                    help="Position among the BM25 keyword candidates (empty = not found by it)"
                ),
                "BM25": st.column_config.ProgressColumn(
                    format="%.2f", min_value=0.0, max_value=bm25_max, help="BM25 keyword score; higher = better match"
                ),
                "Fused score": st.column_config.ProgressColumn(
                    format="%.4f", min_value=0.0, max_value=2 / (RRF_K + 1),
                    help=f"Reciprocal rank fusion: 1/({RRF_K} + rank) per method, added up",
                ),
            },
        )

        # 4. Why each chunk was picked
        labels = [f"#{r['rank']} · p. {r['page']}" for r in results]
        if mode == "hybrid":
            st.markdown("**4 · Why each chunk was picked** – contribution of each method to the fused score")
            chart = {
                "Chunk": labels,
                "Semantic": [1 / (RRF_K + r["semantic_rank"]) if r["semantic_rank"] else 0.0 for r in results],
                "Keyword": [1 / (RRF_K + r["keyword_rank"]) if r["keyword_rank"] else 0.0 for r in results],
            }
            st.bar_chart(chart, x="Chunk", y=["Semantic", "Keyword"], horizontal=True, sort=False,
                         x_label="Fused score", y_label="", height=60 + 40 * len(results))
        else:
            metric = "Distance (lower = closer)" if mode == "semantic" else "BM25 score"
            key = "semantic_distance" if mode == "semantic" else "keyword_score"
            st.markdown(f"**4 · Scores** – {metric}")
            st.bar_chart({"Chunk": labels, metric: [r[key] for r in results]}, x="Chunk", y=metric,
                         horizontal=True, sort=False, y_label="", height=60 + 40 * len(results))

        # 5. The text the model received
        st.markdown("**5 · Context sent to the model** (search words highlighted)")
        for r in results:
            mark = " · :green[cited]" if r["page"] in cited else ""
            with st.expander(f"#{r['rank']} · {r['file']} · page {r['page']}{mark}", type="step"):
                st.markdown(self.highlight(r["text"], trace["query"]))

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
                    self.show_retrieval(msg.get("steps", []), msg["content"])
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
            self.show_retrieval(steps, answer)
            st.markdown(answer)
            self.show_message_cost(cost)

        if cost:
            totals = st.session_state.chat_costs
            totals["answers"] += 1
            totals["tokens"] += cost["input_tokens"] + cost["output_tokens"]
            totals["agent_usd"] += cost["cost_usd"]
            totals["search_usd"] += cost["search_embedding_usd"]

        self.messages.append({"role": "assistant", "content": answer, "steps": steps, "cost": cost})
        st.rerun()

    def run(self) -> None:
        """Draw the page; called on every Streamlit rerun."""
        self.render_header()
        self.render_sidebar()
        self.render_history()
        if prompt := self.get_prompt():
            self.handle_prompt(prompt)


ChatApp().run()
