"""Streamlit settings page: add PDFs to a Chroma collection, or reset a collection.

Opened from main_app.py after login, or on its own:
    uv run streamlit run src/llmragap/settings_gui.py
"""

import os

import streamlit as st

from llmragap.app_resources import COLLECTIONS, get_files_embeddings
from llmragap.session_cost import init_session_costs, money, save_session_cost, show_cost_save_error


def plural(n: int, word: str) -> str:
    return f"{n:,} {word}{'' if n == 1 else 's'}"


def format_cost(usage: dict) -> str:
    return f"{usage['tokens']:,} tokens, {money(usage['cost_usd'])}"


# ---------- collections overview ----------


def chunk_count(collection_name: str) -> int | None:
    try:
        return get_files_embeddings().load_chroma_db(collection_name)._collection.count()
    except Exception:
        return None


def render_overview() -> None:
    st.subheader("Collections")
    counts = {label: chunk_count(name) for label, name in COLLECTIONS.items()}
    for col, (label, count) in zip(st.columns(len(counts)), counts.items()):
        col.metric(label, "?" if count is None else f"{count:,}", help="Chunks stored", border=True)


# ---------- adding files ----------


def save_upload(upload) -> str:
    """Save an uploaded file into the data folder and return its full path."""
    vectors = get_files_embeddings()
    os.makedirs(vectors.data_dir, exist_ok=True)
    # basename() keeps a crafted file name from writing outside the data folder.
    file_path = os.path.join(vectors.data_dir, os.path.basename(upload.name))
    with open(file_path, "wb") as f:
        f.write(upload.getbuffer())
    return file_path


def add_files(uploads, collection_name: str, replace: bool) -> None:
    """Embed each uploaded PDF into the collection and report what happened."""
    vectors = get_files_embeddings()
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
                overall.progress((index + fraction) / n_files, text=f"{index}/{n_files} files done · embedding {name}")

        try:
            file_path = save_upload(upload)
            if replace:
                count = vectors.update_document(collection_name, file_path, progress=show_progress)
            else:
                count = vectors.add_document(collection_name, file_path, progress=show_progress)
        except Exception as e:
            totals["failed"] += 1
            results.error(f"{name}: {e}")
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
                results.success(f"{name}: {count} chunks {action} in '{collection_name}' ({format_cost(usage)}).")
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
        st.markdown(f"**{n_files} files:** {', '.join(parts)} · {totals['chunks']:,} chunks · {format_cost(totals)}")
    if totals["added"]:  # something was embedded, so the session cost changed
        save_session_cost()


def render_upload_form() -> None:
    st.subheader("Add documents")
    with st.form("upload_form", clear_on_submit=True):
        uploads = st.file_uploader(
            "PDF files",
            type="pdf",
            accept_multiple_files=True,
            help="Select or drop several PDFs at once; they are embedded one after another.",
        )
        collection_label = st.selectbox("Collection", list(COLLECTIONS))
        replace = st.checkbox(
            "Replace if already stored",
            help="Off: files already in the collection are skipped. "
            "On: their chunks are re-embedded and updated in place.",
        )
        submitted = st.form_submit_button("Add to database", icon=":material/upload:", type="primary", width="stretch")

    if submitted:
        if not uploads:
            st.warning("Choose at least one PDF first.")
        else:
            add_files(uploads, COLLECTIONS[collection_label], replace)

    uploads_done = st.session_state.upload_costs
    if uploads_done:
        total = {"tokens": sum(u["tokens"] for u in uploads_done), "cost_usd": sum(u["cost_usd"] for u in uploads_done)}
        with st.expander(f"Uploaded this session: {plural(len(uploads_done), 'file')} · {format_cost(total)}"):
            for u in reversed(uploads_done):  # latest first
                st.caption(f"{u['file']} → {u['collection']}: {u['chunks']} chunks · {format_cost(u)}")


# ---------- resetting a collection ----------


def reset_collection(collection_name: str) -> None:
    """Empty the collection and report how many chunks were deleted."""
    try:
        with st.spinner(f"Resetting {collection_name}..."):
            deleted = get_files_embeddings().reset_collection(collection_name)
        st.success(f"'{collection_name}' is now empty ({deleted} chunks deleted).")
    except Exception as e:
        st.error(f"Could not reset '{collection_name}': {e}")


def render_reset_form() -> None:
    st.subheader("Reset collection")
    st.caption("Deletes every chunk in the collection. The PDF files themselves are kept.")
    with st.form("reset_form", clear_on_submit=True):
        collection_label = st.selectbox("Collection to reset", list(COLLECTIONS))
        confirmed = st.checkbox("I understand this deletes all chunks in the collection")
        submitted = st.form_submit_button(
            "Reset collection", icon=":material/delete_forever:", type="primary", width="stretch"
        )

    if submitted:
        if not confirmed:
            st.warning("Tick the confirmation box to reset the collection.")
        else:
            reset_collection(COLLECTIONS[collection_label])


# ---------- page ----------

st.set_page_config(page_title="Settings", page_icon=":material/settings:")
init_session_costs()  # upload costs are part of the session cost
show_cost_save_error()
st.title(":material/settings: Settings")

add_tab, reset_tab = st.tabs([":material/upload_file: Add documents", ":material/delete_forever: Reset collection"])
with add_tab:
    render_upload_form()
with reset_tab:
    render_reset_form()

st.divider()
render_overview()  # drawn last, so the counts include what was just added or reset
