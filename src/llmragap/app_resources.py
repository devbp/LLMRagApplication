"""Streamlit resources shared by all pages.

One FilesEmbeddings for the whole app: it caches each collection's keyword (BM25)
index and clears it when documents are added or a collection is reset, so every
page must use the same instance to see those changes.
"""

import streamlit as st

from llmragap.files_embeddings import CIVIL_CODE_COLLECTION, INSURANCE_COLLECTION, MEDICINE_COLLECTION, FilesEmbeddings

# Label shown in the UI -> Chroma collection name.
COLLECTIONS = {
    "Insurance law (VVG)": INSURANCE_COLLECTION,
    "Civil code (BGB)": CIVIL_CODE_COLLECTION,
    "Medicine": MEDICINE_COLLECTION,
}


@st.cache_resource
def get_files_embeddings() -> FilesEmbeddings:
    """Open the Chroma vector collections once for all pages and sessions."""
    return FilesEmbeddings()
