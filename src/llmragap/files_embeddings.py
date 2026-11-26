"""Embedding PDF files into Chroma collections, and searching them.

FilesEmbeddings handles storage (load, split, embed, add/update/reset files in a
collection) and retrieval (semantic, keyword/BM25 or hybrid search). It also records
the embedding cost OpenRouter bills. The RAG agent itself is in RAG_Pdf.py.
"""

import json
import os
import threading
from collections.abc import Callable
from contextlib import contextmanager
from typing import TypeAlias

import chromadb
import httpx
import pypdf
from dotenv import find_dotenv, load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from llmragap.bm25_search import KeywordIndex

load_dotenv(find_dotenv())  # OPENROUTER_API_KEY from the project .env

PERSIST_DIR = "C:\\Users\\Lenovo\\Downloads\\llmragap\\chroma_langchain_db"
DATA_DIR = "C:\\Users\\Lenovo\\Downloads\\llmragap\\example_data"

# One collection per document set, all inside the same Chroma database folder.
INSURANCE_COLLECTION = "insurance_docs"
INSURANCE_PDF = "C:\\Users\\Lenovo\\Downloads\\llmragap\\example_data\\englisch_vvg.pdf"

CIVIL_CODE_COLLECTION = "civil_code_docs"
CIVIL_CODE_PDF = "C:\\Users\\Lenovo\\Downloads\\llmragap\\example_data\\englisch_bgb.pdf"

# Medicine documents (drug leaflets, guidelines, ...); filled by uploading PDFs in the chat app.
MEDICINE_COLLECTION = "medicine_docs"

# How the search tools find passages. The user picks one:
#   "semantic" - embedding similarity (finds passages with the same meaning)
#   "keyword"  - BM25 keyword match (finds exact terms like "section 40")
#   "hybrid"   - both, merged with reciprocal rank fusion
SEARCH_MODES = ("semantic", "keyword", "hybrid")
DEFAULT_SELECTION_MODE = "hybrid"

# Chunks are embedded in batches of this size, so progress can be reported between batches.
EMBED_BATCH_SIZE = 50
ProgressCallback: TypeAlias = Callable[[int, int], None]  # progress(done_chunks, total_chunks)


class FilesEmbeddings:
    """Embeds PDF files into Chroma collections and searches them.

    Storage: load, split and embed PDFs; add or update files in a collection.
    Retrieval: semantic, keyword (BM25) or hybrid search in a collection.
    """

    def __init__(
        self,
        persist_dir: str = PERSIST_DIR,
        data_dir: str = DATA_DIR,
        embedding_model: str = "text-embedding-3-large",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.persist_dir = persist_dir
        self.data_dir = data_dir
        # Embedding usage as billed by OpenRouter (read from each API response).
        # total_usage: every embedding call since this object was created (uploads and searches).
        # last_upload_usage: the most recent embed_into_chroma/add_document/update_document.
        self._usage_lock = threading.Lock()
        self.total_usage = {"tokens": 0, "cost_usd": 0.0}
        self.last_upload_usage = {"tokens": 0, "cost_usd": 0.0}
        # The same embedding model must be used for storing and for searching.
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            check_embedding_ctx_length=False,
            # LangChain drops the "usage" block of the response; this hook reads it first.
            http_client=httpx.Client(event_hooks={"response": [self._record_usage]}),
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
        )
        # Opened collections, and keyword indexes built from their chunks.
        # Writing to a collection clears its keyword index so it is rebuilt.
        self._stores: dict[str, Chroma] = {}
        # collection name -> (chunks, BM25 index over their text)
        self._keyword_indexes: dict[str, tuple[list[Document], KeywordIndex]] = {}

    # --- Embedding cost ---

    def _record_usage(self, response: httpx.Response) -> None:
        """httpx response hook: add the tokens and cost OpenRouter reports to total_usage.

        The response ends with {..., "usage": {"prompt_tokens", "cost", ...}, ...}.
        Only that small block is decoded, not the (possibly huge) vector data.
        """
        if not response.url.path.endswith("/embeddings") or response.status_code != 200:
            return
        response.read()
        text = response.text
        start = text.rfind('"usage"')
        if start == -1:
            return
        try:
            usage, _ = json.JSONDecoder().raw_decode(text, text.index("{", start))
        except ValueError:
            return
        with self._usage_lock:
            self.total_usage["tokens"] += int(usage.get("prompt_tokens", 0))
            self.total_usage["cost_usd"] += float(usage.get("cost", 0.0))

    @contextmanager
    def track_upload_cost(self):
        """Measure the embedding cost of the code inside the block into last_upload_usage."""
        before = dict(self.total_usage)
        try:
            yield
        finally:
            self.last_upload_usage = {
                "tokens": self.total_usage["tokens"] - before["tokens"],
                "cost_usd": self.total_usage["cost_usd"] - before["cost_usd"],
            }
            if self.last_upload_usage["tokens"]:
                print(
                    f"Embedding cost: {self.last_upload_usage['tokens']:,} tokens = "
                    f"${self.last_upload_usage['cost_usd']:.6f}"
                )

    # --- Storage ---

    def load_chroma_db(self, collection_name: str) -> Chroma:
        """Open a Chroma collection (once) without embedding anything new."""
        if collection_name not in self._stores:
            self._stores[collection_name] = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_dir,
            )
        return self._stores[collection_name]

    def is_already_embedded(self, collection_name: str) -> bool:
        """Return True if the named collection exists and already contains documents."""
        client = chromadb.PersistentClient(path=self.persist_dir)
        names = [c.name for c in client.list_collections()]
        return collection_name in names and client.get_collection(collection_name).count() > 0

    def resolve_path(self, file_name: str) -> str:
        """Turn a file name inside data_dir (or a full path) into a full path that exists."""
        file_path = file_name if os.path.isabs(file_name) else os.path.join(self.data_dir, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        return file_path

    @staticmethod
    def load_pdf_pages(file_path: str) -> list[Document]:
        """Read a PDF into one Document per page."""
        reader = pypdf.PdfReader(file_path)
        return [
            Document(
                page_content=page.extract_text() or "",
                metadata={"source": file_path, "page": i},
            )
            for i, page in enumerate(reader.pages)
        ]

    def split_pdf(self, file_path: str) -> list[Document]:
        """Load a PDF and split it into overlapping chunks."""
        return self.text_splitter.split_documents(self.load_pdf_pages(file_path))

    @staticmethod
    def make_chunk_ids(file_path: str, splits: list[Document]) -> list[str]:
        """Give each chunk a stable ID like "englisch_vvg.pdf::chunk-12".

        The same file always produces the same IDs, which lets update_documents()
        overwrite a file's chunks in place.
        """
        name = os.path.basename(file_path)
        return [f"{name}::chunk-{i}" for i in range(len(splits))]

    def store_chunks(
        self,
        vector_store: Chroma,
        to_add: list[tuple[str, Document]],
        to_update: list[tuple[str, Document]] = (),
        progress: ProgressCallback | None = None,
    ) -> None:
        """Embed and store (id, chunk) pairs in batches, reporting progress after each batch.

        Args:
            to_add: Chunks to add with add_documents().
            to_update: Chunks to overwrite in place with update_documents().
            progress: Called as progress(done, total) after each batch.
        """
        total = len(to_add) + len(to_update)
        done = 0
        if progress:
            progress(0, total)
        for pairs, store in ((to_update, vector_store.update_documents), (to_add, vector_store.add_documents)):
            for start in range(0, len(pairs), EMBED_BATCH_SIZE):
                ids, docs = zip(*pairs[start:start + EMBED_BATCH_SIZE])
                store(ids=list(ids), documents=list(docs))
                done += len(ids)
                if progress:
                    progress(done, total)

    def embed_into_chroma(
        self, file_path: str, collection_name: str, progress: ProgressCallback | None = None
    ) -> Chroma:
        """Load a PDF, split it into chunks, embed them and store them in the given collection."""
        all_splits = self.split_pdf(file_path)
        vector_store = self.load_chroma_db(collection_name)
        with self.track_upload_cost():
            pairs = list(zip(self.make_chunk_ids(file_path, all_splits), all_splits))
            self.store_chunks(vector_store, to_add=pairs, progress=progress)
        self._keyword_indexes.pop(collection_name, None)
        print(f"Stored {len(all_splits)} chunks in '{collection_name}'.")
        return vector_store

    def ensure_embedded(self, collection_name: str, file_path: str) -> None:
        """Embed the PDF into the collection only if the collection is still empty."""
        if self.is_already_embedded(collection_name):
            print(f"'{collection_name}' already exists, skipping embedding.")
        elif not os.path.exists(file_path):
            print(f"'{collection_name}' is empty and {file_path} was not found, skipping.")
        else:
            print(f"'{collection_name}' not found, embedding {file_path}...")
            self.embed_into_chroma(file_path, collection_name)

    def update_document(
        self, collection_name: str, file_name: str, progress: ProgressCallback | None = None
    ) -> int:
        """Re-embed one file and update its chunks in place with Chroma's update_documents().

        Chunks are matched by their stable IDs (see make_chunk_ids):
        - IDs already stored        -> overwritten with update_documents()
        - IDs that are new          -> added (the file now has more chunks)
        - stored IDs no longer used -> deleted (the file now has fewer chunks, or the
                                       chunks were stored earlier with random IDs)
        Other files in the collection are untouched. A file never stored before is added.

        Args:
            collection_name: Collection to update, e.g. "insurance_docs".
            file_name: PDF file name inside data_dir (e.g. "englisch_vvg.pdf"),
                or a full path to the file.
            progress: Optional progress(done, total) callback, called after each batch of chunks.

        Returns:
            Number of chunks now stored for the file.
        """
        file_path = self.resolve_path(file_name)
        vector_store = self.load_chroma_db(collection_name)
        name = os.path.basename(file_path)

        new_splits = self.split_pdf(file_path)
        new_ids = self.make_chunk_ids(file_path, new_splits)

        # IDs currently stored for this file (matched by the "source" metadata).
        old_ids = set(vector_store.get(where={"source": file_path})["ids"])

        to_update = [(i, d) for i, d in zip(new_ids, new_splits) if i in old_ids]
        to_add = [(i, d) for i, d in zip(new_ids, new_splits) if i not in old_ids]
        to_delete = list(old_ids - set(new_ids))

        with self.track_upload_cost():
            self.store_chunks(vector_store, to_add=to_add, to_update=to_update, progress=progress)
        if to_delete:
            vector_store.delete(ids=to_delete)
        self._keyword_indexes.pop(collection_name, None)

        print(
            f"{name} in '{collection_name}': updated {len(to_update)}, "
            f"added {len(to_add)}, deleted {len(to_delete)} chunks."
        )
        return len(new_splits)

    def add_document(
        self, collection_name: str, file_name: str, progress: ProgressCallback | None = None
    ) -> int:
        """Add a new PDF to an existing collection without touching what's already there.

        If the file is already stored in the collection, it is skipped so that running
        this twice does not create duplicate chunks. Use update_document() instead to
        replace a file whose contents have changed.

        Args:
            collection_name: Collection to add to, e.g. "insurance_docs".
            file_name: PDF file name inside data_dir (e.g. "new_policy.pdf"),
                or a full path to the file.
            progress: Optional progress(done, total) callback, called after each batch of chunks.

        Returns:
            Number of chunks added (0 if the file was already stored).
        """
        file_path = self.resolve_path(file_name)
        vector_store = self.load_chroma_db(collection_name)

        # Chunks are matched by the "source" metadata set in load_pdf_pages().
        if vector_store.get(where={"source": file_path}, limit=1)["ids"]:
            print(f"{os.path.basename(file_path)} is already in '{collection_name}', skipping.")
            self.last_upload_usage = {"tokens": 0, "cost_usd": 0.0}  # nothing embedded
            return 0

        new_splits = self.split_pdf(file_path)
        with self.track_upload_cost():
            pairs = list(zip(self.make_chunk_ids(file_path, new_splits), new_splits))
            self.store_chunks(vector_store, to_add=pairs, progress=progress)
        self._keyword_indexes.pop(collection_name, None)
        print(f"Added {len(new_splits)} chunks of {os.path.basename(file_path)} to '{collection_name}'.")
        return len(new_splits)

    def reset_collection(self, collection_name: str) -> int:
        """Delete every chunk in a collection, leaving it empty (the collection itself stays).

        Uses Chroma's reset_collection(), which deletes the collection and recreates
        it empty. Other collections are untouched. Use add_document() or
        embed_into_chroma() afterwards to fill it again.

        Args:
            collection_name: Collection to empty, e.g. "insurance_docs".

        Returns:
            Number of chunks that were deleted.
        """
        vector_store = self.load_chroma_db(collection_name)
        deleted = len(vector_store.get(include=[])["ids"])
        vector_store.reset_collection()
        self._keyword_indexes.pop(collection_name, None)
        print(f"Reset '{collection_name}': deleted {deleted} chunks.")
        return deleted

    # --- Retrieval ---

    def get_keyword_index(self, collection_name: str) -> tuple[list[Document], KeywordIndex]:
        """Build (once) and return a collection's chunks and the BM25 index over them."""
        if collection_name not in self._keyword_indexes:
            data = self.load_chroma_db(collection_name).get(include=["documents", "metadatas"])
            docs = [
                Document(page_content=text or "", metadata=meta or {}, id=doc_id)
                for doc_id, text, meta in zip(data["ids"], data["documents"], data["metadatas"])
            ]
            index = KeywordIndex([d.page_content for d in docs])
            self._keyword_indexes[collection_name] = (docs, index)
        return self._keyword_indexes[collection_name]

    def keyword_search(self, collection_name: str, query: str, k: int) -> list[Document]:
        """BM25 keyword search (via bm25_search) over a collection's chunks."""
        docs, index = self.get_keyword_index(collection_name)
        return [docs[i] for i, _ in index.search(query, top_n=k, mode="text")]

    @staticmethod
    def reciprocal_rank_fusion(result_lists: list[list[Document]], k: int, rrf_k: int = 60) -> list[Document]:
        """Merge several ranked lists. A chunk ranked high in either list (or in both) wins.

        Each chunk scores 1 / (rrf_k + rank) per list it appears in; scores are summed.
        """
        scores: dict[str, float] = {}
        by_key: dict[str, Document] = {}
        for results in result_lists:
            for rank, doc in enumerate(results, start=1):
                key = doc.id or doc.page_content
                scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
                by_key.setdefault(key, doc)
        ranked = sorted(scores, key=scores.get, reverse=True)
        return [by_key[key] for key in ranked[:k]]

    def search(
        self,
        collection_name: str,
        query: str,
        selection_mode: str = DEFAULT_SELECTION_MODE,
        k: int = 4,
    ) -> list[Document]:
        """Search one collection using the selected mode: semantic, keyword or hybrid."""
        if selection_mode not in SEARCH_MODES:
            raise ValueError(f"selection_mode must be one of {SEARCH_MODES}, got {selection_mode!r}")

        store = self.load_chroma_db(collection_name)
        if selection_mode == "semantic":
            return store.similarity_search(query, k=k)
        if selection_mode == "keyword":
            return self.keyword_search(collection_name, query, k)

        # hybrid: take extra candidates from each method, then fuse the rankings
        candidates = k * 3
        semantic = store.similarity_search(query, k=candidates)
        keyword = self.keyword_search(collection_name, query, candidates)
        return self.reciprocal_rank_fusion([semantic, keyword], k)
