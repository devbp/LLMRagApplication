# llmragap

Retrieval-augmented generation (RAG) over German law PDFs, with evaluation.

- **Retrieval:** the Insurance Contract Act (VVG) and the Civil Code (BGB) are split
  into chunks, embedded through OpenRouter and stored in Chroma. The agent searches
  them with semantic, keyword (BM25) or hybrid search.
- **Evaluation:** DeepEval generates test cases and scores the agent's answers for
  answer relevancy and faithfulness to the retrieved passages.

## Files

| File | Purpose |
|---|---|
| `src/llmragap/RAG_Pdf.py` | Embedding into Chroma, search modes, the RAG agent |
| `src/llmragap/bm25_search.py` | BM25 keyword search |
| `src/llmragap/chat_gui.py` | Streamlit chat: ask questions, upload PDFs, see costs |
| `src/llmragap/generate_test_data.py` | Test cases generated from a PDF |
| `src/llmragap/generate_test_data_chroma.py` | Test cases generated from chunks already in Chroma |
| `src/llmragap/test_data_gui.py` | Streamlit page for the same: pick a collection, generate, review, save |
| `src/llmragap/eval_rag.py` | Runs test cases through the agent and scores them |
| `eval_data/` | Test cases and evaluation results (JSON) |
| `example_data/` | The law PDFs (not in git) |

## Setup

1. Copy `.env.example` to `.env` and set `OPENROUTER_API_KEY`.
2. Install dependencies: `uv sync`
3. Put `englisch_vvg.pdf` and `englisch_bgb.pdf` in `example_data/`.

## Run

```powershell
# Embed the PDFs (only if not embedded yet) and ask sample questions
uv run python src/llmragap/RAG_Pdf.py

# Chat app with login: start the auth API, then the app (login page -> chat page).
# Once, in the Supabase SQL editor: run supabase/session_costs.sql (cost of each session per user).
# (Table created before the test generation/evaluation costs? Run supabase/session_costs_add_eval.sql.)
uv run uvicorn llmragap.auth_api:app --port 8000
uv run streamlit run src/llmragap/main_app.py

# Chat app without login
uv run streamlit run src/llmragap/chat_gui.py

# Generate test cases from the stored chunks, then evaluate
uv run python src/llmragap/generate_test_data_chroma.py --collection insurance_docs
# ...or in the browser: "Test case generator" in main_app's sidebar, or on its own:
uv run streamlit run src/llmragap/test_data_gui.py
uv run python src/llmragap/eval_rag.py --input eval_data/insurance_docs_test_cases.json --mode hybrid
```

Evaluation results are written to `eval_data/law_eval_results.json` (change with `--output`).
