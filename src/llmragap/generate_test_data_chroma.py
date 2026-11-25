"""Generate RAG test cases (goldens) from chunks already stored in Chroma.

Unlike generate_test_data.py, the PDF is not read or embedded again: each context is a
random stored chunk plus its nearest neighbours, found with the vectors already in
Chroma, so no embedding calls are made. Only the question/answer model is called.

The output is a JSON list of {"input", "expected_output", "context"} objects, which
eval_rag.py can read directly:
    uv run python src/llmragap/generate_test_data_chroma.py
    uv run python src/llmragap/generate_test_data_chroma.py --collection civil_code_docs --contexts 10
    uv run python src/llmragap/eval_rag.py --input eval_data/insurance_docs_test_cases.json
"""

import argparse
import json
import os
import random
from pathlib import Path

from deepeval.synthesizer import Synthesizer
from dotenv import load_dotenv

from llmragap.llm_costs import TrackedOpenRouterModel
from llmragap.RAG_Pdf import CIVIL_CODE_COLLECTION, INSURANCE_COLLECTION, MEDICINE_COLLECTION, FilesEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

MIN_CHUNK_CHARS = 300  # skip short chunks (headers, table of contents lines) as starting points
COLLECTIONS = (INSURANCE_COLLECTION, CIVIL_CODE_COLLECTION, MEDICINE_COLLECTION)
GENERATOR_MODEL = "openai/gpt-4o-mini"


def default_output(collection_name: str) -> Path:
    return PROJECT_ROOT / "eval_data" / f"{collection_name}_test_cases.json"


def contexts_from_chroma(
    collection_name: str,
    n_contexts: int = 5,
    chunks_per_context: int = 3,
    seed: int | None = None,
) -> list[list[str]]:
    """Build contexts from stored chunks: a random chunk plus its nearest stored neighbours.

    Raises ValueError if the collection is empty or has too few long chunks.
    """
    collection = FilesEmbeddings().load_chroma_db(collection_name)._collection  # raw Chroma collection
    data = collection.get(include=["documents", "embeddings"])
    if not data["ids"]:
        raise ValueError(f"Collection '{collection_name}' is empty. Upload/embed its PDF first.")

    candidates = [i for i, doc in enumerate(data["documents"]) if len(doc) >= MIN_CHUNK_CHARS]
    if len(candidates) < n_contexts:
        raise ValueError(
            f"Only {len(candidates)} chunks of {MIN_CHUNK_CHARS}+ characters in '{collection_name}'; "
            f"asked for {n_contexts} contexts."
        )

    rng = random.Random(seed)
    contexts = []
    for i in rng.sample(candidates, n_contexts):
        # Neighbours found with the STORED vector: no new embedding call.
        hits = collection.query(query_embeddings=[data["embeddings"][i]], n_results=chunks_per_context)
        contexts.append(hits["documents"][0])  # the seed chunk itself comes first
    return contexts


def generate_test_cases(contexts: list[list[str]], per_context: int = 2) -> tuple[list[dict], dict]:
    """Ask the model for question/answer pairs grounded in each context (costs model calls).

    Returns (test cases, usage); usage is {"calls", "tokens", "cost_usd"} as billed by OpenRouter.
    Raises ValueError if OPENROUTER_API_KEY is missing.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is missing in .env")
    # Without an explicit model, DeepEval uses OpenAI directly and needs OPENAI_API_KEY.
    model = TrackedOpenRouterModel(model=GENERATOR_MODEL, api_key=api_key)
    goldens = Synthesizer(model=model).generate_goldens_from_contexts(
        contexts=contexts,
        max_goldens_per_context=per_context,
    )
    cases = [
        {"input": g.input, "expected_output": g.expected_output, "context": g.context or []}
        for g in goldens
    ]
    return cases, model.usage


def save_test_cases(cases: list[dict], output: Path) -> Path:
    """Write the test cases as JSON (the format eval_rag.py reads)."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(cases, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate test cases from chunks stored in Chroma.")
    parser.add_argument(
        "--collection",
        default=INSURANCE_COLLECTION,
        choices=COLLECTIONS,
        help="Chroma collection to read chunks from",
    )
    parser.add_argument("--contexts", type=int, default=5, help="number of contexts to build")
    parser.add_argument("--chunks-per-context", type=int, default=3, help="chunks grouped into one context")
    parser.add_argument("--per-context", type=int, default=2, help="test cases generated per context")
    parser.add_argument("--seed", type=int, default=None, help="random seed, for repeatable contexts")
    parser.add_argument("--output", default=None, help="output JSON (default: eval_data/<collection>_test_cases.json)")
    args = parser.parse_args()

    try:
        contexts = contexts_from_chroma(args.collection, args.contexts, args.chunks_per_context, args.seed)
        print(f"Built {len(contexts)} contexts from '{args.collection}'.")
        cases, usage = generate_test_cases(contexts, args.per_context)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    print(f"Generated {len(cases)} golden test cases.")
    print(f"Cost: ${usage['cost_usd']:.6f} ({usage['calls']} model calls, {usage['tokens']:,} tokens)")

    output = save_test_cases(cases, Path(args.output) if args.output else default_output(args.collection))
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
