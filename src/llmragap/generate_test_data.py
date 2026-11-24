"""Generate RAG test cases (goldens) from a law PDF with DeepEval's Synthesizer.

The output is a JSON list of {"input", "expected_output", "context"} objects, which
eval_rag.py can read directly:
    uv run python src/llmragap/generate_test_data.py
    uv run python src/llmragap/eval_rag.py --input eval_data/vvg_test_cases.json
"""

import json
import os
from pathlib import Path

from deepeval.dataset import EvaluationDataset
from deepeval.models import DeepEvalBaseEmbeddingModel, OpenRouterModel
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import ContextConstructionConfig
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

PDF_PATH = PROJECT_ROOT / "example_data" / "englisch_vvg.pdf"
OUTPUT_JSON = PROJECT_ROOT / "eval_data" / "vvg_test_cases.json"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterEmbeddingModel(DeepEvalBaseEmbeddingModel):
    """DeepEval embedder that calls OpenRouter.

    DeepEval's own OpenAIEmbeddingModel only accepts plain OpenAI model names, but
    OpenRouter needs the "openai/..." prefix, so the LangChain client is wrapped instead.
    """

    def __init__(self, model: str = "openai/text-embedding-3-large"):
        self.name = model
        self.model = self.load_model()

    def load_model(self):
        return OpenAIEmbeddings(
            model=self.name,
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url=OPENROUTER_BASE_URL,
            check_embedding_ctx_length=False,  # OpenRouter takes text, not token ids
        )

    def embed_text(self, text: str) -> list[float]:
        return self.model.embed_query(text)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self.model.embed_documents(texts)

    async def a_embed_text(self, text: str) -> list[float]:
        return await self.model.aembed_query(text)

    async def a_embed_texts(self, texts: list[str]) -> list[list[float]]:
        return await self.model.aembed_documents(texts)

    def get_model_name(self) -> str:
        return self.name


def main() -> None:
    if not PDF_PATH.exists():
        raise SystemExit(f"PDF not found: {PDF_PATH}")
    if not os.getenv("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY is missing in .env")

    # Without these, DeepEval uses OpenAI directly and needs OPENAI_API_KEY.
    llm = OpenRouterModel(model="openai/gpt-4o-mini", api_key=os.getenv("OPENROUTER_API_KEY"))
    synthesizer = Synthesizer(model=llm)
    goldens = synthesizer.generate_goldens_from_docs(
        document_paths=[str(PDF_PATH)],
        context_construction_config=ContextConstructionConfig(
            embedder=OpenRouterEmbeddingModel(),
            critic_model=llm,  # grades the chunks picked as contexts
        ),
    )
    dataset = EvaluationDataset(goldens=goldens)
    print(f"Generated {len(dataset.goldens)} golden test cases from {PDF_PATH.name}.")

    # Save in the format eval_rag.py reads.
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    cases = [
        {"input": g.input, "expected_output": g.expected_output, "context": g.context or []}
        for g in dataset.goldens
    ]
    OUTPUT_JSON.write_text(json.dumps(cases, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
