"""Evaluate the law RAG agent from RAG_Pdf.py with DeepEval.

Test cases are read from a JSON file containing a list of objects:
    [
      {"input": "the question to ask the agent",
       "expected_output": "the correct answer, taken from the law PDFs"},
      ...
    ]

Metrics (judged by an LLM through OpenRouter):
  - AnswerRelevancyMetric: does the answer address the question?
  - FaithfulnessMetric:    is every claim in the answer supported by the retrieved chunks?

Results are written to a JSON file with a summary (pass count, average scores)
and, per test case, the input, expected output, actual output, retrieved chunks,
and each metric's score, pass/fail and reason.

Run:
    uv run python src/llmragap/eval_rag.py
    uv run python src/llmragap/eval_rag.py --mode keyword
    uv run python src/llmragap/eval_rag.py --input eval_data/my_cases.json --output eval_data/my_results.json
"""

import argparse
import json
import os
from datetime import datetime

from deepeval import evaluate
from deepeval.dataset import Golden
from deepeval.evaluate.configs import CacheConfig
from deepeval.evaluate.types import EvaluationResult
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.models import OpenRouterModel
from deepeval.test_case import LLMTestCase
from langchain_core.messages import ToolMessage

from llmragap.RAG_Pdf import (
    CIVIL_CODE_COLLECTION,
    CIVIL_CODE_PDF,
    DEFAULT_SELECTION_MODE,
    INSURANCE_COLLECTION,
    INSURANCE_PDF,
    SEARCH_MODES,
    FilesEmbeddings,
    RagAgent,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_INPUT_JSON = os.path.join(PROJECT_ROOT, "eval_data", "law_test_cases.json")
DEFAULT_OUTPUT_JSON = os.path.join(PROJECT_ROOT, "eval_data", "law_eval_results.json")


class RagEvaluator:
    """Runs JSON test cases through a RagAgent, scores them with DeepEval and saves the results."""

    # format_results() in RAG_Pdf.py joins chunks with this separator.
    CHUNK_SEPARATOR = "\n\n---\n\n"
    NO_RESULTS = "No matching passages found."

    def __init__(
        self,
        rag_agent: RagAgent,
        judge_model: str = "openai/gpt-6-luna",
        threshold: float = 0.7,
    ):
        self.rag_agent = rag_agent
        # The model that grades the answers (not the one being tested).
        self.judge = OpenRouterModel(model=judge_model, api_key=os.getenv("OPENROUTER_API_KEY"))
        self.metrics = [
            AnswerRelevancyMetric(threshold=threshold, model=self.judge),
            FaithfulnessMetric(threshold=threshold, model=self.judge),
        ]

    @staticmethod
    def load_goldens(json_path: str) -> list[Golden]:
        """Read test cases from a JSON list of {"input": ..., "expected_output": ...} objects."""
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"{json_path} must contain a JSON list of test cases")

        goldens = []
        for i, case in enumerate(data, start=1):
            missing = {"input", "expected_output"} - set(case)
            if missing:
                raise ValueError(f"Test case {i} in {json_path} is missing: {', '.join(sorted(missing))}")
            question = str(case["input"]).strip()
            if not question:  # skip empty test cases
                continue
            goldens.append(Golden(input=question, expected_output=str(case["expected_output"]).strip()))

        if not goldens:
            raise ValueError(f"No test cases found in {json_path}")
        print(f"Loaded {len(goldens)} test cases from {json_path}")
        return goldens

    def run_pipeline(self, question: str) -> tuple[str, list[str]]:
        """Ask the agent a question and return (answer, retrieved_chunks).

        retrieved_chunks are the passages the agent's search tools returned, which is
        what FaithfulnessMetric checks the answer against.
        """
        messages = self.rag_agent.invoke(question)["messages"]
        answer = messages[-1].text
        retrieved_chunks = [
            chunk
            for msg in messages
            if isinstance(msg, ToolMessage)
            for chunk in msg.text.split(self.CHUNK_SEPARATOR)
            if chunk.strip() and chunk.strip() != self.NO_RESULTS
        ]
        return answer, retrieved_chunks

    def build_test_cases(self, goldens: list[Golden]) -> list[LLMTestCase]:
        """Run every golden through the agent and turn it into a DeepEval test case."""
        test_cases = []
        for i, golden in enumerate(goldens, start=1):
            print(f"[{i}/{len(goldens)}] {golden.input}")
            answer, retrieved_chunks = self.run_pipeline(golden.input)
            test_cases.append(
                LLMTestCase(
                    input=golden.input,
                    actual_output=answer,
                    expected_output=golden.expected_output,
                    retrieval_context=retrieved_chunks,
                )
            )
        return test_cases

    def evaluate(self, test_cases: list[LLMTestCase]) -> EvaluationResult:
        """Score the test cases with the configured metrics.

        DeepEval's result cache is turned off: on Windows it often fails to lock
        .deepeval/.deepeval-cache.json, which crashes with
        "'NoneType' object has no attribute 'test_cases_lookup_map'".
        """
        return evaluate(
            test_cases=test_cases,
            metrics=self.metrics,
            cache_config=CacheConfig(write_cache=False, use_cache=False),
        )

    def save_results(self, result: EvaluationResult, output_path: str) -> dict:
        """Write a JSON report: a summary, then each test case with its answers, chunks and metrics."""
        # Keep the results in the same order as the input file.
        test_results = sorted(
            result.test_results,
            key=lambda r: r.index if r.index is not None else 0,
        )
        metric_names = [m.__name__ for m in self.metrics]

        cases = []
        for r in test_results:
            metrics = {
                m.name: {
                    "score": round(m.score, 3) if m.score is not None else None,
                    "threshold": m.threshold,
                    "passed": m.success,
                    "reason": m.reason,
                    "error": m.error,
                }
                for m in r.metrics_data or []
            }
            cases.append(
                {
                    "input": r.input,
                    "expected_output": r.expected_output,
                    "actual_output": r.actual_output,
                    "retrieval_context": r.retrieval_context or [],
                    "passed": r.success,
                    "metrics": metrics,
                }
            )

        def average(name: str) -> float | None:
            scores = [c["metrics"][name]["score"] for c in cases
                      if name in c["metrics"] and c["metrics"][name]["score"] is not None]
            return round(sum(scores) / len(scores), 3) if scores else None

        passed = sum(1 for c in cases if c["passed"])
        report = {
            "summary": {
                "run_at": datetime.now().isoformat(timespec="seconds"),
                "selection_mode": self.rag_agent.selection_mode,
                "model": self.rag_agent.model,
                "judge_model": self.judge.get_model_name(),
                "total": len(cases),
                "passed": passed,
                "failed": len(cases) - passed,
                "average_scores": {name: average(name) for name in metric_names},
            },
            "results": cases,
        }

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            # ensure_ascii=False keeps characters like § readable in the file.
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"Saved results to {output_path} ({passed}/{len(cases)} test cases passed)")
        print("Average scores:", report["summary"]["average_scores"])
        return report

    def run(self, input_json: str, output_json: str) -> dict:
        """Load test cases, run the agent, score the answers and save the results."""
        goldens = self.load_goldens(input_json)
        test_cases = self.build_test_cases(goldens)
        result = self.evaluate(test_cases)
        return self.save_results(result, output_json)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the law RAG agent with DeepEval.")
    parser.add_argument("--mode", choices=SEARCH_MODES, default=DEFAULT_SELECTION_MODE, help="search mode")
    parser.add_argument("--input", default=DEFAULT_INPUT_JSON, help="JSON list of {input, expected_output} test cases")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_JSON, help="where to save the results JSON")
    args = parser.parse_args()

    # Make sure both law files are in Chroma (skipped if already embedded).
    files = FilesEmbeddings()
    #files.ensure_embedded(INSURANCE_COLLECTION, INSURANCE_PDF)
    #files.ensure_embedded(CIVIL_CODE_COLLECTION, CIVIL_CODE_PDF)

    rag_agent = RagAgent(files, args.mode)
    print(f"Evaluating with {args.mode} search.")

    RagEvaluator(rag_agent).run(args.input, args.output)


if __name__ == "__main__":
    main()
