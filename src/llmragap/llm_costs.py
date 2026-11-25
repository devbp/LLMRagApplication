"""Cost tracking for DeepEval model calls (test case generation and LLM-judge scoring).

DeepEval's OpenRouterModel takes each call's cost from OpenRouter's response
(usage.cost, the billed amount). TrackedOpenRouterModel adds those up so a run can
report what it cost.
"""

import threading
import uuid

from deepeval.models import OpenRouterModel

# Totals live here, keyed by tracker id, not on the model object: DeepEval may copy
# models/metrics, and every copy must still add to the same totals.
_TOTALS: dict[str, dict] = {}
_LOCK = threading.Lock()  # DeepEval runs model calls concurrently (async)


def empty_usage() -> dict:
    return {"calls": 0, "tokens": 0, "cost_usd": 0.0}


class TrackedOpenRouterModel(OpenRouterModel):
    """OpenRouterModel that adds up calls, tokens and billed cost of every generation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tracker_id = str(uuid.uuid4())
        _TOTALS[self.tracker_id] = empty_usage()

    @property
    def usage(self) -> dict:
        """{"calls", "tokens", "cost_usd"} so far (a copy)."""
        with _LOCK:
            return dict(_TOTALS[self.tracker_id])

    def calculate_cost(self, input_tokens, output_tokens, response=None):
        cost = super().calculate_cost(input_tokens, output_tokens, response=response)
        with _LOCK:
            totals = _TOTALS[self.tracker_id]
            totals["calls"] += 1
            totals["tokens"] += (input_tokens or 0) + (output_tokens or 0)
            totals["cost_usd"] += float(cost or 0.0)
        return cost
