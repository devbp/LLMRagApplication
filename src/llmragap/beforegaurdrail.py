import re
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langgraph.runtime import Runtime


class ContentFilterMiddleware(AgentMiddleware):
    """Deterministic guardrail: Block requests containing banned keywords.

    Checks the LATEST user message: the chat sends the whole history on every turn,
    so checking only the first message would let a banned word in a later question
    through. Keywords match whole words only ("hack" blocks "hack a server" but not
    "shackle"; "exploit" does not block "exploitation").
    """

    def __init__(self, banned_keywords: list[str]):
        super().__init__()
        self.banned_keywords = [kw.lower() for kw in banned_keywords]
        self.pattern = re.compile(
            r"\b(?:" + "|".join(re.escape(kw) for kw in self.banned_keywords) + r")\b", re.IGNORECASE
        )

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        # Get the latest user message
        last_question = next((m for m in reversed(state["messages"]) if m.type == "human"), None)
        if last_question is None:
            return None

        # Check for banned keywords (whole words)
        if self.pattern.search(last_question.text):
            # Block execution before any processing
            return {
                "messages": [{
                    "role": "assistant",
                    "content": "I cannot process requests containing inappropriate content. Please rephrase your request."
                }],
                "jump_to": "end"
            }

        return None
