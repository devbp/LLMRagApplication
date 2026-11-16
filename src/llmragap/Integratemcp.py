"""LangChain agent for searching and booking hotels via the Booking.com MCP server.

The server (Booking.com Demand API MCP) only accepts approved partners. Each
request carries two headers (see the Booking.com MCP implementation guide):

    x-booking-affiliate-id   your affiliate ID (digits only)
    x-booking-authorization  your Demand API key, unchanged (no "Bearer " prefix)

Put these in .env:

    BOOKING_AFFILIATE_ID=1234567
    BOOKING_API_KEY=...
    OPENROUTER_API_KEY=...

Run:
    python src/llmragap/Integratemcp.py

Any tool that creates, changes or cancels a reservation pauses for your
approval in the terminal before it runs.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import uuid

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

load_dotenv()

MCP_URL = os.getenv("BOOKING_MCP_URL", "https://demandapi-mcp.booking.com/v1/sp/mcp/")
MODEL = os.getenv("BOOKING_AGENT_MODEL", "openrouter:openai/gpt-4o-mini")

# Tool names matching this pattern change real reservations and need approval.
WRITE_TOOL_PATTERN = re.compile(
    r"book|order|reserv|create|cancel|modify|update|confirm|pay|checkout",
    re.IGNORECASE,
)

SYSTEM_PROMPT = """You are a hotel booking assistant connected to Booking.com.

Workflow:
1. Collect what you need to search: destination, check-in and check-out dates
   (YYYY-MM-DD), number of adults/children and rooms. Ask for anything missing.
2. Search for available hotels and present the best options as a short list:
   name, area, review score, total price with currency, and cancellation policy.
3. When the user picks a hotel/room, re-check availability and the final price.
4. Before booking, collect the guest details the booking tool requires
   (name, email, phone, etc.) and show a summary: hotel, room, dates, guests,
   total price and cancellation terms.
5. Only call a booking tool after the user has explicitly said to book.

Never invent hotels, prices or booking references; use only what the tools
return. If a tool fails, explain the error plainly and suggest a next step.
"""


def _mcp_headers() -> dict[str, str]:
    affiliate_id = os.getenv("BOOKING_AFFILIATE_ID", "").strip()
    api_key = os.getenv("BOOKING_API_KEY", "").strip()
    if not affiliate_id or not api_key:
        sys.exit(
            "Missing credentials: set BOOKING_AFFILIATE_ID and BOOKING_API_KEY in .env "
            "(Booking.com MCP access is limited to approved Demand API partners)."
        )
    if not affiliate_id.isdigit():
        sys.exit("BOOKING_AFFILIATE_ID must contain digits only.")
    return {
        "x-booking-affiliate-id": affiliate_id,
        "x-booking-authorization": api_key,
    }


async def build_agent():
    """Connect to the MCP server, load its tools and build the agent."""
    client = MultiServerMCPClient(
        {
            "booking": {
                "transport": "streamable_http",
                "url": MCP_URL,
                "headers": _mcp_headers(),
            }
        }
    )
    tools = await client.get_tools()
    if not tools:
        sys.exit("The Booking.com MCP server returned no tools.")

    write_tools = [t.name for t in tools if WRITE_TOOL_PATTERN.search(t.name)]
    print(f"Loaded {len(tools)} tools: {', '.join(t.name for t in tools)}")
    if write_tools:
        print(f"Approval required for: {', '.join(write_tools)}")

    middleware = []
    if write_tools:
        middleware.append(
            HumanInTheLoopMiddleware(
                interrupt_on={
                    name: {"allowed_decisions": ["approve", "reject"]}
                    for name in write_tools
                },
                description_prefix="This action will change a real Booking.com reservation",
            )
        )

    agent = create_agent(
        model=MODEL,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        middleware=middleware,
        checkpointer=InMemorySaver(),
    )
    return agent


def _ask_decisions(action_requests: list[dict]) -> list[dict]:
    """Ask the user to approve or reject each pending write action."""
    decisions = []
    for req in action_requests:
        print("\n--- Approval needed ---")
        print(f"Tool: {req['name']}")
        print(json.dumps(req.get("args", {}), indent=2, ensure_ascii=False))
        answer = input("Approve? [y/N]: ").strip().lower()
        if answer in ("y", "yes"):
            decisions.append({"type": "approve"})
        else:
            reason = input("Reason (optional): ").strip() or "User declined."
            decisions.append({"type": "reject", "message": reason})
    return decisions


async def run_turn(agent, payload, config) -> None:
    """Run one turn, resolving approval interrupts until the agent replies."""
    result = await agent.ainvoke(payload, config)
    while result.get("__interrupt__"):
        decisions = []
        for interrupt in result["__interrupt__"]:
            decisions.extend(_ask_decisions(interrupt.value["action_requests"]))
        result = await agent.ainvoke(Command(resume={"decisions": decisions}), config)
    print(f"\nAgent: {result['messages'][-1].content}\n")


async def main() -> None:
    agent = await build_agent()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    print("\nHotel booking agent ready. Type 'exit' to quit.\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue
        try:
            await run_turn(
                agent, {"messages": [{"role": "user", "content": user_input}]}, config
            )
        except Exception as exc:  # keep the session alive on tool/network errors
            print(f"\nError: {exc}\n")


if __name__ == "__main__":
    asyncio.run(main())
