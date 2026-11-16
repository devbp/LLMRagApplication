import asyncio
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()


async def main():
    client = MultiServerMCPClient(
        {
            "hotels": {
                "url": "https://mcp.trivago.com/mcp",
                "transport": "http",
                # Trivago doesn't support the session-close request (HTTP 501),
                # so skip it to avoid the "Session termination failed" message.
                "terminate_on_close": False,
            },

            "flights": {
                "url": "https://mcp.airtreks.com/mcp",
                "transport": "http",
               
                #"terminate_on_close": False,
            }
        }
        
    )
    tools = await client.get_tools()  # async: must be awaited
    print(f"Loaded {len(tools)} tools: {', '.join(t.name for t in tools)}")
    agent = create_agent(model="openrouter:openai/gpt-4o-mini", tools=tools)

    # MCP tools are async-only, so the agent must be run with ainvoke.
    response = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "find flight from Munich to Kathmanduon 2027-07-01, and find hotel for 1 week in Kathmandu",
                }
            ]
        }
    )
    print(response["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
