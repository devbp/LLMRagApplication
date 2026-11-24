# Tested with: langchain==1.2.17, langchain-core==1.3.2, langchain-openai==1.2.1, langgraph==1.1.10
import os
from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
# Create persistent checkpointer
conn = sqlite3.connect("checkpoints.db", check_same_thread=False)

# Define tools - same pattern as before!
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # In production, this would call a real weather API
    return f"The weather in {city} is sunny, 72°F"

@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression safely."""
    try:
        # Note: In production, use a safer eval alternative
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

# Create model (API key from environment)
model = ChatOpenAI(
    model="openai/gpt-6-luna",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

# Create checkpointer for memory
#checkpointer = InMemorySaver()
checkpointer = SqliteSaver(conn)

# Create agent - this handles all the tool calling logic!
agent = create_agent(
    model=model,
    tools=[get_weather, calculate],
    system_prompt="You are a helpful assistant that can check weather and do calculations.",
    checkpointer=checkpointer,
)

# Use the agent with thread-based memory
config = {"configurable": {"thread_id": "conversation-1"}}

# First message
response = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in Paris?"}]},
    config=config
)
print(response["messages"][-1].content)

# Continue the conversation - the agent remembers context
response = agent.invoke(
    {"messages": [{"role": "user", "content": "Now calculate 25 * 4"}]},
    config=config
)
print(response["messages"][-1].content)

# Different thread - fresh conversation
config2 = {"configurable": {"thread_id": "conversation-2"}}
response = agent.invoke(
    {"messages": [{"role": "user", "content": "Hello!"}]},
    config2
)
print(response["messages"][-1].content)