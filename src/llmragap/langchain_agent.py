from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy
from pydantic import BaseModel

class Weather(BaseModel):
    temperature: float
    condition: str

def weather_tool(location: str) -> str:
    """Get the weather at a location."""
    return "Sunny and 75 degrees F."

agent = create_agent(
  model="openrouter:openai/gpt-4o-mini",

    tools=[weather_tool],
    response_format=ProviderStrategy(Weather),
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "What's the weather in SF?"}]
})
print(result["structured_response"].temperature)  # 75.0
print(result["structured_response"].condition)  # Sunny