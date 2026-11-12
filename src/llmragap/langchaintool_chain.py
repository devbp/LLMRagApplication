from langchain_openrouter import ChatOpenRouter
import os
from pydantic import BaseModel, Field
class GetWeather(BaseModel):
    """Get the current weather for a city."""
    city: str = Field(description="City name, e.g. 'Lisbon'")
    condition: str = Field(description="Weather condition, e.g. 'Sunny'")

def weather_tool(location: str) -> str:
    """Get the weather at a location."""
    return "Sunny and 75 degrees F."
#model=ChatOpenRouter(model="anthropic/claude-sonnet-4.5", temperature=0.7, max_tokens=1000, api_key=os.getenv("OPENROUTER_API_KEY"))
model = ChatOpenRouter(
    model="openai/gpt-4o-mini",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0.7,
)

model_with_tool = model.bind_tools([weather_tool],strict=True) 
structured = model_with_tool.with_structured_output(GetWeather, method="json_schema")
result = structured.invoke("What's the weather in New York City?")
print(result.condition)  # The model's response to the user query