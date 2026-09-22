import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage

load_dotenv()

api_key = os.getenv("NVIDIA_API_KEY")


@tool
def get_stock_price(ticker: str) -> str:
    """Get the current stock price for a given ticker."""
    return f"The current stock price for {ticker} is $123.45."


llm = ChatOpenAI(
    model="moonshotai/kimi-k3",
    api_key=api_key,
    base_url="https://integrate.api.nvidia.com/v1",
    temperature=1,
    max_tokens=1024,
)

llm_with_tools = llm.bind_tools([get_stock_price])

messages = [
    ("user", "What is the current stock price of AAPL?")
]

# 1. Ask the model
response = llm_with_tools.invoke(messages)

print("MODEL RESPONSE:")
print(response)

# 2. Execute the requested tools
tool_messages = []

for tool_call in response.tool_calls:
    if tool_call["name"] == "get_stock_price":
        result = get_stock_price.invoke(tool_call["args"])

        tool_messages.append(
            ToolMessage(
                content=result,
                tool_call_id=tool_call["id"],
            )
        )

# 3. Send the tool result back to the model
messages = messages + [response] + tool_messages

final_response = llm_with_tools.invoke(messages)

print("\nFINAL RESPONSE:")
print(final_response.content)