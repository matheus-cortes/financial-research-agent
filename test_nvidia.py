import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()


llm = ChatOpenAI(
    model="moonshotai/kimi-k3",
    api_key=os.getenv("NVIDIA_API_KEY"),
    base_url="https://integrate.api.nvidia.com/v1",
    temperature=1,
    max_tokens=2048,
)


json_llm = llm.with_structured_output(
    method="json_mode",
)


result = json_llm.invoke(
    """
    Return ONLY valid JSON.

    Use exactly this structure:

    {
        "area": "test",
        "agent": "test_agent",
        "findings": [
            "test finding"
        ],
        "sources": [
            "test source"
        ]
    }
    """
)


print("========== RESULT ==========")
print(result)

print("\n========== TYPE ==========")
print(type(result))
