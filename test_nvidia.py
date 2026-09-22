import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

api_key = os.getenv("NVIDIA_API_KEY")

llm = ChatOpenAI(
    model="moonshotai/kimi-k3",
    api_key=api_key,
    base_url="https://integrate.api.nvidia.com/v1",
    temperature=1,
    max_tokens=1024,
)

#response = llm.invoke(
#    "Explain in one sentence what an AI agent is."
#)

#print(response.content)


response = llm.stream(
    "Explain in one sentence what an AI agent is."
)

for chunk in response:
    print(chunk.content, end="", flush=True)

print()