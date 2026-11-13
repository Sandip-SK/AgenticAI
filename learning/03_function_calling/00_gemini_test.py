
import os

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

response = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "system",
            "content": "You are an expert AI engineering tutor."
        },
        {
            "role": "user",
            "content": "Explain function calling in AI agents."
        }
    ],
    temperature=0.2
)

print(response.choices[0].message.content)
