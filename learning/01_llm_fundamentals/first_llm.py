
from ollama import chat

MODEL = "qwen2.5:1.5b"

response = chat(
    model=MODEL,
    messages=[
        {
            "role": "system",
            "content": "You are a helpful AI tutor teaching Agentic AI."
        },
        {
            "role": "user",
            "content": "Explain how an AI agent works."
        }
    ],
    options={
        "num_ctx": 2048,
        "temperature": 0.7
    }
)

print("\nAI Response:\n")
print(response.message.content)
