
from ollama import chat

response = chat(
    model="qwen2.5:1.5b",
    messages=[
        {
            "role": "user",
            "content": "Explain Kubernetes in 5 bullet points."
        }
    ],
    options={
        "num_ctx": 2048,
        "temperature": 0.2
    }
)

print(response.message.content)
