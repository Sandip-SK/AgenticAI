
from ollama import chat

messages = [
    {
        "role": "system",
        "content": """
        You are a senior Site Reliability Engineer.
        Explain technical concepts using:
        1. Simple language
        2. Practical examples
        3. Production scenarios
        4. Interview-focused explanations
        """
    },
    {
        "role": "user",
        "content": "Explain Kubernetes liveness probes."
    }
]

response = chat(
    model="qwen2.5:1.5b",
    messages=messages,
    options={
        "num_ctx": 2048,
        "temperature": 0.2
    }
)

print(response.message.content)
