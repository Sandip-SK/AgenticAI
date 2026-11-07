
from ollama import chat

response = chat(
    model="qwen2.5:1.5b",
    messages=[
        {
            "role": "system",
            "content": """
            You are an SRE incident analysis assistant.
            Analyze the incident and return structured JSON.
            """
        },
        {
            "role": "user",
            "content": """
            Incident:
            Kubernetes pod is restarting repeatedly.

            Container exit code: 137.
            Memory limit: 512Mi.
            Previous memory usage: 510Mi.

            Identify the likely issue and recommend
            diagnostic actions.
            """
        }
    ],
    format="json",
    options={
        "num_ctx": 2048,
        "temperature": 0.1
    }
)

print(response.message.content)
