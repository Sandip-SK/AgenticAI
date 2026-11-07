
from ollama import chat

response = chat(
    model="qwen2.5:1.5b",
    messages=[
        {
            "role": "user",
            "content": "Explain how Kubernetes pods work."
        }
    ],
    options={
        "num_ctx": 2048
    }
)

print("Response:")
print(response.message.content)

print("\nToken Statistics:")
print("Input tokens:", response.prompt_eval_count)
print("Output tokens:", response.eval_count)
