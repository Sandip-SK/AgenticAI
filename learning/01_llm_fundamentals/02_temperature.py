
from ollama import chat

temperatures = [0.1, 0.7, 1.2]

prompt = "Explain what an AI agent is in two sentences."

for temp in temperatures:

    print(f"\nTemperature: {temp}")

    response = chat(
        model="qwen2.5:1.5b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "num_ctx": 2048,
            "temperature": temp
        }
    )

    print(response.message.content)
