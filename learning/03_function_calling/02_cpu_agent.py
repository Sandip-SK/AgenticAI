
import os
import json
import psutil

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Initialize Gemini through OpenAI SDK
client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


# Actual Python function
def get_cpu_usage():
    cpu = psutil.cpu_percent(interval=1)

    return {
        "cpu_usage": cpu,
        "status": "High" if cpu > 80 else "Normal"
    }


# Tool definition
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_cpu_usage",
            "description": (
                "Retrieve actual CPU utilization "
                "from the user's computer."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]


# Conversation
messages = [
    {
        "role": "system",
        "content": (
            "You are an SRE monitoring assistant. "
            "When the user asks about CPU utilization, "
            "use the get_cpu_usage tool. Never invent metrics."
        )
    },
    {
        "role": "user",
        "content": "Check my current CPU utilization."
    }
]


# First LLM request
response = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    tools=tools,
    tool_choice="auto",
    temperature=0.1
)

assistant_message = response.choices[0].message

print("LLM response received.")

# Check whether the model requested a tool
if assistant_message.tool_calls:

    messages.append(assistant_message)

    for tool_call in assistant_message.tool_calls:

        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)

        print("\nTool selected:", function_name)
        print("Arguments:", arguments)

        # Execute only approved functions
        if function_name == "get_cpu_usage":
            result = get_cpu_usage()
        else:
            raise ValueError("Unknown tool requested")

        print("Actual tool result:", result)

        # Return result to the LLM
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result)
        })

    # Second LLM request
    final_response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.1
    )

    print("\nFinal AI Agent Response:")
    print(final_response.choices[0].message.content)

else:
    print("No tool call generated.")
    print(assistant_message.content)
