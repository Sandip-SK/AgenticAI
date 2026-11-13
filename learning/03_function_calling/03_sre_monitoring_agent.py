
import os
import json
import psutil

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


# -------------------------------
# TOOL IMPLEMENTATIONS
# -------------------------------

def get_cpu_usage():
    cpu = psutil.cpu_percent(interval=1)

    return {
        "cpu_usage_percent": cpu,
        "status": "High" if cpu > 80 else "Normal"
    }


def get_memory_usage():
    memory = psutil.virtual_memory()

    return {
        "total_gb": round(memory.total / (1024 ** 3), 2),
        "used_gb": round(memory.used / (1024 ** 3), 2),
        "usage_percent": memory.percent,
        "status": "High" if memory.percent > 85 else "Normal"
    }


def get_disk_usage():
    disk = psutil.disk_usage("C:\\")

    return {
        "total_gb": round(disk.total / (1024 ** 3), 2),
        "used_gb": round(disk.used / (1024 ** 3), 2),
        "free_gb": round(disk.free / (1024 ** 3), 2),
        "usage_percent": disk.percent,
        "status": "High" if disk.percent > 85 else "Normal"
    }


# -------------------------------
# TOOL DEFINITIONS
# -------------------------------

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_cpu_usage",
            "description": "Get current CPU utilization.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_memory_usage",
            "description": "Get current RAM utilization.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_disk_usage",
            "description": "Get disk utilization of the C drive.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]


# -------------------------------
# TOOL REGISTRY
# -------------------------------

available_functions = {
    "get_cpu_usage": get_cpu_usage,
    "get_memory_usage": get_memory_usage,
    "get_disk_usage": get_disk_usage
}


# -------------------------------
# AGENT EXECUTION
# -------------------------------

user_input = input("Ask your SRE agent: ")

messages = [
    {
        "role": "system",
        "content": """
        You are an SRE monitoring assistant.

        Use the available tools to retrieve actual system metrics.
        Never invent monitoring data.
        Only report metrics returned by tools.
        Explain potential resource pressure clearly.
        """
    },
    {
        "role": "user",
        "content": user_input
    }
]

response = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    tools=tools,
    tool_choice="auto",
    temperature=0.1
)

assistant_message = response.choices[0].message

messages.append(assistant_message)

# Execute requested tools
if assistant_message.tool_calls:

    for tool_call in assistant_message.tool_calls:

        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)

        print(f"\nExecuting: {function_name}")

        if function_name not in available_functions:
            raise ValueError("Unauthorized tool requested")

        try:
            result = available_functions[function_name](**arguments)
        except Exception as error:
            result = {
                "error": str(error)
            }

        print("Tool result:", result)

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result)
        })

    # Send tool results back to Gemini
    final_response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.1
    )

    print("\nAI Agent Analysis:")
    print(final_response.choices[0].message.content)

else:
    print("\nAgent response:")
    print(assistant_message.content)
