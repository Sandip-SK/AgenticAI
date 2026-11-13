
import os
import json
import time
import queue
import logging
import multiprocessing
from datetime import datetime, timezone

import psutil
from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# 1. CONFIGURATION
# ============================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found. Check your .env file."
    )

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

client = OpenAI(
    api_key=API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

MAX_ITERATIONS = 5
TOOL_TIMEOUT = 5


# ============================================================
# 2. STRUCTURED LOGGING
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "agent.log")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(message)s",
    encoding="utf-8"
)

logger = logging.getLogger("sre_agent")


def log_event(event, **details):

    log_entry = {
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "event": event,
        **details
    }

    logger.info(json.dumps(log_entry, default=str))


# ============================================================
# 3. MONITORING TOOLS
# ============================================================

def get_cpu_usage():

    cpu = psutil.cpu_percent(interval=1)

    return {
        "cpu_usage_percent": cpu,
        "status": "High" if cpu >= 80 else "Normal"
    }


def get_memory_usage():

    memory = psutil.virtual_memory()

    return {
        "total_gb": round(
            memory.total / (1024 ** 3), 2
        ),
        "used_gb": round(
            memory.used / (1024 ** 3), 2
        ),
        "usage_percent": memory.percent,
        "status": "High" if memory.percent >= 80 else "Normal"
    }


def get_top_memory_processes():

    processes = []
    skipped = 0

    for process in psutil.process_iter(
        ["pid", "name", "memory_percent"]
    ):

        try:
            info = process.info

            processes.append({
                "pid": info["pid"],
                "name": info["name"],
                "memory_percent": round(
                    info["memory_percent"] or 0, 2
                )
            })

        except (
            psutil.NoSuchProcess,
            psutil.AccessDenied
        ):
            skipped += 1

    processes.sort(
        key=lambda p: p["memory_percent"],
        reverse=True
    )

    return {
        "top_processes": processes[:5],
        "processes_skipped": skipped
    }


# ============================================================
# 4. GEMINI TOOL DEFINITIONS
# ============================================================

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
            "description": "Get current RAM utilization and status.",
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
            "name": "get_top_memory_processes",
            "description": "Get the five processes consuming the most RAM.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }

]


# ============================================================
# 5. PYTHON FUNCTION REGISTRY
# ============================================================

available_functions = {

    "get_cpu_usage": get_cpu_usage,

    "get_memory_usage": get_memory_usage,

    "get_top_memory_processes": get_top_memory_processes

}


# ============================================================
# 6. TOOL ARGUMENT VALIDATION
# ============================================================

def validate_tool_arguments(function_name, raw_arguments):

    # Check tool allowlist
    if function_name not in available_functions:
        raise ValueError(
            f"Unknown tool: {function_name}"
        )

    # Parse JSON
    arguments = json.loads(raw_arguments or "{}")

    # Arguments must be a dictionary
    if not isinstance(arguments, dict):
        raise ValueError(
            "Tool arguments must be a JSON object"
        )

    # Current tools do not accept arguments
    allowed_arguments = {
        "get_cpu_usage": set(),
        "get_memory_usage": set(),
        "get_top_memory_processes": set()
    }

    allowed = allowed_arguments[function_name]

    unexpected = set(arguments.keys()) - allowed

    if unexpected:
        raise ValueError(
            f"Unexpected arguments for {function_name}: {unexpected}"
        )

    return arguments


# ============================================================
# 7. MULTIPROCESSING TOOL EXECUTION
# ============================================================

def tool_worker(result_queue, function, arguments):

    try:

        result = function(**arguments)

        result_queue.put({
            "success": True,
            "result": result
        })

    except Exception as error:

        result_queue.put({
            "success": False,
            "error": str(error)
        })


def execute_tool_with_timeout(
    function,
    arguments,
    timeout=TOOL_TIMEOUT
):

    context = multiprocessing.get_context("spawn")

    result_queue = context.Queue()

    process = context.Process(
        target=tool_worker,
        args=(
            result_queue,
            function,
            arguments
        )
    )

    try:

        process.start()

        process.join(timeout)

        # Check whether the process is still running
        if process.is_alive():

            process.terminate()
            process.join()

            # Ensure termination if necessary
            if process.is_alive():
                process.kill()
                process.join()

            raise TimeoutError(
                f"Tool exceeded {timeout} seconds and was terminated."
            )

        # Retrieve the result
        try:

            result = result_queue.get(timeout=1)

        except queue.Empty:

            raise RuntimeError(
                "Tool process exited without returning a result."
            )

        if not result["success"]:
            raise RuntimeError(result["error"])

        return result["result"]

    finally:

        # Cleanup child process
        if process.pid is not None:

            if process.is_alive():
                process.terminate()
                process.join()

            process.close()

        result_queue.close()
        result_queue.join_thread()


# ============================================================
# 8. AGENT EXECUTION REPORT
# ============================================================

def print_execution_report(agent_stats, agent_start_time):

    total_time = round(
        time.perf_counter() - agent_start_time, 2
    )

    print("\n========== AGENT EXECUTION REPORT ==========")

    print(
        f"Iterations: {agent_stats['iterations']}"
    )

    print(
        f"Tool calls: {agent_stats['tool_calls']}"
    )

    print(
        f"Tool errors: {agent_stats['tool_errors']}"
    )

    print(
        f"Total execution time: {total_time} seconds"
    )

    print(
        f"Total LLM time: "
        f"{agent_stats['llm_time_seconds']:.2f} seconds"
    )

    print("============================================")

    return total_time


# ============================================================
# 9. AUTONOMOUS AGENT LOOP
# ============================================================

def run_agent(user_query):

    agent_start_time = time.perf_counter()

    agent_stats = {
        "iterations": 0,
        "tool_calls": 0,
        "tool_errors": 0,
        "llm_time_seconds": 0
    }

    log_event("agent_started")

    messages = [

        {
            "role": "system",

            "content": """
            You are an SRE monitoring agent.

            Your responsibilities:

            1. Investigate system health using available monitoring tools.

            2. Use CPU and memory metrics to identify potential issues.

            3. If memory usage is 80% or higher,
               investigate the top memory-consuming processes.

            4. Base your conclusions on actual tool results.

            5. Never invent monitoring data.

            6. Recommend safe, read-only diagnostic actions.

            7. Never terminate processes or modify system settings.

            8. If a tool fails, acknowledge the missing evidence.
               Do not treat a failed tool as a successful result.
            """
        },

        {
            "role": "user",
            "content": user_query
        }

    ]

    # Agent decision loop
    for iteration in range(1, MAX_ITERATIONS + 1):

        agent_stats["iterations"] = iteration

        print(f"\n--- Agent Iteration {iteration} ---")

        log_event(
            "agent_iteration",
            iteration=iteration
        )

        # ----------------------------------------------------
        # Call Gemini
        # ----------------------------------------------------

        llm_start = time.perf_counter()

        try:

            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )

        except Exception as error:

            log_event(
                "llm_request_failed",
                iteration=iteration,
                error=str(error)
            )

            print(f"Gemini API request failed: {error}")

            agent_stats["tool_errors"] += 1

            break

        llm_duration = (
            time.perf_counter() - llm_start
        )

        agent_stats["llm_time_seconds"] += llm_duration

        print(
            f"Gemini API latency: {llm_duration:.2f} seconds"
        )
        log_event(
            "llm_request_completed",
            iteration=iteration,
            latency_seconds=round(llm_duration, 2)
        )

        assistant_message = response.choices[0].message

        # ----------------------------------------------------
        # Check for final response
        # ----------------------------------------------------

        if not assistant_message.tool_calls:

            print("\nFinal AI Agent Response:")

            print(
                assistant_message.content or "No response generated."
            )

            total_time = print_execution_report(
                agent_stats,
                agent_start_time
            )

            log_event(
                "agent_completed",
                iterations=agent_stats["iterations"],
                tool_calls=agent_stats["tool_calls"],
                tool_errors=agent_stats["tool_errors"],
                total_execution_time_seconds=total_time,
                total_llm_time_seconds=round(
                    agent_stats["llm_time_seconds"], 2
                )
            )

            return

        # Preserve assistant tool-call message
        messages.append(assistant_message)

        # ----------------------------------------------------
        # Execute requested tools
        # ----------------------------------------------------

        for tool_call in assistant_message.tool_calls:

            # Extract function name BEFORE using it
            function_name = tool_call.function.name

            agent_stats["tool_calls"] += 1

            print(
                f"\nExecuting tool: {function_name}"
            )

            log_event(
                "tool_started",
                tool_name=function_name,
                iteration=iteration
            )

            start_time = time.perf_counter()

            try:

                # Validate arguments
                arguments = validate_tool_arguments(
                    function_name,
                    tool_call.function.arguments
                )

                # Retrieve registered Python function
                function_to_call = available_functions[function_name]

                # Execute with timeout
                result = execute_tool_with_timeout(
                    function_to_call,
                    arguments,
                    timeout=TOOL_TIMEOUT
                )

                execution_time = round(
                    (time.perf_counter() - start_time) * 1000,
                    2
                )

                print(f"Tool result: {result}")

                print(
                    f"Execution time: {execution_time} ms"
                )

                tool_result = {
                    "success": True,
                    "data": result,
                    "execution_time_ms": execution_time
                }

                log_event(
                    "tool_completed",
                    tool_name=function_name,
                    success=True,
                    execution_time_ms=execution_time
                )

            except Exception as error:

                agent_stats["tool_errors"] += 1

                execution_time = round(
                    (time.perf_counter() - start_time) * 1000,
                    2
                )

                print(
                    f"Tool execution failed: {error}"
                )

                tool_result = {
                    "success": False,
                    "error": str(error),
                    "execution_time_ms": execution_time
                }

                log_event(
                    "tool_failed",
                    tool_name=function_name,
                    error=str(error),
                    execution_time_ms=execution_time
                )

            # ------------------------------------------------
            # Return result to Gemini
            # ------------------------------------------------

            messages.append({

                "role": "tool",

                "tool_call_id": tool_call.id,

                "content": json.dumps(tool_result)

            })

    # --------------------------------------------------------
    # Maximum iteration limit reached or API failure
    # --------------------------------------------------------

    print(
        "\nAgent stopped: Maximum iteration limit reached "
        "or an API request failed."
    )

    total_time = print_execution_report(
        agent_stats,
        agent_start_time
    )

    log_event(
        "agent_stopped",
        iterations=agent_stats["iterations"],
        tool_calls=agent_stats["tool_calls"],
        tool_errors=agent_stats["tool_errors"],
        total_execution_time_seconds=total_time,
        total_llm_time_seconds=round(
            agent_stats["llm_time_seconds"], 2
        )
    )


# ============================================================
# 10. APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":

    print("Autonomous SRE Monitoring Agent")
    print("--------------------------------")

    while True:

        query = input(
            "\nAsk your SRE agent (or type exit): "
        )

        if query.lower().strip() == "exit":

            print("Agent stopped.")
            break

        if not query.strip():
            continue

        run_agent(query)
