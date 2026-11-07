
from ollama import chat

def build_sre_prompt(metrics):

    prompt = f"""
    ROLE:
    You are a Senior Site Reliability Engineer.

    TASK:
    Investigate the production incident.

    INCIDENT METRICS:
    {metrics}

    CONSTRAINTS:
- Never invent facts or metrics.
- Clearly distinguish confirmed facts from hypotheses.
- Do not interpret timestamps as delays unless
  explicitly stated.
- Prioritize diagnostic commands over generic advice.
- Explain what each diagnostic command will verify.
- Identify missing information.
- Do not recommend rollback or pod deletion
  without sufficient evidence and approval.

  EVIDENCE REQUIREMENTS:

For every suspected root cause, provide:
1. Supporting evidence.
2. Evidence that is currently missing.
3. A diagnostic command or query to verify it.
4. The expected observation if the hypothesis is true.

    OUTPUT:
    1. Incident Overview
    2. Potential Root Causes
    3. Diagnostic Steps
    4. Business Impact
    """

    return prompt


incident_metrics = {
    "error_rate": "25%",
    "api_latency": "4 seconds",
    "database_cpu": "30%",
    "pod_restarts": "12 in 10 minutes",
    "last_deployment": "15 minutes ago"
}

prompt = build_sre_prompt(incident_metrics)

response = chat(
    model="qwen2.5:1.5b",
    messages=[
        {"role": "user", "content": prompt}
    ],
    options={
        "temperature": 0.2,
        "num_ctx": 2048
    }
)

print(response.message.content)
