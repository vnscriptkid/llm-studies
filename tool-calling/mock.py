import json
import re
from typing import Any, Dict, List, Optional

# -----------------------------
# (a) Mock tools (business-meaningful)
# -----------------------------

MOCK_DRIVE = [
    {
        "id": "file_001",
        "title": "AcmeCo_Q4_2025_Financial_Report",
        "content": """
ACMECO - Q4 2025 FINANCIAL REPORT
Revenue: $128.4M (YoY +22%)
Gross Margin: 61% (up from 58%)
Operating Income: $18.2M (YoY +35%)
Key Drivers:
- Enterprise subscriptions grew +30%
- Churn improved from 4.2% to 3.5%
Risks:
- EU expansion delayed by regulatory review
Outlook:
- Q1 2026 revenue guidance: $130M - $136M
        """.strip()
    },
    {
        "id": "file_002",
        "title": "AcmeCo_Internal_Weekly_Status",
        "content": "This week: infra upgrades, some hiring updates, blah blah..."
    },
]

def search_google_drive(query: str) -> Dict[str, Any]:
    """Mock: returns matching files (like Drive search)."""
    q = query.lower()
    matches = []
    for f in MOCK_DRIVE:
        if q in f["title"].lower() or q in f["content"].lower():
            matches.append({"id": f["id"], "title": f["title"], "content": f["content"]})
    return {"query": query, "count": len(matches), "files": matches}

def summarize_financial_report(text: str) -> str:
    """Mock: returns a summary (pretend this is an LLM summarizer tool)."""
    # In reality you'd do proper parsing; here we just extract a few lines.
    def pick(label: str) -> Optional[str]:
        m = re.search(rf"{re.escape(label)}\s*:\s*(.*)", text)
        return m.group(1).strip() if m else None

    revenue = pick("Revenue")
    gm = pick("Gross Margin")
    op = pick("Operating Income")
    guidance = pick("Q1 2026 revenue guidance")

    bullets = []
    if revenue: bullets.append(f"- Revenue: {revenue}")
    if gm: bullets.append(f"- Gross margin: {gm}")
    if op: bullets.append(f"- Operating income: {op}")
    if guidance: bullets.append(f"- Q1 guidance: {guidance}")

    # Add “business” sections if present
    if "Key Drivers" in text:
        bullets.append("- Key drivers: Enterprise subscriptions growth, improved churn")
    if "Risks" in text:
        bullets.append("- Risks: EU expansion delay (regulatory review)")

    return "Summary:\n" + "\n".join(bullets)

def send_discord_message(channel_id: str, message: str) -> Dict[str, Any]:
    """Mock: confirms sending."""
    return {
        "ok": True,
        "channel_id": channel_id,
        "preview": message[:140] + ("..." if len(message) > 140 else "")
    }


# -----------------------------
# (b) Tool schemas (JSON-Schema-ish)
# -----------------------------

TOOL_SCHEMAS = [
    {
        "name": "search_google_drive",
        "description": "Search company Google Drive for documents by keyword and return matching file contents.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query, e.g. 'Q4 2025 financial report'."}
            },
            "required": ["query"]
        }
    },
    {
        "name": "summarize_financial_report",
        "description": "Summarize a financial report into executive bullets (revenue, margin, income, guidance, risks).",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Full report text to summarize."}
            },
            "required": ["text"]
        }
    },
    {
        "name": "send_discord_message",
        "description": "Send a message to a Discord channel.",
        "parameters": {
            "type": "object",
            "properties": {
                "channel_id": {"type": "string", "description": "Discord channel id, e.g. 'finance'."},
                "message": {"type": "string", "description": "Message body to send."}
            },
            "required": ["channel_id", "message"]
        }
    }
]


# -----------------------------
# (c) Registry
# -----------------------------

TOOLS_BY_NAME = {
    "search_google_drive": search_google_drive,
    "summarize_financial_report": summarize_financial_report,
    "send_discord_message": send_discord_message,
}

TOOLS = {
    schema["name"]: {"schema": schema, "handler": TOOLS_BY_NAME[schema["name"]]}
    for schema in TOOL_SCHEMAS
}

TOOLS_SCHEMA = [t["schema"] for t in TOOLS.values()]


# -----------------------------
# (d) System prompt & envelope
# -----------------------------

SYSTEM_PROMPT = """
You are a helpful assistant that can call tools.
When you need to use a tool, respond ONLY with:

<tool_call>
{"name": "tool_name", "args": {...}}
</tool_call>

Do not add extra text outside the tags.
""".strip()


# -----------------------------
# (e) Parse tool calls & execute
# -----------------------------

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

def extract_tool_call(text: str) -> Optional[Dict[str, Any]]:
    m = TOOL_CALL_RE.search(text)
    if not m:
        return None
    payload = m.group(1)
    return json.loads(payload)

def validate_args(schema: Dict[str, Any], args: Dict[str, Any]) -> None:
    required = schema["parameters"].get("required", [])
    for r in required:
        if r not in args:
            raise ValueError(f"Missing required arg '{r}' for tool '{schema['name']}'")

def run_tool(tool_name: str, args: Dict[str, Any]) -> Any:
    if tool_name not in TOOLS:
        raise ValueError(f"Unknown tool '{tool_name}'")
    schema = TOOLS[tool_name]["schema"]
    validate_args(schema, args)
    handler = TOOLS[tool_name]["handler"]
    return handler(**args)


# -----------------------------
# Mock "LLM" that emits tool calls (so you see the mechanics)
# Replace this later with a real model call.
# -----------------------------

def mock_llm(messages: List[Dict[str, str]]) -> str:
    """
    Very simple scripted behavior:
    - If user asks to find report -> call search_google_drive
    - If tool returned files -> call summarize_financial_report on first match
    - If tool returned summary -> call send_discord_message
    - Otherwise respond normally (no tool call)
    """
    last = messages[-1]
    role, content = last["role"], last["content"]

    # If user request
    if role == "user":
        return """
<tool_call>
{"name":"search_google_drive","args":{"query":"Q4 2025 financial report"}}
</tool_call>
""".strip()

    # If tool result from search
    if role == "tool" and last.get("name") == "search_google_drive":
        data = json.loads(content)
        if data["count"] == 0:
            return "No report found."
        report_text = data["files"][0]["content"]
        return f"""
<tool_call>
{{"name":"summarize_financial_report","args":{{"text":{json.dumps(report_text)}}}}}
</tool_call>
""".strip()

    # If tool result from summarizer
    if role == "tool" and last.get("name") == "summarize_financial_report":
        summary = content
        return f"""
<tool_call>
{{"name":"send_discord_message","args":{{"channel_id":"finance","message":{json.dumps("📌 Q4 2025 Report Summary\\n" + summary)}}}}}
</tool_call>
""".strip()

    # If tool result from discord
    if role == "tool" and last.get("name") == "send_discord_message":
        data = json.loads(content)
        return f"Sent! Preview:\n{data['preview']}"

    return "Done."


def agent_loop(user_request: str, max_steps: int = 6) -> str:
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_request},
    ]

    for _ in range(max_steps):
        # LLM proposes either a tool call or final answer
        model_text = mock_llm(messages)

        tool_call = extract_tool_call(model_text)
        if not tool_call:
            # Final answer
            return model_text

        tool_name = tool_call["name"]
        args = tool_call.get("args", {})

        # Execute tool
        result = run_tool(tool_name, args)

        # Append tool result back (this is the core mechanic)
        messages.append({
            "role": "tool",
            "name": tool_name,
            "content": json.dumps(result) if not isinstance(result, str) else result
        })

    return "Stopped: reached max_steps"


if __name__ == "__main__":
    print(agent_loop("Find the latest Q4 report in Drive, summarize it, and post to Discord."))
