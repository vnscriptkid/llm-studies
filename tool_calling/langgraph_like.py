import inspect
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, get_origin, get_args

# -----------------------------
# Tiny framework: ToolFunction + registry + @tool decorator
# -----------------------------

TOOLS_REGISTRY: List["ToolFunction"] = []

def _pytype_to_jsonschema(t: Any) -> Dict[str, Any]:
    """
    Very small mapping from Python annotations to JSON schema types.
    Extend as needed (arrays, objects, enums, etc).
    """
    if t is inspect._empty:
        return {"type": "string"}  # default fallback

    origin = get_origin(t)
    if origin is list or origin is List:
        return {"type": "array", "items": _pytype_to_jsonschema(get_args(t)[0] if get_args(t) else str)}
    if origin is dict or origin is Dict:
        return {"type": "object"}
    if origin is Optional:
        # Optional[T] is tricky; in typing it appears as Union[T, None]
        return {"type": "string"}  # keep simple

    if t in (str,):
        return {"type": "string"}
    if t in (int,):
        return {"type": "integer"}
    if t in (float,):
        return {"type": "number"}
    if t in (bool,):
        return {"type": "boolean"}

    # fallback for unknown types
    return {"type": "string"}


@dataclass
class ToolFunction:
    func: Callable[..., Any]
    name: str
    description: str
    schema: Dict[str, Any]

    def __call__(self, **kwargs) -> Any:
        return self.func(**kwargs)


def tool(name: Optional[str] = None):
    """
    Decorator that:
    - inspects signature & type annotations
    - marks params with no default as required
    - uses docstring (first line or full) as description
    - builds a schema automatically
    - registers ToolFunction globally
    """
    def decorator(func: Callable[..., Any]) -> ToolFunction:
        sig = inspect.signature(func)
        func_name = name or func.__name__
        doc = inspect.getdoc(func) or ""
        desc = doc.strip() if doc.strip() else f"Tool: {func_name}"

        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            ann = param.annotation
            properties[param_name] = {
                **_pytype_to_jsonschema(ann),
                "description": ""  # could be improved by parsing docstring, Google-style docs, etc.
            }
            if param.default is inspect._empty:
                required.append(param_name)

        schema = {
            "name": func_name,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            }
        }

        tf = ToolFunction(func=func, name=func_name, description=desc, schema=schema)
        TOOLS_REGISTRY.append(tf)
        return tf

    return decorator


def tools_by_name() -> Dict[str, Callable[..., Any]]:
    return {t.name: t for t in TOOLS_REGISTRY}

def tools_schema() -> List[Dict[str, Any]]:
    return [t.schema for t in TOOLS_REGISTRY]

# -----------------------------
# Mock data
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
        "content": "This week: infra upgrades, some hiring updates..."
    },
]

@tool()
def search_google_drive(query: str) -> Dict[str, Any]:
    """Search company Google Drive for documents by keyword and return matching file contents."""
    q = query.lower()
    matches = []
    for f in MOCK_DRIVE:
        if q in f["title"].lower() or q in f["content"].lower():
            matches.append({"id": f["id"], "title": f["title"], "content": f["content"]})
    return {"query": query, "count": len(matches), "files": matches}


@tool()
def summarize_financial_report(text: str) -> str:
    """Summarize a financial report into executive bullets (revenue, margin, income, guidance, risks)."""
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

    if "Churn improved" in text:
        bullets.append("- Retention: churn improved (4.2% → 3.5%)")
    if "Risks" in text:
        bullets.append("- Risks: EU expansion delay (regulatory review)")

    return "Summary:\n" + "\n".join(bullets)


@tool()
def send_discord_message(channel_id: str, message: str) -> Dict[str, Any]:
    """Send a message to a Discord channel."""
    return {
        "ok": True,
        "channel_id": channel_id,
        "preview": message[:140] + ("..." if len(message) > 140 else "")
    }

if __name__ == "__main__":
    for tool in TOOLS_REGISTRY:
        print("=" * 10)
        print(tool.name)
        print(tool.description)
        import pprint
        pprint.pprint(tool.schema)