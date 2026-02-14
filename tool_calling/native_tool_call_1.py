# pip install google-genai
import os
import json
from typing import Any, Dict, Callable

try:
    from dotenv import load_dotenv
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: python-dotenv. Install with `python3 -m pip install python-dotenv` "
        "or `pip install -r requirements.txt`."
    ) from exc

try:
    from google import genai
    from google.genai import types
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: google-genai. Install with `python3 -m pip install google-genai` "
        "or `pip install -r requirements.txt`."
    ) from exc

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise SystemExit("Missing GEMINI_API_KEY. Add it to your environment or `.env` file.")

client = genai.Client(api_key=api_key)


# ----------------------------
# 1) Real tool implementations
# ----------------------------
def get_weather(city: str, unit: str = "C") -> Dict[str, Any]:
    # Demo stub (replace with real weather API call)
    fake = {"city": city, "unit": unit, "temp": 31, "condition": "Partly cloudy"}
    return fake


def convert_currency(amount: float, from_: str, to: str) -> Dict[str, Any]:
    # Demo stub (replace with real FX API)
    rate = 1.35  # pretend
    return {"amount": amount, "from": from_, "to": to, "rate": rate, "converted": amount * rate}


TOOL_REGISTRY: Dict[str, Callable[..., Any]] = {
    "get_weather": get_weather,
    "convert_currency": convert_currency,
}


# ----------------------------
# 2) Dispatcher helper
# ----------------------------
def dispatch_function_call(function_call: types.FunctionCall) -> Dict[str, Any]:
    """
    Takes Gemini's structured FunctionCall and executes the mapped Python function.
    """
    name = function_call.name
    args = dict(function_call.args or {})

    if name not in TOOL_REGISTRY:
        raise ValueError(f"Unknown tool: {name}")

    fn = TOOL_REGISTRY[name]

    # Normalize SDK/model argument aliases to our Python function signature.
    if name == "convert_currency":
        alias_map = {
            "from": "from_",
            "from_": "from_",
            "from1_": "from_",
            "to": "to",
            "to_": "to",
        }
        normalized: Dict[str, Any] = {}
        for key, value in args.items():
            normalized[alias_map.get(key, key)] = value
        args = normalized

    result = fn(**args)
    return {"tool_name": name, "result": result}


# ----------------------------
# 3) Tool schemas (explicit)
# ----------------------------
weather_decl = types.FunctionDeclaration(
    name="get_weather",
    description="Get the current weather for a city.",
    parameters={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name, e.g. Singapore"},
            "unit": {"type": "string", "enum": ["C", "F"], "description": "Temperature unit"},
        },
        "required": ["city"],
    },
)

fx_decl = types.FunctionDeclaration(
    name="convert_currency",
    description="Convert money between currencies.",
    parameters={
        "type": "object",
        "properties": {
            "amount": {"type": "number", "description": "Amount to convert"},
            "from_": {"type": "string", "description": "ISO currency code, e.g. SGD"},
            "to": {"type": "string", "description": "ISO currency code, e.g. USD"},
        },
        "required": ["amount", "from_", "to"],
    },
)

tools = [types.Tool(function_declarations=[weather_decl, fx_decl])]


# ----------------------------
# 4) Ask model + execute tool call
# ----------------------------
prompt = "I'm flying tomorrow. What's the weather in Singapore in C, and convert 200 SGD to USD."

resp = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=prompt,
    config=types.GenerateContentConfig(tools=tools),
)

# Gemini may return a normal text response OR one/many function calls.
parts = resp.candidates[0].content.parts

function_calls = [part.function_call for part in parts if part.function_call]

if function_calls:
    tool_parts = []
    for fc in function_calls:
        print(fc)
        tool_output = dispatch_function_call(fc)
        tool_parts.append(
            types.Part(
                function_response=types.FunctionResponse(
                    name=fc.name,
                    response=tool_output["result"],  # important: give actual tool result
                )
            )
        )

    followup = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            prompt,
            types.Content(
                role="tool",
                parts=tool_parts,
            ),
        ],
    )

    print(followup.text)
else:
    print(resp.text)
