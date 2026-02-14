import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

def get_weather(city: str, unit: str = "C") -> dict:
    """Get current weather for a city. unit must be C or F."""
    return {"city": city, "unit": unit, "temp": 31, "condition": "Partly cloudy"}

def run():
    prompt = "Weather in Tokyo in F?"

    # Some versions accept Python callables directly, e.g. tools=[get_weather]
    # If this errors in your environment, use Variant A with FunctionDeclaration.
    resp = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[get_weather]  # <-- auto schema (if supported)
        ),
    )

    part = resp.candidates[0].content.parts[0]
    if part.function_call:
        fc = part.function_call
        result = get_weather(**(fc.args or {}))

        followup = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                prompt,
                types.Content(
                    role="tool",
                    parts=[
                        types.Part(
                            function_response=types.FunctionResponse(
                                name=fc.name,
                                response=result,
                            )
                        )
                    ],
                ),
            ],
        )
        print(followup.text)
    else:
        print(resp.text)

run()
