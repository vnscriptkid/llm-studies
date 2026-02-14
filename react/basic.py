from __future__ import annotations

import os
from enum import Enum
from typing import Callable, Any, Optional, Union

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from google import genai
from google.genai import types


# ---------------------------
# 1) ENV + CLIENT SETUP
# ---------------------------

def load_env() -> None:
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "Missing GOOGLE_API_KEY. Put it in your environment or in a .env file."
        )

load_env()

client = genai.Client()
MODEL_ID = "gemini-2.5-flash"


# ---------------------------
# 2) TOOL LAYER (MOCK TOOL)
# ---------------------------

def search(query: str) -> str:
    """Search for information about a specific topic or query.

    Args:
        query (str): The search query or topic to look up.

    Returns:
        str: A short factual snippet if found, or a not found message.
    """
    q = query.lower().strip()

    # Predefined responses for demo
    if all(w in q for w in ["capital", "france"]):
        return "Paris is the capital of France and is known for the Eiffel Tower."
    if all(w in q for w in ["capital", "japan"]):
        return "Tokyo is the capital of Japan."
    if "react" in q:
        return (
            "ReAct interleaves reasoning (thought) with acting (tool calls) and "
            "uses tool observations to refine subsequent steps."
        )

    return f"Information about '{query}' was not found."

TOOL_REGISTRY: dict[str, Callable[..., str]] = {
    search.__name__: search,
}


# ---------------------------
# 3) CLASSIC ReAct - THOUGHT PHASE
# ---------------------------

def build_tools_xml_description(tool_registry: dict[str, Callable[..., str]]) -> str:
    """Build a minimal XML description of tools using only their docstrings."""
    lines: list[str] = []
    for tool_name, fn in tool_registry.items():
        doc = (fn.__doc__ or "").strip()
        lines.append(f'\t<tool name="{tool_name}">')
        if doc:
            lines.append("\t\t<description>")
            for line in doc.split("\n"):
                lines.append(f"\t\t\t{line}")
            lines.append("\t\t</description>")
        lines.append("\t</tool>")
    return "\n".join(lines)


PROMPT_TEMPLATE_THOUGHT = """
You are deciding the next best step for reaching the user goal. You have some tools available to you.

Available tools:
<tools>
{tools_xml}
</tools>

Conversation so far:
<conversation>
{conversation}
</conversation>

State your next **thought** about what to do next as one short paragraph focused on the next action you intend to take and why.
Avoid repeating the same strategies that didn't work previously. Prefer different approaches.

Remember:
- Return ONLY plain natural language text.
- Do NOT emit JSON, XML, function calls, or code.
""".strip()


def generate_thought(conversation: str, tool_registry: dict[str, Callable[..., str]]) -> str:
    """Generate a thought as plain text (no structured output)."""
    tools_xml = build_tools_xml_description(tool_registry)
    prompt = PROMPT_TEMPLATE_THOUGHT.format(tools_xml=tools_xml, conversation=conversation)

    response: types.GenerateContentResponse = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
    )
    return (response.text or "").strip()


# ---------------------------
# 4) CLASSIC ReAct - ACTION PHASE
# ---------------------------

PROMPT_TEMPLATE_ACTION = """
You are selecting the best next action to reach the user goal.

Conversation so far:
<conversation>
{conversation}
</conversation>

Respond either with a tool call (with arguments) or a final answer, but only if you can confidently conclude.
""".strip()

PROMPT_TEMPLATE_ACTION_FORCED = """
You must now provide a final answer to the user.

Conversation so far:
<conversation>
{conversation}
</conversation>

Provide a concise final answer that best addresses the user's goal.
""".strip()


class ToolCallRequest(BaseModel):
    """A request to call a tool with its name and arguments."""
    tool_name: str = Field(description="The name of the tool to call.")
    arguments: dict = Field(description="The arguments to pass to the tool.")


class FinalAnswer(BaseModel):
    """A final answer to present to the user when no further action is needed."""
    text: str = Field(description="The final answer text to present to the user.")


def generate_action(
    conversation: str,
    tool_registry: Optional[dict[str, Callable[..., str]]] = None,
    force_final: bool = False,
) -> Union[ToolCallRequest, FinalAnswer]:
    """Generate an action by passing tools to the LLM and parsing function calls or final text.

    When force_final is True or tool_registry is None/empty, the model is instructed to produce a final answer
    and tool calls are disabled.
    """
    # Forced final (or no tools): just ask for final answer text
    if force_final or not tool_registry:
        prompt = PROMPT_TEMPLATE_ACTION_FORCED.format(conversation=conversation)
        response = client.models.generate_content(model=MODEL_ID, contents=prompt)
        return FinalAnswer(text=(response.text or "").strip())

    # Otherwise: allow the model to "choose" a tool call via function calling
    prompt = PROMPT_TEMPLATE_ACTION.format(conversation=conversation)

    tools = list(tool_registry.values())
    config = types.GenerateContentConfig(
        tools=tools,
        # Disable auto execution: model will *propose* function calls; we run them ourselves.
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    response: types.GenerateContentResponse = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config=config,
    )

    # Parse function call from response parts
    candidate = response.candidates[0]
    for part in candidate.content.parts:
        if getattr(part, "function_call", None):
            name = part.function_call.name
            args = dict(part.function_call.args or {})
            return ToolCallRequest(tool_name=name, arguments=args)

    # Otherwise treat as final answer text
    final_text = "".join(part.text for part in candidate.content.parts if getattr(part, "text", None))
    return FinalAnswer(text=final_text.strip())


# ---------------------------
# 5) CLASSIC ReAct - OBSERVATION PHASE
# ---------------------------

def observe(action_request: ToolCallRequest, tool_registry: dict[str, Callable[..., str]]) -> str:
    """Execute the selected tool and return observation text (result or error)."""
    name = action_request.tool_name
    args = action_request.arguments

    if name not in tool_registry:
        return f"Unknown tool '{name}'. Available: {', '.join(tool_registry)}"

    try:
        return tool_registry[name](**args)
    except Exception as e:
        return f"Error executing tool '{name}': {e}"


# ---------------------------
# 6) SCRATCHPAD + TRACE
# ---------------------------

class MessageRole(str, Enum):
    USER = "user"
    THOUGHT = "thought"
    TOOL_REQUEST = "tool request"
    OBSERVATION = "observation"
    FINAL_ANSWER = "final answer"


class Message(BaseModel):
    role: MessageRole = Field(description="The role of the message in the ReAct loop.")
    content: str = Field(description="The textual content of the message.")

    def __str__(self) -> str:
        return f"{self.role.value.capitalize()}: {self.content}"


class Scratchpad:
    """Container for ReAct messages with optional printing."""
    def __init__(self, max_turns: int) -> None:
        self.messages: list[Message] = []
        self.max_turns = max_turns
        self.current_turn = 1

    def set_turn(self, turn: int) -> None:
        self.current_turn = turn

    def append(self, message: Message, verbose: bool = False, forced: bool = False) -> None:
        self.messages.append(message)
        if verbose:
            forced_tag = " (Forced)" if forced else ""
            print(f"\n--- {message.role.value.upper()}{forced_tag} (Turn {self.current_turn}/{self.max_turns}) ---")
            print(message.content)

    def to_string(self) -> str:
        return "\n".join(str(m) for m in self.messages)


# ---------------------------
# 7) CLASSIC ReAct LOOP
# ---------------------------

def react_agent_loop(
    initial_question: str,
    tool_registry: dict[str, Callable[..., str]],
    max_turns: int = 5,
    verbose: bool = True,
) -> str:
    """Classic ReAct: Thought -> Action -> Observation loop with a scratchpad."""
    scratchpad = Scratchpad(max_turns=max_turns)

    scratchpad.append(Message(role=MessageRole.USER, content=initial_question), verbose=verbose)

    for turn in range(1, max_turns + 1):
        scratchpad.set_turn(turn)

        # THOUGHT
        thought = generate_thought(scratchpad.to_string(), tool_registry)
        scratchpad.append(Message(role=MessageRole.THOUGHT, content=thought), verbose=verbose)

        # ACTION
        action = generate_action(scratchpad.to_string(), tool_registry=tool_registry)

        # If final answer -> done
        if isinstance(action, FinalAnswer):
            scratchpad.append(Message(role=MessageRole.FINAL_ANSWER, content=action.text), verbose=verbose)
            return action.text

        # TOOL REQUEST -> OBSERVATION
        params_str = ", ".join(f"{k}={repr(v)}" for k, v in action.arguments.items())
        scratchpad.append(
            Message(role=MessageRole.TOOL_REQUEST, content=f"{action.tool_name}({params_str})"),
            verbose=verbose,
        )

        obs = observe(action, tool_registry)
        scratchpad.append(Message(role=MessageRole.OBSERVATION, content=obs), verbose=verbose)

        # If last turn: force a final answer
        if turn == max_turns:
            forced = generate_action(scratchpad.to_string(), force_final=True)
            final_text = forced.text if isinstance(forced, FinalAnswer) else "Unable to answer within limits."
            scratchpad.append(
                Message(role=MessageRole.FINAL_ANSWER, content=final_text),
                verbose=verbose,
                forced=True,
            )
            return final_text

    return "Unable to answer."


# ---------------------------
# 8) MODEL-NATIVE THINKING ReAct LOOP
# ---------------------------

THINKING_CONFIG = types.ThinkingConfig(
    include_thoughts=True,   # return thought summaries (NOT raw chain-of-thought)
    thinking_budget=1024,    # tune depth vs latency; -1 lets model decide
)

def extract_thought_summary(response: types.GenerateContentResponse) -> Optional[str]:
    """Collect human-readable thought summaries if present."""
    parts = getattr(response.candidates[0].content, "parts", []) or []
    chunks = [
        p.text
        for p in parts
        if getattr(p, "thought", False) and getattr(p, "text", None)
    ]
    joined = "\n".join(chunks).strip()
    return joined if joined else None


def extract_first_function_call(response: types.GenerateContentResponse) -> Optional[tuple[str, dict]]:
    """Return (name, args) for first function call, or None if not present."""
    if getattr(response, "function_calls", None):
        fc = response.function_calls[0]
        return fc.name, dict(fc.args or {})

    parts = getattr(response.candidates[0].content, "parts", []) or []
    for p in parts:
        if getattr(p, "function_call", None):
            return p.function_call.name, dict(p.function_call.args or {})
    return None


def build_config_with_tools(tools: list[Callable[..., str]]) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        tools=tools,
        thinking_config=THINKING_CONFIG,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )


def react_agent_loop_thinking(
    initial_question: str,
    tool_registry: dict[str, Callable[..., str]],
    max_turns: int = 5,
    verbose: bool = True,
) -> str:
    """ReAct relying on model-native thinking + thought signatures preserved in Content history."""
    human_log = Scratchpad(max_turns=max_turns)
    human_log.append(Message(role=MessageRole.USER, content=initial_question), verbose=verbose)

    # This is the important part: keep *structured* contents and append model Content back each turn
    contents: list[types.Content] = [
        types.Content(role="user", parts=[types.Part(text=initial_question)])
    ]

    tools = list(tool_registry.values())
    config = build_config_with_tools(tools)

    for turn in range(1, max_turns + 1):
        human_log.set_turn(turn)

        response = client.models.generate_content(
            model=MODEL_ID,
            contents=contents,
            config=config,
        )

        # Log thought summary if present
        thoughts = extract_thought_summary(response)
        if thoughts:
            human_log.append(Message(role=MessageRole.THOUGHT, content=thoughts), verbose=verbose)

        # Tool call?
        fc = extract_first_function_call(response)
        if fc:
            name, args = fc

            # Preserve thought signatures by appending model content
            contents.append(response.candidates[0].content)

            params_str = ", ".join(f"{k}={repr(v)}" for k, v in args.items())
            human_log.append(
                Message(role=MessageRole.TOOL_REQUEST, content=f"{name}({params_str})"),
                verbose=verbose,
            )

            obs = observe(ToolCallRequest(tool_name=name, arguments=args), tool_registry)
            human_log.append(Message(role=MessageRole.OBSERVATION, content=obs), verbose=verbose)

            # Feed tool result back via function response protocol
            fn_resp_part = types.Part.from_function_response(
                name=name,
                response={"result": obs},
            )
            contents.append(types.Content(role="user", parts=[fn_resp_part]))
            continue

        # No tool call => final answer
        final_text = (response.text or "").strip()
        human_log.append(Message(role=MessageRole.FINAL_ANSWER, content=final_text), verbose=verbose)
        return final_text

    # Forced finish: disallow tool calling, ask for final
    forced_config = types.GenerateContentConfig(
        thinking_config=THINKING_CONFIG,
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode=types.FunctionCallingConfigMode.NONE
            )
        ),
    )
    forced_response = client.models.generate_content(
        model=MODEL_ID,
        contents=contents,
        config=forced_config,
    )
    final_text = (forced_response.text or "Unable to answer within limits.").strip()
    human_log.append(Message(role=MessageRole.FINAL_ANSWER, content=final_text), verbose=verbose, forced=True)
    return final_text


# ---------------------------
# 9) RUN DEMOS
# ---------------------------

if __name__ == "__main__":
    print("\n====================")
    print("DEMO: CLASSIC ReAct")
    print("====================")
    ans1 = react_agent_loop("What is the capital of France?", TOOL_REGISTRY, max_turns=2, verbose=True)
    print("\nFinal:", ans1)

    print("\n===============================")
    print("DEMO: THINKING (MODEL-NATIVE)")
    print("===============================")
    ans2 = react_agent_loop_thinking("What is the capital of France?", TOOL_REGISTRY, max_turns=3, verbose=True)
    print("\nFinal:", ans2)

    print("\n=======================================")
    print("DEMO: FAILURE + FORCED FINAL (CLASSIC)")
    print("=======================================")
    ans3 = react_agent_loop("What is the capital of Italy?", TOOL_REGISTRY, max_turns=2, verbose=True)
    print("\nFinal:", ans3)
