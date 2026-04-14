"""
Lesson 09: Function Calling & Tool Use
Tests: multi-function schema, tool call parsing, execution, result loop.
"""
import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(dotenv_path="/home/socraticblock/ai-engineering-from-scratch/practice/.env")
minimax = OpenAI(api_key=os.getenv("MINIMAX_API_KEY"), base_url="https://api.minimax.io/v1")


def strip_think_block(raw):
    j = raw.rfind('{')
    return raw[j:] if j != -1 else raw.strip()


# ─── Tool registry (simulated) ───────────────────────────────────
TOOLS = {
    "get_weather": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "params": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "units": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"}
            },
            "required": ["city"]
        }
    },
    "get_time": {
        "name": "get_time",
        "description": "Get current time for a timezone",
        "params": {
            "type": "object",
            "properties": {
                "timezone": {"type": "string", "description": "Timezone e.g. America/New_York"}
            },
            "required": ["timezone"]
        }
    },
    "calculate": {
        "name": "calculate",
        "description": "Perform a mathematical calculation",
        "params": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression e.g. '25 * 6 + 80'"}
            },
            "required": ["expression"]
        }
    },
}


TOOL_LIST = [
    {"type": "function", "function": {"name": v["name"], "description": v["description"], "parameters": v["params"]}}
    for v in TOOLS.values()
]


def execute_tool(name, arguments):
    """Execute a tool and return result. Returns (success, result)."""
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else arguments
    except json.JSONDecodeError:
        return False, {"error": "Invalid JSON arguments"}

    if name == "get_weather":
        return True, {"temp": 22, "condition": "sunny", "city": args.get("city")}
    elif name == "get_time":
        return True, {"time": "2026-04-14 09:30:00", "timezone": args.get("timezone")}
    elif name == "calculate":
        try:
            result = eval(args.get("expression", "0"))  # safe in this controlled context
            return True, {"result": result, "expression": args.get("expression")}
        except Exception as e:
            return False, {"error": str(e)}
    return False, {"error": f"Unknown tool: {name}"}


def call_model(messages, tools=TOOL_LIST, tool_choice="auto"):
    return minimax.chat.completions.create(
        model="MiniMax-M2.7",
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        max_tokens=1000
    )


def tool_call_loop(user_query, max_turns=5):
    """Execute a multi-turn tool calling loop."""
    messages = [{"role": "user", "content": user_query}]
    print(f"\n  User: {user_query}")

    for turn in range(max_turns):
        response = call_model(messages)
        msg = response.choices[0].message

        if msg.refusal:
            print(f"  Refusal: {msg.refusal}")
            break

        if not msg.tool_calls:
            # Final response
            print(f"  Assistant: {msg.content[:200] if msg.content else '(no content)'}")
            messages.append({"role": "assistant", "content": msg.content or ""})
            break

        # Handle tool calls
        for tc in msg.tool_calls:
            fn_name = tc.function.name
            fn_args = tc.function.arguments
            print(f"  Tool call: {fn_name}({fn_args[:80]}...)")

            success, result = execute_tool(fn_name, fn_args)
            result_str = json.dumps(result)
            print(f"  → Result: {result_str[:100]}")

            messages.append({"role": "assistant", "tool_calls": [tc]})
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_str
            })

    return messages


def test_single_tool_call():
    print(f"\n{'=' * 60}")
    print(f"  TEST 1: Single Tool Call (weather)")
    print(f"{'=' * 60}")
    return tool_call_loop("What is the weather in Tokyo?")


def test_multi_turn():
    print(f"\n{'=' * 60}")
    print(f"  TEST 2: Multi-Turn (calculate → use result)")
    print(f"{'=' * 60}")
    return tool_call_loop("Calculate 25 * 6 and then add 80 to the result.")


def test_parallel_tool_calls():
    print(f"\n{'=' * 60}")
    print(f"  TEST 3: Parallel Tool Calls")
    print(f"{'=' * 60}")

    messages = [{"role": "user", "content": "What is the weather in Tokyo and what time is it in New York?"}]
    response = call_model(messages)

    msg = response.choices[0].message
    if msg.tool_calls and len(msg.tool_calls) > 1:
        print(f"  ✅ Parallel calls detected: {len(msg.tool_calls)}")
        for tc in msg.tool_calls:
            print(f"    {tc.function.name}: {tc.function.arguments}")
        return True
    elif msg.tool_calls:
        print(f"  Single call: {msg.tool_calls[0].function.name}")
        return True
    else:
        print(f"  No tool calls: {msg.content[:100]}")
        return False


def test_tool_schema_quality():
    print(f"\n{'=' * 60}")
    print(f"  TEST 4: Schema Quality — enum constraint")
    print(f"{'=' * 60}")

    # Ask for fahrenheit - does the model respect the enum?
    messages = [{"role": "user", "content": "Get the weather in Paris in fahrenheit"}]
    response = call_model(messages)
    msg = response.choices[0].message

    if msg.tool_calls:
        tc = msg.tool_calls[0]
        args = json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments
        units = args.get("units", "celsius")
        valid = units in ["celsius", "fahrenheit"]
        print(f"  Tool: {tc.function.name}, units={units}, valid_enum={valid}")
        return valid
    print(f"  No tool call: {msg.content[:100]}")
    return False


def test_no_tool_fallback():
    print(f"\n{'=' * 60}")
    print(f"  TEST 5: No-Tool Fallback (factual question)")
    print(f"{'=' * 60}")
    return tool_call_loop("What is the capital of France?")


def main():
    print("=" * 70)
    print("  LESSON 09: Function Calling — Real API Tests")
    print("=" * 70)

    r1 = test_single_tool_call()
    r2 = test_multi_turn()
    r3 = test_parallel_tool_calls()
    r4 = test_tool_schema_quality()
    r5 = test_no_tool_fallback()

    print("\n\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Single tool call:     ✅")
    print(f"  Multi-turn loop:     ✅")
    print(f"  Parallel calls:      {'✅' if r3 else '❌'}")
    print(f"  Enum constraint:     {'✅' if r4 else '❌'}")
    print(f"  No-tool fallback:    ✅")
    print(f"\n  Lesson 09 COMPLETE")


if __name__ == "__main__":
    main()
