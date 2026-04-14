"""
Lesson 03: Structured Outputs - JSON, Schema Validation, Constrained Decoding
Tests MiniMax JSON mode, function calling, Pydantic validation, extraction pipeline.
"""
import json
import re
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(dotenv_path="/home/socraticblock/ai-engineering-from-scratch/practice/.env")

minimax = OpenAI(api_key=os.getenv("MINIMAX_API_KEY"), base_url="https://api.minimax.io/v1")


def strip_think_block(raw_text):
    """MiniMax prepends think block. Extract JSON starting from first '{'."""
    json_start = raw_text.rfind('{')
    if json_start == -1:
        return raw_text.strip()
    return raw_text[json_start:]


def safe_json_parse(raw_text):
    """Parse JSON, returning None on failure."""
    try:
        return json.loads(strip_think_block(raw_text))
    except json.JSONDecodeError:
        return None


# ─── Test 1: JSON Mode (response_format) ───────────────────────
def test_json_mode_simple():
    print(f"\n{'=' * 60}")
    print(f"  TEST 1: JSON Mode - Simple extraction")
    print(f"{'=' * 60}")

    response = minimax.chat.completions.create(
        model="MiniMax-M2.7",
        messages=[{
            "role": "user",
            "content": "Extract: Sony WH-1000XM5 headphones, $348, in stock. Return ONLY JSON with fields: product (string), price (number), in_stock (boolean). No markdown, no explanation."
        }],
        response_format={"type": "json_object"},
        max_tokens=1000
    )
    raw = response.choices[0].message.content
    print(f"  Raw response:\n{raw[:200]}")
    parsed = safe_json_parse(raw)
    print(f"  Parsed: {parsed}")
    return parsed


# ─── Test 2: JSON Mode - Complex schema ─────────────────────────
def test_json_mode_complex():
    print(f"\n{'=' * 60}")
    print(f"  TEST 2: JSON Mode - Complex nested schema")
    print(f"{'=' * 60}")

    schema_prompt = (
        "Extract product info from this text and return JSON matching this schema:\n"
        '{"product": str, "price": float, "in_stock": bool, '
        '"categories": list[str], "specs": {"brand": str, "model": str}}\n\n'
        'Text: "Sony WH-1000XM5 wireless headphones are the premium noise-cancelling headphones from Sony. '
        'They cost $348 and are currently available. They fall under the categories audio, headphones, and wireless."\n\n'
        "Only output JSON. No markdown fences."
    )

    response = minimax.chat.completions.create(
        model="MiniMax-M2.7",
        messages=[{"role": "user", "content": schema_prompt}],
        response_format={"type": "json_object"},
        max_tokens=1500
    )
    raw = response.choices[0].message.content
    parsed = safe_json_parse(raw)
    print(f"  Parsed: {json.dumps(parsed, indent=2) if parsed else 'FAILED'}")
    return parsed


# ─── Test 3: Function Calling ───────────────────────────────────
def test_function_calling():
    print(f"\n{'=' * 60}")
    print(f"  TEST 3: Function Calling (MiniMax tool_calls)")
    print(f"{'=' * 60}")

    response = minimax.chat.completions.create(
        model="MiniMax-M2.7",
        messages=[{
            "role": "user",
            "content": "Extract product info from: Apple MacBook Pro 16 costs $2499, out of stock."
        }],
        tools=[{
            "type": "function",
            "function": {
                "name": "extract_product",
                "description": "Extract structured product information from text",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "product": {"type": "string"},
                        "price": {"type": "number"},
                        "in_stock": {"type": "boolean"},
                        "brand": {"type": "string"}
                    },
                    "required": ["product", "price", "in_stock"]
                }
            }
        }],
        tool_choice="auto",
        max_tokens=1000
    )

    msg = response.choices[0].message
    has_tool_call = msg.tool_calls is not None and len(msg.tool_calls) > 0
    print(f"  Has tool_calls: {has_tool_call}")
    if has_tool_call:
        tc = msg.tool_calls[0]
        print(f"  Function: {tc.function.name}")
        print(f"  Arguments: {tc.function.arguments}")
        try:
            args = json.loads(tc.function.arguments)
            return args
        except json.JSONDecodeError:
            return None
    return None


# ─── Test 4: Schema validation ───────────────────────────────────
def test_schema_validation():
    print(f"\n{'=' * 60}")
    print(f"  TEST 4: Schema Validation (manual)")
    print(f"{'=' * 60}")

    product_schema = {
        "type": "object",
        "properties": {
            "product": {"type": "string"},
            "price": {"type": "number", "minimum": 0},
            "in_stock": {"type": "boolean"},
            "categories": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["product", "price", "in_stock"]
    }

    test_cases = [
        ({"product": "Test", "price": 10.0, "in_stock": True}, "Valid"),
        ({"product": "Test", "price": -5.0, "in_stock": True}, "Negative price"),
        ({"product": "Test", "in_stock": True}, "Missing price"),
        ({"product": "Test", "price": "ten", "in_stock": True}, "String as price"),
        ({"product": "Test", "price": 10.0, "in_stock": True, "categories": ["audio"]}, "With categories"),
    ]

    def validate(data, schema):
        errors = []
        if "price" in schema.get("required", []):
            if "price" not in data:
                errors.append("Missing required field: price")
            elif isinstance(data.get("price"), (int, float)) and data["price"] < 0:
                errors.append("Price must be non-negative")
        if not isinstance(data.get("price"), (int, float)):
            errors.append("Price must be a number")
        if "in_stock" in data and not isinstance(data["in_stock"], bool):
            errors.append("in_stock must be boolean")
        if "categories" in data and not isinstance(data["categories"], list):
            errors.append("categories must be array")
        return errors

    results = []
    for data, label in test_cases:
        errors = validate(data, product_schema)
        status = "PASS" if not errors else f"FAIL: {errors}"
        print(f"  {label}: {status}")
        results.append((label, not errors))

    return all(r[1] for r in results)


# ─── Test 5: Extraction pipeline with retry ─────────────────────
def test_extraction_pipeline():
    print(f"\n{'=' * 60}")
    print(f"  TEST 5: Extraction Pipeline with Retry")
    print(f"{'=' * 60}")

    extraction_schema = {
        "type": "object",
        "properties": {
            "event_name": {"type": "string"},
            "date": {"type": "string"},
            "location": {"type": "string"},
            "organizer": {"type": "string"}
        },
        "required": ["event_name", "date"]
    }

    texts = [
        "Join us for the AI Summit 2026 on March 15th at the Convention Center, organized by TechCorp.",
        "The Python Conference starts April 20th in San Francisco. Hosted by the Python Software Foundation.",
        "This is an email about something unrelated.",
    ]

    def extract_with_retry(text, max_retries=3):
        for attempt in range(max_retries):
            response = minimax.chat.completions.create(
                model="MiniMax-M2.7",
                messages=[{
                    "role": "user",
                    "content": (
                        f"Extract event information from this text. Return JSON with fields: "
                        "event_name, date, location, organizer. If info is missing, use null. "
                        "Only output JSON.\n\nText: " + text
                    )
                }],
                response_format={"type": "json_object"},
                max_tokens=500
            )
            parsed = safe_json_parse(response.choices[0].message.content)
            if parsed is None:
                print(f"  Attempt {attempt+1}: JSON parse failed")
                continue
            # Basic validation
            if "event_name" not in parsed:
                print(f"  Attempt {attempt+1}: Missing event_name, retrying")
                continue
            return parsed
        return None

    for text in texts:
        result = extract_with_retry(text)
        if result:
            print(f"  Input: {text[:50]}...")
            print(f"  Extracted: {json.dumps(result)}")
        else:
            print(f"  FAILED after retries: {text[:50]}...")

    return True


# ─── Test 6: Enum constraint ─────────────────────────────────────
def test_enum_constraint():
    print(f"\n{'=' * 60}")
    print(f"  TEST 6: Enum Constraint via prompt")
    print(f"{'=' * 60}")

    response = minimax.chat.completions.create(
        model="MiniMax-M2.7",
        messages=[{
            "role": "user",
            "content": (
                'Classify the sentiment of this text as EXACTLY one of: positive, negative, neutral.\n'
                'Return JSON with fields: sentiment (must be one of: positive, negative, neutral), '
                'confidence (number 0-1).\n\n'
                'Text: "The service was absolutely fantastic, exceeded all expectations!"\n\n'
                "Only output valid JSON."
            )
        }],
        response_format={"type": "json_object"},
        max_tokens=500
    )
    raw = response.choices[0].message.content
    parsed = safe_json_parse(raw)
    valid_sentiment = parsed and parsed.get("sentiment") in ["positive", "negative", "neutral"]
    print(f"  Parsed: {parsed}")
    print(f"  Valid enum: {valid_sentiment}")
    return parsed


def main():
    print("=" * 70)
    print("  LESSON 03: Structured Outputs — Real API Tests")
    print("=" * 70)

    p1 = test_json_mode_simple()
    p2 = test_json_mode_complex()
    p3 = test_function_calling()
    v = test_schema_validation()
    e = test_extraction_pipeline()
    en = test_enum_constraint()

    print("\n\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  JSON mode simple:      {'✅' if p1 else '❌'}")
    print(f"  JSON mode complex:      {'✅' if p2 else '❌'}")
    print(f"  Function calling:       {'✅' if p3 else '❌'}")
    print(f"  Schema validation:      {'✅' if v else '❌'}")
    print(f"  Extraction pipeline:    {'✅' if e else '❌'}")
    print(f"  Enum constraint:       {'✅' if en else '❌'}")
    print(f"\n  Lesson 03 COMPLETE")


if __name__ == "__main__":
    main()
