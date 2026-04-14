"""
Lesson 01: Prompt Engineering - Real MiniMax + Ollama version
Adapted from curriculum code to use actual API calls.
"""
import json
import re
import time
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(dotenv_path="/home/socraticblock/ai-engineering-from-scratch/practice/.env")

# ─── Model clients ──────────────────────────────────────────────
minimax = OpenAI(
    api_key=os.getenv("MINIMAX_API_KEY"),
    base_url="https://api.minimax.io/v1"
)
ollama = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

# ─── Prompt Patterns (from curriculum) ─────────────────────────
PROMPT_PATTERNS = {
    "persona": {
        "name": "Persona Pattern",
        "template": (
            "You are {role} with {experience}.\n"
            "Your communication style is {style}.\n"
            "You prioritize {priority}.\n\n"
            "{task}"
        ),
        "variables": ["role", "experience", "style", "priority", "task"],
        "temperature": 0.7,
        "description": "Activates a specific expert distribution in the model's training data",
    },
    "few_shot": {
        "name": "Few-Shot Pattern",
        "template": (
            "Here are examples of the expected input/output format:\n\n"
            "{examples}\n\n"
            "Now process this input:\n{input}"
        ),
        "variables": ["examples", "input"],
        "temperature": 0.0,
        "description": "Provides concrete examples to anchor the output format and style",
    },
    "chain_of_thought": {
        "name": "Chain-of-Thought Pattern",
        "template": (
            "Think through this step by step.\n\n"
            "Problem: {problem}\n\n"
            "Steps:\n"
            "1. Identify the key components\n"
            "2. Analyze each component\n"
            "3. Synthesize your findings\n"
            "4. State your conclusion\n\n"
            "Show your reasoning before giving the final answer."
        ),
        "variables": ["problem"],
        "temperature": 0.3,
        "description": "Forces explicit reasoning steps before the final answer",
    },
    "template_fill": {
        "name": "Template Fill Pattern",
        "template": (
            "Extract information from the following text and fill in the template.\n\n"
            "Text: {text}\n\n"
            "Template:\n{template_structure}\n\n"
            "Fill in every field. If information is not available, write 'N/A'."
        ),
        "variables": ["text", "template_structure"],
        "temperature": 0.0,
        "description": "Constrains output to a specific structure with named fields",
    },
    "guardrail": {
        "name": "Guardrail Pattern",
        "template": (
            "You are a {role}.\n\n"
            "Rules:\n"
            "- ONLY answer questions about {domain}\n"
            "- If the question is outside {domain}, say: 'This is outside my scope.'\n"
            "- NEVER make up information. If unsure, say 'I don't know.'\n"
            "- {additional_rules}\n\n"
            "User question: {question}"
        ),
        "variables": ["role", "domain", "additional_rules", "question"],
        "temperature": 0.3,
        "description": "Constrains the model to a specific domain with explicit boundaries",
    },
    "boundary": {
        "name": "Boundary Pattern",
        "template": (
            "You are an assistant that ONLY handles {scope}.\n\n"
            "If the user's request is within scope, help them fully.\n"
            "If the user's request is outside scope, respond exactly with:\n"
            "'{refusal_message}'\n\n"
            "Do not attempt to answer out-of-scope questions.\n\n"
            "User: {user_input}"
        ),
        "variables": ["scope", "refusal_message", "user_input"],
        "temperature": 0.0,
        "description": "Hard boundary on what the model will and will not respond to",
    },
}


def build_prompt(pattern_name, variables, system_override=None):
    pattern = PROMPT_PATTERNS.get(pattern_name)
    if not pattern:
        raise ValueError(f"Unknown pattern: {pattern_name}")
    rendered = pattern["template"].format(**variables)
    system = system_override or f"You are an AI assistant using the {pattern['name']}."
    return {
        "system": system,
        "user": rendered,
        "temperature": pattern["temperature"],
        "pattern": pattern_name,
    }


def call_minimax(system, user, temperature=0.7, max_tokens=1500):
    response = minimax.chat.completions.create(
        model="MiniMax-M2.7",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )
    raw = response.choices[0].message.content
    u = response.usage
    return raw, {
        "total_tokens": getattr(u, "total_tokens", 0),
        "prompt_tokens": getattr(u, "prompt_tokens", 0),
        "completion_tokens": getattr(u, "completion_tokens", 0),
    }


def call_ollama(system, user, model="qwen3:8b", temperature=0.7, max_tokens=1500):
    response = ollama.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content, dict(response.usage)


def strip_think_block(raw_text):
    """MiniMax outputs a think block before JSON. Extract the actual content."""
    json_start = raw_text.rfind('{')
    if json_start == -1:
        return raw_text.strip()
    return raw_text[json_start:]


def run_pattern(name, prompt_def, test_fn=None):
    print(f"\n{'=' * 60}")
    print(f"  PATTERN: {name} ({prompt_def['name']})")
    print(f"  Temp: {prompt_def['temperature']} | {prompt_def['description']}")
    print(f"{'=' * 60}")

    vars_for_pattern = test_fn if test_fn else {
        "role": "a senior backend engineer at Stripe",
        "experience": "10 years of API design experience",
        "style": "precise, concise, example-driven",
        "priority": "correctness over speed",
        "task": "Explain what API rate limiting is and why it matters.",
    }
    p = build_prompt(name, vars_for_pattern)
    print(f"\nSystem: {p['system']}")
    print(f"User: {p['user'][:120]}...")

    # Call MiniMax
    start = time.time()
    mm_response, mm_usage = call_minimax(p["system"], p["user"], p["temperature"])
    mm_latency = (time.time() - start) * 1000
    total_t = mm_usage.get("total_tokens", "?")
    print(f"\n[MiniMax M2.7] ({mm_latency:.0f}ms, {total_t}t)")
    print(f"  {mm_response[:300]}...")

    return mm_response


def main():
    print("=" * 70)
    print("  LESSON 01: PROMPT ENGINEERING — REAL API TESTS")
    print("=" * 70)

    # ── Pattern 1: Persona ────────────────────────────────────────
    run_pattern("persona", PROMPT_PATTERNS["persona"])

    # ── Pattern 2: Few-Shot ────────────────────────────────────────
    run_pattern("few_shot", PROMPT_PATTERNS["few_shot"], {
        "examples": (
            'Input: "The food was amazing but service was slow"\n'
            'Output: {"sentiment": "mixed", "food": "positive", "service": "negative"}\n\n'
            'Input: "Terrible experience, never coming back"\n'
            'Output: {"sentiment": "negative", "food": null, "service": "negative"}'
        ),
        "input": "\"Great ambiance and the pasta was perfect, though a bit pricey\""
    })

    # ── Pattern 3: Chain-of-Thought ───────────────────────────────
    run_pattern("chain_of_thought", PROMPT_PATTERNS["chain_of_thought"], {
        "problem": (
            "A store offers 20% off all items. An item costs $85. "
            "There is also a $10 coupon. Which saves more: discount first then coupon, or coupon first then discount?"
        )
    })

    # ── Pattern 4: Template Fill ──────────────────────────────────
    run_pattern("template_fill", PROMPT_PATTERNS["template_fill"], {
        "text": "John Smith is a software engineer at Google with 5 years of experience. "
                "He graduated from MIT with a BS in Computer Science in 2019. "
                "He specializes in distributed systems and Go.",
        "template_structure": (
            "Name: [full name]\n"
            "Company: [current employer]\n"
            "Years of Experience: [number]\n"
            "Education: [degree, school, year]\n"
            "Specialties: [comma-separated list]"
        )
    })

    # ── Pattern 5: Guardrail ───────────────────────────────────────
    run_pattern("guardrail", PROMPT_PATTERNS["guardrail"], {
        "role": "Python programming tutor",
        "domain": "Python programming",
        "additional_rules": "Do not write complete solutions. Guide with hints.",
        "question": "How do I sort a list of dictionaries by a specific key?"
    })

    # ── Pattern 6: Boundary ───────────────────────────────────────
    run_pattern("boundary", PROMPT_PATTERNS["boundary"], {
        "scope": "explaining Git version control",
        "refusal_message": "I'm sorry, but I only answer questions about Git.",
        "user_input": "What's the weather in Tbilisi?"
    })

    # ── JSON Mode Test ─────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  JSON MODE (MiniMax with response_format)")
    print(f"{'=' * 60}")
    r = minimax.chat.completions.create(
        model="MiniMax-M2.7",
        messages=[{"role": "user", "content": "Return a short recipe for eggs as JSON: fields name, ingredients (list of 3 strings), steps (list of 2 strings). Only output JSON."}],
        response_format={"type": "json_object"},
        max_tokens=1500
    )
    raw = r.choices[0].message.content
    parsed = json.loads(strip_think_block(raw))
    print(f"  Parsed JSON: {json.dumps(parsed, indent=2)}")

    print("\n\n" + "=" * 70)
    print("  LESSON 01 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
