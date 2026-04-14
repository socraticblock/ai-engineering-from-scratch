"""
Lesson 02: Few-Shot, Chain-of-Thought, Tree-of-Thought - Real MiniMax + Ollama
Tests: few-shot, zero-shot CoT, few-shot CoT, self-consistency, tree-of-thought.
"""
import json
import re
import time
import os
from collections import Counter
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(dotenv_path="/home/socraticblock/ai-engineering-from-scratch/practice/.env")

minimax = OpenAI(api_key=os.getenv("MINIMAX_API_KEY"), base_url="https://api.minimax.io/v1")
ollama = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# ─── GSM8K-style math examples (from curriculum) ───────────────
GSM8K_EXAMPLES = [
    {
        "question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells every egg at the farmers' market for $2. How much does she make every day at the farmers' market?",
        "reasoning": "Janet's ducks lay 16 eggs per day. She eats 3 and bakes 4, using 3 + 4 = 7 eggs. So she has 16 - 7 = 9 eggs left. She sells each for $2, so she makes 9 * 2 = $18 per day.",
        "answer": "18"
    },
    {
        "question": "A store has 48 apples. They sell 1/3 on Monday and 1/4 of the rest on Tuesday. How many are left?",
        "reasoning": "48 apples total. Sell 1/3: 48/3 = 16 sold, 48 - 16 = 32 remain. Sell 1/4 of rest: 32/4 = 8 sold, 32 - 8 = 24 remain. Answer: 24.",
        "answer": "24"
    },
    {
        "question": "Marco has 15 stickers. He buys 3 packs of 5 stickers each. Then he gives 7 stickers to his sister. How many stickers does Marco have now?",
        "reasoning": "Marco starts with 15 stickers. He buys 3 packs of 5 = 15 more. Total: 15 + 15 = 30. He gives 7 to his sister: 30 - 7 = 23. Answer: 23.",
        "answer": "23"
    },
]

# ─── Math test questions ────────────────────────────────────────
TEST_QUESTIONS = [
    "A baker has 36 cupcakes. She puts them in boxes of 6. She sells each box for $8. How much does she earn if she sells all boxes?",
    "There are 72 students on a field trip. They need to split into vans that hold 8 students each. How many vans are needed?",
    "John saves $25 each month. After 6 months, he spends $80 on a game. How much does he have left?",
]

# ─── Helpers ────────────────────────────────────────────────────
def extract_answer(text: str):
    """Extract the final numeric answer from a reasoning chain."""
    if not text:
        return None
    # Look for "The answer is N" or just a trailing number
    patterns = [
        r"[Tt]he answer is (\d+)",
        r"(\d+)\.?$",
    ]
    for p in patterns:
        m = re.search(p, text.strip())
        if m:
            return m.group(1)
    return None


def build_cot_prompt(question, examples, num_examples=3, include_answer_format=True):
    """Build a few-shot CoT prompt with GSM8K examples."""
    system = (
        "You are a math problem solver. "
        "Show your step-by-step reasoning, then give the final answer "
        "in the format: 'The answer is [number]'."
    )
    example_text = ""
    for ex in examples[:num_examples]:
        example_text += f"Q: {ex['question']}\n"
        example_text += f"A: {ex['reasoning']} The answer is {ex['answer']}.\n\n"

    user = f"{example_text}Q: {question}\nA:"
    return system, user


def call_minimax(system, user, temperature=0.7, max_tokens=1500):
    response = minimax.chat.completions.create(
        model="MiniMax-M2.7",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
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
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content, dict(response.usage)


def strip_think_block(raw_text):
    """MiniMax prepends a think block to JSON. Extract the actual content."""
    json_start = raw_text.rfind('{')
    if json_start == -1:
        return raw_text.strip()
    return raw_text[json_start:]


# ─── Test 1: Zero-Shot vs Few-Shot ─────────────────────────────
def test_zero_vs_few_shot():
    print(f"\n{'=' * 60}")
    print(f"  TEST: Zero-Shot vs Few-Shot (MiniMax)")
    print(f"{'=' * 60}")

    question = TEST_QUESTIONS[0]
    # Zero-shot
    system_z = "You are a math solver. Answer the question. Show your reasoning then state the final answer as 'The answer is [number]'."
    user_z = f"Q: {question}\nA:"
    raw_z, _ = call_minimax(system_z, user_z, temperature=0.0)
    ans_z = extract_answer(raw_z)
    print(f"\n[Zero-Shot] Q: {question[:60]}...")
    print(f"  Answer extracted: {ans_z}")
    print(f"  Full: {raw_z[:200]}...")

    # Few-shot
    system_fs, user_fs = build_cot_prompt(question, GSM8K_EXAMPLES)
    raw_fs, _ = call_minimax(system_fs, user_fs, temperature=0.0)
    ans_fs = extract_answer(raw_fs)
    print(f"\n[Few-Shot] Q: {question[:60]}...")
    print(f"  Answer extracted: {ans_fs}")
    print(f"  Full: {raw_fs[:200]}...")

    return ans_z, ans_fs


# ─── Test 2: Zero-Shot CoT ──────────────────────────────────────
def test_zero_shot_cot():
    print(f"\n{'=' * 60}")
    print(f"  TEST: Zero-Shot Chain-of-Thought ('think step by step')")
    print(f"{'=' * 60}")

    question = TEST_QUESTIONS[1]
    system_cot = (
        "You are a math problem solver. Think step by step, "
        "then give the final answer as 'The answer is [number]'."
    )
    user_cot = f"Q: {question}\nA: Let's think step by step."

    raw, _ = call_minimax(system_cot, user_cot, temperature=0.0)
    ans = extract_answer(raw)
    print(f"\n[Zero-Shot CoT] Q: {question[:60]}...")
    print(f"  Answer: {ans}")
    print(f"  Response: {raw[:300]}...")

    return ans


# ─── Test 3: Few-Shot CoT ───────────────────────────────────────
def test_few_shot_cot():
    print(f"\n{'=' * 60}")
    print(f"  TEST: Few-Shot Chain-of-Thought (3 examples)")
    print(f"{'=' * 60}")

    question = TEST_QUESTIONS[2]
    system, user = build_cot_prompt(question, GSM8K_EXAMPLES, num_examples=3)
    raw, usage = call_minimax(system, user, temperature=0.0)
    ans = extract_answer(raw)
    print(f"\n[Few-Shot CoT] Q: {question[:60]}...")
    print(f"  Answer: {ans} | Tokens: {usage['total_tokens']}")
    print(f"  Response: {raw[:300]}...")

    return ans


# ─── Test 4: Self-Consistency Voting ─────────────────────────────
def test_self_consistency():
    print(f"\n{'=' * 60}")
    print(f"  TEST: Self-Consistency (N=5, majority vote)")
    print(f"{'=' * 60}")

    question = TEST_QUESTIONS[0]
    system, user = build_cot_prompt(question, GSM8K_EXAMPLES, num_examples=3)

    answers = []
    reasonings = []
    n_samples = 5

    for i in range(n_samples):
        raw, _ = call_minimax(system, user, temperature=0.7, max_tokens=1000)
        answers.append(extract_answer(raw))
        reasonings.append(raw[:150])
        time.sleep(0.2)  # brief pause

    vote_counts = Counter(answers)
    best_answer, vote_count = vote_counts.most_common(1)[0]
    confidence = vote_count / len(answers)

    print(f"\n[Self-Consistency] Q: {question[:60]}...")
    print(f"  Votes: {dict(vote_counts)}")
    print(f"  Winner: {best_answer} ({vote_count}/{n_samples} = {confidence:.0%} confidence)")
    for i, (r, a) in enumerate(zip(reasonings, answers)):
        print(f"  Sample {i+1}: ans={a}, reasoning={r[:80]}...")

    return best_answer, confidence


# ─── Test 5: Ollama qwen3 for comparison ───────────────────────
def test_ollama_comparison():
    print(f"\n{'=' * 60}")
    print(f"  TEST: Ollama qwen3:8b vs MiniMax (few-shot CoT)")
    print(f"{'=' * 60}")

    question = TEST_QUESTIONS[1]
    system, user = build_cot_prompt(question, GSM8K_EXAMPLES, num_examples=3)

    raw, usage = call_ollama(system, user, model="qwen3:8b", temperature=0.0)
    ans = extract_answer(raw)
    print(f"\n[Ollama qwen3:8b] Q: {question[:60]}...")
    print(f"  Answer: {ans} | Tokens: {usage.get('total_tokens', '?')}")
    print(f"  Response: {raw[:300]}...")

    return ans


# ─── Test 6: Structured extraction with JSON mode ──────────────
def test_structured_math_response():
    print(f"\n{'=' * 60}")
    print(f"  TEST: Structured JSON output (MiniMax JSON mode)")
    print(f"{'=' * 60}")

    question = TEST_QUESTIONS[2]
    prompt = (
        "Solve the math problem. Return your answer as JSON with fields:\n"
        "reasoning (string with step-by-step), final_answer (string with the number).\n"
        "Only output JSON, no markdown.\n\n"
        f"Problem: {question}"
    )

    response = minimax.chat.completions.create(
        model="MiniMax-M2.7",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        max_tokens=1500
    )
    raw = response.choices[0].message.content
    parsed = json.loads(strip_think_block(raw))
    print(f"  Parsed: {json.dumps(parsed, indent=2)}")

    return parsed


def main():
    print("=" * 70)
    print("  LESSON 02: Few-Shot, CoT, ToT — Real API Tests")
    print("=" * 70)

    ans_z, ans_fs = test_zero_vs_few_shot()
    ans_cot = test_zero_shot_cot()
    ans_fscot = test_few_shot_cot()
    best_sc, conf_sc = test_self_consistency()
    n_samples = 5
    ans_ollama = test_ollama_comparison()
    parsed = test_structured_math_response()

    print("\n\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Zero-Shot answer:              {ans_z}")
    print(f"  Few-Shot answer:               {ans_fs}")
    print(f"  Zero-Shot CoT answer:          {ans_cot}")
    print(f"  Few-Shot CoT answer:           {ans_fscot}")
    print(f"  Self-Consistency ({n_samples} samples): {best_sc} ({conf_sc:.0%})")
    print(f"  Ollama qwen3:8b:              {ans_ollama}")
    print(f"  Structured JSON parsed:        {parsed.get('final_answer')}")
    print(f"\n  Lesson 02 COMPLETE")


if __name__ == "__main__":
    main()
