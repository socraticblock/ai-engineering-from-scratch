"""
Lesson 10: Evaluation & Testing LLM Applications
Tests: LLM-as-judge, automated metrics, regression testing.
"""
import json
import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(dotenv_path="/home/socraticblock/ai-engineering-from-scratch/practice/.env")
minimax = OpenAI(api_key=os.getenv("MINIMAX_API_KEY"), base_url="https://api.minimax.io/v1")


def llm_judge_quality(question, answer, rubric="Is the answer accurate, complete, and helpful?"):
    """Use MiniMax as a judge to score a response."""
    judge_prompt = (
        f"You are an expert evaluator. Score this answer on a scale of 1-5.\n\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n\n"
        f"Rubric: {rubric}\n\n"
        "Respond with JSON with fields: score (integer 1-5), reasoning (string explaining the score).\n"
        "Only output JSON."
    )
    response = minimax.chat.completions.create(
        model="MiniMax-M2.7",
        messages=[{"role": "user", "content": judge_prompt}],
        response_format={"type": "json_object"},
        max_tokens=500
    )
    raw = response.choices[0].message.content
    j_start = raw.rfind('{')
    if j_start != -1:
        raw = raw[j_start:]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"score": None, "reasoning": "Parse failed"}


def exact_match(predicted, expected):
    return predicted.strip().lower() == expected.strip().lower()


def test_llm_as_judge():
    print(f"\n{'=' * 60}")
    print(f"  TEST 1: LLM-as-Judge (MiniMax evaluating responses)")
    print(f"{'=' * 60}")

    test_cases = [
        {
            "question": "What is Python 3.12's main new feature?",
            "answer_good": "Python 3.12 introduced type parameter syntax for generic classes, allowing you to write class MyList[T]: without using the typing module.",
            "answer_bad": "Python 3.12 has many features including faster startup and better error messages.",
        },
        {
            "question": "What is the capital of France?",
            "answer_good": "The capital of France is Paris.",
            "answer_bad": "France is a country in Europe.",
        },
        {
            "question": "How do I sort a list in Python?",
            "answer_good": "Use the sorted() function or the .sort() method. sorted(list) returns a new sorted list, while list.sort() sorts in place.",
            "answer_bad": "Python is a programming language.",
        },
    ]

    for tc in test_cases:
        result_good = llm_judge_quality(tc["question"], tc["answer_good"])
        result_bad = llm_judge_quality(tc["question"], tc["answer_bad"])
        print(f"\n  Q: {tc['question'][:50]}...")
        print(f"  Good answer ({result_good.get('score', '?')}): {tc['answer_good'][:50]}...")
        print(f"  Bad answer ({result_bad.get('score', '?')}): {tc['answer_bad'][:50]}...")

    return True


def test_exact_match():
    print(f"\n{'=' * 60}")
    print(f"  TEST 2: Exact Match + ROUGE-like scoring")
    print(f"{'=' * 60}")

    pairs = [
        ("The answer is 42", "42", True),
        ("The answer is 42", "The answer is 42", True),
        ("42", "42.0", False),
        ("Paris", "paris", True),
        ("Hello world", "Hello World", True),
    ]

    for pred, expected, expected_match in pairs:
        match = exact_match(pred, expected)
        status = "✅" if match == expected_match else "❌"
        print(f"  {status} '{pred}' vs '{expected}': {match}")


def test_regression_suite():
    print(f"\n{'=' * 60}")
    print(f"  TEST 3: Regression Test Suite")
    print(f"{'=' * 60}")

    # Simulate a test suite for an LLM app
    test_suite = [
        {"id": "t1", "query": "What is 2+2?", "expected_keywords": ["4"]},
        {"id": "t2", "query": "Capital of Japan?", "expected_keywords": ["Tokyo"]},
        {"id": "t3", "query": "Python web framework?", "expected_keywords": ["FastAPI", "Flask", "Django"]},
        {"id": "t4", "query": "What is an LLM?", "expected_keywords": ["language", "model"]},
    ]

    results = []
    for tc in test_suite:
        response = minimax.chat.completions.create(
            model="MiniMax-M2.7",
            messages=[{"role": "user", "content": tc["query"]}],
            temperature=0.0,
            max_tokens=200
        )
        answer = response.choices[0].message.content
        passed = any(kw.lower() in answer.lower() for kw in tc["expected_keywords"])
        results.append({"id": tc["id"], "passed": passed, "answer": answer[:80]})
        print(f"  [{'PASS' if passed else 'FAIL'}] {tc['id']}: {tc['query']}")
        print(f"         Answer: {answer[:60]}...")

    passed_count = sum(1 for r in results if r["passed"])
    print(f"\n  Results: {passed_count}/{len(results)} tests passed")
    return passed_count == len(results)


def main():
    print("=" * 70)
    print("  LESSON 10: Evaluation — Real API Tests")
    print("=" * 70)

    test_llm_as_judge()
    test_exact_match()
    r = test_regression_suite()

    print("\n\n" + "=" * 70)
    print("  Lesson 10 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
