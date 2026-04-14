"""
Lesson 11: Caching, Rate Limiting & Cost Optimization
Tests: exact-match cache, semantic cache, cost calculation, model routing.
"""
import json
import hashlib
import time
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(dotenv_path="/home/socraticblock/ai-engineering-from-scratch/practice/.env")
minimax = OpenAI(api_key=os.getenv("MINIMAX_API_KEY"), base_url="https://api.minimax.io/v1")
ollama = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")


def embed_texts(texts):
    r = ollama.embeddings.create(model="nomic-embed-text", input=texts)
    return [d.embedding for d in r.data]


def cosine_sim(a, b):
    return sum(x * y for x, y in zip(a, b))


# ─── Exact-match cache ──────────────────────────────────────────
class ExactCache:
    def __init__(self):
        self.store = {}

    def make_key(self, text):
        return hashlib.sha256(text.encode()).hexdigest()

    def get(self, text):
        return self.store.get(self.make_key(text))

    def set(self, text, result):
        self.store[self.make_key(text)] = result


# ─── Semantic cache ─────────────────────────────────────────────
class SemanticCache:
    def __init__(self, similarity_threshold=0.95):
        self.entries = []  # list of (embedding, text, result)
        self.threshold = similarity_threshold

    def get(self, text, embedding):
        for emb, orig_text, result in self.entries:
            sim = cosine_sim(embedding, emb)
            if sim >= self.threshold:
                return result, sim
        return None, 0.0

    def set(self, text, embedding, result):
        self.entries.append((embedding, text, result))


# ─── Cost calculator ─────────────────────────────────────────────
COSTS = {
    "MiniMax-M2.7": {"input": 0.002, "output": 0.002},  # per 1K tokens (approximate)
    "qwen3:8b": {"input": 0.0, "output": 0.0},  # local, free
}


def calc_cost(model, prompt_tokens, completion_tokens):
    rates = COSTS.get(model, {"input": 0.0, "output": 0.0})
    return prompt_tokens * rates["input"] / 1000 + completion_tokens * rates["output"] / 1000


# ─── Tests ──────────────────────────────────────────────────────
def test_exact_cache():
    print(f"\n{'=' * 60}")
    print(f"  TEST 1: Exact-Match Cache")
    print(f"{'=' * 60}")

    cache = ExactCache()
    query = "What is Python's walrus operator?"

    # First call - cache miss
    t0 = time.time()
    r1 = minimax.chat.completions.create(
        model="MiniMax-M2.7",
        messages=[{"role": "user", "content": query}],
        temperature=0.0,
        max_tokens=300
    )
    miss_time = (time.time() - t0) * 1000
    result1 = r1.choices[0].message.content

    # Cache it
    cache.set(query, result1)

    # Second call - cache hit
    t0 = time.time()
    result2 = cache.get(query)
    hit_time = (time.time() - t0) * 1000

    print(f"  Cache miss: {miss_time:.1f}ms | Cache hit: {hit_time:.3f}ms")
    print(f"  Speedup: {miss_time/hit_time:.0f}x")
    print(f"  Content match: {result1 == result2}")
    return result2 is not None


def test_semantic_cache():
    print(f"\n{'=' * 60}")
    print(f"  TEST 2: Semantic Cache")
    print(f"{'=' * 60}")

    cache = SemanticCache(similarity_threshold=0.90)

    q1 = "How do I reset my password?"
    emb1 = embed_texts([q1])[0]
    result1 = "Use the forgot password link on the login page."
    cache.set(q1, emb1, result1)

    # Very similar query
    q2 = "I can't reset my password"
    emb2 = embed_texts([q2])[0]
    hit_result, sim = cache.get(q2, emb2)
    print(f"  Q1: '{q1}'")
    print(f"  Q2: '{q2}'")
    print(f"  Similarity: {sim:.4f} | Cache hit: {hit_result is not None}")

    # Very different query
    q3 = "What is the weather today?"
    emb3 = embed_texts([q3])[0]
    hit_result3, sim3 = cache.get(q3, emb3)
    print(f"  Q3: '{q3}' | Similarity: {sim3:.4f} | Cache hit: {hit_result3 is not None}")

    return True


def test_cost_calculation():
    print(f"\n{'=' * 60}")
    print(f"  TEST 3: Cost Calculation")
    print(f"{'=' * 60}")

    # Simulate a batch
    calls = [
        {"model": "MiniMax-M2.7", "prompt_tokens": 500, "completion_tokens": 150},
        {"model": "MiniMax-M2.7", "prompt_tokens": 800, "completion_tokens": 200},
        {"model": "MiniMax-M2.7", "prompt_tokens": 1200, "completion_tokens": 300},
    ]

    total = 0.0
    for c in calls:
        cost = calc_cost(c["model"], c["prompt_tokens"], c["completion_tokens"])
        c["cost"] = cost
        total += cost
        print(f"  {c['model']}: {c['prompt_tokens']} in + {c['completion_tokens']} out = ${cost:.6f}")

    print(f"\n  Batch total: ${total:.6f}")
    print(f"  Extrapolated 1000 calls: ${total * 1000:.2f}")


def test_model_routing():
    print(f"\n{'=' * 60}")
    print(f"  TEST 4: Model Routing (simple)")
    print(f"{'=' * 60}")

    def route_query(query):
        """Route to cheap or expensive model based on query complexity."""
        simple_keywords = ["what is", "who is", "define", "capital of"]
        complex_keywords = ["explain", "analyze", "compare", "write code", "debug"]

        q_lower = query.lower()
        if any(kw in q_lower for kw in simple_keywords):
            return "cheap"
        elif any(kw in q_lower for kw in complex_keywords):
            return "expensive"
        return "medium"

    queries = [
        "What is Python?",
        "Compare FastAPI vs Django for building APIs",
        "Explain how async/await works in Python",
        "Capital of Georgia",
    ]

    for q in queries:
        route = route_query(q)
        print(f"  [{route:>9}] {q}")


def main():
    print("=" * 70)
    print("  LESSON 11: Caching & Cost — Real API Tests")
    print("=" * 70)

    test_exact_cache()
    test_semantic_cache()
    test_cost_calculation()
    test_model_routing()

    print("\n\n" + "=" * 70)
    print("  Lesson 11 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
