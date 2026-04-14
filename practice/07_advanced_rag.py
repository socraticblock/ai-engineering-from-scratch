"""
Lesson 07: Advanced RAG - Hybrid Search, Query Transformation
Tests: BM25 keyword search, Reciprocal Rank Fusion, query expansion.
"""
import json
import math
import os
import re
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(dotenv_path="/home/socraticblock/ai-engineering-from-scratch/practice/.env")

ollama = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
minimax = OpenAI(api_key=os.getenv("MINIMAX_API_KEY"), base_url="https://api.minimax.io/v1")


def embed_texts(texts, model="nomic-embed-text"):
    r = ollama.embeddings.create(model=model, input=texts)
    return [d.embedding for d in r.data]


def cosine_sim(a, b):
    return sum(x * y for x, y in zip(a, b))


# ─── BM25 Keyword Search ────────────────────────────────────────
class BM25:
    def __init__(self, k1=1.2, b=0.75):
        self.k1 = k1
        self.b = b
        self.doc_lengths = []
        self.avgdl = 0
        self.doc_freqs = {}
        self.idf = {}
        self.corpus = []

    def fit(self, documents):
        self.corpus = [doc.lower().split() for doc in documents]
        self.doc_lengths = [len(d) for d in self.corpus]
        self.avgdl = sum(self.doc_lengths) / len(self.corpus) if self.corpus else 1
        df = {}
        for doc in self.corpus:
            for word in set(doc):
                df[word] = df.get(word, 0) + 1
        n = len(self.corpus)
        self.idf = {word: math.log((n - df[word] + 0.5) / (df[word] + 0.5) + 1)
                    for word in df}

    def score(self, query, doc_idx):
        doc = self.corpus[doc_idx]
        dl = self.doc_lengths[doc_idx]
        score = 0.0
        for term in query.lower().split():
            if term not in self.idf:
                continue
            tf = doc.count(term)
            idf = self.idf[term]
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += idf * numerator / denominator
        return score

    def search(self, query, top_k=5):
        scores = [(i, self.score(query, i)) for i in range(len(self.corpus))]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [(i, s) for i, s in scores[:top_k]]


# ─── Reciprocal Rank Fusion ─────────────────────────────────────
def rrf_fusion(vector_results, bm25_results, k=60):
    """Combine rankings using Reciprocal Rank Fusion."""
    scores = {}
    for rank, (idx, score) in enumerate(vector_results):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)
    for rank, (idx, score) in enumerate(bm25_results):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def test_bm25():
    print(f"\n{'=' * 60}")
    print(f"  TEST 1: BM25 Keyword Search")
    print(f"{'=' * 60}")

    docs = [
        "Python 3.12 introduced type parameter syntax for generic classes",
        "FastAPI is a modern Python web framework for building APIs",
        "PostgreSQL database with JSON support and full-text search",
        "Docker containers wrap application code with dependencies",
        "Kubernetes orchestrates containerized applications at scale",
        "TypeScript is a typed superset of JavaScript",
        "GitHub Actions provides CI/CD workflows for repositories",
        "React is a JavaScript library for building user interfaces",
    ]

    bm25 = BM25()
    bm25.fit(docs)

    queries = [
        "Python web framework",
        "Docker Kubernetes",
        "TypeScript JavaScript",
        "database SQL",
    ]

    for q in queries:
        results = bm25.search(q, top_k=3)
        print(f"\n  Query: '{q}'")
        for idx, score in results:
            print(f"    [{score:.4f}] {docs[idx][:60]}...")


def test_hybrid_search():
    print(f"\n{'=' * 60}")
    print(f"  TEST 2: Hybrid Search (Vector + BM25 + RRF)")
    print(f"{'=' * 60}")

    docs = [
        "Python 3.12 introduced type parameter syntax for generic classes",
        "FastAPI is a modern Python web framework for building APIs",
        "PostgreSQL database with JSON support and full-text search",
        "Docker containers wrap application code with dependencies",
        "Kubernetes orchestrates containerized applications at scale",
        "TypeScript is a typed superset of JavaScript",
        "GitHub Actions provides CI/CD workflows for repositories",
        "React is a JavaScript library for building user interfaces",
    ]

    bm25 = BM25()
    bm25.fit(docs)

    t0 = time.time()
    doc_embeddings = embed_texts(docs)
    embed_time = (time.time() - t0) * 1000

    query = "Python web framework"
    q_emb = embed_texts([query])[0]

    # Vector search
    vec_scores = sorted(
        [(i, cosine_sim(q_emb, e)) for i, e in enumerate(doc_embeddings)],
        key=lambda x: x[1], reverse=True
    )[:5]

    # BM25
    bm25_scores = bm25.search(query, top_k=5)

    # RRF fusion
    fused = rrf_fusion(vec_scores, bm25_scores)

    print(f"  Query: '{query}' | Embed time: {embed_time:.1f}ms")
    print(f"\n  {'Rank':<5} {'Fused Score':>12} {'Vector':>8} {'BM25':>8} {'Doc'}")
    print(f"  {'-'*80}")
    for rank, (idx, fused_score) in enumerate(fused[:5]):
        v_score = next((s for i, s in vec_scores if i == idx), 0)
        b_score = next((s for i, s in bm25_scores if i == idx), 0)
        print(f"  {rank+1:<5} {fused_score:>12.4f} {v_score:>8.4f} {b_score:>8.4f} {docs[idx][:40]}...")


def test_query_transformation():
    print(f"\n{'=' * 60}")
    print(f"  TEST 3: Query Transformation (multi-query expansion)")
    print(f"{'=' * 60}")

    original_query = "Python async programming"

    expansion_prompt = (
        "Generate 3 different versions of this search query that express the same information need "
        "using different words. Return JSON with field 'queries' (list of strings).\n\n"
        f"Query: {original_query}\n\n"
        "Only output JSON."
    )

    response = minimax.chat.completions.create(
        model="MiniMax-M2.7",
        messages=[{"role": "user", "content": expansion_prompt}],
        response_format={"type": "json_object"},
        max_tokens=500
    )
    raw = response.choices[0].message.content
    j_start = raw.rfind('{')
    if j_start != -1:
        raw = raw[j_start:]
    try:
        parsed = json.loads(raw)
        expanded = parsed.get("queries", [])
        print(f"  Original: '{original_query}'")
        print(f"  Expanded queries:")
        for q in expanded:
            print(f"    - '{q}'")
    except json.JSONDecodeError:
        print(f"  Parse failed: {raw[:100]}")


import time


def main():
    print("=" * 70)
    print("  LESSON 07: Advanced RAG — Real API Tests")
    print("=" * 70)

    test_bm25()
    test_hybrid_search()
    test_query_transformation()

    print("\n\n" + "=" * 70)
    print("  Lesson 07 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
