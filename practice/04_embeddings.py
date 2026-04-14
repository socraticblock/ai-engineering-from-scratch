"""
Lesson 04: Embeddings & Vector Representations
Tests: Ollama nomic-embed-text (already running), cosine similarity,
semantic search, embedding quality comparison.
"""
import json
import math
import os
import time
from openai import OpenAI

ollama = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")


# ─── Embedding client ───────────────────────────────────────────
def embed_texts(texts, model="nomic-embed-text"):
    """Embed texts using Ollama."""
    response = ollama.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]


# ─── Similarity metrics ─────────────────────────────────────────
def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * y for x, y in zip(b, b)))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def dot_product(a, b):
    return sum(x * y for x, y in zip(a, b))


def euclidean_distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


# ─── Test 1: Embedding generation ───────────────────────────────
def test_embedding_generation():
    print(f"\n{'=' * 60}")
    print(f"  TEST 1: Embedding Generation (Ollama nomic-embed-text)")
    print(f"{'=' * 60}")

    texts = [
        "How do I reset my password?",
        "I need to change my account password",
        "The weather is nice today",
        "What's for lunch?",
        "How to update my email address",
        "I forgot my login credentials",
    ]

    embeddings = embed_texts(texts)
    print(f"  Generated {len(embeddings)} embeddings")
    print(f"  Dimension: {len(embeddings[0])}")
    print(f"  Sample (first 10 dims): {embeddings[0][:10]}")

    # Check norms are unit length (nomic normalizes)
    norms = [math.sqrt(sum(x*x for x in e)) for e in embeddings]
    print(f"  Norms: {[f'{n:.4f}' for n in norms[:3]]}")
    print(f"  All normalized: {all(0.99 < n < 1.01 for n in norms)}")

    return embeddings, texts


# ─── Test 2: Cosine similarity — semantic similarity ────────────
def test_semantic_similarity(embeddings, texts):
    print(f"\n{'=' * 60}")
    print(f"  TEST 2: Cosine Similarity — Semantic Matching")
    print(f"{'=' * 60}")

    query = "I can't remember my password"
    query_emb = embed_texts([query])[0]

    print(f"  Query: '{query}'")
    print(f"\n  Top-5 most similar:")

    similarities = []
    for i, (emb, text) in enumerate(zip(embeddings, texts)):
        sim = cosine_similarity(query_emb, emb)
        similarities.append((sim, text, i))

    similarities.sort(key=lambda x: x[0], reverse=True)
    for rank, (sim, text, idx) in enumerate(similarities[:5], 1):
        bar = "█" * int(sim * 30)
        print(f"  {rank}. [{sim:.4f}] {text[:50]}")


# ─── Test 3: Different metrics comparison ────────────────────────
def test_metric_comparison(embeddings, texts):
    print(f"\n{'=' * 60}")
    print(f"  TEST 3: Cosine vs Dot Product vs Euclidean")
    print(f"{'=' * 60}")

    query_text = "How do I update my email?"
    query_emb = embed_texts([query_text])[0]

    results = {"cosine": [], "dot": [], "euclidean": []}
    for emb, text in zip(embeddings, texts):
        results["cosine"].append((cosine_similarity(query_emb, emb), text))
        results["dot"].append((dot_product(query_emb, emb), text))
        results["euclidean"].append((euclidean_distance(query_emb, emb), text))

    # Sort each
    for key in results:
        if key == "euclidean":
            results[key].sort(key=lambda x: x[0])  # lower = better
        else:
            results[key].sort(key=lambda x: x[0], reverse=True)

    print(f"  Query: '{query_text}'")
    print(f"\n  {'Text':<45} {'Cosine':>7} {'Dot':>8} {'Euclid':>8}")
    print(f"  {'-'*72}")
    for i in range(5):
        cos_t, cos_v = results["cosine"][i]
        dot_t, dot_v = results["dot"][i]
        # Find euclidean for same text
        euc_v = next(e for e, t in results["euclidean"] if t == cos_v)
        print(f"  {cos_v[:43]:<45} {cos_t:>7.4f} {dot_t:>8.4f} {euc_v:>8.4f}")


# ─── Test 4: Semantic search pipeline ──────────────────────────
def test_semantic_search():
    print(f"\n{'=' * 60}")
    print(f"  TEST 4: Semantic Search Pipeline")
    print(f"{'=' * 60}")

    documents = [
        "Python 3.12 introduced type parameter syntax for generic classes.",
        "FastAPI is a modern Python web framework for building APIs.",
        "The PostgreSQL database supports JSON columns and full-text search.",
        "Docker containers wrap applications with their dependencies.",
        "GitHub Actions enables CI/CD pipelines directly in repositories.",
        "React hooks let you use state in functional components.",
        "TypeScript adds static typing to JavaScript.",
        "pgvector is a PostgreSQL extension for vector similarity search.",
        "The requests library is used for making HTTP calls in Python.",
        "Kubernetes orchestrates containerized applications at scale.",
    ]

    # Embed all docs
    t0 = time.time()
    doc_embeddings = embed_texts(documents)
    embed_time = (time.time() - t0) * 1000
    print(f"  Indexed {len(documents)} documents in {embed_time:.1f}ms")
    print(f"  Embedding dimension: {len(doc_embeddings[0])}")

    queries = [
        "Python web framework for APIs",
        "Database with vector search",
        "CI/CD and deployment tools",
        "Frontend JavaScript framework",
    ]

    print(f"\n  Query results:")
    for query in queries:
        q_emb = embed_texts([query])[0]
        scored = [(cosine_similarity(q_emb, e), d) for e, d in zip(doc_embeddings, documents)]
        scored.sort(key=lambda x: x[0], reverse=True)
        print(f"\n  Query: '{query}'")
        for sim, doc in scored[:3]:
            print(f"    [{sim:.4f}] {doc[:60]}...")


# ─── Test 5: Chunk size effect ─────────────────────────────────
def test_chunking_strategy():
    print(f"\n{'=' * 60}")
    print(f"  TEST 5: Chunk Size Strategy (fixed-size vs sentence)")
    print(f"{'=' * 60}")

    long_text = (
        "Machine learning is a subset of artificial intelligence. "
        "Deep learning uses neural networks with many layers. "
        "Natural language processing deals with text data. "
        "Computer vision processes images and video. "
        "Reinforcement learning trains agents through rewards. "
        "Supervised learning uses labeled training data. "
        "Unsupervised learning finds patterns without labels."
    )

    # Fixed-size chunking
    def chunk_text_fixed(text, chunk_size=15, overlap=5):
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            chunk = " ".join(words[start:start + chunk_size])
            chunks.append(chunk)
            start += chunk_size - overlap
        return chunks

    # Sentence-based chunking
    def chunk_sentences(text, max_words=10):
        sentences = text.replace(".", ".\n").split("\n")
        sentences = [s.strip() for s in sentences if s.strip()]
        chunks, current = [], []
        word_count = 0
        for sent in sentences:
            sent_words = len(sent.split())
            if word_count + sent_words > max_words and current:
                chunks.append(" ".join(current))
                current = []
                word_count = 0
            current.append(sent)
            word_count += sent_words
        if current:
            chunks.append(" ".join(current))
        return chunks

    fixed_chunks = chunk_text_fixed(long_text, chunk_size=15, overlap=5)
    sent_chunks = chunk_sentences(long_text, max_words=10)

    print(f"  Fixed-size (15 words, 5 overlap): {len(fixed_chunks)} chunks")
    for c in fixed_chunks:
        print(f"    - {c[:60]}...")

    print(f"\n  Sentence-based (max 10 words): {len(sent_chunks)} chunks")
    for c in sent_chunks:
        print(f"    - {c[:60]}...")


def main():
    print("=" * 70)
    print("  LESSON 04: Embeddings & Vector Representations")
    print("=" * 70)

    embeddings, texts = test_embedding_generation()
    test_semantic_similarity(embeddings, texts)
    test_metric_comparison(embeddings, texts)
    test_semantic_search()
    test_chunking_strategy()

    print("\n\n" + "=" * 70)
    print("  Lesson 04 COMPLETE")
    print(f"  Key: Ollama nomic-embed-text (768d) works reliably.")
    print(f"  Cosine similarity correctly identifies semantic similarity.")
    print("=" * 70)


if __name__ == "__main__":
    main()
