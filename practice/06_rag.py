"""
Lesson 06: RAG (Retrieval-Augmented Generation)
Build: document chunking, embedding, vector index, retrieval, augmentation, generation.
"""
import json
import math
import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(dotenv_path="/home/socraticblock/ai-engineering-from-scratch/practice/.env")

minimax = OpenAI(api_key=os.getenv("MINIMAX_API_KEY"), base_url="https://api.minimax.io/v1")
ollama = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")


def embed_texts(texts, model="nomic-embed-text"):
    r = ollama.embeddings.create(model=model, input=texts)
    return [d.embedding for d in r.data]


def cosine_sim(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot  # both normalized


def chunk_text_fixed(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunk = " ".join(words[start:start + chunk_size])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def build_rag_prompt(context_docs, query):
    context = "\n\n".join(f"Document {i+1}:\n{doc}" for i, doc in enumerate(context_docs))
    return (
        "You are a helpful assistant. Use the provided context to answer the user's question.\n"
        "If the answer is not in the context, say so.\n\n"
        f"{context}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )


class VectorIndex:
    def __init__(self):
        self.vectors = []
        self.texts = []
        self.metadata = []

    def add(self, vec, text, meta=None):
        self.vectors.append(vec)
        self.texts.append(text)
        self.metadata.append(meta or {})

    def search(self, query_vec, top_k=5):
        scores = [(i, cosine_sim(query_vec, v)) for i, v in enumerate(self.vectors)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [{"index": i, "score": s, "text": self.texts[i], "meta": self.metadata[i]}
                for i, s in scores[:top_k]]

    def size(self):
        return len(self.vectors)


def test_rag_pipeline():
    print(f"\n{'=' * 60}")
    print(f"  TEST: RAG Pipeline (full build + retrieval + generation)")
    print(f"{'=' * 60}")

    # ─── Knowledge base ────────────────────────────────────────────
    docs = [
        "Python 3.12 introduced type parameter syntax for generic classes: class MyList[T]: pass",
        "FastAPI is a Python web framework for building APIs, built on Starlette and Pydantic.",
        "PostgreSQL is a powerful open-source relational database with ACID compliance.",
        "GitHub Actions enables continuous integration and deployment through YAML workflows.",
        "Docker packages applications and dependencies into containers for consistent environments.",
        "Kubernetes automates deployment, scaling, and management of containerized applications.",
        "TypeScript is a statically typed superset of JavaScript that compiles to plain JS.",
        "React is a JavaScript library for building user interfaces with component-based architecture.",
        "The WALRUS operator := in Python 3.8 assigns values within expressions.",
        "Claude AI is an AI assistant by Anthropic designed to be helpful, harmless, and honest.",
    ]

    # ─── Chunk and index ──────────────────────────────────────────
    all_chunks = []
    for doc in docs:
        chunks = chunk_text_fixed(doc, chunk_size=50, overlap=10)
        all_chunks.extend(chunks)

    print(f"  Indexing {len(all_chunks)} chunks from {len(docs)} documents...")
    t0 = time.time()
    chunk_embeddings = embed_texts(all_chunks)
    index_time = (time.time() - t0) * 1000
    print(f"  Embedded in {index_time:.1f}ms ({len(all_chunks)} chunks, 768d)")

    index = VectorIndex()
    for i, (chunk, emb) in enumerate(zip(all_chunks, chunk_embeddings)):
        index.add(emb, chunk, {"doc_idx": i % len(docs)})

    # ─── Retrieval test ──────────────────────────────────────────
    queries = [
        "What is new in Python 3.12?",
        "Python web framework for APIs",
        "Containers and orchestration",
        "Static typing for JavaScript",
    ]

    for query in queries:
        print(f"\n  Query: '{query}'")
        q_emb = embed_texts([query])[0]
        results = index.search(q_emb, top_k=3)
        context_docs = [r["text"] for r in results]
        print(f"  Top-3 retrieved:")
        for r in results:
            print(f"    [{r['score']:.4f}] {r['text'][:60]}...")

        # ─── Generate with context ───────────────────────────────
        prompt = build_rag_prompt(context_docs, query)
        r = minimax.chat.completions.create(
            model="MiniMax-M2.7",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300
        )
        answer = r.choices[0].message.content
        print(f"  Generated: {answer[:200]}...")

    # ─── RAG vs No-RAG comparison ───────────────────────────────
    print(f"\n  --- RAG vs No-RAG comparison ---")
    question = "What Python feature was introduced in version 3.12?"

    # Without RAG
    r_no = minimax.chat.completions.create(
        model="MiniMax-M2.7",
        messages=[{"role": "user", "content": f"Question: {question}\nAnswer concisely."}],
        temperature=0.0,
        max_tokens=200
    )
    print(f"\n  [No-RAG] Q: {question}")
    print(f"  {r_no.choices[0].message.content[:200]}...")

    # With RAG
    q_emb = embed_texts([question])[0]
    results = index.search(q_emb, top_k=2)
    context = "\n".join(r["text"] for r in results)
    r_rag = minimax.chat.completions.create(
        model="MiniMax-M2.7",
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\nAnswer concisely."}],
        temperature=0.0,
        max_tokens=200
    )
    print(f"\n  [RAG] Q: {question}")
    print(f"  {r_rag.choices[0].message.content[:200]}...")

    return True


def main():
    print("=" * 70)
    print("  LESSON 06: RAG — Real API Tests")
    print("=" * 70)
    test_rag_pipeline()
    print("\n  Lesson 06 COMPLETE")


if __name__ == "__main__":
    main()
