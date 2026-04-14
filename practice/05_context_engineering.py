"""
Lesson 05: Context Engineering - Token Budgets, Window Management, Retrieval
Tests: token counting, context budget manager, lost-in-the-middle ordering,
conversation history summarization, dynamic tool selection.
"""
import json
import os
import time
from collections import OrderedDict
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(dotenv_path="/home/socraticblock/ai-engineering-from-scratch/practice/.env")
minimax = OpenAI(api_key=os.getenv("MINIMAX_API_KEY"), base_url="https://api.minimax.io/v1")


def count_tokens(text):
    """Approximate token count: whitespace split × 1.3."""
    if not text:
        return 0
    return int(len(text.split()) * 1.3)


def strip_think_block(raw):
    j = raw.rfind('{')
    return raw[j:] if j != -1 else raw.strip()


# ─── Token Budget Manager ──────────────────────────────────────
class ContextBudget:
    def __init__(self, max_tokens=128000, generation_reserve=4000):
        self.max_tokens = max_tokens
        self.generation_reserve = generation_reserve
        self.available = max_tokens - generation_reserve
        self.allocations = OrderedDict()

    def allocate(self, component, content, max_tokens=None):
        tokens = count_tokens(content)
        if max_tokens and tokens > max_tokens:
            words = content.split()
            target_words = int(max_tokens / 1.3)
            content = " ".join(words[:target_words])
            tokens = count_tokens(content)

        used = sum(self.allocations.values())
        if used + tokens > self.available:
            allowed = self.available - used
            if allowed <= 0:
                return None, 0
            words = content.split()
            target_words = int(allowed / 1.3)
            content = " ".join(words[:target_words])
            tokens = count_tokens(content)

        self.allocations[component] = tokens
        return content, tokens

    def remaining(self):
        return self.available - sum(self.allocations.values())

    def report(self):
        lines = [f"Context Budget ({self.max_tokens:,} token window)"]
        lines.append("-" * 50)
        total_used = sum(self.allocations.values())
        for comp, toks in self.allocations.items():
            pct = toks / self.max_tokens * 100
            bar = "█" * int(pct / 2)
            lines.append(f"  {comp:<28} {toks:>6} ({pct:>5.1f}%) {bar}")
        lines.append("-" * 50)
        lines.append(f"  {'Used':<28} {total_used:>6} ({total_used/self.max_tokens*100:.1f}%)")
        lines.append(f"  {'Generation reserve':<28} {self.generation_reserve:>6}")
        lines.append(f"  {'Remaining':<28} {self.remaining():>6}")
        return "\n".join(lines)


# ─── Lost-in-the-middle reordering ─────────────────────────────
def reorder_lost_in_middle(items, scores):
    """Put most relevant first and last, least relevant in middle."""
    paired = sorted(zip(scores, items), key=lambda x: x[0], reverse=True)
    sorted_items = [item for _, item in paired]
    if len(sorted_items) <= 2:
        return sorted_items
    first_half = sorted_items[::2]
    second_half = list(reversed(sorted_items[1::2]))
    return first_half + second_half


def score_by_overlap(query, documents):
    """Simple word-overlap scoring."""
    q_words = set(query.lower().split())
    if not q_words:
        return [0.0] * len(documents)
    scores = []
    for doc in documents:
        d_words = set(doc.lower().split())
        overlap = len(q_words & d_words) / len(q_words)
        scores.append(round(overlap, 3))
    return scores


# ─── Conversation History Compressor ────────────────────────────
class ConversationManager:
    def __init__(self, max_history_tokens=5000):
        self.turns = []
        self.summaries = []
        self.max_history_tokens = max_history_tokens

    def add_turn(self, role, content):
        self.turns.append({"role": role, "content": content})
        self._compress_if_needed()

    def _compress_if_needed(self):
        total = sum(count_tokens(t["content"]) for t in self.turns)
        if total <= self.max_history_tokens:
            return
        while total > self.max_history_tokens and len(self.turns) > 4:
            old = self.turns[:2]
            summary = self._summarize_turns(old)
            self.summaries.append(summary)
            self.turns = self.turns[2:]
            total = sum(count_tokens(t["content"]) for t in self.turns)

    def _summarize_turns(self, turns):
        parts = []
        for t in turns:
            content = t["content"]
            if len(content) > 100:
                content = content[:100] + "..."
            parts.append(f"{t['role']}: {content}")
        return "Previous: " + " | ".join(parts)

    def token_count(self):
        return count_tokens(self.get_context())

    def get_context(self):
        parts = []
        if self.summaries:
            parts.append("[Summary of Earlier Conversation]")
            parts.extend(self.summaries)
        parts.append("[Recent Conversation]")
        parts.extend(f"{t['role']}: {t['content']}" for t in self.turns)
        return "\n".join(parts)


# ─── Dynamic Tool Selector ───────────────────────────────────────
TOOL_REGISTRY = {
    "read_file": {"description": "Read file contents", "tokens": 120, "categories": ["code", "files"]},
    "write_file": {"description": "Write content to file", "tokens": 150, "categories": ["code", "files"]},
    "search_code": {"description": "Search code patterns", "tokens": 130, "categories": ["code"]},
    "run_command": {"description": "Execute shell command", "tokens": 140, "categories": ["code", "system"]},
    "web_search": {"description": "Search the web", "tokens": 140, "categories": ["research"]},
    "send_email": {"description": "Send an email", "tokens": 200, "categories": ["email"]},
}


INTENT_KEYWORDS = {
    "code": ["code", "function", "bug", "error", "file", "implement", "refactor", "debug"],
    "email": ["email", "mail", "send", "inbox"],
    "research": ["search", "find", "what is", "how does", "explain", "look up"],
}


def classify_intent(query):
    q = query.lower()
    scores = {}
    for intent, keywords in INTENT_KEYWORDS.items():
        scores[intent] = sum(1 for kw in keywords if kw in q)
    if not scores:
        return ["code"]
    max_score = max(scores.values())
    return [i for i, s in scores.items() if s >= max_score * 0.5]


def select_tools(query, budget=2000):
    intents = classify_intent(query)
    selected = {}
    used = 0
    for name, tool in TOOL_REGISTRY.items():
        if any(cat in intents for cat in tool["categories"]):
            if used + tool["tokens"] <= budget:
                selected[name] = tool
                used += tool["tokens"]
    return selected, used


# ─── Test 1: Token counting ─────────────────────────────────────
def test_token_counting():
    print(f"\n{'=' * 60}")
    print(f"  TEST 1: Token Counting (approximation)")
    print(f"{'=' * 60}")

    samples = [
        "Hello world",
        "The quick brown fox jumps over the lazy dog. This is a longer sentence for testing.",
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "Python 3.12 introduced type parameter syntax for generic classes using bracket notation.",
    ]

    for s in samples:
        words = len(s.split())
        tokens = count_tokens(s)
        print(f"  Words={words:3d} | Tokens~{tokens:4d} | '{s[:50]}...' ")


# ─── Test 2: Context Budget Manager ─────────────────────────────
def test_budget_manager():
    print(f"\n{'=' * 60}")
    print(f"  TEST 2: Context Budget Manager")
    print(f"{'=' * 60}")

    budget = ContextBudget(max_tokens=10000, generation_reserve=2000)
    system = "You are a helpful AI assistant." * 50
    tools = json.dumps(list(TOOL_REGISTRY.keys())) * 10
    context = "Relevant context from the knowledge base. " * 50
    history = "user: Hello\nassistant: Hi there!\n" * 20
    query = "What is Python?"

    alloc1, _ = budget.allocate("system_prompt", system, max_tokens=500)
    alloc2, _ = budget.allocate("tools", tools, max_tokens=1000)
    alloc3, _ = budget.allocate("retrieved_context", context, max_tokens=2000)
    alloc4, _ = budget.allocate("conversation_history", history, max_tokens=2000)
    alloc5, _ = budget.allocate("user_query", query, max_tokens=200)

    print(budget.report())


# ─── Test 3: Lost-in-the-middle ───────────────────────────────
def test_lost_in_middle():
    print(f"\n{'=' * 60}")
    print(f"  TEST 3: Lost-in-the-Middle Reordering")
    print(f"{'=' * 60}")

    docs = [
        "Python is a programming language.",
        "FastAPI is a modern web framework.",
        "PostgreSQL is a relational database.",
        "GitHub Actions is a CI/CD tool.",
        "React is a JavaScript UI library.",
        "Docker containers wrap application code.",
        "Kubernetes orchestrates containers.",
        "TypeScript adds static types to JavaScript.",
    ]

    query = "Python programming and web frameworks"
    scores = score_by_overlap(query, docs)
    reordered = reorder_lost_in_middle(docs, scores)

    print(f"  Query: '{query}'")
    print(f"\n  Original order (by relevance score):")
    for d, s in sorted(zip(docs, scores), key=lambda x: x[1], reverse=True):
        print(f"    [{s:.3f}] {d}")

    print(f"\n  Lost-in-middle order (high, low, high, low...):")
    for d in reordered:
        s = score_by_overlap(query, [d])[0]
        print(f"    [{s:.3f}] {d}")


# ─── Test 4: Conversation History Compression ───────────────────
def test_conversation_compression():
    print(f"\n{'=' * 60}")
    print(f"  TEST 4: Conversation History Compression")
    print(f"{'=' * 60}")

    cm = ConversationManager(max_history_tokens=500)

    for i in range(1, 11):
        cm.add_turn("user", f"Tell me about topic {i} in detail with lots of information")
        cm.add_turn("assistant", f"Here is detailed information about topic {i}. " * 50)

    print(f"  After 10 turns:")
    print(f"  Turns: {len(cm.turns)}, Summaries: {len(cm.summaries)}")
    print(f"  Total tokens: {cm.token_count()}")
    print(f"  Context preview:\n{cm.get_context()[:300]}")


# ─── Test 5: Dynamic Tool Selection ─────────────────────────────
def test_tool_selection():
    print(f"\n{'=' * 60}")
    print(f"  TEST 5: Dynamic Tool Selection")
    print(f"{'=' * 60}")

    queries = [
        "Fix the bug in the auth module",
        "Search for information about Llama 3",
        "Send an email to the team",
    ]

    for q in queries:
        tools, tokens = select_tools(q, budget=300)
        intents = classify_intent(q)
        print(f"\n  Query: '{q}'")
        print(f"  Intent: {intents} | Selected: {list(tools.keys())} ({tokens} tokens)")


# ─── Test 6: Full Pipeline with MiniMax ───────────────────────
def test_full_pipeline():
    print(f"\n{'=' * 60}")
    print(f"  TEST 6: Full Context Assembly → MiniMax Call")
    print(f"{'=' * 60}")

    budget = ContextBudget(max_tokens=4000, generation_reserve=1000)
    system_prompt = "You are a helpful Python programming tutor. Answer questions about Python concisely."
    retrieved_context = (
        "Python 3.12 introduced type parameter syntax for generic classes: class MyList[T]: ...\n"
        "Python's walrus operator := assigns values within expressions: if (n := len(data)) > 10: ...\n"
        "Match statements (structural pattern matching) were added in Python 3.10.\n"
        "The pathlib module provides object-oriented filesystem paths."
    )
    conversation = "user: What is new in Python 3.12?\nassistant: Python 3.12 introduced type parameter syntax!"
    query = "Tell me about Python 3.12 features"

    alloc_sys, _ = budget.allocate("system", system_prompt, max_tokens=500)
    alloc_ctx, _ = budget.allocate("context", retrieved_context, max_tokens=2000)
    alloc_hist, _ = budget.allocate("history", conversation, max_tokens=500)
    alloc_q, _ = budget.allocate("query", query, max_tokens=200)

    print(budget.report())

    full_prompt = f"System: {alloc_sys}\n\nContext:\n{alloc_ctx}\n\nHistory:\n{alloc_hist}\n\nUser: {alloc_q}"
    print(f"\n  Assembled prompt ({count_tokens(full_prompt)} tokens)")

    t0 = time.time()
    response = minimax.chat.completions.create(
        model="MiniMax-M2.7",
        messages=[{"role": "user", "content": full_prompt}],
        temperature=0.3,
        max_tokens=500
    )
    latency = (time.time() - t0) * 1000
    raw = response.choices[0].message.content
    print(f"\n  MiniMax response ({latency:.0f}ms):")
    print(f"  {raw[:300]}...")


def main():
    print("=" * 70)
    print("  LESSON 05: Context Engineering — Real API Tests")
    print("=" * 70)

    test_token_counting()
    test_budget_manager()
    test_lost_in_middle()
    test_conversation_compression()
    test_tool_selection()
    test_full_pipeline()

    print("\n\n" + "=" * 70)
    print("  Lesson 05 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
