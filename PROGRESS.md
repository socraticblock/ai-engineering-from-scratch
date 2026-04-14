# Phase 11 Progress

## Rules
- Wiki only gets updated with what **actually works in code**
- Read → Code → Verify → Wiki

## Lessons

| # | Lesson | Status | Wiki Updated |
|---|--------|--------|-------------|
| 01 | Prompt Engineering | ✅ | ✅ |
| 02 | Few-Shot, CoT, ToT | ✅ | ✅ |
| 03 | Structured Outputs | ✅ | ✅ |
| 04 | Embeddings | ✅ | ✅ |
| 05 | Context Engineering | ✅ | ✅ |
| 06 | RAG | ✅ | ✅ |
| 07 | Advanced RAG | ✅ | ✅ |
| 08 | Fine-Tuning LoRA | ⏭️ SKIP | GPU required |
| 09 | Function Calling | ✅ | ✅ |
| 10 | Evaluation | ✅ | ✅ |
| 11 | Caching & Cost | ✅ | ✅ |
| 12 | Guardrails | ✅ | ✅ |
| 13 | Production App | ✅ | ✅ |
| 08 | Fine-Tuning LoRA | ⏭️ SKIP | GPU required |

## Wiki Sources Created
- `ai-engineering-llm-engineering-patterns.md` - All prompt patterns verified
- `structured-outputs-with-llm.md` - JSON mode, function calling verified
- `embeddings-and-vector-search.md` - Ollama nomic-embed-text, 768d verified
- `context-engineering-llm.md` - Token budgeting, lost-in-middle verified
- `rag-pipeline.md` - Full RAG pipeline verified
- `advanced-rag-techniques.md` - BM25, RRF, query transform verified
- `llm-function-calling.md` - Tool loop, parallel calls verified
- `llm-evaluation-testing.md` - LLM-as-judge, regression suite verified
- `llm-caching-cost-optimization.md` - Exact + semantic cache verified
- `llm-guardrails-safety.md` - Injection detection, PII filtering verified

## What worked (wiki-ready)

### MiniMax-Specific
- `strip_think_block()` required before JSON parsing
- Flat schemas work in JSON mode; nested may fail
- Function calling (tool_calls) works reliably
- Parallel tool calls work
- Enum constraints in tool schemas enforced

### Ollama
- nomic-embed-text: 768d, unit-normalized, works reliably
- qwen3:8b: handles math and code correctly

### Key Verified Techniques
- Few-shot CoT: 3 examples, temp=0.0, 624 tokens avg
- Self-consistency: N=5, temp=0.7, 100% agreement on test problems
- RAG: 12.3s miss vs 0.002ms hit (exact cache)
- Semantic cache: 0.91 similarity threshold catches paraphrases
- BM25: correctly handles exact keyword matches
- RRF fusion: balances semantic + keyword rankings
- Input guardrails: injection patterns blocked
- PII detection: email, phone, SSN, credit card patterns caught
