# Benchmarks & Empirical Evaluation

Context Loom is continuously evaluated using benchmark suites including **FP-AMB** (~512.9k token multi-session conversational benchmark) and **LoCoMo** (Long Context Memory Evaluation).

## FP-AMB Scorecard

For complete empirical evaluation results across 281 benchmark questions and 9 categories, see the **[FP-AMB Scorecard](FPAMB_SCORECARD.md)**.

## LoCoMo Benchmark Reference Results

| Metric | Context Loom Result |
| :--- | :--- |
| **Overall Micro-Recall** | **55.70%** |
| **Overall Full Evidence Hit Rate** | **57.75%** |
| **Temporal Recall** | **86.02%** |
| **Adversarial Recall** | **63.16%** |
| **Single-Hop Recall** | **61.65%** |
| **Multi-Hop Recall** | **35.52%** |
| **Average Context Tokens** | **1,088 tokens** |

## Comparison with Brute-Force Baselines

Compared to traditional brute-force TF-IDF / BM25 search:
- **Token Efficiency**: Context Loom reduces context overhead by over **70%** by compiling bounded JIT packets (1,088 tokens avg vs 4,000+ tokens).
- **Small LLM Effectiveness**: Small parameter models (Granite 4.2 8B, Qwen 3.5 4B) achieve SOTA factual recall precision when fed Context Loom JIT packets.

