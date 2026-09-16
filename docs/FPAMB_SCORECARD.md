# FP-AMB Benchmark Scorecard: MightyMem

This scorecard documents the empirical performance of **MightyMem** (powered by the **Context Loom JIT Memory Compiler**) evaluated on the **FP-AMB** (~512.9k token multi-session conversational memory benchmark).

## Executive Summary

| Metric | Score |
| :--- | :--- |
| **Total Benchmark Questions** | `262` |
| **Overall Context Recall %** | **`79.4%`** |
| **Passed Benchmark Items** | `208.0 / 262` |
| **Total Corpus Size** | `~512,889 tokens` (679 turns across 60 sessions) |
| **Average Retrieval Latency** | `35826.46 ms / query` |
| **Average Injected Payload** | `719.6 tokens` (`5187.0 chars`) |
| **Token Efficiency Ratio** | `110.32 accuracy points / 1k injected tokens` |
| **Evaluation Mode** | `Pure Context Recall (Zero-LLM Overhead)` |
| **Embedding Provider** | `qwen3-embedding:0.6b` |

## Performance by Category

| Category Name | Passed / Total | Context Recall % |
| :--- | :---: | :---: |
| `Adaptability & Fact Correction Overwrites` | 14.0 / 18 | **77.8%** |
| `Adversarial Defense & Gaslighting Robustness` | 34.0 / 43 | **79.1%** |
| `Cross-Session Multi-Hop Reasoning` | 38.0 / 44 | **86.4%** |
| `Self-Referential & Procedural Tool Memory` | 28.0 / 31 | **90.3%** |
| `Single-Hop Fact Recall` | 26.0 / 35 | **74.3%** |
| `Source Credibility & Conflict Resolution` | 6.0 / 7 | **85.7%** |
| `Speaker Attribution Traps` | 14.0 / 14 | **100.0%** |
| `Temporal Reasoning & Session Math` | 18.0 / 35 | **51.4%** |
| `Unanswerable & Absent Memory Refusal` | 30.0 / 35 | **85.7%** |

## Evaluation Details & Methodology

- **Ingestion Duration**: `40.82s`
- **Total Exam Duration**: `9386.57s`
- **Retrieval Engine**: MightyMem JIT Compiler with BM25, Cosine Vector Search, Temporal Anchoring, and Multi-Hop Entity Graph reasoning.
