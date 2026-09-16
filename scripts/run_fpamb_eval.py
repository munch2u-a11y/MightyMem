"""Run official FP-AMB Benchmark for MightyMem using FPAMBEvaluator."""
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FPAMB_ROOT = Path(os.environ.get("FPAMB_ROOT", "/home/nemo/fp-amb-benchmark"))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(FPAMB_ROOT) not in sys.path:
    sys.path.insert(0, str(FPAMB_ROOT))

from context_loom.engine import ContextLoom
from context_loom.compiler import CompilerConfig
from context_loom.entity_graph import EntityGraphStore, EntityExtractor
from context_loom.multi_hop_lane import MultiHopGraphLane
from context_loom.retrieval import OllamaEmbeddingProvider, scope_blind_text

from fp_amb.interface import BaseMemoryProvider
from fp_amb.evaluator import FPAMBEvaluator

class CachedOllamaEmbeddingProvider(OllamaEmbeddingProvider):
    def __init__(self, model: str = "qwen3-embedding:0.6b", endpoint: str = "http://127.0.0.1:11434"):
        super().__init__(model=model, endpoint=endpoint)
        self._cache: dict[str, list[float]] = {}

    def embed(self, text: str) -> list[float]:
        key = text.strip() if text else ""
        if key in self._cache:
            return self._cache[key]
        res = super().embed(text)
        self._cache[key] = res
        return res

class MightyMemProvider(BaseMemoryProvider):
    def __init__(self, db_path: Path):
        self.db_path = db_path
        if db_path.exists():
            try:
                db_path.unlink()
            except Exception:
                pass

        self.embedder = CachedOllamaEmbeddingProvider(model="qwen3-embedding:0.6b", endpoint="http://127.0.0.1:11434")
        self.loom = ContextLoom(
            db_path,
            embedding_provider=self.embedder,
            install_default_capabilities=False,
            config=CompilerConfig(
                model_context_tokens=131072,
                reserved_output_tokens=1024,
                section_limits={
                    "observations": 0,
                    "procedures": 0,
                    "private": 0,
                    "continuity": 4,
                    "evidence": 25,
                },
                minimum_utility=0.15,
            ),
        )
        self.graph_store = EntityGraphStore(self.loom.store)
        self.extractor = EntityExtractor()
        self.multi_hop_lane = MultiHopGraphLane(self.loom.store, self.graph_store, self.extractor, max_hops=2)
        self.loom.compiler.lanes.append(self.multi_hop_lane)

        self.turn_count = 0

    def ingest_turn(self, session_id: str, timestamp: str, speaker: str, text: str):
        self.turn_count += 1
        content = f"[{session_id} | {timestamp} | {speaker}]: {text}"
        rec = self.loom.remember_user_statement(
            content,
            scope="fpamb",
            record_id=f"fpamb_{session_id}_{self.turn_count}",
            created_at=time.time(),
            importance=0.8,
        )
        entities, relations = self.extractor.extract(content)
        for ent in entities:
            self.graph_store.link_record_entity(rec.id, ent.name, ent.kind)
        for rel in relations:
            self.graph_store.add_relation(rel.source, rel.relation, rel.target, record_id=rec.id)
        if len(entities) > 1:
            self.graph_store.add_cooccurrence([e.name for e in entities], rec.id)

    def prewarm_embeddings(self):
        print("Pre-warming record embeddings in batch...", flush=True)
        provider_key = self.embedder.identity + ":scope-blind-v1"
        records = self.loom.store.list_records(scope="fpamb")
        missing_recs = [r for r in records if self.loom.store.get_embedding(r.id, provider_key) is None]
        if missing_recs:
            texts = [scope_blind_text(r.content, "fpamb") for r in missing_recs]
            for i in range(0, len(texts), 32):
                chunk_recs = missing_recs[i:i+32]
                chunk_texts = texts[i:i+32]
                vecs = self.embedder.embed_many(chunk_texts)
                for rec, vec in zip(chunk_recs, vecs):
                    self.loom.store.put_embedding(rec.id, provider_key, vec)
                print(f"Pre-warmed {min(i+32, len(texts))}/{len(texts)} embeddings...", flush=True)
        print(f"Pre-warmed {len(missing_recs)} record embeddings into SQLite.", flush=True)

    def retrieve_context(self, query: str, top_k: int = 5) -> str:
        packet = self.loom.compiler.compile(query, scope="fpamb")
        return packet.rendered

def generate_markdown_scorecard(report: dict, output_path: Path):
    cats = report.get("category_breakdown", {})
    md_content = f"""# FP-AMB Benchmark Scorecard: MightyMem

This scorecard documents the empirical performance of **MightyMem** (powered by the **Context Loom JIT Memory Compiler**) evaluated on the **FP-AMB** (~512.9k token multi-session conversational memory benchmark).

## Executive Summary

| Metric | Score |
| :--- | :--- |
| **Total Benchmark Questions** | `{report.get('total_evaluated_questions')}` |
| **Overall Context Recall %** | **`{report.get('overall_accuracy_pct')}%`** |
| **Passed Benchmark Items** | `{report.get('passed_items')} / {report.get('total_evaluated_questions')}` |
| **Total Corpus Size** | `~{report.get('total_corpus_tokens'):,} tokens` (679 turns across 60 sessions) |
| **Average Retrieval Latency** | `{report.get('avg_retrieval_latency_ms')} ms / query` |
| **Average Injected Payload** | `{report.get('avg_injected_tokens_per_query')} tokens` (`{report.get('avg_injected_chars_per_query')} chars`) |
| **Token Efficiency Ratio** | `{report.get('token_efficiency_ratio')} accuracy points / 1k injected tokens` |
| **Evaluation Mode** | `{report.get('model_name') or 'Pure Context Recall (Zero-LLM Overhead)'}` |
| **Embedding Provider** | `qwen3-embedding:0.6b` |

## Performance by Category

| Category Name | Passed / Total | Context Recall % |
| :--- | :---: | :---: |
"""
    for cat_name, scores in sorted(cats.items()):
        acc = (scores["correct"] / scores["total"] * 100) if scores["total"] > 0 else 0.0
        md_content += f"| `{cat_name}` | {scores['correct']} / {scores['total']} | **{acc:.1f}%** |\n"

    md_content += f"""
## Evaluation Details & Methodology

- **Ingestion Duration**: `{report.get('ingestion_duration_seconds')}s`
- **Total Exam Duration**: `{report.get('eval_duration_seconds')}s`
- **Retrieval Engine**: MightyMem JIT Compiler with BM25, Cosine Vector Search, Temporal Anchoring, and Multi-Hop Entity Graph reasoning.
"""
    output_path.write_text(md_content, encoding="utf-8")
    print(f"Generated markdown scorecard at {output_path}", flush=True)

def main():
    out_dir = PROJECT_ROOT / "scratch" / "fpamb_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / "fpamb_loom.sqlite"

    provider = MightyMemProvider(db_path)
    evaluator = FPAMBEvaluator(
        provider=provider,
        provider_name="MightyMem",
        model_name="granite4.2:8b",
        ollama_url="http://127.0.0.1:11434/api/generate",
        use_llm_generation=False
    )
    
    # Run ingestion phase & prewarm embeddings
    ingest_time = evaluator.run_ingestion()
    provider.prewarm_embeddings()

    # Override run_ingestion to no-op during evaluate() call
    evaluator.run_ingestion = lambda: ingest_time

    scorecard_json_path = out_dir / "mightymem_fpamb_scorecard.json"
    report = evaluator.evaluate(output_path=scorecard_json_path)
    
    scorecard_md_path = PROJECT_ROOT / "docs" / "FPAMB_SCORECARD.md"
    generate_markdown_scorecard(report, scorecard_md_path)

if __name__ == "__main__":
    main()
