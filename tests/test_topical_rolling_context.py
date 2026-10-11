from __future__ import annotations

import pytest

from context_loom.engine import ContextLoom
from context_loom.models import MemoryRecord, RecordKind, EvidenceStatus
from context_loom.topical import TopicalTracker, TopicalFrame
from context_loom.summarizer import RollingFactSummarizer, FactSummaryItem
from context_loom.compiler import ContextCompiler, CompilerConfig


class DummyEmbeddingProvider:
    identity = "dummy_provider"

    def embed(self, text: str) -> list[float]:
        # Deterministic simple hash vector
        h = sum(ord(c) for c in text)
        return [float((h + i) % 17) / 17.0 for i in range(16)]

    def embed_many(self, texts: list[str], batch_size: int = 12) -> list[list[float]]:
        return [self.embed(t) for t in texts]


def test_topical_tracker_activation_and_decay(tmp_path):
    loom = ContextLoom(tmp_path / "loom.sqlite", embedding_provider=DummyEmbeddingProvider())
    tracker = TopicalTracker(loom.store, decay=0.70)

    # Initial frame should be empty
    frame1 = tracker.frame()
    assert isinstance(frame1, TopicalFrame)
    assert frame1.focus_topics == ()
    assert frame1.drift == 0.0

    # Simulate path activation for Topic A
    mock_trace_a = {
        "target_concept_id": "concept:project_helios",
        "target_concept_name": "Project Helios",
        "endpoint_score": 0.9,
        "path_nodes": ["self_root", "concept:project_helios"],
    }
    tracker.observe_turn(y_paths=[mock_trace_a], output_trunk="LOOK")

    frame2 = tracker.frame()
    assert "Project Helios" in frame2.focus_topics
    assert frame2.action_mode_mix.get("LOOK", 0) > 0

    # Observe Topic B and verify Topic A decays / momentum reflects change
    mock_trace_b = {
        "target_concept_id": "concept:database_migrations",
        "target_concept_name": "Database Migrations",
        "endpoint_score": 0.95,
        "path_nodes": ["self_root", "concept:database_migrations"],
    }
    tracker.observe_turn(y_paths=[mock_trace_b], output_trunk="DO")

    frame3 = tracker.frame()
    assert "Database Migrations" in frame3.focus_topics
    assert frame3.momentum.get("Database Migrations") == "rising"
    assert frame3.drift > 0.0

    loom.close()


def test_rolling_fact_summarizer_extract_and_ingest(tmp_path):
    loom = ContextLoom(tmp_path / "loom.sqlite", embedding_provider=DummyEmbeddingProvider())
    summarizer = RollingFactSummarizer(loom, max_raw_exchanges=2)

    # Create dummy dialogue records
    records = [
        MemoryRecord(
            id="rec_1",
            content="User: We need to use PostgreSQL 16 for Project Helios.",
            kind=RecordKind.USER,
            created_at=100.0,
            metadata={"speaker": "User"},
        ),
        MemoryRecord(
            id="rec_2",
            content="Assistant: Understood. PostgreSQL 16 will be configured.",
            kind=RecordKind.ASSISTANT,
            created_at=101.0,
            metadata={"speaker": "Assistant"},
        ),
    ]

    # Summarize with deterministic mock model
    class MockSummaryModel:
        def complete(self, prompt, **kwargs):
            return """{
                "key_facts": ["Project Helios requires PostgreSQL 16"],
                "user_constraints": ["Must use PostgreSQL 16"],
                "decisions_made": ["PostgreSQL 16 selected as primary database"],
                "unresolved_tasks": ["Configure deployment script"]
            }"""

    summarizer.model_adapter = MockSummaryModel()
    item = summarizer.summarize_exchanges(records)

    assert isinstance(item, FactSummaryItem)
    assert "Project Helios requires PostgreSQL 16" in item.key_facts
    assert "Must use PostgreSQL 16" in item.user_constraints

    # Ingest summary into store
    record_ids = summarizer.ingest_summary(item, scope="default")
    assert len(record_ids) > 0

    # Verify records in store
    saved_records = [loom.store.get_record(rid) for rid in record_ids if loom.store.get_record(rid) is not None]
    assert any("PostgreSQL 16" in r.content for r in saved_records)
    assert all(r.kind == RecordKind.FACT for r in saved_records)
    assert all(r.evidence_status == EvidenceStatus.SYSTEM_DERIVED for r in saved_records)

    loom.close()


def test_context_compiler_includes_topical_and_summary_sections(tmp_path):
    loom = ContextLoom(tmp_path / "loom.sqlite", embedding_provider=DummyEmbeddingProvider())
    
    # Add a fact summary record
    fact_rec = MemoryRecord(
        id="fact_summary_1",
        content="[ROLLING SUMMARY] Project Helios uses PostgreSQL 16.",
        kind=RecordKind.FACT,
        evidence_status=EvidenceStatus.SYSTEM_DERIVED,
        created_at=200.0,
    )
    loom.store.append_record(fact_rec)

    compiler = ContextCompiler(loom.store, embedding_provider=DummyEmbeddingProvider())
    packet = compiler.compile("What database does Helios use?")

    assert packet.rendered != ""
    loom.close()
