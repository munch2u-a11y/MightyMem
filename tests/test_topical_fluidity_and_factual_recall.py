from __future__ import annotations

import pytest
from pathlib import Path

from context_loom import ContextLoom
from context_loom.models import MemoryRecord, RecordKind, EvidenceStatus
from context_loom.topical import TopicalTracker, TopicalFrame
from context_loom.compiler import ContextCompiler, CompilerConfig


class DeterministicMockEmbedder:
    identity = "det_mock_embedder"

    def embed(self, text: str) -> list[float]:
        words = text.casefold().split()
        v = [0.0] * 16
        for w in words:
            idx = sum(ord(c) for c in w) % 16
            v[idx] += 1.0
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / norm for x in v]

    def embed_many(self, texts: list[str], batch_size: int = 12) -> list[list[float]]:
        return [self.embed(t) for t in texts]


def test_topical_frame_fluidity_and_momentum_transitions(tmp_path: Path):
    """Test that topic activations evolve smoothly and momentum transitions cleanly across multi-subtopic turns."""
    db_path = tmp_path / "fluidity_mind.sqlite"
    loom = ContextLoom(db_path, embedding_provider=DeterministicMockEmbedder())
    tracker = TopicalTracker(loom.store, decay=0.75, tau=1.0, max_focus=3)

    # Turn 1: Subtopic 1 - Database Schema
    path_schema = [{
        "target_concept_id": "concept:database_schema",
        "target_concept_name": "Database Schema",
        "endpoint_score": 0.90,
        "path_nodes": ["self_root", "concept:database_schema"],
    }]
    frame1 = tracker.observe_turn(y_paths=path_schema, output_trunk="LOOK")
    assert "Database Schema" in frame1.focus_topics
    assert frame1.momentum.get("Database Schema") in {"rising", "sustained"}
    assert frame1.drift == 0.0

    # Turn 2: Subtopic 2 - Indexing Strategy (Subtopic 1 is on the path, so both are active & fluid)
    path_index = [{
        "target_concept_id": "concept:indexing_strategy",
        "target_concept_name": "Indexing Strategy",
        "endpoint_score": 0.95,
        "path_nodes": ["self_root", "concept:database_schema", "concept:indexing_strategy"],
    }]
    frame2 = tracker.observe_turn(y_paths=path_index, output_trunk="DO")
    assert "Indexing Strategy" in frame2.focus_topics
    assert "Database Schema" in frame2.focus_topics  # Coherent overlap maintained
    assert frame2.momentum.get("Indexing Strategy") == "rising"

    # Turn 3: Subtopic 3 - Table Partitioning (Subtopic 1 decays as conversation moves forward)
    path_part = [{
        "target_concept_id": "concept:table_partitioning",
        "target_concept_name": "Table Partitioning",
        "endpoint_score": 0.88,
        "path_nodes": ["self_root", "concept:indexing_strategy", "concept:table_partitioning"],
    }]
    frame3 = tracker.observe_turn(y_paths=path_part, output_trunk="DO")
    assert "Table Partitioning" in frame3.focus_topics
    assert frame3.momentum.get("Table Partitioning") == "rising"
    assert frame3.momentum.get("Database Schema") == "fading"  # Smooth decay to fading status

    loom.close()


def test_topical_orientation_assists_factual_recall(tmp_path: Path):
    """Test that active topical orientation helps JIT context compiler retrieve the exact factual record for ambiguous queries."""
    db_path = tmp_path / "recall_assist_mind.sqlite"
    loom = ContextLoom(db_path, embedding_provider=DeterministicMockEmbedder())

    # Store exact factual records across multiple topics
    rec_db = loom.remember_preference(
        "User table schema requires a B-Tree index on user_id and email_hash.",
        scope="default",
        record_id="rec_db_indexing",
        created_at=100.0,
        importance=0.9,
    )
    rec_cache = loom.remember_preference(
        "Redis cache TTL is strictly set to 3600 seconds for session tokens.",
        scope="default",
        record_id="rec_cache_ttl",
        created_at=101.0,
        importance=0.9,
    )

    # Link concepts in Y-cypher entity graph
    loom.graph_store.create_vector_concept_node(
        name="User Table Indexing",
        embedding=DeterministicMockEmbedder().embed("User table schema B-Tree index user_id"),
        y_depth=1,
    )
    loom.graph_store.compact_concept_nodes(embedding_provider=DeterministicMockEmbedder())

    # Initialize tracker and observe database indexing topic
    tracker = TopicalTracker(loom.store)
    tracker.observe_turn(
        y_paths=[{
            "target_concept_id": "concept:user_table_indexing",
            "target_concept_name": "User Table Indexing",
            "endpoint_score": 0.92,
            "path_nodes": ["self_root", "concept:user_table_indexing"],
        }],
        output_trunk="LOOK",
    )

    # Ambiguous query that lacks exact table name
    ambiguous_query = "What index type did we choose?"
    
    # Compile packet with topical orientation injected
    topical_header = tracker.frame().rendered
    packet = loom.compiler.compile(
        current_input=f"{topical_header}\n\nQuery: {ambiguous_query}",
        scope="default",
    )

    # Verify that the DB indexing fact was admitted into the JIT context packet
    assert "B-Tree index on user_id" in packet.rendered
    assert rec_db.id in [c.source_record_ids[0] for c in packet.manifest if c.admitted and c.source_record_ids]

    loom.close()
