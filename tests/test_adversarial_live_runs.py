from __future__ import annotations

import pytest
import time
from pathlib import Path

from context_loom import ContextLoom
from context_loom.models import MemoryRecord, RecordKind, EvidenceStatus
from context_loom.topical import TopicalTracker, TopicalFrame
from context_loom.summarizer import RollingFactSummarizer, FactSummaryItem
from context_loom.compiler import ContextCompiler, CompilerConfig
from context_loom.habitus_agent import HabitusLoomAgent


class MockEmbeddingProvider:
    identity = "mock_provider"

    def embed(self, text: str) -> list[float]:
        h = sum(ord(c) for c in text)
        return [float((h + i) % 19) / 19.0 for i in range(16)]

    def embed_many(self, texts: list[str], batch_size: int = 12) -> list[list[float]]:
        return [self.embed(t) for t in texts]


def test_adversarial_topic_switch_and_drift(tmp_path: Path):
    """Test multi-turn conversation with sudden topic switches (distractors) and verify topical drift."""
    db_path = tmp_path / "adversarial_mind.sqlite"
    loom = ContextLoom(db_path, embedding_provider=MockEmbeddingProvider())
    tracker = TopicalTracker(loom.store, decay=0.70)

    # Turn 1: Topic A (Kubernetes)
    path_k8s = [{
        "target_concept_id": "concept:k8s_deployment",
        "target_concept_name": "Kubernetes Deployment",
        "endpoint_score": 0.92,
        "path_nodes": ["self_root", "concept:k8s_deployment"],
    }]
    frame1 = tracker.observe_turn(y_paths=path_k8s, output_trunk="LOOK")
    assert "Kubernetes Deployment" in frame1.focus_topics
    assert frame1.drift == 0.0

    # Turn 2: Topic A continued (Kubernetes scaling)
    path_k8s_scale = [{
        "target_concept_id": "concept:k8s_deployment",
        "target_concept_name": "Kubernetes Deployment",
        "endpoint_score": 0.88,
        "path_nodes": ["self_root", "concept:k8s_deployment"],
    }]
    frame2 = tracker.observe_turn(y_paths=path_k8s_scale, output_trunk="LOOK")
    assert "Kubernetes Deployment" in frame2.focus_topics
    assert frame2.drift < 0.05  # Low drift on topic continuation

    # Turn 3: Distractor Topic B (Baking Recipes) - sudden topic switch!
    path_baking = [{
        "target_concept_id": "concept:sourdough_bread",
        "target_concept_name": "Sourdough Bread Recipe",
        "endpoint_score": 0.96,
        "path_nodes": ["self_root", "concept:sourdough_bread"],
    }]
    frame3 = tracker.observe_turn(y_paths=path_baking, output_trunk="SPEAK")
    assert "Sourdough Bread Recipe" in frame3.focus_topics
    assert frame3.drift > 0.10  # Drift detected due to adversarial topic switch

    loom.close()


def test_adversarial_routine_task_execution_with_receipts(tmp_path: Path):
    """Test routine task execution (file creation/inspection) with receipt verification."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    db_path = tmp_path / "task_mind.sqlite"

    loom = ContextLoom(db_path, embedding_provider=MockEmbeddingProvider())
    
    # Log user preference
    loom.remember_preference("My preferred build script format is Python setuptools.", scope="alpha", importance=0.9)

    # Begin task
    loom.begin_task(
        task_id="task_build_1",
        goal="Create build.py script and verify file hash",
        scope="alpha",
        stage="inspect",
        focus_stack=["inspect workspace", "write build.py"],
        now=100.0,
    )

    # Unverified tool result claim (should fail closed)
    unverified_obs = loom.record_tool_result(
        content="Assistant claims it created build.py without running runtime.",
        task_id="task_build_1",
        scope="alpha",
        verified=False,
        receipt_id="unverified-1",
        verifier="none",
        next_stage="act",
        now=105.0,
    )
    packet1 = loom.compiler.compile("Check if build.py exists", scope="alpha", task_id="task_build_1", now=110.0)
    assert loom.store.get_task("task_build_1").stage == "inspect"  # Stage did NOT advance

    # Verified tool execution receipt
    verified_obs = loom.record_tool_result(
        content="Verified execution: build.py created, bytes=412, sha256=a1b2c3d4",
        task_id="task_build_1",
        scope="alpha",
        verified=True,
        receipt_id="verified-receipt-1",
        verifier="runtime_hash",
        next_stage="act",
        now=115.0,
    )
    packet2 = loom.compiler.compile("Continue to build execution", scope="alpha", task_id="task_build_1", now=120.0)
    assert loom.store.get_task("task_build_1").stage == "act"  # Stage successfully advanced

    loom.close()


def test_adversarial_cross_restart_memory_persistence(tmp_path: Path):
    """Test memory retention over process restarts with rolling factual summary ingestion."""
    db_path = tmp_path / "restart_mind.sqlite"

    # --- Session 1 ---
    loom1 = ContextLoom(db_path, embedding_provider=MockEmbeddingProvider())
    loom1.remember_preference("User name is Maya, Senior Systems Reliability Engineer.", scope="default", importance=0.95)
    loom1.remember_preference("Critical constraint: Database maintenance window is strictly Sundays 02:00 UTC.", scope="default", importance=0.98)
    
    # Ingest rolling fact summary
    summarizer1 = RollingFactSummarizer(loom1)
    summary_item = FactSummaryItem(
        key_facts=("Maya is Lead SRE", "Maintenance window is Sundays 02:00 UTC"),
        user_constraints=("Maintenance window is Sundays 02:00 UTC",),
        decisions_made=("Approved Sundays 02:00 UTC schedule",),
        unresolved_tasks=("Deploy Prometheus alert rules",),
    )
    summarizer1.ingest_summary(summary_item, scope="default")
    loom1.close()

    # --- Session 2 (Cold Process Restart) ---
    loom2 = ContextLoom(db_path, embedding_provider=MockEmbeddingProvider())
    packet = loom2.compiler.compile("When is our database maintenance window and who leads SRE?", scope="default")

    # Verify 100% factual recall across cold restart
    assert "Sundays 02:00 UTC" in packet.rendered
    assert "Maya" in packet.rendered
    loom2.close()
