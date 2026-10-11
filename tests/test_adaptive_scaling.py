from __future__ import annotations

import pytest
from pathlib import Path

from context_loom import ContextLoom, CompilerConfig
from context_loom.compiler import AdaptiveBudgetPolicy
from context_loom.topical import TopicalTracker, TopicalFrame


class MockEmbedder:
    identity = "mock_embedder"

    def embed(self, text: str) -> list[float]:
        v = [0.1] * 16
        return v

    def embed_many(self, texts: list[str], batch_size: int = 12) -> list[list[float]]:
        return [self.embed(t) for t in texts]


def test_topical_frame_exposes_max_depth():
    """Verify TopicalFrame tracks the maximum Y-cypher depth of active focus topics."""
    db_path = ":memory:"
    loom = ContextLoom(db_path, embedding_provider=MockEmbedder())
    tracker = TopicalTracker(loom.store)

    path_deep = [{
        "target_concept_id": "concept:partitioning",
        "target_concept_name": "Table Partitioning",
        "endpoint_score": 0.90,
        "path_nodes": ["self_root", "concept:database", "concept:indexing", "concept:partitioning"],
    }]
    frame = tracker.observe_turn(y_paths=path_deep, output_trunk="LOOK")
    
    assert hasattr(frame, "max_depth")
    assert frame.max_depth == 3
    loom.close()


def test_adaptive_context_budgeting_scales_with_depth_and_intent(tmp_path: Path):
    """Verify CompilerConfig dynamic budget scaling from 800 tokens (shallow) to 1300 tokens (deep multi-hop)."""
    db_path = tmp_path / "adaptive.sqlite"
    policy = AdaptiveBudgetPolicy(
        enabled=True,
        base_tokens=800,
        min_tokens=750,
        max_tokens=1450,
        depth_scaling_step=250,
        multi_hop_boost=300,
    )
    config = CompilerConfig(
        model_context_tokens=4096,
        reserved_output_tokens=400,
        adaptive_budget=policy,
    )

    # Test shallow single-hop query budget
    budget_shallow = config.calculate_adaptive_target(max_depth=1, is_multi_hop=False)
    assert budget_shallow == 800

    # Test medium multi-hop query budget
    budget_deep = config.calculate_adaptive_target(max_depth=2, is_multi_hop=True)
    assert budget_deep == 1350  # 800 + (250 * (2 - 1)) + 300 = 1350

    # Test budget capping at max_tokens (1450 tokens)
    budget_capped = config.calculate_adaptive_target(max_depth=3, is_multi_hop=True)
    assert budget_capped == 1450  # 800 + (250 * 2) + 300 = 1600 -> capped at 1450
