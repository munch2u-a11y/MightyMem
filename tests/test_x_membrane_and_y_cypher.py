"""Tests for X-Axis Membrane & Y-Axis Experience Graph Dual-Cypher Architecture."""
from __future__ import annotations

import pytest
from pathlib import Path

from context_loom.x_membrane import XMembrane, XMembraneNode, ShadowXCypher
from context_loom.entity_graph import EntityGraphStore, VectorConceptNode
from context_loom.store import LoomStore
from context_loom.engine import ContextLoom
from context_loom.models import RecordKind


class DummyEmbeddingProvider:
    def __init__(self, dimension: int = 8):
        self.dimension = dimension

    def embed(self, text: str) -> list[float]:
        return [0.1] * self.dimension


def test_x_membrane_counters_and_regional_softmax(tmp_path: Path):
    store = LoomStore(tmp_path / "test_loom.sqlite")
    membrane = XMembrane(store)

    # Initial activation of n-grams / entities
    membrane.activate(["Tom", "enjoys", "painting", "Tom enjoys painting"])
    membrane.activate(["Tom", "enjoys", "painting"])

    tom_node = membrane.get_node("tom")
    assert tom_node is not None
    assert tom_node.activation_count == 2

    phrase_node = membrane.get_node("tom enjoys painting")
    assert phrase_node is not None
    assert phrase_node.activation_count == 1

    # Regional softmax weights should be non-zero and normalized
    weights = membrane.get_regional_softmax_weights(["tom", "enjoys", "painting"])
    assert "tom" in weights
    assert weights["tom"] > 0.0
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-3)
    store.close()


def test_shadow_x_cypher_parsing():
    cypher = ShadowXCypher()
    terms, tuple_groupings = cypher.parse_input("Tom enjoys painting in Paris.")

    assert "tom" in terms
    assert "painting" in terms
    assert "paris" in terms
    assert any("tom" in tup for tup in tuple_groupings)


def test_y_cypher_path_of_least_resistance(tmp_path: Path):
    store = LoomStore(tmp_path / "test_loom.sqlite")
    graph_store = EntityGraphStore(store)
    embedder = DummyEmbeddingProvider(dimension=8)

    # Ensure Self node exists at root Y=0
    self_node = graph_store.get_or_create_self_node()
    assert self_node.id == "self_root"

    # Create Y-axis concept nodes branching from Self
    concept1 = graph_store.create_vector_concept_node(
        name="Artistic Hobbies",
        embedding=[0.1] * 8,
        parent_id=self_node.id,
        y_depth=1,
    )
    concept2 = graph_store.create_vector_concept_node(
        name="Oil Painting",
        embedding=[0.1] * 8,
        parent_id=concept1.id,
        y_depth=2,
    )

    # Cross-axis link to X-membrane nodes
    graph_store.link_x_membrane_to_y_concept("painting", concept2.id)

    # Execute Y-Cypher path search from query
    paths = graph_store.y_cypher_find_paths(
        query="painting",
        x_terms=["painting"],
        embedding_provider=embedder,
        top_k=5,
    )

    assert len(paths) > 0
    top_path = paths[0]
    assert top_path.target_concept_id == concept2.id
    assert self_node.id in top_path.node_ids
    store.close()


def test_dual_ingestion_and_vault_routing(tmp_path: Path):
    embedder = DummyEmbeddingProvider(dimension=8)
    loom = ContextLoom(
        database=tmp_path / "test_loom.sqlite",
        embedding_provider=embedder,
        install_default_capabilities=False,
    )

    # Log turn input -> should update X-membrane counters (+1) and route into Y-concept vaults
    rec = loom.log(
        speaker="Tom",
        content="I really enjoy oil painting in my studio.",
        kind=RecordKind.OBSERVATION,
    )

    assert rec.id is not None

    # Check X-membrane counter updated
    tom_node = loom.x_membrane.get_node("tom")
    assert tom_node is not None
    assert tom_node.activation_count >= 1

    # Check vault routing
    vault_records = loom.graph_store.get_records_in_concept_vaults_for_term("painting")
    assert rec.id in vault_records

    loom.close()
