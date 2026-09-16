"""Unit tests for 3-Section JIT Context Window layout and Y-Vault utility boost."""
from __future__ import annotations

import pytest
from pathlib import Path

from context_loom.compiler import CompilerConfig, ContextCompiler
from context_loom.models import RecordKind, MemoryRecord
from context_loom.engine import ContextLoom


class DummyEmbeddingProvider:
    identity = "dummy_provider"

    def __init__(self, dimension: int = 8):
        self.dimension = dimension

    def embed(self, text: str) -> list[float]:
        return [0.1] * self.dimension


def test_3_section_jit_rendering_order(tmp_path: Path):
    embedder = DummyEmbeddingProvider(dimension=8)
    loom = ContextLoom(
        database=tmp_path / "test_loom.sqlite",
        embedding_provider=embedder,
        install_default_capabilities=False,
    )

    # 1. Log conversational turns (recency)
    loom.log(speaker="User", content="Hi there, how are you today?", kind=RecordKind.USER)
    loom.log(speaker="Assistant", content="I am doing well!", kind=RecordKind.ASSISTANT)

    # 2. Log direct evidence observation
    direct_obs = loom.log(speaker="User", content="Melanie stated that her favorite hobby is oil painting.", kind=RecordKind.FACT)

    # 3. Log observation for vault evidence & create Y-concept link for subjective relevance / multi-hop
    obs = loom.log(speaker="User", content="Melanie loves painting sunrises on canvas.", kind=RecordKind.OBSERVATION)
    cnode = loom.graph_store.create_vector_concept_node(name="Artistic Hobbies", parent_id="self_root", y_depth=1)
    loom.graph_store.link_x_membrane_to_y_concept("hobby", cnode.id)
    loom.graph_store.link_concept_record(cnode.id, obs.id)

    # Compile context packet for query
    packet = loom.compiler.compile("What is Melanie's favorite hobby?", scope="default", mode="reply")

    rendered = packet.rendered
    print("\n--- RENDERED PACKET ---\n", rendered)
    for m in packet.manifest:
        print("MANIFEST ITEM:", m)

    # Assert 3-section titles exist in exact order
    pos_direct = rendered.find("[DIRECT SEMANTIC & COSINE OVERLAP]")
    pos_subjective = rendered.find("[SUBJECTIVE RELEVANCE & MULTI-HOP VAULTS]")
    pos_recency = rendered.find("[RECENCY & CONVERSATIONAL CONTINUITY]")
    pos_input = rendered.find("[CURRENT INPUT]")

    assert pos_direct != -1
    assert pos_subjective != -1
    assert pos_recency != -1
    assert pos_input != -1

    # Recency & Continuity MUST be presented LAST (after subjective relevance and before current input)
    assert pos_direct < pos_subjective < pos_recency < pos_input
    loom.close()


def test_y_vault_utility_boost_prevents_gold_starvation(tmp_path: Path):
    embedder = DummyEmbeddingProvider(dimension=8)
    loom = ContextLoom(
        database=tmp_path / "test_loom.sqlite",
        embedding_provider=embedder,
        install_default_capabilities=False,
    )

    # Log 10 recency turns
    for i in range(10):
        loom.log(speaker="User", content=f"Filler chat turn number {i}", kind=RecordKind.USER)

    # Log gold observation
    gold_rec = loom.log(
        speaker="User",
        content="Melanie's friend Sarah lives in Seattle.",
        kind=RecordKind.OBSERVATION,
    )

    loom.compiler.config.minimum_utility = 0.0

    packet = loom.compiler.compile("Where does Melanie's friend live?", scope="default", mode="reply")

    # Gold record MUST be admitted
    admitted_ids = {
        rid
        for entry in packet.manifest
        if entry.admitted
        for rid in entry.source_record_ids
    }

    assert gold_rec.id in admitted_ids
    loom.close()
