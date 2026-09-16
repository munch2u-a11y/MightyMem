"""Unit tests for Topical Node Density & Overlap Reflection Engine."""
from __future__ import annotations

from pathlib import Path

from context_loom.compiler import ContextCompiler
from context_loom.engine import ContextLoom
from context_loom.models import RecordKind, EvidenceStatus
from context_loom.reflection import TopicalReflectionEngine


class DummyEmbeddingProvider:
    identity = "dummy_provider"

    def __init__(self, dimension: int = 8):
        self.dimension = dimension

    def embed(self, text: str) -> list[float]:
        return [0.1] * self.dimension


def test_topical_node_density_detection(tmp_path: Path):
    embedder = DummyEmbeddingProvider()
    loom = ContextLoom(
        database=tmp_path / "test_reflection.sqlite",
        embedding_provider=embedder,
        install_default_capabilities=False,
    )

    # Create a Y-concept node "Classical Music"
    cnode = loom.graph_store.create_vector_concept_node(
        name="Classical Music", parent_id="self_root", y_depth=1
    )

    # Log 3 related observations
    r1 = loom.log(speaker="User", content="I love listening to Bach concertos.", kind=RecordKind.OBSERVATION)
    r2 = loom.log(speaker="User", content="Mozart symphonies always make me feel relaxed.", kind=RecordKind.OBSERVATION)
    r3 = loom.log(speaker="User", content="Classical piano music is my favorite genre.", kind=RecordKind.OBSERVATION)

    # Link all 3 records to the concept node
    loom.graph_store.link_concept_record(cnode.id, r1.id)
    loom.graph_store.link_concept_record(cnode.id, r2.id)
    loom.graph_store.link_concept_record(cnode.id, r3.id)

    engine = TopicalReflectionEngine(loom)
    dense_nodes = engine.find_dense_concept_nodes(min_records=3)

    assert len(dense_nodes) >= 1
    assert dense_nodes[0]["concept_id"] == cnode.id
    assert dense_nodes[0]["record_count"] == 3


def test_topical_reflection_consolidation_and_ingestion(tmp_path: Path):
    embedder = DummyEmbeddingProvider()
    loom = ContextLoom(
        database=tmp_path / "test_reflection_ingest.sqlite",
        embedding_provider=embedder,
        install_default_capabilities=False,
    )

    cnode = loom.graph_store.create_vector_concept_node(
        name="Classical Music", parent_id="self_root", y_depth=1
    )

    r1 = loom.log(speaker="User", content="I love listening to Bach concertos.", kind=RecordKind.OBSERVATION)
    r2 = loom.log(speaker="User", content="Mozart symphonies always make me feel relaxed.", kind=RecordKind.OBSERVATION)
    r3 = loom.log(speaker="User", content="Classical piano music is my favorite genre.", kind=RecordKind.OBSERVATION)

    loom.graph_store.link_concept_record(cnode.id, r1.id)
    loom.graph_store.link_concept_record(cnode.id, r2.id)
    loom.graph_store.link_concept_record(cnode.id, r3.id)

    engine = TopicalReflectionEngine(loom)
    ref_record = engine.consolidate_and_ingest_topical_node(
        cnode.id,
        synthesized_text="User is a passionate fan of classical music including Bach and Mozart, preferring orchestral and piano compositions.",
        scope="default",
    )

    assert ref_record is not None
    assert ref_record.kind == RecordKind.FACT
    assert ref_record.evidence_status == EvidenceStatus.SYSTEM_DERIVED
    assert ref_record.metadata.get("origin") == "topical_reflection"
    assert ref_record.metadata.get("y_concept_id") == cnode.id

    # Verify concept record link was created for the new reflection record
    linked_recs = loom.graph_store.get_records_in_concept_vaults_for_term("music")
    assert ref_record.id in linked_recs or ref_record.id in loom.graph_store.connection.execute(
        "SELECT record_id FROM concept_record_links WHERE concept_id = ?", (cnode.id,)
    ).fetchall()[0]["record_id"]


def test_topical_reflection_retrieval_on_inferential_query(tmp_path: Path):
    embedder = DummyEmbeddingProvider()
    loom = ContextLoom(
        database=tmp_path / "test_reflection_retrieval.sqlite",
        embedding_provider=embedder,
        install_default_capabilities=False,
    )

    cnode = loom.graph_store.create_vector_concept_node(
        name="Classical Music Preferences", parent_id="self_root", y_depth=1
    )
    loom.graph_store.link_x_membrane_to_y_concept("vivaldi", cnode.id)

    r1 = loom.log(speaker="User", content="I love listening to Bach concertos.", kind=RecordKind.OBSERVATION)
    r2 = loom.log(speaker="User", content="Mozart symphonies always make me feel relaxed.", kind=RecordKind.OBSERVATION)
    r3 = loom.log(speaker="User", content="Classical piano music is my favorite genre.", kind=RecordKind.OBSERVATION)

    loom.graph_store.link_concept_record(cnode.id, r1.id)
    loom.graph_store.link_concept_record(cnode.id, r2.id)
    loom.graph_store.link_concept_record(cnode.id, r3.id)

    engine = TopicalReflectionEngine(loom)
    engine.run_reflection_pass(
        scope="default",
        min_records=3,
        synthesizer=lambda concept_name, text_list: f"User enjoys classical music like Bach, Mozart, and Vivaldi."
    )

    packet = loom.compiler.compile("Would User enjoy Vivaldi?", scope="default", mode="reply")

    rendered = packet.rendered
    print("\n--- REFLECTION RENDERED PACKET ---\n", rendered)
    print("\n--- MANIFEST ---")
    for m in packet.manifest:
        print(f"ID: {m.candidate_id}, admitted: {m.admitted}, reason: {m.reason}")

    # The synthesized reflection record containing 'Vivaldi' must be in the rendered packet body!
    assert "User enjoys classical music like Bach, Mozart, and Vivaldi." in rendered
    assert any(m.candidate_id.startswith("record:") for m in packet.manifest if m.admitted)
