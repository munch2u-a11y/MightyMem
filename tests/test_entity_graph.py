from __future__ import annotations

from pathlib import Path
import pytest

from context_loom.entity_graph import EntityExtractor, EntityGraphStore, EntityKind, RelationKind
from context_loom.store import LoomStore


def test_entity_extractor_extracts_typed_entities_and_relations() -> None:
    extractor = EntityExtractor()
    text = "Alex's brother Sam drives a vintage 1972 Volvo 1800ES in Seattle."
    entities, relations = extractor.extract(text)

    entity_names = {e.name.casefold() for e in entities}
    assert "alex" in entity_names
    assert "sam" in entity_names
    assert "seattle" in entity_names

    rel_tuples = {(r.source.casefold(), r.relation, r.target.casefold()) for r in relations}
    assert any("alex" in src and "sam" in tgt for src, rel, tgt in rel_tuples)


def test_entity_graph_store_persists_nodes_edges_and_record_links(tmp_path: Path) -> None:
    store = LoomStore(tmp_path / "loom.sqlite")
    graph = EntityGraphStore(store)

    graph.add_entity("Alex", EntityKind.PERSON)
    graph.add_entity("Sam", EntityKind.PERSON)
    graph.add_relation("Alex", RelationKind.FAMILY_OF, "Sam", record_id="rec-1")
    graph.link_record_entity("rec-1", "Alex")
    graph.link_record_entity("rec-1", "Sam")

    neighbors = graph.neighbors("Alex")
    assert any(n["entity"] == "Sam" and n["relation"] == RelationKind.FAMILY_OF for n in neighbors)

    records = graph.records_for_entity("Sam")
    assert "rec-1" in records
    store.close()


def test_vector_concept_node_compaction_and_y_cipher_traversal(tmp_path: Path) -> None:
    from context_loom.retrieval import HashingSemanticEmbedder

    store = LoomStore(tmp_path / "loom.sqlite")
    graph = EntityGraphStore(store)
    embedder = HashingSemanticEmbedder()

    # Add relations
    graph.add_relation("Alex", RelationKind.INTERESTED_IN, "Camping", record_id="rec-camp-1")
    graph.add_relation("Alex", RelationKind.INTERESTED_IN, "Outdoors", record_id="rec-camp-2")

    nodes = graph.compact_concept_nodes(embedding_provider=embedder)
    assert len(nodes) >= 1
    assert nodes[0].id.startswith("concept:")

    results = graph.y_cipher_depth_search("What does Alex like to do?", embedding_provider=embedder)
    assert len(results) >= 1
    assert "record_id" in results[0]
    store.close()


def test_entity_extractor_owns_pattern_and_relations() -> None:
    extractor = EntityExtractor()
    text = "Alex drives a vintage Volvo 1800ES in Seattle. Jon owns a bakery."
    entities, relations = extractor.extract(text)

    entity_names = {e.canonical_name for e in entities}
    assert "alex" in entity_names
    assert "jon" in entity_names

    owns_relations = [r for r in relations if r.relation == RelationKind.OWNS]
    assert len(owns_relations) >= 1
    assert any(r.source.casefold() == "jon" and "bakery" in r.target.casefold() for r in owns_relations) or \
           any(r.source.casefold() == "alex" and "volvo" in r.target.casefold() for r in owns_relations)


def test_entity_extractor_multiword_and_salient_concepts() -> None:
    extractor = EntityExtractor()
    text = "Gina is opening a dance studio. Jon started a clothing store after working as a banker."
    entities, relations = extractor.extract(text)

    entity_names = {e.canonical_name for e in entities}
    assert "dance studio" in entity_names
    assert "clothing store" in entity_names
    assert "banker" in entity_names

    # Stopwords and verbs should not be entities
    assert "is" not in entity_names
    assert "opening" not in entity_names
    assert "after" not in entity_names
    assert "working" not in entity_names


def test_entity_extractor_temporal_entities() -> None:
    extractor = EntityExtractor()
    text = "On 20 January, 2023 at 4:04 pm, Jon met Gina. Yesterday they discussed the lease."
    entities, relations = extractor.extract(text)

    temporal_entities = [e for e in entities if e.kind == EntityKind.TIME_INTERVAL]
    temporal_names = {e.canonical_name for e in temporal_entities}
    assert any("january" in name and "2023" in name for name in temporal_names)
    assert "yesterday" in temporal_names


def test_entity_extractor_question_stopword_protection() -> None:
    extractor = EntityExtractor()
    query = "When did Jon lose his job as a banker, and where does Gina teach?"
    entities, _ = extractor.extract(query)

    canonical_names = {e.canonical_name for e in entities}
    assert "when" not in canonical_names
    assert "where" not in canonical_names
    assert "did" not in canonical_names
    assert "and" not in canonical_names
    assert "jon" in canonical_names
    assert "gina" in canonical_names


def test_entity_graph_store_kind_preservation_and_upgrade(tmp_path: Path) -> None:
    store = LoomStore(tmp_path / "loom.sqlite")
    graph = EntityGraphStore(store)

    # 1. Add as PERSON initially (or generic)
    graph.add_entity("EcoDesign", EntityKind.PERSON)
    # 2. Add relation with specific kinds - should upgrade kind to ORGANIZATION, not pollute with PERSON
    graph.add_relation(
        "Sam",
        RelationKind.WORKS_AT,
        "EcoDesign",
        source_kind=EntityKind.PERSON,
        target_kind=EntityKind.ORGANIZATION,
    )

    rows = graph.connection.execute(
        "SELECT kind FROM entity_nodes WHERE canonical_name = 'ecodesign'"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["kind"] == EntityKind.ORGANIZATION.value

    # 3. link_record_entity should preserve existing specific kind
    graph.link_record_entity("rec-1", "EcoDesign")
    rows = graph.connection.execute(
        "SELECT kind FROM entity_nodes WHERE canonical_name = 'ecodesign'"
    ).fetchall()
    assert rows[0]["kind"] == EntityKind.ORGANIZATION.value
    store.close()


def test_entity_graph_store_cooccurrence_and_indexes(tmp_path: Path) -> None:
    store = LoomStore(tmp_path / "loom.sqlite")
    graph = EntityGraphStore(store)

    graph.add_cooccurrence(["Jon", "banker", "clothing store"], record_id="rec-1", temporal_weight=1.0)

    # Reciprocal CO_OCCURRED_WITH edges
    neighbors_jon = graph.neighbors("Jon")
    co_occurred = [n for n in neighbors_jon if n["relation"] == RelationKind.CO_OCCURRED_WITH]
    co_entities = {n["canonical_name"] for n in co_occurred}
    assert "banker" in co_entities
    assert "clothing store" in co_entities

    neighbors_banker = graph.neighbors("banker")
    banker_co = {n["canonical_name"] for n in neighbors_banker if n["relation"] == RelationKind.CO_OCCURRED_WITH}
    assert "jon" in banker_co
    assert "clothing store" in banker_co

    # Verify SQLite indexes exist
    indexes = [
        row["name"]
        for row in graph.connection.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()
    ]
    assert "idx_record_entities_record_id" in indexes
    assert "idx_entity_edges_relation" in indexes

    store.close()


def test_index_record_links_speaker_to_first_person_concepts(tmp_path: Path) -> None:
    store = LoomStore(tmp_path / "loom.sqlite")
    graph = EntityGraphStore(store)

    graph.index_record(
        "rec-1",
        "I opened a dance studio after leaving my banking job.",
        speaker="Jon",
    )

    assert set(graph.records_for_entity("Jon")) == {"rec-1"}
    neighbors = graph.neighbors("Jon")
    assert any(
        row["canonical_name"] == "dance studio"
        and row["relation"] == RelationKind.CO_OCCURRED_WITH
        and row["record_id"] == "rec-1"
        for row in neighbors
    )
    kind = graph.connection.execute(
        "SELECT kind FROM entity_nodes WHERE canonical_name = 'dance studio'"
    ).fetchone()["kind"]
    assert kind == EntityKind.TOPIC.value
    store.close()


def test_concept_compaction_does_not_conflate_cooccurring_predicates(tmp_path: Path) -> None:
    from context_loom.retrieval import HashingSemanticEmbedder

    store = LoomStore(tmp_path / "loom_dynamic.sqlite")
    graph = EntityGraphStore(store)
    embedder = HashingSemanticEmbedder()

    graph.add_relation("Alice", "works_with", "Acme", record_id="rec-work")
    graph.add_relation("Alice", "owns", "bicycle", record_id="rec-own")
    graph.record_relation_cooccurrence("works_with", "owns", shared_target="Acme")
    graph.record_relation_cooccurrence("works_with", "owns", shared_target="Acme")

    concept_nodes = graph.compact_concept_nodes(embedding_provider=embedder)
    names = {node.name for node in concept_nodes}
    assert "Alice works with: acme" in names
    assert "Alice owns: bicycle" in names
    assert all("works with: acme, bicycle" not in name for name in names)

    work_links = graph.connection.execute(
        "SELECT record_id FROM concept_record_links WHERE concept_id = 'concept:alice:works_with'"
    ).fetchall()
    assert {row["record_id"] for row in work_links} == {"rec-work"}

    store.close()


def test_graph_ingestion_is_idempotent_per_source_record(tmp_path: Path) -> None:
    store = LoomStore(tmp_path / "loom_idempotent.sqlite")
    graph = EntityGraphStore(store)

    for _ in range(2):
        graph.add_relation("Alex", "owns", "bicycle", record_id="rec-1")
        graph.add_cooccurrence(["Alex", "bicycle"], record_id="rec-1")

    rows = graph.connection.execute(
        "SELECT source, relation, target, record_id FROM entity_edges WHERE record_id = 'rec-1'"
    ).fetchall()
    assert len(rows) == 3

    store.close()


def test_failed_recompaction_preserves_existing_concept_vector(tmp_path: Path) -> None:
    from context_loom.retrieval import HashingSemanticEmbedder

    store = LoomStore(tmp_path / "loom_vector.sqlite")
    graph = EntityGraphStore(store)
    graph.add_relation("Alex", "owns", "bicycle", record_id="rec-1")
    graph.compact_concept_nodes(HashingSemanticEmbedder())
    before = graph.connection.execute(
        "SELECT dense_vector FROM concept_nodes WHERE id = 'concept:alex:owns'"
    ).fetchone()["dense_vector"]
    graph.add_relation("Alex", "owns", "car", record_id="rec-2")

    class BrokenEmbedder:
        def embed(self, text):
            raise RuntimeError("embedding unavailable")

    with pytest.raises(RuntimeError, match="embedding unavailable"):
        graph.compact_concept_nodes(BrokenEmbedder())

    after = graph.connection.execute(
        "SELECT dense_vector FROM concept_nodes WHERE id = 'concept:alex:owns'"
    ).fetchone()["dense_vector"]
    assert after == before
    store.close()
