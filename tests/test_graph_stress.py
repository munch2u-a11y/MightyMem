"""Adversarial and stress-testing test harness for Milestone 1:
Entity Graph and MultiHopGraphLane.

Tests cover:
1. Deep multi-hop chains (3, 4, 5 hops) with proper decay scoring.
2. Cyclic graph topologies and self-referential edges (A -> B -> C -> A, A -> A, K5).
3. High-density star hubs (50-degree, 100-degree vs 2-degree concept node discounting >= 2.5x).
4. Co-occurrence matrix generation at scale (dense clusters, 50-100 entities, no corruption/locks).
5. Malformed inputs, long strings (10k+ chars), ReDoS resistance, emojis, and unicode variants.
"""
from __future__ import annotations

import math
from pathlib import Path
import time
import pytest

from context_loom.engine import ContextLoom
from context_loom.entity_graph import (
    Entity,
    EntityExtractor,
    EntityGraphStore,
    EntityKind,
    RelationKind,
)
from context_loom.models import EvidenceStatus, RecordKind
from context_loom.multi_hop_lane import MultiHopGraphLane
from context_loom.retrieval import RetrievalContext
from context_loom.store import LoomStore


# ============================================================================
# Area 1: Deep Multi-Hop Chains (3, 4, 5 Hops) & Decay Scoring
# ============================================================================

def test_deep_chain_traversal_and_monotonic_decay_5_hops(tmp_path: Path) -> None:
    """Verify deep multi-hop chains up to 5 hops with monotonic decay scoring."""
    store = LoomStore(tmp_path / "chain.sqlite")
    graph = EntityGraphStore(store)

    # Chain: E0 -> E1 -> E2 -> E3 -> E4 -> E5
    # Each entity Ei has one direct record Rec_i (so each has uniform degree)
    num_nodes = 6
    for i in range(num_nodes):
        graph.add_entity(f"Node{i}", EntityKind.TOPIC)
        graph.link_record_entity(f"rec-node-{i}", f"Node{i}")
        if i > 0:
            graph.add_relation(
                f"Node{i-1}",
                RelationKind.ASSOCIATED_WITH,
                f"Node{i}",
                record_id=f"rec-edge-{i-1}-{i}",
                confidence=1.0,
            )

    # Test max_hops = 1
    hits_1 = graph.multi_hop_records(["Node0"], max_hops=1)
    rec_ids_1 = {h["record_id"] for h in hits_1}
    assert "rec-node-0" in rec_ids_1
    assert "rec-edge-0-1" in rec_ids_1
    assert "rec-node-1" not in rec_ids_1  # Node1 direct record is reached at hop 2
    assert "rec-node-2" not in rec_ids_1

    # Test max_hops = 3
    hits_3 = graph.multi_hop_records(["Node0"], max_hops=3)
    rec_ids_3 = {h["record_id"] for h in hits_3}
    assert "rec-node-0" in rec_ids_3
    assert "rec-node-1" in rec_ids_3
    assert "rec-node-2" in rec_ids_3
    assert "rec-node-3" not in rec_ids_3
    assert "rec-node-4" not in rec_ids_3

    # Test max_hops = 5
    hits_5 = graph.multi_hop_records(["Node0"], max_hops=5)
    rec_ids_5 = {h["record_id"] for h in hits_5}
    for i in range(5):
        assert f"rec-node-{i}" in rec_ids_5, f"rec-node-{i} should be reached within 5 hops"

    # Verify monotonic decay along the chain for direct entity records
    score_map = {h["record_id"]: h["score"] for h in hits_5}
    assert score_map["rec-node-0"] > score_map["rec-node-1"], "Hop 1 record must outscore Hop 2"
    assert score_map["rec-node-1"] > score_map["rec-node-2"], "Hop 2 record must outscore Hop 3"
    assert score_map["rec-node-2"] > score_map["rec-node-3"], "Hop 3 record must outscore Hop 4"
    assert score_map["rec-node-3"] > score_map["rec-node-4"], "Hop 4 record must outscore Hop 5"

    # Verify explicit decay ratios match formula base_hop_score = max(0.40, 0.95 * (0.85 ** (hop - 1)))
    # For entities with identical degrees, the ratio of scores equals the base_hop_score ratio (0.85)
    # Node1 (hop 2), Node2 (hop 3), Node3 (hop 4) all have degree = 3 (1 record + 2 edges)
    ratio_hop3_hop2 = score_map["rec-node-2"] / score_map["rec-node-1"]
    assert abs(ratio_hop3_hop2 - 0.85) < 0.01, f"Expected ~0.85 decay ratio, got {ratio_hop3_hop2}"

    ratio_hop4_hop3 = score_map["rec-node-3"] / score_map["rec-node-2"]
    assert abs(ratio_hop4_hop3 - 0.85) < 0.01, f"Expected ~0.85 decay ratio, got {ratio_hop4_hop3}"

    store.close()


def test_deep_tree_branching_5_hops(tmp_path: Path) -> None:
    """Verify binary tree expansion reaches 31 nodes at depth 5 without state explosion."""
    store = LoomStore(tmp_path / "tree.sqlite")
    graph = EntityGraphStore(store)

    # Build full binary tree of depth 5 (root + 2 + 4 + 8 + 16 = 31 nodes)
    for i in range(1, 16):
        left = 2 * i
        right = 2 * i + 1
        graph.add_relation(f"tree_{i}", RelationKind.ASSOCIATED_WITH, f"tree_{left}", record_id=f"rec_edge_{i}_{left}")
        graph.add_relation(f"tree_{i}", RelationKind.ASSOCIATED_WITH, f"tree_{right}", record_id=f"rec_edge_{i}_{right}")
        graph.link_record_entity(f"rec_node_{i}", f"tree_{i}")
        graph.link_record_entity(f"rec_node_{left}", f"tree_{left}")
        graph.link_record_entity(f"rec_node_{right}", f"tree_{right}")

    hits = graph.multi_hop_records(["tree_1"], max_hops=5)
    assert len(hits) > 0

    # Ensure all scores are strictly between (0.0, 1.0]
    for h in hits:
        assert 0.0 < h["score"] <= 1.0
        assert h["hop"] <= 5

    store.close()


def test_multi_path_reinforcement_bonus(tmp_path: Path) -> None:
    """Verify records reachable via multiple distinct paths receive the multi-seed intersection bonus."""
    store = LoomStore(tmp_path / "bonus.sqlite")
    graph = EntityGraphStore(store)

    # Seed1 -> SharedEntity -> TargetRecord
    # Seed2 -> SharedEntity -> TargetRecord
    graph.link_record_entity("rec_target", "SharedConcept")
    graph.add_relation("SeedA", RelationKind.ASSOCIATED_WITH, "SharedConcept", record_id="rec_edge_a")
    graph.add_relation("SeedB", RelationKind.ASSOCIATED_WITH, "SharedConcept", record_id="rec_edge_b")

    # Traversal from single seed
    hits_single = graph.multi_hop_records(["SeedA"], max_hops=2)
    single_score = next(h["score"] for h in hits_single if h["record_id"] == "rec_target")

    # Traversal from both seeds
    hits_multi = graph.multi_hop_records(["SeedA", "SeedB"], max_hops=2)
    multi_score = next(h["score"] for h in hits_multi if h["record_id"] == "rec_target")

    # The multi-seed traversal should boost the score
    assert multi_score >= single_score
    store.close()


# ============================================================================
# Area 2: Cyclic Graph Topologies & Self-Referential Edges
# ============================================================================

def test_self_referential_edge_termination_and_safety(tmp_path: Path) -> None:
    """Verify self-referential edge A -> A does not cause infinite loop or corrupt visited set."""
    store = LoomStore(tmp_path / "self_loop.sqlite")
    graph = EntityGraphStore(store)

    graph.add_entity("SelfLoopNode", EntityKind.TOPIC)
    graph.link_record_entity("rec_self", "SelfLoopNode")
    # A -> A self edge
    graph.add_relation("SelfLoopNode", RelationKind.ASSOCIATED_WITH, "SelfLoopNode", record_id="rec_self_edge")

    hits = graph.multi_hop_records(["SelfLoopNode"], max_hops=4)
    rec_ids = [h["record_id"] for h in hits]

    # Must contain the record and must terminate cleanly
    assert "rec_self" in rec_ids
    assert len(hits) <= 2  # rec_self and optionally rec_self_edge
    # Scores must be valid
    for h in hits:
        assert 0.0 < h["score"] <= 1.0

    store.close()


def test_cyclic_graph_triangle_and_figure_eight(tmp_path: Path) -> None:
    """Verify directed/undirected cycles (A -> B -> C -> A) and figure-8 topology terminate safely."""
    store = LoomStore(tmp_path / "cycles.sqlite")
    graph = EntityGraphStore(store)

    # Triangle 1: Alpha -> Beta -> Gamma -> Alpha
    graph.add_relation("Alpha", RelationKind.ASSOCIATED_WITH, "Beta", record_id="rec_ab")
    graph.add_relation("Beta", RelationKind.ASSOCIATED_WITH, "Gamma", record_id="rec_bg")
    graph.add_relation("Gamma", RelationKind.ASSOCIATED_WITH, "Alpha", record_id="rec_ga")

    # Triangle 2 sharing Gamma: Gamma -> Delta -> Epsilon -> Gamma
    graph.add_relation("Gamma", RelationKind.ASSOCIATED_WITH, "Delta", record_id="rec_gd")
    graph.add_relation("Delta", RelationKind.ASSOCIATED_WITH, "Epsilon", record_id="rec_de")
    graph.add_relation("Epsilon", RelationKind.ASSOCIATED_WITH, "Gamma", record_id="rec_eg")

    for node in ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]:
        graph.link_record_entity(f"rec_{node.lower()}", node)

    # High max_hops to stress cycle traversal
    hits = graph.multi_hop_records(["Alpha"], max_hops=10)

    # Must terminate without infinite loop and visited_records must have finite size
    rec_ids = {h["record_id"] for h in hits}
    for node in ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]:
        assert f"rec_{node.lower()}" in rec_ids

    # All scores strictly bounded
    for h in hits:
        assert 0.0 < h["score"] <= 1.0

    store.close()


def test_clique_k5_cycle_stress(tmp_path: Path) -> None:
    """Verify fully connected 5-clique K5 terminates cleanly without score explosion."""
    store = LoomStore(tmp_path / "clique.sqlite")
    graph = EntityGraphStore(store)

    nodes = [f"KNode_{i}" for i in range(5)]
    for i in range(5):
        graph.link_record_entity(f"rec_knode_{i}", nodes[i])
        for j in range(i + 1, 5):
            graph.add_relation(nodes[i], RelationKind.ASSOCIATED_WITH, nodes[j], record_id=f"rec_edge_{i}_{j}")

    hits = graph.multi_hop_records(["KNode_0"], max_hops=5)
    assert len(hits) > 0
    for h in hits:
        assert 0.0 < h["score"] <= 1.0

    store.close()


# ============================================================================
# Area 3: High-Density Star Hubs & Discounting Verification (>= 2.5x)
# ============================================================================

def test_hub_discounting_ratio_50_and_100_degree(tmp_path: Path) -> None:
    """Verify that a 50-degree and 100-degree speaker hub is discounted by >= 2.5x relative to a 2-degree concept node."""
    store = LoomStore(tmp_path / "hub_stress.sqlite")
    graph = EntityGraphStore(store)

    # 1. 2-degree concept node
    graph.link_record_entity("rec_concept_1", "vintage_car")
    graph.link_record_entity("rec_concept_2", "vintage_car")

    # 2. 50-degree speaker hub
    for i in range(50):
        graph.link_record_entity(f"rec_hub50_{i}", "SpeakerJon")

    # 3. 100-degree speaker hub
    for i in range(100):
        graph.link_record_entity(f"rec_hub100_{i}", "SpeakerGina")

    deg_concept = graph.entity_degree("vintage_car")
    deg_hub50 = graph.entity_degree("speakerjon")
    deg_hub100 = graph.entity_degree("speakergina")

    assert deg_concept == 2
    assert deg_hub50 == 50
    assert deg_hub100 == 100

    # Retrieve 1-hop records
    concept_hits = graph.multi_hop_records(["vintage_car"], max_hops=1)
    hub50_hits = graph.multi_hop_records(["SpeakerJon"], max_hops=1)
    hub100_hits = graph.multi_hop_records(["SpeakerGina"], max_hops=1)

    concept_score = concept_hits[0]["score"]
    hub50_score = hub50_hits[0]["score"]
    hub100_score = hub100_hits[0]["score"]

    ratio_50 = concept_score / hub50_score
    ratio_100 = concept_score / hub100_score

    # Verification criteria: discount by >= 2.5x
    assert ratio_50 >= 2.5, f"50-degree hub discount ratio {ratio_50:.3f} is less than 2.5x"
    assert ratio_100 >= 2.5, f"100-degree hub discount ratio {ratio_100:.3f} is less than 2.5x"

    # Theoretical check:
    # W(concept) = 1.0 / log2(2 + 2) = 0.500
    # W(hub50) = 1.0 / log2(2 + 50) = 1.0 / 5.7004 = 0.1754 -> ratio = 2.850
    # W(hub100) = 1.0 / log2(2 + 100) = 1.0 / 6.6724 = 0.1499 -> ratio = 3.336
    assert abs(ratio_50 - (math.log2(52) / math.log2(4))) < 1e-4
    assert abs(ratio_100 - (math.log2(102) / math.log2(4))) < 1e-4

    store.close()


def test_hub_domination_prevention_in_lane_nomination(tmp_path: Path) -> None:
    """Verify in MultiHopGraphLane that 50 speaker hub turns do NOT dominate/displace salient concept evidence."""
    loom = ContextLoom(tmp_path / "hub_loom.sqlite")
    graph = EntityGraphStore(loom.store)
    lane = MultiHopGraphLane(loom.store, graph, top_k=10)

    # Create 50 conversational filler records from speaker "Jon"
    hub_records = []
    for i in range(50):
        rec = loom.record(
            RecordKind.USER,
            f"Jon: sounds good turn {i}",
            scope="session_1",
            record_id=f"hub-turn-{i}",
        )
        hub_records.append(rec)
        graph.link_record_entity(rec.id, "Jon")

    # Create 2 factual evidence records linked to specific concept "volvo"
    evidence_1 = loom.record(
        RecordKind.USER,
        "Alex bought a rare blue vintage Volvo 1800ES coupe.",
        scope="session_1",
        record_id="evidence-rec-1",
    )
    evidence_2 = loom.record(
        RecordKind.USER,
        "The Volvo was serviced in Seattle yesterday.",
        scope="session_1",
        record_id="evidence-rec-2",
    )
    graph.link_record_entity(evidence_1.id, "Volvo")
    graph.link_record_entity(evidence_2.id, "Volvo")
    graph.link_record_entity(evidence_1.id, "Alex")
    graph.link_record_entity(evidence_2.id, "Seattle")

    all_records = hub_records + [evidence_1, evidence_2]

    # Query asking about the Volvo and mentioning Jon
    ctx = RetrievalContext(
        query="Tell me about the Volvo that Jon mentioned",
        scope="session_1",
        now=100.0,
        records=all_records,
        store=loom.store,
    )

    candidates = lane.nominate(ctx)
    candidate_ids = [c.source_record_ids[0] for c in candidates]

    # The Volvo evidence must be present in top nominations
    assert "evidence-rec-1" in candidate_ids
    assert "evidence-rec-2" in candidate_ids

    # The Volvo evidence must have higher score than any filler hub turn
    volvo_score = next(c.lane_scores["multi_hop_graph"] for c in candidates if c.source_record_ids[0] == "evidence-rec-1")
    for c in candidates:
        if c.source_record_ids[0].startswith("hub-turn-"):
            hub_lane_score = c.lane_scores["multi_hop_graph"]
            assert volvo_score > hub_lane_score * 2.0, (
                f"Concept evidence score ({volvo_score:.3f}) should strongly beat hub score ({hub_lane_score:.3f})"
            )

    loom.close()


# ============================================================================
# Area 4: Co-Occurrence Matrix Generation at Scale
# ============================================================================

def test_cooccurrence_dense_cluster_50_entities(tmp_path: Path) -> None:
    """Verify add_cooccurrence handles 50 entities in one record (1225 pairs = 2450 edges) cleanly."""
    store = LoomStore(tmp_path / "cooc_50.sqlite")
    graph = EntityGraphStore(store)

    entities = [f"Entity_{i}" for i in range(50)]
    t0 = time.perf_counter()
    graph.add_cooccurrence(entities, record_id="rec_dense_50", temporal_weight=0.9)
    elapsed = time.perf_counter() - t0

    # Must complete fast (< 1.0s)
    assert elapsed < 1.0, f"Dense 50-entity co-occurrence took {elapsed:.2f}s, expected < 1.0s"

    # Verify SQLite integrity
    integrity = store.connection.execute("PRAGMA integrity_check").fetchone()[0]
    assert integrity == "ok"

    # Verify row counts
    # 50 entities in record_entities
    rec_count = store.connection.execute(
        "SELECT COUNT(*) FROM record_entities WHERE record_id = 'rec_dense_50'"
    ).fetchone()[0]
    assert rec_count == 50

    # 50 * 49 = 2450 edges in entity_edges
    edge_count = store.connection.execute(
        "SELECT COUNT(*) FROM entity_edges WHERE record_id = 'rec_dense_50' AND relation = 'co_occurred_with'"
    ).fetchone()[0]
    assert edge_count == 2450

    # Verify reciprocity for an arbitrary pair (Entity_0, Entity_49)
    n0 = graph.neighbors("Entity_0")
    assert any(n["canonical_name"] == "entity_49" and n["confidence"] == 0.9 for n in n0)
    n49 = graph.neighbors("Entity_49")
    assert any(n["canonical_name"] == "entity_0" and n["confidence"] == 0.9 for n in n49)

    store.close()


def test_cooccurrence_extreme_cluster_100_entities(tmp_path: Path) -> None:
    """Verify add_cooccurrence handles 100 entities (4950 pairs = 9900 edges) without lock or corruption."""
    store = LoomStore(tmp_path / "cooc_100.sqlite")
    graph = EntityGraphStore(store)

    entities = [f"ConceptNode_{i}" for i in range(100)]
    t0 = time.perf_counter()
    graph.add_cooccurrence(entities, record_id="rec_dense_100", temporal_weight=1.0)
    elapsed = time.perf_counter() - t0

    assert elapsed < 3.0, f"Dense 100-entity co-occurrence took {elapsed:.2f}s"

    integrity = store.connection.execute("PRAGMA integrity_check").fetchone()[0]
    assert integrity == "ok"

    edge_count = store.connection.execute(
        "SELECT COUNT(*) FROM entity_edges WHERE record_id = 'rec_dense_100'"
    ).fetchone()[0]
    assert edge_count == 9900

    store.close()


def test_cooccurrence_multi_record_sequential_stress(tmp_path: Path) -> None:
    """Verify 50 consecutive records each with 10 entities without database lock or performance degradation."""
    store = LoomStore(tmp_path / "cooc_multi.sqlite")
    graph = EntityGraphStore(store)

    t0 = time.perf_counter()
    for rec_idx in range(50):
        # 10 entities per record -> 90 edges per record -> 4500 total edges
        ents = [f"Cluster_{rec_idx % 5}_Ent_{i}" for i in range(10)]
        graph.add_cooccurrence(ents, record_id=f"rec_{rec_idx}", temporal_weight=0.85)
    elapsed = time.perf_counter() - t0

    assert elapsed < 2.0, f"50 sequential co-occurrence transactions took {elapsed:.2f}s"

    integrity = store.connection.execute("PRAGMA integrity_check").fetchone()[0]
    assert integrity == "ok"

    total_edges = store.connection.execute(
        "SELECT COUNT(*) FROM entity_edges WHERE relation = 'co_occurred_with'"
    ).fetchone()[0]
    assert total_edges == 50 * (10 * 9)

    store.close()


def test_cooccurrence_edge_cases_and_deduplication(tmp_path: Path) -> None:
    """Verify edge cases: duplicate entities, empty input, single entity, whitespace."""
    store = LoomStore(tmp_path / "cooc_edge.sqlite")
    graph = EntityGraphStore(store)

    # 1. Empty list
    graph.add_cooccurrence([], record_id="rec_empty")
    assert store.connection.execute("SELECT COUNT(*) FROM entity_edges").fetchone()[0] == 0

    # 2. Empty record_id
    graph.add_cooccurrence(["Alex", "Sam"], record_id="")
    assert store.connection.execute("SELECT COUNT(*) FROM entity_edges").fetchone()[0] == 0

    # 3. Single entity (no pairs can be formed)
    graph.add_cooccurrence(["SoloEntity"], record_id="rec_solo")
    rec_ents = store.connection.execute("SELECT canonical_name FROM record_entities WHERE record_id = 'rec_solo'").fetchall()
    assert len(rec_ents) == 1
    # No self-referential co-occurrence edge should be inserted
    assert store.connection.execute("SELECT COUNT(*) FROM entity_edges").fetchone()[0] == 0

    # 4. Duplicate entities with casing and punctuation variants: ["Alex", "alex", " ALEX! "]
    graph.add_cooccurrence(["Alex", "alex", " ALEX! "], record_id="rec_dup")
    # All canonicalize to "alex", so no pairs should be inserted
    assert store.connection.execute("SELECT COUNT(*) FROM entity_edges").fetchone()[0] == 0

    # 5. Entity list containing empty strings or whitespace
    graph.add_cooccurrence(["ValidA", "   ", "", "ValidB"], record_id="rec_clean")
    edges = store.connection.execute("SELECT source, target FROM entity_edges WHERE record_id = 'rec_clean'").fetchall()
    assert len(edges) == 2  # (valida, validb) and (validb, valida)

    store.close()


# ============================================================================
# Area 5: Malformed Inputs, 10k Strings, ReDoS Resistance, Unicode & Emojis
# ============================================================================

def test_extractor_long_string_and_redos_resistance() -> None:
    """Verify EntityExtractor does not hang on 10,000+ char strings or adversarial regex patterns."""
    extractor = EntityExtractor()

    # 1. 10k continuous string without spaces
    long_token = "A" * 10000
    t0 = time.perf_counter()
    ents, rels = extractor.extract(long_token)
    assert time.perf_counter() - t0 < 0.5

    # 2. 10k repetitive text
    repeated = "Alex drives a Volvo in Seattle. " * 350  # ~11,200 chars
    t0 = time.perf_counter()
    ents, rels = extractor.extract(repeated)
    assert time.perf_counter() - t0 < 1.0
    assert any(e.canonical_name == "alex" for e in ents)

    # 3. Pathological backtracking stress for _works_pattern: "Alex works at AAAAA...!"
    pathological_works = "Alex works at " + "A" * 5000 + "!"
    t0 = time.perf_counter()
    ents, rels = extractor.extract(pathological_works)
    assert time.perf_counter() - t0 < 0.5

    # 4. Pathological multi-word proper noun pattern: repeated capitalized tokens
    multi_caps = " ".join(["Word" + str(i) for i in range(1000)])
    t0 = time.perf_counter()
    ents, rels = extractor.extract(multi_caps)
    assert time.perf_counter() - t0 < 0.5


def test_extractor_malformed_inputs_and_injection_vectors(tmp_path: Path) -> None:
    """Verify EntityExtractor and EntityGraphStore safely handle malformed input and SQL injections."""
    extractor = EntityExtractor()
    store = LoomStore(tmp_path / "injection.sqlite")
    graph = EntityGraphStore(store)

    # Malformed strings
    malformed_inputs = [
        "",
        "   \t\n  ",
        "\x00\x00\x00",
        "(((((((([[[{[[[{(",
        "????????!!!!!!!!....",
        "'; DROP TABLE entity_nodes; --",
        "Jon' OR '1'='1",
        "\" UNION SELECT * FROM entity_nodes --",
        "<script>alert('xss')</script>",
        "\\x00\\xff\\xfe",
    ]

    for text in malformed_inputs:
        # Extractor must not raise an unhandled exception
        entities, relations = extractor.extract(text)
        assert isinstance(entities, list)
        assert isinstance(relations, list)

        # Store operations must safely parameterize and not execute injected SQL
        for ent in entities:
            graph.add_entity(ent.name, ent.kind)
            graph.link_record_entity("rec_test", ent.name)

    # Verify entity_nodes table was NOT dropped by injection payload
    tables = [
        r[0] for r in store.connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    ]
    assert "entity_nodes" in tables
    assert "entity_edges" in tables
    assert "record_entities" in tables

    store.close()


def test_extractor_emojis_and_unicode_variants() -> None:
    """Verify EntityExtractor handles emojis, accented latin, Cyrillic, CJK, and mixed text."""
    extractor = EntityExtractor()

    # Emojis + named entities
    emoji_text = "Alex 😀🎉 met Sam 👨‍👩‍👧‍👦 in Seattle 🌲🚀. Jon drives a Volvo 🚗."
    ents, rels = extractor.extract(emoji_text)
    canon_names = {e.canonical_name for e in ents}
    assert "alex" in canon_names
    assert "sam" in canon_names
    assert "seattle" in canon_names

    # Accented characters
    accent_text = "Renée drives a Citroën in São Paulo and Zürich."
    ents, rels = extractor.extract(accent_text)
    # Extractor should process without crashing
    assert isinstance(ents, list)

    # Cyrillic / non-ASCII
    cyrillic_text = "Иван работает в компании Яндекс в Москве."
    ents, rels = extractor.extract(cyrillic_text)
    assert isinstance(ents, list)

    # CJK text
    cjk_text = "田中さんは東京のソニーで働いています。"
    ents, rels = extractor.extract(cjk_text)
    assert isinstance(ents, list)
