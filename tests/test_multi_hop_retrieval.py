from __future__ import annotations

from pathlib import Path
import pytest

from context_loom.engine import ContextLoom
from context_loom.engine import ContextLoom
from context_loom.entity_graph import EntityGraphStore, EntityKind, RelationKind
from context_loom.models import EvidenceStatus, RecordKind
from context_loom.multi_hop_lane import MultiHopGraphLane, YCipherDepthSearchLane
from context_loom.store import LoomStore


def test_multi_hop_lane_expands_2hop_and_3hop_evidence(tmp_path: Path) -> None:
    loom = ContextLoom(tmp_path / "loom.sqlite")
    graph = EntityGraphStore(loom.store)

    # 1. Rec 1: Alex says "My brother Sam drives a Volvo"
    rec1 = loom.record(
        RecordKind.USER,
        "Alex: My brother Sam drives a vintage 1972 Volvo 1800ES.",
        scope="test",
        record_id="rec-1",
        created_at=10,
        evidence_status=EvidenceStatus.USER_REPORTED,
    )
    # 2. Rec 2: Sam works at a Seattle green building firm
    rec2 = loom.record(
        RecordKind.FACT,
        "Sam works as lead architect at EcoDesign in Seattle.",
        scope="test",
        record_id="rec-2",
        created_at=20,
        evidence_status=EvidenceStatus.TOOL_VERIFIED,
    )

    graph.add_relation("Alex", RelationKind.FAMILY_OF, "Sam", record_id=rec1.id)
    graph.link_record_entity(rec1.id, "Alex")
    graph.link_record_entity(rec1.id, "Sam")

    graph.add_relation("Sam", RelationKind.WORKS_AT, "EcoDesign", record_id=rec2.id)
    graph.link_record_entity(rec2.id, "Sam")
    graph.link_record_entity(rec2.id, "EcoDesign")

    lane = MultiHopGraphLane(loom.store, graph)
    loom.compiler.lanes.append(lane)

    packet = loom.compiler.compile("Where does the brother of Alex work?", scope="test", now=30)
    admitted = [rec_id for entry in packet.manifest if entry.admitted for rec_id in entry.source_record_ids]

    assert rec1.id in admitted
    assert rec2.id in admitted
    loom.close()


def test_multi_hop_seed_filtering(tmp_path: Path) -> None:
    loom = ContextLoom(tmp_path / "loom.sqlite")
    graph = EntityGraphStore(loom.store)
    lane = MultiHopGraphLane(loom.store, graph)

    rec = loom.record(
        RecordKind.USER,
        "Jon was hired as a banker.",
        scope="test",
        record_id="rec-1",
    )
    graph.link_record_entity(rec.id, "Jon")

    from context_loom.retrieval import RetrievalContext
    ctx = RetrievalContext(
        query="When did Jon lose his job as a banker and what did he do?",
        scope="test",
        now=100.0,
        records=[rec],
        store=loom.store,
    )

    # Calling nominate should NOT query seeds like "when", "what", "did", "his"
    # To verify, spy on graph_store.multi_hop_records
    called_seeds = []
    orig_multi_hop = graph.multi_hop_records
    def spy_multi_hop(seed_entity_names, max_hops=2, **kwargs):
        called_seeds.extend(seed_entity_names)
        return orig_multi_hop(seed_entity_names, max_hops=max_hops, **kwargs)
    graph.multi_hop_records = spy_multi_hop

    lane.nominate(ctx)
    called_lower = {s.casefold() for s in called_seeds}
    assert "when" not in called_lower
    assert "what" not in called_lower
    assert "did" not in called_lower
    assert "jon" in called_lower
    loom.close()


def test_multi_hop_inverse_degree_discounting(tmp_path: Path) -> None:
    store = LoomStore(tmp_path / "loom.sqlite")
    graph = EntityGraphStore(store)

    # Create a hub entity "Jon" with 50 record links (simulating speaker hub in 100-turn chat)
    for i in range(50):
        rec_id = f"hub-rec-{i}"
        graph.link_record_entity(rec_id, "Jon")

    # Create a specific entity "banker" with only 2 record links
    graph.link_record_entity("spec-rec-1", "banker")
    graph.link_record_entity("spec-rec-2", "banker")

    # Traversal from specific seed vs hub seed
    hub_records = graph.multi_hop_records(["Jon"], max_hops=1)
    spec_records = graph.multi_hop_records(["banker"], max_hops=1)

    hub_score = hub_records[0]["score"]
    spec_score = spec_records[0]["score"]

    # Degree discounting should make spec_score strictly greater than hub_score
    # W(hub=50) ~ 1/log2(52) = 0.175, W(spec=2) ~ 1/log2(4) = 0.5
    assert spec_score >= hub_score * 2.5
    store.close()


def test_multi_hop_lane_top_k_capping(tmp_path: Path) -> None:
    loom = ContextLoom(tmp_path / "loom.sqlite")
    graph = EntityGraphStore(loom.store)
    lane = MultiHopGraphLane(loom.store, graph, top_k=15)

    records = []
    # Link 40 records to "Jon"
    for i in range(40):
        rec = loom.record(
            RecordKind.USER,
            f"Turn {i} mentioning Jon and projects.",
            scope="test",
            record_id=f"rec-{i}",
        )
        records.append(rec)
        graph.link_record_entity(rec.id, "Jon")

    from context_loom.retrieval import RetrievalContext
    ctx = RetrievalContext(
        query="Tell me about Jon",
        scope="test",
        now=100.0,
        records=records,
        store=loom.store,
    )
    candidates = lane.nominate(ctx)
    assert len(candidates) <= 15
    loom.close()


def test_graph_lanes_respect_time_scope_and_current_event_exclusion(tmp_path: Path) -> None:
    loom = ContextLoom(tmp_path / "loom_temporal.sqlite")
    graph = EntityGraphStore(loom.store)
    loom.compiler.lanes.extend(
        [MultiHopGraphLane(loom.store, graph), YCipherDepthSearchLane(loom.store, graph)]
    )

    past = loom.remember_user_statement(
        "Alex works at the library.", scope="test", record_id="past", created_at=10
    )
    future = loom.remember_user_statement(
        "Alex owns a time machine.", scope="test", record_id="future", created_at=30
    )
    current = loom.remember_user_statement(
        "Alex just entered the room.", scope="test", record_id="current", created_at=20
    )
    other_scope = loom.remember_user_statement(
        "Alex keeps a submarine.", scope="other", record_id="other", created_at=10
    )
    for record, relation, target in (
        (past, "works_at", "library"),
        (future, "owns", "time machine"),
        (current, "entered", "room"),
        (other_scope, "owns", "submarine"),
    ):
        graph.add_relation("Alex", relation, target, record_id=record.id)
        graph.link_record_entity(record.id, "Alex")

    packet = loom.compiler.compile(
        "What is known about Alex?",
        scope="test",
        current_event_id=current.id,
        now=20,
    )
    nominated = {
        source_id
        for entry in packet.manifest
        for source_id in entry.source_record_ids
    }
    assert past.id in nominated
    assert future.id not in nominated
    assert current.id not in nominated
    assert other_scope.id not in nominated
    loom.close()

