"""Adversarial and Empirical Challenge Test Suite for MultiHopGraphLane and Candidate Quality.

Challenger: Challenger 2 (Retrieval Quality Challenger)
Scope: Question stopword filtering, filler turn downranking, top-k candidate capping,
and candidate text/reason formatting conformance with compiler expectations.
"""
from __future__ import annotations

from pathlib import Path
import pytest

from context_loom.compiler import ContextBudgetExceeded
from context_loom.engine import ContextLoom
from context_loom.entity_graph import EntityGraphStore, EntityKind, RelationKind
from context_loom.models import Candidate, EvidenceStatus, RecordKind
from context_loom.multi_hop_lane import MultiHopGraphLane, QUESTION_STOPWORDS
from context_loom.retrieval import RetrievalContext, candidate_for, render_record, section_for
from context_loom.store import LoomStore


# ==============================================================================
# Suite 1: Question Stopword Filtering
# ==============================================================================

@pytest.mark.parametrize(
    "interrogative",
    ["Where", "What", "When", "Who", "Why", "How", "Which", "Whom", "Whose"],
)
def test_interrogative_prefix_does_not_produce_entity_seeds(
    tmp_path: Path, interrogative: str
) -> None:
    """Verify queries starting with Where, What, When, Who, Why, How, etc. never produce interrogatives as seeds."""
    loom = ContextLoom(tmp_path / "loom.sqlite")
    graph = EntityGraphStore(loom.store)
    lane = MultiHopGraphLane(loom.store, graph)

    rec = loom.record(
        RecordKind.USER,
        "Alice founded a robotics venture in Boston.",
        scope="test",
        record_id="rec-alice-1",
    )
    graph.link_record_entity(rec.id, "Alice")

    query = f"{interrogative} did Alice launch her venture?"
    ctx = RetrievalContext(
        query=query,
        scope="test",
        now=100.0,
        records=[rec],
        store=loom.store,
    )

    captured_seeds: list[str] = []
    orig_multi_hop = graph.multi_hop_records

    def spy_multi_hop(seed_entity_names, max_hops=2, **kwargs):
        captured_seeds.extend(seed_entity_names)
        return orig_multi_hop(seed_entity_names, max_hops=max_hops, **kwargs)

    graph.multi_hop_records = spy_multi_hop

    candidates = lane.nominate(ctx)
    captured_lower = {s.casefold() for s in captured_seeds}

    # Interrogative must NOT appear in seeds
    assert interrogative.casefold() not in captured_lower, (
        f"Interrogative '{interrogative}' leaked into multi-hop seeds: {captured_seeds}"
    )
    # Target entity must be extracted
    assert "alice" in captured_lower
    # Auxiliaries/stopwords must not leak
    for stop in ["did", "her", "venture", "launch"]:
        if stop in QUESTION_STOPWORDS:
            assert stop not in captured_lower

    assert len(candidates) >= 1
    assert candidates[0].source_record_ids[0] == rec.id
    loom.close()


@pytest.mark.parametrize(
    "query",
    [
        "Where?",
        "What is this?",
        "When was it?",
        "Who is that?",
        "Why did this happen?",
        "How do you do that?",
        "Where, when, why, and how?",
        "What's what?",
        "Tell me about this and that.",
    ],
)
def test_pure_stopword_queries_produce_zero_seeds_and_empty_candidates(
    tmp_path: Path, query: str
) -> None:
    """Adversarial stress: queries containing solely interrogatives, auxiliaries, or stopwords return empty."""
    loom = ContextLoom(tmp_path / "loom.sqlite")
    graph = EntityGraphStore(loom.store)
    lane = MultiHopGraphLane(loom.store, graph)

    rec = loom.record(
        RecordKind.USER,
        "General background conversation without named entities.",
        scope="test",
        record_id="rec-dummy",
    )

    ctx = RetrievalContext(
        query=query,
        scope="test",
        now=100.0,
        records=[rec],
        store=loom.store,
    )

    called_seeds: list[str] = []
    graph.multi_hop_records = lambda seeds, **kw: called_seeds.extend(seeds) or []

    candidates = lane.nominate(ctx)
    assert candidates == [], f"Expected empty candidates for query '{query}', got {candidates}"
    assert len(called_seeds) == 0, f"Expected no seed expansion for query '{query}', got {called_seeds}"
    loom.close()


def test_adversarial_casing_and_punctuation_interrogatives(tmp_path: Path) -> None:
    """Stress test variations: WHERE (all caps), Where's (contraction), **Where** (markdown)."""
    loom = ContextLoom(tmp_path / "loom.sqlite")
    graph = EntityGraphStore(loom.store)
    lane = MultiHopGraphLane(loom.store, graph)

    rec = loom.record(
        RecordKind.USER,
        "Bob lives in Denver.",
        scope="test",
        record_id="rec-bob-1",
    )
    graph.link_record_entity(rec.id, "Bob")

    queries = [
        "WHERE did Bob relocate?",
        "Where's Bob living now?",
        "**Where** did Bob go?",
        "Bob relocated where exactly?",
    ]
    orig = graph.multi_hop_records
    for q in queries:
        called: list[str] = []

        def spy(seeds, **kw):
            called.extend(seeds)
            return orig(seeds, **kw)

        graph.multi_hop_records = spy
        ctx = RetrievalContext(query=q, scope="test", now=100.0, records=[rec], store=loom.store)
        lane.nominate(ctx)

        called_lower = {s.casefold() for s in called}
        assert "where" not in called_lower, f"Failed on query '{q}': 'where' found in seeds: {called}"
        assert "bob" in called_lower, f"Failed on query '{q}': 'bob' missing from seeds: {called}"

    loom.close()


# ==============================================================================
# Suite 2: Filler Turn Downranking
# ==============================================================================

def test_filler_turns_receive_exact_0_35_penalty_and_rank_below_substantive(
    tmp_path: Path,
) -> None:
    """Empirically verify 1-word and 2-word conversational greetings receive 0.35x penalty and rank below evidence."""
    loom = ContextLoom(tmp_path / "loom.sqlite")
    graph = EntityGraphStore(loom.store)
    lane = MultiHopGraphLane(loom.store, graph)

    # 1-word greetings
    rec_1word_1 = loom.record(RecordKind.USER, "Bye!", scope="test", record_id="filler-1w-1")
    rec_1word_2 = loom.record(RecordKind.USER, "Thanks!", scope="test", record_id="filler-1w-2")

    # 2-word conversational turns (per prompt specification: "Jon: Bye!", "Gina: Thanks!")
    rec_2word_1 = loom.record(RecordKind.USER, "Jon: Bye!", scope="test", record_id="filler-2w-1")
    rec_2word_2 = loom.record(RecordKind.USER, "Gina: Thanks!", scope="test", record_id="filler-2w-2")

    # 3-word conversational turn
    rec_3word_1 = loom.record(RecordKind.USER, "Jon: See ya!", scope="test", record_id="filler-3w-1")

    # Substantive turns with >= 4 words
    rec_substantive_1 = loom.record(
        RecordKind.USER,
        "Jon relocated his robotics engineering firm to Seattle.",
        scope="test",
        record_id="substantive-1",
    )
    rec_substantive_2 = loom.record(
        RecordKind.FACT,
        "Jon was appointed lead systems architect at Meridian.",
        scope="test",
        record_id="substantive-2",
    )

    all_records = [
        rec_1word_1,
        rec_1word_2,
        rec_2word_1,
        rec_2word_2,
        rec_3word_1,
        rec_substantive_1,
        rec_substantive_2,
    ]

    # Link all records to "Jon" so base graph score is identical for all turns
    for r in all_records:
        graph.link_record_entity(r.id, "Jon")

    ctx = RetrievalContext(
        query="Tell me about Jon's career background.",
        scope="test",
        now=100.0,
        records=all_records,
        store=loom.store,
    )

    candidates = lane.nominate(ctx)
    candidate_map = {c.source_record_ids[0]: c for c in candidates}

    # 1. Check all candidates are nominated
    assert len(candidates) == len(all_records)

    # 2. Check substantive turns rank strictly above all filler turns
    substantive_ids = {"substantive-1", "substantive-2"}
    filler_ids = {"filler-1w-1", "filler-1w-2", "filler-2w-1", "filler-2w-2", "filler-3w-1"}

    top_2_ids = {c.source_record_ids[0] for c in candidates[:2]}
    assert top_2_ids == substantive_ids, (
        f"Substantive turns did not take top 2 ranks! Candidates order: {[c.source_record_ids[0] for c in candidates]}"
    )

    # 3. Verify exact score dampening factor:
    # Substantive turns have length_factor = 1.0; filler turns have length_factor = 0.35.
    sub1_score = candidate_map["substantive-1"].lane_scores["multi_hop_graph"]
    sub2_score = candidate_map["substantive-2"].lane_scores["multi_hop_graph"]
    filler_2w_1_score = candidate_map["filler-2w-1"].lane_scores["multi_hop_graph"]
    filler_2w_2_score = candidate_map["filler-2w-2"].lane_scores["multi_hop_graph"]
    filler_1w_1_score = candidate_map["filler-1w-1"].lane_scores["multi_hop_graph"]
    filler_1w_2_score = candidate_map["filler-1w-2"].lane_scores["multi_hop_graph"]

    assert sub1_score > filler_2w_1_score * 2.0, (
        f"Substantive turn ({sub1_score}) is not significantly higher than 'Jon: Bye!' ({filler_2w_1_score})"
    )
    assert sub2_score > filler_2w_2_score * 2.0, (
        f"Substantive turn ({sub2_score}) is not significantly higher than 'Gina: Thanks!' ({filler_2w_2_score})"
    )
    assert sub1_score > filler_1w_1_score * 2.0
    assert sub2_score > filler_1w_2_score * 2.0

    # 4. Check that 2-word greetings without query match receive exactly 0.35 * base_score
    # For rec_2word_2 ("Gina: Thanks!"), query terms are {"jon", "career", "background"}.
    # Overlap between {"gina", "thanks"} and query terms is 0 -> relevance_boost = 1.0.
    # Therefore, final_score == base_score * 1.0 * 0.35.
    base_score = graph.multi_hop_records(["Jon"], max_hops=1)[0]["score"]
    expected_gina_thanks = max(0.05, min(1.0, base_score * 1.0 * 0.35))
    assert abs(filler_2w_2_score - expected_gina_thanks) < 1e-6, (
        f"Gina: Thanks! score {filler_2w_2_score} != expected {expected_gina_thanks}"
    )

    loom.close()


# ==============================================================================
# Suite 3: Top-K Candidate Capping
# ==============================================================================

def test_default_multi_hop_lane_has_no_hidden_nomination_cap(tmp_path: Path) -> None:
    loom = ContextLoom(tmp_path / "loom.sqlite")
    lane = MultiHopGraphLane(loom.store)
    assert lane.top_k is None
    loom.close()


def test_top_k_candidate_capping_bounds_nomination_pool(tmp_path: Path) -> None:
    """Verify that when a seed matches 100+ records, MultiHopGraphLane bounds nominations to top_k."""
    loom = ContextLoom(tmp_path / "loom.sqlite")
    graph = EntityGraphStore(loom.store)

    total_records = 125  # 100+ records
    records = []
    for i in range(total_records):
        # Sprinkle query terms on some records to create distinct score strata
        if i % 10 == 0:
            content = f"Record {i} provides detailed engineering blueprints for Charlie's quantum computer."
        else:
            content = f"Record {i} mentions Charlie and informal discussion {i}."
        rec = loom.record(RecordKind.USER, content, scope="test", record_id=f"rec-charlie-{i:03d}")
        records.append(rec)
        graph.link_record_entity(rec.id, "Charlie")

    ctx = RetrievalContext(
        query="Tell me about Charlie's quantum computer engineering.",
        scope="test",
        now=100.0,
        records=records,
        store=loom.store,
    )

    # 1. Bounding to top_k=30
    lane_30 = MultiHopGraphLane(loom.store, graph, top_k=30)
    candidates_30 = lane_30.nominate(ctx)
    assert len(candidates_30) == 30, (
        f"Expected exactly 30 candidates from 125 matches, got {len(candidates_30)}"
    )

    # Verify descending score sorting
    scores_30 = [c.lane_scores["multi_hop_graph"] for c in candidates_30]
    assert all(scores_30[i] >= scores_30[i + 1] for i in range(len(scores_30) - 1)), (
        f"Candidates are not sorted descending by score: {scores_30}"
    )

    # Verify that high-overlap records (blueprint turns) are all within the top 30
    admitted_ids = {c.source_record_ids[0] for c in candidates_30}
    for i in range(0, total_records, 10):
        expected_high_scorer = f"rec-charlie-{i:03d}"
        assert expected_high_scorer in admitted_ids, (
            f"High-scoring record {expected_high_scorer} was truncated despite top-30 capacity!"
        )

    # 2. Custom top_k=10
    lane_10 = MultiHopGraphLane(loom.store, graph, top_k=10)
    candidates_10 = lane_10.nominate(ctx)
    assert len(candidates_10) == 10

    # 3. Custom top_k=50
    lane_50 = MultiHopGraphLane(loom.store, graph, top_k=50)
    candidates_50 = lane_50.nominate(ctx)
    assert len(candidates_50) == 50

    # 4. Custom top_k=200 (greater than total records)
    lane_200 = MultiHopGraphLane(loom.store, graph, top_k=200)
    candidates_200 = lane_200.nominate(ctx)
    assert len(candidates_200) == total_records

    # 5. Boundary case top_k=0
    lane_0 = MultiHopGraphLane(loom.store, graph, top_k=0)
    candidates_0 = lane_0.nominate(ctx)
    assert len(candidates_0) == 0

    loom.close()


def test_default_instance_returns_every_eligible_graph_candidate(tmp_path: Path) -> None:
    loom = ContextLoom(tmp_path / "loom.sqlite")
    graph = EntityGraphStore(loom.store)
    lane_default = MultiHopGraphLane(loom.store, graph)

    records = []
    for i in range(120):
        rec = loom.record(RecordKind.USER, f"Turn {i} with Dana.", scope="test", record_id=f"rec-dana-{i}")
        records.append(rec)
        graph.link_record_entity(rec.id, "Dana")

    ctx = RetrievalContext(
        query="Tell me about Dana.",
        scope="test",
        now=100.0,
        records=records,
        store=loom.store,
    )
    candidates = lane_default.nominate(ctx)
    assert len(candidates) == len(records)
    loom.close()


# ==============================================================================
# Suite 4: Candidate Object Formatting and Compiler Conformance
# ==============================================================================

def test_candidate_text_reason_formatting_and_compiler_expectations(tmp_path: Path) -> None:
    """Verify Candidate objects conform to exact compiler expectations and pipeline integration."""
    loom = ContextLoom(tmp_path / "loom.sqlite")
    graph = EntityGraphStore(loom.store)
    lane = MultiHopGraphLane(loom.store, graph)

    # 1. Direct 1-hop relation
    rec_direct = loom.record(
        RecordKind.USER,
        "Diana started a graphic design agency in Chicago.",
        scope="test",
        record_id="rec-diana-direct",
        created_at=100.0,
        evidence_status=EvidenceStatus.USER_REPORTED,
        importance=0.8,
    )
    graph.link_record_entity(rec_direct.id, "Diana")

    # 2. 2-hop relation: Diana -> WORKS_WITH -> Edward
    rec_indirect = loom.record(
        RecordKind.FACT,
        "Edward is head of typography at the studio.",
        scope="test",
        record_id="rec-edward-indirect",
        created_at=100.0,
        evidence_status=EvidenceStatus.TOOL_VERIFIED,
        importance=0.9,
    )
    graph.add_relation("Diana", RelationKind.WORKS_WITH, "Edward", record_id=rec_indirect.id)
    graph.link_record_entity(rec_indirect.id, "Edward")

    ctx = RetrievalContext(
        query="Where does Diana and her partner work?",
        scope="test",
        now=200.0,
        records=[rec_direct, rec_indirect],
        store=loom.store,
    )

    candidates = lane.nominate(ctx)
    assert len(candidates) == 2

    # Check candidate structure
    for c in candidates:
        # id contract
        assert c.id.startswith("record:")
        assert c.id in {"record:rec-diana-direct", "record:rec-edward-indirect"}

        # text rendering contract
        assert c.text.startswith("- ")
        assert f"[{c.id}]" in c.text or f"[{c.id} " in c.text

        # section contract: must match section_for(record)
        matched_rec = rec_direct if c.source_record_ids[0] == rec_direct.id else rec_indirect
        assert c.section == section_for(matched_rec)

        # source record IDs tuple contract
        assert isinstance(c.source_record_ids, tuple)
        assert len(c.source_record_ids) == 1

        # lane scores contract
        assert "multi_hop_graph" in c.lane_scores
        score = c.lane_scores["multi_hop_graph"]
        assert 0.05 <= score <= 1.0

        # reasons list contract
        assert isinstance(c.reasons, list)
        assert len(c.reasons) >= 1
        assert c.reasons[0].startswith("multi-hop graph propagation (")
        assert c.reasons[0].endswith(")")

        # metadata contract
        assert "kind" in c.metadata
        assert "source_ids" in c.metadata

    # Check specific 1-hop vs 2-hop reason string
    c_direct = [c for c in candidates if c.source_record_ids[0] == rec_direct.id][0]
    c_indirect = [c for c in candidates if c.source_record_ids[0] == rec_indirect.id][0]

    assert c_direct.reasons == ["multi-hop graph propagation (diana)"]
    assert c_indirect.reasons == ["multi-hop graph propagation (diana->works_with->edward)"]

    # Check EvidenceStatus preservation
    assert c_direct.evidence_status == EvidenceStatus.USER_REPORTED
    assert c_indirect.evidence_status == EvidenceStatus.TOOL_VERIFIED

    # Check End-to-End Compiler Pipeline Execution
    loom.compiler.lanes.append(lane)
    packet = loom.compiler.compile("Tell me about Diana and typography", scope="test", now=200.0)

    # Verify context packet validity
    assert packet.scope == "test"
    assert len(packet.rendered) > 0
    assert "CURRENT INPUT" in packet.rendered

    # Verify manifest entries
    manifest_candidate_ids = {entry.candidate_id for entry in packet.manifest}
    assert "record:rec-diana-direct" in manifest_candidate_ids
    assert "record:rec-edward-indirect" in manifest_candidate_ids

    # Verify manifest entries for multi_hop_graph lane are admitted
    admitted_multi_hop = [
        entry for entry in packet.manifest
        if "multi_hop_graph" in entry.lanes and entry.admitted
    ]
    assert len(admitted_multi_hop) >= 1, "Expected at least one multi_hop candidate to be admitted in packet!"
    for entry in admitted_multi_hop:
        assert entry.lanes["multi_hop_graph"] >= 0.05
        assert entry.utility > 0.0

    loom.close()


# ==============================================================================
# Suite 5: Adversarial Boundary Conditions & Robustness
# ==============================================================================

def test_scope_isolation_in_multi_hop_lane(tmp_path: Path) -> None:
    """Adversarial challenge: records belonging to other scopes must NEVER be nominated."""
    loom = ContextLoom(tmp_path / "loom.sqlite")
    graph = EntityGraphStore(loom.store)
    lane = MultiHopGraphLane(loom.store, graph)

    # Record in 'allowed_scope'
    rec_allowed = loom.record(
        RecordKind.USER,
        "Alice submitted the proposal.",
        scope="allowed_scope",
        record_id="rec-scope-allowed",
    )
    graph.link_record_entity(rec_allowed.id, "Alice")

    # Record in 'private_scope' linked to same entity
    rec_private = loom.record(
        RecordKind.USER,
        "Alice leaked private financial credentials.",
        scope="private_scope",
        record_id="rec-scope-private",
    )
    graph.link_record_entity(rec_private.id, "Alice")

    ctx = RetrievalContext(
        query="What did Alice do?",
        scope="allowed_scope",
        now=100.0,
        records=[rec_allowed],  # Context only contains allowed_scope records
        store=loom.store,
    )

    candidates = lane.nominate(ctx)
    candidate_record_ids = [c.source_record_ids[0] for c in candidates]

    assert rec_allowed.id in candidate_record_ids
    assert rec_private.id not in candidate_record_ids, (
        f"Cross-scope security violation! Private record {rec_private.id} nominated in scope 'allowed_scope'!"
    )
    loom.close()


def test_missing_or_corrupted_record_in_graph_store(tmp_path: Path) -> None:
    """Adversarial challenge: graph links to a record ID that was deleted or missing from LoomStore."""
    loom = ContextLoom(tmp_path / "loom.sqlite")
    graph = EntityGraphStore(loom.store)
    lane = MultiHopGraphLane(loom.store, graph)

    # Link a ghost record that does not exist in SQLite memory_records
    graph.link_record_entity("ghost-record-999", "Casper")

    ctx = RetrievalContext(
        query="Where is Casper?",
        scope="test",
        now=100.0,
        records=[],
        store=loom.store,
    )

    # Must NOT crash with NoneType attribute error
    candidates = lane.nominate(ctx)
    assert candidates == []
    loom.close()


def test_candidate_merging_with_another_lane(tmp_path: Path) -> None:
    """Verify Candidate.merge correctly merges multi_hop_graph with another lane."""
    loom = ContextLoom(tmp_path / "loom.sqlite")
    rec = loom.record(RecordKind.USER, "Meeting notes with George", scope="test", record_id="rec-geo")

    c1 = candidate_for(rec, "multi_hop_graph", 0.60, "multi-hop graph propagation (george)")
    c2 = candidate_for(rec, "semantic", 0.85, "semantic similarity 0.85")

    c1.merge(c2)

    assert c1.lane_scores == {"multi_hop_graph": 0.60, "semantic": 0.85}
    assert c1.reasons == ["multi-hop graph propagation (george)", "semantic similarity 0.85"]

    # Higher score should be picked up if same lane merges
    c3 = candidate_for(rec, "multi_hop_graph", 0.90, "multi-hop propagation boost")
    c1.merge(c3)
    assert c1.lane_scores["multi_hop_graph"] >= 0.90
    assert len(c1.reasons) == 3
    loom.close()
