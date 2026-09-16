from __future__ import annotations

import math
import pytest

from context_loom import (
    CompilerConfig,
    ContextCompiler,
    ContextLoom,
    EvidenceStatus,
    LoomStore,
    MemoryRecord,
    RecordKind,
)
from context_loom.compiler import Candidate
from context_loom.locomo_evaluation import seed_conversation
from context_loom.retrieval import (
    STOPWORDS,
    ExactLane,
    HashingSemanticEmbedder,
    RetrievalContext,
    SemanticLane,
    clean_embedding_payload,
    terms,
)


def test_clean_embedding_payload_strips_headers_timestamps_speakers() -> None:
    # 1. Turn headers with ID and date/time + speaker
    text1 = "[D1:1 | 4:04 pm on 20 January, 2023] Jon: Lost my job as a banker yesterday, so I'm gonna take a shot at starting my own business."
    assert clean_embedding_payload(text1) == "Lost my job as a banker yesterday, so I'm gonna take a shot at starting my own business."

    # 2. Bracketed ISO timestamp + speaker
    text2 = "[2023-01-20T16:04:00+00:00] Gina: I opened a clothing store by the bay."
    assert clean_embedding_payload(text2) == "I opened a clothing store by the bay."

    # 3. Simple speaker prefix without turn header
    text3 = "Jon: I love ice cream."
    assert clean_embedding_payload(text3) == "I love ice cream."

    # 4. Scope term removal when scope is provided
    text4 = "[D1:2 | 4:05 pm] Gina: We are evaluating locomo results."
    assert "locomo" not in clean_embedding_payload(text4, scope="locomo").lower()

    # 5. Plain text without formatting
    text5 = "Plain conversational statement."
    assert clean_embedding_payload(text5) == "Plain conversational statement."


def test_semantic_lane_uses_clean_payload_v2_cache_key(tmp_path) -> None:
    store = LoomStore(tmp_path / "test.sqlite")
    embedder = HashingSemanticEmbedder()
    lane = SemanticLane(embedder)

    record = MemoryRecord(
        id="rec1",
        kind=RecordKind.USER,
        content="[D1:1 | 4:04 pm on 20 January, 2023] Jon: I love ice cream.",
        scope="test_scope",
        created_at=100.0,
        importance=0.8,
        confidence=1.0,
        evidence_status=EvidenceStatus.USER_REPORTED,
    )
    store.append_record(record)

    context = RetrievalContext(
        query="What does Jon like?",
        scope="test_scope",
        now=101.0,
        records=[record],
        store=store,
    )

    vector = lane._embedding(context, record)
    assert len(vector) == embedder.dimensions

    # Check that cache key is provider.identity + ':clean-payload-v2'
    expected_key = embedder.identity + ":clean-payload-v2"
    cached = store.get_embedding("rec1", expected_key)
    assert cached is not None
    assert cached == vector

    # Old scope-blind-v1 key must NOT be populated
    old_key = embedder.identity + ":scope-blind-v1"
    assert store.get_embedding("rec1", old_key) is None
    store.close()


def test_current_task_tool_return_skips_redundant_semantic_indexing(tmp_path) -> None:
    class CountingEmbedder:
        identity = "counting-v1"

        def __init__(self) -> None:
            self.inputs: list[str] = []

        def embed(self, text: str) -> list[float]:
            self.inputs.append(text)
            return [1.0, 0.0]

    embedder = CountingEmbedder()
    loom = ContextLoom(tmp_path / "deferred.sqlite", embedding_provider=embedder)
    try:
        loom.begin_task(
            "task",
            "Inspect and then write the file.",
            scope="work",
            stage="inspect",
        )
        tool_text = '{"ability":"ability:workspace-read","status":"success"}'
        tool_record = loom.record_tool_result(
            tool_text,
            task_id="task",
            verified=True,
            receipt_id="receipt:1",
            verifier="test",
            scope="work",
            metadata={"deferred_semantic_for_task": "task"},
        )
        packet = loom.compiler.compile(
            "Inspect and then write the file.",
            scope="work",
            task_id="task",
        )
        assert tool_text not in embedder.inputs
        assert loom.store.get_embedding(
            tool_record.id, embedder.identity + ":clean-payload-v2"
        ) is None
        assert any(
            entry.admitted and tool_record.id in entry.source_record_ids
            for entry in packet.manifest
        )
    finally:
        loom.close()


def test_terms_strips_leading_and_trailing_punctuation() -> None:
    raw = "Jon: I like ice cream."
    assert terms(raw) == ["jon", "i", "like", "ice", "cream"]

    mixed = "Google... 1800ES! /path/ (parentheses) [brackets] {braces} 'quotes' \"double\""
    expected = ["google", "1800es", "path", "parentheses", "brackets", "braces", "quotes", "double"]
    assert terms(mixed) == expected


def test_stopwords_expansion() -> None:
    # Interrogatives
    for word in ["who", "whom", "whose", "why", "where", "what", "when", "how"]:
        assert word in STOPWORDS, f"Missing interrogative stopword: {word}"

    # Inquiry verbs
    for word in ["tell", "show", "find", "know"]:
        assert word in STOPWORDS, f"Missing inquiry verb stopword: {word}"

    # Contraction tails
    for tail in ["s", "d", "ll", "m", "re", "ve"]:
        assert tail in STOPWORDS, f"Missing contraction tail stopword: {tail}"


def test_exact_lane_matches_cleaned_speaker_and_sentence_tokens(tmp_path) -> None:
    store = LoomStore(tmp_path / "test_exact.sqlite")
    record = MemoryRecord(
        id="rec1",
        kind=RecordKind.USER,
        content="[D1:1 | 4:04 pm on 20 January, 2023] Jon: I like ice cream.",
        scope="chat",
        created_at=100.0,
        importance=0.8,
        confidence=1.0,
        evidence_status=EvidenceStatus.USER_REPORTED,
    )
    store.append_record(record)

    context = RetrievalContext(
        query="What does Jon like?",
        scope="chat",
        now=101.0,
        records=[record],
        store=store,
    )

    lane = ExactLane()
    candidates = lane.nominate(context)
    assert len(candidates) >= 1
    # Both 'jon' and 'like' should match cleanly
    assert candidates[0].lane_scores["exact"] >= 1.0
    store.close()


def test_compiler_config_defaults() -> None:
    config = CompilerConfig()
    assert config.lane_weights.get("multi_hop_graph") == 1.25
    assert config.lane_weights.get("exact") == 1.15
    assert config.lane_weights.get("semantic") == 1.05
    assert config.lane_weights.get("correction") == 1.35


def test_compiler_knapsack_sublinear_density_dampening(tmp_path) -> None:
    store = LoomStore(tmp_path / "test_knapsack.sqlite")
    # Budget of 120 tokens
    config = CompilerConfig(
        model_context_tokens=320,
        reserved_output_tokens=200,  # leaves max_input_tokens = 120
    )
    compiler = ContextCompiler(store, config=config)

    # Informative turn: 80 tokens, utility ~ 0.90
    # Density under sublinear: 0.90 / sqrt(80 / 24) = 0.90 / 1.825 = 0.493
    # Density under old linear: 0.90 / (80 / 24) = 0.90 / 3.333 = 0.270
    informative_text = "Jon: " + "We discussed opening a dance studio by the bay and funding it. " * 5
    informative = Candidate(
        id="informative",
        text=informative_text,
        section="continuity",
        source_record_ids=("info1",),
        lane_scores={"exact": 0.78},  # utility = 1.15 * 0.78 = 0.897
        reasons=["exact"],
        evidence_status=EvidenceStatus.USER_REPORTED,
    )

    # Filler turn: 24 tokens, utility ~ 0.45
    # Density under sublinear: 0.45 / sqrt(24 / 24) = 0.45
    # Density under old linear: 0.45 / 1.0 = 0.45
    filler_text = "Jon: Yeah sounds good."
    filler = Candidate(
        id="filler",
        text=filler_text,
        section="continuity",
        source_record_ids=("fill1",),
        lane_scores={"semantic": 0.42},  # utility = 1.05 * 0.42 = 0.441
        reasons=["semantic"],
        evidence_status=EvidenceStatus.USER_REPORTED,
    )

    # Under sublinear: informative utility density (0.493) > filler utility density (0.441)
    # So informative is sorted before filler!
    cost_info = compiler.token_counter.count(informative.text)
    cost_fill = compiler.token_counter.count(filler.text)
    u_info = compiler._utility(informative)
    u_fill = compiler._utility(filler)

    dampened_info = u_info / max(1.0, math.sqrt(cost_info / 24.0))
    dampened_fill = u_fill / max(1.0, math.sqrt(cost_fill / 24.0))
    assert dampened_info > dampened_fill

    # Now verify ContextCompiler.compile() soft knapsack ordering
    class MockLane:
        name = "mock"
        def nominate(self, ctx):
            return [informative, filler]

    compiler.lanes = [MockLane()]
    packet = compiler.compile("dance studio", scope="chat", now=10.0)
    admitted = [m.candidate_id for m in packet.manifest if m.admitted]
    assert "informative" in admitted
    store.close()


def test_locomo_evaluation_uses_compact_token_budget_without_item_caps(tmp_path) -> None:
    conversation = {
        "speaker_a": "Jon",
        "speaker_b": "Gina",
        "session_1_date_time": "4:04 pm on 20 January, 2023",
        "session_1": [
            {"speaker": "Jon", "dia_id": "D1:1", "text": "Hello."},
        ],
    }
    loom, _, _ = seed_conversation(
        conversation,
        database=tmp_path / "locomo_eval.sqlite",
        embedding_provider=HashingSemanticEmbedder(),
    )
    try:
        assert loom.compiler.config.max_input_tokens == 1100
        assert "continuity" not in loom.compiler.config.section_limits
        assert "evidence" not in loom.compiler.config.section_limits
    finally:
        loom.close()
