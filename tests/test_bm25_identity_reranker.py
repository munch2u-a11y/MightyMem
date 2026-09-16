"""Tests for BM25 and Regex Identity Intent Reranking in Context Loom."""
from __future__ import annotations

import pytest

from context_loom import (
    CompilerConfig,
    ContextCompiler,
    EvidenceStatus,
    LoomStore,
    MemoryRecord,
    RecordKind,
)
from context_loom.retrieval import (
    BM25IdentityReranker,
    ExactLane,
    RetrievalContext,
    clean_embedding_payload,
)


def test_exact_lane_uses_cleaned_header_payload(tmp_path) -> None:
    store = LoomStore(tmp_path / "test_exact_header.sqlite")
    # Filler turn spoken by Caroline
    record_filler = MemoryRecord(
        id="rec_filler",
        kind=RecordKind.USER,
        content="[D1:1 | 4:04 pm on 20 January, 2023 | Caroline]: Thanks, Mel! That sounds awesome.",
        scope="chat",
        created_at=100.0,
        importance=0.5,
        confidence=1.0,
        evidence_status=EvidenceStatus.USER_REPORTED,
    )
    # Gold evidence turn spoken by Caroline
    record_gold = MemoryRecord(
        id="rec_gold",
        kind=RecordKind.USER,
        content="[D1:2 | 4:05 pm on 20 January, 2023 | Caroline]: I got hired as an architect at a studio.",
        scope="chat",
        created_at=101.0,
        importance=0.9,
        confidence=1.0,
        evidence_status=EvidenceStatus.USER_REPORTED,
    )
    store.append_record(record_filler)
    store.append_record(record_gold)

    context = RetrievalContext(
        query="What is Caroline's job as an architect?",
        scope="chat",
        now=102.0,
        records=[record_filler, record_gold],
        store=store,
    )

    lane = ExactLane()
    candidates = lane.nominate(context)
    
    # Fillers that only contain 'Caroline' in the header prefix must NOT match 'caroline' term
    # when header cleaning is applied, because 'Caroline:' is header metadata, not message body.
    matched_ids = [c.source_record_ids[0] for c in candidates if c.lane_scores.get("exact", 0.0) > 0.0]
    assert record_gold.id in matched_ids
    assert record_filler.id not in matched_ids, "Filler turn matched ExactLane solely due to speaker label in header"
    store.close()


def test_bm25_identity_reranker_boosts_gold_identity_fact(tmp_path) -> None:
    store = LoomStore(tmp_path / "test_reranker.sqlite")
    
    record_filler = MemoryRecord(
        id="rec1",
        kind=RecordKind.USER,
        content="[D1:1 | 4:04 pm | Caroline]: Thanks, Mel!",
        scope="locomo",
        created_at=100.0,
        importance=0.5,
        confidence=1.0,
        evidence_status=EvidenceStatus.USER_REPORTED,
    )
    record_unrelated_job = MemoryRecord(
        id="rec2",
        kind=RecordKind.USER,
        content="[D1:2 | 4:05 pm | Bob]: I work as a software engineer at a tech startup.",
        scope="locomo",
        created_at=101.0,
        importance=0.8,
        confidence=1.0,
        evidence_status=EvidenceStatus.USER_REPORTED,
    )
    record_gold = MemoryRecord(
        id="rec3",
        kind=RecordKind.USER,
        content="[D1:3 | 4:06 pm | Caroline]: I was hired as an architect last week.",
        scope="locomo",
        created_at=102.0,
        importance=0.9,
        confidence=1.0,
        evidence_status=EvidenceStatus.USER_REPORTED,
    )
    record_weaker_identity_match = MemoryRecord(
        id="rec4",
        kind=RecordKind.USER,
        content="[D1:4 | 4:07 pm | Caroline]: I am considering renting a studio.",
        scope="locomo",
        created_at=103.0,
        importance=0.5,
        confidence=1.0,
        evidence_status=EvidenceStatus.USER_REPORTED,
    )
    store.append_record(record_filler)
    store.append_record(record_unrelated_job)
    store.append_record(record_gold)
    store.append_record(record_weaker_identity_match)

    context = RetrievalContext(
        query="What is Caroline's profession as an architect?",
        scope="locomo",
        now=103.0,
        records=[record_filler, record_unrelated_job, record_gold, record_weaker_identity_match],
        store=store,
    )

    reranker = BM25IdentityReranker()
    scores = reranker.score_candidates(
        context,
        [record_filler, record_unrelated_job, record_gold, record_weaker_identity_match],
    )

    assert all(0.0 <= score <= 1.0 for score in scores.values())
    assert scores[record_gold.id] == 1.0
    assert scores[record_gold.id] > scores[record_weaker_identity_match.id]
    assert scores[record_gold.id] > scores[record_unrelated_job.id], "Gold identity turn must score higher than generic job statement"
    assert scores[record_gold.id] > scores[record_filler.id], "Gold identity turn must score higher than filler turn"
    store.close()
