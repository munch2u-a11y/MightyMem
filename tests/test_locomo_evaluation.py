from __future__ import annotations

import json

import pytest

import context_loom.locomo_evaluation as locomo_evaluation
from context_loom.cli import parser
from context_loom.locomo_evaluation import (
    build_grade_prompt,
    parse_session_time,
    parse_verdict,
    ollama_generate,
    retrieval_summary,
    seed_conversation,
    summarise,
)
from context_loom.retrieval import HashingSemanticEmbedder


def test_parse_session_time_preserves_order() -> None:
    assert parse_session_time("4:04 pm on 20 January, 2023") < parse_session_time(
        "9:00 am on 21 January, 2023"
    )


def test_locomo_cli_defaults_to_compact_receipted_packet_budget() -> None:
    args = parser().parse_args(
        ["locomo-retrieval-diagnostic", "--ollama-server-pid", "421"]
    )
    assert args.compiler_context_tokens == 1500
    assert args.compiler_reserved_output_tokens == 400
    assert args.ollama_url == "http://127.0.0.1:11434"
    assert args.ollama_server_pid == 421


def test_adversarial_grade_contract_treats_dataset_answer_as_distractor() -> None:
    prompt = build_grade_prompt(
        "What instrument did Gina play?",
        "",
        "That was Jon, not Gina.",
        adversarial=True,
        distractor="piano",
    )
    assert "false premise" in prompt
    assert "Distractor: piano" in prompt
    assert "Expected answer" not in prompt


def test_parse_verdict_requires_standalone_verdict() -> None:
    assert parse_verdict("PASS") is True
    assert parse_verdict("FAIL") is False
    assert parse_verdict("ungraded") is None


def test_seed_conversation_keeps_gold_dialog_identity(tmp_path) -> None:
    conversation = {
        "speaker_a": "Jon",
        "speaker_b": "Gina",
        "session_1_date_time": "4:04 pm on 20 January, 2023",
        "session_1": [
            {"speaker": "Jon", "dia_id": "D1:1", "text": "I opened a dance studio by the water."},
            {"speaker": "Gina", "dia_id": "D1:2", "text": "That sounds wonderful."},
            {"speaker": "Jon", "dia_id": "D1:3", "text": "Gina visited the dance studio."},
        ],
    }
    loom, record_map, now = seed_conversation(
        conversation,
        database=tmp_path / "locomo.sqlite",
        embedding_provider=HashingSemanticEmbedder(),
    )
    try:
        packet = loom.compiler.compile(
            "Who opened a dance studio by the water?", scope="locomo", now=now
        )
        admitted = {
            record_map[source]
            for item in packet.manifest
            if item.admitted
            for source in item.source_record_ids
            if source in record_map
        }
        assert "D1:1" in admitted
        assert "D1:3" in admitted
        last_record = loom.store.get_record(next(record_id for record_id, dialog_id in record_map.items() if dialog_id == "D1:3"))
        assert last_record is not None
        assert now > last_record.created_at

        cooccurrences = loom.store.connection.execute(
            "SELECT COUNT(*) FROM entity_edges WHERE relation = 'co_occurred_with'"
        ).fetchone()[0]
        assert cooccurrences > 0
        assert packet.token_count <= packet.max_input_tokens
    finally:
        loom.close()


def test_summary_separates_retrieval_and_reader_misses() -> None:
    base = {
        "category": 4,
        "evidence_count": 1,
        "injected_tokens": 100,
        "compiler_tokens": 120,
        "answer_metrics": {"prompt_eval_count": 10, "eval_count": 2},
        "judge_metrics": {"prompt_eval_count": 5, "eval_count": 1},
    }
    summary = summarise(
        [
            {**base, "correct": False, "evidence_hits": 1, "full_evidence_hit": True},
            {**base, "correct": False, "evidence_hits": 0, "full_evidence_hit": False},
        ]
    )
    assert summary["misses_with_full_evidence"] == 1
    assert summary["misses_with_incomplete_evidence"] == 1
    assert summary["evidence_micro_recall"] == 0.5


def test_retrieval_summary_scores_exact_evidence_ids_and_packet_efficiency() -> None:
    rows = [
        {
            "category": 4,
            "evidence_count": 2,
            "evidence_hits": 2,
            "evidence_recall": 1.0,
            "any_evidence_hit": True,
            "full_evidence_hit": True,
            "injected_tokens": 500,
            "compiler_tokens": 800,
        },
        {
            "category": 4,
            "evidence_count": 1,
            "evidence_hits": 0,
            "evidence_recall": 0.0,
            "any_evidence_hit": False,
            "full_evidence_hit": False,
            "injected_tokens": 500,
            "compiler_tokens": 800,
        },
    ]
    summary = retrieval_summary(rows)
    assert summary["evidence_micro_recall"] == 2 / 3
    assert summary["full_evidence_hit_rate"] == 0.5
    assert summary["recall_token_efficiency_ratio"] == 100 * (2 / 3) / 0.5


def test_ollama_generation_declares_context_and_has_no_implicit_output_cap(monkeypatch) -> None:
    captured: dict = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def read(self):
            return json.dumps({"response": "complete", "done": True, "done_reason": "stop"}).encode()

    def fake_urlopen(request, timeout=None):
        captured.update(json.loads(request.data))
        return Response()

    monkeypatch.setattr(locomo_evaluation.urllib.request, "urlopen", fake_urlopen)
    text, metrics = ollama_generate("prompt", model="local", num_ctx=4096)

    assert text == "complete"
    assert metrics["done"] is True
    assert captured["options"]["num_ctx"] == 4096
    assert "num_predict" not in captured["options"]


def test_ollama_generation_rejects_nonempty_partial_output(monkeypatch) -> None:
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def read(self):
            return json.dumps(
                {"response": "plausible but unfinished", "done": True, "done_reason": "length"}
            ).encode()

    monkeypatch.setattr(locomo_evaluation.urllib.request, "urlopen", lambda *args, **kwargs: Response())
    with pytest.raises(RuntimeError, match="generation boundary"):
        ollama_generate("prompt", model="local", num_ctx=4096)
