from __future__ import annotations

import json

import pytest

from context_loom.capabilities import default_capabilities
from context_loom.evaluation import BenchmarkCase
from context_loom.llm_evaluation import (
    HARNESS_VERSION,
    answer_requirements,
    grade_answer,
    ollama_answer,
    render_llm_report,
    run_llm_benchmark,
)
from context_loom.models import EvidenceStatus, MemoryRecord, RecordKind


def case(category: str, fragments: tuple[str, ...] = ()) -> BenchmarkCase:
    return BenchmarkCase("case", category, "scope", "question", 10, expected_fragments=fragments)


@pytest.mark.parametrize(
    ("category", "fragments", "answer"),
    [
        ("current_correction", ("16",), "It currently uses PostgreSQL 16."),
        ("historical_state", ("14", "16"), "It originally used 14 and now uses 16."),
        ("negation", ("not",), "No, it has not been deleted."),
        ("exact_values", ("5200", "2026-10-01"), "Port 5200; deadline 2026-10-01."),
        ("exact_values", ("5200", "2026-10-01"), "Port 5200; deadline October 1, 2026."),
        ("semantic_paraphrase", ("north workshop",), "It will launch at the north workshop."),
        ("subjective_preference", (), "Use compact Markdown, citations, and an uncertainty line."),
        ("subjective_preference", (), "Use Markdown with references and a line describing uncertainties."),
        ("verified_procedure", (), "Inspect the source, write it, reread it, and compare the hash."),
        ("unsupported_claim_gate", (), "The available evidence does not establish that claim."),
    ],
)
def test_correct_answers_pass(category: str, fragments: tuple[str, ...], answer: str) -> None:
    assert grade_answer(case(category, fragments), answer)["correct"] is True


@pytest.mark.parametrize(
    ("category", "fragments", "answer"),
    [
        ("current_correction", ("16",), "It uses PostgreSQL 14."),
        ("historical_state", ("14", "16"), "It uses 16."),
        ("negation", ("not",), "Yes, it was deleted."),
        ("exact_values", ("5200", "2026-10-01"), "The port is 5200."),
        ("subjective_preference", (), "Use Markdown."),
        ("verified_procedure", (), "Write the file."),
        ("unsupported_claim_gate", (), "Yes, Maple owns an interplanetary spaceport."),
    ],
)
def test_incomplete_or_unsupported_answers_fail(category: str, fragments: tuple[str, ...], answer: str) -> None:
    assert grade_answer(case(category, fragments), answer)["correct"] is False


def test_unknown_category_has_no_silent_grading_fallback() -> None:
    with pytest.raises(ValueError, match="no answer requirements"):
        answer_requirements(case("unknown"))


def test_correct_core_fact_with_invented_date_fails_grounded_grade() -> None:
    result = grade_answer(
        case("current_correction", ("16",)),
        "It uses PostgreSQL 16, effective on 2023-10-10.",
        context="The current record says PostgreSQL 16.",
    )
    assert result["concept_complete"] is True
    assert result["unsupported_numeric_claims"] == ["2023-10-10"]
    assert result["correct"] is False


def test_refusal_that_repeats_the_question_is_not_an_affirmative_claim() -> None:
    result = grade_answer(
        case("unsupported_claim_gate"),
        "The available evidence does not establish whether Project Maple owns an interplanetary spaceport.",
    )
    assert result["affirmative_unsupported_claim"] is False
    assert result["correct"] is True


def _tiny_seed(store):
    for capability in default_capabilities():
        store.register_capability(capability)
    current = MemoryRecord(
        id="alpha-current",
        kind=RecordKind.CORRECTION,
        content="Alpha now uses PostgreSQL 16.",
        created_at=10,
        scope="alpha",
        evidence_status=EvidenceStatus.USER_REPORTED,
    )
    unsupported = MemoryRecord(
        id="alpha-unsupported",
        kind=RecordKind.FACT,
        content="Alpha owns an interplanetary spaceport.",
        created_at=11,
        scope="alpha",
        evidence_status=EvidenceStatus.MODEL_INFERRED,
    )
    store.append_record(current)
    store.append_record(unsupported)
    return [
        BenchmarkCase(
            "alpha-current", "current_correction", "alpha",
            "Which PostgreSQL version does Alpha use now?", 20,
            gold_record_ids=(current.id,), expected_fragments=("16",),
        ),
        BenchmarkCase(
            "alpha-unsupported", "unsupported_claim_gate", "alpha",
            "Does Alpha own an interplanetary spaceport?", 20,
            prohibited_record_ids=(unsupported.id,),
        ),
    ]


def test_resumable_llm_benchmark_checkpoints_before_regrading(monkeypatch, tmp_path) -> None:
    import context_loom.llm_evaluation as module

    monkeypatch.setattr(module, "_seed_benchmark", _tiny_seed)
    calls = []

    def fake_answer(prompt, **_kwargs):
        calls.append(prompt)
        answer = (
            "It uses PostgreSQL 16."
            if "Which PostgreSQL version" in prompt
            else "The available evidence does not establish that claim."
        )
        return answer, {
            "latency_ms": 5.0, "done": True, "done_reason": "stop",
            "prompt_eval_count": 20, "eval_count": 5, "load_duration_ns": 0,
            "prompt_eval_duration_ns": 1, "eval_duration_ns": 1,
        }

    monkeypatch.setattr(module, "ollama_answer", fake_answer)
    output = tmp_path / "qa"
    first = run_llm_benchmark(output, model="fake", arms=("context_loom",))
    assert len(calls) == 2
    assert first["summary"]["context_loom"]["answer_accuracy"] == 1
    assert first["summary"]["context_loom"]["evidence_micro_recall"] == 1
    assert (output / "results.json").exists()
    checkpoints = sorted((output / "run_state" / "context_loom").glob("*.json"))
    assert len(checkpoints) == 2
    assert all(json.loads(path.read_text())["harness_version"] == HARNESS_VERSION for path in checkpoints)

    def forbidden_call(*_args, **_kwargs):
        raise AssertionError("resume should not regenerate completed answers")

    monkeypatch.setattr(module, "ollama_answer", forbidden_call)
    second = run_llm_benchmark(output, model="fake", arms=("context_loom",))
    assert second["summary"] == first["summary"]
    report = render_llm_report(second)
    assert "2`" in report and "100.0%" in report


def test_unknown_llm_benchmark_arm_fails_closed(tmp_path) -> None:
    with pytest.raises(ValueError, match="unknown benchmark arms"):
        run_llm_benchmark(tmp_path / "qa", model="fake", arms=("mystery",))


def test_ollama_answer_parses_metrics_without_live_server(monkeypatch) -> None:
    import context_loom.llm_evaluation as module

    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return json.dumps(
                {
                    "response": "  PostgreSQL 16.  ", "done": True, "done_reason": "stop",
                    "prompt_eval_count": 21, "eval_count": 4, "load_duration": 1,
                    "prompt_eval_duration": 2, "eval_duration": 3,
                }
            ).encode()

    def urlopen(request, timeout):
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr(module.urllib.request, "urlopen", urlopen)
    answer, metrics = ollama_answer("packet", model="fake", timeout=12, seed=9)
    assert answer == "PostgreSQL 16."
    assert metrics["prompt_eval_count"] == 21 and metrics["eval_count"] == 4
    assert captured["payload"]["options"]["temperature"] == 0
    assert captured["payload"]["options"]["seed"] == 9
    assert captured["timeout"] == 12
