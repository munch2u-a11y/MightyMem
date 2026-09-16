from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from context_loom import full_system_benchmark as benchmark


def _composition() -> dict:
    return {
        "entrypoint": "context_loom.habitus_agent.HabitusLoomAgent.handle",
        "memory_lanes": benchmark.EXPECTED_LANES,
        "context_loom_max_input_tokens": benchmark.LOOM_MAX_INPUT_TOKENS,
        "context_loom_renders_capability_schemas": False,
        "context_loom_response_instruction": "",
        "model_response_instruction": benchmark.DIRECT_ANSWER_RESPONSE_INSTRUCTION,
        "response_surface_schema": benchmark.DIRECT_ANSWER_SURFACE_SCHEMA,
        "response_surface_protocol": benchmark.DIRECT_ANSWER_SURFACE_PROTOCOL,
        "chat_model": "chat@digest",
        "embedding_model": "embedding@digest",
        "test_model_adapter": False,
        "private_thinking_enabled": True,
        "private_thinking_policy": "adaptive",
        "chat_temperature": 0.25,
        "action_budget": None,
        "history_turns": None,
        "checkpoint_retention": None,
    }


class _Models:
    chat_identity = "chat@digest"
    space_id = "embedding@digest"
    temperature = 0.25


class _FakeAgent:
    def __init__(self, state: Path, question_states: list[str]) -> None:
        self.state = state
        self.question_states = question_states
        self.last_model_inputs: list[str] = []
        self._loom_round_packets: list[dict] = []

    def composition(self) -> dict:
        return _composition()

    def handle(self, question: str) -> dict:
        before = (self.state / "seed.txt").read_text(encoding="utf-8")
        self.question_states.append(before)
        (self.state / "seed.txt").write_text(before + "|" + question, encoding="utf-8")
        (self.state / "events.jsonl").write_text(question + "\n", encoding="utf-8")
        self.last_model_inputs = [f"memory={before}\nquestion={question}"]
        return {
            "response": before,
            "turn_id": "turn-" + question,
            "speech_cycle_id": "speech-" + question,
            "model_rounds": [{
                "usage": {"prompt_eval_count": 9, "eval_count": 1},
                "thinking_chars": 2,
                "tool_call_count": 0,
            }],
            "full_system_receipt": {
                "entrypoint": "context_loom.habitus_agent.HabitusLoomAgent.handle",
                "lanes": benchmark.EXPECTED_LANES,
                "chat_model": "chat@digest",
                "embedding_model": "embedding@digest",
                "test_model_adapter": False,
                "private_thinking_enabled": True,
                "private_thinking_policy": "adaptive",
                "chat_temperature": 0.25,
                "private_thinking_rounds": [False],
                "private_thinking_reasons": ["direct_reply"],
                "response_surface_schema": benchmark.DIRECT_ANSWER_SURFACE_SCHEMA,
                "response_surface_protocol": benchmark.DIRECT_ANSWER_SURFACE_PROTOCOL,
                "response_surface_rounds": [{
                    "schema": benchmark.DIRECT_ANSWER_SURFACE_SCHEMA,
                    "task_stage": "understand",
                    "protocol": benchmark.DIRECT_ANSWER_SURFACE_PROTOCOL,
                    "applied": False,
                    "raw_content": before,
                    "raw_content_sha256": hashlib.sha256(before.encode()).hexdigest(),
                    "surfaced_content": before,
                    "surfaced_content_sha256": hashlib.sha256(before.encode()).hexdigest(),
                }],
                "generation_token_limit": None,
                "model_request_timeout": None,
                "action_budget": None,
                "context_rounds": [{
                    "admitted_record_ids": [],
                    "packet_max_input_tokens": benchmark.LOOM_MAX_INPUT_TOKENS,
                    "packet_tokens": 10,
                    "private_thinking_enabled_for_round": False,
                    "private_thinking_reason_for_round": "direct_reply",
                }],
                "model_input_sha256": [
                    hashlib.sha256(self.last_model_inputs[0].encode()).hexdigest()
                ],
            },
        }

    def close(self) -> None:
        pass


def test_parser_has_no_case_cap_and_requires_server_attestation() -> None:
    parser = benchmark.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--output", "/tmp/example"])
    options = {option for action in parser._actions for option in action.option_strings}
    assert "--max-questions" not in options
    assert "--sample-size" not in options
    assert "--timeout" not in options
    args = parser.parse_args(["--output", "/tmp/example", "--ollama-server-pid", "123"])
    assert args.conversation_indices is None
    assert args.source_checkpoint is None
    assert args.sensory_assets is None
    assert args.temperature == 0.25
    assert args.case_state_retention == "audit"


def test_source_inventory_covers_complete_habitus_source_tree() -> None:
    inventory = benchmark.source_inventory()
    expected = {
        str(path.resolve())
        for path in (benchmark.HABITUS_ROOT / "habitus_ai").rglob("*.py")
    }
    assert expected
    assert expected <= set(inventory)
    assert str(Path(benchmark.__file__).resolve()) in inventory


def test_each_question_starts_from_an_unchanged_pristine_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "run"
    conversation = output / "conversations" / "sample"
    pristine = conversation / "pristine-state"
    pristine.mkdir(parents=True)
    (pristine / "seed.txt").write_text("pristine", encoding="utf-8")
    observed_states: list[str] = []

    def fake_build_agent(state, workspace, models, **_kwargs):
        workspace.mkdir(parents=True, exist_ok=True)
        return _FakeAgent(Path(state), observed_states)

    grades: list[tuple[str, str]] = []

    def evaluator(rows, **_kwargs):
        grades.append((rows[0]["prediction"], rows[0]["answer"]))
        return [1.0], 0.0, [1.0]

    monkeypatch.setattr(benchmark, "build_agent", fake_build_agent)
    baseline = {"scope": "locomo:sample", "state_inventory_sha256": "baseline-sha"}
    sample = {"sample_id": "sample"}
    first = {"question": "first", "answer": "pristine", "category": 2, "evidence": []}
    second = {"question": "second", "answer": "pristine", "category": 2, "evidence": []}

    row_one = benchmark.run_case(
        output, conversation, baseline, sample, 0, first, _Models(), evaluator
    )
    row_two = benchmark.run_case(
        output, conversation, baseline, sample, 1, second, _Models(), evaluator
    )

    assert observed_states == ["pristine", "pristine"]
    assert (pristine / "seed.txt").read_text(encoding="utf-8") == "pristine"
    assert grades == [("pristine", "pristine"), ("pristine", "pristine")]
    assert row_one["official_grade_recorded_before_progress"] is True
    assert row_two["official_grade_recorded_before_progress"] is True
    benchmark.validate_case_artifacts(output, row_one)
    benchmark.validate_case_artifacts(output, row_two)
    first_case = output / row_one["case_artifact_directory"]
    assert not (first_case / "state/seed.txt").exists()
    assert (first_case / "state/events.jsonl").read_text(encoding="utf-8") == "first\n"
    assert row_one["state_retention"]["mode"] == "audit"
    assert row_one["state_retention"][
        "applied_after_model_completion_and_official_grading"
    ] is True

    first_input = output / row_one["case_artifact_directory"] / "model-inputs/round-001.txt"
    first_input.write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="completed case artifact changed"):
        benchmark.validate_case_artifacts(output, row_one)

    first_input.write_text("memory=pristine\nquestion=first", encoding="utf-8")
    (first_case / "state/events.jsonl").write_text("changed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="retained state changed"):
        benchmark.validate_case_artifacts(output, row_one)


def test_official_f1_rejects_non_scalar_case_result() -> None:
    item = {"answer": "x", "category": 2, "evidence": []}
    with pytest.raises(RuntimeError, match="exactly one score"):
        benchmark.official_f1(lambda *_args, **_kwargs: ([], 0, []), item, "x")


def test_turn_receipt_rejects_tampered_response_surface_hash(tmp_path: Path) -> None:
    state = tmp_path / "state"
    state.mkdir()
    (state / "seed.txt").write_text("answer", encoding="utf-8")
    agent = _FakeAgent(state, [])
    turn = agent.handle("question")
    benchmark.validate_turn_receipt(turn, agent.last_model_inputs, _Models())

    turn["full_system_receipt"]["response_surface_rounds"][0][
        "surfaced_content_sha256"
    ] = "0" * 64
    with pytest.raises(RuntimeError, match="response_surface_rounds"):
        benchmark.validate_turn_receipt(turn, agent.last_model_inputs, _Models())
