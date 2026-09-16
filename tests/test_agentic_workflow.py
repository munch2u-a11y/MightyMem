from __future__ import annotations

import pytest

from context_loom import ContextLoom

from conftest import admitted_ids


def test_capability_visibility_changes_with_task_phase(loom) -> None:
    loom.begin_task(
        "task", "Read the notes, then create a summary.", scope="alpha", stage="inspect",
        focus_stack=["read source", "create summary"], mandatory_capability_ids=["read-file"], now=10,
    )
    inspect = loom.compiler.compile("Inspect the notes.", scope="alpha", task_id="task", now=20)
    inspect_tools = {tool.id for tool in inspect.capabilities}
    loom.update_task("task", stage="act", mandatory_capability_ids=["write-file"], now=21)
    act = loom.compiler.compile("Create the summary.", scope="alpha", task_id="task", now=22)
    act_tools = {tool.id for tool in act.capabilities}
    assert "read-file" in inspect_tools and "write-file" not in inspect_tools
    assert "write-file" in act_tools and "read-file" not in act_tools
    assert len(inspect_tools) <= 3 and len(act_tools) <= 3


def test_verified_tool_observation_advances_state_and_is_mandatory(loom) -> None:
    loom.begin_task("task", "Inspect then write.", scope="alpha", stage="inspect", now=10)
    observation = loom.record_tool_result(
        "Verified bytes and sha256=abc.", task_id="task", scope="alpha", verified=True,
        receipt_id="read-1", verifier="content_and_sha256", next_stage="act", now=20,
    )
    packet = loom.compiler.compile("Continue.", scope="alpha", task_id="task", now=21)
    assert loom.store.get_task("task").stage == "act"
    assert loom.store.get_task("task").last_observation_id == observation.id
    assert observation.id in admitted_ids(packet)


def test_unverified_tool_result_cannot_advance_task(loom) -> None:
    loom.begin_task("task", "Inspect then write.", scope="alpha", stage="inspect", now=10)
    result = loom.record_tool_result(
        "Tool claimed success without a receipt.", task_id="task", scope="alpha", verified=False,
        receipt_id="missing", verifier="none", next_stage="act", now=20,
    )
    packet = loom.compiler.compile("Did it succeed?", scope="alpha", task_id="task", now=21)
    assert loom.store.get_task("task").stage == "inspect"
    assert result.id not in admitted_ids(packet)


def test_positive_learning_requires_verified_outcome(loom) -> None:
    memory = loom.remember_preference(
        "I prefer JSON reports.", scope="alpha", record_id="preference", created_at=10, importance=1,
    )
    packet = loom.compiler.compile("How should you format my report?", scope="alpha", now=20)
    assert memory.id in admitted_ids(packet)
    with pytest.raises(ValueError, match="verified outcome"):
        loom.record_outcome(
            packet.id, verified=False, task_progress=0.5, helpful_record_ids=(memory.id,), now=21
        )
    assert loom.store.learned_utility(memory.id) == 0


def test_feedback_can_credit_only_records_that_were_used(loom) -> None:
    used = loom.remember_preference(
        "I prefer JSON reports.", scope="alpha", record_id="used", created_at=10, importance=1,
    )
    unused = loom.remember_user_statement(
        "Unrelated snow report.", scope="beta", record_id="unused", created_at=1,
    )
    packet = loom.compiler.compile("How should you format my report?", scope="alpha", now=20)
    assert used.id in admitted_ids(packet) and unused.id not in admitted_ids(packet)
    with pytest.raises(ValueError, match="only records admitted"):
        loom.record_outcome(
            packet.id, verified=True, task_progress=0.5, helpful_record_ids=(unused.id,), now=21
        )


def test_verified_candidate_specific_feedback_updates_utility(loom) -> None:
    helpful = loom.remember_preference(
        "I prefer JSON reports.", scope="alpha", record_id="helpful", created_at=10, importance=1,
    )
    packet = loom.compiler.compile("How should you format my report?", scope="alpha", now=20)
    loom.record_outcome(
        packet.id, verified=True, task_progress=0.75, helpful_record_ids=(helpful.id,), now=21,
        feedback_id="feedback-1",
    )
    assert loom.store.learned_utility(helpful.id) == pytest.approx(0.75)
    # Exact replay is idempotent and cannot double-count credit.
    loom.record_outcome(
        packet.id, verified=True, task_progress=0.75, helpful_record_ids=(helpful.id,), now=21,
        feedback_id="feedback-1",
    )
    assert loom.store.learned_utility(helpful.id) == pytest.approx(0.75)


def test_harmful_feedback_reduces_only_named_candidate(loom) -> None:
    first = loom.remember_preference(
        "I prefer JSON reports.", scope="alpha", record_id="first", created_at=10, importance=1,
    )
    second = loom.remember_preference(
        "For audits I prefer Markdown.", scope="alpha", record_id="second", created_at=11, importance=1,
    )
    packet = loom.compiler.compile("How should you format my audit report?", scope="alpha", now=20)
    assert {first.id, second.id} <= admitted_ids(packet)
    loom.record_outcome(
        packet.id, verified=False, task_progress=-0.5, harmful_record_ids=(first.id,), now=21,
    )
    assert loom.store.learned_utility(first.id) < 0
    assert loom.store.learned_utility(second.id) == 0


def test_task_scope_mismatch_fails_closed(loom) -> None:
    loom.begin_task("task", "Work in alpha.", scope="alpha", now=10)
    with pytest.raises(ValueError, match="scope"):
        loom.compiler.compile("Continue", scope="beta", task_id="task", now=20)


def test_unknown_task_and_response_packet_are_rejected(loom) -> None:
    with pytest.raises(ValueError, match="unknown task"):
        loom.compiler.compile("Continue", scope="alpha", task_id="missing", now=20)
    with pytest.raises(ValueError, match="source context packet"):
        loom.record_response("Done", scope="alpha", source_packet_id="missing", now=20)


def test_empty_query_is_rejected(loom) -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        loom.compiler.compile("   ", scope="alpha", now=20)
