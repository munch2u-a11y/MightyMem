from __future__ import annotations

from pathlib import Path
import pytest

from context_loom.gestation import GestationConfig, gestate_loom_agent
from context_loom.habitus_agent import (
    HabitusLoomAgent,
    MODEL_POLICY,
    explicit_preference_key,
    is_explicit_user_preference,
    resolve_relative_dates,
)
from context_loom.models import EvidenceStatus, RecordKind
from context_loom.retrieval import render_record
from habitus_ai.types import InputTrunk, RecordType


def test_gestate_loom_agent_creates_gestated_mind(tmp_path: Path) -> None:
    directory = tmp_path / "agent_dir"
    workspace = tmp_path / "workspace"
    config = GestationConfig(
        human_name="Alex",
        agent_name="LoomMind",
        taste_schema="builder",
        model_name="granite4.2:8b",
        scope="test_scope",
        allow_test_models=True,
    )
    result = gestate_loom_agent(
        directory=directory,
        workspace=workspace,
        config=config,
    )
    assert result["profile"]["agent_name"] == "LoomMind"
    assert result["profile"]["taste_schema"] == "builder"
    assert (directory / "loom.sqlite").exists()
    assert (directory / "mind.sqlite").exists()
    assert (directory / "LATEST.json").exists()


def test_uncapped_token_and_timeout_config(tmp_path: Path) -> None:
    directory = tmp_path / "agent_dir"
    workspace = tmp_path / "workspace"
    config = GestationConfig(
        human_name="Alex",
        agent_name="LoomMind",
        model_name="granite4.2:8b",
        timeout=None,
        maximum_tokens=None,
        temperature=0.15,
        allow_test_models=True,
    )
    result = gestate_loom_agent(
        directory=directory,
        workspace=workspace,
        config=config,
    )
    assert result["models"]["timeout"] is None
    assert result["models"]["maximum_tokens"] is None
    assert result["models"]["temperature"] == 0.15
    assert result["composition"]["chat_temperature"] == 0.15
    assert result["composition"]["private_thinking_policy"] == "adaptive"


def test_habitus_loom_agent_autogestates_when_profile_missing(tmp_path: Path) -> None:
    directory = tmp_path / "agent_dir_auto"
    workspace = tmp_path / "workspace_auto"
    workspace.mkdir(parents=True, exist_ok=True)
    # Instantiate without a pre-existing checkpoint
    agent = HabitusLoomAgent(
        directory=directory,
        workspace=workspace,
        models=None,  # mock or fallback
        scope="auto_scope",
        allow_test_models=True,
    )
    assert agent.loom is not None
    assert (directory / "loom.sqlite").exists()
    agent.close()


def test_production_agent_rejects_an_implicit_test_model(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with pytest.raises(ValueError, match="explicit installed model adapter"):
        HabitusLoomAgent(
            directory=tmp_path / "agent",
            workspace=workspace,
            models=None,
        )


def test_agent_rejects_unknown_private_thinking_policy(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="private thinking policy"):
        HabitusLoomAgent(
            directory=tmp_path / "agent",
            workspace=tmp_path / "workspace",
            models=None,
            allow_test_models=True,
            private_thinking_policy="sometimes",
        )


def test_direct_answer_surface_is_explicit_and_raw_response_remains_auditable(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    agent = HabitusLoomAgent(
        directory=tmp_path / "agent",
        workspace=workspace,
        models=None,
        allow_test_models=True,
    )
    responses = [
        "ANSWER: 19 January, 2023\nThe source statement was dated one day later.",
        "The source statement directly names the subject.\nANSWER: adoption agencies",
        "ANSWER: 18 January, 2023\nANSWER: 19 January, 2023",
        "The evidence conflicts, so I cannot give one date.",
    ]

    def scripted_complete(_messages, _tools):
        content = responses.pop(0)
        agent.models.last_usage = {"done_reason": "stop"}
        agent.models.last_response = {
            "done": True,
            "model": "scripted-test-model",
            "message": {"role": "assistant", "content": content},
        }
        return {"role": "assistant", "content": content}

    agent.models.complete = scripted_complete
    try:
        direct = agent.handle("When did Jon lose his job as a banker?")
        assert direct["response"] == "19 January, 2023"
        direct_surface = direct["full_system_receipt"]["response_surface_rounds"]
        assert len(direct_surface) == 1
        assert direct_surface[0]["applied"] is True
        assert direct_surface[0]["answer_field_position"] == "leading"
        assert direct_surface[0]["raw_content"].endswith("dated one day later.")
        assert direct_surface[0]["surfaced_content"] == direct["response"]

        terminal = agent.handle("What did Caroline research?")
        assert terminal["response"] == "adoption agencies"
        terminal_surface = terminal["full_system_receipt"]["response_surface_rounds"]
        assert terminal_surface[0]["applied"] is True
        assert terminal_surface[0]["answer_field_position"] == "terminal"
        assert terminal_surface[0]["raw_content"].startswith("The source statement")

        multiple = agent.handle("When did the two candidate events happen?")
        assert multiple["response"].count("ANSWER:") == 2
        multiple_surface = multiple["full_system_receipt"]["response_surface_rounds"]
        assert multiple_surface[0]["applied"] is False
        assert multiple_surface[0]["answer_field_position"] is None

        conflicted = agent.handle("When did the conflicting event happen?")
        assert conflicted["response"] == "The evidence conflicts, so I cannot give one date."
        conflict_surface = conflicted["full_system_receipt"]["response_surface_rounds"]
        assert len(conflict_surface) == 1
        assert conflict_surface[0]["applied"] is False
        assert conflict_surface[0]["raw_content"] == conflicted["response"]
    finally:
        agent.close()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("I prefer concise answers with evidence.", True),
        ("Please remember this working preference: cite exact files.", True),
        ("My favorite report format is Markdown.", True),
        ("I no longer prefer long introductions.", True),
        ("What report format do I prefer?", False),
        ("Please create a concise report.", False),
    ],
)
def test_explicit_preference_classifier_is_conservative(text: str, expected: bool) -> None:
    assert is_explicit_user_preference(text) is expected


def test_only_explicit_preference_domains_become_supersession_keys() -> None:
    assert explicit_preference_key("My response-style preference is concise.") == "preference:response style"
    assert explicit_preference_key("My favorite report format is Markdown.") == "favorite:report format"
    assert explicit_preference_key("Remember this working preference: concise answers.") == ""
    assert explicit_preference_key("I prefer Markdown.") == ""


def test_transcript_relative_date_is_derived_without_changing_raw_text(tmp_path: Path) -> None:
    created_at = 1_674_172_800.0  # 2023-01-20 UTC
    assert resolve_relative_dates("I lost it yesterday.", created_at) == {
        "yesterday": "2023-01-19"
    }
    assert resolve_relative_dates("I painted it last year.", created_at) == {
        "last year": "2022"
    }
    (tmp_path / "workspace").mkdir()
    agent = HabitusLoomAgent(
        directory=tmp_path / "agent",
        workspace=tmp_path / "workspace",
        models=None,
        allow_test_models=True,
        scope="relative-date",
    )
    try:
        agent.import_transcript(
            [{
                "source_id": "D1:2",
                "session_id": "session_1",
                "timestamp": "4:04 pm on 20 January, 2023",
                "created_at": created_at,
                "speaker": "Jon",
                "text": "I lost my job as a banker yesterday.",
            }],
            corpus_id="temporal-test",
        )
        record = next(
            item for item in agent.loom.store.list_records(scope="relative-date")
            if item.metadata.get("source_record_id") == "D1:2"
        )
        assert record.content.endswith("Jon: I lost my job as a banker yesterday.")
        assert record.metadata["resolved_relative_dates"] == {"yesterday": "2023-01-19"}
        assert "system-derived relative dates: yesterday=2023-01-19" in render_record(record)

        thinking_at_completion = []

        def scripted_complete(_messages, _tools):
            thinking_at_completion.append(agent.models.think)
            agent.models.last_usage = {"done_reason": "stop"}
            agent.models.last_response = {
                "done": True,
                "model": "scripted-test-model",
            }
            return {"role": "assistant", "content": "ANSWER: 19 January, 2023"}

        agent.models.complete = scripted_complete
        result = agent.handle("When did Jon lose his job as a banker?")
        assert result["response"] == "19 January, 2023"
        assert thinking_at_completion == [True]
        assert result["full_system_receipt"]["private_thinking_rounds"] == [True]
        assert result["full_system_receipt"]["private_thinking_reasons"] == [
            "temporal_resolution"
        ]
    finally:
        agent.close()


def test_inferential_question_enables_private_thinking(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    agent = HabitusLoomAgent(
        directory=tmp_path / "agent",
        workspace=workspace,
        models=None,
        allow_test_models=True,
    )
    thinking_at_completion = []

    def scripted_complete(_messages, _tools):
        thinking_at_completion.append(agent.models.think)
        agent.models.last_usage = {"done_reason": "stop"}
        agent.models.last_response = {
            "done": True,
            "model": "scripted-test-model",
        }
        return {"role": "assistant", "content": "ANSWER: counseling"}

    agent.models.complete = scripted_complete
    try:
        result = agent.handle("What field would Caroline likely pursue?")
        assert result["response"] == "counseling"
        assert thinking_at_completion == [True]
        assert result["full_system_receipt"]["private_thinking_rounds"] == [True]
        assert result["full_system_receipt"]["private_thinking_reasons"] == [
            "inference_resolution"
        ]
    finally:
        agent.close()


def test_live_preference_and_gestation_taste_are_grounded_behavior_state(tmp_path: Path) -> None:
    directory = tmp_path / "behavior_agent"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    agent = HabitusLoomAgent(
        directory=directory,
        workspace=workspace,
        models=None,
        scope="behavior",
        human_name="Alex",
        agent_name="LoomMind",
        taste_schema="builder",
        allow_test_models=True,
    )
    try:
        profile = agent.behavioral_profile()
        assert "learned_pattern_record_ids" not in profile
        prior = agent.loom.store.get_record(profile["starting_taste_record_id"])
        assert prior is not None
        assert prior.kind == RecordKind.PREFERENCE
        assert prior.evidence_status == EvidenceStatus.SYSTEM_DERIVED
        assert prior.metadata["origin"] == "gestation_config"

        agent._begin_turn_context(
            "Please remember this working preference: keep answers concise and cite evidence."
        )
        preference_id = agent._loom_current_event_id
        preference = agent.loom.store.get_record(preference_id)
        assert preference is not None
        assert preference.kind == RecordKind.PREFERENCE
        assert preference.metadata["preference_owner"] == "human"

        agent._begin_turn_context("What response style do I prefer?")
        question = agent.loom.store.get_record(agent._loom_current_event_id)
        assert question is not None and question.kind == RecordKind.USER
        packet = agent.loom.compiler.compile(
            "What response style do I prefer?",
            scope="behavior",
            task_id=agent._loom_task_id,
            current_event_id=agent._loom_current_event_id,
            now=agent._loom_turn_now,
        )
        admitted = {
            source_id
            for entry in packet.manifest
            if entry.admitted
            for source_id in entry.source_record_ids
        }
        assert preference_id in admitted
        assert any("User preference" in line for line in packet.sections["personal"])

        agent._begin_turn_context("My response style preference is concise.")
        old_style_id = agent._loom_current_event_id
        agent._begin_turn_context("My response style preference is detailed.")
        new_style = agent.loom.store.get_record(agent._loom_current_event_id)
        assert new_style is not None
        assert new_style.supersedes_id == old_style_id
        assert new_style.metadata["preference_key"] == "preference:response style"
    finally:
        agent.close()


def test_verified_workspace_receipts_advance_multistep_task_and_accept_json_null_spelling(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    agent = HabitusLoomAgent(
        directory=tmp_path / "task_agent",
        workspace=workspace,
        models=None,
        scope="tasks",
        allow_test_models=True,
    )
    try:
        agent._begin_turn_context(
            "Create result.txt. Inspect the workspace first, write exactly ready, "
            "then read it back and verify it."
        )
        task = agent.loom.store.get_task(agent._loom_task_id)
        assert task is not None
        assert task.stage == "inspect"
        assert task.mandatory_capability_ids == ["list-files"]

        inspected = agent._tool_call("list_files", {"path": "."})
        assert inspected.status == "success" and inspected.verified
        task = agent.loom.store.get_task(agent._loom_task_id)
        assert task is not None
        assert task.stage == "act"
        assert task.mandatory_capability_ids == ["write-file"]

        written = agent._tool_call(
            "write_file",
            {"path": "result.txt", "content": "ready", "expected_sha256": "null"},
        )
        assert written.status == "success" and written.verified
        assert (workspace / "result.txt").read_text() == "ready"
        task = agent.loom.store.get_task(agent._loom_task_id)
        assert task is not None
        assert task.stage == "verify"
        assert task.mandatory_capability_ids == ["read-file"]

        reread = agent._tool_call("read_file", {"path": "result.txt"})
        assert reread.status == "success" and reread.verified
        assert reread.output["content"] == "ready"
        task = agent.loom.store.get_task(agent._loom_task_id)
        assert task is not None
        assert task.stage == "report"
        assert task.mandatory_capability_ids == ["report-result"]
    finally:
        agent.close()


def test_promoted_habitus_pattern_becomes_inspectable_loom_evidence(tmp_path: Path) -> None:
    directory = tmp_path / "pattern_agent"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    agent = HabitusLoomAgent(
        directory=directory,
        workspace=workspace,
        models=None,
        scope="patterns",
        allow_test_models=True,
    )
    try:
        for index in range(2):
            record = agent.mind.remember(
                "Cobalt garden synchronization routine repeats.",
                record_id=f"pattern-source-{index}",
                event_id=f"pattern-event-{index}",
                metadata={
                    "preference_signals": [0.8],
                    "preference_confidence": 1.0,
                },
                allow_growth=False,
            )
            agent.mind.graph.grow_trunk_breadth(
                record,
                input_trunk=InputTrunk.HEAR,
                pulse=agent.mind.pulse,
            )
        pattern_ids = agent._sync_habitus_patterns()
        assert pattern_ids
        records = [agent.loom.store.get_record(record_id) for record_id in pattern_ids]
        assert all(record is not None for record in records)
        pattern = records[-1]
        assert pattern is not None
        assert pattern.kind == RecordKind.PREFERENCE
        assert pattern.evidence_status == EvidenceStatus.SYSTEM_DERIVED
        assert pattern.metadata["origin"] == "habitus_promoted_pattern"
        assert len(pattern.metadata["habitus_experience_ids"]) >= 2
        assert {"cobalt", "garden", "synchronization"} <= set(
            pattern.metadata["shared_terms"]
        )
        receipt_profile = agent.behavioral_profile(include_record_ids=True)
        assert receipt_profile["learned_pattern_record_ids"] == sorted(pattern_ids)

        agent._begin_turn_context("How should I handle cobalt garden synchronization?")
        packet = agent.loom.compiler.compile(
            "How should I handle cobalt garden synchronization?",
            scope="patterns",
            task_id=agent._loom_task_id,
            current_event_id=agent._loom_current_event_id,
            now=agent._loom_turn_now,
        )
        assert any(
            "Learned Habitus behavioral pattern" in line
            for line in packet.sections.get("personal", [])
        )
    finally:
        agent.close()


def test_tool_return_repetition_cannot_become_a_personality_pattern(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    agent = HabitusLoomAgent(
        directory=tmp_path / "tool_pattern_agent",
        workspace=workspace,
        models=None,
        scope="tool-patterns",
        allow_test_models=True,
    )
    try:
        for index in range(2):
            record = agent.mind.remember(
                '{"ability":"ability:memory-recall","record_id":"noise","cosine":0.7}',
                source_id="ability:memory-recall",
                record_type=RecordType.TOOL_RESULT,
                record_id=f"tool-pattern-source-{index}",
                event_id=f"tool-pattern-event-{index}",
                input_trunk=InputTrunk.SEE,
                metadata={
                    "preference_signals": [0.8],
                    "preference_confidence": 1.0,
                    "verified": True,
                },
                allow_growth=False,
            )
            agent.mind.graph.grow_trunk_breadth(
                record,
                input_trunk=InputTrunk.SEE,
                pulse=agent.mind.pulse,
            )
        assert agent.mind.store.list_concepts(kind="experiential_pattern")
        assert agent._sync_habitus_patterns() == []
        assert not any(
            record.metadata.get("origin") == "habitus_promoted_pattern"
            for record in agent.loom.store.list_records(scope="tool-patterns")
        )
    finally:
        agent.close()


def test_legacy_nonbehavioral_pattern_is_superseded_by_retraction(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    agent = HabitusLoomAgent(
        directory=tmp_path / "legacy_pattern_agent",
        workspace=workspace,
        models=None,
        scope="legacy-patterns",
        allow_test_models=True,
    )
    try:
        legacy = agent.loom.record(
            RecordKind.PREFERENCE,
            "Habitus learned record_id and cosine as a behavior.",
            scope="legacy-patterns",
            record_id="legacy-tool-pattern",
            evidence_status=EvidenceStatus.SYSTEM_DERIVED,
            metadata={
                "origin": "habitus_promoted_pattern",
                "habitus_cluster_id": "legacy-tool-cluster",
                "preference_owner": "agent",
            },
        )
        agent._habitus_pattern_versions = None
        agent._habitus_patterns_synced_pulse = -1
        agent._sync_habitus_patterns()
        replacement = agent.loom.store.latest_version(legacy.id)
        assert replacement.id != legacy.id
        assert replacement.supersedes_id == legacy.id
        assert replacement.metadata["origin"] == "habitus_pattern_retraction"
        assert replacement.metadata["applicability"] == "retracted"
    finally:
        agent.close()


def test_workspace_observation_embeddings_are_batched_after_actions(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    agent = HabitusLoomAgent(
        directory=tmp_path / "batched_observation_agent",
        workspace=workspace,
        models=None,
        scope="batched-observations",
        allow_test_models=True,
    )
    calls: list[list[str]] = []
    original_embed = agent.models.embed

    def counted_embed(texts):
        calls.append(list(texts))
        return original_embed(texts)

    agent.models.embed = counted_embed
    try:
        agent._begin_turn_context(
            "Inspect first, create result.txt containing ready, then read it back."
        )
        agent._tool_call("list_files", {"path": "."})
        agent._tool_call(
            "write_file",
            {"path": "result.txt", "content": "ready", "expected_sha256": None},
        )
        agent._tool_call("read_file", {"path": "result.txt"})
        assert calls == []
        assert len(agent._pending_workspace_observations) == 3

        receipt = agent._flush_workspace_observation_embeddings()
        assert receipt["status"] == "indexed"
        assert receipt["observation_count"] == 3
        assert receipt["embedding_request_count"] == 1
        assert len(calls) == 1
        assert agent._pending_workspace_observations == []
        assert agent.mind.store.connection.execute(
            "SELECT COUNT(*) FROM agent_semantic_index WHERE vault='observations'"
        ).fetchone()[0] == 3
        for record_id in receipt["loom_record_ids"]:
            assert agent.loom.store.get_embedding(
                record_id, agent.loom_provider.identity + ":clean-payload-v2"
            ) is not None
            assert agent.loom.store.get_embedding(
                record_id, agent.loom_provider.identity + ":scope-blind-v1"
            ) is not None
    finally:
        agent.close()


def test_report_capability_request_does_not_regress_to_memory_search(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    agent = HabitusLoomAgent(
        directory=tmp_path / "report_stage_agent",
        workspace=workspace,
        models=None,
        scope="report-stage",
        allow_test_models=True,
    )
    try:
        assert agent._stage_and_capability("report_result") == ("report", "report-result")
        assert agent._stage_and_capability("final response") == ("report", "report-result")
    finally:
        agent.close()


def test_interrupted_deferred_observation_index_is_recovered(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    directory = tmp_path / "recovery_agent"
    first = HabitusLoomAgent(
        directory=directory,
        workspace=workspace,
        models=None,
        scope="recovery",
        allow_test_models=True,
    )
    models = first.models
    calls: list[list[str]] = []
    original_embed = models.embed

    def counted_embed(texts):
        calls.append(list(texts))
        return original_embed(texts)

    models.embed = counted_embed
    first._begin_turn_context("Inspect the workspace.")
    action = first._tool_call("list_files", {"path": "."})
    native_id = action.return_record_id
    loom_id = first._pending_workspace_observations[0]["loom_record"].id
    assert calls == []
    first.close()

    calls.clear()
    recovered = HabitusLoomAgent(
        directory=directory,
        workspace=workspace,
        models=models,
        scope="recovery",
        recover_working=True,
        allow_test_models=True,
    )
    try:
        receipt = recovered._recovered_workspace_observation_index
        assert receipt["status"] == "indexed"
        assert receipt["recovered_return_record_ids"] == [native_id]
        assert receipt["embedding_request_count"] == 1
        assert len(calls) == 1
        assert recovered.mind.store.connection.execute(
            "SELECT vault FROM agent_semantic_index WHERE record_id=?", (native_id,)
        ).fetchone()[0] == "observations"
        for provider in ("clean-payload-v2", "scope-blind-v1"):
            assert recovered.loom.store.get_embedding(
                loom_id, recovered.loom_provider.identity + ":" + provider
            ) is not None
    finally:
        recovered.close()


def test_handle_keeps_embedding_work_out_of_multistep_action_loop(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    agent = HabitusLoomAgent(
        directory=tmp_path / "full_loop_agent",
        workspace=workspace,
        models=None,
        scope="full-loop",
        allow_test_models=True,
    )
    embed_calls: list[list[str]] = []
    original_embed = agent.models.embed

    def counted_embed(texts):
        embed_calls.append(list(texts))
        return original_embed(texts)

    agent.models.embed = counted_embed
    embedding_counts_at_chat: list[int] = []
    visible_tools_at_chat: list[list[str]] = []
    private_thinking_at_chat: list[bool] = []
    responses = [
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "list-call",
            "function": {"name": "list_files", "arguments": {"path": "."}},
        }]},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "write-call",
            "function": {"name": "write_file", "arguments": {
                "path": "result.txt", "content": "ready", "expected_sha256": None,
            }},
        }]},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "read-call",
            "function": {"name": "read_file", "arguments": {"path": "result.txt"}},
        }]},
        {"role": "assistant", "content": "Verified result.txt contains ready."},
    ]

    def scripted_complete(messages, tools):
        embedding_counts_at_chat.append(len(embed_calls))
        visible_tools_at_chat.append([item["function"]["name"] for item in tools])
        private_thinking_at_chat.append(agent.models.think)
        agent.models.last_usage = {"done_reason": "stop"}
        agent.models.last_response = {
            "done": True, "model": "scripted-test-model", "done_reason": "stop"
        }
        return responses.pop(0)

    agent.models.complete = scripted_complete
    agent.models.last_usage = {}
    agent.models.last_response = {}
    try:
        result = agent.handle(
            "Inspect the workspace first, write result.txt containing exactly ready, "
            "then read it back and verify it."
        )
        assert result["response"] == "Verified result.txt contains ready."
        assert (workspace / "result.txt").read_text() == "ready"
        assert [item["tool_id"] for item in result["receipts"]] == [
            "ability:memory-recall",
            "ability:workspace-read",
            "ability:workspace-write",
            "ability:workspace-read",
        ]
        # The first context build may issue several embeddings. Every later
        # model round must reuse them while the chat model is active.
        assert len(embedding_counts_at_chat) == 4
        assert len(set(embedding_counts_at_chat)) == 1
        assert visible_tools_at_chat[-1] == []
        assert private_thinking_at_chat == [True, True, True, False]
        assert result["full_system_receipt"]["private_thinking_rounds"] == [
            True, True, True, False
        ]
        deferred = result["deferred_observation_index"]
        assert deferred["observation_count"] == 3
        assert deferred["embedding_request_count"] == 1
        assert not agent._pending_workspace_observations
    finally:
        agent.close()


def test_repeated_live_human_preference_becomes_admitted_habitus_pattern(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    agent = HabitusLoomAgent(
        directory=tmp_path / "preference_growth_agent",
        workspace=workspace,
        models=None,
        scope="preference-growth",
        human_name="LocalUser",
        allow_test_models=True,
    )

    message_batches = []
    private_thinking_at_chat = []

    def scripted_complete(messages, tools):
        message_batches.append([dict(message) for message in messages])
        private_thinking_at_chat.append(agent.models.think)
        agent.models.last_usage = {"done_reason": "stop"}
        agent.models.last_response = {
            "done": True, "model": "scripted-test-model", "done_reason": "stop"
        }
        return {"role": "assistant", "content": "Understood."}

    agent.models.complete = scripted_complete
    agent.models.last_usage = {}
    agent.models.last_response = {}
    preference = "I prefer concise answers with concrete evidence."
    try:
        first = agent.handle(preference)
        second = agent.handle(preference)
        recalled = agent.handle("What stable response style have I expressed?")

        for result in (first, second):
            signal = result["full_system_receipt"]["direct_human_preference"]
            assert signal["source_id"] == "LocalUser"
            assert signal["preference"] > 0
            assert signal["confidence"] > 0
        assert second["full_system_receipt"]["behavioral_profile"][
            "learned_pattern_count"
        ] >= 1
        assert recalled["full_system_receipt"]["direct_human_preference"] is None

        pattern_ids = recalled["full_system_receipt"]["behavioral_profile"][
            "learned_pattern_record_ids"
        ]
        assert pattern_ids
        pattern = agent.loom.store.get_record(pattern_ids[-1])
        assert pattern is not None
        assert "evidence." not in pattern.metadata["shared_terms"]
        native_sources = agent.mind.store.get_records(
            pattern.metadata["habitus_source_record_ids"]
        )
        assert len(native_sources) >= 2
        assert {record.source_id for record in native_sources} == {"LocalUser"}

        personal = agent._loom_packet.sections["personal"]
        assert sum("User preference:" in line for line in personal) == 1
        assert any("Learned Habitus behavioral pattern:" in line for line in personal)

        audit = agent.context_audit
        assert audit["messages"][0]["content"] == MODEL_POLICY
        assert all(
            any("Acknowledge it once and briefly" in message["content"] for message in batch)
            for batch in message_batches[:2]
        )
        assert not any(
            "Acknowledge it once and briefly" in message["content"]
            for message in message_batches[2]
        )
        assert private_thinking_at_chat == [False, False, False]
        assert recalled["full_system_receipt"]["private_thinking_rounds"] == [False]
        assert "sha256" not in audit["messages"][1]["content"].casefold()
        assert audit["habitus_state_receipt"]["recurrent_state_sha256"]
        assert audit["habitus_state_receipt"]["habitus_recall_trace"][
            "habitus_trace_hash"
        ]
    finally:
        agent.close()
