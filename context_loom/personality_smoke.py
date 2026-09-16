"""Receipt-backed local-model smoke for grounded Habitus personality growth."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

from habitus_ai.agent_memory import LocalModels
from habitus_ai.types import RecordType

from .deployment import (
    local_model_memory_preflight,
    require_physical_context,
    verify_ollama_server_runtime,
)
from .deployment_smoke import (
    EXPECTED_LANES,
    admitted_ids,
    atomic_json,
    source_inventory,
)
from .habitus_agent import HabitusLoomAgent


SCHEMA = "habitus-context-loom-personality-smoke-v1"
PREFERENCE = "I prefer concise answers with concrete evidence."
QUERY = "What stable response style have I expressed?"


def run_personality_smoke(
    output: Path,
    *,
    chat_model: str,
    embedding_model: str,
    ollama_url: str,
    context_tokens: int | None,
    kv_cache_type: str,
    ollama_server_pid: int | None,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.mkdir(parents=True)
    state_dir = output / "state"
    workspace = output / "workspace"
    workspace.mkdir()

    models = LocalModels(
        chat=chat_model,
        embedding=embedding_model,
        base_url=ollama_url,
        timeout=None,
        context_tokens=context_tokens,
        maximum_tokens=None,
        num_gpu=-1,
        think=True,
    )
    models.kv_cache_type = kv_cache_type
    if kv_cache_type != "f16" or ollama_server_pid is not None:
        if ollama_server_pid is None:
            raise ValueError("quantized KV cache requires --ollama-server-pid verification")
        verify_ollama_server_runtime(
            models,
            server_pid=ollama_server_pid,
            kv_cache_type=kv_cache_type,
        )
    inventory = source_inventory()
    preflight = local_model_memory_preflight(models)
    manifest = {
        "schema": SCHEMA,
        "created_at": time.time(),
        "status": "preflight",
        "entrypoint": "context_loom.habitus_agent.HabitusLoomAgent.handle",
        "evaluation_entrypoint": "context_loom.personality_smoke.run_personality_smoke",
        "source_revision": "exact file hashes; Context Loom has no usable Git metadata",
        "source_inventory": inventory,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "chat_model": models.chat_identity,
        "embedding_model": models.space_id,
        "model_context_tokens": models.context_tokens,
        "kv_cache_type": kv_cache_type,
        "ollama_server_runtime": getattr(models, "ollama_server_runtime", None),
        "model_memory_preflight": preflight,
        "private_thinking": models.think,
        "chat_temperature": models.temperature,
        "model_request_timeout": models.timeout,
        "generation_token_limit": models.maximum_tokens,
        "ollama_num_gpu": models.num_gpu,
        "memory_store": str((state_dir / "mind.sqlite").resolve()),
        "loom_store": str((state_dir / "loom.sqlite").resolve()),
        "workspace": str(workspace.resolve()),
        "intended_memory_lanes": EXPECTED_LANES,
        "loom_compiler_model_tokens": 1500,
        "loom_reserved_output_tokens": 400,
        "loom_max_input_regex_tokens": 1100,
        "context_token_counter": "regex-v1",
        "action_budget": None,
        "checkpoint_retention": None,
        "graph_hop_limit": None,
        "retrieval_candidate_limit": None,
        "context_section_item_limit": None,
        "dataset": None,
        "grader": "explicit source/state/admission checks plus required response concepts",
        "test_cases": [
            "first explicit preference declaration",
            "independent repeated preference declaration",
            "related query after native pattern promotion",
            "fresh-process recovery of source and pattern state",
        ],
        "parity_verified": False,
    }
    atomic_json(output / "composition-manifest.json", manifest)
    try:
        require_physical_context(preflight)
    except Exception as error:
        manifest.update(
            status="failed",
            failed_at=time.time(),
            failure_stage="physical_context_preflight",
            error_type=type(error).__name__,
            error=str(error),
        )
        atomic_json(output / "composition-manifest.json", manifest)
        raise

    turns: list[dict[str, Any]] = []
    agent = None
    failure_stage = "product_initialization"
    try:
        agent = HabitusLoomAgent(
            directory=state_dir,
            workspace=workspace,
            models=models,
            action_budget=None,
            checkpoint_retention=None,
            scope="personality-smoke",
            loom_context_tokens=1500,
            loom_reserved_output_tokens=400,
            human_name="LocalUser",
            agent_name="LoomMind",
            taste_schema="builder",
        )
        composition = agent.composition()
        parity_checks = {
            "entrypoint": composition["entrypoint"] == manifest["entrypoint"],
            "memory_lanes": composition["memory_lanes"] == EXPECTED_LANES,
            "packet_budget": composition["context_loom_max_input_tokens"] == 1100,
            "chat_model": composition["chat_model"] == models.chat_identity,
            "embedding_model": composition["embedding_model"] == models.space_id,
            "real_models": not composition["test_model_adapter"],
            "no_action_budget": composition["action_budget"] is None,
        }
        if not all(parity_checks.values()):
            failed = sorted(name for name, passed in parity_checks.items() if not passed)
            raise RuntimeError("personality-smoke composition mismatch: " + ", ".join(failed))
        manifest.update(
            parity_verified=True,
            parity_checks=parity_checks,
            actual_composition=composition,
            capabilities=[item.id for item in agent.loom.store.list_capabilities()],
            status="running",
        )
        atomic_json(output / "composition-manifest.json", manifest)

        failure_stage = "product_turns"
        for index, prompt in enumerate((PREFERENCE, PREFERENCE, QUERY), start=1):
            turn = agent.handle(prompt)
            turns.append(turn)
            atomic_json(output / "turns" / f"turn-{index}.json", turn)

        first_signal = turns[0]["full_system_receipt"]["direct_human_preference"]
        second_signal = turns[1]["full_system_receipt"]["direct_human_preference"]
        third_signal = turns[2]["full_system_receipt"]["direct_human_preference"]
        pattern_ids = turns[2]["full_system_receipt"]["behavioral_profile"][
            "learned_pattern_record_ids"
        ]
        patterns = [agent.loom.store.get_record(record_id) for record_id in pattern_ids]
        source_ids = sorted({
            source_id
            for pattern in patterns
            if pattern is not None
            for source_id in pattern.metadata.get("habitus_source_record_ids", [])
        })
        native_sources = agent.mind.store.get_records(source_ids)
        experience_states = []
        for record in native_sources:
            experience_id = str(
                record.metadata.get("experience_id") or record.event_id or record.record_id
            )
            state = agent.mind.store.get_experience_state(experience_id)
            experience_states.append({
                "record_id": record.record_id,
                "source_id": record.source_id,
                "record_type": record.record_type.value,
                "experience_id": experience_id,
                "preference": state.preference_mean if state else None,
                "confidence": min(1.0, state.preference_weight) if state else None,
                "imported_transcript": bool(record.metadata.get("imported_transcript")),
            })
        response = str(turns[2].get("response", ""))
        outcome_checks = {
            "two_distinct_direct_signals": bool(
                first_signal
                and second_signal
                and first_signal["source_record_id"] != second_signal["source_record_id"]
            ),
            "signals_attributed_to_configured_human": bool(
                first_signal
                and second_signal
                and {first_signal["source_id"], second_signal["source_id"]} == {"LocalUser"}
            ),
            "query_not_misclassified_as_preference": third_signal is None,
            "pattern_promoted": bool(pattern_ids),
            "pattern_admitted_on_related_query": bool(set(pattern_ids) & admitted_ids(turns[2])),
            "pattern_has_two_native_human_sources": (
                len(native_sources) >= 2
                and all(record.source_id == "LocalUser" for record in native_sources)
                and all(record.record_type == RecordType.INBOUND_MESSAGE for record in native_sources)
                and all(not record.metadata.get("imported_transcript") for record in native_sources)
            ),
            "source_experiences_have_positive_preference": bool(experience_states)
            and all(
                row["preference"] is not None
                and row["preference"] > 0
                and row["confidence"] is not None
                and row["confidence"] > 0
                for row in experience_states
            ),
            "response_mentions_concise": "concise" in response.casefold(),
            "response_mentions_evidence": "evidence" in response.casefold(),
            "all_model_calls_stopped_normally": all(
                round_["usage"].get("done_reason") == "stop"
                for turn in turns
                for round_ in turn.get("model_rounds", [])
            ),
            "loom_integrity": agent.loom.store.integrity_check() == "ok",
        }
        before_close = {
            "outcome_checks": outcome_checks,
            "direct_preference_signals": [first_signal, second_signal],
            "pattern_record_ids": pattern_ids,
            "pattern_records": [
                {
                    "id": pattern.id,
                    "content": pattern.content,
                    "metadata": pattern.metadata,
                }
                for pattern in patterns
                if pattern is not None
            ],
            "native_source_experience_states": experience_states,
            "related_response": response,
            "state": agent.state(),
        }
        if not all(outcome_checks.values()):
            failed = sorted(name for name, passed in outcome_checks.items() if not passed)
            failure_stage = "product_outcome_validation"
            raise RuntimeError("personality-smoke checks failed: " + ", ".join(failed))
    except (Exception, KeyboardInterrupt) as error:
        manifest.update(
            status="failed",
            failed_at=time.time(),
            failure_stage=failure_stage,
            error_type=type(error).__name__,
            error=str(error),
        )
        atomic_json(output / "composition-manifest.json", manifest)
        atomic_json(output / "failure.json", {
            "schema": SCHEMA,
            "stage": failure_stage,
            "error_type": type(error).__name__,
            "error": str(error),
            "completed_turn_count": len(turns),
        })
        raise
    finally:
        if agent is not None:
            agent.close()

    failure_stage = "durable_recovery"
    try:
        recovered = HabitusLoomAgent(
            directory=state_dir,
            workspace=workspace,
            models=models,
            action_budget=None,
            checkpoint_retention=None,
            recover_working=True,
            scope="personality-smoke",
            loom_context_tokens=1500,
            loom_reserved_output_tokens=400,
        )
        try:
            recovered_profile = recovered.behavioral_profile(include_record_ids=True)
            recovered_patterns = [
                recovered.loom.store.get_record(record_id) for record_id in pattern_ids
            ]
            recovered_sources = [
                recovered.mind.store.get_record(record_id) for record_id in source_ids
            ]
            recovery_checks = {
                "pattern_ids_recovered": set(pattern_ids)
                <= set(recovered_profile["learned_pattern_record_ids"]),
                "pattern_records_recovered": all(record is not None for record in recovered_patterns),
                "native_sources_recovered": all(record is not None for record in recovered_sources),
                "loom_integrity": recovered.loom.store.integrity_check() == "ok",
            }
            recovery = {
                "outcome_checks": recovery_checks,
                "behavioral_profile": recovered_profile,
                "state": recovered.state(),
            }
            if not all(recovery_checks.values()):
                failed = sorted(name for name, passed in recovery_checks.items() if not passed)
                raise RuntimeError("personality recovery checks failed: " + ", ".join(failed))
        finally:
            recovered.close()
    except (Exception, KeyboardInterrupt) as error:
        manifest.update(
            status="failed",
            failed_at=time.time(),
            failure_stage=failure_stage,
            error_type=type(error).__name__,
            error=str(error),
        )
        atomic_json(output / "composition-manifest.json", manifest)
        atomic_json(output / "failure.json", {
            "schema": SCHEMA,
            "stage": failure_stage,
            "error_type": type(error).__name__,
            "error": str(error),
            "completed_turn_count": len(turns),
        })
        raise

    result = {
        "schema": SCHEMA,
        "composition_manifest": str((output / "composition-manifest.json").resolve()),
        "turn_count": len(turns),
        "turn_artifacts": [
            str((output / "turns" / f"turn-{index}.json").resolve())
            for index in range(1, len(turns) + 1)
        ],
        "before_close": before_close,
        "recovery": recovery,
    }
    manifest.update(status="complete", completed_at=time.time())
    atomic_json(output / "composition-manifest.json", manifest)
    atomic_json(output / "result.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/personality-smoke-current"))
    parser.add_argument("--model", default="granite4.2:8b")
    parser.add_argument("--embedding-model", default="qwen3-embedding:0.6b")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--context-tokens", type=int, default=None)
    parser.add_argument("--kv-cache-type", choices=["f16", "q8_0", "q4_0"], default="f16")
    parser.add_argument("--ollama-server-pid", type=int, default=None)
    args = parser.parse_args(argv)
    result = run_personality_smoke(
        args.output,
        chat_model=args.model,
        embedding_model=args.embedding_model,
        ollama_url=args.ollama_url,
        context_tokens=args.context_tokens,
        kv_cache_type=args.kv_cache_type,
        ollama_server_pid=args.ollama_server_pid,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
