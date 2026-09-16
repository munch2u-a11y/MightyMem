"""Run independent LoCoMo questions through the exact composed local product.

Retrieval-only LoCoMo harnesses remain component diagnostics and must not be
reported as scores for this product.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
import types
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
HABITUS_ROOT = PROJECT_ROOT
DEFAULT_DATASET = WORKSPACE_ROOT / "locomo" / "data" / "locomo10.json"
DEFAULT_GRADER = WORKSPACE_ROOT / "locomo" / "task_eval" / "evaluation.py"
for path in (PROJECT_ROOT, PROJECT_ROOT):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from habitus_ai.agent_memory import LocalModels

from .deployment import (
    local_model_memory_preflight,
    require_physical_context,
    verify_ollama_server_runtime,
)
from .deployment_smoke import EXPECTED_LANES
from .compiler import (
    DIRECT_ANSWER_RESPONSE_INSTRUCTION,
    DIRECT_ANSWER_SURFACE_PROTOCOL,
    DIRECT_ANSWER_SURFACE_SCHEMA,
)
from .habitus_agent import HabitusLoomAgent

RUN_SCHEMA = "habitus-context-loom-locomo-full-product-v2"
LOOM_CONTEXT_TOKENS = 1500
LOOM_RESERVED_OUTPUT_TOKENS = 400
LOOM_MAX_INPUT_TOKENS = LOOM_CONTEXT_TOKENS - LOOM_RESERVED_OUTPUT_TOKENS


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def source_inventory() -> dict[str, str]:
    """Hash the complete composed Python source, including nested Habitus modules."""
    files = list((PROJECT_ROOT / "context_loom").rglob("*.py"))
    files += list((PROJECT_ROOT / "habitus_ai").rglob("*.py"))
    files += [
        PROJECT_ROOT / "pyproject.toml",
        PROJECT_ROOT / "requirements-product.txt",
    ]
    return {
        str(path.resolve()): sha256_file(path)
        for path in sorted(set(files), key=lambda item: str(item.resolve()))
    }


def directory_inventory(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(directory)): sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def directory_bytes(directory: Path) -> int:
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def _retain_case_state_file(relative: Path) -> bool:
    """Keep inspectable memory/event state after a scored case, excluding checkpoints."""
    return len(relative.parts) == 1 and (
        relative.name in {
            "mind.sqlite", "loom.sqlite", "events.jsonl", "combined-runtime.json",
            "config.json", "status.json",
        }
        or (relative.name.startswith("import-") and relative.suffix == ".json")
    )


def retain_case_state(state: Path, mode: str) -> dict[str, Any]:
    """Apply a post-score artifact policy without changing product execution."""
    if mode not in {"audit", "full"}:
        raise ValueError("case state retention must be audit or full")
    original = directory_inventory(state)
    original_bytes = directory_bytes(state)
    if mode == "audit":
        for path in sorted(state.rglob("*"), key=lambda item: len(item.parts), reverse=True):
            if path.is_file() and not _retain_case_state_file(path.relative_to(state)):
                path.unlink()
            elif path.is_dir():
                try:
                    path.rmdir()
                except OSError:
                    pass
    retained = directory_inventory(state)
    retained_bytes = directory_bytes(state)
    return {
        "schema": "locomo-case-state-retention-v1",
        "mode": mode,
        "applied_after_model_completion_and_official_grading": True,
        "affects_product_execution": False,
        "physical_requirement": (
            "retain auditable databases, event journal, configuration, status, and import "
            "receipt while avoiding hundreds of gigabytes of redundant per-question checkpoints"
            if mode == "audit" else None
        ),
        "restorable_product_checkpoint_retained": mode == "full",
        "original_file_count": len(original),
        "original_bytes": original_bytes,
        "original_inventory": original,
        "original_inventory_sha256": canonical_sha256(original),
        "retained_file_count": len(retained),
        "retained_bytes": retained_bytes,
        "retained_inventory": retained,
        "retained_inventory_sha256": canonical_sha256(retained),
    }


def external_input_receipt(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = path.resolve()
    if resolved.is_file():
        return {"path": str(resolved), "kind": "file", "sha256": sha256_file(resolved)}
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    inventory = directory_inventory(resolved)
    return {
        "path": str(resolved),
        "kind": "directory",
        "inventory": inventory,
        "inventory_sha256": canonical_sha256(inventory),
    }


def git_receipt(path: Path) -> dict[str, Any]:
    def call(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments], cwd=path, check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stdout.strip()
    try:
        return {"root": str(path.resolve()), "available": True,
                "head": call("rev-parse", "HEAD"), "status": call("status", "--short")}
    except (OSError, subprocess.CalledProcessError) as error:
        return {"root": str(path.resolve()), "available": False, "error": str(error)}


def parse_locomo_time(value: str, fallback: float) -> float:
    try:
        return datetime.strptime(value, "%I:%M %p on %d %B, %Y").replace(
            tzinfo=timezone.utc
        ).timestamp()
    except (ValueError, TypeError):
        return fallback


def conversation_turns(sample: dict[str, Any]) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    conversation = sample["conversation"]
    session_numbers = sorted(
        int(key.removeprefix("session_")) for key in conversation
        if key.startswith("session_") and key.removeprefix("session_").isdigit()
    )
    sequence = 0
    for session_number in session_numbers:
        session_id = f"session_{session_number}"
        timestamp = str(conversation.get(session_id + "_date_time", ""))
        created_at = parse_locomo_time(timestamp, 1_600_000_000.0 + session_number * 1000)
        for raw in conversation[session_id]:
            text = str(raw.get("text", ""))
            if raw.get("blip_caption"):
                text += " and shared " + str(raw["blip_caption"])
            turns.append({
                "source_id": str(raw["dia_id"]), "session_id": session_id,
                "timestamp": timestamp, "created_at": created_at + sequence * 0.001,
                "speaker": raw.get("speaker", "external"), "text": text,
            })
            sequence += 1
    return turns


def load_official_evaluator(grader_path: Path):
    if "bert_score" not in sys.modules:
        placeholder = types.ModuleType("bert_score")
        placeholder.score = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("bert_score is not used by LoCoMo's official QA F1 evaluator")
        )
        sys.modules["bert_score"] = placeholder
    spec = importlib.util.spec_from_file_location("locomo_official_evaluation", grader_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load official LoCoMo evaluator: {grader_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_question_answering


def official_f1(evaluator, item: dict[str, Any], response: str) -> float:
    rows = [{"prediction": response, "answer": item["answer"],
             "category": int(item["category"]), "evidence": list(item.get("evidence", []))}]
    scores, _, _ = evaluator(rows, eval_key="prediction", metric="f1")
    if len(scores) != 1:
        raise RuntimeError("official LoCoMo evaluator did not return exactly one score")
    return float(scores[0])


def exposed_source_ids(agent: HabitusLoomAgent) -> list[str]:
    output: list[str] = []
    for round_receipt in agent._loom_round_packets:
        for record_id in round_receipt["admitted_record_ids"]:
            record = agent.loom.store.get_record(record_id)
            if record is not None and record.metadata.get("source_record_id") is not None:
                output.append(str(record.metadata["source_record_id"]))
    return list(dict.fromkeys(output))


def usage_summary(turn: dict[str, Any]) -> dict[str, Any]:
    rounds = turn.get("model_rounds", [])
    prompts = [int(row.get("usage", {}).get("prompt_eval_count") or 0) for row in rounds]
    generated = [int(row.get("usage", {}).get("eval_count") or 0) for row in rounds]
    return {
        "model_calls": len(rounds), "prompt_eval_counts": prompts,
        "total_prompt_eval_tokens": sum(prompts), "generated_tokens": sum(generated),
        "thinking_rounds": sum(bool(row.get("thinking_chars")) for row in rounds),
        "thinking_chars": sum(int(row.get("thinking_chars") or 0) for row in rounds),
        "tool_calls": sum(int(row.get("tool_call_count") or 0) for row in rounds),
    }


def validate_turn_receipt(
    turn: dict[str, Any], model_inputs: list[str], models: LocalModels
) -> None:
    receipt = turn.get("full_system_receipt", {})
    expected_hashes = [hashlib.sha256(value.encode()).hexdigest() for value in model_inputs]
    surface_rounds = receipt.get("response_surface_rounds", [])
    thinking_rounds = receipt.get("private_thinking_rounds", [])
    thinking_reasons = receipt.get("private_thinking_reasons", [])
    context_rounds = receipt.get("context_rounds", [])
    thinking_receipts_valid = (
        len(thinking_rounds) == len(model_inputs)
        and len(thinking_reasons) == len(model_inputs)
        and len(context_rounds) == len(model_inputs)
        and all(
            row.get("private_thinking_enabled_for_round") is enabled
            and row.get("private_thinking_reason_for_round") == reason
            and reason in {
                "model_thinking_disabled", "policy_never", "policy_always",
                "action_stage", "temporal_resolution", "inference_resolution", "direct_reply",
            }
            for row, enabled, reason in zip(
                context_rounds, thinking_rounds, thinking_reasons
            )
        )
    )
    surface_rounds_valid = len(surface_rounds) == len(model_inputs) and all(
        row.get("schema") == DIRECT_ANSWER_SURFACE_SCHEMA
        and row.get("protocol") == DIRECT_ANSWER_SURFACE_PROTOCOL
        and row.get("raw_content_sha256")
        == hashlib.sha256(str(row.get("raw_content", "")).encode()).hexdigest()
        and row.get("surfaced_content_sha256")
        == hashlib.sha256(str(row.get("surfaced_content", "")).encode()).hexdigest()
        and (
            not row.get("applied")
            or (
                row.get("task_stage") == "understand"
                and bool(str(row.get("surfaced_content", "")).strip())
                and re.fullmatch(
                    r"ANSWER:\s*(.+?)\s*",
                    str(row.get("raw_content", "")).strip().splitlines()[0],
                    re.IGNORECASE,
                ) is not None
            )
        )
        for row in surface_rounds
    )
    checks = {
        "entrypoint": receipt.get("entrypoint")
        == "context_loom.habitus_agent.HabitusLoomAgent.handle",
        "lanes": receipt.get("lanes") == EXPECTED_LANES,
        "chat_model": receipt.get("chat_model") == models.chat_identity,
        "embedding_model": receipt.get("embedding_model") == models.space_id,
        "real_model_adapter": receipt.get("test_model_adapter") is False,
        "private_thinking": receipt.get("private_thinking_enabled") is True,
        "adaptive_private_thinking": receipt.get("private_thinking_policy") == "adaptive",
        "chat_temperature": receipt.get("chat_temperature") == models.temperature,
        "response_surface_schema": receipt.get("response_surface_schema")
        == DIRECT_ANSWER_SURFACE_SCHEMA,
        "response_surface_protocol": receipt.get("response_surface_protocol")
        == DIRECT_ANSWER_SURFACE_PROTOCOL,
        "response_surface_rounds": surface_rounds_valid,
        "final_surface_is_response": bool(surface_rounds)
        and str(surface_rounds[-1].get("surfaced_content", "")).strip()
        == str(turn.get("response", "")),
        "adaptive_thinking_receipts": thinking_receipts_valid,
        "model_inputs": bool(model_inputs) and receipt.get("model_input_sha256") == expected_hashes,
        "no_generation_limit": receipt.get("generation_token_limit") is None,
        "no_request_timeout": receipt.get("model_request_timeout") is None,
        "no_action_budget": receipt.get("action_budget") is None,
        "context_rounds": bool(receipt.get("context_rounds")),
        "packet_boundaries": all(
            row.get("packet_max_input_tokens") == LOOM_MAX_INPUT_TOKENS
            and int(row.get("packet_tokens", LOOM_MAX_INPUT_TOKENS + 1))
            <= LOOM_MAX_INPUT_TOKENS
            for row in receipt.get("context_rounds", [])
        ),
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise RuntimeError("full-product turn receipt mismatch: " + ", ".join(failed))


def runtime_signature_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    """Select runtime semantics that may safely persist across a server restart."""
    return {key: receipt[key] for key in (
        "endpoint", "kv_cache_type", "flash_attention", "cloud_disabled",
        "selected_server_environment",
    )}


def build_agent(state: Path, workspace: Path, models: LocalModels, *, scope: str,
                source_checkpoint: Path | None = None,
                sensory_assets: Path | None = None) -> HabitusLoomAgent:
    workspace.mkdir(parents=True, exist_ok=True)
    return HabitusLoomAgent(
        directory=state, workspace=workspace, models=models,
        source_checkpoint=source_checkpoint, sensory_assets=sensory_assets,
        history_turns=None, action_budget=None, checkpoint_retention=None, scope=scope,
        loom_context_tokens=LOOM_CONTEXT_TOKENS,
        loom_reserved_output_tokens=LOOM_RESERVED_OUTPUT_TOKENS,
    )


def parity_checks(composition: dict[str, Any], models: LocalModels) -> dict[str, bool]:
    return {
        "authoritative_entrypoint": composition["entrypoint"]
        == "context_loom.habitus_agent.HabitusLoomAgent.handle",
        "all_production_lanes": composition["memory_lanes"] == EXPECTED_LANES,
        "compact_packet_budget": composition["context_loom_max_input_tokens"] == LOOM_MAX_INPUT_TOKENS,
        "capability_schemas_only_at_model_api": (
            composition["context_loom_renders_capability_schemas"] is False
        ),
        "retrieval_packet_formatting_neutral": (
            composition["context_loom_response_instruction"] == ""
        ),
        "direct_answer_instruction": composition["model_response_instruction"]
        == DIRECT_ANSWER_RESPONSE_INSTRUCTION,
        "response_surface_schema": composition["response_surface_schema"]
        == DIRECT_ANSWER_SURFACE_SCHEMA,
        "response_surface_protocol": composition["response_surface_protocol"]
        == DIRECT_ANSWER_SURFACE_PROTOCOL,
        "declared_chat_model": composition["chat_model"] == models.chat_identity,
        "declared_embedding_model": composition["embedding_model"] == models.space_id,
        "real_model_adapter": not composition["test_model_adapter"],
        "private_thinking": composition["private_thinking_enabled"] is True,
        "adaptive_private_thinking": composition["private_thinking_policy"] == "adaptive",
        "chat_temperature": composition["chat_temperature"] == models.temperature,
        "no_action_budget": composition["action_budget"] is None,
        "full_conversation_history": composition["history_turns"] is None,
        "no_checkpoint_retention": composition["checkpoint_retention"] is None,
    }


def prepare_baseline(conversation_directory: Path, sample: dict[str, Any],
                     models: LocalModels, *, source_checkpoint: Path | None,
                     sensory_assets: Path | None) -> dict[str, Any]:
    baseline = conversation_directory / "pristine-state"
    receipt_path = conversation_directory / "baseline-receipt.json"
    if baseline.exists() or receipt_path.exists():
        if not baseline.is_dir() or not receipt_path.is_file():
            raise RuntimeError("incomplete pristine baseline; preserve output and start a new run")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if canonical_sha256(directory_inventory(baseline)) != receipt["state_inventory_sha256"]:
            raise RuntimeError("pristine baseline state changed after import")
        return receipt

    scope = f"locomo:{sample['sample_id']}"
    agent = build_agent(
        baseline, conversation_directory / "baseline-workspace", models, scope=scope,
        source_checkpoint=source_checkpoint, sensory_assets=sensory_assets,
    )
    try:
        composition = agent.composition()
        checks = parity_checks(composition, models)
        if not all(checks.values()):
            failed = sorted(name for name, passed in checks.items() if not passed)
            raise RuntimeError("full-product composition mismatch: " + ", ".join(failed))
        imported = agent.import_transcript(conversation_turns(sample), corpus_id=str(sample["sample_id"]))
    finally:
        agent.close()
    inventory = directory_inventory(baseline)
    receipt = {
        "schema": "locomo-pristine-baseline-v1", "sample_id": sample["sample_id"],
        "scope": scope, "created_at": time.time(), "composition": composition,
        "parity_checks": checks, "parity_verified": all(checks.values()),
        "import_receipt": imported, "state_inventory": inventory,
        "state_inventory_sha256": canonical_sha256(inventory),
    }
    atomic_json(receipt_path, receipt)
    return receipt


def archive_incomplete(path: Path) -> Path:
    archived = path.with_name(path.name + f".failed-{time.time_ns()}")
    path.replace(archived)
    return archived


def write_model_inputs(directory: Path, values: Iterable[str]) -> list[dict[str, Any]]:
    receipts = []
    for index, value in enumerate(values, 1):
        path = directory / "model-inputs" / f"round-{index:03d}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")
        receipts.append({"round": index, "path": str(path.relative_to(directory)),
                         "sha256": sha256_file(path), "bytes": path.stat().st_size})
    return receipts


def validate_case_artifacts(output: Path, row: dict[str, Any]) -> None:
    case_directory = output / row["case_artifact_directory"]
    receipt_path = case_directory / "case-receipt.json"
    if not receipt_path.is_file() or sha256_file(receipt_path) != row["case_receipt_sha256"]:
        raise RuntimeError(f"completed case receipt changed: {row['case_id']}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    artifacts = [
        receipt["turn_artifact"],
        receipt["state_retention_artifact"],
        *receipt["model_input_artifacts"],
    ]
    for artifact in artifacts:
        path = case_directory / artifact["path"]
        if not path.is_file() or sha256_file(path) != artifact["sha256"]:
            raise RuntimeError(f"completed case artifact changed: {row['case_id']} {artifact['path']}")
    retention_path = case_directory / receipt["state_retention_artifact"]["path"]
    retention = json.loads(retention_path.read_text(encoding="utf-8"))
    retained_inventory = directory_inventory(case_directory / "state")
    if (
        retained_inventory != retention["retained_inventory"]
        or canonical_sha256(retained_inventory) != retention["retained_inventory_sha256"]
    ):
        raise RuntimeError(f"completed case retained state changed: {row['case_id']}")


def run_case(output: Path, conversation_directory: Path,
             baseline_receipt: dict[str, Any], sample: dict[str, Any],
             question_index: int, item: dict[str, Any], models: LocalModels,
             evaluator, *, case_state_retention: str = "audit") -> dict[str, Any]:
    case_id = f"{sample['sample_id']}:{question_index + 1}"
    final_directory = conversation_directory / "cases" / f"question-{question_index + 1:04d}"
    if final_directory.exists():
        raise RuntimeError(f"unrecorded completed case directory exists: {final_directory}")
    temporary = final_directory.with_name(final_directory.name + ".in-progress")
    if temporary.exists():
        archive_incomplete(temporary)
    temporary.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(conversation_directory / "pristine-state", temporary / "state")
    began = time.monotonic()
    agent = None
    try:
        agent = build_agent(temporary / "state", temporary / "workspace", models,
                            scope=str(baseline_receipt["scope"]))
        checks = parity_checks(agent.composition(), models)
        if not all(checks.values()):
            failed = sorted(name for name, passed in checks.items() if not passed)
            raise RuntimeError("case composition mismatch: " + ", ".join(failed))
        turn = agent.handle(str(item["question"]))
        response = str(turn.get("response", ""))
        model_inputs = list(agent.last_model_inputs)
        validate_turn_receipt(turn, model_inputs, models)
        score = official_f1(evaluator, item, response)
        exposed = exposed_source_ids(agent)
    except BaseException:
        if agent is not None:
            agent.close()
        if temporary.exists():
            archive_incomplete(temporary)
        raise
    else:
        agent.close()

    turn_path = temporary / "turn.json"
    atomic_json(turn_path, turn)
    input_artifacts = write_model_inputs(temporary, model_inputs)
    state_retention = retain_case_state(temporary / "state", case_state_retention)
    state_retention_path = temporary / "state-retention.json"
    atomic_json(state_retention_path, state_retention)
    evidence = [str(value) for value in item.get("evidence", [])]
    case_receipt = {
        "schema": "locomo-full-product-case-v1", "case_id": case_id,
        "sample_id": sample["sample_id"], "question_index": question_index + 1,
        "question": item["question"], "answer": item["answer"],
        "category": int(item["category"]), "response": response, "official_f1": score,
        "evidence": evidence, "exposed_source_ids": exposed,
        "evidence_recall": len(set(evidence) & set(exposed)) / len(evidence) if evidence else 1.0,
        "full_evidence_recall": set(evidence).issubset(exposed),
        "entrypoint": "context_loom.habitus_agent.HabitusLoomAgent.handle",
        "baseline_state_inventory_sha256": baseline_receipt["state_inventory_sha256"],
        "turn_id": turn["turn_id"], "speech_cycle_id": turn["speech_cycle_id"],
        "usage": usage_summary(turn), "full_system_receipt": turn["full_system_receipt"],
        "turn_artifact": {"path": "turn.json", "sha256": sha256_file(turn_path)},
        "state_retention": {
            key: state_retention[key] for key in (
                "mode", "applied_after_model_completion_and_official_grading",
                "affects_product_execution", "restorable_product_checkpoint_retained",
                "original_file_count", "original_bytes", "original_inventory_sha256",
                "retained_file_count", "retained_bytes", "retained_inventory_sha256",
            )
        },
        "state_retention_artifact": {
            "path": "state-retention.json", "sha256": sha256_file(state_retention_path)
        },
        "model_input_artifacts": input_artifacts,
        "elapsed_seconds": round(time.monotonic() - began, 3),
        "official_grade_recorded_before_progress": True,
    }
    receipt_path = temporary / "case-receipt.json"
    atomic_json(receipt_path, case_receipt)
    temporary.replace(final_directory)
    row = {key: value for key, value in case_receipt.items()
           if key not in {
               "full_system_receipt", "turn_artifact", "state_retention_artifact",
               "model_input_artifacts",
           }}
    row["case_artifact_directory"] = str(final_directory.relative_to(output))
    row["case_receipt_sha256"] = sha256_file(final_directory / "case-receipt.json")
    return row


def category_breakdown(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        groups[str(row["category"])].append(float(row["official_f1"]))
    return {category: {"questions": len(scores),
                       "mean_official_f1": round(statistics.mean(scores), 6)}
            for category, scores in sorted(groups.items())}


def complete_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    prompts = [count for row in rows for count in row["usage"]["prompt_eval_counts"] if count > 0]
    return {
        "questions": len(rows),
        "official_mean_f1": round(statistics.mean(row["official_f1"] for row in rows), 6),
        "category_breakdown": category_breakdown(rows),
        "mean_evidence_recall": round(statistics.mean(row["evidence_recall"] for row in rows), 6),
        "full_evidence_recall_rate": round(statistics.mean(row["full_evidence_recall"] for row in rows), 6),
        "model_calls": sum(row["usage"]["model_calls"] for row in rows),
        "tool_calls": sum(row["usage"]["tool_calls"] for row in rows),
        "mean_actual_prompt_tokens_per_model_call": round(statistics.mean(prompts), 2) if prompts else 0.0,
        "median_actual_prompt_tokens_per_model_call": round(statistics.median(prompts), 2) if prompts else 0.0,
        "empty_responses": sum(not row["response"].strip() for row in rows),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--grader", type=Path, default=DEFAULT_GRADER)
    parser.add_argument("--conversation-index", type=int, action="append",
                        dest="conversation_indices",
                        help="full conversation; repeat as needed (default: 0 and 1, 304 questions)")
    parser.add_argument("--model", default="granite4.2:8b")
    parser.add_argument("--embedding-model", default="qwen3-embedding:0.6b")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--context-tokens", type=int, default=None,
                        help="physical model context; default uses the installed model native context")
    parser.add_argument("--kv-cache-type", choices=["f16", "q8_0", "q4_0"], default="f16")
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument(
        "--case-state-retention", choices=["audit", "full"], default="audit",
        help=(
            "post-score artifact policy: audit keeps databases/events/configuration; "
            "full also keeps every redundant per-question checkpoint"
        ),
    )
    parser.add_argument("--ollama-server-pid", type=int, required=True,
                        help="serving PID used to attest endpoint, local-only mode, and KV config")
    parser.add_argument("--source-checkpoint", type=Path)
    parser.add_argument("--sensory-assets", type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output = args.output.resolve()
    dataset_path, grader_path = args.dataset.resolve(), args.grader.resolve()
    conversation_indices = args.conversation_indices or [0, 1]
    if len(conversation_indices) != len(set(conversation_indices)):
        raise ValueError("conversation indices must be unique")
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    for index in conversation_indices:
        if index < 0 or index >= len(dataset):
            raise IndexError(f"LoCoMo conversation index out of range: {index}")
    if args.output.exists() and not args.resume:
        raise FileExistsError("output exists; pass --resume to verify and continue it")
    args.output.mkdir(parents=True, exist_ok=True)

    models = LocalModels(chat=args.model, embedding=args.embedding_model,
                         base_url=args.ollama_url, timeout=None,
                         context_tokens=args.context_tokens, maximum_tokens=None,
                         num_gpu=-1, think=True, temperature=args.temperature)
    models.kv_cache_type = args.kv_cache_type
    runtime = verify_ollama_server_runtime(
        models, server_pid=args.ollama_server_pid, kv_cache_type=args.kv_cache_type)
    memory_preflight = local_model_memory_preflight(models)
    require_physical_context(memory_preflight)
    evaluator = load_official_evaluator(grader_path)
    inventory, source_hash = source_inventory(), ""
    source_hash = canonical_sha256(inventory)
    selected = [dataset[index] for index in conversation_indices]
    total_questions = sum(len(sample["qa"]) for sample in selected)
    configuration = {
        "entrypoint": "context_loom.habitus_agent.HabitusLoomAgent.handle",
        "chat_model": models.chat_identity, "embedding_model": models.space_id,
        "ollama_base_url": models.base_url, "ollama_num_gpu": models.num_gpu,
        "private_thinking": models.think, "model_context_tokens": models.context_tokens,
        "private_thinking_policy": "adaptive",
        "chat_temperature": models.temperature,
        "response_surface_schema": DIRECT_ANSWER_SURFACE_SCHEMA,
        "response_surface_protocol": DIRECT_ANSWER_SURFACE_PROTOCOL,
        "kv_cache_type": args.kv_cache_type,
        "runtime_semantics": runtime_signature_receipt(runtime),
        "loom_compiler_model_tokens": LOOM_CONTEXT_TOKENS,
        "loom_reserved_output_tokens": LOOM_RESERVED_OUTPUT_TOKENS,
        "loom_max_input_regex_tokens": LOOM_MAX_INPUT_TOKENS,
        "memory_lanes": EXPECTED_LANES, "history_turns": None, "action_budget": None,
        "checkpoint_retention": None, "generation_token_limit": models.maximum_tokens,
        "model_request_timeout": models.timeout, "retrieval_candidate_limit": None,
        "context_section_item_limit": None, "graph_hop_limit": None,
        "question_or_sample_limit": None,
        "postscore_case_state_retention": args.case_state_retention,
        "postscore_retention_affects_product_execution": False,
        "source_checkpoint": external_input_receipt(args.source_checkpoint),
        "sensory_assets": external_input_receipt(args.sensory_assets),
    }
    signature_value = {
        "source_inventory_sha256": source_hash, "configuration": configuration,
        "dataset_sha256": sha256_file(dataset_path), "grader_sha256": sha256_file(grader_path),
        "conversation_indices": conversation_indices,
    }
    system_signature = canonical_sha256(signature_value)
    manifest_path = args.output / "composition-manifest.json"
    manifest = {
        "schema": RUN_SCHEMA, "created_at": time.time(), "status": "preflight",
        "system_signature": system_signature,
        "source_revision": "exact file inventory; Context Loom Git metadata unavailable",
        "source_inventory_sha256": source_hash, "source_inventory": inventory,
        "configuration": configuration,
        "dataset": {"path": str(dataset_path), "sha256": sha256_file(dataset_path),
                    "conversation_indices": conversation_indices,
                    "sample_ids": [sample["sample_id"] for sample in selected],
                    "questions": total_questions},
        "grader": {"path": str(grader_path), "sha256": sha256_file(grader_path),
                   "function": "eval_question_answering", "metric": "f1"},
        "environment": {"python_executable": sys.executable, "python": sys.version,
                        "platform": platform.platform(), "cpu_count": os.cpu_count(),
                        "output_filesystem_free_bytes_at_preflight": shutil.disk_usage(args.output).free},
        "git": {"dataset_repository": git_receipt(dataset_path.parents[1])},
        "ollama_server_sessions": [runtime], "model_memory_preflight": memory_preflight,
        "test_contract": {
            "real_ingestion": True, "actual_persistent_memory": True,
            "all_production_retrieval_lanes": True, "production_context_compiler": True,
            "declared_local_model": True, "real_answer_path": True,
            "official_grader": True, "independent_pristine_state_per_question": True,
            "questions_used_during_ingestion": False,
            "answer_keys_used_before_generation": False,
            "component_metric_reported_as_product_score": False,
            "postscore_state_retention_declared": True,
        },
        "parity_verified": False,
    }
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous["system_signature"] != system_signature:
            raise RuntimeError("resume requested with a different full-system composition")
        previous.setdefault("ollama_server_sessions", []).append(runtime)
        manifest = previous
    atomic_json(manifest_path, manifest)

    baseline_receipts: dict[str, Any] = {}
    for sample in selected:
        conversation_directory = args.output / "conversations" / str(sample["sample_id"])
        baseline_receipts[str(sample["sample_id"])] = prepare_baseline(
            conversation_directory, sample, models,
            source_checkpoint=args.source_checkpoint, sensory_assets=args.sensory_assets)
    all_checks = {sample_id: receipt["parity_checks"]
                  for sample_id, receipt in baseline_receipts.items()}
    if not all(all(checks.values()) for checks in all_checks.values()):
        raise RuntimeError("one or more pristine conversation baselines failed parity")
    manifest.update(
        status="running", parity_verified=True, parity_checks=all_checks,
        baselines={sample_id: {
            "state_inventory_sha256": receipt["state_inventory_sha256"],
            "turns_supplied": receipt["import_receipt"]["turns_supplied"],
        } for sample_id, receipt in baseline_receipts.items()},
    )
    atomic_json(manifest_path, manifest)

    progress_path = args.output / "results.json"
    progress = (json.loads(progress_path.read_text(encoding="utf-8"))
                if progress_path.exists() else {
                    "schema": RUN_SCHEMA, "system_signature": system_signature,
                    "complete": False, "completed": 0, "total": total_questions, "rows": []})
    if progress["system_signature"] != system_signature:
        raise RuntimeError("result state belongs to a different full-system composition")
    completed = {row["case_id"]: row for row in progress["rows"]}
    for row in progress["rows"]:
        validate_case_artifacts(args.output, row)

    for sample in selected:
        sample_id = str(sample["sample_id"])
        conversation_directory = args.output / "conversations" / sample_id
        for question_index, item in enumerate(sample["qa"]):
            case_id = f"{sample_id}:{question_index + 1}"
            if case_id in completed:
                continue
            try:
                row = run_case(args.output, conversation_directory,
                               baseline_receipts[sample_id], sample, question_index,
                               item, models, evaluator,
                               case_state_retention=args.case_state_retention)
            except BaseException as error:
                progress["failure"] = {"case_id": case_id, "type": type(error).__name__,
                                       "error": str(error), "time": time.time()}
                atomic_json(progress_path, progress)
                manifest.update(status="failed", failure=progress["failure"], failed_at=time.time())
                atomic_json(manifest_path, manifest)
                raise
            progress["rows"].append(row)
            progress.pop("failure", None)
            progress["completed"] = len(progress["rows"])
            atomic_json(progress_path, progress)
            print(f"FULL PRODUCT LOCOMO {progress['completed']}/{total_questions} "
                  f"case={case_id} f1={row['official_f1']:.4f} "
                  f"calls={row['usage']['model_calls']} "
                  f"prompt_tokens={row['usage']['total_prompt_eval_tokens']}", flush=True)

    rows = progress["rows"]
    if len(rows) != total_questions:
        raise RuntimeError("run ended without evaluating every selected question")
    progress.update(complete=True, summary=complete_summary(rows), completed_at=time.time())
    atomic_json(progress_path, progress)
    manifest.update(status="complete", completed_at=time.time())
    atomic_json(manifest_path, manifest)
    final = {"schema": RUN_SCHEMA, "system_signature": system_signature, "complete": True,
             "summary": progress["summary"], "composition_manifest": str(manifest_path),
             "results": str(progress_path)}
    atomic_json(args.output / "summary.json", final)
    return final


def main(argv: list[str] | None = None) -> int:
    result = run(build_parser().parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
