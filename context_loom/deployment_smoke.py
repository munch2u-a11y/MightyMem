"""Receipt-backed local deployment smoke for the composed Habitus + Context Loom agent."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

from habitus_ai.agent_memory import LocalModels

from .habitus_agent import HabitusLoomAgent
from .deployment import (
    local_model_memory_preflight,
    require_physical_context,
    verify_ollama_server_runtime,
)
from .models import RecordKind


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HABITUS_ROOT = PROJECT_ROOT
SCHEMA = "habitus-context-loom-deployment-smoke-v1"
EXPECTED_LANES = [
    "semantic",
    "exact",
    "recent",
    "subjective",
    "procedure",
    "correction",
    "habitus_graph",
    "multi_hop_graph",
    "y_cipher_depth",
]
DEPLOYMENT_FILE_CONTENT = "Context Loom local deployment works."


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


def source_inventory() -> dict[str, str]:
    paths = sorted((PROJECT_ROOT / "context_loom").glob("*.py"))
    paths += sorted((PROJECT_ROOT / "habitus_ai").glob("*.py"))
    paths += [PROJECT_ROOT / "pyproject.toml"]
    return {str(path.resolve()): sha256_file(path) for path in paths}


def admitted_ids(turn: dict[str, Any]) -> set[str]:
    return {
        record_id
        for context_round in turn.get("full_system_receipt", {}).get("context_rounds", [])
        for record_id in context_round.get("admitted_record_ids", [])
    }


def run_smoke(
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
    source_digest = hashlib.sha256(
        json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    memory_preflight = local_model_memory_preflight(models)
    manifest = {
        "schema": SCHEMA,
        "created_at": time.time(),
        "status": "preflight",
        "entrypoint": "context_loom.habitus_agent.HabitusLoomAgent.handle",
        "source_revision": "exact file hashes; Context Loom has no usable Git metadata",
        "source_inventory_sha256": source_digest,
        "source_inventory": inventory,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "torch": {
            "version": __import__("torch").__version__,
            "cuda_available": __import__("torch").cuda.is_available(),
            "hip_version": getattr(__import__("torch").version, "hip", None),
        },
        "chat_model": models.chat_identity,
        "embedding_model": models.space_id,
        "model_context_tokens": models.context_tokens,
        "kv_cache_type": kv_cache_type,
        "ollama_server_runtime": getattr(models, "ollama_server_runtime", None),
        "private_thinking": models.think,
        "chat_temperature": models.temperature,
        "model_request_timeout": models.timeout,
        "generation_token_limit": models.maximum_tokens,
        "ollama_num_gpu": models.num_gpu,
        "model_memory_preflight": memory_preflight,
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
        "test_cases": [
            "remember a response-style preference",
            "inspect, write, and read back a workspace file",
            "recall the preference through the normal answer path",
            "reopen the saved agent and verify durable state",
        ],
        "parity_verified": False,
    }
    atomic_json(output / "composition-manifest.json", manifest)
    try:
        require_physical_context(memory_preflight)
    except Exception as error:
        manifest["status"] = "failed"
        manifest["failed_at"] = time.time()
        manifest["failure_stage"] = "physical_context_preflight"
        manifest["error_type"] = type(error).__name__
        manifest["error"] = str(error)
        atomic_json(output / "composition-manifest.json", manifest)
        atomic_json(
            output / "failure.json",
            {
                "schema": SCHEMA,
                "stage": "physical_context_preflight",
                "error_type": type(error).__name__,
                "error": str(error),
                "model_memory_preflight": memory_preflight,
                "completed_turn_count": 0,
            },
        )
        raise

    agent = HabitusLoomAgent(
        directory=state_dir,
        workspace=workspace,
        models=models,
        action_budget=None,
        checkpoint_retention=None,
        scope="deployment-smoke",
        loom_context_tokens=1500,
        loom_reserved_output_tokens=400,
        human_name="LocalUser",
        agent_name="LoomMind",
        taste_schema="builder",
    )
    composition = agent.composition()
    if composition["entrypoint"] != manifest["entrypoint"]:
        agent.close()
        raise RuntimeError("authoritative entrypoint mismatch")
    if composition["memory_lanes"] != EXPECTED_LANES:
        agent.close()
        raise RuntimeError(f"memory lane mismatch: {composition['memory_lanes']!r}")
    if composition["context_loom_max_input_tokens"] != 1100:
        agent.close()
        raise RuntimeError("Context Loom packet budget mismatch")
    if composition["chat_model"] != models.chat_identity:
        agent.close()
        raise RuntimeError("chat model mismatch")
    if composition["embedding_model"] != models.space_id:
        agent.close()
        raise RuntimeError("embedding model mismatch")
    if composition["test_model_adapter"]:
        agent.close()
        raise RuntimeError("deployment smoke cannot use a test model adapter")
    manifest["parity_verified"] = True
    manifest["status"] = "running"
    manifest["actual_composition"] = composition
    manifest["capabilities"] = [capability.id for capability in agent.loom.store.list_capabilities()]
    atomic_json(output / "composition-manifest.json", manifest)

    prompts = [
        "Please remember this working preference: keep answers concise and cite concrete evidence.",
        (
            f"Create deployment-smoke.txt containing exactly: {DEPLOYMENT_FILE_CONTENT} "
            "Inspect the workspace first, write the file, then read it back and verify its exact content."
        ),
        "What working preference did I ask you to remember?",
    ]
    turns: list[dict[str, Any]] = []
    failure_stage = "product_turns"
    try:
        preference_record_id = ""
        for index, prompt in enumerate(prompts):
            turn = agent.handle(prompt)
            turns.append(turn)
            if index == 0:
                row = agent.loom.store.connection.execute(
                    "SELECT id FROM records WHERE scope=? AND content=? ORDER BY created_at DESC LIMIT 1",
                    ("deployment-smoke", prompt),
                ).fetchone()
                preference_record_id = str(row["id"]) if row else ""
            atomic_json(output / "turns" / f"turn-{index + 1}.json", turn)

        expected_file_content = DEPLOYMENT_FILE_CONTENT
        written_path = workspace / "deployment-smoke.txt"
        actual_file_content = written_path.read_text(encoding="utf-8") if written_path.exists() else None
        third_response = str(turns[2].get("response", ""))
        reflection_rows = agent.loom.store.connection.execute(
            "SELECT id,source_ids,metadata FROM records WHERE kind=? AND scope=?",
            (RecordKind.THOUGHT.value, "deployment-smoke"),
        ).fetchall()
        before_close = {
            "workspace_file_exists": written_path.is_file(),
            "workspace_file_exact": actual_file_content == expected_file_content,
            "workspace_file_sha256": (
                hashlib.sha256(actual_file_content.encode()).hexdigest()
                if actual_file_content is not None
                else None
            ),
            "preference_record_id": preference_record_id,
            "preference_admitted_on_recall": preference_record_id in admitted_ids(turns[2]),
            "recall_response_mentions_concise": "concise" in third_response.casefold(),
            "recall_response_mentions_evidence": "evidence" in third_response.casefold(),
            "verified_tool_receipts": [
                receipt["receipt_id"]
                for turn in turns
                for receipt in turn.get("receipts", [])
                if receipt.get("verified")
            ],
            "successful_verified_tool_abilities": [
                receipt["tool_id"]
                for turn in turns
                for receipt in turn.get("receipts", [])
                if receipt.get("verified") and receipt.get("status") == "success"
            ],
            "learned_procedures": [turn.get("learned_procedure") for turn in turns],
            "reflection_checkpoint_count": len(reflection_rows),
            "loom_integrity": agent.loom.store.integrity_check(),
            "state": agent.state(),
        }
        required_abilities = {"ability:workspace-write", "ability:workspace-read"}
        outcome_checks = {
            "workspace_file_exact": before_close["workspace_file_exact"],
            "preference_admitted_on_recall": before_close["preference_admitted_on_recall"],
            "recall_response_mentions_concise": before_close["recall_response_mentions_concise"],
            "recall_response_mentions_evidence": before_close["recall_response_mentions_evidence"],
            "required_workspace_abilities_succeeded": required_abilities <= set(
                before_close["successful_verified_tool_abilities"]
            ),
            "loom_integrity": before_close["loom_integrity"] == "ok",
        }
        before_close["outcome_checks"] = outcome_checks
        if not all(outcome_checks.values()):
            failure_stage = "product_outcome_validation"
            failed = sorted(name for name, passed in outcome_checks.items() if not passed)
            raise RuntimeError("product smoke outcome checks failed: " + ", ".join(failed))
    except Exception as error:
        manifest["status"] = "failed"
        manifest["failed_at"] = time.time()
        manifest["failure_stage"] = failure_stage
        manifest["error_type"] = type(error).__name__
        manifest["error"] = str(error)
        atomic_json(output / "composition-manifest.json", manifest)
        atomic_json(
            output / "failure.json",
            {
                "schema": SCHEMA,
                "stage": failure_stage,
                "error_type": type(error).__name__,
                "error": str(error),
                "completed_turn_count": len(turns),
                "completed_turn_artifacts": [
                    str((output / "turns" / f"turn-{i + 1}.json").resolve())
                    for i in range(len(turns))
                ],
            },
        )
        raise
    finally:
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
            scope="deployment-smoke",
            loom_context_tokens=1500,
            loom_reserved_output_tokens=400,
        )
        try:
            recovered_record = recovered.loom.store.get_record(before_close["preference_record_id"])
            recovered_file_content = (
                written_path.read_text(encoding="utf-8") if written_path.is_file() else None
            )
            recovery = {
                "preference_record_recovered": recovered_record is not None,
                "preference_text_recovered": (
                    recovered_record is not None and recovered_record.content == prompts[0]
                ),
                "workspace_file_exact": recovered_file_content == expected_file_content,
                "loom_integrity": recovered.loom.store.integrity_check(),
                "composition": recovered.composition(),
                "state": recovered.state(),
            }
            recovery_checks = {
                "preference_record_recovered": recovery["preference_record_recovered"],
                "preference_text_recovered": recovery["preference_text_recovered"],
                "workspace_file_exact": recovery["workspace_file_exact"],
                "loom_integrity": recovery["loom_integrity"] == "ok",
            }
            recovery["outcome_checks"] = recovery_checks
            if not all(recovery_checks.values()):
                failed = sorted(name for name, passed in recovery_checks.items() if not passed)
                raise RuntimeError("durable recovery checks failed: " + ", ".join(failed))
        finally:
            recovered.close()
    except Exception as error:
        manifest["status"] = "failed"
        manifest["failed_at"] = time.time()
        manifest["failure_stage"] = failure_stage
        manifest["error_type"] = type(error).__name__
        manifest["error"] = str(error)
        atomic_json(output / "composition-manifest.json", manifest)
        atomic_json(
            output / "failure.json",
            {
                "schema": SCHEMA,
                "stage": failure_stage,
                "error_type": type(error).__name__,
                "error": str(error),
                "completed_turn_count": len(turns),
            },
        )
        raise

    result = {
        "schema": SCHEMA,
        "composition_manifest": str((output / "composition-manifest.json").resolve()),
        "turn_count": len(turns),
        "before_close": before_close,
        "recovery": recovery,
        "turn_artifacts": [str((output / "turns" / f"turn-{i + 1}.json").resolve()) for i in range(len(turns))],
    }
    manifest["status"] = "complete"
    manifest["completed_at"] = time.time()
    atomic_json(output / "composition-manifest.json", manifest)
    atomic_json(output / "result.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/deployment-smoke-current"))
    parser.add_argument("--model", default="granite4.2:8b")
    parser.add_argument("--embedding-model", default="qwen3-embedding:0.6b")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument(
        "--context-tokens",
        type=int,
        default=None,
        help="explicit physical model context; default uses the model's native context",
    )
    parser.add_argument("--kv-cache-type", choices=["f16", "q8_0", "q4_0"], default="f16")
    parser.add_argument(
        "--ollama-server-pid", type=int, default=None,
        help="PID used to verify endpoint, KV format, and Flash Attention",
    )
    args = parser.parse_args(argv)
    result = run_smoke(
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
