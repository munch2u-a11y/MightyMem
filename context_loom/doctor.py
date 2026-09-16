"""Read-only deployment readiness receipt for the composed local product."""
from __future__ import annotations

import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Callable

from .deployment import (
    local_model_memory_preflight,
    require_physical_context,
    verify_ollama_server_runtime,
)


SCHEMA = "context-loom-deployment-doctor-v1"


def product_dependency_receipt() -> dict[str, Any]:
    """Inspect the actual interpreter without changing or installing anything."""
    packages: dict[str, dict[str, Any]] = {}
    required = (
        ("numpy", "numpy"),
        ("torch", "torch"),
        ("habitus-mind", "habitus_ai"),
    )
    for distribution, module_name in required:
        row: dict[str, Any] = {"distribution": distribution, "module": module_name}
        try:
            module = importlib.import_module(module_name)
            row["available"] = True
            row["version"] = importlib.metadata.version(distribution)
            module_file = getattr(module, "__file__", None)
            row["module_file"] = str(Path(module_file).resolve()) if module_file else None
        except (ImportError, importlib.metadata.PackageNotFoundError) as error:
            row.update(available=False, error=f"{type(error).__name__}: {error}")
        packages[module_name] = row
    return {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "ready": all(row.get("available") is True for row in packages.values()),
    }


def run_deployment_doctor(
    *,
    chat_model: str,
    embedding_model: str,
    ollama_url: str,
    context_tokens: int | None,
    kv_cache_type: str,
    ollama_server_pid: int | None,
    timeout: float | None = None,
    temperature: float = 0.25,
    models: Any | None = None,
    dependency_probe: Callable[[], dict[str, Any]] = product_dependency_receipt,
) -> dict[str, Any]:
    """Return a complete readiness or explicit-blocker receipt."""
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at": time.time(),
        "status": "checking",
        "entrypoint": "context_loom.habitus_agent.HabitusLoomAgent.handle",
        "requested": {
            "chat_model": chat_model,
            "embedding_model": embedding_model,
            "ollama_url": ollama_url,
            "context_tokens": context_tokens,
            "kv_cache_type": kv_cache_type,
            "ollama_server_pid": ollama_server_pid,
            "model_request_timeout": timeout,
            "chat_temperature": temperature,
            "generation_token_limit": None,
            "action_budget": None,
        },
    }
    stage = "python_environment"
    try:
        dependencies = dependency_probe()
        receipt["environment"] = dependencies
        if dependencies.get("ready") is not True:
            missing = sorted(
                name
                for name, row in dependencies.get("packages", {}).items()
                if row.get("available") is not True
            )
            raise RuntimeError("missing composed-product dependencies: " + ", ".join(missing))

        stage = "installed_models"
        if models is None:
            from habitus_ai.agent_memory import LocalModels

            models = LocalModels(
                chat=chat_model,
                embedding=embedding_model,
                base_url=ollama_url,
                timeout=timeout,
                context_tokens=context_tokens,
                maximum_tokens=None,
                num_gpu=-1,
                think=True,
                temperature=temperature,
            )
        models.kv_cache_type = kv_cache_type
        receipt["models"] = {
            "chat": models.chat_identity,
            "embedding": models.space_id,
            "context_tokens": models.context_tokens,
            "private_thinking": models.think,
            "chat_temperature": getattr(models, "temperature", temperature),
            "ollama_num_gpu": models.num_gpu,
        }

        stage = "ollama_server_runtime"
        if kv_cache_type != "f16" or ollama_server_pid is not None:
            if ollama_server_pid is None:
                raise ValueError("quantized KV cache requires --ollama-server-pid verification")
            receipt["ollama_server_runtime"] = verify_ollama_server_runtime(
                models,
                server_pid=ollama_server_pid,
                kv_cache_type=kv_cache_type,
            )
        else:
            receipt["ollama_server_runtime"] = None

        stage = "physical_context"
        preflight = local_model_memory_preflight(models)
        receipt["model_memory_preflight"] = preflight
        if preflight.get("status") != "calculated":
            raise RuntimeError(
                "model memory preflight could not be verified: "
                + str(preflight.get("reason") or "installed metadata unavailable")
            )
        require_physical_context(preflight)
        receipt.update(status="ready", completed_at=time.time())
    except Exception as error:
        receipt.update(
            status="blocked",
            completed_at=time.time(),
            blocker_stage=stage,
            error_type=type(error).__name__,
            error=str(error),
        )
    return receipt


def write_doctor_receipt(path: Path, receipt: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
