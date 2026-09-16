"""Gestation and Model Pairing for Context Loom Agents."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any

# Ensure habitus-mind src is on sys.path if not already present
HABITUS_SRC = Path(__file__).resolve().parents[2] / "habitus-ai-experiments" / "habitus-mind" / "src"
if HABITUS_SRC.exists() and str(HABITUS_SRC) not in sys.path:
    sys.path.insert(0, str(HABITUS_SRC))

from habitus_ai.agent_memory import LocalModels
from .deployment import (
    local_model_memory_preflight,
    require_physical_context,
    verify_ollama_server_runtime,
)


@dataclass(frozen=True)
class GestationConfig:
    human_name: str = "User"
    agent_name: str = "LoomAgent"
    taste_schema: str = "builder"
    model_backend: str = "ollama"
    model_name: str = "granite4.2:8b"
    embedding_model: str = "qwen3-embedding:0.6b"
    ollama_url: str = "http://127.0.0.1:11434"
    timeout: float | None = None  # None = no timeout limitation
    maximum_tokens: int | None = None  # None = uncapped generation
    temperature: float = 0.25
    context_tokens: int | None = None  # None = model native context length
    kv_cache_type: str = "f16"
    ollama_server_pid: int | None = None
    scope: str = "default"
    loom_context_tokens: int | None = 1500
    loom_reserved_output_tokens: int = 400
    allow_test_models: bool = False
    private_thinking_policy: str = "adaptive"


def gestate_loom_agent(
    *,
    directory: Path | str,
    workspace: Path | str,
    config: GestationConfig | None = None,
    models: LocalModels | None = None,
    allowed_programs: dict[str, Path] | None = None,
) -> dict[str, Any]:
    """Gestate and pair a Context Loom + Habitus Mind without artificial limitations."""
    if config is None:
        config = GestationConfig()

    dir_path = Path(directory).resolve()
    ws_path = Path(workspace).resolve()
    dir_path.mkdir(parents=True, exist_ok=True)
    ws_path.mkdir(parents=True, exist_ok=True)

    if models is None and config.model_name:
        try:
            models = LocalModels(
                chat=config.model_name,
                embedding=config.embedding_model,
                base_url=config.ollama_url,
                timeout=config.timeout,
                context_tokens=config.context_tokens,
                maximum_tokens=config.maximum_tokens,
                temperature=config.temperature,
                num_gpu=-1,
                think=True,
            )
        except Exception:
            if not config.allow_test_models:
                raise
            class DummyModels:
                chat_identity = config.model_name + "@dummy"
                space_id = config.embedding_model + "@dummy"
                chat = config.model_name
                embedding = config.embedding_model
                context_tokens = config.context_tokens or 131072
                maximum_tokens = config.maximum_tokens
                timeout = config.timeout
                num_gpu = -1
                think = True
                temperature = config.temperature
                base_url = config.ollama_url

                def embed(self, texts):
                    import numpy as np
                    return np.ones((len(texts), 16), dtype=np.float32)

                def complete(self, messages, tools):
                    raise RuntimeError(
                        "dummy completion requires an explicit scripted response"
                    )

            models = DummyModels()

    if models is None:
        raise ValueError("gestation requires an explicit installed model adapter")

    models.kv_cache_type = config.kv_cache_type
    if config.kv_cache_type != "f16" or config.ollama_server_pid is not None:
        if config.ollama_server_pid is None:
            raise ValueError("quantized KV cache requires --ollama-server-pid verification")
        verify_ollama_server_runtime(
            models,
            server_pid=config.ollama_server_pid,
            kv_cache_type=config.kv_cache_type,
        )

    memory_preflight = local_model_memory_preflight(models)
    (dir_path / "model-memory-preflight.json").write_text(
        json.dumps(memory_preflight, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not config.allow_test_models:
        require_physical_context(memory_preflight)

    from .habitus_agent import HabitusLoomAgent

    agent = HabitusLoomAgent(
        directory=dir_path,
        workspace=ws_path,
        models=models,
        scope=config.scope,
        loom_context_tokens=config.loom_context_tokens,
        loom_reserved_output_tokens=config.loom_reserved_output_tokens,
        allowed_programs=allowed_programs,
        human_name=config.human_name,
        agent_name=config.agent_name,
        taste_schema=config.taste_schema,
        allow_test_models=config.allow_test_models,
        private_thinking_policy=config.private_thinking_policy,
    )
    profile = agent.profile

    loom = agent.loom
    agent.save(label="gestated-and-paired")
    receipt = {
        "status": "gestated",
        "directory": str(dir_path),
        "workspace": str(ws_path),
        "profile": asdict(profile),
        "scope": config.scope,
        "models": {
            "chat": getattr(models, "chat_identity", config.model_name),
            "embedding": getattr(models, "space_id", config.embedding_model),
            "timeout": getattr(models, "timeout", config.timeout),
            "maximum_tokens": getattr(models, "maximum_tokens", config.maximum_tokens),
            "temperature": getattr(models, "temperature", config.temperature),
            "context_tokens": getattr(models, "context_tokens", config.context_tokens),
            "kv_cache_type": getattr(models, "kv_cache_type", config.kv_cache_type),
            "ollama_server_runtime": getattr(models, "ollama_server_runtime", None),
            "test_adapter": str(getattr(models, "chat_identity", "")).endswith("@dummy"),
        },
        "model_memory_preflight": memory_preflight,
        "composition": agent.composition(),
        "loom": {
            "max_input_tokens": loom.compiler.config.max_input_tokens,
            "reserved_output_tokens": loom.compiler.config.reserved_output_tokens,
        },
    }
    agent.close()
    return receipt
