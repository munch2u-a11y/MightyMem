"""Inspectable physical-memory preflight for local Ollama contexts."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


KV_CACHE_BYTES_PER_ELEMENT = {
    "f16": 2.0,
    # GGML blocks store a scale alongside 32 quantized values.
    "q8_0": 34 / 32,
    "q4_0": 18 / 32,
}


class PhysicalContextUnavailable(RuntimeError):
    """The declared model context cannot fit the host's current memory."""


def _endpoint(value: str) -> tuple[str, int]:
    parsed = urlparse(value if "://" in value else "//" + value)
    return parsed.hostname or "", parsed.port or 11434


def verify_ollama_server_runtime(
    models: Any,
    *,
    server_pid: int,
    kv_cache_type: str,
    proc_root: Path = Path("/proc"),
) -> dict[str, Any]:
    """Prove a quantized-cache declaration against the serving process."""
    cache_type = str(kv_cache_type).casefold()
    if cache_type not in KV_CACHE_BYTES_PER_ELEMENT:
        raise ValueError(f"unsupported Ollama KV cache type: {kv_cache_type!r}")
    process = proc_root / str(int(server_pid))
    permission_restricted = False
    environ_bytes = b""
    try:
        cmdline_bytes = (process / "cmdline").read_bytes()
        argv = tuple(
            value.decode("utf-8", errors="replace")
            for value in cmdline_bytes.split(b"\0")
            if value
        )
    except OSError as error:
        raise RuntimeError(f"cannot inspect declared Ollama server PID {server_pid}: {error}") from error

    try:
        environ_bytes = (process / "environ").read_bytes()
        environment = {
            key.decode("utf-8", errors="replace"): value.decode("utf-8", errors="replace")
            for item in environ_bytes.split(b"\0")
            if item and b"=" in item
            for key, value in (item.split(b"=", 1),)
        }
    except PermissionError:
        permission_restricted = True
        environment = {}
        environ_bytes = b""
    except OSError as error:
        raise RuntimeError(f"cannot inspect declared Ollama server PID {server_pid}: {error}") from error

    if len(argv) < 2 or Path(argv[0]).name != "ollama" or argv[1] != "serve":
        raise RuntimeError(f"PID {server_pid} is not an Ollama serve process")
    actual_cache_type = environment.get("OLLAMA_KV_CACHE_TYPE", "f16").casefold()
    if actual_cache_type != cache_type:
        raise RuntimeError(
            f"Ollama server PID {server_pid} uses KV cache {actual_cache_type!r}, "
            f"not declared {cache_type!r}"
        )
    flash_value = environment.get("OLLAMA_FLASH_ATTENTION", "").casefold()
    flash_enabled = flash_value in {"1", "true", "yes", "on"}
    if cache_type != "f16" and not flash_enabled:
        raise RuntimeError("quantized Ollama KV cache requires verified Flash Attention")
    no_cloud_value = environment.get("OLLAMA_NO_CLOUD", "").casefold()
    cloud_disabled = no_cloud_value in {"1", "true", "yes", "on"} or permission_restricted
    if not cloud_disabled:
        raise RuntimeError(
            "local-model runtime verification requires OLLAMA_NO_CLOUD to be enabled"
        )
    configured_endpoint = _endpoint(environment.get("OLLAMA_HOST", "127.0.0.1:11434"))
    requested_endpoint = _endpoint(str(getattr(models, "base_url", "http://127.0.0.1:11434")))
    if not permission_restricted and configured_endpoint != requested_endpoint:
        raise RuntimeError(
            f"Ollama server PID {server_pid} listens at {configured_endpoint!r}, "
            f"but the model adapter targets {requested_endpoint!r}"
        )
    receipt = {
        "schema": "ollama-server-runtime-v1",
        "verified": True,
        "server_pid": int(server_pid),
        "server_argv": list(argv),
        "endpoint": f"http://{configured_endpoint[0]}:{configured_endpoint[1]}",
        "kv_cache_type": cache_type,
        "flash_attention": flash_enabled,
        "cloud_disabled": cloud_disabled,
        "evidence": f"{process}/cmdline and {process}/environ",
        "server_cmdline_sha256": hashlib.sha256(cmdline_bytes).hexdigest(),
        "server_environ_sha256": hashlib.sha256(environ_bytes).hexdigest(),
        "selected_server_environment": {
            key: environment.get(key)
            for key in (
                "OLLAMA_HOST",
                "OLLAMA_KV_CACHE_TYPE",
                "OLLAMA_FLASH_ATTENTION",
                "OLLAMA_MAX_LOADED_MODELS",
                "OLLAMA_NUM_PARALLEL",
                "OLLAMA_MODELS",
                "OLLAMA_NO_CLOUD",
            )
        },
    }
    models.kv_cache_type = cache_type
    models.ollama_server_runtime = dict(receipt)
    return receipt


def available_memory_bytes() -> int | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def local_model_memory_preflight(models: Any) -> dict[str, Any]:
    """Calculate KV allocation from model dimensions and proven server config."""
    kv_cache_type = str(getattr(models, "kv_cache_type", "f16")).casefold()
    if kv_cache_type not in KV_CACHE_BYTES_PER_ELEMENT:
        raise ValueError(f"unsupported Ollama KV cache type: {kv_cache_type!r}")
    runtime_receipt = getattr(models, "ollama_server_runtime", None)
    if kv_cache_type != "f16" and not (
        isinstance(runtime_receipt, dict)
        and runtime_receipt.get("verified") is True
        and runtime_receipt.get("kv_cache_type") == kv_cache_type
    ):
        raise ValueError(
            "quantized KV cache preflight requires a verified Ollama server runtime receipt"
        )
    cached = getattr(models, "context_memory_preflight", None)
    if (
        isinstance(cached, dict)
        and cached.get("context_tokens") == models.context_tokens
        and cached.get("kv_cache_type") == kv_cache_type
    ):
        return dict(cached)
    kv_element_bytes = KV_CACHE_BYTES_PER_ELEMENT[kv_cache_type]
    receipt: dict[str, Any] = {
        "schema": "ollama-context-memory-preflight-v2",
        "chat_model": getattr(models, "chat_identity", getattr(models, "chat", "unknown")),
        "context_tokens": int(models.context_tokens),
        "kv_cache_type": kv_cache_type,
        "kv_element_bytes": kv_element_bytes,
        "kv_element_assumption": (
            "Ollama default f16 KV cache"
            if kv_cache_type == "f16"
            else "verified Ollama server setting including GGML quantization block scales"
        ),
        "ollama_server_runtime": runtime_receipt,
        "available_memory_bytes": available_memory_bytes(),
        "status": "unavailable",
    }
    if not hasattr(models, "request"):
        receipt["reason"] = "model adapter does not expose installed model metadata"
        return receipt
    try:
        shown = models.request("show", {"model": models.chat})
        info = shown.get("model_info", {})
        architecture = str(info.get("general.architecture", "")).strip()
        layers = int(info[f"{architecture}.block_count"])
        embedding = int(info[f"{architecture}.embedding_length"])
        heads = int(info[f"{architecture}.attention.head_count"])
        kv_heads = int(info[f"{architecture}.attention.head_count_kv"])
        if min(layers, embedding, heads, kv_heads) <= 0 or embedding % heads:
            raise ValueError("invalid attention dimensions")
        head_dimension = embedding // heads
        kv_bytes = (
            2
            * layers
            * int(models.context_tokens)
            * kv_heads
            * head_dimension
            * kv_element_bytes
        )
        kv_bytes = int(kv_bytes)
        model_blob_bytes = None
        tags = models.request("tags").get("models", [])
        model_row = next((row for row in tags if row.get("name") == models.chat), None)
        if model_row is not None and model_row.get("size") is not None:
            model_blob_bytes = int(model_row["size"])
        resident_model = None
        try:
            running = models.request("ps").get("models", [])
            resident_model = next(
                (
                    row
                    for row in running
                    if row.get("name") == models.chat or row.get("model") == models.chat
                ),
                None,
            )
        except RuntimeError:
            # Older Ollama versions may not expose /api/ps. Memory feasibility can
            # still be checked conservatively as a new allocation.
            resident_model = None
        declared_identity = str(receipt["chat_model"])
        declared_digest = declared_identity.rsplit("@", 1)[1] if "@" in declared_identity else None
        resident_digest = str(resident_model.get("digest", "")) if resident_model else None
        resident_context = (
            int(resident_model["context_length"])
            if resident_model and resident_model.get("context_length") is not None
            else None
        )
        resident_context_verified = bool(
            resident_model
            and declared_digest
            and resident_digest == declared_digest
            and resident_context == int(models.context_tokens)
        )
        available = receipt["available_memory_bytes"]
        receipt.update(
            {
                "status": "calculated",
                "architecture": architecture,
                "layer_count": layers,
                "embedding_length": embedding,
                "attention_head_count": heads,
                "attention_kv_head_count": kv_heads,
                "attention_head_dimension": head_dimension,
                "kv_cache_bytes": kv_bytes,
                "model_blob_bytes": model_blob_bytes,
                "resident_model": resident_model,
                "resident_model_matches_declared_context": resident_context_verified,
                "new_model_allocation_required": not resident_context_verified,
                "model_plus_kv_bytes": (
                    None if model_blob_bytes is None else model_blob_bytes + kv_bytes
                ),
                "kv_only_fits_available_memory": (
                    None if available is None else kv_bytes < available
                ),
                "model_plus_kv_fits_available_memory": (
                    None
                    if available is None or model_blob_bytes is None
                    else model_blob_bytes + kv_bytes < available
                ),
                "scope": (
                    "model-blob plus KV lower bound; decoded/repacked weights, compute buffers, "
                    "prompt cache, embeddings, Python, and other processes require headroom"
                ),
            }
        )
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        receipt["reason"] = str(error)
    try:
        models.context_memory_preflight = dict(receipt)
    except (AttributeError, TypeError):
        pass
    return receipt


def require_physical_context(receipt: dict[str, Any]) -> None:
    """Fail before loading when a known model/KV lower bound cannot fit."""
    if receipt.get("status") != "calculated":
        return
    if receipt.get("resident_model_matches_declared_context") is True:
        return
    if receipt.get("model_plus_kv_fits_available_memory") is False:
        lower_mib = int(receipt["model_plus_kv_bytes"]) / (1024 * 1024)
        available_mib = int(receipt["available_memory_bytes"]) / (1024 * 1024)
        raise PhysicalContextUnavailable(
            f"declared {receipt['context_tokens']}-token context with {receipt['kv_cache_type']} "
            f"KV has a {lower_mib:.0f} MiB model-plus-KV lower bound; only "
            f"{available_mib:.0f} MiB host memory is currently available"
        )
    if receipt.get("kv_only_fits_available_memory") is not False:
        return
    kv_mib = int(receipt["kv_cache_bytes"]) / (1024 * 1024)
    available_mib = int(receipt["available_memory_bytes"]) / (1024 * 1024)
    raise PhysicalContextUnavailable(
        f"declared {receipt['context_tokens']}-token context requires {kv_mib:.0f} MiB "
        f"for KV cache alone; only {available_mib:.0f} MiB host memory is currently available"
    )
