from __future__ import annotations

import hashlib

import pytest

from habitus_ai.agent_memory import LocalModels

from context_loom.deployment import (
    PhysicalContextUnavailable,
    local_model_memory_preflight,
    require_physical_context,
    verify_ollama_server_runtime,
)
from context_loom.doctor import run_deployment_doctor, write_doctor_receipt
from context_loom.cli import parser


def test_local_model_completion_sends_declared_sampling_temperature() -> None:
    models = object.__new__(LocalModels)
    models.chat = "chat"
    models.context_tokens = 8192
    models.num_gpu = -1
    models.think = False
    models.maximum_tokens = None
    models.temperature = 0.15
    models.last_response = {}
    models.last_usage = {}
    observed = {}

    def request(endpoint, payload=None):
        observed.update(endpoint=endpoint, payload=payload)
        return {
            "model": "chat",
            "done": True,
            "done_reason": "stop",
            "message": {"role": "assistant", "content": "ready"},
        }

    models.request = request
    assert models.complete([{"role": "user", "content": "hi"}], [])[
        "content"
    ] == "ready"
    assert observed["endpoint"] == "chat"
    assert observed["payload"]["options"]["temperature"] == 0.15


class InstalledGranite:
    chat = "granite4.2:8b"
    chat_identity = "granite4.2:8b@digest"
    context_tokens = 131072
    base_url = "http://127.0.0.1:11436"

    def request(self, endpoint, payload=None):
        if endpoint == "show":
            return {
                "model_info": {
                    "general.architecture": "granite",
                    "granite.block_count": 40,
                    "granite.embedding_length": 4096,
                    "granite.attention.head_count": 32,
                    "granite.attention.head_count_kv": 8,
                }
            }
        if endpoint == "tags":
            return {"models": [{"name": self.chat, "size": 5_300_000_000}]}
        if endpoint == "ps":
            return {"models": []}
        raise AssertionError(endpoint)


def test_granite_native_context_preflight_reproduces_ollama_kv_allocation() -> None:
    receipt = local_model_memory_preflight(InstalledGranite())
    assert receipt["status"] == "calculated"
    assert receipt["kv_cache_bytes"] == 20_480 * 1024 * 1024
    assert receipt["context_tokens"] == 131072
    assert receipt["model_blob_bytes"] == 5_300_000_000


def test_verified_q4_server_preflight_matches_observed_5760_mib_cache(tmp_path) -> None:
    process = tmp_path / "421"
    process.mkdir()
    (process / "cmdline").write_bytes(b"/usr/local/bin/ollama\0serve\0")
    (process / "environ").write_bytes(
        b"OLLAMA_HOST=127.0.0.1:11436\0"
        b"OLLAMA_KV_CACHE_TYPE=q4_0\0"
        b"OLLAMA_FLASH_ATTENTION=1\0"
        b"OLLAMA_NO_CLOUD=true\0"
    )
    models = InstalledGranite()
    server = verify_ollama_server_runtime(
        models, server_pid=421, kv_cache_type="q4_0", proc_root=tmp_path
    )
    receipt = local_model_memory_preflight(models)
    assert server["verified"] is True
    assert server["server_cmdline_sha256"] == hashlib.sha256(
        b"/usr/local/bin/ollama\0serve\0"
    ).hexdigest()
    assert server["selected_server_environment"]["OLLAMA_KV_CACHE_TYPE"] == "q4_0"
    assert server["cloud_disabled"] is True
    assert receipt["kv_cache_type"] == "q4_0"
    assert receipt["kv_cache_bytes"] == 5_760 * 1024 * 1024
    assert receipt["ollama_server_runtime"] == server


def test_quantized_preflight_rejects_unverified_server_declaration() -> None:
    models = InstalledGranite()
    models.kv_cache_type = "q4_0"
    with pytest.raises(ValueError, match="verified Ollama server"):
        local_model_memory_preflight(models)


def test_resident_exact_context_does_not_require_a_second_full_allocation(
    tmp_path, monkeypatch
) -> None:
    class ResidentGranite(InstalledGranite):
        def request(self, endpoint, payload=None):
            if endpoint == "ps":
                return {
                    "models": [
                        {
                            "name": self.chat,
                            "digest": "digest",
                            "context_length": self.context_tokens,
                            "size": 12_336_790_239,
                        }
                    ]
                }
            return super().request(endpoint, payload)

    process = tmp_path / "423"
    process.mkdir()
    (process / "cmdline").write_bytes(b"ollama\0serve\0")
    (process / "environ").write_bytes(
        b"OLLAMA_HOST=127.0.0.1:11436\0"
        b"OLLAMA_KV_CACHE_TYPE=q4_0\0"
        b"OLLAMA_FLASH_ATTENTION=1\0"
        b"OLLAMA_NO_CLOUD=true\0"
    )
    monkeypatch.setattr("context_loom.deployment.available_memory_bytes", lambda: 512 * 1024**2)
    models = ResidentGranite()
    verify_ollama_server_runtime(
        models, server_pid=423, kv_cache_type="q4_0", proc_root=tmp_path
    )
    receipt = local_model_memory_preflight(models)
    assert receipt["resident_model_matches_declared_context"] is True
    assert receipt["new_model_allocation_required"] is False
    assert receipt["model_plus_kv_fits_available_memory"] is False
    require_physical_context(receipt)


def test_server_verification_rejects_a_cache_type_mismatch(tmp_path) -> None:
    process = tmp_path / "422"
    process.mkdir()
    (process / "cmdline").write_bytes(b"ollama\0serve\0")
    (process / "environ").write_bytes(
        b"OLLAMA_HOST=127.0.0.1:11436\0OLLAMA_KV_CACHE_TYPE=f16\0"
    )
    with pytest.raises(RuntimeError, match="not declared"):
        verify_ollama_server_runtime(
            InstalledGranite(), server_pid=422, kv_cache_type="q4_0", proc_root=tmp_path
        )


def test_server_verification_rejects_cloud_enabled_runtime(tmp_path) -> None:
    process = tmp_path / "424"
    process.mkdir()
    (process / "cmdline").write_bytes(b"ollama\0serve\0")
    (process / "environ").write_bytes(
        b"OLLAMA_HOST=127.0.0.1:11436\0"
        b"OLLAMA_KV_CACHE_TYPE=q4_0\0"
        b"OLLAMA_FLASH_ATTENTION=1\0"
    )
    with pytest.raises(RuntimeError, match="OLLAMA_NO_CLOUD"):
        verify_ollama_server_runtime(
            InstalledGranite(), server_pid=424, kv_cache_type="q4_0", proc_root=tmp_path
        )


def test_physical_context_failure_is_explicit_before_model_loading() -> None:
    receipt = {
        "status": "calculated",
        "context_tokens": 131072,
        "kv_cache_bytes": 20_480 * 1024 * 1024,
        "available_memory_bytes": 12_000 * 1024 * 1024,
        "kv_only_fits_available_memory": False,
    }
    with pytest.raises(PhysicalContextUnavailable, match="KV cache alone"):
        require_physical_context(receipt)


def test_unknown_model_metadata_does_not_invent_a_memory_limit() -> None:
    class UnknownAdapter:
        chat = "unknown"
        chat_identity = "unknown@digest"
        context_tokens = 4096

    receipt = local_model_memory_preflight(UnknownAdapter())
    assert receipt["status"] == "unavailable"
    require_physical_context(receipt)


def test_product_cli_exposes_the_physical_context_boundary() -> None:
    gestate = parser().parse_args(["gestate", "--context-tokens", "16384"])
    chat = parser().parse_args(["chat", "--context-tokens", "16384"])
    assert gestate.context_tokens == 16384
    assert chat.context_tokens == 16384
    quantized = parser().parse_args(
        ["chat", "--kv-cache-type", "q4_0", "--ollama-server-pid", "421"]
    )
    assert quantized.kv_cache_type == "q4_0"
    assert quantized.ollama_server_pid == 421

    doctor = parser().parse_args(
        ["doctor", "--context-tokens", "131072", "--kv-cache-type", "q4_0"]
    )
    assert doctor.context_tokens == 131072
    assert doctor.kv_cache_type == "q4_0"


def test_deployment_doctor_writes_a_ready_composition_receipt(tmp_path, monkeypatch) -> None:
    class DoctorGranite(InstalledGranite):
        context_tokens = 4096
        space_id = "qwen3-embedding:0.6b@embedding-digest"
        embedding = "qwen3-embedding:0.6b"
        think = True
        num_gpu = -1

    dependencies = {
        "ready": True,
        "python_executable": "/declared/product/python",
        "packages": {
            "numpy": {"available": True},
            "torch": {"available": True},
            "habitus_ai": {"available": True},
        },
    }
    monkeypatch.setattr(
        "context_loom.deployment.available_memory_bytes", lambda: 16 * 1024**3
    )
    receipt = run_deployment_doctor(
        chat_model="granite4.2:8b",
        embedding_model="qwen3-embedding:0.6b",
        ollama_url="http://127.0.0.1:11436",
        context_tokens=4096,
        kv_cache_type="f16",
        ollama_server_pid=None,
        models=DoctorGranite(),
        dependency_probe=lambda: dependencies,
    )
    output = tmp_path / "doctor.json"
    write_doctor_receipt(output, receipt)
    assert receipt["status"] == "ready"
    assert receipt["entrypoint"] == "context_loom.habitus_agent.HabitusLoomAgent.handle"
    assert receipt["models"]["chat"] == "granite4.2:8b@digest"
    assert receipt["model_memory_preflight"]["status"] == "calculated"
    assert receipt["requested"]["generation_token_limit"] is None
    assert receipt["requested"]["action_budget"] is None
    assert output.read_text().endswith("\n")


def test_deployment_doctor_names_a_missing_dependency_blocker() -> None:
    receipt = run_deployment_doctor(
        chat_model="granite4.2:8b",
        embedding_model="qwen3-embedding:0.6b",
        ollama_url="http://127.0.0.1:11436",
        context_tokens=None,
        kv_cache_type="f16",
        ollama_server_pid=None,
        dependency_probe=lambda: {
            "ready": False,
            "packages": {
                "numpy": {"available": True},
                "torch": {"available": False},
                "habitus_ai": {"available": True},
            },
        },
    )
    assert receipt["status"] == "blocked"
    assert receipt["blocker_stage"] == "python_environment"
    assert receipt["error_type"] == "RuntimeError"
    assert "torch" in receipt["error"]


def test_product_cli_does_not_advertise_retracted_two_case_harness() -> None:
    help_text = parser().format_help()
    assert "long-mem-benchmark" not in help_text
    assert "qa-benchmark" not in help_text
    assert "{demo,gestate,chat,doctor,locomo-retrieval-diagnostic,channel,credentials}" in help_text

