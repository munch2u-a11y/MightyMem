"""Opt-in frozen pretrained RWKV-7 backend; not the scratch-trained cortex.

Only the explicitly pinned, locally captured checkpoint is supported. No Hub
loader, chat history, text retrieval, generation wrapper, or optimized custom
kernel is used. Dependencies and GPU access are needed only on construction.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import types

MODEL_ID = "RWKV/RWKV7-G1j-1.5B-20260831"
REVISION = "2c18b29ab7fbece25ff6112281eea0fa41fcb30f"
AUDITED_CODE = {
    "configuration_rwkv7": "14050c631e27d66cd2f52ca5ec53b62eef04e602c4f1fc4e3696c328bc8e200e",
    "modeling_rwkv7": "28fc373673b2c98805275cc55381db0dd1dd9325ec0c1b34264d49ca1bc16958",
    "tokenization_rwkv7": "58406b0aa5f51d5ca3a698602825ae9b299f6621e6c20bb0a2915d43c9f8bf65",
}
SHARD_SHA256 = {
    "model-00001-of-00004.safetensors": "7a1bea185d7e9ab1a271eb0da8a26f9a005a92860c8af42a0bc3d27ac7142a1d",
    "model-00002-of-00004.safetensors": "6c168e242175ae5e5b881ff354330d3fb80a9b616e6be02a69d2a656d04f3f43",
    "model-00003-of-00004.safetensors": "ce2ba2dcaef2af7490c320080ebbb546cb38311d51c44bd48b29a6310e4869a4",
    "model-00004-of-00004.safetensors": "04135fbdff8ebb1dc86328e953f0b740800d6725e727c5de7b4a863640f98806",
}
ASSET_SHA256 = {
    "config.json": "95f3524181116fca466ce5b2e8c827dcf1a026ebfb4967d3ad28af9feee96d1f",
    "tokenizer.json": "1c0bc8e5f355de60a4587a1e9abb9aa4ef65508ff59641344e7f9f10f76feacf",
    "model.safetensors.index.json": "9466d25dd34ea227d5056b599ec7afa80a22c645ffc93aceb3f61262a3520d8a",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checkpoint(directory: Path, *, weights: bool = True) -> None:
    expected = {f"{name}.py": sha for name, sha in AUDITED_CODE.items()} | ASSET_SHA256
    if weights:
        expected.update(SHARD_SHA256)
    for name, sha in expected.items():
        if file_sha256(directory / name) != sha:
            raise ValueError(f"pinned checkpoint checksum mismatch: {name}")


def _import_audited(directory: Path):
    package_name = "_habitus_audited_rwkv7_2c18b29"
    package = types.ModuleType(package_name)
    package.__path__ = [str(directory)]
    sys.modules[package_name] = package
    modules = {}
    for name in AUDITED_CODE:
        qualified = f"{package_name}.{name}"
        spec = importlib.util.spec_from_file_location(qualified, directory / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        spec.loader.exec_module(module)
        modules[name] = module
    return modules


def state_tensors(state) -> dict:
    """RWKV uses integer-keyed slot dictionaries, not a tensor sequence."""
    result = {}
    for index, layer in enumerate(state.layers):
        if set(layer.recurrent_states) != {0, 1, 2}:
            raise ValueError("RWKV requires a matrix and both shift slots")
        for slot in range(3):
            tensor = layer.recurrent_states[slot]
            if tensor is None:
                raise ValueError("RWKV state slot is unallocated")
            result[f"layer.{index:02d}.slot.{slot}"] = tensor
    return result


class Rwkv7Backend:
    eos_token_id = 0

    def __init__(
        self,
        directory: Path,
        *,
        device: str | None = None,
        gpu_memory_gib: float | None = None,
    ):
        began = time.monotonic()
        directory = directory.resolve()
        verify_checkpoint(directory)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        import torch
        import transformers
        from safetensors import safe_open
        from transformers.initialization import no_init_weights

        self.torch = torch
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if self.device.type == "cuda" and gpu_memory_gib is not None:
            if gpu_memory_gib <= 0:
                raise ValueError("an explicit GPU allocation must be positive")
            total = torch.cuda.get_device_properties(self.device).total_memory
            torch.cuda.set_per_process_memory_fraction(min(gpu_memory_gib * 1024**3 / total, 1.0), self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        modules = _import_audited(directory)
        config = modules["configuration_rwkv7"].Rwkv7Config.from_dict(json.loads((directory / "config.json").read_text()))
        config.wkv_implementation = "eager"
        config.wkv_state_dtype = "float32"
        model_type = modules["modeling_rwkv7"].Rwkv7ForCausalLM
        # Do not create a second full model in host RAM. Materialize each verified
        # tensor directly on the requested device, one at a time, from mmap shards.
        with no_init_weights(), torch.device("meta"):
            self.model = model_type(config)
        index = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]
        expected = self.model.state_dict()
        if set(index) != set(expected) or set(index.values()) != set(SHARD_SHA256):
            raise ValueError("weight names do not exactly match the pinned architecture")
        for shard in SHARD_SHA256:
            with safe_open(directory / shard, framework="pt", device="cpu") as source:
                keys = set(source.keys())
                if keys != {key for key, value in index.items() if value == shard}:
                    raise ValueError("shard keys differ from index")
                for key in sorted(keys):
                    tensor = source.get_tensor(key)
                    if tensor.shape != expected[key].shape or tensor.dtype != torch.bfloat16:
                        raise ValueError(f"unexpected shape or dtype: {key}")
                    tensor = tensor.to(self.device)
                    module_path, leaf = key.rsplit(".", 1)
                    module = self.model.get_submodule(module_path)
                    if leaf in module._parameters:
                        setattr(module, leaf, torch.nn.Parameter(tensor, requires_grad=False))
                    else:
                        module._buffers[leaf] = tensor
            print(json.dumps({"loaded_shard": shard}), flush=True)
        if any(t.is_meta for t in self.model.state_dict().values()):
            raise ValueError("unmaterialized model tensor")
        self.model.eval().requires_grad_(False)
        self.config = config
        self.tokenizer = modules["tokenization_rwkv7"].Rwkv7Tokenizer(tokenizer_file=str(directory / "tokenizer.json"))
        self.state = self.model.rwkv7.allocate_state(1, self.device, torch.bfloat16)
        self.identity = f"{MODEL_ID}@{REVISION}:eager:bf16:state-fp32:torch-{torch.__version__}:transformers-{transformers.__version__}"
        self.statistics = {"model_id": MODEL_ID, "revision": REVISION,
                           "pretrained": True, "weight_tensor_elements": sum(t.numel() for t in self.model.state_dict().values()),
                           "torch_version": str(torch.__version__), "transformers_version": transformers.__version__,
                           "device": str(self.device), "load_seconds": time.monotonic() - began,
                           "wkv_implementation": "eager", "gpu_allocator_cap_gib": gpu_memory_gib if self.device.type == "cuda" else None}

    def encode(self, text: str) -> list[int]:
        # Treat event content as bytes, including any literal special-token text.
        return self.tokenizer._encode_bytes(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, clean_up_tokenization_spaces=False)

    def consume(self, ids: list[int]) -> int:
        if not ids:
            raise ValueError("cannot consume an empty token sequence")
        if any(not 0 <= token < self.config.vocab_size for token in ids):
            raise ValueError("invalid token event")
        with self.torch.inference_mode():
            result = self.model(input_ids=self.torch.tensor([ids], device=self.device),
                                state=self.state, use_cache=True, logits_to_keep=1)
            logits = result.logits[0, -1]
            if not self.torch.isfinite(logits).all():
                raise ValueError("nonfinite model logits")
            self.state = result.state
            return int(logits.argmax().item())

    @property
    def embedding_dimension(self) -> int:
        return self.config.hidden_size

    def stream_context(self) -> list[float]:
        """Read the final block's contextual attention input without advancing."""
        attention, _, _ = self.state.read(len(self.state.layers) - 1)
        return attention[0].detach().float().cpu().tolist()

    def stream_embedding(self, ids: list[int]) -> list[float]:
        """Pool actual token embeddings, without another recurrent model step."""
        if not ids or any(not 0 <= token < self.config.vocab_size for token in ids):
            raise ValueError("invalid embedding token event")
        with self.torch.inference_mode():
            total = self.torch.zeros(
                self.embedding_dimension, device=self.device, dtype=self.torch.float32
            )
            for start in range(0, len(ids), 4096):
                rows = self.model.rwkv7.emb(
                    self.torch.tensor(ids[start:start + 4096], device=self.device)
                )
                total += rows.float().sum(dim=0)
            return (total / len(ids)).cpu().tolist()

    def consume_with_feedback(self, token: int, feedback: list[float], *, gain: float) -> int:
        """Add a bounded numeric residual to one existing token's embedding.

        No extra token, weight update, or prompt is introduced. A zero gain or
        zero field takes the original consume path exactly.
        """
        import math
        if (len(feedback) != self.embedding_dimension or not 0 <= gain <= 0.25
                or not all(math.isfinite(value) for value in feedback)
                or not 0 <= token < self.config.vocab_size):
            raise ValueError("invalid numeric stream feedback")
        if gain == 0 or not any(feedback):
            return self.consume([token])
        with self.torch.inference_mode():
            row = self.model.rwkv7.emb(self.torch.tensor([[token]], device=self.device))
            field = self.torch.tensor(feedback, device=self.device, dtype=self.torch.float32)
            field = field / field.norm().clamp_min(1.0)
            embedded = (row.float() + gain * row.float().norm() * field).to(row.dtype)
            result = self.model(inputs_embeds=embedded, state=self.state,
                                use_cache=True, logits_to_keep=1)
            logits = result.logits[0, -1]
            if not self.torch.isfinite(logits).all():
                raise ValueError("nonfinite coupled model logits")
            self.state = result.state
            return int(logits.argmax().item())

    def dump_state(self) -> bytes:
        from safetensors.torch import save
        tensors = {key: tensor.detach().cpu().contiguous()
                   for key, tensor in state_tensors(self.state).items()}
        if not all(self.torch.isfinite(tensor).all() for tensor in tensors.values()):
            raise ValueError("nonfinite recurrent state")
        return save(tensors)

    def load_state(self, data: bytes) -> None:
        from safetensors.torch import load
        tensors = load(data)
        targets = state_tensors(self.state)
        if set(tensors) != set(targets):
            raise ValueError("recurrent state slots mismatch")
        for key, tensor in tensors.items():
            if tensor.shape != targets[key].shape or tensor.dtype != targets[key].dtype:
                raise ValueError("recurrent state shape or dtype mismatch")
            if not self.torch.isfinite(tensor).all():
                raise ValueError("nonfinite saved recurrent state")
        with self.torch.inference_mode():
            for key, tensor in tensors.items():
                targets[key].copy_(tensor.to(self.device))

    def memory_statistics(self) -> dict:
        tensors = list(state_tensors(self.state).values())
        result = {"numeric_state_bytes": sum(t.numel() * t.element_size() for t in tensors),
                  "numeric_state_tensor_count": len(tensors)}
        if self.device.type == "cuda":
            result.update(peak_gpu_allocated_bytes=self.torch.cuda.max_memory_allocated(self.device),
                          peak_gpu_reserved_bytes=self.torch.cuda.max_memory_reserved(self.device),
                          device_name=self.torch.cuda.get_device_name(self.device))
        return result
