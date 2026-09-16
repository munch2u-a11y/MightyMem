"""Experimental scratch-born retention LM, separate from the GRU lineages.

Implements rotary, decayed matrix retention in a local rotating basis. Only
fixed-size matrices persist: no token keys/values, transcript or position cache.
Parallel chunks and single-symbol recurrence express the same causal operator.
This small configuration is not a reproduction of RetNet's large-model results.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

from .developmental_cortex import CortexLineageError, DIRECTION_IDS, require_torch, torch
from .language_vocabulary import LanguageVocabulary
from .stateful_language import StatefulConversation, language_model_sha256

SCHEMA = "habitus.retentive-language.v1"


@dataclass(frozen=True)
class RetentionConfig:
    width: int = 256
    layers: int = 6
    heads: int = 4
    value_factor: int = 2
    feedforward_factor: int = 2
    chunk_symbols: int = 128
    shortest_decay_scale: float = 32.0
    longest_decay_scale: float = 512.0
    maximum_event_bytes: int = 2048
    learned_stop_threshold: float = 0.8
    gradient_clip: float = 1.0
    weight_decay: float = 0.05
    seed: int = 2081
    # Adapter shape only. This core does not consume any graph or sensor field.
    graph_width: int = 16

    def __post_init__(self):
        for key in ("width", "layers", "heads", "value_factor", "feedforward_factor", "chunk_symbols", "maximum_event_bytes"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.width % self.heads or (self.width // self.heads) % 2:
            raise ValueError("each retention head needs an even key dimension")
        if not (math.isfinite(self.shortest_decay_scale) and math.isfinite(self.longest_decay_scale)
                and 1 < self.shortest_decay_scale <= self.longest_decay_scale):
            raise ValueError("invalid decay scales")
        if not 0.5 < self.learned_stop_threshold < 1 or self.maximum_event_bytes < 2:
            raise ValueError("invalid language event limits")
        if self.graph_width != 16 or type(self.seed) is not int:
            raise ValueError("invalid adapter width or random seed")
        if not (math.isfinite(self.gradient_clip) and self.gradient_clip > 0
                and math.isfinite(self.weight_decay) and self.weight_decay >= 0):
            raise ValueError("invalid optimization bounds")


def rotate_pairs(value, phase):
    """Rotate the last dimension in real two-dimensional planes."""
    even, odd = value[..., 0::2], value[..., 1::2]
    cosine, sine = phase.cos(), phase.sin()
    return torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), dim=-1).flatten(-2)


def retention(q, k, v, previous, decay, angles, lengths=None):
    """Causal chunk operator; inputs [batch, heads, symbols, features].

    The saved state is expressed at the current token's rotary phase, so the
    recurrence rotates old matrix rows by -theta before inserting a new outer
    product. There is no ever-growing absolute position or phase counter.
    Padding is not an event and cannot advance the returned numeric state.
    """
    batch, heads, length, dimension = q.shape
    if lengths is None:
        lengths = torch.full((batch,), length, dtype=torch.long, device=q.device)
    if length == 1:
        shifted = rotate_pairs(previous.transpose(-1, -2), -angles).transpose(-1, -2)
        following = decay[None, :, None, None] * shifted + k[..., 0, :, None] * v[..., 0, None, :]
        valid = lengths[:, None, None, None] > 0
        following = torch.where(valid, following, previous)
        output = (q[..., 0, None, :] @ following) / math.sqrt(dimension)
        return output * valid, following
    positions = torch.arange(1, length + 1, device=q.device, dtype=q.dtype)
    phase = positions[:, None] * angles
    qr = rotate_pairs(q, phase)
    kr = rotate_pairs(k, phase)
    distance = positions[:, None] - positions[None, :]
    causal = distance >= 0
    mask = decay[:, None, None].pow(distance.clamp(min=0)) * causal
    valid = positions[None, :] <= lengths[:, None]
    scores = (qr @ kr.transpose(-1, -2)) * mask[None, :, :, :]
    scores = scores * valid[:, None, None, :]
    cross_decay = decay[:, None].pow(positions)
    output = (scores @ v + (qr @ previous) * cross_decay[None, :, :, None]) / math.sqrt(dimension)
    output = output * valid[:, None, :, None]
    remaining = (lengths[:, None] - positions[None, :]).clamp(min=0)
    tail_weights = decay[None, :, None].pow(remaining[:, None, :]) * valid[:, None, :]
    final_global = (kr.transpose(-1, -2) @ (v * tail_weights[..., None])
                    + previous * decay[None, :].pow(lengths[:, None])[..., None, None])
    final_phase = -lengths[:, None, None, None].to(q.dtype) * angles
    following = rotate_pairs(final_global.transpose(-1, -2), final_phase).transpose(-1, -2)
    return output, following


if torch is not None:
    class RetentionBlock(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            width, value_width = config.width, config.width * config.value_factor
            dimension = width // config.heads
            self.input_norm = torch.nn.LayerNorm(width)
            self.qkv = torch.nn.Linear(width, 2 * width + value_width, bias=False)
            self.gate = torch.nn.Linear(width, value_width, bias=False)
            self.output = torch.nn.Linear(value_width, width, bias=False)
            self.head_scale = torch.nn.Parameter(torch.ones(config.heads, dimension * config.value_factor))
            self.head_bias = torch.nn.Parameter(torch.zeros(config.heads, dimension * config.value_factor))
            self.ffn_norm = torch.nn.LayerNorm(width)
            self.ffn_in = torch.nn.Linear(width, width * config.feedforward_factor, bias=False)
            self.ffn_out = torch.nn.Linear(width * config.feedforward_factor, width, bias=False)
            scales = torch.linspace(math.log(config.shortest_decay_scale), math.log(config.longest_decay_scale), config.heads)
            self.register_buffer("decay", 1 - torch.exp(-scales))
            self.register_buffer("angles", 10000.0 ** (-torch.arange(0, dimension, 2).float() / dimension))

        def forward(self, value, previous, lengths):
            batch, length, width = value.shape
            heads = self.config.heads
            source = self.input_norm(value)
            q, k, v = self.qkv(source).split((width, width, width * self.config.value_factor), dim=-1)
            q, k, v = (x.reshape(batch, length, heads, -1).transpose(1, 2) for x in (q, k, v))
            outputs, state = [], previous
            # Matrix accumulation stays float32 under CPU bfloat16 training;
            # float64 is retained for independent mathematical tests.
            compute_dtype = torch.float64 if value.dtype == torch.float64 else torch.float32
            with torch.autocast(device_type="cpu", enabled=False):
                q, k, v = (x.to(compute_dtype) for x in (q, k, v))
                for start in range(0, length, self.config.chunk_symbols):
                    end = min(start + self.config.chunk_symbols, length)
                    output, state = retention(q[:, :, start:end], k[:, :, start:end], v[:, :, start:end], state,
                        self.decay.to(compute_dtype), self.angles.to(compute_dtype), (lengths - start).clamp(0, end - start))
                    outputs.append(output)
                retained = torch.cat(outputs, dim=2).transpose(1, 2)
                retained = torch.nn.functional.layer_norm(retained, (retained.shape[-1],))
                retained = retained * self.head_scale + self.head_bias
            retained = retained.reshape(batch, length, -1).to(source.dtype)
            value = value + self.output(torch.nn.functional.silu(self.gate(source)) * retained)
            value = value + self.ffn_out(torch.nn.functional.gelu(self.ffn_in(self.ffn_norm(value))))
            return value, state


    class RetentiveLanguageCore(torch.nn.Module):
        def __init__(self, config, vocabulary):
            super().__init__()
            self.config, self.vocabulary = config, vocabulary
            self.byte_embedding = torch.nn.Embedding(len(vocabulary), config.width)
            self.boundary_embedding = torch.nn.Parameter(torch.empty(config.width))
            self.direction_embedding = torch.nn.Embedding(len(DIRECTION_IDS), config.width)
            self.blocks = torch.nn.ModuleList(RetentionBlock(config) for _ in range(config.layers))
            self.final_norm = torch.nn.LayerNorm(config.width)
            self.output_bias = torch.nn.Parameter(torch.zeros(len(vocabulary)))
            self.stop_head = torch.nn.Linear(config.width, 1)
            identity = json.dumps({"schema": SCHEMA, "config": asdict(config)}, sort_keys=True).encode()
            self.register_buffer("architecture_identity", torch.tensor(list(hashlib.sha256(identity).digest()), dtype=torch.uint8))
            for module in self.modules():
                if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
                    torch.nn.init.normal_(module.weight, std=0.02)
                    if getattr(module, "bias", None) is not None:
                        torch.nn.init.zeros_(module.bias)
            torch.nn.init.normal_(self.boundary_embedding, std=0.02)
            for block in self.blocks:
                torch.nn.init.normal_(block.output.weight, std=0.02 / math.sqrt(2 * config.layers))
                torch.nn.init.normal_(block.ffn_out.weight, std=0.02 / math.sqrt(2 * config.layers))

        def initial_hidden(self, batch_size, *, device, dtype):
            dimension = self.config.width // self.config.heads
            return torch.zeros(self.config.layers, batch_size,
                self.config.heads * dimension * dimension * self.config.value_factor, device=device, dtype=dtype)

        def forward(self, previous_bytes, graph_field_tensor, direction_ids, hidden=None, *, sequence_lengths=None):
            if previous_bytes.ndim != 2 or previous_bytes.shape[1] < 1:
                raise ValueError("symbols must have nonempty [batch, sequence] shape")
            if previous_bytes.device.type != "cpu":
                raise ValueError("this experimental runner is CPU-only")
            if torch.any(previous_bytes < -1) or torch.any(previous_bytes >= len(self.vocabulary)):
                raise ValueError("symbol ID is outside the vocabulary")
            batch, length = previous_bytes.shape
            if graph_field_tensor.shape != (batch, self.config.graph_width) or torch.any(graph_field_tensor != 0):
                raise ValueError("retention language core requires an empty numeric field")
            if direction_ids.shape == (batch,):
                direction_ids = direction_ids[:, None].expand(batch, length)
            if direction_ids.shape != (batch, length) or torch.any(direction_ids < 0) or torch.any(direction_ids >= len(DIRECTION_IDS)):
                raise ValueError("invalid language direction IDs")
            if sequence_lengths is None:
                sequence_lengths = torch.full((batch,), length, dtype=torch.long, device=previous_bytes.device)
            if (sequence_lengths.shape != (batch,) or sequence_lengths.dtype != torch.long
                    or torch.any(sequence_lengths < 1) or torch.any(sequence_lengths > length)):
                raise ValueError("invalid sequence lengths")
            if hidden is None:
                hidden = self.initial_hidden(batch, device=previous_bytes.device, dtype=self.byte_embedding.weight.dtype)
            expected = (self.config.layers, batch, self.config.heads * (self.config.width // self.config.heads) ** 2 * self.config.value_factor)
            if hidden.shape != expected or not torch.isfinite(hidden).all():
                raise ValueError("invalid retained matrix state")
            lexical = self.byte_embedding(previous_bytes.clamp(min=0))
            value = torch.where((previous_bytes < 0)[..., None], self.boundary_embedding, lexical)
            value = value + self.direction_embedding(direction_ids)
            states = []
            for index, block in enumerate(self.blocks):
                prior = hidden[index].reshape(batch, self.config.heads, self.config.width // self.config.heads, -1)
                value, following = block(value, prior, sequence_lengths)
                states.append(following.reshape(batch, -1))
            value = self.final_norm(value)
            return {"byte_logits": torch.nn.functional.linear(value, self.byte_embedding.weight, self.output_bias),
                    "stop_logits": self.stop_head(value).squeeze(-1)}, torch.stack(states)


def new_retentive_core(config, vocabulary):
    require_torch()
    caller_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            model = RetentiveLanguageCore(config, vocabulary)
    finally:
        torch.set_num_threads(caller_threads)
    model.language_initialization = {"threads": 2, "torch_version": str(torch.__version__)}
    model.language_initial_sha256 = language_model_sha256(model)
    return model


def save_retentive_checkpoint(path, model, *, training):
    if not all(torch.isfinite(v).all() for v in model.state_dict().values()):
        raise CortexLineageError("non-finite retentive checkpoint")
    payload = {"schema": SCHEMA, "config": asdict(model.config), "vocabulary": model.vocabulary.payload(),
        "weights": model.state_dict(), "initial_model_sha256": model.language_initial_sha256,
        "model_sha256": language_model_sha256(model), "initialization": model.language_initialization,
        "pretrained_model_loaded": False, "training": training}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        torch.save(payload, stream)
    return payload["model_sha256"]


def load_retentive_checkpoint(path):
    require_torch()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != SCHEMA or payload.get("pretrained_model_loaded") is not False:
        raise CortexLineageError("not a scratch-born retentive checkpoint")
    model = new_retentive_core(RetentionConfig(**payload["config"]), LanguageVocabulary.from_payload(payload["vocabulary"]))
    if model.language_initial_sha256 != payload.get("initial_model_sha256"):
        raise CortexLineageError("retentive random-birth identity mismatch")
    model.load_state_dict(payload["weights"], strict=True)
    if not all(torch.isfinite(v).all() for v in model.state_dict().values()):
        raise CortexLineageError("non-finite retentive checkpoint")
    if language_model_sha256(model) != payload.get("model_sha256"):
        raise CortexLineageError("retentive weight identity mismatch")
    return model.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--once", required=True)
    parser.add_argument("--maximum-output-bytes", type=int, default=None)
    args = parser.parse_args()
    require_torch()
    torch.set_num_threads(2)
    model = load_retentive_checkpoint(args.checkpoint)
    with StatefulConversation(model, state_path=args.state) as conversation:
        receipt = conversation.reply(args.once, maximum_bytes=args.maximum_output_bytes)
    print(json.dumps({"event": "language", "architecture": SCHEMA, "receipt": asdict(receipt)}, sort_keys=True))


if __name__ == "__main__":
    main()
