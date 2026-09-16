"""Trainable RWKV language inside the developmental Habitus runtime.

The pinned pretrained embedding and lower blocks retain their weights. Native
RWKV tail blocks, the native 18M cortex and a bidirectional numeric interface
learn together. The imported RWKV block implementation performs the recurrence;
this module does not substitute an association predictor for language training.
"""
from __future__ import annotations

from contextlib import nullcontext
import hashlib
import io
import json
import math
from pathlib import Path
import time

from .developmental_cortex import ACTION_IDS, DIRECTION_IDS, pulse_field
from .habitus_stream import _NumericSpace, unit
from .rwkv7_language import Rwkv7Backend, state_tensors
from .types import OutputTrunk


class RecurrentEmbeddingSpace(_NumericSpace):
    pretrained_geometry = True
    def __init__(self, backend, *, space_id=None):
        super().__init__(backend.embedding_dimension, backend.identity)
        self.backend = backend
        if space_id is not None:
            self.space_id = space_id

    def embed(self, text):
        ids = self.backend.encode(str(text))
        if not ids:
            return [0.0] * self.dimension
        # All bytes contribute; chunking only bounds embedding allocation.
        torch = self.backend.torch
        value = torch.zeros(self.dimension)
        with torch.no_grad():
            for start in range(0, len(ids), 2048):
                value += self.backend.model.rwkv7.emb(torch.tensor(ids[start:start + 2048])).float().sum(0)
        return unit(value.tolist())


def tensor_digest(tensors):
    import torch
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class ContinualRwkv(Rwkv7Backend):
    """One language state, with causal truncated BPTT and attributable updates."""

    def __init__(self, directory, *, trainable_blocks=1, chunk_tokens=128):
        super().__init__(Path(directory), device=None)
        if not 1 <= trainable_blocks <= len(self.model.rwkv7.blocks):
            raise ValueError("invalid number of trainable RWKV blocks")
        if not 2 <= chunk_tokens <= 256:
            raise ValueError("training chunk must contain 2 to 256 tokens")
        self.chunk_tokens = chunk_tokens
        self.first_trainable = len(self.model.rwkv7.blocks) - trainable_blocks
        # The audited reference includes a differentiable, torch-only chunked
        # recurrence. Batch matrix operations also avoid tokenwise CPU matmuls.
        self.config.wkv_implementation = "chunked"
        for block in self.model.rwkv7.blocks[self.first_trainable:]:
            block.float().requires_grad_(True)
        self.model.rwkv7.ln_out.float().requires_grad_(True)
        for layer in self.state.layers[self.first_trainable:]:
            for slot in (self.state.ATT_SHIFT, self.state.FFN_SHIFT):
                layer.recurrent_states[slot] = layer.recurrent_states[slot].float()
        self.identity += f":continual-v1:chunked:tail-{trainable_blocks}:chunk-{chunk_tokens}"
        self.runtime = None
        self.optimizer = None
        self.interface = None
        self.field = None
        self.public_candidate = None
        self.replay_graph = None
        self.source = "external_language"
        self.training_enabled = True
        self.sampling = {"temperature": 1.0, "top_p": 0.5, "seed": 20260910}
        self.sampling_generator = self.torch.Generator(device="cpu").manual_seed(self.sampling["seed"])
        self.counters = {"updates": 0, "language_training_tokens": 0, "corpus_training_tokens": 0,
                         "interaction_training_tokens": 0, "reinforced_tokens": 0,
                         "feedback_updates": 0, "action_updates": 0, "generated_tokens": 0}
        self.receipt_sink = lambda receipt: None

    def attach(self, runtime):
        torch = self.torch
        self.runtime = runtime
        cortex = runtime.cortex
        cortex.continuous_hidden = cortex._latest_hidden(cortex.connection).detach().clone()
        width = cortex.config.graph_width
        # Both matrices are learned through the actual language objective.
        self.interface = torch.nn.ModuleDict({
            "language_to_cortex": torch.nn.Linear(self.embedding_dimension, width, bias=False),
            "graph_to_cortex": torch.nn.Linear(self.embedding_dimension, width, bias=False),
            "cortex_to_language": torch.nn.Linear(cortex.config.hidden_width + width, self.embedding_dimension),
        })
        with torch.no_grad():
            self.interface["cortex_to_language"].weight.mul_(0.02)
            self.interface["cortex_to_language"].bias.zero_()
        self.optimizer = torch.optim.AdamW([
            {"params": list(self.trainable_rwkv().values()), "lr": 1e-5},
            {"params": list(cortex.model.parameters()), "lr": 1e-4},
            {"params": list(self.interface.parameters()), "lr": 3e-4},
        ], weight_decay=0.001, foreach=False)
        self.begin_slice()

    def trainable_rwkv(self):
        return {name: value for name, value in self.model.named_parameters() if value.requires_grad}

    def _graph_vectors(self):
        if self.replay_graph is not None:
            return self.replay_graph
        mind = self.runtime.mind
        snapshot = mind.recurrent.snapshot(pulse=mind.pulse)
        vector = pulse_field(snapshot, (), width=self.runtime.cortex.config.graph_width)
        weighted = self.torch.zeros(self.embedding_dimension)
        total = 0.0
        for state in sorted(snapshot.node_states, key=lambda row: -row.activation):
            if state.activation <= 0:
                continue
            node = mind.store.get_concept(state.node_id)
            if node is not None and len(node.embedding) == self.embedding_dimension:
                weighted += self.torch.tensor(node.embedding) * state.activation
                total += state.activation
        return self.torch.tensor([vector]), (weighted / max(1.0, total)).unsqueeze(0)

    def _cognitive_step(self):
        torch = self.torch
        graph, semantic = self._graph_vectors()
        context = torch.tensor([self.stream_context()])
        context = torch.nn.functional.normalize(context, dim=-1)
        field = graph + 0.1 * torch.tanh(self.interface["language_to_cortex"](context)
                                         + self.interface["graph_to_cortex"](semantic))
        cortex = self.runtime.cortex
        outputs, hidden = cortex.model(torch.tensor([[-1]]), field,
            torch.tensor([DIRECTION_IDS["quiet"]]), cortex.continuous_hidden)
        cortex.continuous_hidden = hidden
        residual = self.interface["cortex_to_language"](torch.cat((hidden[-1], field), dim=-1))
        residual = torch.tanh(residual)
        residual = residual / residual.norm(dim=-1, keepdim=True).clamp_min(1.0)
        self.field = 0.02 * residual
        return outputs

    def begin_slice(self):
        with self.torch.no_grad():
            self._cognitive_step()
        self.detach_state()

    def detach_state(self):
        for layer in self.state.layers:
            for slot, value in layer.recurrent_states.items():
                layer.recurrent_states[slot] = value.detach()
        if self.runtime is not None:
            self.runtime.cortex.continuous_hidden = self.runtime.cortex.continuous_hidden.detach()
        if self.field is not None:
            self.field = self.field.detach()

    def _forward(self, ids, *, learning=False):
        torch = self.torch
        if not ids or any(not 0 <= token < self.config.vocab_size for token in ids):
            raise ValueError("invalid RWKV token sequence")
        self.config.wkv_implementation = "eager" if len(ids) == 1 else "chunked"
        with nullcontext() if learning else torch.no_grad():
            # Lower blocks have no gradient-bearing input or parameters.
            with torch.no_grad():
                hidden = self.model.rwkv7.emb(torch.tensor([ids]))
                first = None
                for block in self.model.rwkv7.blocks[:self.first_trainable]:
                    hidden, first, self.state = block(hidden, first, self.state)
            hidden = hidden.float()
            if self.field is not None:
                hidden = hidden + hidden.detach().norm(dim=-1, keepdim=True) * self.field.unsqueeze(1)
            first = first.float() if first is not None else None
            for block in self.model.rwkv7.blocks[self.first_trainable:]:
                hidden, first, self.state = block(hidden, first, self.state)
            hidden = self.model.rwkv7.ln_out(hidden)
            if not learning:
                hidden = hidden[:, -1:]
            logits = torch.nn.functional.linear(hidden.to(self.model.head.weight.dtype), self.model.head.weight).float()
            if not torch.isfinite(logits).all():
                raise ValueError("nonfinite integrated RWKV logits")
            return logits

    def consume(self, ids):
        result = None
        for start in range(0, len(ids), self.chunk_tokens):
            result = self._forward(ids[start:start + self.chunk_tokens])
        if result is None:
            raise ValueError("empty input")
        return self.sample(result[0, -1])

    def sample(self, logits):
        """Use the pinned model's documented nucleus decoding, with saved RNG."""
        torch = self.torch
        probabilities = torch.softmax(logits.detach() / self.sampling["temperature"], dim=-1)
        probabilities, indices = torch.sort(probabilities, descending=True)
        # Keep the token that crosses the nucleus threshold, including when a
        # single token already has more probability than top_p.
        outside = probabilities.cumsum(0) - probabilities >= self.sampling["top_p"]
        probabilities.masked_fill_(outside, 0.0)
        choice = torch.multinomial(probabilities, 1, generator=self.sampling_generator)
        return int(indices[choice].item())

    def consume_generated(self, token, *, private):
        if not private:
            if self.public_candidate is None:
                self.public_candidate = {"state": self.dump_state(), "tokens": [],
                                         "graph": self._graph_vectors()}
            self.public_candidate["tokens"].append(token)
        self.counters["generated_tokens"] += 1
        return self.consume([token])

    def consume_input(self, ids):
        if not self.training_enabled:
            return self.consume(ids)
        return self.learn_tokens(ids, source=self.source)

    def learn_tokens(self, ids, *, source, next_token=None):
        """Consume each current token once; targets are subsequent external tokens.

        Call before sensory admission of this material. The graph/cortex field
        can therefore carry prior experience, never the not-yet-read targets.
        """
        if not ids:
            raise ValueError("empty learning event")
        torch = self.torch
        result = None
        for start in range(0, len(ids), self.chunk_tokens):
            inputs = ids[start:start + self.chunk_tokens]
            targets = ids[start + 1:start + len(inputs) + 1]
            if len(targets) < len(inputs) and next_token is not None:
                targets.append(next_token)
            if not targets:
                result = self.consume(inputs)
                continue
            began = time.monotonic()
            self.optimizer.zero_grad(set_to_none=True)
            self._cognitive_step()
            logits = self._forward(inputs, learning=True)
            loss = torch.nn.functional.cross_entropy(logits[0, :len(targets)], torch.tensor(targets))
            receipt = self._update(loss, {"source": source, "tokens": len(targets),
                "payload_token_sha256": hashlib.sha256(json.dumps(inputs).encode()).hexdigest(),
                "language_loss": float(loss.detach()), "elapsed_seconds": time.monotonic() - began})
            receipt["elapsed_seconds"] = time.monotonic() - began
            self.counters["language_training_tokens"] += len(targets)
            key = "corpus_training_tokens" if source.startswith("corpus:") else "interaction_training_tokens"
            self.counters[key] += len(targets)
            self.receipt_sink({**receipt, "counters": dict(self.counters)})
            result = int(logits[0, -1].detach().argmax())
            # No replay after the optimizer step. This cursor was produced just
            # before the update, as in ordinary online truncated BPTT.
        return result

    def _update(self, loss, evidence, *, accumulated=False):
        torch = self.torch
        if not torch.isfinite(loss):
            raise ValueError("nonfinite joint training loss")
        if not accumulated:
            loss.backward()
        groups = {"rwkv": list(self.trainable_rwkv().values()),
                  "cortex": list(self.runtime.cortex.model.parameters()),
                  "interface": list(self.interface.parameters())}
        norms = {}
        for name, parameters in groups.items():
            norm = torch.nn.utils.clip_grad_norm_(parameters, 0.5, error_if_nonfinite=True)
            norms[name] = float(norm)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.runtime.cortex._model_sha256_cache = None
        self.detach_state()
        self.counters["updates"] += 1
        return {"update": self.counters["updates"], "kind": "joint_gradient_update",
                "gradient_norms": norms, **evidence}

    def reinforce(self, eligibility, reward, *, evidence):
        if not math.isfinite(reward) or not -1 <= reward <= 1 or reward == 0:
            raise ValueError("feedback must be a nonzero finite value in [-1,1]")
        if eligibility is None or len(eligibility["tokens"]) < 2:
            return {"applied": False, "reason": "no delivered language eligibility"}
        current = self.dump_state()
        torch = self.torch
        losses = []
        try:
            self.load_state(eligibility["state"])
            self.replay_graph = eligibility["graph"]
            ids = eligibility["tokens"]
            self.optimizer.zero_grad(set_to_none=True)
            objective = 0.0
            for start in range(0, len(ids) - 1, self.chunk_tokens):
                chunk = ids[start:start + self.chunk_tokens]
                targets = ids[start + 1:start + len(chunk) + 1]
                self._cognitive_step()
                logits = self._forward(chunk, learning=True)
                nll = torch.nn.functional.cross_entropy(logits[0, :len(targets)], torch.tensor(targets))
                weighted = 0.2 * reward * nll * (len(targets) / (len(ids) - 1))
                if not torch.isfinite(weighted):
                    raise ValueError("nonfinite feedback objective")
                weighted.backward()
                objective += float(weighted.detach())
                losses.append({"tokens": len(targets), "nll": float(nll.detach())})
                self.detach_state()
            receipt = self._update(torch.tensor(objective), {"source": "explicit_teacher_feedback",
                "reward": reward, "evidence": evidence, "tokens": len(ids) - 1,
                "feedback_loss": objective, "training_replay": True, "external_input_admissions": 0,
                "one_update_per_assessment": True, "chunks": losses}, accumulated=True)
            self.counters["reinforced_tokens"] += len(ids) - 1
            self.receipt_sink(receipt)
            self.counters["feedback_updates"] += 1
        finally:
            # Learning updates remain; the ongoing state never travels backward
            # to the credited utterance or re-admits its text as a current input.
            self.load_state(current)
            self.replay_graph = None
        return {"applied": True, "updates": losses, "reward": reward}

    def learn_action(self, trunk, *, success, receipt_id, eligibility=None):
        torch = self.torch
        current = self.dump_state()
        if eligibility is not None:
            self.load_state(eligibility["state"])
            self.replay_graph = eligibility["graph"]
        self.optimizer.zero_grad(set_to_none=True)
        outputs = self._cognitive_step()
        target = torch.tensor([ACTION_IDS[OutputTrunk(trunk)]])
        loss = torch.nn.functional.cross_entropy(outputs["action_logits"], target)
        loss = loss + torch.nn.functional.mse_loss(outputs["consequence"], torch.tensor([1.0 if success else -1.0]))
        result = self._update(loss, {"source": "verified_tool_receipt", "receipt_id": receipt_id,
            "action": str(trunk), "success": bool(success), "action_loss": float(loss.detach())})
        self.counters["action_updates"] += 1
        self.load_state(current)
        self.replay_graph = None
        self.receipt_sink(result)
        return result

    def dump_state(self):
        torch = self.torch
        stream = io.BytesIO()
        torch.save({"language": {key: tensor.detach().clone() for key, tensor in state_tensors(self.state).items()},
                    "cortex": None if self.runtime is None else self.runtime.cortex.continuous_hidden.detach().clone(),
                    "field": None if self.field is None else self.field.detach().clone(),
                    "sampling_rng": self.sampling_generator.get_state()}, stream)
        return stream.getvalue()

    def load_state(self, data):
        payload = self.torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        targets = state_tensors(self.state)
        if set(payload["language"]) != set(targets):
            raise ValueError("joint language state slots differ")
        for key, value in payload["language"].items():
            allowed = (self.torch.float32,) if int(key.rsplit(".", 1)[1]) == self.state.WKV else (self.torch.bfloat16, self.torch.float32)
            if value.shape != targets[key].shape or value.dtype not in allowed or not self.torch.isfinite(value).all():
                raise ValueError(f"invalid joint language state: {key}, {value.shape}, {value.dtype}")
        if self.runtime is not None:
            hidden = payload["cortex"]
            if hidden is None or hidden.shape != self.runtime.cortex.continuous_hidden.shape or not self.torch.isfinite(hidden).all():
                raise ValueError("invalid joint cortex state")
            self.runtime.cortex.continuous_hidden = hidden.detach().clone()
        with self.torch.no_grad():
            for key, value in payload["language"].items():
                # Training can promote a native shift cache to float32. Restore
                # its saved precision rather than casting into a fresh bf16 slot.
                _, layer, _, slot = key.split(".")
                self.state.layers[int(layer)].recurrent_states[int(slot)] = value.detach().clone()
        self.field = payload["field"]
        if "sampling_rng" in payload:
            self.sampling_generator.set_state(payload["sampling_rng"])

    def weight_report(self):
        groups = {"rwkv": self.trainable_rwkv(), "cortex": self.runtime.cortex.model.state_dict(),
                  "interface": self.interface.state_dict()}
        return {name: {"parameters": sum(value.numel() for value in tensors.values()),
                       "sha256": tensor_digest(tensors)} for name, tensors in groups.items()}

    def checkpoint_payload(self):
        return {"identity": self.identity, "rwkv": self.trainable_rwkv(),
                "cortex": self.runtime.cortex.model.state_dict(), "interface": self.interface.state_dict(),
                "optimizer": self.optimizer.state_dict(), "numeric": self.dump_state(),
                "counters": dict(self.counters), "sampling": dict(self.sampling)}

    def restore_payload(self, payload):
        if payload["identity"] != self.identity or set(payload["rwkv"]) != set(self.trainable_rwkv()):
            raise ValueError("joint model checkpoint identity mismatch")
        with self.torch.no_grad():
            for name, parameter in self.trainable_rwkv().items():
                parameter.copy_(payload["rwkv"][name])
        self.runtime.cortex.model.load_state_dict(payload["cortex"], strict=True)
        self.interface.load_state_dict(payload["interface"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer"])
        self.load_state(payload["numeric"])
        self.counters = dict(payload["counters"])
        if "sampling" in payload:
            self.sampling = dict(payload["sampling"])
        self.runtime.cortex._model_sha256_cache = None
