"""Language cores for the constant-flow Habitus mind.

A recurrent language model supplies private thought and public speech. The
born-in cortex runs beside it as a smaller recurrent tissue. On every flow tick
the cortex reads the live graph field and the language model's current
context, advances its own hidden state, and writes one bounded residual into
the language model's middle layers, below the blocks that still have to
process it. Language prediction, verified action consequences, explicit
teacher feedback and prediction of the next sensory event train the language
tail, the cortex and their numeric interface together.

``FlowRwkv`` couples the pinned pretrained RWKV-7 checkpoint.
``ByteRecurrentCore`` is a small randomly initialized byte model with the same
coupling. It exists for behavioral tests and CPU wiring checks and is not a
useful language model.
"""
from __future__ import annotations

from contextlib import nullcontext
import hashlib
import io
import json
import math
from pathlib import Path
import time

from .continual_rwkv import tensor_digest
from .developmental_cortex import ACTION_IDS, DIRECTION_IDS, pulse_field
from .rwkv7_language import Rwkv7Backend, state_tensors
from .types import OutputTrunk


EXPECTATION = "expectation"


class CoupledCore:
    """Joint language/cortex learning shared by every flow language core.

    Subclasses supply token geometry, a forward pass split at the residual
    injection point, recurrent language state and trainable language
    parameters. Every optimizer update, replay and checkpoint is common.
    """

    eos_token_id = 0

    def _init_coupling(self, *, chunk_tokens, residual_scale, sampling):
        if not 2 <= int(chunk_tokens) <= 256:
            raise ValueError("training chunk must contain 2 to 256 tokens")
        if not math.isfinite(residual_scale) or not 0.0 <= residual_scale <= 0.25:
            raise ValueError("cortex residual scale must be within [0, 0.25]")
        self.chunk_tokens = int(chunk_tokens)
        self.residual_scale = float(residual_scale)
        self.runtime = None
        self.interface = None
        self.optimizer = None
        self.field = None
        self.public_candidate = None
        self.replay_graph = None
        self.source = "external_language"
        self.training_enabled = True
        self.sampling = dict(sampling)
        self.sampling_generator = self.torch.Generator(device="cpu").manual_seed(int(self.sampling["seed"]))
        self.counters = {"updates": 0, "language_training_tokens": 0, "reinforced_tokens": 0,
                         "feedback_updates": 0, "action_updates": 0, "expectation_updates": 0,
                         "generated_tokens": 0}
        self.receipt_sink = lambda receipt: None

    # ------------------------------------------------------------ coupling

    def attach(self, runtime):
        torch = self.torch
        cortex = runtime.cortex
        if torch.device(cortex.device).type != self.device.type:
            raise ValueError("the cortex and the language core must run on one device")
        self.runtime = runtime
        cortex.continuous_hidden = cortex._latest_hidden(cortex.connection).detach().clone()
        width = cortex.config.graph_width
        joint = cortex.config.hidden_width + width
        dimension = self.embedding_dimension
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(self.sampling["seed"]))
            self.interface = torch.nn.ModuleDict({
                "language_to_cortex": torch.nn.Linear(dimension, width, bias=False),
                "graph_to_cortex": torch.nn.Linear(dimension, width, bias=False),
                "cortex_to_language": torch.nn.Linear(joint, dimension),
                EXPECTATION: torch.nn.Linear(joint, dimension),
            }).to(self.device)
        with torch.no_grad():
            self.interface["cortex_to_language"].weight.mul_(0.02)
            self.interface["cortex_to_language"].bias.zero_()
        self.optimizer = torch.optim.AdamW([
            {"params": list(self.trainable_language().values()), "lr": 1e-5},
            {"params": list(cortex.model.parameters()), "lr": 1e-4},
            {"params": list(self.interface.parameters()), "lr": 3e-4},
        ], weight_decay=0.001, foreach=False)
        self.begin_slice()

    def _graph_vectors(self):
        torch = self.torch
        if self.replay_graph is not None:
            return tuple(value.to(self.device) for value in self.replay_graph)
        mind = self.runtime.mind
        snapshot = mind.recurrent.snapshot(pulse=mind.pulse)
        field = pulse_field(snapshot, (), width=self.runtime.cortex.config.graph_width)
        weighted = torch.zeros(self.embedding_dimension)
        total = 0.0
        for state in snapshot.node_states:
            if state.activation <= 0:
                continue
            node = mind.store.get_concept(state.node_id)
            if node is not None and len(node.embedding) == self.embedding_dimension and any(node.embedding):
                weighted += torch.tensor(node.embedding) * state.activation
                total += state.activation
        return (torch.tensor([field], device=self.device),
                (weighted / max(1.0, total)).unsqueeze(0).to(self.device))

    def _cognitive_step(self):
        """Advance the cortex once and derive the residual it writes into language."""
        torch = self.torch
        graph, semantic = self._graph_vectors()
        context = torch.tensor([self.stream_context()], device=self.device, dtype=torch.float32)
        context = torch.nn.functional.normalize(context, dim=-1)
        field = graph + 0.1 * torch.tanh(self.interface["language_to_cortex"](context)
                                         + self.interface["graph_to_cortex"](semantic))
        cortex = self.runtime.cortex
        outputs, hidden = cortex.model(torch.tensor([[-1]], device=self.device), field,
                                       torch.tensor([DIRECTION_IDS["quiet"]], device=self.device),
                                       cortex.continuous_hidden)
        cortex.continuous_hidden = hidden
        joint = torch.cat((hidden[-1], field), dim=-1)
        residual = torch.tanh(self.interface["cortex_to_language"](joint))
        residual = residual / residual.norm(dim=-1, keepdim=True).clamp_min(1.0)
        self.field = self.residual_scale * residual
        return outputs, joint

    def begin_slice(self):
        with self.torch.no_grad():
            self._cognitive_step()
        self.detach_state()

    def detach_state(self):
        self._detach_language()
        if self.runtime is not None:
            self.runtime.cortex.continuous_hidden = self.runtime.cortex.continuous_hidden.detach()
        if self.field is not None:
            self.field = self.field.detach()

    # ------------------------------------------------------------- language

    def _forward(self, ids, *, learning=False):
        torch = self.torch
        if not ids or any(not 0 <= int(token) < self.vocabulary_size for token in ids):
            raise ValueError("invalid language token sequence")
        with nullcontext() if learning else torch.no_grad():
            hidden, carry = self._below(ids)
            if self.field is not None and self.residual_scale > 0:
                # Relative to each row's own norm: the cortex can steer, not replace, language.
                injected = hidden.detach().float().norm(dim=-1, keepdim=True) * self.field.unsqueeze(1)
                hidden = hidden + injected.to(hidden.dtype)
            hidden = self._above(hidden, carry)
            if not learning:
                hidden = hidden[:, -1:]
            logits = self._head(hidden)
            if not torch.isfinite(logits).all():
                raise ValueError("nonfinite coupled language logits")
            return logits

    def consume(self, ids):
        result = None
        for start in range(0, len(ids), self.chunk_tokens):
            result = self._forward(list(ids[start:start + self.chunk_tokens]))
        if result is None:
            raise ValueError("empty input")
        return self.sample(result[0, -1])

    def sample(self, logits):
        """Nucleus decoding with a checkpointed generator."""
        torch = self.torch
        probabilities = torch.softmax(logits.detach().float().cpu() / self.sampling["temperature"], dim=-1)
        probabilities, indices = torch.sort(probabilities, descending=True)
        outside = probabilities.cumsum(0) - probabilities >= self.sampling["top_p"]
        probabilities.masked_fill_(outside, 0.0)
        choice = torch.multinomial(probabilities, 1, generator=self.sampling_generator)
        return int(indices[choice].item())

    def consume_generated(self, token, *, private):
        if not private:
            if self.public_candidate is None:
                self.public_candidate = {"state": self.dump_state(), "tokens": [],
                                         "graph": tuple(value.detach().cpu() for value in self._graph_vectors())}
            self.public_candidate["tokens"].append(int(token))
        self.counters["generated_tokens"] += 1
        return self.consume([int(token)])

    def consume_input(self, ids):
        if not self.training_enabled:
            return self.consume(ids)
        return self.learn_tokens(list(ids), source=self.source)

    def learn_tokens(self, ids, *, source, next_token=None):
        """Consume each external token once and train on the ones that follow.

        Call before the same material reaches graph/cortex admission, so the
        residual field carries prior experience and never its own targets.
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
            loss = torch.nn.functional.cross_entropy(logits[0, :len(targets)],
                                                     torch.tensor(targets, device=logits.device))
            receipt = self._update(loss, {"source": source, "tokens": len(targets),
                "payload_token_sha256": hashlib.sha256(json.dumps(inputs).encode()).hexdigest(),
                "language_loss": float(loss.detach())})
            receipt["elapsed_seconds"] = time.monotonic() - began
            self.counters["language_training_tokens"] += len(targets)
            self.receipt_sink({**receipt, "counters": dict(self.counters)})
            # Learning consumes no decoding randomness; callers ignore this cursor.
            result = int(logits[0, -1].detach().argmax())
        return result

    def _update(self, loss, evidence, *, accumulated=False):
        torch = self.torch
        if not torch.isfinite(loss):
            raise ValueError("nonfinite joint training loss")
        if not accumulated:
            loss.backward()
        groups = {"language": list(self.trainable_language().values()),
                  "cortex": list(self.runtime.cortex.model.parameters()),
                  "interface": list(self.interface.parameters())}
        norms = {name: float(torch.nn.utils.clip_grad_norm_(parameters, 0.5, error_if_nonfinite=True))
                 for name, parameters in groups.items()}
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.runtime.cortex._model_sha256_cache = None
        self.detach_state()
        self.counters["updates"] += 1
        return {"update": self.counters["updates"], "kind": "joint_gradient_update",
                "gradient_norms": norms, **evidence}

    # ------------------------------------------------------------- learning

    def reinforce(self, eligibility, reward, *, evidence):
        if not math.isfinite(reward) or not -1 <= reward <= 1 or reward == 0:
            raise ValueError("feedback must be a nonzero finite value in [-1,1]")
        if eligibility is None or len(eligibility["tokens"]) < 2:
            return {"applied": False, "reason": "no delivered language eligibility"}
        torch = self.torch
        current = self.dump_state()
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
                nll = torch.nn.functional.cross_entropy(logits[0, :len(targets)],
                                                        torch.tensor(targets, device=logits.device))
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
                "chunks": losses}, accumulated=True)
            self.counters["reinforced_tokens"] += len(ids) - 1
            self.counters["feedback_updates"] += 1
            self.receipt_sink(receipt)
        finally:
            # The ongoing stream never travels back to the credited utterance.
            self.load_state(current)
            self.replay_graph = None
        return {"applied": True, "updates": losses, "reward": reward}

    def learn_action(self, trunk, *, success, receipt_id, eligibility=None):
        """Train the cortex's action and consequence heads from one verified return."""
        torch = self.torch
        current = self.dump_state()
        try:
            if eligibility is not None:
                self.load_state(eligibility["state"])
                self.replay_graph = eligibility["graph"]
            self.optimizer.zero_grad(set_to_none=True)
            outputs, _ = self._cognitive_step()
            target = torch.tensor([ACTION_IDS[OutputTrunk(trunk)]], device=self.device)
            loss = torch.nn.functional.cross_entropy(outputs["action_logits"], target)
            loss = loss + torch.nn.functional.mse_loss(
                outputs["consequence"], torch.tensor([1.0 if success else -1.0], device=self.device))
            result = self._update(loss, {"source": "verified_tool_receipt", "receipt_id": receipt_id,
                "action": OutputTrunk(trunk).value, "success": bool(success), "action_loss": float(loss.detach())})
            self.counters["action_updates"] += 1
            self.receipt_sink(result)
        finally:
            self.load_state(current)
            self.replay_graph = None
        return result

    def learn_expectation(self, vectors, *, evidence, eligibility=None):
        """Score and learn the cortex's prediction of the sensation about to arrive.

        The prediction uses only what preceded the sensation: either the
        current state before admission, or an ``eligibility`` state captured
        before the action whose return is being predicted. Surprise is one
        minus the cosine between the prediction and the event's token geometry.
        """
        torch = self.torch
        rows = [list(vector) for vector in vectors if any(vector)]
        if not rows:
            return None
        target = torch.nn.functional.normalize(
            torch.tensor(rows, dtype=torch.float32, device=self.device).mean(0), dim=0)
        current = self.dump_state()
        try:
            if eligibility is not None:
                self.load_state(eligibility["state"])
                self.replay_graph = eligibility["graph"]
            self.optimizer.zero_grad(set_to_none=True)
            _, joint = self._cognitive_step()
            prediction = torch.nn.functional.normalize(self.interface[EXPECTATION](joint), dim=-1)[0]
            loss = 1.0 - (prediction * target).sum()
            surprise = float(loss.detach())
            receipt = self._update(loss, {"source": "sensory_expectation", "surprise": surprise, **dict(evidence)})
            self.counters["expectation_updates"] += 1
            self.receipt_sink(receipt)
        finally:
            # Prediction is a learning replay; the next real cortex step happens in the tick.
            self.load_state(current)
            self.replay_graph = None
        return {"surprise": surprise, "update": receipt["update"]}

    # ---------------------------------------------------------- persistence

    def token_geometry(self, ids):
        """Pooled pretrained token geometry, without advancing any recurrent state."""
        torch = self.torch
        if not ids:
            return [0.0] * self.embedding_dimension
        total = torch.zeros(self.embedding_dimension, device=self.device)
        with torch.no_grad():
            for start in range(0, len(ids), 2048):
                total += self._token_rows(list(ids[start:start + 2048])).sum(0)
        norm = float(total.norm())
        return [0.0] * self.embedding_dimension if norm <= 1e-12 else (total / norm).cpu().tolist()

    def dump_state(self):
        torch = self.torch
        stream = io.BytesIO()
        torch.save({"language": {key: value.detach().cpu().clone() for key, value in self._language_state().items()},
                    "cortex": None if self.runtime is None else self.runtime.cortex.continuous_hidden.detach().cpu().clone(),
                    "field": None if self.field is None else self.field.detach().cpu().clone(),
                    "sampling_rng": self.sampling_generator.get_state()}, stream)
        return stream.getvalue()

    def load_state(self, data):
        torch = self.torch
        payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        self._restore_language_state(payload["language"])
        if self.runtime is not None:
            hidden = payload["cortex"]
            expected = self.runtime.cortex.continuous_hidden
            if hidden is None or hidden.shape != expected.shape or not torch.isfinite(hidden).all():
                raise ValueError("invalid joint cortex state")
            self.runtime.cortex.continuous_hidden = hidden.detach().clone().to(self.device)
        self.field = None if payload["field"] is None else payload["field"].to(self.device)
        self.sampling_generator.set_state(payload["sampling_rng"])

    @staticmethod
    def numeric_states_equal(torch, left, right):
        first = torch.load(io.BytesIO(left), weights_only=True)
        second = torch.load(io.BytesIO(right), weights_only=True)
        if set(first["language"]) != set(second["language"]):
            return False
        pairs = [(first["language"][key], second["language"][key]) for key in first["language"]]
        pairs += [(first[key], second[key]) for key in ("cortex", "field", "sampling_rng")]
        return all((a is None and b is None) or (a is not None and b is not None and torch.equal(a, b))
                   for a, b in pairs)

    def weight_report(self):
        groups = {"language": self.trainable_language(), "cortex": self.runtime.cortex.model.state_dict(),
                  "interface": self.interface.state_dict()}
        return {name: {"parameters": sum(value.numel() for value in tensors.values()),
                       "sha256": tensor_digest(tensors)} for name, tensors in groups.items()}

    def checkpoint_payload(self):
        return {"identity": self.identity,
                "language": {name: value.detach().cpu() for name, value in self.trainable_language().items()},
                "cortex": self.runtime.cortex.model.state_dict(), "interface": self.interface.state_dict(),
                "optimizer": self.optimizer.state_dict(), "numeric": self.dump_state(),
                "counters": dict(self.counters), "sampling": dict(self.sampling)}

    def restore_payload(self, payload):
        if payload["identity"] != self.identity or set(payload["language"]) != set(self.trainable_language()):
            raise ValueError("flow core checkpoint identity mismatch")
        with self.torch.no_grad():
            for name, parameter in self.trainable_language().items():
                parameter.copy_(payload["language"][name].to(parameter.device, parameter.dtype))
        self.runtime.cortex.model.load_state_dict(payload["cortex"], strict=True)
        self.interface.load_state_dict(payload["interface"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer"])
        self.load_state(payload["numeric"])
        self.counters = dict(payload["counters"])
        self.sampling = dict(payload["sampling"])
        self.runtime.cortex._model_sha256_cache = None


class ByteRecurrentCore(CoupledCore):
    """A small random byte-level recurrent model with the complete coupling.

    For behavioral tests and CPU wiring checks only: it has no pretrained
    knowledge and does not produce useful language.
    """

    vocabulary_size = 256

    def __init__(self, *, width=32, seed=17, chunk_tokens=64, residual_scale=0.02):
        import torch
        self.torch = torch
        self.device = torch.device("cpu")
        self.width = int(width)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.embedding = torch.nn.Embedding(256, self.width)
            self.lower = torch.nn.GRU(self.width, self.width, batch_first=True)
            self.upper = torch.nn.GRU(self.width, self.width, batch_first=True)
            self.norm = torch.nn.LayerNorm(self.width)
            self.head = torch.nn.Linear(self.width, 256, bias=False)
        for module in (self.embedding, self.lower, self.head):
            module.requires_grad_(False)
        self.state = {"lower": torch.zeros(1, 1, self.width), "upper": torch.zeros(1, 1, self.width)}
        self.geometry_identity = f"byte-recurrent-core-v1:width-{self.width}:seed-{int(seed)}"
        self._init_coupling(chunk_tokens=chunk_tokens, residual_scale=residual_scale,
                            sampling={"temperature": 1.0, "top_p": 0.9, "seed": int(seed)})
        self.identity = (f"{self.geometry_identity}:flow-coupling-v1:chunk-{self.chunk_tokens}"
                         f":residual-{self.residual_scale:g}")

    @property
    def embedding_dimension(self):
        return self.width

    def encode(self, text):
        return list(str(text).encode("utf-8"))

    def decode(self, ids):
        return bytes(int(token) for token in ids).decode("utf-8", errors="replace")

    def token_bytes(self, token):
        return bytes([int(token)])

    def _token_rows(self, ids):
        return self.embedding(self.torch.tensor(ids)).float()

    def _below(self, ids):
        with self.torch.no_grad():
            hidden, self.state["lower"] = self.lower(self.embedding(self.torch.tensor([ids])), self.state["lower"])
        return hidden, None

    def _above(self, hidden, carry):
        output, self.state["upper"] = self.upper(hidden, self.state["upper"])
        return self.norm(output)

    def _head(self, hidden):
        return self.head(hidden).float()

    def _language_state(self):
        return dict(self.state)

    def _restore_language_state(self, payload):
        torch = self.torch
        if set(payload) != set(self.state):
            raise ValueError("byte core state slots differ")
        for key, value in payload.items():
            if value.shape != self.state[key].shape or not torch.isfinite(value).all():
                raise ValueError(f"invalid byte core state: {key}")
        self.state = {key: value.detach().clone() for key, value in payload.items()}

    def _detach_language(self):
        self.state = {key: value.detach() for key, value in self.state.items()}

    def stream_context(self):
        return self.state["upper"][0, 0].detach().float().tolist()

    def trainable_language(self):
        return {**{f"upper.{name}": value for name, value in self.upper.named_parameters()},
                **{f"norm.{name}": value for name, value in self.norm.named_parameters()}}


class FlowRwkv(CoupledCore, Rwkv7Backend):
    """The pinned RWKV-7 G1j model with the cortex residual entering mid-depth.

    Blocks below ``injection_block`` run without gradients. The cortex
    residual is added to the hidden stream at that block. Frozen pretrained
    blocks between it and the trainable tail still carry gradient back to the
    interface and cortex, so the pretrained layers learn to be steered without
    their weights changing. Only the last ``trainable_blocks`` blocks and the
    output normalization update.
    """

    def __init__(self, directory, *, device="cpu", trainable_blocks=1, injection_block=None,
                 chunk_tokens=128, residual_scale=0.02, gpu_memory_gib=None):
        Rwkv7Backend.__init__(self, Path(directory), device=device, gpu_memory_gib=gpu_memory_gib)
        self._configure_flow(trainable_blocks=trainable_blocks, injection_block=injection_block,
                             chunk_tokens=chunk_tokens, residual_scale=residual_scale)

    @classmethod
    def from_model(cls, model, tokenizer, *, identity, device="cpu", **options):
        """Couple an already constructed RWKV-7 model (for architecture tests)."""
        import torch
        core = cls.__new__(cls)
        core.torch = torch
        core.device = torch.device(device)
        core.model = model.to(core.device).eval().requires_grad_(False)
        core.config = model.config
        core.tokenizer = tokenizer
        core.state = model.rwkv7.allocate_state(1, core.device, torch.bfloat16)
        core.identity = identity
        core.statistics = {"pretrained": False, "weight_tensor_elements": sum(
            value.numel() for value in model.state_dict().values())}
        core._configure_flow(**options)
        return core

    def _configure_flow(self, *, trainable_blocks=1, injection_block=None, chunk_tokens=128, residual_scale=0.02):
        blocks = self.model.rwkv7.blocks
        if not 1 <= int(trainable_blocks) <= len(blocks):
            raise ValueError("invalid number of trainable RWKV blocks")
        self.first_trainable = len(blocks) - int(trainable_blocks)
        self.injection_block = self.first_trainable if injection_block is None else int(injection_block)
        if not 0 <= self.injection_block <= self.first_trainable:
            raise ValueError("the cortex residual must enter at or below the first trainable block")
        self.config.wkv_implementation = "chunked"
        for block in blocks[self.first_trainable:]:
            block.float().requires_grad_(True)
        self.model.rwkv7.ln_out.float().requires_grad_(True)
        for layer in self.state.layers[self.first_trainable:]:
            for slot in (self.state.ATT_SHIFT, self.state.FFN_SHIFT):
                layer.recurrent_states[slot] = layer.recurrent_states[slot].float()
        # Token embeddings stay pretrained; sensory geometry is independent of coupling options.
        self.geometry_identity = self.identity
        self._init_coupling(chunk_tokens=chunk_tokens, residual_scale=residual_scale,
                            sampling={"temperature": 1.0, "top_p": 0.5, "seed": 20260912})
        self.identity = (f"{self.identity}:flow-coupling-v1:tail-{int(trainable_blocks)}"
                         f":inject-{self.injection_block}:chunk-{self.chunk_tokens}:residual-{self.residual_scale:g}")

    @property
    def vocabulary_size(self):
        return self.config.vocab_size

    def token_bytes(self, token):
        return self.tokenizer._token_bytes.get(self.tokenizer.decoder[int(token)], b"")

    def _token_rows(self, ids):
        return self.model.rwkv7.emb(self.torch.tensor(ids, device=self.device)).float()

    def _below(self, ids):
        torch = self.torch
        self.config.wkv_implementation = "eager" if len(ids) == 1 else "chunked"
        with torch.no_grad():
            hidden = self.model.rwkv7.emb(torch.tensor([ids], device=self.device))
            first = None
            for block in self.model.rwkv7.blocks[:self.injection_block]:
                hidden, first, self.state = block(hidden, first, self.state)
        return hidden, first

    def _above(self, hidden, first):
        blocks = self.model.rwkv7.blocks
        for block in blocks[self.injection_block:self.first_trainable]:
            hidden, first, self.state = block(hidden, first, self.state)
        hidden = hidden.float()
        first = first.float() if first is not None else None
        for block in blocks[self.first_trainable:]:
            hidden, first, self.state = block(hidden, first, self.state)
        return self.model.rwkv7.ln_out(hidden)

    def _head(self, hidden):
        return self.torch.nn.functional.linear(hidden.to(self.model.head.weight.dtype), self.model.head.weight).float()

    def _language_state(self):
        return state_tensors(self.state)

    def _restore_language_state(self, payload):
        torch = self.torch
        targets = state_tensors(self.state)
        if set(payload) != set(targets):
            raise ValueError("joint language state slots differ")
        for key, value in payload.items():
            slot = int(key.rsplit(".", 1)[1])
            allowed = (torch.float32,) if slot == self.state.WKV else (torch.bfloat16, torch.float32)
            if value.shape != targets[key].shape or value.dtype not in allowed or not torch.isfinite(value).all():
                raise ValueError(f"invalid joint language state: {key}")
        for key, value in payload.items():
            _, layer, _, slot = key.split(".")
            # Training promotes tail shift caches to float32; keep the saved precision.
            self.state.layers[int(layer)].recurrent_states[int(slot)] = value.detach().clone().to(self.device)

    def _detach_language(self):
        for layer in self.state.layers:
            for slot, value in layer.recurrent_states.items():
                layer.recurrent_states[slot] = value.detach()

    def trainable_language(self):
        return {name: value for name, value in self.model.named_parameters() if value.requires_grad}
