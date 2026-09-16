"""Experimental numeric connection between recurrent language and Habitus.

Real inputs grow Habitus sensory fibers in the frozen model's input-embedding
space. Private token activity excites those fibers through the existing SELF
internal-output API. The active numeric field feeds back into private token
embeddings. Neither canonical record text nor private language is retrieved.
The born-in cortex and external action/speech authorization are separate paths.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import sqlite3
import struct

from .graph import SENSORY_FEATURE_KIND
from .pipeline import BaseAgenticMemoryRAG
from .self_pulse import AdmissionEffect, SelfPulseKernel
from .types import InputTrunk


SCHEMA = "habitus-embedding-stream-v1"
MAGIC = b"HABSTREAM1\0"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encode_json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def unit(vector) -> list[float]:
    values = [float(value) for value in vector]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("nonfinite stream vector")
    norm = math.sqrt(sum(value * value for value in values))
    return [value / norm for value in values] if norm > 1e-12 else [0.0] * len(values)


@dataclass(frozen=True)
class StreamBridgeConfig:
    gain: float = 0.05
    interval: int = 32
    feature_banks: int = 4
    signature_bits: int = 6

    def __post_init__(self):
        if (not 0 <= self.gain <= 0.25 or not 1 <= self.interval <= 256
                or not 1 <= self.feature_banks <= 6 or not 1 <= self.signature_bits <= 8):
            raise ValueError("invalid stream bridge configuration")


class _NumericSpace:
    def __init__(self, dimension, model_identity):
        self.dimension = int(dimension)
        self.space_id = f"{SCHEMA}:{sha(model_identity.encode())}:{dimension}"

    def embed(self, text):
        # Seed labels are not language inputs. Actual events always provide
        # their numeric embedding explicitly.
        return [0.0] * self.dimension


class HabitusStream:
    """An isolated Habitus lineage, saved together with the language state.

    SQLite lives in memory during processing. Complete checkpoints include its
    graph, recurrent field, canonical admission hashes, and internal receipts.
    A user database is never modified or cleared by an intervention.
    """

    def __init__(self, dimension: int, model_identity: str, *, config=None):
        self.config = config or StreamBridgeConfig()
        self.dimension = int(dimension)
        self.identity = f"{SCHEMA}:{model_identity}:{sha(encode_json(asdict(self.config)))}"
        self.mind = BaseAgenticMemoryRAG(":memory:", embedder=_NumericSpace(dimension, model_identity))
        # Audit timestamps must not change the numeric result across restart.
        self.mind.graph.recency_strength = 0.0
        self.kernel = SelfPulseKernel(self.mind)
        self.mind.store.set_metadata("stream_identity", self.identity)
        self.pending = [0.0] * self.dimension
        self.pending_count = 0
        self.inputs = self.private_tokens = self.internal_updates = self.feedback_steps = 0

    def close(self):
        self.kernel.close()
        self.mind.close()

    def _features(self):
        return self.mind.store.list_concepts(kind=SENSORY_FEATURE_KIND)

    def hear(self, vector, message_sha256: str):
        self.flush()
        vector = unit(vector)
        if len(vector) != self.dimension:
            raise ValueError("stream input dimension mismatch")
        self.inputs += 1
        bucket = self.kernel.enqueue_input(
            f"external-input-sha256:{message_sha256}", lane=InputTrunk.HEAR,
            source_id="external_language", embedding=vector, token_cost=1,
            item_id=f"stream-input:{self.inputs}", allow_growth=False,
            metadata={"origin": "external_language", "message_sha256": message_sha256,
                      "external_action_verified": False, "numeric_stream_input": True},
        )

        def grow(record, admitted, created):
            growth = self.mind.graph.grow_trunk_breadth(
                record, input_trunk=InputTrunk.HEAR, pulse=self.mind.pulse,
                sensory_rows=(vector,), feature_banks=self.config.feature_banks,
                signature_bits=self.config.signature_bits,
            )
            return AdmissionEffect(excitation={node: 1.0 for node in growth.feature_node_ids},
                                   context_node_ids=growth.feature_node_ids)

        self.kernel.advance_cycle(item_ids=(bucket.item_id,), admission_observer=grow,
                                  output_candidate_provider=lambda *_: ())

    def internal_opportunity(self):
        self.flush()
        self.kernel.advance_cycle(advance_when_idle=True, output_candidate_provider=lambda *_: ())

    def observe_private(self, vector):
        values = unit(vector)
        if len(values) != self.dimension:
            raise ValueError("private stream dimension mismatch")
        self.pending = [left + right for left, right in zip(self.pending, values)]
        self.pending_count += 1
        self.private_tokens += 1
        if self.pending_count >= self.config.interval:
            self.flush()

    def flush(self):
        if not self.pending_count:
            return
        vector = unit(self.pending)
        scored = [(max(0.0, sum(a * b for a, b in zip(vector, node.embedding))) ** 2,
                   node.concept_id) for node in self._features()]
        # Private associations activate learned geometry; they never create
        # external records, strengthen edges, or supply positive outcome credit.
        excitation = {
            node: score
            for score, node in sorted(scored, key=lambda item: (-item[0], item[1]))
            if score > 0
        }
        # A private compute window is also elapsed Habitus time. This gives the
        # existing field its normal decay and propagation without an input.
        self.kernel.advance_cycle(advance_when_idle=True, output_candidate_provider=lambda *_: ())
        self.kernel.advance_internal_output(
            excitation=excitation, kind="model_private_numeric",
            metadata={"origin": "model_generated_private_unverified",
                      "tokens": self.pending_count, "vector_sha256": sha(encode_json(vector)),
                      "external_evidence": False, "verified_external_action": False},
        )
        self.internal_updates += 1
        self.pending = [0.0] * self.dimension
        self.pending_count = 0

    def feedback(self) -> list[float]:
        states = {state.node_id: state.activation for state in self.mind.store.list_node_dynamics()}
        features = self._features()
        weights = [max(0.0, states.get(node.concept_id, 0.0)) for node in features]
        total = max(1.0, sum(weights))
        # Convex combination: norm cannot exceed one, even with many active
        # nodes. Low activity retains its low amplitude instead of being amplified.
        return [sum(weight * node.embedding[index] for node, weight in zip(features, weights)) / total
                for index in range(self.dimension)]

    def state(self) -> dict:
        """All bridge computation state, excluding audit UUIDs and wall clocks."""
        graph = self.mind.graph
        return {"identity": self.identity, "pulse": self.mind.pulse,
                "nodes": [asdict(node) for node in self._features()],
                "edges": [{key: value for key, value in asdict(edge).items() if key != "last_active_time"}
                          for edge in self.mind.store.list_edges()],
                "recurrent": self.mind.recurrent.snapshot(pulse=self.mind.pulse).state_sha256,
                "pending": self.pending, "pending_count": self.pending_count,
                "inputs": self.inputs, "private_tokens": self.private_tokens,
                "internal_updates": self.internal_updates, "feedback_steps": self.feedback_steps,
                "feedback": self.feedback(), "graph_invariants": graph.validate_invariants()}

    def summary(self) -> dict:
        state = self.state()
        connection = self.mind.store.connection
        return {"computational_sha256": sha(encode_json(state)),
                "recurrent_sha256": state["recurrent"],
                "feedback_sha256": sha(encode_json(state["feedback"])),
                "feedback_norm": math.sqrt(sum(value * value for value in state["feedback"])),
                "features": len(state["nodes"]), "edges": len(state["edges"]),
                "inputs": self.inputs, "private_tokens": self.private_tokens,
                "internal_updates": self.internal_updates, "feedback_steps": self.feedback_steps,
                "pending_tokens": self.pending_count,
                "canonical_records": connection.execute("SELECT COUNT(*) FROM records").fetchone()[0],
                "internal_receipts": connection.execute("SELECT COUNT(*) FROM self_internal_output_events").fetchone()[0],
                "experience_cycles": connection.execute("SELECT COUNT(*) FROM experience_cycles").fetchone()[0],
                "graph_invariants": state["graph_invariants"]}

    def checkpoint(self) -> tuple[bytes, dict]:
        numeric = {"identity": self.identity, "pending": self.pending, "pending_count": self.pending_count,
                   "inputs": self.inputs, "private_tokens": self.private_tokens,
                   "internal_updates": self.internal_updates, "feedback_steps": self.feedback_steps}
        return self.mind.store.connection.serialize(), json.loads(encode_json(numeric))

    def restore(self, database: bytes, numeric: dict):
        expected = set(self.checkpoint()[1])
        if set(numeric) != expected or numeric["identity"] != self.identity:
            raise ValueError("Habitus stream checkpoint identity mismatch")
        if (len(numeric["pending"]) != self.dimension
                or not all(math.isfinite(value) for value in numeric["pending"])
                or any(not isinstance(numeric[key], int) or numeric[key] < 0
                       for key in ("pending_count", "inputs", "private_tokens", "internal_updates", "feedback_steps"))
                or numeric["pending_count"] >= self.config.interval):
            raise ValueError("invalid Habitus stream numeric checkpoint")
        check = sqlite3.connect(":memory:")
        try:
            check.deserialize(database)
            identity = check.execute("SELECT value FROM metadata WHERE key='stream_identity'").fetchone()
            if identity != (self.identity,) or check.execute("PRAGMA quick_check").fetchone() != ("ok",):
                raise ValueError("Habitus stream database identity or integrity mismatch")
            if check.execute("PRAGMA foreign_key_check").fetchall():
                raise ValueError("Habitus stream database foreign key mismatch")
        finally:
            check.close()
        self.mind.store.connection.deserialize(database)
        self.mind.pulse = int(self.mind.store.get_metadata("pulse_counter", "0"))
        for key, value in numeric.items():
            if key != "identity":
                setattr(self, key, value)


class HabitusStreamBackend:
    """Origin-aware backend for RecurrentFlow, with joint rollback/checkpoints."""

    def __init__(self, backend, *, config=None):
        self.base = backend
        self.bridge = HabitusStream(backend.embedding_dimension, backend.identity, config=config)
        self.identity = self.bridge.identity
        self.eos_token_id = backend.eos_token_id
        self.feedback_enabled = True

    def encode(self, text):
        return self.base.encode(text)

    def decode(self, ids):
        return self.base.decode(ids)

    def consume(self, ids):
        # Actual external input and protocol framing always use the base path.
        return self.base.consume(ids)

    def admit_input(self, message):
        self.bridge.hear(self.base.stream_embedding(self.encode(message)), sha(message.encode()))

    def internal_opportunity(self):
        self.bridge.internal_opportunity()

    def consume_generated(self, token, *, private):
        if not private or token == self.eos_token_id:
            self.bridge.flush()
            return self.base.consume([token])
        feedback = self.bridge.feedback()
        gain = self.bridge.config.gain if self.feedback_enabled else 0.0
        result = self.base.consume_with_feedback(token, feedback, gain=gain)
        if gain and any(feedback):
            self.bridge.feedback_steps += 1
        self.bridge.observe_private(self.base.stream_embedding([token]))
        return result

    def finish_generation(self):
        self.bridge.flush()

    def dump_state(self) -> bytes:
        language = self.base.dump_state()
        database, numeric = self.bridge.checkpoint()
        metadata = encode_json({"bridge": numeric, "feedback_enabled": self.feedback_enabled,
                                "language_sha256": sha(language),
                                "database_sha256": sha(database)})
        return MAGIC + struct.pack(">QQ", len(metadata), len(language)) + metadata + language + database

    def load_state(self, data: bytes):
        if not data.startswith(MAGIC) or len(data) < len(MAGIC) + 16:
            raise ValueError("invalid coupled checkpoint header")
        offset = len(MAGIC) + 16
        meta_length, language_length = struct.unpack(">QQ", data[len(MAGIC):offset])
        if not 0 < meta_length < len(data) - offset or language_length > len(data) - offset - meta_length:
            raise ValueError("invalid coupled checkpoint lengths")
        metadata = json.loads(data[offset:offset + meta_length])
        if not isinstance(metadata.get("feedback_enabled"), bool):
            raise ValueError("invalid coupled feedback switch")
        offset += meta_length
        language = data[offset:offset + language_length]
        database = data[offset + language_length:]
        if sha(language) != metadata["language_sha256"] or sha(database) != metadata["database_sha256"]:
            raise ValueError("coupled checkpoint checksum mismatch")
        before_language = self.base.dump_state()
        before_bridge = self.bridge.checkpoint()
        try:
            self.bridge.restore(database, metadata["bridge"])
            self.base.load_state(language)
            self.feedback_enabled = metadata["feedback_enabled"]
        except BaseException:
            self.bridge.restore(*before_bridge)
            self.base.load_state(before_language)
            raise

    def memory_statistics(self):
        return {**self.base.memory_statistics(), "habitus": self.bridge.summary(),
                "habitus_database_bytes": len(self.bridge.checkpoint()[0])}

    def close(self):
        self.bridge.close()
