"""Experimental numeric continuation through a corpus-trained Habitus graph.

The saved native graph supplies conditional sensory routes. A bounded recurrent
activation field follows observed tokens, including private computation. Private
tokens never update corpus counts, become source documents, or earn rewards.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import struct

import numpy as np

from .corpus_associations import canonical, read_graph, sha
from .rwkv7_language import file_sha256


MAGIC = b"HABCORPUS1\0"


class CorpusStreamBackend:
    def __init__(self, backend, graph_path, codes, *, gain=0.005, retention=0.9):
        if not math.isfinite(gain) or not 0 <= gain <= 0.25 or not 0 <= retention < 1:
            raise ValueError("invalid corpus feedback gain or retention")
        manifest, centers, probabilities = read_graph(graph_path)
        self.base = backend
        self.codes = np.array(codes, dtype=np.uint16, copy=True)
        self.levels = tuple(manifest["levels"])
        if (self.codes.ndim != 2 or self.codes.shape[1] != len(self.levels)
                or manifest["dimension"] != backend.embedding_dimension
                or any(np.any(self.codes[:, bank] >= level) for bank, level in enumerate(self.levels))):
            raise ValueError("corpus readout geometry mismatch")
        self.routes = [probability @ center for probability, center in zip(probabilities, centers)]
        self.activity = [np.zeros(level, dtype=np.float64) for level in self.levels]
        self.graph_sha256 = file_sha256(Path(graph_path))
        self.identity = "habitus-corpus-stream-v1:" + sha(canonical({
            "base": backend.identity, "graph": self.graph_sha256,
            "codes": sha(self.codes.tobytes()), "gain": gain, "retention": retention}))
        self.gain, self.retention = float(gain), float(retention)
        self.eos_token_id = backend.eos_token_id
        self.feedback_enabled = True
        self.inputs = self.external_tokens = self.private_tokens = self.feedback_steps = 0

    def encode(self, text):
        return self.base.encode(text)

    def decode(self, ids):
        return self.base.decode(ids)

    def consume(self, ids):
        return self.base.consume(ids)

    def _observe(self, ids):
        for token in ids:
            if not 0 <= token < len(self.codes):
                raise ValueError("token outside frozen vocabulary")
            for bank, activity in enumerate(self.activity):
                activity *= self.retention
                activity[self.codes[token, bank]] += 1 - self.retention

    def admit_input(self, message):
        ids = self.encode(message)
        self._observe(ids)
        self.inputs += 1
        self.external_tokens += len(ids)

    def internal_opportunity(self):
        # Scheduling alone does not manufacture tokens or change the field.
        pass

    def feedback(self):
        vector = sum(activity @ routes for activity, routes in zip(self.activity, self.routes)) / len(self.levels)
        # Convex routes have norm <= 1; absorb numerical round-off only.
        return (vector / max(1.0, float(np.linalg.norm(vector)))).tolist()

    def consume_generated(self, token, *, private):
        if token == self.eos_token_id:
            return self.base.consume([token])
        self._observe([token])
        if not private:
            return self.base.consume([token])
        self.private_tokens += 1
        gain = self.gain if self.feedback_enabled else 0.0
        if not gain:
            return self.base.consume([token])
        field = self.feedback()
        result = self.base.consume_with_feedback(token, field, gain=gain)
        if any(field):
            self.feedback_steps += 1
        return result

    def finish_generation(self):
        pass

    def numeric_state(self):
        return {"activity": [value.tolist() for value in self.activity], "inputs": self.inputs,
            "external_tokens": self.external_tokens, "private_tokens": self.private_tokens,
            "feedback_steps": self.feedback_steps, "feedback_enabled": self.feedback_enabled}

    def dump_state(self):
        language = self.base.dump_state()
        metadata = canonical({"identity": self.identity, "numeric": self.numeric_state(),
                              "language_sha256": sha(language)})
        return MAGIC + struct.pack(">Q", len(metadata)) + metadata + language

    def load_state(self, data):
        offset = len(MAGIC) + 8
        if not data.startswith(MAGIC) or len(data) <= offset:
            raise ValueError("invalid corpus stream checkpoint")
        length = struct.unpack(">Q", data[len(MAGIC):offset])[0]
        if not 0 < length < len(data) - offset:
            raise ValueError("invalid corpus checkpoint length")
        metadata = json.loads(data[offset:offset + length])
        numeric = metadata["numeric"]
        language = data[offset + length:]
        if (metadata["identity"] != self.identity or sha(language) != metadata["language_sha256"]
                or set(numeric) != set(self.numeric_state())
                or not isinstance(numeric["feedback_enabled"], bool)
                or any(type(numeric[key]) is not int or numeric[key] < 0 for key in (
                    "inputs", "external_tokens", "private_tokens", "feedback_steps"))
                or numeric["feedback_steps"] > numeric["private_tokens"]):
            raise ValueError("corpus checkpoint identity or counters mismatch")
        activity = [np.asarray(row, dtype=np.float64) for row in numeric["activity"]]
        if len(activity) != len(self.levels) or any(
                row.shape != (level,) or not np.isfinite(row).all() or np.any(row < 0)
                or row.sum() > 1.000000001 for row, level in zip(activity, self.levels)):
            raise ValueError("invalid corpus activation field")
        before = self.base.dump_state()
        try:
            self.base.load_state(language)
        except BaseException:
            self.base.load_state(before)
            raise
        for key, value in numeric.items():
            setattr(self, key, activity if key == "activity" else value)

    def memory_statistics(self):
        field = self.feedback()
        return {"graph_sha256": self.graph_sha256, "numeric_sha256": sha(canonical(self.numeric_state())),
            "field_sha256": sha(canonical(field)), "field_norm": float(np.linalg.norm(field)),
            "inputs": self.inputs, "private_tokens": self.private_tokens, "feedback_steps": self.feedback_steps,
            "corpus_learning_during_generation": False, "reward_events": 0}
