"""Read-only numeric transition capture for future cortex bridge experiments."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


class StreamDataCollector:
    """Pair existing graph fields with contextual language-state vectors.

    This collector never advances a model, changes a graph, supplies rewards,
    or creates training examples asserting that generated claims are true.
    Full recurrent matrices remain in separately saved flow checkpoints.
    """

    def __init__(self):
        self.frames = []
        self.rows = {key: [] for key in ("graph_feedback", "cortex_field_256",
                    "rwkv_attention_shift", "rwkv_ffn_shift")}

    def capture(self, backend, *, kind: str, phase: str, input_number: int,
                generated_tokens: int, private_tokens_in_slice: int = 0):
        import numpy as np
        from .developmental_cortex import pulse_field

        bridge, base = backend.bridge, backend.base
        recurrent = bridge.mind.recurrent.snapshot(pulse=bridge.mind.pulse)
        contexts = tuple(node.concept_id for node in bridge._features())
        # RWKV's final block stores the actual normalized, contextual inputs to
        # its attention and feed-forward token shifts, not word lookup vectors.
        attention, ffn, _ = base.state.read(len(base.state.layers) - 1)
        values = {"graph_feedback": np.asarray(bridge.feedback(), dtype=np.float32),
                  "cortex_field_256": np.asarray(pulse_field(recurrent, contexts, width=256), dtype=np.float32),
                  "rwkv_attention_shift": attention[0].detach().float().cpu().numpy().copy(),
                  "rwkv_ffn_shift": ffn[0].detach().float().cpu().numpy().copy()}
        if any(not np.isfinite(row).all() for row in values.values()):
            raise ValueError("nonfinite numeric capture")
        for key, row in values.items():
            expected = 256 if key == "cortex_field_256" else base.embedding_dimension
            if row.shape != (expected,):
                raise ValueError("numeric capture dimension mismatch")
            self.rows[key].append(row)
        self.frames.append({"frame": len(self.frames), "kind": kind, "phase": phase,
            "input_number": input_number, "generated_tokens": generated_tokens,
            "private_tokens_in_slice": private_tokens_in_slice,
            "language_sha256": hashlib.sha256(base.dump_state()).hexdigest(),
            "habitus_computational_sha256": bridge.summary()["computational_sha256"],
            "schedule_state": backend.schedule_state(), "next_requested_gain": backend.current_gain(),
            "graph_activations": {state.node_id: state.activation for state in recurrent.node_states},
            "representation_origin": "model_and_graph_numeric_state",
            "verified_environmental_evidence": False, "verified_external_action": False})

    def save(self, directory: Path):
        import numpy as np
        if not self.frames:
            raise ValueError("cannot save an empty numeric capture")
        directory.mkdir(parents=True, exist_ok=False)
        arrays = {key: np.stack(values).astype(np.float32) for key, values in self.rows.items()}
        path = directory / "numeric.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != set(arrays) or not all(np.array_equal(saved[key], array) for key, array in arrays.items()):
                raise ValueError("numeric dataset read-back mismatch")
        manifest = {"schema": "habitus.stream-numeric-pairs.v1", "frames": self.frames,
                    "shapes": {key: list(array.shape) for key, array in arrays.items()},
                    "numeric_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "dtype": "float32", "cortex_weights_used": False, "training_performed": False,
                    "labels_or_expected_answers_in_vectors": False,
                    "full_state_location": "separate flow checkpoint",
                    "private_language_is_external_evidence": False}
        with (directory / "frames.json").open("x") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
        return {"frames": len(self.frames), "numeric_sha256": manifest["numeric_sha256"],
                "numeric_bytes": path.stat().st_size, "shapes": manifest["shapes"]}
