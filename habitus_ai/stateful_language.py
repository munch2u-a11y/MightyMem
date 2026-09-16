"""Communication-only runner for Habitus's random-born recurrent language core.

There is no chat adapter, prompt builder, transcript store, retrieval interface,
or phrase selector here. One current message crosses HEAR once; the same network
then generates SPEAK bytes and a learned boundary. Only a fixed-size neural
state survives the turn. This research runner does not replace IntegratedMind.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
from pathlib import Path
import sqlite3
from typing import Any, Sequence

from .developmental_cortex import (
    BornInRecurrentCortex,
    CortexConfig,
    CortexLineageError,
    DIRECTION_IDS,
    _complete_utf8,
    _deserialize_hidden,
    _serialize_hidden,
    _tensor_sha256,
    _valid_utf8_prefix,
    require_torch,
    torch,
)
from .language_vocabulary import LanguageVocabulary


CHECKPOINT_SCHEMA = "habitus.stateful-language.v1"
LEXICAL_CHECKPOINT_SCHEMA = "habitus.stateful-language.lexical.v1"
SENSORY_STATE_POLICIES = ("shared", "protected", "coupled")
SENSORY_READOUT_MODES = ("static", "language_query", "sensory_query")
LANGUAGE_INPUT_SKIPS = ("none", "mean_embedding")


if torch is not None:
    class StateConditionedCortex(BornInRecurrentCortex):
        """Read this core's own retained state through its numeric field input."""

        def __init__(self, config: CortexConfig):
            super().__init__(config)
            self.state_readout = torch.nn.Linear(
                config.recurrent_layers * config.hidden_width, config.graph_width
            )


def language_field(model: BornInRecurrentCortex, hidden: Any) -> Any:
    if hasattr(model, "retained_field"):
        return model.retained_field(hidden)
    if hasattr(model, "state_readout"):
        field = model.state_readout(hidden.transpose(0, 1).flatten(1))
        if hasattr(model, "state_readout_norm"):
            field = model.state_readout_norm(field)
        return torch.tanh(field)
    return torch.zeros(hidden.shape[1], model.config.graph_width, device=hidden.device)


def new_language_core(
    config: CortexConfig, *, state_readout: bool = False, vocabulary: LanguageVocabulary | None = None,
    initialization_threads: int = 2, stream_normalization: bool = False,
    sensory_trunk: bool = False,
    sensory_state_policy: str = "shared",
    sensory_readout_mode: str = "static",
    language_input_skip: str = "none",
) -> BornInRecurrentCortex:
    require_torch()
    if type(stream_normalization) is not bool or (stream_normalization and (vocabulary is None or not state_readout)):
        raise ValueError("stream normalization requires a lexical state-readout lineage")
    if type(sensory_trunk) is not bool or (sensory_trunk and (vocabulary is None or not state_readout)):
        raise ValueError("a sensory trunk requires a lexical state-readout lineage")
    if sensory_state_policy not in SENSORY_STATE_POLICIES or (
        sensory_state_policy != "shared" and not (sensory_trunk and vocabulary is not None and state_readout)
    ):
        raise ValueError("sensory state policy requires a lexical state-readout sensory trunk")
    if sensory_readout_mode not in SENSORY_READOUT_MODES or (sensory_readout_mode != "static" and sensory_state_policy == "shared"):
        raise ValueError("a sensory query readout requires a numeric reservoir")
    if language_input_skip not in LANGUAGE_INPUT_SKIPS or (language_input_skip != "none" and
            (sensory_state_policy != "protected" or sensory_readout_mode != "static")):
        raise ValueError("language input skip requires a protected static reservoir")
    if type(initialization_threads) is not int or initialization_threads < 1:
        raise ValueError("initialization thread count must be a positive integer")
    # Orthogonal initialization's floating-point QR depends on CPU thread
    # count. Pin and record its recipe, independently of inference/training
    # threads, while preserving the caller's RNG and thread setting.
    caller_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(initialization_threads)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            if sensory_state_policy != "shared":
                from .sensory_reservoir import SensoryReservoirCortex
                model = SensoryReservoirCortex(config, vocabulary, stream_normalization=stream_normalization,
                                               sensory_state_policy=sensory_state_policy,
                                               sensory_readout_mode=sensory_readout_mode,
                                               language_input_skip=language_input_skip)
            elif vocabulary is not None:
                from .lexical_cortex import LexicalRecurrentCortex
                model = LexicalRecurrentCortex(config, vocabulary, state_readout=state_readout,
                                               stream_normalization=stream_normalization, sensory_trunk=sensory_trunk)
            else:
                model = StateConditionedCortex(config) if state_readout else BornInRecurrentCortex(config)
    finally:
        torch.set_num_threads(caller_threads)
    model = model.to(device="cpu", dtype=torch.float32)
    model.language_initialization = {"threads": initialization_threads, "torch_version": str(torch.__version__)}
    model.language_initial_sha256 = language_model_sha256(model)
    return model


def language_model_sha256(model: BornInRecurrentCortex) -> str:
    weights = _tensor_sha256(model)
    vocabulary = getattr(model, "vocabulary", None)
    if vocabulary is None:
        return weights  # preserve every existing byte checkpoint/state identity
    identity = f"{LEXICAL_CHECKPOINT_SCHEMA}\0{weights}\0{vocabulary.sha256}"
    policy = getattr(model, "sensory_state_policy", "shared")
    if policy != "shared":
        identity += f"\0sensory-state:{policy}"
    readout = getattr(model, "sensory_readout_mode", "static")
    if readout != "static":
        identity += f"\0sensory-readout:{readout}"
    input_skip = getattr(model, "language_input_skip", "none")
    if input_skip != "none":
        identity += f"\0language-input-skip:{input_skip}"
    return hashlib.sha256(identity.encode()).hexdigest()


def language_symbols(model: BornInRecurrentCortex, text: str) -> tuple[int, ...]:
    vocabulary = getattr(model, "vocabulary", None)
    return vocabulary.encode(text) if vocabulary is not None else tuple(text.encode("utf-8"))


def save_language_checkpoint(
    path: Path,
    model: BornInRecurrentCortex,
    *,
    initial_model_sha256: str,
    training: dict[str, Any],
) -> str:
    """Create a new inspectable weight artifact; never overwrite a checkpoint."""
    if initial_model_sha256 != model.language_initial_sha256:
        raise CortexLineageError("checkpoint initial hash differs from its verified lineage")
    path.parent.mkdir(parents=True, exist_ok=True)
    model_hash = language_model_sha256(model)
    vocabulary = getattr(model, "vocabulary", None)
    payload = {
        "schema": LEXICAL_CHECKPOINT_SCHEMA if vocabulary is not None else CHECKPOINT_SCHEMA,
        "config": asdict(model.config),
        "weights": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "initial_model_sha256": str(initial_model_sha256),
        "initialization": dict(model.language_initialization),
        "model_sha256": model_hash,
        "training": training,
        "pretrained_model_loaded": False,
        "state_readout": hasattr(model, "state_readout"),
        "stream_normalization": getattr(model, "stream_normalization", False),
        "sensory_trunk": hasattr(model, "sensory_trunk"),
        "sensory_state_policy": getattr(model, "sensory_state_policy", "shared"),
        "sensory_readout_mode": getattr(model, "sensory_readout_mode", "static"),
        "language_input_skip": getattr(model, "language_input_skip", "none"),
    }
    if vocabulary is not None:
        payload["vocabulary"] = vocabulary.payload()
        payload["weights_sha256"] = _tensor_sha256(model)
    if getattr(model, "language_architecture_origin", None) is not None:
        payload["architecture_origin"] = dict(model.language_architecture_origin)
    with path.open("xb") as destination:
        torch.save(payload, destination)
    return model_hash


def load_language_checkpoint(path: Path) -> BornInRecurrentCortex:
    require_torch()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") not in {CHECKPOINT_SCHEMA, LEXICAL_CHECKPOINT_SCHEMA}:
        raise CortexLineageError("not a Habitus stateful-language checkpoint")
    if payload.get("pretrained_model_loaded") is not False:
        raise CortexLineageError("checkpoint must declare a from-scratch lineage")
    config = CortexConfig(**payload["config"])
    vocabulary = None
    if payload["schema"] == LEXICAL_CHECKPOINT_SCHEMA:
        vocabulary = LanguageVocabulary.from_payload(payload["vocabulary"])
    elif "vocabulary" in payload:
        raise CortexLineageError("a byte checkpoint cannot silently acquire a lexical alphabet")
    stream_normalization = payload.get("stream_normalization", False)
    if type(stream_normalization) is not bool or (
        stream_normalization and (vocabulary is None or not payload.get("state_readout", False))
    ):
        raise CortexLineageError("invalid stream-normalization checkpoint shape")
    sensory_trunk = payload.get("sensory_trunk", False)
    if type(sensory_trunk) is not bool or (sensory_trunk and (vocabulary is None or not payload.get("state_readout", False))):
        raise CortexLineageError("invalid sensory-trunk checkpoint shape")
    sensory_state_policy = payload.get("sensory_state_policy", "shared")
    if sensory_state_policy not in SENSORY_STATE_POLICIES or (sensory_state_policy != "shared" and not sensory_trunk):
        raise CortexLineageError("invalid sensory-state checkpoint policy")
    sensory_readout_mode = payload.get("sensory_readout_mode", "static")
    if sensory_readout_mode not in SENSORY_READOUT_MODES or (sensory_readout_mode != "static" and sensory_state_policy == "shared"):
        raise CortexLineageError("invalid sensory-readout checkpoint mode")
    language_input_skip = payload.get("language_input_skip", "none")
    if language_input_skip not in LANGUAGE_INPUT_SKIPS or (language_input_skip != "none" and
            (sensory_state_policy != "protected" or sensory_readout_mode != "static")):
        raise CortexLineageError("invalid language-input-skip checkpoint mode")
    initialization = payload.get("initialization")
    if initialization is not None:
        if not isinstance(initialization, dict) or "threads" not in initialization:
            raise CortexLineageError("checkpoint has an invalid initialization recipe")
        initialization_threads = initialization["threads"]
    else:
        # Earlier training artifacts already recorded their thread count in
        # run provenance. Artifacts with neither recipe retain the historical
        # caller-thread behavior; the initial hash must still match exactly.
        initialization_threads = payload.get("training", {}).get("run_provenance", {}).get("arguments", {}).get(
            "threads", torch.get_num_threads(),
        )
    if type(initialization_threads) is not int or initialization_threads < 1:
        raise CortexLineageError("checkpoint has an invalid initialization thread count")
    model = new_language_core(config, state_readout=payload.get("state_readout", False), vocabulary=vocabulary,
                             initialization_threads=initialization_threads, stream_normalization=stream_normalization,
                             sensory_trunk=sensory_trunk, sensory_state_policy=sensory_state_policy,
                             sensory_readout_mode=sensory_readout_mode, language_input_skip=language_input_skip)
    if language_model_sha256(model) != payload.get("initial_model_sha256"):
        raise CortexLineageError("checkpoint does not match its random initialization; use its matching initialization environment")
    model.load_state_dict(payload["weights"], strict=True)
    if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
        raise CortexLineageError("checkpoint contains non-finite weights")
    if language_model_sha256(model) != payload.get("model_sha256"):
        raise CortexLineageError("checkpoint weight hash mismatch")
    if vocabulary is not None and _tensor_sha256(model) != payload.get("weights_sha256"):
        raise CortexLineageError("checkpoint tensor-only hash mismatch")
    origin = payload.get("architecture_origin")
    if origin is not None:
        expected_origin = "habitus.native-sensory-growth.v1" if sensory_readout_mode == "static" else "habitus.native-query-growth.v1"
        if language_input_skip != "none":
            expected_origin = "habitus.native-language-skip-growth.v1"
        if (not isinstance(origin, dict) or origin.get("schema") != expected_origin
                or origin.get("sensory_state_policy") != sensory_state_policy
                or origin.get("sensory_readout_mode", "static") != sensory_readout_mode
                or origin.get("language_input_skip", "none") != language_input_skip
                or origin.get("factory_initial_model_sha256") != model.language_initial_sha256
                or origin.get("external_pretrained_model_loaded") is not False
                or origin.get("native_learned_parent_loaded") is not True):
            raise CortexLineageError("invalid native architectural origin")
        model.language_architecture_origin = origin
    return model.eval()


@dataclass(frozen=True)
class LanguageTurn:
    response: str
    status: str
    turn: int
    input_byte_count: int
    input_sha256: str
    output_sha256: str | None
    state_before_sha256: str
    state_after_sha256: str
    state_bytes: int
    model_sha256: str
    generated_byte_count: int
    stop_probability: float
    transcript_records_used: int = 0
    system_prompt_used: bool = False
    retrieval_used: bool = False
    pretrained_model_used: bool = False
    generation_mode: str = "cortex_bytes"
    input_symbol_count: int | None = None
    generated_symbol_count: int | None = None
    vocabulary_sha256: str | None = None


class StatefulConversation:
    """One continuing neural state, optionally persisted in a one-row database."""

    def __init__(
        self,
        model: BornInRecurrentCortex,
        *,
        state_path: Path | None = None,
    ) -> None:
        require_torch()
        self.model = model.eval()
        self.config = model.config
        self.model_sha256 = language_model_sha256(model)
        self.hidden = model.initial_hidden(1, device="cpu", dtype=torch.float32)
        self.turn = 0
        self._closed = False
        self.connection: sqlite3.Connection | None = None
        if state_path is not None:
            state_path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(state_path)
            try:
                self.connection.execute(
                    """CREATE TABLE IF NOT EXISTS language_state (
                        id INTEGER PRIMARY KEY CHECK (id = 1),
                        model_sha256 TEXT NOT NULL,
                        turn INTEGER NOT NULL,
                        hidden BLOB NOT NULL,
                        hidden_sha256 TEXT NOT NULL
                    )"""
                )
                blob, _ = _serialize_hidden(self.hidden)
                with self.connection:
                    self.connection.execute(
                        "INSERT OR IGNORE INTO language_state VALUES (1, ?, 0, ?, ?)",
                        (self.model_sha256, blob, hashlib.sha256(blob).hexdigest()),
                    )
                row = self.connection.execute(
                    "SELECT model_sha256, turn, hidden, hidden_sha256 FROM language_state WHERE id=1"
                ).fetchone()
                if row[0] != self.model_sha256:
                    raise CortexLineageError("conversation state belongs to different weights")
                if hashlib.sha256(row[2]).hexdigest() != row[3]:
                    raise CortexLineageError("conversation state hash mismatch")
                self.hidden = _deserialize_hidden(
                    row[2], self.hidden.shape, device="cpu", dtype=torch.float32
                )
                if not torch.isfinite(self.hidden).all():
                    raise CortexLineageError("conversation state is non-finite")
                self.turn = int(row[1])
            except BaseException:
                self.close()
                raise

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        self._closed = True

    def __enter__(self) -> StatefulConversation:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _commit(self, hidden: Any) -> bytes:
        if not torch.isfinite(hidden).all():
            raise CortexLineageError("refusing to persist non-finite conversation state")
        blob, _ = _serialize_hidden(hidden)
        if self.connection is not None:
            with self.connection:
                result = self.connection.execute(
                    """UPDATE language_state SET turn=?, hidden=?, hidden_sha256=?
                       WHERE id=1 AND turn=? AND model_sha256=?""",
                    (
                        self.turn + 1, blob, hashlib.sha256(blob).hexdigest(),
                        self.turn, self.model_sha256,
                    ),
                )
                if result.rowcount != 1:
                    raise RuntimeError("conversation state advanced in another session; reopen it")
        self.hidden = hidden.detach()
        self.turn += 1
        return blob

    def reply(self, current_message: str, *, maximum_bytes: int | None = None) -> LanguageTurn:
        if self._closed:
            raise RuntimeError("conversation is closed")
        if language_model_sha256(self.model) != self.model_sha256:
            raise CortexLineageError("weights changed during this conversation")
        if not isinstance(current_message, str) or not current_message.strip():
            raise ValueError("current message must be nonempty text")
        incoming = current_message.encode("utf-8")
        if maximum_bytes is not None and maximum_bytes < 1:
            raise ValueError("an explicit diagnostic output limit must be positive")
        before, _ = _serialize_hidden(self.hidden)
        hear = torch.tensor([DIRECTION_IDS["hear"]])
        speak = torch.tensor([DIRECTION_IDS["speak"]])
        generated = bytearray()
        incoming_symbols = language_symbols(self.model, current_message)
        vocabulary = getattr(self.model, "vocabulary", None)
        pieces = vocabulary.pieces if vocabulary is not None else tuple(bytes((value,)) for value in range(256))
        generated_symbols = 0
        stopped = False
        stop_probability = 0.0
        with torch.no_grad():
            field = language_field(self.model, self.hidden)
            _, heard_hidden = self.model(
                torch.tensor([[-1, *incoming_symbols]]), field, hear, self.hidden
            )
            field = language_field(self.model, heard_hidden)
            hidden = heard_hidden
            previous = -1
            steps = itertools.count() if maximum_bytes is None else range(maximum_bytes + 1)
            for step in steps:
                outputs, hidden = self.model(torch.tensor([[previous]]), field, speak, hidden)
                stop_probability = float(torch.sigmoid(outputs["stop_logits"][0, -1]))
                if (
                    generated
                    and _complete_utf8(bytes(generated))
                    and stop_probability >= self.config.learned_stop_threshold
                ):
                    stopped = True
                    break
                if maximum_bytes is not None and len(generated) >= maximum_bytes:
                    break
                logits = outputs["byte_logits"][0, -1]
                valid = torch.tensor([
                    (maximum_bytes is None or len(generated) + len(piece) <= maximum_bytes)
                    and _valid_utf8_prefix(bytes(generated) + piece)
                    for piece in pieces
                ])
                if not bool(valid.any()):
                    break
                previous = int(logits.masked_fill(~valid, float("-inf")).argmax())
                generated.extend(pieces[previous])
                generated_symbols += 1
        response = bytes(generated).decode("utf-8") if stopped else ""
        if not response.strip():
            stopped = False
            response = ""
        # An unfinished private decode is not a spoken experience. The current
        # human input is retained even when the model cannot finish its reply.
        after = self._commit(hidden if stopped else heard_hidden)
        return LanguageTurn(
            response=response,
            status="spoken" if stopped else "incomplete",
            turn=self.turn,
            input_byte_count=len(incoming),
            input_sha256=hashlib.sha256(incoming).hexdigest(),
            output_sha256=hashlib.sha256(response.encode("utf-8")).hexdigest() if stopped else None,
            state_before_sha256=hashlib.sha256(before).hexdigest(),
            state_after_sha256=hashlib.sha256(after).hexdigest(),
            state_bytes=len(after),
            model_sha256=self.model_sha256,
            generated_byte_count=len(generated),
            stop_probability=stop_probability,
            generation_mode="cortex_subwords" if vocabulary is not None else "cortex_bytes",
            input_symbol_count=len(incoming_symbols), generated_symbol_count=generated_symbols,
            vocabulary_sha256=vocabulary.sha256 if vocabulary is not None else None,
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--once")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--maximum-bytes", type=int, default=None)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    require_torch()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    model = load_language_checkpoint(args.checkpoint)
    with StatefulConversation(model, state_path=args.state) as conversation:
        while True:
            try:
                current = args.once if args.once is not None else input("You: ")
            except (EOFError, KeyboardInterrupt):
                return 0
            if current == "/quit":
                return 0
            result = conversation.reply(current, maximum_bytes=args.maximum_bytes)
            if args.json:
                print(json.dumps(asdict(result), sort_keys=True))
            elif result.status == "spoken":
                print(f"Habitus: {result.response}")
            else:
                print("[No completed neural utterance; no substitute response was generated.]")
            if args.once is not None:
                return 0


if __name__ == "__main__":
    raise SystemExit(main())
