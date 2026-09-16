"""A randomly initialized, pulse-native recurrent cortex for Habitus.

The cortex is intentionally unable to load a pretrained model or tokenizer.
It senses UTF-8 bytes, receives a numeric projection of the current Habitus
field, and persists only a compact recurrent state with each SELF pulse.
Canonical records remain the authority for experience; this module never
builds a transcript window or a text retrieval packet.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import struct
import tempfile
import uuid
from typing import Any, Callable, Mapping, Sequence

from .self_pulse import PulseCommitFrame
from .types import InputTrunk, OutputTrunk, RecordType, RecurrentSnapshot

try:  # Optional until the local ROCm environment is installed.
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - exercised by dependency-free tests
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[assignment,misc]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


CORTEX_SCHEMA = "habitus.developmental-cortex.v2"
CORTEX_INITIALIZATION = "random_from_seed"
DIRECTION_IDS = {"hear": 0, "speak": 1, "quiet": 2}
ACTION_IDS = {
    OutputTrunk.DO: 0,
    OutputTrunk.LOOK: 1,
    OutputTrunk.SPEAK: 2,
}


class CortexDependencyError(RuntimeError):
    """Raised when neural execution is requested without PyTorch."""


class CortexLineageError(RuntimeError):
    """Raised when a checkpoint does not belong to this random-born cortex."""


class CortexNumericalError(RuntimeError):
    """Raised before non-finite neural state can receive a learning receipt."""


def require_torch() -> None:
    if torch is None or nn is None or F is None:
        raise CortexDependencyError(
            "developmental cortex requires PyTorch; install the isolated ROCm "
            "environment or a CPU PyTorch build"
        )


@dataclass(frozen=True)
class CortexConfig:
    """Shape and plasticity defaults for the first local developmental model."""

    byte_embedding_width: int = 256
    graph_width: int = 256
    direction_width: int = 32
    hidden_width: int = 1024
    recurrent_layers: int = 3
    maximum_event_bytes: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    gradient_clip: float = 1.0
    learned_stop_threshold: float = 0.80
    seed: int = 1701
    initialization: str = CORTEX_INITIALIZATION
    architecture_revision: str = "born-in-recurrent-cortex-v2-stop"

    def __post_init__(self) -> None:
        if self.initialization != CORTEX_INITIALIZATION:
            raise CortexLineageError(
                "the born-in cortex only accepts random_from_seed initialization"
            )
        for name in (
            "byte_embedding_width",
            "graph_width",
            "direction_width",
            "hidden_width",
            "recurrent_layers",
            "maximum_event_bytes",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if self.graph_width < 16:
            raise ValueError("graph_width must be at least 16")
        if self.maximum_event_bytes < 2:
            raise ValueError("maximum_event_bytes must be at least 2")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if not math.isfinite(self.gradient_clip) or self.gradient_clip <= 0.0:
            raise ValueError("gradient_clip must be finite and positive")
        if (
            not math.isfinite(self.learned_stop_threshold)
            or not 0.5 < self.learned_stop_threshold < 1.0
        ):
            raise ValueError("learned_stop_threshold must be within (0.5, 1.0)")
        if not self.architecture_revision:
            raise ValueError("architecture_revision cannot be empty")

    @property
    def architecture_id(self) -> str:
        encoded = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return "cortex:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]


@dataclass(frozen=True)
class CortexPulseState:
    pulse: int
    pulse_id: str
    state_sha256: str
    model_sha256: str
    graph_field: tuple[float, ...]
    graph_field_sha256: str
    hidden_shape: tuple[int, ...]
    sensed_byte_count: int
    input_record_ids: tuple[str, ...]
    route_ids: tuple[str, ...]
    curriculum_stage: str


@dataclass(frozen=True)
class CortexOutputState:
    event_id: str
    parent_pulse: int
    sequence: int
    state_sha256: str
    model_sha256: str
    graph_field_sha256: str
    payload_sha256: str
    byte_count: int
    stopped: bool
    stop_probability: float
    stop_source: str


@dataclass(frozen=True)
class CortexTrainingEpisode:
    """One grounded learning item derived from committed experience.

    ``payload`` exists only in the active training call. The plasticity receipt
    stores its hash and causal record IDs, not a duplicate language corpus.
    """

    episode_id: str
    pulse: int
    record_ids: tuple[str, ...]
    route_ids: tuple[str, ...]
    graph_field: tuple[float, ...]
    payload: bytes = b""
    conditioning_payloads: tuple[bytes, ...] = ()
    direction: str = "hear"
    kind: str = "sensory"
    consequence: float = 0.0
    valence_delta: float = 0.0
    selected_action: OutputTrunk | None = None
    verified: bool = False
    reward: float = 0.0
    negative_graph_field: tuple[float, ...] | None = None
    experience_cycle_id: str | None = None
    return_record_id: str | None = None
    motor_rehearsal: bool = False


@dataclass(frozen=True)
class PlasticityReceipt:
    update_id: str
    episode_ids: tuple[str, ...]
    before_model_sha256: str
    after_model_sha256: str
    checkpoint_path: str
    checkpoint_sha256: str
    evidence_sha256: str
    losses: Mapping[str, float]
    verified_behavior_examples: int
    optimization_steps: int
    curriculum_stage: str


@dataclass(frozen=True)
class CortexEpisodeEvaluation:
    episode_id: str
    byte_top_1: float | None
    byte_top_5: float | None
    route_cosine: float
    predicted_consequence: float
    predicted_valence: float
    selected_action: OutputTrunk
    expected_action_probability: float | None
    stop_accuracy: float | None = None


@dataclass(frozen=True)
class CortexGeneration:
    payload: bytes
    step_probabilities: tuple[float, ...]
    initial_state_sha256: str
    final_hidden: bytes
    final_hidden_shape: tuple[int, ...]
    stopped: bool = False
    stop_probability: float = 0.0
    stop_source: str = "none"


@dataclass(frozen=True)
class CortexSequenceScore:
    """Read-only likelihood of one graph-offered outward byte trajectory."""

    payload_sha256: str
    byte_count: int
    mean_byte_log_probability: float
    boundary_log_probability: float
    sequence_score: float
    competition_probability: float
    initial_state_sha256: str
    model_sha256: str


@dataclass(frozen=True)
class CortexProposal:
    """A read-only learned valuation offered to the current SELF pulse.

    The proposal is not an authorization.  It carries only numeric predictions
    tied to the exact model, graph field, and current sensory byte material that
    produced them; the pulse kernel remains the authority that may use or ignore
    that evidence when ranking its ordinary graph affordances.
    """

    proposal_id: str
    model_sha256: str
    graph_field_sha256: str
    sensory_payload_sha256: str
    sensed_byte_count: int
    action_probabilities: Mapping[str, float]
    predicted_consequence: float
    predicted_valence: float
    route_alignment: float
    confidence: float

    def support_for(self, trunk: OutputTrunk) -> float:
        return float(self.action_probabilities.get(trunk.value, 0.0))


def _hash_slot(material: str, width: int) -> tuple[int, float]:
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % width, (1.0 if digest[8] & 1 else -1.0)


def pulse_field(
    recurrent: RecurrentSnapshot,
    context_node_ids: Sequence[str],
    *,
    selected_outputs: Sequence[Any] = (),
    width: int = 256,
) -> tuple[float, ...]:
    """Project opaque live graph state without labels, terms, or record text."""
    if width < 16:
        raise ValueError("pulse field width must be at least 16")
    result = [0.0] * width
    states = {state.node_id: state for state in recurrent.node_states}
    for node_id in sorted(set(states) | {str(item) for item in context_node_ids}):
        state = states.get(node_id)
        values = (
            ("presence", 1.0 if state is None else max(0.05, state.activation)),
            ("activation", 0.0 if state is None else state.activation),
            ("pressure", 0.0 if state is None else state.pressure),
            ("valence", 0.0 if state is None else state.valence),
            ("momentum", 0.0 if state is None else state.momentum),
        )
        for feature, value in values:
            if abs(value) <= 1e-12:
                continue
            index, sign = _hash_slot(f"node\0{node_id}\0{feature}", width)
            result[index] += sign * float(value)
    for selected in selected_outputs:
        trunk = getattr(selected, "trunk", None)
        node_id = str(getattr(selected, "node_id", ""))
        score = float(getattr(selected, "score", 0.0))
        index, sign = _hash_slot(
            f"output\0{getattr(trunk, 'value', trunk)}\0{node_id}", width
        )
        result[index] += sign * score
    norm = math.sqrt(sum(value * value for value in result))
    if norm > 1.0:
        result = [value / norm for value in result]
    return tuple(result)


def active_route_ids(
    recurrent: RecurrentSnapshot,
    context_node_ids: Sequence[str],
    *,
    minimum_activation: float = 0.05,
    maximum: int | None = None,
) -> tuple[str, ...]:
    ranked = sorted(
        (
            (state.activation + 0.25 * state.pressure, state.node_id)
            for state in recurrent.node_states
            if state.activation >= minimum_activation or state.pressure >= minimum_activation
        ),
        key=lambda item: (-item[0], item[1]),
    )
    ordered = [str(item) for item in context_node_ids]
    ordered.extend(node_id for _, node_id in ranked)
    result = tuple(dict.fromkeys(ordered))
    return result if maximum is None else result[: max(1, int(maximum))]


if nn is not None:

    class BornInRecurrentCortex(nn.Module):
        """A blank recurrent language tissue conditioned by the live graph."""

        def __init__(self, config: CortexConfig):
            super().__init__()
            self.config = config
            self.byte_embedding = nn.Embedding(256, config.byte_embedding_width)
            self.boundary_embedding = nn.Parameter(
                torch.empty(config.byte_embedding_width)
            )
            self.direction_embedding = nn.Embedding(3, config.direction_width)
            input_width = (
                config.byte_embedding_width
                + config.graph_width
                + config.direction_width
            )
            self.input_norm = nn.LayerNorm(input_width)
            self.recurrent = nn.GRU(
                input_width,
                config.hidden_width,
                num_layers=config.recurrent_layers,
                batch_first=True,
            )
            self.byte_head = nn.Linear(config.hidden_width, 256)
            self.stop_head = nn.Linear(config.hidden_width, 1)
            self.route_head = nn.Linear(config.hidden_width, config.graph_width)
            self.consequence_head = nn.Linear(config.hidden_width, 1)
            self.valence_head = nn.Linear(config.hidden_width, 1)
            self.action_head = nn.Linear(config.hidden_width, len(ACTION_IDS))
            self.reset_parameters()

        def reset_parameters(self) -> None:
            nn.init.normal_(self.byte_embedding.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.boundary_embedding, mean=0.0, std=0.02)
            nn.init.normal_(self.direction_embedding.weight, mean=0.0, std=0.02)
            for name, parameter in self.recurrent.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(parameter)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(parameter)
                elif "bias" in name:
                    nn.init.zeros_(parameter)
            for head in (
                self.byte_head,
                self.stop_head,
                self.route_head,
                self.consequence_head,
                self.valence_head,
                self.action_head,
            ):
                nn.init.xavier_uniform_(head.weight)
                nn.init.zeros_(head.bias)

        def initial_hidden(self, batch_size: int, *, device: Any, dtype: Any) -> Tensor:
            return torch.zeros(
                self.config.recurrent_layers,
                batch_size,
                self.config.hidden_width,
                device=device,
                dtype=dtype,
            )

        def forward(
            self,
            previous_bytes: Tensor,
            graph_field_tensor: Tensor,
            direction_ids: Tensor,
            hidden: Tensor | None = None,
            *,
            sequence_lengths: Tensor | None = None,
        ) -> tuple[Mapping[str, Tensor], Tensor]:
            if previous_bytes.ndim != 2:
                raise ValueError("previous_bytes must have shape [batch, sequence]")
            batch, length = previous_bytes.shape
            boundary_mask = previous_bytes < 0
            lexical = self.byte_embedding(previous_bytes.clamp(min=0, max=255))
            boundary = self.boundary_embedding.view(1, 1, -1).expand(batch, length, -1)
            lexical = torch.where(boundary_mask.unsqueeze(-1), boundary, lexical)
            graph_rows = graph_field_tensor.unsqueeze(1).expand(batch, length, -1)
            if direction_ids.shape == (batch,):
                directions = self.direction_embedding(direction_ids).unsqueeze(1).expand(
                    batch, length, -1
                )
            elif direction_ids.shape == (batch, length):
                # Training may unroll several HEAR/SPEAK events without
                # detaching hidden state at each turn. Inference still admits
                # only the current event and its carried numeric state.
                directions = self.direction_embedding(direction_ids)
            else:
                raise ValueError("direction_ids must have shape [batch] or [batch, sequence]")
            inputs = self.input_norm(torch.cat((lexical, graph_rows, directions), dim=-1))
            if sequence_lengths is None:
                sequence, final_hidden = self.recurrent(inputs, hidden)
                final = sequence[:, -1, :]
            else:
                if sequence_lengths.shape != (batch,):
                    raise ValueError("sequence_lengths must have shape [batch]")
                if torch.any(sequence_lengths < 1) or torch.any(sequence_lengths > length):
                    raise ValueError("sequence lengths must be within the padded sequence")
                packed = nn.utils.rnn.pack_padded_sequence(
                    inputs, sequence_lengths.cpu(), batch_first=True, enforce_sorted=False
                )
                packed_sequence, final_hidden = self.recurrent(packed, hidden)
                sequence, _ = nn.utils.rnn.pad_packed_sequence(
                    packed_sequence, batch_first=True, total_length=length
                )
                final = final_hidden[-1]
            return {
                "byte_logits": self.byte_head(sequence),
                "stop_logits": self.stop_head(sequence).squeeze(-1),
                "route": torch.tanh(self.route_head(final)),
                "consequence": torch.tanh(self.consequence_head(final)).squeeze(-1),
                "valence": torch.tanh(self.valence_head(final)).squeeze(-1),
                "action_logits": self.action_head(final),
            }, final_hidden

else:

    class BornInRecurrentCortex:  # pragma: no cover - dependency error path
        def __init__(self, *_: Any, **__: Any):
            require_torch()


def _tensor_sha256(model: Any) -> str:
    require_torch()
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _valid_utf8_prefix(payload: bytes) -> bool:
    try:
        payload.decode("utf-8", errors="strict")
        return True
    except UnicodeDecodeError as error:
        return error.reason == "unexpected end of data" and error.end == len(payload)


def _complete_utf8(payload: bytes) -> bool:
    try:
        payload.decode("utf-8", errors="strict")
        return True
    except UnicodeDecodeError:
        return False


def _serialize_hidden(hidden: Tensor) -> tuple[bytes, tuple[int, ...]]:
    values = hidden.detach().to(device="cpu", dtype=torch.float32).contiguous().view(-1)
    shape = tuple(int(item) for item in hidden.shape)
    return struct.pack(f"<{values.numel()}f", *values.tolist()), shape


def _deserialize_hidden(payload: bytes, shape: Sequence[int], *, device: Any, dtype: Any) -> Tensor:
    count = math.prod(int(item) for item in shape)
    if len(payload) != count * 4:
        raise CortexLineageError("persisted cortex hidden-state size is invalid")
    values = struct.unpack(f"<{count}f", payload)
    return torch.tensor(values, device=device, dtype=dtype).reshape(tuple(shape))


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


class DevelopmentalCortex:
    """Own the blank neural tissue and its auditable Habitus coupling."""

    def __init__(
        self,
        mind: Any,
        *,
        config: CortexConfig | None = None,
        checkpoint_directory: str | Path | None = None,
        device: str = "auto",
        curriculum_stage: str = "prelinguistic",
    ) -> None:
        require_torch()
        self.mind = mind
        self.config = config or CortexConfig()
        self.device = self._resolve_device(device)
        # This cortex is small enough for full precision on the target 16 GiB
        # APU.  Persistent recurrent learning is substantially more sensitive
        # to accumulated FP16 optimizer error than a one-step inference probe,
        # so precision is an invariant across CPU and ROCm lineages.
        self.compute_dtype = torch.float32
        self.curriculum_stage = str(curriculum_stage)
        torch.manual_seed(self.config.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.config.seed)
        self.model = BornInRecurrentCortex(self.config).to(
            device=self.device, dtype=self.compute_dtype
        )
        self._model_sha256_cache: str | None = None
        # An integrated continual trainer owns the corresponding joint checkpoint.
        # Legacy consolidation retains its existing rebase behavior.
        self.continuous_hidden: Tensor | None = None
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        default_checkpoints = (
            Path(mind.store.path).resolve().parent / "developmental_cortex_checkpoints"
            if mind.store.path != ":memory:"
            else Path(tempfile.gettempdir()) / "habitus-developmental-cortex"
        )
        self.checkpoint_directory = Path(
            checkpoint_directory or default_checkpoints
        ).expanduser().resolve()
        self.checkpoint_directory.mkdir(parents=True, exist_ok=True)
        self._create_schema()
        self._bind_lineage()
        self._load_latest_checkpoint()

    @staticmethod
    def _resolve_device(requested: str) -> Any:
        value = str(requested).casefold()
        if value in {"amd", "rocm"}:
            if not torch.cuda.is_available() or torch.version.hip is None:
                raise CortexDependencyError(
                    "AMD execution requested but ROCm PyTorch cannot see a HIP device"
                )
            return torch.device("cuda")
        if value == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if value == "cpu":
            return torch.device("cpu")
        raise ValueError("device must be auto, amd, rocm, or cpu")

    @property
    def connection(self) -> Any:
        return self.mind.store.connection

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters())

    @property
    def model_sha256(self) -> str:
        if self._model_sha256_cache is None:
            self._model_sha256_cache = _tensor_sha256(self.model)
        return self._model_sha256_cache

    def _require_finite_model(self, *, context: str) -> None:
        for name, parameter in self.model.named_parameters():
            if not bool(torch.isfinite(parameter).all().item()):
                raise CortexNumericalError(
                    f"non-finite cortex parameter {name!r} {context}"
                )

    def _create_schema(self) -> None:
        with self.connection:
            self.connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS cortex_lineage (
                    lineage_id TEXT PRIMARY KEY,
                    schema_id TEXT NOT NULL,
                    architecture_json TEXT NOT NULL,
                    initialization TEXT NOT NULL,
                    seed INTEGER NOT NULL,
                    parameter_count INTEGER NOT NULL,
                    pretrained_source TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    CHECK(pretrained_source IS NULL),
                    CHECK(initialization = 'random_from_seed')
                );

                CREATE TABLE IF NOT EXISTS cortex_pulse_states (
                    pulse INTEGER PRIMARY KEY,
                    pulse_id TEXT NOT NULL UNIQUE,
                    lineage_id TEXT NOT NULL REFERENCES cortex_lineage(lineage_id),
                    model_sha256 TEXT NOT NULL,
                    state_sha256 TEXT NOT NULL UNIQUE,
                    graph_field BLOB NOT NULL,
                    graph_field_sha256 TEXT NOT NULL,
                    hidden_shape_json TEXT NOT NULL,
                    hidden_state BLOB NOT NULL,
                    sensed_byte_count INTEGER NOT NULL,
                    input_record_ids_json TEXT NOT NULL,
                    route_ids_json TEXT NOT NULL,
                    curriculum_stage TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TRIGGER IF NOT EXISTS cortex_pulse_states_immutable_update
                BEFORE UPDATE ON cortex_pulse_states BEGIN
                    SELECT RAISE(ABORT, 'cortex pulse states are immutable');
                END;

                CREATE TRIGGER IF NOT EXISTS cortex_pulse_states_immutable_delete
                BEFORE DELETE ON cortex_pulse_states BEGIN
                    SELECT RAISE(ABORT, 'cortex pulse states are immutable');
                END;

                CREATE TABLE IF NOT EXISTS cortex_checkpoints (
                    checkpoint_id TEXT PRIMARY KEY,
                    lineage_id TEXT NOT NULL REFERENCES cortex_lineage(lineage_id),
                    update_id TEXT,
                    model_sha256 TEXT NOT NULL,
                    artifact_path TEXT NOT NULL,
                    artifact_sha256 TEXT NOT NULL,
                    curriculum_stage TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS cortex_output_states (
                    event_id TEXT PRIMARY KEY
                        REFERENCES self_internal_output_events(event_id),
                    parent_pulse INTEGER NOT NULL
                        REFERENCES self_pulse_states(pulse),
                    sequence INTEGER NOT NULL,
                    lineage_id TEXT NOT NULL REFERENCES cortex_lineage(lineage_id),
                    model_sha256 TEXT NOT NULL,
                    state_sha256 TEXT NOT NULL UNIQUE,
                    graph_field_sha256 TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    hidden_shape_json TEXT NOT NULL,
                    hidden_state BLOB NOT NULL,
                    byte_count INTEGER NOT NULL,
                    stopped INTEGER NOT NULL,
                    stop_probability REAL NOT NULL,
                    stop_source TEXT NOT NULL DEFAULT 'none',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(parent_pulse, sequence)
                );

                CREATE TRIGGER IF NOT EXISTS cortex_output_states_immutable_update
                BEFORE UPDATE ON cortex_output_states BEGIN
                    SELECT RAISE(ABORT, 'cortex output states are immutable');
                END;

                CREATE TRIGGER IF NOT EXISTS cortex_output_states_immutable_delete
                BEFORE DELETE ON cortex_output_states BEGIN
                    SELECT RAISE(ABORT, 'cortex output states are immutable');
                END;

                CREATE TABLE IF NOT EXISTS cortex_plasticity_receipts (
                    update_id TEXT PRIMARY KEY,
                    lineage_id TEXT NOT NULL REFERENCES cortex_lineage(lineage_id),
                    episode_ids_json TEXT NOT NULL,
                    record_ids_json TEXT NOT NULL,
                    route_ids_json TEXT NOT NULL,
                    payload_hashes_json TEXT NOT NULL,
                    before_model_sha256 TEXT NOT NULL,
                    after_model_sha256 TEXT NOT NULL,
                    losses_json TEXT NOT NULL,
                    episode_evidence_json TEXT NOT NULL,
                    verified_behavior_examples INTEGER NOT NULL,
                    optimization_steps INTEGER NOT NULL,
                    checkpoint_id TEXT NOT NULL REFERENCES cortex_checkpoints(checkpoint_id),
                    curriculum_stage TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TRIGGER IF NOT EXISTS cortex_plasticity_immutable_update
                BEFORE UPDATE ON cortex_plasticity_receipts BEGIN
                    SELECT RAISE(ABORT, 'cortex plasticity receipts are immutable');
                END;

                CREATE TRIGGER IF NOT EXISTS cortex_plasticity_immutable_delete
                BEFORE DELETE ON cortex_plasticity_receipts BEGIN
                    SELECT RAISE(ABORT, 'cortex plasticity receipts are immutable');
                END;
                """
            )
            pulse_columns = {
                row["name"]
                for row in self.connection.execute(
                    "PRAGMA table_info(cortex_pulse_states)"
                ).fetchall()
            }
            if "graph_field" not in pulse_columns:
                self.connection.execute(
                    "ALTER TABLE cortex_pulse_states ADD COLUMN graph_field BLOB"
                )
            if "graph_field_sha256" not in pulse_columns:
                self.connection.execute(
                    "ALTER TABLE cortex_pulse_states ADD COLUMN graph_field_sha256 TEXT"
                )
            plasticity_columns = {
                row["name"]
                for row in self.connection.execute(
                    "PRAGMA table_info(cortex_plasticity_receipts)"
                ).fetchall()
            }
            if "episode_evidence_json" not in plasticity_columns:
                self.connection.execute(
                    """ALTER TABLE cortex_plasticity_receipts
                       ADD COLUMN episode_evidence_json TEXT NOT NULL DEFAULT '{}'"""
                )
            if "optimization_steps" not in plasticity_columns:
                self.connection.execute(
                    """ALTER TABLE cortex_plasticity_receipts
                       ADD COLUMN optimization_steps INTEGER NOT NULL DEFAULT 1"""
                )
            output_columns = {
                row["name"]
                for row in self.connection.execute(
                    "PRAGMA table_info(cortex_output_states)"
                ).fetchall()
            }
            if "stop_source" not in output_columns:
                self.connection.execute(
                    """ALTER TABLE cortex_output_states
                       ADD COLUMN stop_source TEXT NOT NULL DEFAULT 'none'"""
                )

    def _bind_lineage(self) -> None:
        architecture = _json(asdict(self.config))
        rows = self.connection.execute("SELECT * FROM cortex_lineage").fetchall()
        if rows:
            if len(rows) != 1:
                raise CortexLineageError("mind contains multiple cortex lineages")
            row = rows[0]
            if row["lineage_id"] != self.config.architecture_id:
                raise CortexLineageError("cortex architecture does not match persisted lineage")
            if row["architecture_json"] != architecture:
                raise CortexLineageError("cortex configuration drifted from persisted lineage")
            if row["pretrained_source"] is not None:
                raise CortexLineageError("pretrained cortex lineage is forbidden")
            return
        with self.connection:
            self.connection.execute(
                """INSERT INTO cortex_lineage(
                       lineage_id, schema_id, architecture_json, initialization,
                       seed, parameter_count, pretrained_source
                   ) VALUES (?, ?, ?, ?, ?, ?, NULL)""",
                (
                    self.config.architecture_id,
                    CORTEX_SCHEMA,
                    architecture,
                    self.config.initialization,
                    self.config.seed,
                    self.parameter_count,
                ),
            )

    def _load_latest_checkpoint(self) -> None:
        row = self.connection.execute(
            "SELECT * FROM cortex_checkpoints ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return
        path = Path(row["artifact_path"]).expanduser().resolve()
        if self.checkpoint_directory not in path.parents:
            raise CortexLineageError("cortex checkpoint escaped its lineage directory")
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != row["artifact_sha256"]:
            raise CortexLineageError("cortex checkpoint artifact hash mismatch")
        checkpoint = torch.load(
            io.BytesIO(payload), map_location=self.device, weights_only=True
        )
        if checkpoint.get("lineage_id") != self.config.architecture_id:
            raise CortexLineageError("checkpoint belongs to another cortex lineage")
        if checkpoint.get("initialization") != CORTEX_INITIALIZATION:
            raise CortexLineageError("checkpoint is not descended from random birth")
        self.model.load_state_dict(checkpoint["model"])
        self._model_sha256_cache = None
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(self.device)
        self._require_finite_model(context="while loading a checkpoint")
        if self.model_sha256 != row["model_sha256"]:
            raise CortexLineageError("loaded model hash does not match its receipt")

    def _latest_hidden(self, connection: Any, *, before_pulse: int | None = None) -> Tensor:
        if before_pulse is None and self.continuous_hidden is not None:
            return self.continuous_hidden.detach().clone()
        if before_pulse is None:
            pulse_row = connection.execute(
                """SELECT hidden_state, hidden_shape_json, model_sha256
                          , pulse AS state_pulse, -1 AS state_sequence
                   FROM cortex_pulse_states ORDER BY pulse DESC LIMIT 1"""
            ).fetchone()
            output_row = connection.execute(
                """SELECT hidden_state, hidden_shape_json, model_sha256,
                          parent_pulse AS state_pulse, sequence AS state_sequence
                   FROM cortex_output_states
                   ORDER BY parent_pulse DESC, sequence DESC LIMIT 1"""
            ).fetchone()
        else:
            pulse_row = connection.execute(
                """SELECT hidden_state, hidden_shape_json, model_sha256
                          , pulse AS state_pulse, -1 AS state_sequence
                   FROM cortex_pulse_states
                   WHERE pulse < ? ORDER BY pulse DESC LIMIT 1""",
                (int(before_pulse),),
            ).fetchone()
            output_row = connection.execute(
                """SELECT hidden_state, hidden_shape_json, model_sha256,
                          parent_pulse AS state_pulse, sequence AS state_sequence
                   FROM cortex_output_states
                   WHERE parent_pulse < ?
                   ORDER BY parent_pulse DESC, sequence DESC LIMIT 1""",
                (int(before_pulse),),
            ).fetchone()
        rows = [row for row in (pulse_row, output_row) if row is not None]
        row = (
            max(
                rows,
                key=lambda item: (
                    int(item["state_pulse"]), int(item["state_sequence"])
                ),
            )
            if rows
            else None
        )
        if row is None or str(row["model_sha256"]) != self.model_sha256:
            # Consolidation changes the recurrent coordinate system. Until a
            # learned state migrator exists, wake from graph state rather than
            # replaying text or pretending an old hidden vector is compatible.
            return self.model.initial_hidden(
                1, device=self.device, dtype=self.compute_dtype
            )
        return _deserialize_hidden(
            bytes(row["hidden_state"]),
            json.loads(row["hidden_shape_json"]),
            device=self.device,
            dtype=self.compute_dtype,
        )

    def _advance_sequence(
        self,
        payload: bytes,
        field: Sequence[float],
        *,
        direction: str,
        hidden: Tensor,
    ) -> Tensor:
        field_tensor = torch.tensor(
            [list(field)], device=self.device, dtype=self.compute_dtype
        )
        direction_tensor = torch.tensor(
            [DIRECTION_IDS[direction]], device=self.device, dtype=torch.long
        )
        self.model.eval()
        with torch.no_grad():
            raw = bytes(payload)
            if not raw:
                raw_chunks = (b"",)
            else:
                chunk_bytes = self.config.maximum_event_bytes
                raw_chunks = tuple(
                    raw[start : start + chunk_bytes]
                    for start in range(0, len(raw), chunk_bytes)
                )
            for chunk in raw_chunks:
                previous = list(chunk) if chunk else [-1]
                previous_tensor = torch.tensor(
                    [previous], device=self.device, dtype=torch.long
                )
                _, hidden = self.model(
                    previous_tensor, field_tensor, direction_tensor, hidden
                )
        return hidden

    def propose(
        self,
        graph_field: Sequence[float],
        *,
        heard_payloads: Sequence[bytes] = (),
    ) -> CortexProposal:
        """Evaluate the current state without mutating cortex or SELF state."""
        if len(graph_field) != self.config.graph_width:
            raise ValueError("proposal graph field width mismatch")
        if not all(math.isfinite(float(value)) for value in graph_field):
            raise ValueError("proposal graph field contains a non-finite value")

        sensed_payloads = tuple(bytes(payload) for payload in heard_payloads if payload)
        payload_digest = hashlib.sha256()
        for payload in sensed_payloads:
            payload_digest.update(len(payload).to_bytes(4, "big"))
            payload_digest.update(payload)
        payload_hash = payload_digest.hexdigest()
        field_payload = struct.pack(
            f"<{self.config.graph_width}f", *(float(value) for value in graph_field)
        )
        field_hash = hashlib.sha256(field_payload).hexdigest()
        graph_tensor = torch.tensor(
            [[float(value) for value in graph_field]],
            device=self.device,
            dtype=self.compute_dtype,
        )
        hidden = self._latest_hidden(self.connection)
        outputs: Mapping[str, Tensor] | None = None
        payload_sequence = sensed_payloads or (b"",)
        self.model.eval()
        with torch.no_grad():
            for payload in payload_sequence:
                chunk_bytes = self.config.maximum_event_bytes
                chunks = (
                    tuple(payload[start : start + chunk_bytes]
                          for start in range(0, len(payload), chunk_bytes))
                    if payload else (b"",)
                )
                for chunk in chunks:
                    previous = list(chunk) if chunk else [-1]
                    previous_tensor = torch.tensor(
                        [previous], device=self.device, dtype=torch.long
                    )
                    direction = torch.tensor(
                        [DIRECTION_IDS["hear" if chunk else "quiet"]],
                        device=self.device,
                        dtype=torch.long,
                    )
                    outputs, hidden = self.model(
                        previous_tensor, graph_tensor, direction, hidden
                    )
        assert outputs is not None
        action_probabilities = torch.softmax(
            outputs["action_logits"].to(torch.float32), dim=-1
        )[0]
        probabilities = [float(value.cpu()) for value in action_probabilities]
        ordered = sorted(probabilities, reverse=True)
        entropy = -sum(
            probability * math.log(max(probability, 1e-12))
            for probability in probabilities
        )
        entropy_confidence = max(
            0.0, 1.0 - entropy / math.log(max(2, len(probabilities)))
        )
        margin = ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0]
        confidence = max(0.0, min(1.0, max(entropy_confidence, margin)))
        route_alignment = float(
            F.cosine_similarity(outputs["route"], graph_tensor)
            .to(torch.float32)
            .cpu()[0]
        )
        model_hash = self.model_sha256
        proposal_id = "cortex-proposal:" + hashlib.sha256(
            b"\0".join(
                (
                    model_hash.encode("ascii"),
                    field_hash.encode("ascii"),
                    payload_hash.encode("ascii"),
                )
            )
        ).hexdigest()[:32]
        by_action_id = {value: key for key, value in ACTION_IDS.items()}
        return CortexProposal(
            proposal_id=proposal_id,
            model_sha256=model_hash,
            graph_field_sha256=field_hash,
            sensory_payload_sha256=payload_hash,
            sensed_byte_count=sum(len(payload) for payload in sensed_payloads),
            action_probabilities={
                by_action_id[index].value: probability
                for index, probability in enumerate(probabilities)
            },
            predicted_consequence=float(
                outputs["consequence"].to(torch.float32).cpu()[0]
            ),
            predicted_valence=float(outputs["valence"].to(torch.float32).cpu()[0]),
            route_alignment=route_alignment,
            confidence=confidence,
        )

    def pulse_observer(self, connection: Any, frame: PulseCommitFrame) -> Mapping[str, Any]:
        """Advance and persist active cortex state inside a SELF transaction."""
        field = pulse_field(
            frame.recurrent,
            frame.context_node_ids,
            selected_outputs=frame.selected_outputs,
            width=self.config.graph_width,
        )
        routes = active_route_ids(frame.recurrent, frame.context_node_ids)
        prior = connection.execute(
            "SELECT model_sha256 FROM cortex_pulse_states ORDER BY pulse DESC LIMIT 1"
        ).fetchone()
        state_rebased = bool(
            self.continuous_hidden is None
            and prior is not None and str(prior["model_sha256"]) != self.model_sha256
        )
        hidden = self._latest_hidden(connection)
        sensed = 0
        heard_records = []
        for record in frame.input_records:
            causal_trunk = str(record.metadata.get("causal_trunk", ""))
            if causal_trunk != InputTrunk.HEAR.value:
                continue
            payload = record.text.encode("utf-8")
            hidden = self._advance_sequence(
                payload, field, direction="hear", hidden=hidden
            )
            sensed += len(payload)
            heard_records.append(record.record_id)
        if not heard_records:
            hidden = self._advance_sequence(
                b"", field, direction="quiet", hidden=hidden
            )
        hidden_payload, hidden_shape = _serialize_hidden(hidden)
        if self.continuous_hidden is not None:
            self.continuous_hidden = hidden.detach().clone()
        field_payload = struct.pack(f"<{len(field)}f", *field)
        field_hash = hashlib.sha256(field_payload).hexdigest()
        model_hash = self.model_sha256
        material = b"\0".join(
            (
                frame.pulse_id.encode("utf-8"),
                model_hash.encode("ascii"),
                hashlib.sha256(hidden_payload).hexdigest().encode("ascii"),
                field_hash.encode("ascii"),
            )
        )
        state_hash = hashlib.sha256(material).hexdigest()
        connection.execute(
            """INSERT INTO cortex_pulse_states(
                   pulse, pulse_id, lineage_id, model_sha256, state_sha256,
                   graph_field, graph_field_sha256,
                   hidden_shape_json, hidden_state, sensed_byte_count,
                   input_record_ids_json, route_ids_json, curriculum_stage
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                frame.pulse,
                frame.pulse_id,
                self.config.architecture_id,
                model_hash,
                state_hash,
                field_payload,
                field_hash,
                _json(hidden_shape),
                hidden_payload,
                sensed,
                _json([record.record_id for record in frame.input_records]),
                _json(routes),
                self.curriculum_stage,
            ),
        )
        return {
            "developmental_cortex": {
                "schema": CORTEX_SCHEMA,
                "lineage_id": self.config.architecture_id,
                "state_sha256": state_hash,
                "model_sha256": model_hash,
                "sensed_byte_count": sensed,
                "route_count": len(routes),
                "curriculum_stage": self.curriculum_stage,
                "transcript_window": False,
                "state_rebased_after_plasticity": state_rebased,
            }
        }

    def latest_pulse_state(self) -> CortexPulseState | None:
        row = self.connection.execute(
            "SELECT * FROM cortex_pulse_states ORDER BY pulse DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        field_payload = row["graph_field"]
        field_hash = row["graph_field_sha256"]
        if field_payload is None or field_hash is None:
            raise CortexLineageError(
                "persisted cortex state predates auditable graph fields"
            )
        field_bytes = bytes(field_payload)
        if hashlib.sha256(field_bytes).hexdigest() != str(field_hash):
            raise CortexLineageError("persisted cortex graph-field hash mismatch")
        expected_bytes = self.config.graph_width * 4
        if len(field_bytes) != expected_bytes:
            raise CortexLineageError("persisted cortex graph-field width is invalid")
        return CortexPulseState(
            pulse=int(row["pulse"]),
            pulse_id=str(row["pulse_id"]),
            state_sha256=str(row["state_sha256"]),
            model_sha256=str(row["model_sha256"]),
            graph_field=tuple(
                struct.unpack(f"<{self.config.graph_width}f", field_bytes)
            ),
            graph_field_sha256=str(field_hash),
            hidden_shape=tuple(json.loads(row["hidden_shape_json"])),
            sensed_byte_count=int(row["sensed_byte_count"]),
            input_record_ids=tuple(json.loads(row["input_record_ids_json"])),
            route_ids=tuple(json.loads(row["route_ids_json"])),
            curriculum_stage=str(row["curriculum_stage"]),
        )

    def _episode_loss(
        self,
        episode: CortexTrainingEpisode,
        *,
        hidden: Tensor | None = None,
    ) -> tuple[Tensor, Mapping[str, float], Tensor]:
        if len(episode.graph_field) != self.config.graph_width:
            raise ValueError("training episode graph field width mismatch")
        if episode.direction not in DIRECTION_IDS:
            raise ValueError("unknown cortex direction")
        # Consolidation may begin from zero, but later episodes in the same
        # pulse-ordered life pass receive the detached state produced by what
        # came before. This matches continuous inference without retaining or
        # replaying a transcript and bounds gradients at episode boundaries.
        if hidden is None:
            hidden = self.model.initial_hidden(
                1, device=self.device, dtype=self.compute_dtype
            )
        payload = bytes(episode.payload)
        previous = [-1, *payload]
        targets = (
            torch.tensor([list(payload)], device=self.device, dtype=torch.long)
            if payload
            else None
        )
        previous_tensor = torch.tensor([previous], device=self.device, dtype=torch.long)
        graph_tensor = torch.tensor(
            [list(episode.graph_field)], device=self.device, dtype=self.compute_dtype
        )
        for conditioning_payload in episode.conditioning_payloads:
            full_condition = bytes(conditioning_payload)
            if not full_condition:
                continue
            conditioning_direction = torch.tensor(
                [DIRECTION_IDS["hear"]],
                device=self.device,
                dtype=torch.long,
            )
            chunk_bytes = self.config.maximum_event_bytes
            for start in range(0, len(full_condition), chunk_bytes):
                conditioning_previous = torch.tensor(
                    [list(full_condition[start : start + chunk_bytes])],
                    device=self.device,
                    dtype=torch.long,
                )
                _, hidden = self.model(
                    conditioning_previous,
                    graph_tensor,
                    conditioning_direction,
                    hidden,
                )
        direction = torch.tensor(
            [DIRECTION_IDS[episode.direction]], device=self.device, dtype=torch.long
        )
        outputs, final_hidden = self.model(
            previous_tensor, graph_tensor, direction, hidden
        )
        parts: dict[str, Tensor] = {}
        if targets is not None:
            byte_logits = outputs["byte_logits"][:, :-1, :]
            byte_loss = F.cross_entropy(
                byte_logits.reshape(-1, 256),
                targets.reshape(-1),
            )
            if episode.direction == "speak":
                if episode.motor_rehearsal:
                    byte_loss = byte_loss * 0.25
                elif episode.verified and episode.reward > 0.0:
                    byte_loss = byte_loss * min(1.0, episode.reward)
                elif episode.verified and episode.reward < 0.0:
                    chosen = torch.softmax(byte_logits, dim=-1).gather(
                        -1, targets.unsqueeze(-1)
                    )
                    byte_loss = -torch.log((1.0 - chosen).clamp(min=1e-6)).mean()
                    byte_loss = byte_loss * min(1.0, abs(episode.reward))
                else:
                    byte_loss = byte_loss * 0.0
            parts["byte"] = byte_loss
        stop_targets = torch.zeros_like(outputs["stop_logits"])
        stop_targets[:, -1] = 1.0
        stop_loss = F.binary_cross_entropy_with_logits(
            outputs["stop_logits"], stop_targets
        )
        if episode.direction == "speak":
            if episode.motor_rehearsal:
                stop_loss = stop_loss * 0.25
            elif episode.verified:
                stop_loss = stop_loss * min(1.0, abs(episode.reward))
            else:
                stop_loss = stop_loss * 0.0
        parts["stop"] = stop_loss
        parts["route"] = F.mse_loss(outputs["route"], graph_tensor)
        if episode.negative_graph_field is not None:
            negative = torch.tensor(
                [list(episode.negative_graph_field)],
                device=self.device,
                dtype=self.compute_dtype,
            )
            positive_similarity = F.cosine_similarity(outputs["route"], graph_tensor)
            negative_similarity = F.cosine_similarity(outputs["route"], negative)
            parts["contrastive"] = F.relu(
                torch.tensor(0.25, device=self.device, dtype=self.compute_dtype)
                - positive_similarity
                + negative_similarity
            ).mean()
        if episode.verified:
            consequence_target = torch.tensor(
                [max(-1.0, min(1.0, episode.consequence))],
                device=self.device,
                dtype=self.compute_dtype,
            )
            valence_target = torch.tensor(
                [max(-1.0, min(1.0, episode.valence_delta))],
                device=self.device,
                dtype=self.compute_dtype,
            )
            parts["consequence"] = F.mse_loss(
                outputs["consequence"], consequence_target
            )
            parts["valence"] = F.mse_loss(outputs["valence"], valence_target)
            if episode.selected_action is not None:
                action_id = ACTION_IDS[episode.selected_action]
                probabilities = torch.softmax(outputs["action_logits"], dim=-1)[0]
                if episode.reward > 0.0:
                    parts["action"] = F.cross_entropy(
                        outputs["action_logits"],
                        torch.tensor([action_id], device=self.device),
                    ) * min(1.0, episode.reward)
                elif episode.reward < 0.0:
                    parts["action"] = -torch.log(
                        (1.0 - probabilities[action_id]).clamp(min=1e-6)
                    ) * min(1.0, abs(episode.reward))
        weights = {
            "byte": 1.0,
            "stop": 0.15,
            "route": 0.40,
            "contrastive": 0.30,
            "consequence": 0.20,
            "valence": 0.20,
            "action": 0.30,
        }
        total = sum(parts[name] * weights[name] for name in parts)
        metrics = {
            name: float(value.detach().to(torch.float32).cpu())
            for name, value in parts.items()
        }
        metrics["total"] = float(total.detach().to(torch.float32).cpu())
        if not all(math.isfinite(value) for value in metrics.values()):
            raise CortexNumericalError(
                f"non-finite loss in grounded episode {episode.episode_id!r}"
            )
        return total, metrics, final_hidden

    def _validate_training_episode(self, episode: CortexTrainingEpisode) -> None:
        if not episode.episode_id:
            raise ValueError("training episode ID cannot be empty")
        if episode.direction not in DIRECTION_IDS:
            raise ValueError("unknown cortex direction")
        if episode.selected_action is not None and not isinstance(
            episode.selected_action, OutputTrunk
        ):
            raise ValueError("selected action must be an output trunk")
        if episode.motor_rehearsal and episode.direction != "speak":
            raise ValueError("motor rehearsal requires the speak direction")
        if episode.motor_rehearsal and episode.verified:
            raise ValueError("motor rehearsal is not a verified external action")
        if episode.conditioning_payloads and episode.direction != "speak":
            raise ValueError(
                "current-turn conditioning is only valid for SPEAK learning"
            )
        for name, value in (
            ("consequence", episode.consequence),
            ("valence_delta", episode.valence_delta),
            ("reward", episode.reward),
        ):
            if not math.isfinite(value) or not -1.0 <= value <= 1.0:
                raise ValueError(f"training {name} must be finite and within [-1, 1]")
        if not episode.record_ids:
            raise ValueError("training episode requires canonical record evidence")
        if len(episode.graph_field) != self.config.graph_width:
            raise ValueError("training episode graph field width mismatch")
        if not all(math.isfinite(value) for value in episode.graph_field):
            raise ValueError("training graph field contains a non-finite value")
        if (
            episode.negative_graph_field is not None
            and len(episode.negative_graph_field) != self.config.graph_width
        ):
            raise ValueError("negative graph field width mismatch")
        if episode.negative_graph_field is not None and not all(
            math.isfinite(value) for value in episode.negative_graph_field
        ):
            raise ValueError("negative graph field contains a non-finite value")
        records = self.mind.store.get_records(episode.record_ids)
        if {record.record_id for record in records} != set(episode.record_ids):
            raise ValueError("training episode references a missing canonical record")
        if episode.payload:
            if episode.direction == "speak" and episode.verified:
                allowed_payloads = {
                    record.text.encode("utf-8")
                    for record in records
                    if record.record_type == RecordType.OUTBOUND_MESSAGE
                    and record.metadata.get("membrane_lane") == OutputTrunk.SPEAK.value
                    and record.metadata.get("developmental_self_generated") is True
                }
            else:
                allowed_payloads = {
                    record.text.encode("utf-8")
                    for record in records
                    if record.metadata.get("causal_trunk") == InputTrunk.HEAR.value
                }
            if bytes(episode.payload) not in allowed_payloads:
                raise ValueError(
                    "training payload is not exact admitted sensory or self speech evidence"
                )
        if not episode.route_ids:
            raise ValueError("training episode requires lived route evidence")
        missing_routes = [
            route_id
            for route_id in episode.route_ids
            if not self.mind.store.has_concept(route_id)
        ]
        if missing_routes:
            raise ValueError("training episode references routes absent from the mind")
        pulse_state = self.connection.execute(
            """SELECT graph_field, graph_field_sha256, input_record_ids_json
               FROM cortex_pulse_states WHERE pulse = ?""",
            (int(episode.pulse),),
        ).fetchone()
        if (
            pulse_state is None
            or pulse_state["graph_field"] is None
            or pulse_state["graph_field_sha256"] is None
        ):
            raise ValueError("training episode has no committed cortex pulse state")
        field_payload = struct.pack(
            f"<{self.config.graph_width}f", *episode.graph_field
        )
        if hashlib.sha256(field_payload).hexdigest() != str(
            pulse_state["graph_field_sha256"]
        ) or field_payload != bytes(pulse_state["graph_field"]):
            raise ValueError("training graph field differs from its committed pulse")
        pulse_input_ids = {
            str(item) for item in json.loads(pulse_state["input_record_ids_json"])
        }
        if episode.conditioning_payloads:
            allowed_conditions = {
                record.text.encode("utf-8")
                for record in records
                if record.record_id in pulse_input_ids
                and record.metadata.get("causal_trunk") == InputTrunk.HEAR.value
            }
            for conditioning_payload in episode.conditioning_payloads:
                full_condition = bytes(conditioning_payload)
                if not full_condition or full_condition not in allowed_conditions:
                    raise ValueError(
                        "speech conditioning is not exact current HEAR evidence"
                    )
        primary_record_ids = pulse_input_ids & set(episode.record_ids)
        if not primary_record_ids:
            raise ValueError("training evidence was not sensed in its claimed pulse")
        admissions_table = self.connection.execute(
            """SELECT 1 FROM sqlite_master
               WHERE type='table' AND name='developmental_admissions'"""
        ).fetchone()
        if admissions_table is None:
            raise ValueError("training episode has no developmental curriculum")
        admissions = self.connection.execute(
            """SELECT record_id, episode_kind, route_ids_json
               FROM developmental_admissions
               WHERE record_id IN ({}) AND accepted = 1""".format(
                ",".join("?" for _ in primary_record_ids)
            ),
            tuple(primary_record_ids),
        ).fetchall()
        admitted = any(
            str(row["episode_kind"]) == episode.kind
            and tuple(json.loads(row["route_ids_json"])) == episode.route_ids
            for row in admissions
        )
        if not admitted:
            raise ValueError(
                "training routes and kind do not match an admitted experience"
            )
        if episode.verified:
            if not episode.experience_cycle_id or not episode.return_record_id:
                raise ValueError(
                    "verified behavioral learning requires a cycle and return receipt"
                )
            cycle = self.mind.store.get_experience_cycle(episode.experience_cycle_id)
            if cycle is None:
                raise ValueError("verified training cycle does not exist")
            if cycle.opened_pulse != episode.pulse:
                raise ValueError("verified training cycle belongs to another pulse")
            if (
                cycle.status != "closed"
                or cycle.terminal_return_record_id != episode.return_record_id
            ):
                raise ValueError("verified training requires the terminal cycle return")
            if cycle.output_record_id not in episode.record_ids:
                raise ValueError("training evidence omits its output record")
            if episode.return_record_id not in episode.record_ids:
                raise ValueError("training evidence omits its return record")
            returns = self.mind.store.returns_for_experience_cycle(cycle.cycle_id)
            observed = next(
                (
                    item
                    for item in returns
                    if item.record_id == episode.return_record_id and item.verified
                ),
                None,
            )
            if observed is None:
                raise ValueError("verified training return has no receipt")
            if episode.selected_action != cycle.output_trunk:
                raise ValueError("training action does not match its output cycle")
            if abs(float(episode.consequence) - observed.stability_delta) > 1e-9:
                raise ValueError("training consequence differs from verified return")
        elif (
            episode.selected_action is not None
            or episode.experience_cycle_id is not None
            or episode.return_record_id is not None
        ):
            raise ValueError("unverified episodes cannot carry behavioral evidence")

    def consolidate(
        self,
        episodes: Sequence[CortexTrainingEpisode],
        *,
        curriculum_stage: str,
        optimization_steps: int = 1,
    ) -> PlasticityReceipt:
        """Apply one bounded consolidation and emit an artifact-backed receipt."""
        if not episodes:
            raise ValueError("consolidation requires at least one episode")
        if int(optimization_steps) != optimization_steps or optimization_steps < 1:
            raise ValueError("optimization_steps must be a positive integer")
        if len({episode.episode_id for episode in episodes}) != len(episodes):
            raise ValueError("consolidation episode IDs must be unique")
        for episode in episodes:
            self._validate_training_episode(episode)
        before = self.model_sha256
        self.model.train()
        aggregate: dict[str, float] = {}
        steps = int(optimization_steps)
        ordered_episodes = tuple(
            episode
            for _, episode in sorted(
                enumerate(episodes),
                key=lambda item: (item[1].pulse, item[0]),
            )
        )
        for _ in range(steps):
            self.optimizer.zero_grad(set_to_none=True)
            hidden = self.model.initial_hidden(
                1, device=self.device, dtype=self.compute_dtype
            )
            for episode in ordered_episodes:
                loss, metrics, hidden = self._episode_loss(
                    episode, hidden=hidden
                )
                (loss / len(episodes)).backward()
                hidden = hidden.detach()
                for name, value in metrics.items():
                    aggregate[name] = aggregate.get(name, 0.0) + value / (
                        len(episodes) * steps
                    )
            try:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip,
                    error_if_nonfinite=True,
                )
            except RuntimeError as exc:
                self.optimizer.zero_grad(set_to_none=True)
                raise CortexNumericalError(
                    "non-finite cortex gradient; consolidation was not receipted"
                ) from exc
            self.optimizer.step()
            self._model_sha256_cache = None
        self._require_finite_model(context="after consolidation")
        self._model_sha256_cache = None
        after = self.model_sha256
        update_id = "cortex-update:" + uuid.uuid4().hex
        checkpoint_id = "cortex-checkpoint:" + uuid.uuid4().hex
        record_ids = tuple(
            dict.fromkeys(record for episode in episodes for record in episode.record_ids)
        )
        route_ids = tuple(
            dict.fromkeys(route for episode in episodes for route in episode.route_ids)
        )
        payload_hashes = {
            episode.episode_id: hashlib.sha256(episode.payload).hexdigest()
            for episode in episodes
        }
        episode_evidence = {
            episode.episode_id: {
                "pulse": episode.pulse,
                "record_ids": list(episode.record_ids),
                "route_ids": list(episode.route_ids),
                "graph_field_sha256": hashlib.sha256(
                    struct.pack(
                        f"<{self.config.graph_width}f", *episode.graph_field
                    )
                ).hexdigest(),
                "negative_graph_field_sha256": (
                    hashlib.sha256(
                        struct.pack(
                            f"<{self.config.graph_width}f",
                            *episode.negative_graph_field,
                        )
                    ).hexdigest()
                    if episode.negative_graph_field is not None
                    else None
                ),
                "payload_sha256": payload_hashes[episode.episode_id],
                "conditioning_payload_sha256": [
                    hashlib.sha256(bytes(payload)).hexdigest()
                    for payload in episode.conditioning_payloads
                ],
                "direction": episode.direction,
                "kind": episode.kind,
                "consequence": episode.consequence,
                "valence_delta": episode.valence_delta,
                "selected_action": (
                    episode.selected_action.value
                    if episode.selected_action is not None
                    else None
                ),
                "verified": episode.verified,
                "reward": episode.reward,
                "experience_cycle_id": episode.experience_cycle_id,
                "return_record_id": episode.return_record_id,
                "motor_rehearsal": episode.motor_rehearsal,
                "training_state_mode": "pulse_order_complete_chunked_recurrent_carry",
            }
            for episode in episodes
        }
        evidence_json = _json(episode_evidence)
        evidence_hash = hashlib.sha256(evidence_json.encode("utf-8")).hexdigest()
        artifact = self.checkpoint_directory / f"{checkpoint_id.replace(':', '-')}.pt"
        buffer = io.BytesIO()
        torch.save(
            {
                "schema": CORTEX_SCHEMA,
                "lineage_id": self.config.architecture_id,
                "initialization": CORTEX_INITIALIZATION,
                "config": asdict(self.config),
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "curriculum_stage": str(curriculum_stage),
                "update_id": update_id,
                "episode_evidence_sha256": evidence_hash,
                "optimization_steps": steps,
            },
            buffer,
        )
        checkpoint_bytes = buffer.getvalue()
        temporary = artifact.with_suffix(".tmp")
        temporary.write_bytes(checkpoint_bytes)
        os.replace(temporary, artifact)
        artifact_hash = hashlib.sha256(checkpoint_bytes).hexdigest()
        with self.mind.store.transaction() as connection:
            connection.execute(
                """INSERT INTO cortex_checkpoints(
                       checkpoint_id, lineage_id, update_id, model_sha256,
                       artifact_path, artifact_sha256, curriculum_stage
                   ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    checkpoint_id,
                    self.config.architecture_id,
                    update_id,
                    after,
                    str(artifact),
                    artifact_hash,
                    str(curriculum_stage),
                ),
            )
            connection.execute(
                """INSERT INTO cortex_plasticity_receipts(
                       update_id, lineage_id, episode_ids_json, record_ids_json,
                       route_ids_json, payload_hashes_json, before_model_sha256,
                       after_model_sha256, losses_json,
                       episode_evidence_json, verified_behavior_examples,
                       optimization_steps, checkpoint_id, curriculum_stage
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    update_id,
                    self.config.architecture_id,
                    _json([episode.episode_id for episode in episodes]),
                    _json(record_ids),
                    _json(route_ids),
                    _json(payload_hashes),
                    before,
                    after,
                    _json(aggregate),
                    evidence_json,
                    sum(
                        1
                        for episode in episodes
                        if episode.verified and episode.selected_action is not None
                    ),
                    steps,
                    checkpoint_id,
                    str(curriculum_stage),
                ),
            )
        self.curriculum_stage = str(curriculum_stage)
        return PlasticityReceipt(
            update_id=update_id,
            episode_ids=tuple(episode.episode_id for episode in episodes),
            before_model_sha256=before,
            after_model_sha256=after,
            checkpoint_path=str(artifact),
            checkpoint_sha256=artifact_hash,
            evidence_sha256=evidence_hash,
            losses=aggregate,
            verified_behavior_examples=sum(
                1
                for episode in episodes
                if episode.verified and episode.selected_action is not None
            ),
            optimization_steps=steps,
            curriculum_stage=str(curriculum_stage),
        )

    def evaluate_episode(
        self, episode: CortexTrainingEpisode
    ) -> CortexEpisodeEvaluation:
        """Read a held-out grounded episode without changing weights or state."""
        if len(episode.graph_field) != self.config.graph_width:
            raise ValueError("evaluation episode graph field width mismatch")
        payload = bytes(episode.payload)
        previous = [-1, *payload]
        targets = (
            torch.tensor([list(payload)], device=self.device, dtype=torch.long)
            if payload
            else None
        )
        hidden = self._latest_hidden(self.connection, before_pulse=episode.pulse)
        previous_tensor = torch.tensor([previous], device=self.device, dtype=torch.long)
        graph_tensor = torch.tensor(
            [list(episode.graph_field)], device=self.device, dtype=self.compute_dtype
        )
        direction = torch.tensor(
            [DIRECTION_IDS[episode.direction]], device=self.device, dtype=torch.long
        )
        self.model.eval()
        with torch.no_grad():
            outputs, _ = self.model(previous_tensor, graph_tensor, direction, hidden)
            top_1 = None
            top_5 = None
            if targets is not None:
                logits = outputs["byte_logits"][:, :-1, :]
                top_1 = float(
                    (logits.argmax(dim=-1) == targets).to(torch.float32).mean().cpu()
                )
                top_5_ids = logits.topk(5, dim=-1).indices
                top_5 = float(
                    (top_5_ids == targets.unsqueeze(-1))
                    .any(dim=-1)
                    .to(torch.float32)
                    .mean()
                    .cpu()
                )
            stop_targets = torch.zeros_like(outputs["stop_logits"])
            stop_targets[:, -1] = 1.0
            stop_accuracy = float(
                (
                    (torch.sigmoid(outputs["stop_logits"]) >= 0.5)
                    == (stop_targets >= 0.5)
                )
                .to(torch.float32)
                .mean()
                .cpu()
            )
            route_cosine = float(
                F.cosine_similarity(outputs["route"], graph_tensor)
                .to(torch.float32)
                .cpu()[0]
            )
            action_probabilities = torch.softmax(
                outputs["action_logits"].to(torch.float32), dim=-1
            )[0]
            selected_action_id = int(action_probabilities.argmax().cpu())
            action_by_id = {value: key for key, value in ACTION_IDS.items()}
            expected_probability = (
                float(action_probabilities[ACTION_IDS[episode.selected_action]].cpu())
                if episode.selected_action is not None
                else None
            )
            return CortexEpisodeEvaluation(
                episode_id=episode.episode_id,
                byte_top_1=top_1,
                byte_top_5=top_5,
                route_cosine=route_cosine,
                predicted_consequence=float(
                    outputs["consequence"].to(torch.float32).cpu()[0]
                ),
                predicted_valence=float(
                    outputs["valence"].to(torch.float32).cpu()[0]
                ),
                selected_action=action_by_id[selected_action_id],
                expected_action_probability=expected_probability,
                stop_accuracy=stop_accuracy,
            )

    def generate_bytes(
        self,
        graph_field: Sequence[float],
        *,
        maximum_bytes: int | None = None,
        minimum_bytes: int = 1,
        temperature: float = 0.0,
        stop_policy: Callable[[bytes, int], bool] | None = None,
        learned_stop_threshold: float | None = None,
        valid_utf8_only: bool = True,
    ) -> CortexGeneration:
        """Generate without text history using a learned motor boundary."""
        if len(graph_field) != self.config.graph_width:
            raise ValueError("generation graph field width mismatch")
        hidden = self._latest_hidden(self.connection)
        initial_payload, _ = _serialize_hidden(hidden)
        initial_hash = hashlib.sha256(initial_payload).hexdigest()
        condition = torch.tensor(
            [list(graph_field)], device=self.device, dtype=self.compute_dtype
        )
        direction = torch.tensor(
            [DIRECTION_IDS["speak"]], device=self.device, dtype=torch.long
        )
        generated = bytearray()
        probabilities = []
        previous = -1
        stopped = False
        stop_probability = 0.0
        stop_source = "none"
        stop_threshold = (
            self.config.learned_stop_threshold
            if learned_stop_threshold is None
            else float(learned_stop_threshold)
        )
        if not 0.5 < stop_threshold < 1.0:
            raise ValueError("learned stop threshold must be within (0.5, 1.0)")
        self.model.eval()
        with torch.no_grad():
            steps = itertools.count() if maximum_bytes is None else range(max(1, int(maximum_bytes)))
            for step in steps:
                previous_tensor = torch.tensor(
                    [[previous]], device=self.device, dtype=torch.long
                )
                outputs, hidden = self.model(
                    previous_tensor, condition, direction, hidden
                )
                stop_probability = float(
                    torch.sigmoid(outputs["stop_logits"][0, -1])
                    .to(torch.float32)
                    .cpu()
                )
                if (
                    len(generated) >= max(1, int(minimum_bytes))
                    and stop_probability >= stop_threshold
                    and (not valid_utf8_only or _complete_utf8(bytes(generated)))
                ):
                    stopped = True
                    stop_source = "learned_neural"
                    break
                logits = outputs["byte_logits"][0, -1].to(torch.float32)
                if valid_utf8_only:
                    valid = torch.tensor(
                        [
                            _valid_utf8_prefix(bytes(generated) + bytes((candidate,)))
                            for candidate in range(256)
                        ],
                        device=self.device,
                        dtype=torch.bool,
                    )
                    logits = logits.masked_fill(~valid, float("-inf"))
                if temperature > 0.0:
                    distribution = torch.softmax(logits / max(0.05, temperature), dim=-1)
                    selected = int(torch.multinomial(distribution, 1).item())
                    probability = float(distribution[selected].cpu())
                else:
                    distribution = torch.softmax(logits, dim=-1)
                    selected = int(torch.argmax(distribution).item())
                    probability = float(distribution[selected].cpu())
                generated.append(selected)
                probabilities.append(probability)
                previous = selected
                if (
                    len(generated) >= max(1, int(minimum_bytes))
                    and stop_policy is not None
                    and stop_policy(bytes(generated), step)
                ):
                    stopped = True
                    stop_source = "external_policy"
                    break
        final_payload, final_shape = _serialize_hidden(hidden)
        return CortexGeneration(
            payload=bytes(generated),
            step_probabilities=tuple(probabilities),
            initial_state_sha256=initial_hash,
            final_hidden=final_payload,
            final_hidden_shape=final_shape,
            stopped=stopped,
            stop_probability=stop_probability,
            stop_source=stop_source,
        )

    def score_sequences(
        self,
        graph_field: Sequence[float],
        payloads: Sequence[bytes],
    ) -> tuple[CortexSequenceScore, ...]:
        """Compare bounded graph-native motor options without changing state.

        The graph decides which whole forms are eligible. This method sees only
        those byte trajectories and the current numeric field; it neither scans
        memory records nor invents a response candidate.
        """
        if len(graph_field) != self.config.graph_width:
            raise ValueError("sequence scoring graph field width mismatch")
        if not all(math.isfinite(float(value)) for value in graph_field):
            raise ValueError("sequence scoring graph field is non-finite")
        candidates = tuple(dict.fromkeys(bytes(item) for item in payloads))
        if not candidates:
            return ()
        for payload in candidates:
            if not payload:
                raise ValueError("sequence scoring payload cannot be empty")
            if not _complete_utf8(payload):
                raise ValueError("sequence scoring payload must be complete UTF-8")

        hidden = self._latest_hidden(self.connection)
        initial_payload, _ = _serialize_hidden(hidden)
        initial_hash = hashlib.sha256(initial_payload).hexdigest()
        graph_tensor = torch.tensor(
            [list(graph_field)], device=self.device, dtype=self.compute_dtype
        )
        direction = torch.tensor(
            [DIRECTION_IDS["speak"]], device=self.device, dtype=torch.long
        )
        raw: list[tuple[bytes, float, float, float]] = []
        self.model.eval()
        with torch.no_grad():
            for payload in candidates:
                previous = torch.tensor(
                    [[-1, *payload]], device=self.device, dtype=torch.long
                )
                outputs, _ = self.model(
                    previous, graph_tensor, direction, hidden.clone()
                )
                targets = torch.tensor(
                    [list(payload)], device=self.device, dtype=torch.long
                )
                byte_log_probabilities = torch.log_softmax(
                    outputs["byte_logits"][:, :-1, :].to(torch.float32), dim=-1
                ).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                mean_byte_log = float(byte_log_probabilities.mean().cpu())

                stop_probabilities = torch.sigmoid(
                    outputs["stop_logits"].to(torch.float32)
                ).clamp(min=1e-7, max=1.0 - 1e-7)
                boundary_parts = []
                if stop_probabilities.shape[1] > 1:
                    boundary_parts.append(
                        torch.log1p(-stop_probabilities[:, :-1]).mean()
                    )
                boundary_parts.append(torch.log(stop_probabilities[:, -1]).mean())
                boundary_log = float(torch.stack(boundary_parts).mean().cpu())
                # Length-normalized scoring lets short and long learned motor
                # units compete without an automatic short-string advantage.
                sequence_score = mean_byte_log + 0.25 * boundary_log
                raw.append((payload, mean_byte_log, boundary_log, sequence_score))

            competition = torch.softmax(
                torch.tensor(
                    [item[3] for item in raw],
                    device=self.device,
                    dtype=torch.float32,
                ),
                dim=0,
            ).cpu()
        model_hash = self.model_sha256
        return tuple(
            CortexSequenceScore(
                payload_sha256=hashlib.sha256(payload).hexdigest(),
                byte_count=len(payload),
                mean_byte_log_probability=mean_byte_log,
                boundary_log_probability=boundary_log,
                sequence_score=sequence_score,
                competition_probability=float(competition[index]),
                initial_state_sha256=initial_hash,
                model_sha256=model_hash,
            )
            for index, (payload, mean_byte_log, boundary_log, sequence_score) in enumerate(raw)
        )

    def motorize_bytes(
        self,
        graph_field: Sequence[float],
        payload: bytes,
        *,
        stop_source: str = "promoted_whole_form",
    ) -> CortexGeneration:
        """Advance the cortex through an exact self-selected learned motor unit."""
        resolved = bytes(payload)
        if len(graph_field) != self.config.graph_width:
            raise ValueError("motor graph field width mismatch")
        if not resolved:
            raise ValueError("motor payload cannot be empty")
        if not _complete_utf8(resolved):
            raise ValueError("motor payload must be complete UTF-8")
        if not str(stop_source):
            raise ValueError("motor stop source cannot be empty")

        hidden = self._latest_hidden(self.connection)
        initial_payload, _ = _serialize_hidden(hidden)
        initial_hash = hashlib.sha256(initial_payload).hexdigest()
        previous = torch.tensor(
            [[-1, *resolved]], device=self.device, dtype=torch.long
        )
        targets = torch.tensor(
            [list(resolved)], device=self.device, dtype=torch.long
        )
        graph_tensor = torch.tensor(
            [list(graph_field)], device=self.device, dtype=self.compute_dtype
        )
        direction = torch.tensor(
            [DIRECTION_IDS["speak"]], device=self.device, dtype=torch.long
        )
        self.model.eval()
        with torch.no_grad():
            outputs, final_hidden = self.model(
                previous, graph_tensor, direction, hidden
            )
            distributions = torch.softmax(
                outputs["byte_logits"][:, :-1, :].to(torch.float32), dim=-1
            )
            step_probabilities = distributions.gather(
                -1, targets.unsqueeze(-1)
            ).squeeze(-1)[0]
            stop_probability = float(
                torch.sigmoid(outputs["stop_logits"][0, -1])
                .to(torch.float32)
                .cpu()
            )
        final_payload, final_shape = _serialize_hidden(final_hidden)
        return CortexGeneration(
            payload=resolved,
            step_probabilities=tuple(float(item) for item in step_probabilities.cpu()),
            initial_state_sha256=initial_hash,
            final_hidden=final_payload,
            final_hidden_shape=final_shape,
            stopped=True,
            stop_probability=stop_probability,
            stop_source=str(stop_source),
        )

    def persist_generation(
        self,
        event_id: str,
        graph_field: Sequence[float],
        generation: CortexGeneration,
    ) -> CortexOutputState:
        """Make a generated motor sequence part of the continuing cortex state."""
        if len(graph_field) != self.config.graph_width:
            raise ValueError("generated output graph field width mismatch")
        internal = self.connection.execute(
            """SELECT parent_pulse, sequence FROM self_internal_output_events
               WHERE event_id = ?""",
            (str(event_id),),
        ).fetchone()
        if internal is None:
            raise ValueError("generated cortex state requires an internal output receipt")
        current_hidden = self._latest_hidden(self.connection)
        current_payload, _ = _serialize_hidden(current_hidden)
        if hashlib.sha256(current_payload).hexdigest() != generation.initial_state_sha256:
            raise ValueError("generated cortex state no longer follows the current state")
        final_hidden = _deserialize_hidden(
            generation.final_hidden,
            generation.final_hidden_shape,
            device=self.device,
            dtype=self.compute_dtype,
        )
        if not bool(torch.isfinite(final_hidden).all().item()):
            raise CortexNumericalError("generated cortex hidden state is non-finite")
        field_payload = struct.pack(
            f"<{self.config.graph_width}f", *(float(value) for value in graph_field)
        )
        field_hash = hashlib.sha256(field_payload).hexdigest()
        payload_hash = hashlib.sha256(generation.payload).hexdigest()
        final_hash = hashlib.sha256(generation.final_hidden).hexdigest()
        state_hash = hashlib.sha256(
            b"\0".join(
                (
                    str(event_id).encode("utf-8"),
                    self.model_sha256.encode("ascii"),
                    field_hash.encode("ascii"),
                    payload_hash.encode("ascii"),
                    final_hash.encode("ascii"),
                    generation.stop_source.encode("utf-8"),
                )
            )
        ).hexdigest()
        stop_probability = float(generation.stop_probability)
        if not math.isfinite(stop_probability) or not 0.0 <= stop_probability <= 1.0:
            raise ValueError("generated stop probability must be within [0, 1]")
        if not generation.stop_source:
            raise ValueError("generated stop source cannot be empty")
        parent_pulse = int(internal["parent_pulse"])
        sequence = int(internal["sequence"])
        with self.mind.store.transaction() as connection:
            connection.execute(
                """INSERT INTO cortex_output_states(
                       event_id, parent_pulse, sequence, lineage_id,
                       model_sha256, state_sha256, graph_field_sha256,
                       payload_sha256, hidden_shape_json, hidden_state,
                       byte_count, stopped, stop_probability, stop_source
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(event_id),
                    parent_pulse,
                    sequence,
                    self.config.architecture_id,
                    self.model_sha256,
                    state_hash,
                    field_hash,
                    payload_hash,
                    _json(generation.final_hidden_shape),
                    generation.final_hidden,
                    len(generation.payload),
                    int(generation.stopped),
                    stop_probability,
                    generation.stop_source,
                ),
            )
        return CortexOutputState(
            event_id=str(event_id),
            parent_pulse=parent_pulse,
            sequence=sequence,
            state_sha256=state_hash,
            model_sha256=self.model_sha256,
            graph_field_sha256=field_hash,
            payload_sha256=payload_hash,
            byte_count=len(generation.payload),
            stopped=generation.stopped,
            stop_probability=stop_probability,
            stop_source=generation.stop_source,
        )


def rocm_runtime_report() -> Mapping[str, Any]:
    """Return a side-effect-free report used by the executable GPU probe."""
    require_torch()
    available = bool(torch.cuda.is_available())
    hip = torch.version.hip
    result: dict[str, Any] = {
        "torch_version": torch.__version__,
        "hip_version": hip,
        "gpu_available": available,
        "backend": "rocm" if available and hip is not None else "cpu",
    }
    if available:
        result.update(
            {
                "device_count": torch.cuda.device_count(),
                "device_name": torch.cuda.get_device_name(0),
                "allocated_bytes": torch.cuda.memory_allocated(0),
                "reserved_bytes": torch.cuda.memory_reserved(0),
            }
        )
    return result


__all__ = [
    "ACTION_IDS",
    "BornInRecurrentCortex",
    "CORTEX_INITIALIZATION",
    "CORTEX_SCHEMA",
    "CortexConfig",
    "CortexDependencyError",
    "CortexEpisodeEvaluation",
    "CortexGeneration",
    "CortexLineageError",
    "CortexNumericalError",
    "CortexOutputState",
    "CortexPulseState",
    "CortexProposal",
    "CortexSequenceScore",
    "CortexTrainingEpisode",
    "DevelopmentalCortex",
    "PlasticityReceipt",
    "active_route_ids",
    "pulse_field",
    "require_torch",
    "rocm_runtime_report",
]
