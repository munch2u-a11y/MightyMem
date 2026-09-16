"""Unified, self-centred pulse authority for the recurrent Habitus runtime.

The kernel does not introduce a second focus model. Sensory routes collapse
into the existing :class:`RecurrentField`, which advances exactly once for a
complete frame. The resulting recurrent state, valued output opportunities,
and one-use output authorizations are committed as one durable transition.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
import hashlib
import json
import math
import time
import uuid
from typing import Any, Callable, Mapping, Sequence

from .graph import INPUT_NODE_IDS, SELF_ID
from .pipeline import BaseAgenticMemoryRAG
from .types import (
    EventKind,
    ExperienceCycle,
    GraphSide,
    InputTrunk,
    MemoryRecord,
    OutputTrunk,
    OutputDecision,
    RecordType,
    RecurrentSnapshot,
    TraversalTrace,
)


SENSORY_SETTLING_ORDER = (
    InputTrunk.NOTICE,
    InputTrunk.SEE,
    InputTrunk.HEAR,
)
ACTION_OPPORTUNITY_ORDER = (
    OutputTrunk.DO,
    OutputTrunk.LOOK,
    OutputTrunk.SPEAK,
)
OUTPUT_RETURN_LANES = {
    OutputTrunk.DO: InputTrunk.NOTICE,
    OutputTrunk.LOOK: InputTrunk.SEE,
    OutputTrunk.SPEAK: InputTrunk.HEAR,
}

_STRUCTURAL_NODE_IDS = {
    SELF_ID,
    *INPUT_NODE_IDS.values(),
    "OUT:DO",
    "OUT:LOOK",
    "OUT:SPEAK",
}


def conservative_token_cost(text: str) -> int:
    """Return a tokenizer-free upper-biased cost for a sensory bucket."""
    return max(1, math.ceil(len(str(text).encode("utf-8")) / 3))


@dataclass(frozen=True)
class SensoryBucket:
    item_id: str
    kind: str
    lane: InputTrunk
    content: str
    content_sha256: str
    source_id: str
    token_cost: int
    enqueued_pulse: int
    sequence_id: int
    concept_scores: Mapping[str, float]
    embedding: tuple[float, ...] | None
    metadata: Mapping[str, Any]
    allow_growth: bool = False
    cycle_id: str | None = None


@dataclass(frozen=True)
class AdmissionEffect:
    """Additional graph-native state produced while admitting one input."""

    excitation: Mapping[str, float]
    pressure_delta: Mapping[str, float] = field(default_factory=dict)
    context_node_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class SelfInputReceipt:
    pulse: int
    item_id: str
    kind: str
    lane: InputTrunk
    record_id: str
    token_cost: int
    concept_ids: tuple[str, ...]
    inward_traces: tuple[TraversalTrace, ...]
    inward_trace_ids: tuple[str, ...]
    pending_after: int


@dataclass(frozen=True)
class OutputCandidate:
    node_id: str
    trunk: OutputTrunk
    path_node_ids: tuple[str, ...]
    path_edge_ids: tuple[str, ...]
    route_probability: float
    transient_pull: float
    travel_time: float
    learned_support: float = 0.0
    learned_confidence: float = 0.0
    learned_expected_stability_delta: float = 0.0
    proposal_id: str | None = None


@dataclass(frozen=True)
class OutputAffordance:
    node_id: str
    trunk: OutputTrunk
    score: float
    relative_probability: float
    route_probability: float
    transient_pull: float
    travel_time: float
    path_node_ids: tuple[str, ...]
    path_edge_ids: tuple[str, ...]
    expected_stability_delta: float = 0.0
    uncertainty: float = 1.0
    learned_support: float = 0.0
    learned_confidence: float = 0.0
    learned_expected_stability_delta: float = 0.0
    proposal_id: str | None = None
    authorization_id: str | None = None


@dataclass(frozen=True)
class OutputOpportunity:
    trunk: OutputTrunk
    affordance: OutputAffordance | None


@dataclass(frozen=True)
class SelfState:
    pulse: int
    pulse_id: str
    cycle_id: str
    recurrent_state_sha256: str
    state_sha256: str
    perceived_stability: float
    free_energy: float
    input_item_ids: tuple[str, ...]
    input_record_ids: tuple[str, ...]
    inward_trace_ids: tuple[str, ...]
    settled_order: tuple[InputTrunk, ...]
    active_node_ids: tuple[str, ...]


@dataclass(frozen=True)
class SettlingCycleReceipt:
    cycle_id: str
    started_pulse: int
    completed_pulse: int
    receipts: tuple[SelfInputReceipt, ...]
    settled_order: tuple[InputTrunk, ...]
    self_state: SelfState


@dataclass(frozen=True)
class CognitiveCycleReceipt:
    cycle_id: str
    sensory: SettlingCycleReceipt
    recurrent: RecurrentSnapshot
    output_candidates: tuple[OutputAffordance, ...]
    output_opportunities: tuple[OutputOpportunity, ...]
    self_state: SelfState
    selected_output: OutputAffordance | None
    selected_outputs: tuple[OutputAffordance, ...]


@dataclass(frozen=True)
class PulseCommitFrame:
    """Read-only state offered to extensions inside the SELF transaction.

    Extensions may persist compact numeric state through the supplied SQLite
    connection.  They receive only records admitted in this pulse, never a
    recalled transcript or rendered context packet.
    """

    pulse: int
    pulse_id: str
    cycle_id: str
    input_records: tuple[MemoryRecord, ...]
    recurrent: RecurrentSnapshot
    context_node_ids: tuple[str, ...]
    selected_outputs: tuple[OutputAffordance, ...]
    perceived_stability: float
    free_energy: float


@dataclass(frozen=True)
class _DeferredOutcome:
    edge_ids: tuple[str, ...]
    stability_delta: float
    verified: bool


AdmissionObserver = Callable[
    [MemoryRecord, SensoryBucket, bool], AdmissionEffect | Mapping[str, float] | None
]
OutputCandidateProvider = Callable[
    [RecurrentSnapshot, str], Sequence[OutputCandidate]
]
PulseStateObserver = Callable[
    [Any, PulseCommitFrame], Mapping[str, Any] | None
]


def _bounded(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))


def _softmax(values: Sequence[float], temperature: float) -> list[float]:
    if not values:
        return []
    temperature = max(0.05, float(temperature))
    maximum = max(values)
    exponentials = [math.exp((value - maximum) / temperature) for value in values]
    total = sum(exponentials) or 1.0
    return [value / total for value in exponentials]


class SelfPulseKernel:
    """Collapse all active senses into one recurrent SELF transition.

    Input arrival order never determines cognitive order. At most a bounded
    number of pending items from NOTICE, SEE, and HEAR are admitted in that
    order under one pulse ID. Outputs are proposals until a caller claims a
    one-use authorization and successfully persists an experience cycle.
    """

    def __init__(
        self,
        mind: BaseAgenticMemoryRAG,
        *,
        maximum_bucket_tokens: int | None = None,
        maximum_output_candidates: int | None = None,
        entropy_temperature: float = 0.35,
        outcome_weight: float = 0.75,
        output_effort_weight: float = 0.08,
        output_uncertainty_weight: float = 0.04,
        learned_output_weight: float = 0.30,
        minimum_output_score: float = 0.0,
    ) -> None:
        self.mind = mind
        self.maximum_bucket_tokens = (
            None if maximum_bucket_tokens is None
            else max(1, int(maximum_bucket_tokens))
        )
        self.maximum_output_candidates = (
            None if maximum_output_candidates is None
            else max(1, int(maximum_output_candidates))
        )
        self.entropy_temperature = max(0.05, float(entropy_temperature))
        self.outcome_weight = max(0.0, float(outcome_weight))
        self.output_effort_weight = max(0.0, float(output_effort_weight))
        self.output_uncertainty_weight = max(
            0.0, float(output_uncertainty_weight)
        )
        self.learned_output_weight = max(0.0, float(learned_output_weight))
        self.minimum_output_score = float(minimum_output_score)
        self._closed = False
        self._create_schema()
        self._recover_claims()

    @property
    def connection(self):
        return self.mind.store.connection

    def _create_schema(self) -> None:
        with self.connection:
            self.connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS self_pulse_inbox (
                    sequence_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    item_id TEXT NOT NULL UNIQUE,
                    lane TEXT NOT NULL,
                    content TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    token_cost INTEGER NOT NULL,
                    enqueued_pulse INTEGER NOT NULL,
                    concept_scores_json TEXT NOT NULL DEFAULT '{}',
                    embedding_json TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    allow_growth INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'pending',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    record_id TEXT,
                    processed_pulse INTEGER,
                    error TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_self_pulse_inbox_status_lane
                    ON self_pulse_inbox(status, lane, sequence_id);

                CREATE TABLE IF NOT EXISTS self_pulse_states (
                    pulse INTEGER PRIMARY KEY,
                    pulse_id TEXT NOT NULL UNIQUE,
                    cycle_id TEXT NOT NULL UNIQUE,
                    recurrent_state_sha256 TEXT NOT NULL,
                    state_sha256 TEXT NOT NULL UNIQUE,
                    state_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TRIGGER IF NOT EXISTS self_pulse_states_are_immutable_update
                BEFORE UPDATE ON self_pulse_states BEGIN
                    SELECT RAISE(ABORT, 'self pulse states are immutable');
                END;

                CREATE TRIGGER IF NOT EXISTS self_pulse_states_are_immutable_delete
                BEFORE DELETE ON self_pulse_states BEGIN
                    SELECT RAISE(ABORT, 'self pulse states are immutable');
                END;

                CREATE TABLE IF NOT EXISTS self_output_authorizations (
                    authorization_id TEXT PRIMARY KEY,
                    pulse INTEGER NOT NULL REFERENCES self_pulse_states(pulse),
                    pulse_id TEXT NOT NULL,
                    trunk TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    path_node_ids_json TEXT NOT NULL,
                    path_edge_ids_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'selected',
                    cycle_id TEXT,
                    claimed_at REAL,
                    consumed_at REAL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_self_output_authorizations_status
                    ON self_output_authorizations(status, pulse, trunk);

                CREATE TABLE IF NOT EXISTS self_internal_output_events (
                    event_id TEXT PRIMARY KEY,
                    parent_pulse INTEGER NOT NULL REFERENCES self_pulse_states(pulse),
                    sequence INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    excitation_json TEXT NOT NULL,
                    pressure_delta_json TEXT NOT NULL,
                    state_before_sha256 TEXT NOT NULL,
                    state_after_sha256 TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(parent_pulse, sequence)
                );

                CREATE TRIGGER IF NOT EXISTS self_internal_events_are_immutable_update
                BEFORE UPDATE ON self_internal_output_events BEGIN
                    SELECT RAISE(ABORT, 'self internal output events are immutable');
                END;

                CREATE TRIGGER IF NOT EXISTS self_internal_events_are_immutable_delete
                BEFORE DELETE ON self_internal_output_events BEGIN
                    SELECT RAISE(ABORT, 'self internal output events are immutable');
                END;
                """
            )
            self.connection.execute(
                """UPDATE self_pulse_inbox
                   SET status = 'pending', error = 'recovered interrupted admission'
                   WHERE status = 'processing'"""
            )

    def _recover_claims(self) -> None:
        claimed = self.connection.execute(
            "SELECT authorization_id FROM self_output_authorizations WHERE status = 'claimed'"
        ).fetchall()
        if not claimed:
            return
        cycles = self.connection.execute(
            "SELECT cycle_id, metadata_json FROM experience_cycles"
        ).fetchall()
        by_authorization: dict[str, str] = {}
        for cycle in cycles:
            metadata = json.loads(cycle["metadata_json"] or "{}")
            authorization_id = metadata.get("self_pulse_authorization_id")
            if authorization_id:
                by_authorization[str(authorization_id)] = str(cycle["cycle_id"])
        with self.connection:
            for row in claimed:
                authorization_id = str(row["authorization_id"])
                cycle_id = by_authorization.get(authorization_id)
                if cycle_id is None:
                    self.connection.execute(
                        """UPDATE self_output_authorizations
                           SET status = 'selected', claimed_at = NULL
                           WHERE authorization_id = ?""",
                        (authorization_id,),
                    )
                else:
                    self.connection.execute(
                        """UPDATE self_output_authorizations
                           SET status = 'consumed', cycle_id = ?, consumed_at = ?
                           WHERE authorization_id = ?""",
                        (cycle_id, time.time(), authorization_id),
                    )

    @staticmethod
    def _bucket_from_row(row: Any) -> SensoryBucket:
        embedding = json.loads(row["embedding_json"]) if row["embedding_json"] else None
        metadata = json.loads(row["metadata_json"] or "{}")
        return SensoryBucket(
            item_id=str(row["item_id"]),
            kind=str(metadata.get("_self_pulse_kind", "input")),
            lane=InputTrunk(row["lane"]),
            content=str(row["content"]),
            content_sha256=str(row["content_sha256"]),
            source_id=str(row["source_id"]),
            token_cost=int(row["token_cost"]),
            enqueued_pulse=int(row["enqueued_pulse"]),
            sequence_id=int(row["sequence_id"]),
            concept_scores={
                str(key): float(value)
                for key, value in json.loads(row["concept_scores_json"] or "{}").items()
            },
            embedding=(
                tuple(float(value) for value in embedding)
                if embedding is not None
                else None
            ),
            metadata=metadata,
            allow_growth=bool(row["allow_growth"]),
            cycle_id=(
                str(metadata["_self_pulse_cycle_id"])
                if metadata.get("_self_pulse_cycle_id")
                else None
            ),
        )

    def enqueue_input(
        self,
        content: str,
        *,
        lane: InputTrunk | str,
        source_id: str = "environment",
        concept_ids: Sequence[str] = (),
        concept_scores: Mapping[str, float] | None = None,
        embedding: Sequence[float] | None = None,
        token_cost: int | None = None,
        metadata: Mapping[str, Any] | None = None,
        allow_growth: bool = False,
        item_id: str | None = None,
    ) -> SensoryBucket:
        if self._closed:
            raise RuntimeError("self pulse kernel is closed")
        resolved_lane = lane if isinstance(lane, InputTrunk) else InputTrunk(lane)
        resolved_content = str(content)
        if not resolved_content:
            raise ValueError("sensory content cannot be empty")
        resolved_embedding = (
            tuple(float(value) for value in embedding)
            if embedding is not None
            else None
        )
        if (
            resolved_embedding is not None
            and len(resolved_embedding) != self.mind.embedder.dimension
        ):
            raise ValueError("sensory embedding dimension mismatch")
        scores = {
            str(node_id): _bounded(float(score))
            for node_id, score in (concept_scores or {}).items()
        }
        for node_id in concept_ids:
            scores.setdefault(str(node_id), 1.0)
        resolved_item_id = item_id or f"self-input:{uuid.uuid4().hex}"
        digest = hashlib.sha256(resolved_content.encode("utf-8")).hexdigest()
        with self.connection:
            self.connection.execute(
                """INSERT INTO self_pulse_inbox(
                       item_id, lane, content, content_sha256, source_id,
                       token_cost, enqueued_pulse, concept_scores_json,
                       embedding_json, metadata_json, allow_growth
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    resolved_item_id,
                    resolved_lane.value,
                    resolved_content,
                    digest,
                    str(source_id),
                    max(1, int(token_cost or conservative_token_cost(resolved_content))),
                    self.mind.pulse,
                    json.dumps(scores, sort_keys=True, separators=(",", ":")),
                    (
                        json.dumps(resolved_embedding, separators=(",", ":"))
                        if resolved_embedding is not None
                        else None
                    ),
                    json.dumps(dict(metadata or {}), sort_keys=True, separators=(",", ":")),
                    int(bool(allow_growth)),
                ),
            )
        row = self.connection.execute(
            "SELECT * FROM self_pulse_inbox WHERE item_id = ?",
            (resolved_item_id,),
        ).fetchone()
        if row is None:
            raise RuntimeError("queued sensory input was not persisted")
        return self._bucket_from_row(row)

    def enqueue_cycle_return(
        self,
        cycle_id: str,
        content: str,
        *,
        lane: InputTrunk | str,
        status: str,
        stability_delta: float,
        verified: bool,
        terminal: bool = True,
        source_id: str = "environment",
        record_type: RecordType | str = RecordType.RECEIPT,
        return_concept_id: str | None = None,
        return_path_node_ids: Sequence[str] = (),
        concept_scores: Mapping[str, float] | None = None,
        evidence_quality: float = 1.0,
        embedding: Sequence[float] | None = None,
        token_cost: int | None = None,
        provenance: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        item_id: str | None = None,
        allow_cross_lane: bool = False,
    ) -> SensoryBucket:
        """Queue a verified or unverified consequence on its sensory lane."""
        for row in self.connection.execute(
            """SELECT * FROM self_pulse_inbox
               WHERE status IN ('pending', 'processing') ORDER BY sequence_id"""
        ).fetchall():
            existing = self._bucket_from_row(row)
            if (
                existing.kind == "cycle_return"
                and existing.cycle_id == str(cycle_id)
            ):
                return existing
        cycle = self.mind.experience_cycle(cycle_id)
        if cycle is None:
            raise KeyError(f"unknown experience cycle: {cycle_id}")
        if cycle.status != "open":
            raise ValueError(f"experience cycle is already closed: {cycle_id}")
        resolved_lane = lane if isinstance(lane, InputTrunk) else InputTrunk(lane)
        expected_lane = OUTPUT_RETURN_LANES[cycle.output_trunk]
        if resolved_lane != expected_lane and not allow_cross_lane:
            raise ValueError(
                f"{cycle.output_trunk.value} returns through {expected_lane.value}, "
                f"not {resolved_lane.value}"
            )
        resolved_scores = {
            str(node_id): float(score)
            for node_id, score in (concept_scores or {}).items()
        }
        if return_concept_id is not None:
            resolved_scores.setdefault(str(return_concept_id), 1.0)
        return self.enqueue_input(
            content,
            lane=resolved_lane,
            source_id=source_id,
            concept_scores=resolved_scores,
            embedding=embedding,
            token_cost=token_cost,
            metadata={
                **dict(metadata or {}),
                "_self_pulse_kind": "cycle_return",
                "_self_pulse_cycle_id": str(cycle_id),
                "_self_pulse_return_status": str(status),
                "_self_pulse_stability_delta": _bounded(
                    stability_delta, -1.0, 1.0
                ),
                "_self_pulse_verified": bool(verified),
                "_self_pulse_terminal": bool(terminal),
                "_self_pulse_record_type": (
                    record_type.value
                    if isinstance(record_type, RecordType)
                    else RecordType(record_type).value
                ),
                "_self_pulse_return_concept_id": return_concept_id,
                "_self_pulse_return_path_node_ids": list(return_path_node_ids),
                "_self_pulse_evidence_quality": _bounded(evidence_quality),
                "_self_pulse_provenance": dict(provenance or {}),
            },
            allow_growth=False,
            item_id=item_id,
        )

    def pending(self) -> tuple[SensoryBucket, ...]:
        rows = self.connection.execute(
            """SELECT * FROM self_pulse_inbox
               WHERE status = 'pending' ORDER BY sequence_id"""
        ).fetchall()
        return tuple(self._bucket_from_row(row) for row in rows)

    def retry_failed(self, item_id: str) -> SensoryBucket:
        with self.connection:
            changed = self.connection.execute(
                """UPDATE self_pulse_inbox
                   SET status = 'pending', error = NULL
                   WHERE item_id = ? AND status = 'failed'""",
                (str(item_id),),
            ).rowcount
        if changed != 1:
            raise ValueError("sensory item is not failed or does not exist")
        row = self.connection.execute(
            "SELECT * FROM self_pulse_inbox WHERE item_id = ?", (str(item_id),)
        ).fetchone()
        return self._bucket_from_row(row)

    def _select_frame(
        self,
        maximum_per_lane: int | None,
        item_ids: Sequence[str] | None = None,
    ) -> tuple[SensoryBucket, ...]:
        pending = self.pending()
        if item_ids is not None:
            requested = set(dict.fromkeys(str(value) for value in item_ids))
            available = {item.item_id for item in pending}
            missing = requested - available
            if missing:
                raise ValueError(
                    "sensory frame contains unavailable items: "
                    + ", ".join(sorted(missing))
                )
            pending = tuple(item for item in pending if item.item_id in requested)
            selected = []
            for lane in SENSORY_SETTLING_ORDER:
                lane_items = [item for item in pending if item.lane == lane]
                if (
                    maximum_per_lane is not None
                    and len(lane_items) > max(1, int(maximum_per_lane))
                ):
                    raise ValueError(
                        f"explicit sensory frame exceeds the {lane.value} lane limit"
                    )
                selected.extend(lane_items)
            return tuple(selected)
        selected = []
        for lane in SENSORY_SETTLING_ORDER:
            lane_items = [item for item in pending if item.lane == lane]
            budget = self.maximum_bucket_tokens
            used = 0
            for item in lane_items:
                if (
                    maximum_per_lane is not None
                    and len([value for value in selected if value.lane == lane])
                    >= max(1, int(maximum_per_lane))
                ):
                    break
                if budget is not None and used and used + item.token_cost > budget:
                    continue
                selected.append(item)
                used += item.token_cost
                if budget is not None and used >= budget:
                    break
        return tuple(selected)

    def _validate_frame(self, buckets: Sequence[SensoryBucket]) -> None:
        for bucket in buckets:
            digest = hashlib.sha256(bucket.content.encode("utf-8")).hexdigest()
            if digest != bucket.content_sha256:
                raise ValueError(f"sensory content hash mismatch: {bucket.item_id}")
            if (
                bucket.embedding is not None
                and len(bucket.embedding) != self.mind.embedder.dimension
            ):
                raise ValueError(f"sensory embedding dimension mismatch: {bucket.item_id}")
            for node_id in bucket.concept_scores:
                if self.mind.store.get_concept(node_id) is None:
                    raise KeyError(f"unknown sensory concept: {node_id}")
            if bucket.kind == "cycle_return":
                cycle = (
                    self.mind.experience_cycle(bucket.cycle_id)
                    if bucket.cycle_id is not None
                    else None
                )
                expected_record_id = f"self-pulse-record:{bucket.item_id}"
                replayable_closed_return = (
                    cycle is not None
                    and cycle.status == "closed"
                    and cycle.terminal_return_record_id == expected_record_id
                )
                if cycle is None or (
                    cycle.status != "open" and not replayable_closed_return
                ):
                    raise ValueError(
                        "sensory return does not reference an admissible cycle: "
                        f"{bucket.cycle_id}"
                    )

    @staticmethod
    def _event_kind(lane: InputTrunk) -> EventKind:
        return {
            InputTrunk.HEAR: EventKind.MESSAGE,
            InputTrunk.SEE: EventKind.OBSERVATION,
            InputTrunk.NOTICE: EventKind.NOTIFICATION,
        }[lane]

    @staticmethod
    def _record_type(lane: InputTrunk) -> RecordType:
        return {
            InputTrunk.HEAR: RecordType.INBOUND_MESSAGE,
            InputTrunk.SEE: RecordType.OBSERVATION,
            InputTrunk.NOTICE: RecordType.NOTIFICATION,
        }[lane]

    @staticmethod
    def _merge_signal(target: dict[str, float], node_id: str, value: float) -> None:
        bounded = _bounded(value, -1.0, 1.0)
        current = target.get(node_id, 0.0)
        if bounded >= 0.0:
            positive = max(0.0, current)
            target[node_id] = 1.0 - (1.0 - positive) * (1.0 - bounded)
        elif current <= 0.0:
            target[node_id] = min(current, bounded)
        else:
            target[node_id] = current * (1.0 + bounded)

    def _admit_record(
        self,
        bucket: SensoryBucket,
        *,
        pulse: int,
    ) -> tuple[MemoryRecord, bool, _DeferredOutcome | None]:
        record_id = f"self-pulse-record:{bucket.item_id}"
        existing = self.mind.store.get_record(record_id)
        if existing is not None:
            if bucket.kind != "cycle_return":
                return existing, False, None
            cycle = self.mind.experience_cycle(str(bucket.cycle_id))
            if cycle is None:
                raise KeyError(f"unknown experience cycle: {bucket.cycle_id}")
            return (
                existing,
                False,
                _DeferredOutcome(
                    cycle.credited_edge_ids,
                    float(bucket.metadata["_self_pulse_stability_delta"]),
                    bool(bucket.metadata["_self_pulse_verified"]),
                ),
            )
        if bucket.kind == "cycle_return":
            if bucket.cycle_id is None:
                raise ValueError("cycle return has no cycle ID")
            result = self.mind.record_cycle_return(
                bucket.cycle_id,
                bucket.content,
                input_trunk=bucket.lane,
                status=str(bucket.metadata["_self_pulse_return_status"]),
                stability_delta=float(
                    bucket.metadata["_self_pulse_stability_delta"]
                ),
                verified=bool(bucket.metadata["_self_pulse_verified"]),
                terminal=bool(bucket.metadata["_self_pulse_terminal"]),
                source_id=bucket.source_id,
                record_type=RecordType(
                    bucket.metadata.get(
                        "_self_pulse_record_type", RecordType.RECEIPT.value
                    )
                ),
                record_id=record_id,
                event_id=f"event:{record_id}",
                return_concept_id=bucket.metadata.get(
                    "_self_pulse_return_concept_id"
                ),
                return_path_node_ids=tuple(
                    bucket.metadata.get("_self_pulse_return_path_node_ids", ())
                ),
                evidence_quality=float(
                    bucket.metadata["_self_pulse_evidence_quality"]
                ),
                provenance=dict(
                    bucket.metadata.get("_self_pulse_provenance", {})
                ),
                metadata={
                    key: value
                    for key, value in bucket.metadata.items()
                    if not str(key).startswith("_self_pulse_")
                },
                embedding=bucket.embedding,
                allow_growth=False,
                pulse_number=pulse,
                defer_recurrent_outcome=True,
            )
            return (
                result.record,
                True,
                _DeferredOutcome(
                    result.outcome.credited_edge_ids,
                    result.outcome.stability_delta,
                    result.outcome.verified,
                ),
            )
        record = self.mind.remember(
            bucket.content,
            kind=self._event_kind(bucket.lane),
            source_id=bucket.source_id,
            record_id=record_id,
            event_id=f"event:{record_id}",
            record_type=self._record_type(bucket.lane),
            concept_ids=tuple(bucket.concept_scores),
            metadata={
                **dict(bucket.metadata),
                "self_pulse_item_id": bucket.item_id,
                "content_sha256": bucket.content_sha256,
                "experience_id": f"self-pulse-experience:{bucket.item_id}",
            },
            allow_growth=bucket.allow_growth,
            input_trunk=bucket.lane,
            embedding=bucket.embedding,
            pulse_number=pulse,
        )
        return record, True, None

    def _collapse_record(
        self,
        record: MemoryRecord,
        bucket: SensoryBucket,
        *,
        pulse: int,
        pulse_id: str,
    ) -> tuple[tuple[TraversalTrace, ...], dict[str, float], tuple[str, ...]]:
        targets = dict(bucket.concept_scores)
        if not targets:
            experience_id = str(record.metadata.get("experience_id") or record.record_id)
            projections = self.mind.experience_projections(experience_id)
            for projection in projections:
                if projection.side != GraphSide.INPUT:
                    continue
                if projection.node_id in {SELF_ID, INPUT_NODE_IDS[bucket.lane]}:
                    continue
                targets[projection.node_id] = max(
                    targets.get(projection.node_id, 0.0), projection.activation
                )
        if not targets:
            targets[INPUT_NODE_IDS[bucket.lane]] = 1.0

        traces = []
        excitation: dict[str, float] = {}
        endpoints = []
        for node_id, score in sorted(targets.items()):
            collapse = self.mind.graph.collapse_to_self(
                pulse_id=f"{pulse_id}:{bucket.lane.value}:{bucket.item_id}",
                source_id=node_id,
                endpoint_score=score,
                required_input_trunk=bucket.lane,
                mark_active=True,
                pulse=pulse,
            )
            if collapse is None:
                continue
            traces.append(collapse)
            endpoints.append(node_id)
            denominator = max(1, len(collapse.path_node_ids) - 1)
            for depth, path_node_id in enumerate(collapse.path_node_ids):
                # SELF is the reconciliation boundary, not a broadcast neuron.
                # Exciting it would immediately diffuse every observation into
                # all six trunks and erase the specificity of the inward route.
                if path_node_id == SELF_ID:
                    continue
                value = _bounded(score) * (1.0 - 0.90 * depth / denominator)
                self._merge_signal(excitation, path_node_id, value)
        return tuple(traces), excitation, tuple(endpoints)

    def default_output_candidates(
        self, recurrent: RecurrentSnapshot
    ) -> tuple[OutputCandidate, ...]:
        # Imported lazily to keep self_pulse independent of the language surface
        # while still letting the standalone kernel use the native output graph.
        from .open_weight import resolve_output_focus

        focus = resolve_output_focus(
            self.mind,
            recurrent,
            pulse_id=f"self-output-probe:{recurrent.pulse}",
            mark_active=False,
        )
        result = []
        for trajectory in focus.candidates:
            edges = [
                self.mind.store.get_edge(edge_id)
                for edge_id in trajectory.path_edge_ids
            ]
            result.append(
                OutputCandidate(
                    node_id=trajectory.terminal_node_id,
                    trunk=trajectory.trunk,
                    path_node_ids=trajectory.path_node_ids,
                    path_edge_ids=trajectory.path_edge_ids,
                    route_probability=trajectory.effective_probability,
                    transient_pull=trajectory.transient_pull,
                    travel_time=sum(
                        edge.delta_y + edge.conflict_penalty
                        for edge in edges
                        if edge is not None
                    ),
                )
            )
        return tuple(
            result if self.maximum_output_candidates is None
            else result[: self.maximum_output_candidates]
        )

    def _verified_output_history(
        self,
    ) -> tuple[tuple[set[str], float, set[str]], ...]:
        rows = self.connection.execute(
            """SELECT c.credited_edge_ids_json, c.metadata_json,
                      r.stability_delta
               FROM experience_cycles c
               JOIN experience_cycle_returns r ON r.cycle_id = c.cycle_id
               WHERE r.verified = 1"""
        ).fetchall()
        result = []
        for row in rows:
            metadata = json.loads(row["metadata_json"] or "{}")
            context = {
                str(value) for value in metadata.get("semantic_node_ids", ())
            }
            if metadata.get("source_node_id"):
                context.add(str(metadata["source_node_id"]))
            result.append(
                (
                    {
                        str(value)
                        for value in json.loads(
                            row["credited_edge_ids_json"] or "[]"
                        )
                    },
                    _bounded(float(row["stability_delta"]), -1.0, 1.0),
                    context,
                )
            )
        return tuple(result)

    def _rank_outputs(
        self,
        candidates: Sequence[OutputCandidate],
        recurrent: RecurrentSnapshot,
        *,
        context_node_ids: Sequence[str],
    ) -> tuple[OutputAffordance, ...]:
        history = self._verified_output_history()
        current_context = {
            str(value)
            for value in context_node_ids
            if str(value) not in _STRUCTURAL_NODE_IDS
        }
        states = {state.node_id: state for state in recurrent.node_states}
        ranked = []
        for candidate in candidates:
            path_edges = set(candidate.path_edge_ids)
            terminal_edge = (
                candidate.path_edge_ids[-1]
                if candidate.path_edge_ids
                else None
            )
            observations = []
            for observed_edges, stability_delta, observed_context in history:
                if terminal_edge is None or terminal_edge not in observed_edges:
                    continue
                union = path_edges | observed_edges
                path_overlap = len(path_edges & observed_edges) / max(1, len(union))
                if current_context and observed_context:
                    context_union = current_context | observed_context
                    context_overlap = len(current_context & observed_context) / max(
                        1, len(context_union)
                    )
                elif not current_context and not observed_context:
                    context_overlap = 1.0
                else:
                    context_overlap = 0.20
                # Disjoint lived contexts should provide only a faint prior;
                # otherwise one broad trunk success dominates unrelated states.
                weight = path_overlap * (0.05 + 0.95 * context_overlap)
                observations.append((stability_delta, weight))

            terminal_state = states.get(candidate.node_id)
            prior_weight = 0.25 if terminal_state is not None else 0.0
            prior_value = terminal_state.valence if terminal_state is not None else 0.0
            observed_weight = sum(weight for _, weight in observations)
            evidence_weight = prior_weight + observed_weight
            expected = (
                prior_value * prior_weight
                + sum(value * weight for value, weight in observations)
            ) / max(1e-9, evidence_weight)
            if evidence_weight <= 0.0:
                expected = 0.0
            uncertainty = 1.0 / (1.0 + evidence_weight)
            outcome_confidence = 1.0 - uncertainty
            effort = candidate.travel_time / (1.0 + candidate.travel_time)
            endogenous_pull = math.sqrt(max(0.0, candidate.transient_pull))
            learned_support = _bounded(candidate.learned_support)
            learned_confidence = _bounded(candidate.learned_confidence)
            learned_expected = _bounded(
                candidate.learned_expected_stability_delta, -1.0, 1.0
            )
            chance_support = 1.0 / len(ACTION_OPPORTUNITY_ORDER)
            learned_value = learned_confidence * (
                (learned_support - chance_support)
                + 0.25 * learned_support * learned_expected
            )
            score = (
                0.55 * max(0.0, candidate.route_probability)
                + 0.35 * endogenous_pull
                + self.outcome_weight * expected * outcome_confidence
                + self.learned_output_weight * learned_value
                - self.output_effort_weight * effort
                - self.output_uncertainty_weight * uncertainty
            )
            ranked.append(
                OutputAffordance(
                    node_id=candidate.node_id,
                    trunk=candidate.trunk,
                    score=score,
                    relative_probability=0.0,
                    route_probability=candidate.route_probability,
                    transient_pull=candidate.transient_pull,
                    travel_time=candidate.travel_time,
                    path_node_ids=candidate.path_node_ids,
                    path_edge_ids=candidate.path_edge_ids,
                    expected_stability_delta=expected,
                    uncertainty=uncertainty,
                    learned_support=learned_support,
                    learned_confidence=learned_confidence,
                    learned_expected_stability_delta=learned_expected,
                    proposal_id=candidate.proposal_id,
                )
            )
        ranked.sort(key=lambda item: (-item.score, item.trunk.value, item.node_id))
        probabilities = _softmax(
            [item.score for item in ranked], self.entropy_temperature
        )
        return tuple(
            replace(item, relative_probability=probability)
            for item, probability in zip(ranked, probabilities)
        )

    def output_affordances(
        self,
        *,
        recurrent: RecurrentSnapshot | None = None,
        context_node_ids: Sequence[str] = (),
        maximum: int | None = None,
    ) -> tuple[OutputAffordance, ...]:
        state = recurrent or self.mind.recurrent.snapshot(pulse=self.mind.pulse)
        ranked = self._rank_outputs(
            self._default_output_candidates(state),
            state,
            context_node_ids=context_node_ids,
        )
        resolved_maximum = maximum if maximum is not None else self.maximum_output_candidates
        return ranked if resolved_maximum is None else ranked[: max(1, int(resolved_maximum))]

    @staticmethod
    def _state_metrics(recurrent: RecurrentSnapshot) -> tuple[float, float]:
        active = [state for state in recurrent.node_states if state.activation > 1e-9]
        if not active:
            return 0.0, 0.0
        total = sum(state.activation for state in active) or 1.0
        masses = [state.activation / total for state in active]
        stability = sum(
            mass * state.valence for mass, state in zip(masses, active)
        )
        entropy = -sum(mass * math.log(max(mass, 1e-12)) for mass in masses)
        normalized_entropy = (
            entropy / math.log(len(masses)) if len(masses) > 1 else 0.0
        )
        ambiguity = sum(
            mass * (1.0 - abs(state.valence))
            for mass, state in zip(masses, active)
        )
        free_energy = ambiguity - 0.35 * normalized_entropy - stability
        return stability, free_energy

    @staticmethod
    def _effect(
        value: AdmissionEffect | Mapping[str, float] | None,
    ) -> AdmissionEffect:
        if value is None:
            return AdmissionEffect({})
        if isinstance(value, AdmissionEffect):
            return value
        return AdmissionEffect(value)

    def advance_cycle(
        self,
        *,
        maximum_per_sensory_lane: int | None = None,
        cycle_id: str | None = None,
        advance_when_idle: bool = False,
        admission_observer: AdmissionObserver | None = None,
        item_ids: Sequence[str] | None = None,
        output_candidate_provider: OutputCandidateProvider | None = None,
        pulse_state_observer: PulseStateObserver | None = None,
    ) -> CognitiveCycleReceipt | None:
        """Commit one complete inward / SELF / outward transition.

        Canonical sensory records may be durable before reconciliation, but no
        recurrent state or self-pulse receipt is exposed unless the complete
        frame and every output authorization commit together.
        """
        if self._closed:
            raise RuntimeError("self pulse kernel is closed")
        started = self.mind.pulse
        buckets = self._select_frame(maximum_per_sensory_lane, item_ids=item_ids)
        if not buckets and not advance_when_idle:
            return None
        self._validate_frame(buckets)
        resolved_cycle_id = cycle_id or f"self-pulse:{uuid.uuid4().hex}"
        pulse, base_pulse_id = self.mind.begin_pulse()
        pulse_id = f"self-pulse:{pulse}"
        if buckets:
            with self.connection:
                self.connection.executemany(
                    """UPDATE self_pulse_inbox
                       SET status = 'processing', attempts = attempts + 1, error = NULL
                       WHERE item_id = ? AND status = 'pending'""",
                    ((bucket.item_id,) for bucket in buckets),
                )

        excitation: dict[str, float] = {}
        pressure_delta: dict[str, float] = {}
        context_node_ids: list[str] = []
        input_receipts = []
        input_records: list[MemoryRecord] = []
        all_trace_ids = []
        deferred_outcomes: list[_DeferredOutcome] = []
        try:
            for lane in SENSORY_SETTLING_ORDER:
                for bucket in (item for item in buckets if item.lane == lane):
                    record, created, deferred_outcome = self._admit_record(
                        bucket, pulse=pulse
                    )
                    input_records.append(record)
                    if deferred_outcome is not None:
                        deferred_outcomes.append(deferred_outcome)
                    traces, trace_excitation, endpoints = self._collapse_record(
                        record,
                        bucket,
                        pulse=pulse,
                        pulse_id=pulse_id,
                    )
                    for node_id, value in trace_excitation.items():
                        self._merge_signal(excitation, node_id, value)
                    context_node_ids.extend(endpoints)
                    if admission_observer is not None:
                        # Developmental observers can project several sparse
                        # fibers, intersections, and lexical bindings for one
                        # admitted sensation.  Keep that growth atomic and
                        # avoid committing every tiny graph mutation alone.
                        with self.mind.store.transaction():
                            effect = self._effect(
                                admission_observer(record, bucket, created)
                            )
                        for node_id, value in effect.excitation.items():
                            self._merge_signal(excitation, str(node_id), float(value))
                        for node_id, value in effect.pressure_delta.items():
                            pressure_delta[str(node_id)] = pressure_delta.get(
                                str(node_id), 0.0
                            ) + float(value)
                        context_node_ids.extend(effect.context_node_ids)
                    trace_ids = tuple(trace.trace_id for trace in traces)
                    all_trace_ids.extend(trace_ids)
                    input_receipts.append(
                        SelfInputReceipt(
                            pulse=pulse,
                            item_id=bucket.item_id,
                            kind=bucket.kind,
                            lane=bucket.lane,
                            record_id=record.record_id,
                            token_cost=bucket.token_cost,
                            concept_ids=tuple(bucket.concept_scores),
                            inward_traces=traces,
                            inward_trace_ids=trace_ids,
                            pending_after=max(0, len(self.pending()) - len(buckets)),
                        )
                    )

            with self.mind.store.transaction() as connection:
                for outcome in deferred_outcomes:
                    self.mind.recurrent.observe_outcome(
                        outcome.edge_ids,
                        stability_delta=outcome.stability_delta,
                        verified=outcome.verified,
                        pulse=pulse,
                    )
                recurrent = self.mind.recurrent.advance(
                    pulse=pulse,
                    pulse_id=pulse_id,
                    excitation=excitation,
                    pressure_delta=pressure_delta,
                )
                active_context = [
                    state.node_id
                    for state in recurrent.node_states
                    if state.activation >= 0.20
                ]
                context_node_ids.extend(active_context)
                proposed_outputs = (
                    tuple(output_candidate_provider(recurrent, pulse_id))
                    if output_candidate_provider is not None
                    else self.default_output_candidates(recurrent)
                )
                visible_outputs = (
                    proposed_outputs
                    if self.maximum_output_candidates is None
                    else proposed_outputs[: self.maximum_output_candidates]
                )
                candidates = self._rank_outputs(
                    visible_outputs,
                    recurrent,
                    context_node_ids=tuple(dict.fromkeys(context_node_ids)),
                )
                best_by_trunk: dict[OutputTrunk, OutputAffordance] = {}
                for candidate in candidates:
                    if candidate.score <= self.minimum_output_score:
                        continue
                    best_by_trunk.setdefault(candidate.trunk, candidate)
                selected_without_ids = tuple(
                    best_by_trunk[trunk]
                    for trunk in ACTION_OPPORTUNITY_ORDER
                    if trunk in best_by_trunk
                )
                selected = []
                for candidate in selected_without_ids:
                    material = (
                        f"{pulse_id}|{candidate.trunk.value}|{candidate.node_id}|"
                        f"{'/'.join(candidate.path_edge_ids)}"
                    )
                    authorization_id = "self-output:" + hashlib.sha256(
                        material.encode("utf-8")
                    ).hexdigest()[:32]
                    selected.append(
                        replace(candidate, authorization_id=authorization_id)
                    )
                selected_outputs = tuple(selected)
                selected_by_key = {
                    (item.trunk, item.node_id): item for item in selected_outputs
                }
                candidates = tuple(
                    selected_by_key.get((item.trunk, item.node_id), item)
                    for item in candidates
                )
                stability, free_energy = self._state_metrics(recurrent)
                settled_order = tuple(
                    lane
                    for lane in SENSORY_SETTLING_ORDER
                    if any(item.lane == lane for item in buckets)
                )
                active_node_ids = tuple(
                    state.node_id
                    for state in sorted(
                        recurrent.node_states,
                        key=lambda item: (-item.activation, item.node_id),
                    )
                    if state.activation > 1e-5
                )
                extension_payload: Mapping[str, Any] = {}
                if pulse_state_observer is not None:
                    extension_payload = dict(
                        pulse_state_observer(
                            connection,
                            PulseCommitFrame(
                                pulse=pulse,
                                pulse_id=pulse_id,
                                cycle_id=resolved_cycle_id,
                                input_records=tuple(input_records),
                                recurrent=recurrent,
                                context_node_ids=tuple(
                                    dict.fromkeys(context_node_ids)
                                ),
                                selected_outputs=selected_outputs,
                                perceived_stability=stability,
                                free_energy=free_energy,
                            ),
                        )
                        or {}
                    )
                payload = {
                    "schema": "habitus.self-pulse.v1",
                    "pulse": pulse,
                    "pulse_id": pulse_id,
                    "reserved_pulse_id": base_pulse_id,
                    "cycle_id": resolved_cycle_id,
                    "recurrent_state_sha256": recurrent.state_sha256,
                    "recurrent_nodes": [
                        {
                            "node_id": state.node_id,
                            "activation": state.activation,
                            "pressure": state.pressure,
                            "valence": state.valence,
                            "momentum": state.momentum,
                            "baseline_growth": state.baseline_growth,
                            "persistence": state.persistence,
                            "satisfaction_gain": state.satisfaction_gain,
                            "frustration_gain": state.frustration_gain,
                            "expression_threshold": state.expression_threshold,
                            "last_pulse": state.last_pulse,
                            "last_selected_pulse": state.last_selected_pulse,
                        }
                        for state in recurrent.node_states
                    ],
                    "inputs": [
                        {
                            "item_id": item.item_id,
                            "kind": item.kind,
                            "lane": item.lane.value,
                            "record_id": item.record_id,
                            "content_sha256": next(
                                bucket.content_sha256
                                for bucket in buckets
                                if bucket.item_id == item.item_id
                            ),
                            "concept_ids": list(item.concept_ids),
                            "inward_trace_ids": list(item.inward_trace_ids),
                        }
                        for item in input_receipts
                    ],
                    "settled_order": [lane.value for lane in settled_order],
                    "inward_trace_ids": list(dict.fromkeys(all_trace_ids)),
                    "perceived_stability": stability,
                    "free_energy": free_energy,
                    "output_candidates": [
                        {
                            "node_id": item.node_id,
                            "trunk": item.trunk.value,
                            "score": item.score,
                            "relative_probability": item.relative_probability,
                            "route_probability": item.route_probability,
                            "transient_pull": item.transient_pull,
                            "travel_time": item.travel_time,
                            "expected_stability_delta": item.expected_stability_delta,
                            "uncertainty": item.uncertainty,
                            "learned_support": item.learned_support,
                            "learned_confidence": item.learned_confidence,
                            "learned_expected_stability_delta": (
                                item.learned_expected_stability_delta
                            ),
                            "proposal_id": item.proposal_id,
                            "path_node_ids": list(item.path_node_ids),
                            "path_edge_ids": list(item.path_edge_ids),
                            "authorization_id": item.authorization_id,
                        }
                        for item in candidates
                    ],
                    "selected_output_authorization_ids": [
                        item.authorization_id for item in selected_outputs
                    ],
                }
                if extension_payload:
                    payload["extensions"] = extension_payload
                encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
                state_sha256 = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
                connection.execute(
                    """INSERT INTO self_pulse_states(
                           pulse, pulse_id, cycle_id, recurrent_state_sha256,
                           state_sha256, state_json
                       ) VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        pulse,
                        pulse_id,
                        resolved_cycle_id,
                        recurrent.state_sha256,
                        state_sha256,
                        encoded,
                    ),
                )
                connection.executemany(
                    """INSERT INTO self_output_authorizations(
                           authorization_id, pulse, pulse_id, trunk, node_id,
                           path_node_ids_json, path_edge_ids_json
                       ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        (
                            item.authorization_id,
                            pulse,
                            pulse_id,
                            item.trunk.value,
                            item.node_id,
                            json.dumps(item.path_node_ids, separators=(",", ":")),
                            json.dumps(item.path_edge_ids, separators=(",", ":")),
                        )
                        for item in selected_outputs
                    ),
                )
                connection.executemany(
                    """UPDATE self_pulse_inbox
                       SET status = 'processed', record_id = ?, processed_pulse = ?
                       WHERE item_id = ? AND status = 'processing'""",
                    (
                        (item.record_id, pulse, item.item_id)
                        for item in input_receipts
                    ),
                )

            state = SelfState(
                pulse=pulse,
                pulse_id=pulse_id,
                cycle_id=resolved_cycle_id,
                recurrent_state_sha256=recurrent.state_sha256,
                state_sha256=state_sha256,
                perceived_stability=stability,
                free_energy=free_energy,
                input_item_ids=tuple(item.item_id for item in input_receipts),
                input_record_ids=tuple(item.record_id for item in input_receipts),
                inward_trace_ids=tuple(dict.fromkeys(all_trace_ids)),
                settled_order=settled_order,
                active_node_ids=active_node_ids,
            )
            sensory = SettlingCycleReceipt(
                cycle_id=resolved_cycle_id,
                started_pulse=started,
                completed_pulse=pulse,
                receipts=tuple(input_receipts),
                settled_order=settled_order,
                self_state=state,
            )
            selected_by_trunk = {item.trunk: item for item in selected_outputs}
            opportunities = tuple(
                OutputOpportunity(trunk, selected_by_trunk.get(trunk))
                for trunk in ACTION_OPPORTUNITY_ORDER
            )
            selected_output = (
                max(selected_outputs, key=lambda item: (item.score, item.node_id))
                if selected_outputs
                else None
            )
            return CognitiveCycleReceipt(
                cycle_id=resolved_cycle_id,
                sensory=sensory,
                recurrent=recurrent,
                output_candidates=candidates,
                output_opportunities=opportunities,
                self_state=state,
                selected_output=selected_output,
                selected_outputs=selected_outputs,
            )
        except Exception as error:
            if buckets:
                with self.connection:
                    self.connection.executemany(
                        """UPDATE self_pulse_inbox
                           SET status = 'failed', error = ?
                           WHERE item_id = ? AND status = 'processing'""",
                        (
                            (f"{type(error).__name__}: {error}", bucket.item_id)
                            for bucket in buckets
                        ),
                    )
            raise

    def latest_state_payload(self) -> Mapping[str, Any] | None:
        row = self.connection.execute(
            "SELECT state_json FROM self_pulse_states ORDER BY pulse DESC LIMIT 1"
        ).fetchone()
        return json.loads(row["state_json"]) if row is not None else None

    def advance_internal_output(
        self,
        *,
        excitation: Mapping[str, float],
        pressure_delta: Mapping[str, float] | None = None,
        selected_node_ids: Sequence[str] = (),
        kind: str = "thought",
        metadata: Mapping[str, Any] | None = None,
    ) -> tuple[RecurrentSnapshot, str]:
        """Advance a private/output substep inside the latest SELF pulse.

        A generated word or private thought is not a new sensory frame. It
        changes the live recurrent field under the current pulse and receives
        its own immutable event receipt, without invalidating the motor
        authorization that permitted the enclosing output.
        """
        parent = self.connection.execute(
            "SELECT pulse FROM self_pulse_states ORDER BY pulse DESC LIMIT 1"
        ).fetchone()
        if parent is None or int(parent["pulse"]) != self.mind.pulse:
            raise ValueError("internal output requires the current self pulse")
        row = self.connection.execute(
            """SELECT COALESCE(MAX(sequence), 0) + 1 AS sequence
               FROM self_internal_output_events WHERE parent_pulse = ?""",
            (self.mind.pulse,),
        ).fetchone()
        sequence = int(row["sequence"])
        event_id = f"self-internal:{self.mind.pulse}:{sequence}:{uuid.uuid4().hex[:12]}"
        before = self.mind.recurrent.snapshot(pulse=self.mind.pulse)
        with self.mind.store.transaction() as connection:
            recurrent = self.mind.recurrent.advance(
                pulse=self.mind.pulse,
                pulse_id=event_id,
                excitation=excitation,
                pressure_delta=pressure_delta,
                selected_node_ids=selected_node_ids,
            )
            connection.execute(
                """INSERT INTO self_internal_output_events(
                       event_id, parent_pulse, sequence, kind,
                       excitation_json, pressure_delta_json,
                       state_before_sha256, state_after_sha256, metadata_json
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    event_id,
                    self.mind.pulse,
                    sequence,
                    str(kind),
                    json.dumps(
                        {str(key): float(value) for key, value in excitation.items()},
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    json.dumps(
                        {
                            str(key): float(value)
                            for key, value in (pressure_delta or {}).items()
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    before.state_sha256,
                    recurrent.state_sha256,
                    json.dumps(
                        dict(metadata or {}), sort_keys=True, separators=(",", ":")
                    ),
                ),
            )
        return recurrent, event_id

    def claim_output(
        self,
        authorization_id: str,
        *,
        trunk: OutputTrunk,
        node_id: str,
        path_node_ids: Sequence[str],
        path_edge_ids: Sequence[str],
    ) -> None:
        row = self.connection.execute(
            "SELECT * FROM self_output_authorizations WHERE authorization_id = ?",
            (str(authorization_id),),
        ).fetchone()
        if row is None:
            raise ValueError("output was not selected by a self pulse")
        latest = self.connection.execute(
            "SELECT MAX(pulse) AS pulse FROM self_pulse_states"
        ).fetchone()
        if int(row["pulse"]) != int(latest["pulse"]):
            raise ValueError("output authorization belongs to a superseded self pulse")
        if row["status"] != "selected":
            raise ValueError(f"output authorization is already {row['status']}")
        if OutputTrunk(row["trunk"]) != trunk or str(row["node_id"]) != str(node_id):
            raise ValueError("output does not match its selected motor route")
        if tuple(json.loads(row["path_node_ids_json"])) != tuple(path_node_ids):
            raise ValueError("output nodes do not match their authorization")
        if tuple(json.loads(row["path_edge_ids_json"])) != tuple(path_edge_ids):
            raise ValueError("output path does not match its authorization")
        with self.connection:
            changed = self.connection.execute(
                """UPDATE self_output_authorizations
                   SET status = 'claimed', claimed_at = ?
                   WHERE authorization_id = ? AND status = 'selected'""",
                (time.time(), str(authorization_id)),
            ).rowcount
        if changed != 1:
            raise ValueError("output authorization could not be claimed")

    def release_output_claim(self, authorization_id: str) -> None:
        with self.connection:
            self.connection.execute(
                """UPDATE self_output_authorizations
                   SET status = 'selected', claimed_at = NULL
                   WHERE authorization_id = ? AND status = 'claimed'""",
                (str(authorization_id),),
            )

    def complete_output(self, authorization_id: str, cycle_id: str) -> None:
        cycle = self.mind.experience_cycle(cycle_id)
        if cycle is None:
            raise KeyError(f"unknown experience cycle: {cycle_id}")
        if str(cycle.metadata.get("self_pulse_authorization_id")) != str(
            authorization_id
        ):
            raise ValueError("experience cycle does not carry this authorization")
        with self.connection:
            changed = self.connection.execute(
                """UPDATE self_output_authorizations
                   SET status = 'consumed', cycle_id = ?, consumed_at = ?
                   WHERE authorization_id = ? AND status = 'claimed'""",
                (str(cycle_id), time.time(), str(authorization_id)),
            ).rowcount
        if changed != 1:
            raise ValueError("output authorization was not actively claimed")

    def actualize_output(
        self,
        affordance: OutputAffordance,
        content: str,
        *,
        trace: TraversalTrace | None = None,
        source_id: str = "self",
        record_type: RecordType | str | None = None,
        metadata: Mapping[str, Any] | None = None,
        credited_edge_ids: Sequence[str] | None = None,
        embedding: Sequence[float] | None = None,
    ) -> ExperienceCycle:
        """Persist one authorized motor act and consume its capability once."""
        authorization_id = affordance.authorization_id
        if authorization_id is None:
            raise ValueError("output affordance has no self-pulse authorization")
        resolved_content = str(content)
        if not resolved_content.strip():
            raise ValueError("externalized output cannot be empty")
        self.claim_output(
            authorization_id,
            trunk=affordance.trunk,
            node_id=affordance.node_id,
            path_node_ids=affordance.path_node_ids,
            path_edge_ids=affordance.path_edge_ids,
        )
        if trace is None:
            edges = [
                self.mind.store.get_edge(edge_id)
                for edge_id in affordance.path_edge_ids
            ]
            trace = TraversalTrace(
                trace_id=(
                    f"trace:{affordance.authorization_id}:"
                    f"output:{affordance.node_id}"
                ),
                side=GraphSide.OUTPUT,
                start_node_id=SELF_ID,
                target_node_id=affordance.path_node_ids[-1],
                path_node_ids=affordance.path_node_ids,
                path_edge_ids=affordance.path_edge_ids,
                total_travel_time=sum(
                    edge.delta_y + edge.conflict_penalty
                    for edge in edges
                    if edge is not None
                ),
                endpoint_score=affordance.relative_probability,
            )
        elif (
            trace.side != GraphSide.OUTPUT
            or trace.start_node_id != SELF_ID
            or trace.path_node_ids[: len(affordance.path_node_ids)]
            != affordance.path_node_ids
            or trace.path_edge_ids[: len(affordance.path_edge_ids)]
            != affordance.path_edge_ids
        ):
            self.release_output_claim(authorization_id)
            raise ValueError("actualized trace does not extend its authorized route")
        authorization = self.connection.execute(
            """SELECT pulse_id FROM self_output_authorizations
               WHERE authorization_id = ?""",
            (authorization_id,),
        ).fetchone()
        if authorization is None:
            self.release_output_claim(authorization_id)
            raise ValueError("output authorization disappeared while claimed")
        decision = OutputDecision(
            pulse_id=str(authorization["pulse_id"]),
            trunk=affordance.trunk,
            confidence=affordance.relative_probability,
            trace=trace,
        )
        resolved_record_type = (
            record_type
            if record_type is not None
            else RecordType.OUTBOUND_MESSAGE
            if affordance.trunk == OutputTrunk.SPEAK
            else RecordType.TOOL_CALL
        )
        try:
            cycle = self.mind.begin_output_cycle(
                resolved_content,
                decision,
                source_id=source_id,
                record_type=resolved_record_type,
                metadata={
                    **dict(metadata or {}),
                    "self_pulse_authorization_id": authorization_id,
                    "self_pulse_id": decision.pulse_id,
                    "authorized_output_node_id": affordance.node_id,
                },
                credited_edge_ids=(
                    credited_edge_ids
                    if credited_edge_ids is not None
                    else trace.path_edge_ids
                ),
                embedding=embedding,
                pulse_number=self.mind.pulse,
            )
        except Exception:
            self.release_output_claim(authorization_id)
            raise
        self.complete_output(authorization_id, cycle.cycle_id)
        self.mind.graph.activate_trace(
            f"self-output:{self.mind.pulse}", trace
        )
        self.mind.recurrent.mark_selected(
            affordance.path_node_ids, pulse=self.mind.pulse
        )
        return cycle

    async def run(
        self,
        *,
        maximum_pulses: int | None = None,
        poll_interval: float = 0.05,
        stop_when_idle: bool = False,
        admission_observer: AdmissionObserver | None = None,
        pulse_state_observer: PulseStateObserver | None = None,
    ) -> tuple[CognitiveCycleReceipt, ...]:
        receipts = []
        while not self._closed:
            if self.pending():
                receipt = self.advance_cycle(
                    admission_observer=admission_observer,
                    pulse_state_observer=pulse_state_observer,
                )
                if receipt is not None:
                    receipts.append(receipt)
                    if maximum_pulses is not None and len(receipts) >= maximum_pulses:
                        break
                continue
            if stop_when_idle:
                break
            await asyncio.sleep(max(0.001, float(poll_interval)))
        return tuple(receipts)

    def close(self) -> None:
        self._closed = True


__all__ = [
    "ACTION_OPPORTUNITY_ORDER",
    "AdmissionEffect",
    "CognitiveCycleReceipt",
    "OutputAffordance",
    "OutputCandidate",
    "OutputCandidateProvider",
    "OutputOpportunity",
    "OUTPUT_RETURN_LANES",
    "SENSORY_SETTLING_ORDER",
    "SelfInputReceipt",
    "SelfPulseKernel",
    "SelfState",
    "SensoryBucket",
    "SettlingCycleReceipt",
    "conservative_token_cost",
]
