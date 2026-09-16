"""Milestone-gated developmental language and byte-form growth.

This module deliberately does not contain a word list, tokenizer, story corpus,
or response template.  It admits canonical experience records by developmental
stage and lets recurring UTF-8 spans compete for graph membership only when
they are both compressive and more route-specific than a shuffled baseline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import enum
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from .graph import INPUT_NODE_IDS
from .types import ConceptNode, GraphSide, InputTrunk, MemoryRecord, as_tuple


BYTE_LEXEME_CANDIDATE_KIND = "byte_lexeme_candidate"
BYTE_LEXEME_KIND = "byte_lexeme"


class CurriculumStage(str, enum.Enum):
    PRELINGUISTIC = "prelinguistic"
    GROUNDED_FORMS = "grounded_forms"
    FUNCTIONAL_EXCHANGE = "functional_exchange"
    EXPERIENCED_NARRATIVE = "experienced_narrative"
    BROADER_NARRATIVE = "broader_narrative"


STAGE_ORDER = tuple(CurriculumStage)


class EpisodeKind(str, enum.Enum):
    SENSORY = "sensory"
    GROUNDED_LABEL = "grounded_label"
    FUNCTIONAL_EXCHANGE = "functional_exchange"
    EXPERIENCED_NARRATIVE = "experienced_narrative"
    EXTERNAL_NARRATIVE = "external_narrative"


@dataclass(frozen=True)
class DevelopmentalMetrics:
    nonverbal_discrimination: float = 0.0
    consequence_prediction: float = 0.0
    lexical_recall_at_5: float = 0.0
    lexical_shuffled_gap: float = 0.0
    grounded_action_accuracy: float = 0.0
    grounding_coverage: float = 0.0
    narrative_route_recall: float = 0.0


@dataclass(frozen=True)
class CurriculumAdmission:
    admission_id: str
    record_id: str
    episode_kind: EpisodeKind
    stage: CurriculumStage
    accepted: bool
    reason: str
    route_ids: tuple[str, ...]


@dataclass(frozen=True)
class FormGrowthReceipt:
    record_id: str
    observed_forms: int
    candidate_node_ids: tuple[str, ...]
    promoted_node_ids: tuple[str, ...]
    promoted_form_ids: tuple[str, ...]


@dataclass(frozen=True)
class FormStatistics:
    form_id: str
    kind: str
    independent_records: int
    byte_length: int
    compression_gain: float
    grounding_score: float
    shuffled_control: float
    grounding_gap: float
    node_id: str | None


@dataclass(frozen=True)
class MotorForm:
    """A whole experienced utterance promoted into an outward motor option."""

    form_id: str
    node_id: str
    payload: bytes
    payload_sha256: str
    route_ids: tuple[str, ...]
    route_alignment: float
    grounding_score: float
    grounding_gap: float
    independent_records: int
    habit_support: float = 0.0
    habit_confidence: float = 0.0
    habit_match_count: int = 0


@dataclass(frozen=True)
class RecognizedForm:
    """One promoted byte pattern detected in the current HEAR membrane.

    Recognition hashes current byte spans and consults only learned form state;
    it does not recover an old record or compare against a text transcript.
    """

    form_id: str
    node_id: str
    start_byte: int
    end_byte: int
    byte_length: int
    activation: float
    grounding_score: float
    grounding_gap: float


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _bounded_score(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _stage_index(stage: CurriculumStage) -> int:
    return STAGE_ORDER.index(stage)


def _form_id(payload: bytes) -> str:
    return "byte-form:" + hashlib.sha256(payload).hexdigest()[:32]


def _lexeme_node_id(form_id: str) -> str:
    return "byte-lexeme:" + form_id.rsplit(":", 1)[-1]


def _eligible_span(payload: bytes) -> bool:
    if len(payload) < 2:
        return False
    # Whitespace and control-only fragments carry timing but should not become
    # independent lexical candidates. No alphabet or word boundary is assumed.
    return any(value > 32 and value != 127 for value in payload)


class DevelopmentalCurriculum:
    """Persist milestone gates and reject language ahead of lived grounding."""

    def __init__(self, mind: Any):
        self.mind = mind
        self.connection = mind.store.connection
        self._create_schema()

    def _create_schema(self) -> None:
        with self.connection:
            self.connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS developmental_curriculum_state (
                    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                    stage TEXT NOT NULL,
                    metrics_json TEXT NOT NULL DEFAULT '{}',
                    transition_count INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS developmental_stage_transitions (
                    transition_id TEXT PRIMARY KEY,
                    from_stage TEXT NOT NULL,
                    to_stage TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TRIGGER IF NOT EXISTS developmental_transitions_immutable_update
                BEFORE UPDATE ON developmental_stage_transitions BEGIN
                    SELECT RAISE(ABORT, 'developmental transitions are immutable');
                END;

                CREATE TRIGGER IF NOT EXISTS developmental_transitions_immutable_delete
                BEFORE DELETE ON developmental_stage_transitions BEGIN
                    SELECT RAISE(ABORT, 'developmental transitions are immutable');
                END;

                CREATE TABLE IF NOT EXISTS developmental_admissions (
                    admission_id TEXT PRIMARY KEY,
                    record_id TEXT NOT NULL REFERENCES records(record_id),
                    episode_kind TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    accepted INTEGER NOT NULL,
                    reason TEXT NOT NULL,
                    route_ids_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TRIGGER IF NOT EXISTS developmental_admissions_immutable_update
                BEFORE UPDATE ON developmental_admissions BEGIN
                    SELECT RAISE(ABORT, 'developmental admissions are immutable');
                END;

                CREATE TRIGGER IF NOT EXISTS developmental_admissions_immutable_delete
                BEFORE DELETE ON developmental_admissions BEGIN
                    SELECT RAISE(ABORT, 'developmental admissions are immutable');
                END;
                """
            )
            self.connection.execute(
                """INSERT OR IGNORE INTO developmental_curriculum_state(
                       singleton, stage, metrics_json, transition_count
                   ) VALUES (1, ?, '{}', 0)""",
                (CurriculumStage.PRELINGUISTIC.value,),
            )

    @property
    def stage(self) -> CurriculumStage:
        row = self.connection.execute(
            "SELECT stage FROM developmental_curriculum_state WHERE singleton = 1"
        ).fetchone()
        return CurriculumStage(row["stage"])

    @staticmethod
    def _ready(stage: CurriculumStage, metrics: DevelopmentalMetrics) -> bool:
        if stage == CurriculumStage.PRELINGUISTIC:
            return (
                metrics.nonverbal_discrimination >= 0.80
                and metrics.consequence_prediction >= 0.80
            )
        if stage == CurriculumStage.GROUNDED_FORMS:
            return (
                metrics.lexical_recall_at_5 >= 0.75
                and metrics.lexical_shuffled_gap >= 0.40
            )
        if stage == CurriculumStage.FUNCTIONAL_EXCHANGE:
            return (
                metrics.grounded_action_accuracy >= 0.80
                and metrics.grounding_coverage >= 0.80
            )
        if stage == CurriculumStage.EXPERIENCED_NARRATIVE:
            return (
                metrics.grounding_coverage >= 0.85
                and metrics.narrative_route_recall >= 0.75
            )
        return False

    def update_metrics(self, metrics: DevelopmentalMetrics) -> CurriculumStage:
        current = self.stage
        next_stage = current
        if current != CurriculumStage.BROADER_NARRATIVE and self._ready(current, metrics):
            next_stage = STAGE_ORDER[_stage_index(current) + 1]
        encoded = _json(asdict(metrics))
        with self.mind.store.transaction() as connection:
            if next_stage != current:
                material = f"{current.value}|{next_stage.value}|{encoded}"
                transition_id = "developmental-stage:" + hashlib.sha256(
                    material.encode("utf-8")
                ).hexdigest()[:32]
                connection.execute(
                    """INSERT OR IGNORE INTO developmental_stage_transitions(
                           transition_id, from_stage, to_stage, metrics_json
                       ) VALUES (?, ?, ?, ?)""",
                    (transition_id, current.value, next_stage.value, encoded),
                )
                connection.execute(
                    """UPDATE developmental_curriculum_state
                       SET stage = ?, metrics_json = ?,
                           transition_count = transition_count + 1,
                           updated_at = CURRENT_TIMESTAMP
                       WHERE singleton = 1""",
                    (next_stage.value, encoded),
                )
            else:
                connection.execute(
                    """UPDATE developmental_curriculum_state
                       SET metrics_json = ?, updated_at = CURRENT_TIMESTAMP
                       WHERE singleton = 1""",
                    (encoded,),
                )
        return next_stage

    def _admission_reason(
        self,
        kind: EpisodeKind,
        route_ids: Sequence[str],
    ) -> tuple[bool, str]:
        stage = self.stage
        required = {
            EpisodeKind.SENSORY: CurriculumStage.PRELINGUISTIC,
            EpisodeKind.GROUNDED_LABEL: CurriculumStage.GROUNDED_FORMS,
            EpisodeKind.FUNCTIONAL_EXCHANGE: CurriculumStage.FUNCTIONAL_EXCHANGE,
            EpisodeKind.EXPERIENCED_NARRATIVE: CurriculumStage.EXPERIENCED_NARRATIVE,
            EpisodeKind.EXTERNAL_NARRATIVE: CurriculumStage.BROADER_NARRATIVE,
        }[kind]
        if _stage_index(stage) < _stage_index(required):
            return False, f"requires {required.value} milestone"
        if kind != EpisodeKind.SENSORY and not route_ids:
            return False, "language episode has no lived route"
        if kind in {EpisodeKind.EXPERIENCED_NARRATIVE, EpisodeKind.EXTERNAL_NARRATIVE}:
            missing = [route for route in route_ids if not self.mind.store.has_concept(route)]
            if missing:
                return False, "narrative references routes absent from this mind"
        return True, "admitted by current developmental milestone"

    def admit(
        self,
        record: MemoryRecord,
        *,
        episode_kind: EpisodeKind | str,
        route_ids: Sequence[str] = (),
    ) -> CurriculumAdmission:
        kind = episode_kind if isinstance(episode_kind, EpisodeKind) else EpisodeKind(episode_kind)
        routes = tuple(dict.fromkeys(str(item) for item in route_ids))
        accepted, reason = self._admission_reason(kind, routes)
        material = f"{record.record_id}|{kind.value}|{self.stage.value}|{'/'.join(routes)}"
        admission_id = "developmental-admission:" + hashlib.sha256(
            material.encode("utf-8")
        ).hexdigest()[:32]
        with self.mind.store.transaction() as connection:
            connection.execute(
                """INSERT OR IGNORE INTO developmental_admissions(
                       admission_id, record_id, episode_kind, stage, accepted,
                       reason, route_ids_json
                   ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    admission_id,
                    record.record_id,
                    kind.value,
                    self.stage.value,
                    int(accepted),
                    reason,
                    _json(routes),
                ),
            )
        return CurriculumAdmission(
            admission_id=admission_id,
            record_id=record.record_id,
            episode_kind=kind,
            stage=self.stage,
            accepted=accepted,
            reason=reason,
            route_ids=routes,
        )


class ByteFormLearner:
    """Grow opaque lexeme candidates from recurrent byte spans and lived routes."""

    def __init__(
        self,
        mind: Any,
        *,
        minimum_span_bytes: int = 2,
        maximum_span_bytes: int | None = None,
        maximum_candidates_per_record: int | None = None,
        candidate_records: int = 2,
        promotion_records: int = 3,
        minimum_grounding: float = 0.67,
        minimum_grounding_gap: float = 0.20,
        minimum_compression_gain: float = 0.0,
    ) -> None:
        if minimum_span_bytes < 2 or (
            maximum_span_bytes is not None
            and maximum_span_bytes < minimum_span_bytes
        ):
            raise ValueError("invalid byte span bounds")
        if candidate_records < 2 or promotion_records < candidate_records:
            raise ValueError("invalid byte-form evidence thresholds")
        self.mind = mind
        self.connection = mind.store.connection
        self.minimum_span_bytes = int(minimum_span_bytes)
        self.maximum_span_bytes = (
            None if maximum_span_bytes is None else int(maximum_span_bytes)
        )
        self.maximum_candidates_per_record = (
            None
            if maximum_candidates_per_record is None
            else max(1, int(maximum_candidates_per_record))
        )
        self.candidate_records = int(candidate_records)
        self.promotion_records = int(promotion_records)
        self.minimum_grounding = _bounded_score(minimum_grounding)
        self.minimum_grounding_gap = _bounded_score(minimum_grounding_gap)
        self.minimum_compression_gain = float(minimum_compression_gain)
        self._create_schema()

    def _create_schema(self) -> None:
        with self.connection:
            self.connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS developmental_byte_forms (
                    form_id TEXT PRIMARY KEY,
                    byte_length INTEGER NOT NULL,
                    kind TEXT NOT NULL DEFAULT 'observed',
                    node_id TEXT,
                    compression_gain REAL NOT NULL DEFAULT 0.0,
                    grounding_score REAL NOT NULL DEFAULT 0.0,
                    shuffled_control REAL NOT NULL DEFAULT 0.0,
                    first_pulse INTEGER NOT NULL,
                    last_pulse INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS developmental_form_occurrences (
                    form_id TEXT NOT NULL REFERENCES developmental_byte_forms(form_id),
                    record_id TEXT NOT NULL REFERENCES records(record_id),
                    start_byte INTEGER NOT NULL,
                    end_byte INTEGER NOT NULL,
                    pulse INTEGER NOT NULL,
                    route_ids_json TEXT NOT NULL,
                    motor_eligible INTEGER NOT NULL DEFAULT 1,
                    utterance_sha256 TEXT NOT NULL DEFAULT '',
                    whole_utterance INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY(form_id, record_id, start_byte, end_byte)
                );

                CREATE TABLE IF NOT EXISTS developmental_route_exposures (
                    record_id TEXT NOT NULL REFERENCES records(record_id),
                    route_id TEXT NOT NULL,
                    pulse INTEGER NOT NULL,
                    PRIMARY KEY(record_id, route_id)
                );

                CREATE INDEX IF NOT EXISTS idx_developmental_forms_kind
                    ON developmental_byte_forms(kind, last_pulse);
                CREATE INDEX IF NOT EXISTS idx_developmental_occurrences_form
                    ON developmental_form_occurrences(form_id, record_id);

                CREATE TABLE IF NOT EXISTS developmental_motor_forms (
                    form_id TEXT PRIMARY KEY
                        REFERENCES developmental_byte_forms(form_id),
                    node_id TEXT NOT NULL,
                    payload BLOB NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    evidence_record_id TEXT NOT NULL REFERENCES records(record_id),
                    promoted_pulse INTEGER NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TRIGGER IF NOT EXISTS developmental_motor_forms_immutable_update
                BEFORE UPDATE ON developmental_motor_forms BEGIN
                    SELECT RAISE(ABORT, 'developmental motor forms are immutable');
                END;

                CREATE TRIGGER IF NOT EXISTS developmental_motor_forms_immutable_delete
                BEFORE DELETE ON developmental_motor_forms BEGIN
                    SELECT RAISE(ABORT, 'developmental motor forms are immutable');
                END;

                CREATE TABLE IF NOT EXISTS developmental_motor_habit_observations (
                    form_id TEXT NOT NULL
                        REFERENCES developmental_motor_forms(form_id),
                    cycle_id TEXT NOT NULL REFERENCES experience_cycles(cycle_id),
                    return_record_id TEXT NOT NULL REFERENCES records(record_id),
                    route_id TEXT NOT NULL REFERENCES concepts(concept_id),
                    stability_delta REAL NOT NULL,
                    pulse INTEGER NOT NULL,
                    PRIMARY KEY(form_id, cycle_id, route_id)
                );

                CREATE INDEX IF NOT EXISTS idx_developmental_motor_habit_form
                    ON developmental_motor_habit_observations(form_id, pulse);

                CREATE TRIGGER IF NOT EXISTS developmental_motor_habits_immutable_update
                BEFORE UPDATE ON developmental_motor_habit_observations BEGIN
                    SELECT RAISE(ABORT, 'developmental motor habits are immutable');
                END;

                CREATE TRIGGER IF NOT EXISTS developmental_motor_habits_immutable_delete
                BEFORE DELETE ON developmental_motor_habit_observations BEGIN
                    SELECT RAISE(ABORT, 'developmental motor habits are immutable');
                END;
                """
            )
            occurrence_columns = {
                str(row["name"])
                for row in self.connection.execute(
                    "PRAGMA table_info(developmental_form_occurrences)"
                ).fetchall()
            }
            if "motor_eligible" not in occurrence_columns:
                self.connection.execute(
                    "ALTER TABLE developmental_form_occurrences "
                    "ADD COLUMN motor_eligible INTEGER NOT NULL DEFAULT 1"
                )
            if "utterance_sha256" not in occurrence_columns:
                self.connection.execute(
                    "ALTER TABLE developmental_form_occurrences "
                    "ADD COLUMN utterance_sha256 TEXT NOT NULL DEFAULT ''"
                )
            if "whole_utterance" not in occurrence_columns:
                self.connection.execute(
                    "ALTER TABLE developmental_form_occurrences "
                    "ADD COLUMN whole_utterance INTEGER NOT NULL DEFAULT 0"
                )
            habit_columns = {
                str(row["name"])
                for row in self.connection.execute(
                    "PRAGMA table_info(developmental_motor_habit_observations)"
                ).fetchall()
            }
            if "return_record_id" not in habit_columns:
                self.connection.execute(
                    "ALTER TABLE developmental_motor_habit_observations "
                    "ADD COLUMN return_record_id TEXT REFERENCES records(record_id)"
                )

    @staticmethod
    def _nonstructural_routes(route_ids: Sequence[str]) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                str(item)
                for item in route_ids
                if str(item) != "SELF"
                and not str(item).startswith(("IN:", "OUT:"))
            )
        )

    def _spans(self, payload: bytes) -> list[tuple[str, int, int, bytes]]:
        upper = (
            len(payload)
            if self.maximum_span_bytes is None
            else min(self.maximum_span_bytes, len(payload))
        )
        candidates = []
        for length in range(self.minimum_span_bytes, upper + 1):
            same_length = {}
            for start in range(0, len(payload) - length + 1):
                end = start + length
                span = payload[start:end]
                if not _eligible_span(span):
                    continue
                form_id = _form_id(span)
                same_length.setdefault(form_id, (form_id, start, end, span))
            candidates.extend(
                sorted(
                    same_length.values(),
                    key=lambda item: hashlib.sha256(item[3]).digest(),
                )
            )
        if (
            self.maximum_candidates_per_record is None
            or len(candidates) <= self.maximum_candidates_per_record
        ):
            return candidates
        # Text already carries pause-like whitespace boundaries. Treat those
        # boundaries as low-level timing receptors, not as a vocabulary: every
        # bounded surface remains only a candidate until recurrence across
        # different utterances and lived-route grounding select it.
        runs = []
        start = None
        for index, value in enumerate(payload):
            if value <= 32 or value == 127:
                if start is not None:
                    runs.append((start, index))
                    start = None
            elif start is None:
                start = index
        if start is not None:
            runs.append((start, len(payload)))
        mandatory: dict[tuple[str, int, int], tuple[str, int, int, bytes]] = {}
        for left in range(len(runs)):
            for width in range(1, min(3, len(runs) - left) + 1):
                span_start = runs[left][0]
                span_end = runs[left + width - 1][1]
                if span_end - span_start > upper:
                    continue
                span = payload[span_start:span_end]
                if not _eligible_span(span):
                    continue
                form_id = _form_id(span)
                mandatory[(form_id, span_start, span_end)] = (
                    form_id,
                    span_start,
                    span_end,
                    span,
                )
        whole = next(
            (
                item
                for item in candidates
                if item[1] == 0 and item[2] == len(payload)
            ),
            None,
        )
        if whole is not None:
            mandatory[(whole[0], whole[1], whole[2])] = whole
        mandatory_values = sorted(
            mandatory.values(),
            key=lambda item: (
                -(item[2] - item[1]),
                item[1],
                item[0],
            ),
        )
        if (
            self.maximum_candidates_per_record is not None
            and len(mandatory_values) >= self.maximum_candidates_per_record
        ):
            selected = mandatory_values[: self.maximum_candidates_per_record]
            if whole is not None and whole not in selected:
                selected[-1] = whole
            return selected
        mandatory_keys = set(mandatory)
        remaining = [
            item
            for item in candidates
            if (item[0], item[1], item[2]) not in mandatory_keys
        ]
        if self.maximum_candidates_per_record is None:
            return mandatory_values + remaining
        # When an explicit candidate limit is requested, stable hash sampling
        # avoids dictionaries or hand-selected terms within that limit.
        sampled = mandatory_values + sorted(
            remaining,
            key=lambda item: hashlib.sha256(
                b"developmental-span\0" + item[3]
            ).digest(),
        )[: self.maximum_candidates_per_record - len(mandatory_values)]
        # Whole experienced utterances are the only forms eligible to become
        # outward motors, so explicit substring limiting must never silently
        # discard that developmental evidence.
        if whole is not None and whole not in sampled:
            sampled[-1] = whole
        return sampled

    def _statistics(self, form_id: str) -> FormStatistics:
        form = self.connection.execute(
            "SELECT * FROM developmental_byte_forms WHERE form_id = ?", (form_id,)
        ).fetchone()
        if form is None:
            raise KeyError(form_id)
        occurrences = self.connection.execute(
            """SELECT record_id, route_ids_json
               FROM developmental_form_occurrences WHERE form_id = ?
               GROUP BY record_id ORDER BY record_id""",
            (form_id,),
        ).fetchall()
        independent_records = len(occurrences)
        byte_length = int(form["byte_length"])
        # Description-length charge prevents every repeated long phrase from
        # winning merely because it is long.
        compression_gain = max(
            0.0,
            (independent_records - 1) * max(1, byte_length - 1)
            - 0.5 * (byte_length + 1),
        )
        total_language_records = int(
            self.connection.execute(
                "SELECT COUNT(DISTINCT record_id) FROM developmental_route_exposures"
            ).fetchone()[0]
        )
        route_counts: dict[str, int] = {}
        for occurrence in occurrences:
            for route_id in json.loads(occurrence["route_ids_json"]):
                route_counts[route_id] = route_counts.get(route_id, 0) + 1
        best_grounding = 0.0
        best_control = 0.0
        best_gap = 0.0
        for route_id, count in route_counts.items():
            conditional = count / max(1, independent_records)
            global_count = int(
                self.connection.execute(
                    """SELECT COUNT(DISTINCT record_id)
                       FROM developmental_route_exposures WHERE route_id = ?""",
                    (route_id,),
                ).fetchone()[0]
            )
            control = global_count / max(1, total_language_records)
            gap = conditional - control
            if (gap, conditional, route_id) > (best_gap, best_grounding, ""):
                best_gap = gap
                best_grounding = conditional
                best_control = control
        return FormStatistics(
            form_id=form_id,
            kind=str(form["kind"]),
            independent_records=independent_records,
            byte_length=byte_length,
            compression_gain=compression_gain,
            grounding_score=best_grounding,
            shuffled_control=best_control,
            grounding_gap=max(0.0, best_gap),
            node_id=form["node_id"],
        )

    def statistics(self, form_id: str) -> FormStatistics:
        return self._statistics(form_id)

    def _developmental_support(
        self, form_id: str
    ) -> tuple[int, int, int]:
        row = self.connection.execute(
            """SELECT COUNT(DISTINCT record_id) AS records,
                      COUNT(DISTINCT utterance_sha256) AS utterances,
                      COUNT(DISTINCT CASE WHEN whole_utterance = 1
                                          THEN record_id END) AS whole_records
               FROM developmental_form_occurrences WHERE form_id = ?""",
            (str(form_id),),
        ).fetchone()
        return (
            int(row["records"]),
            int(row["utterances"]),
            int(row["whole_records"]),
        )

    def _supports_lexical_growth(self, form_id: str, threshold: int) -> bool:
        records, utterances, whole_records = self._developmental_support(
            form_id
        )
        # Exact utterances may become practiced motor units through repeated
        # grounded use. A proper fragment, however, must recur across at least
        # two different utterance shapes; drilling one sentence cannot spawn a
        # dense cloud of every substring it contains.
        return records >= int(threshold) and (
            whole_records >= int(threshold) or utterances >= 2
        )

    def _route_embedding(self, route_ids: Sequence[str]) -> tuple[float, ...]:
        vectors = []
        for route_id in route_ids:
            concept = self.mind.store.get_concept(route_id)
            if concept is not None and any(concept.embedding):
                vectors.append(concept.embedding)
        if not vectors:
            return as_tuple([0.0] * self.mind.embedder.dimension)
        vector = [
            sum(row[index] for row in vectors) / len(vectors)
            for index in range(self.mind.embedder.dimension)
        ]
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return as_tuple(value / norm for value in vector)

    def _ensure_candidate_node(
        self,
        form_id: str,
        statistics: FormStatistics,
        *,
        pulse: int,
    ) -> tuple[str, bool]:
        node_id = statistics.node_id or _lexeme_node_id(form_id)
        occurrences = self.connection.execute(
            """SELECT record_id, route_ids_json, motor_eligible,
                      whole_utterance
               FROM developmental_form_occurrences WHERE form_id = ?
               ORDER BY record_id""",
            (form_id,),
        ).fetchall()
        route_ids = self._nonstructural_routes(
            route
            for occurrence in occurrences
            for route in json.loads(occurrence["route_ids_json"])
        )
        created = not self.mind.store.has_concept(node_id)
        if created:
            self.mind.store.add_concept(
                ConceptNode(
                    concept_id=node_id,
                    label=node_id,
                    kind=BYTE_LEXEME_CANDIDATE_KIND,
                    embedding=self._route_embedding(route_ids),
                    terms=(),
                    vault_id=f"byte-lexeme-vault:{form_id.rsplit(':', 1)[-1]}",
                    created_pulse=int(pulse),
                    last_active_pulse=int(pulse),
                )
            )
        # The surface stays outside the node, but a promoted recurrent byte
        # pattern must be reachable beneath HEAR when that same pattern occurs
        # inside a later utterance. This is the learned receptive fiber.
        receptive_edge = self.mind.graph.add_relation(
            INPUT_NODE_IDS[InputTrunk.HEAR],
            node_id,
            side=GraphSide.INPUT,
            pulse=int(pulse),
            evidence_record_ids=tuple(
                str(occurrence["record_id"]) for occurrence in occurrences
            ),
        )
        if statistics.kind != BYTE_LEXEME_KIND:
            self.mind.store.update_edge_state(
                receptive_edge.edge_id, archived=True
            )
        for occurrence in occurrences:
            record_id = str(occurrence["record_id"])
            concept = self.mind.store.get_concept(node_id)
            if concept is not None and concept.vault_id:
                self.mind.store.add_to_vault(concept.vault_id, record_id, node_id)
            for route_id in self._nonstructural_routes(
                json.loads(occurrence["route_ids_json"])
            ):
                if not self.mind.store.has_concept(route_id):
                    continue
                receptive_relation = self.mind.graph.add_relation(
                    route_id,
                    node_id,
                    side=GraphSide.INPUT,
                    pulse=int(pulse),
                    evidence_record_ids=(record_id,),
                )
                if statistics.kind != BYTE_LEXEME_KIND:
                    self.mind.store.update_edge_state(
                        receptive_relation.edge_id, archived=True
                    )
                if not (
                    bool(occurrence["motor_eligible"])
                    and bool(occurrence["whole_utterance"])
                ):
                    continue
                productive_relation = self.mind.graph.add_relation(
                    route_id,
                    node_id,
                    side=GraphSide.OUTPUT,
                    pulse=int(pulse),
                    evidence_record_ids=(record_id,),
                )
                if statistics.kind != BYTE_LEXEME_KIND:
                    self.mind.store.update_edge_state(
                        productive_relation.edge_id, archived=True
                    )
        next_kind = (
            BYTE_LEXEME_KIND
            if statistics.kind == BYTE_LEXEME_KIND
            else BYTE_LEXEME_CANDIDATE_KIND
        )
        self.connection.execute(
            """UPDATE developmental_byte_forms
               SET kind = ?, node_id = ?, compression_gain = ?,
                   grounding_score = ?, shuffled_control = ?, last_pulse = ?
               WHERE form_id = ?""",
            (
                next_kind,
                node_id,
                statistics.compression_gain,
                statistics.grounding_score,
                statistics.shuffled_control,
                int(pulse),
                form_id,
            ),
        )
        return node_id, created

    def _promote(self, form_id: str, node_id: str, *, pulse: int) -> None:
        self.mind.store.set_concept_kind(node_id, BYTE_LEXEME_KIND)
        for edge in self.mind.store.list_edges(include_archived=True):
            if edge.target_id == node_id and edge.archived:
                self.mind.store.update_edge_state(edge.edge_id, archived=False)
        self.connection.execute(
            """UPDATE developmental_byte_forms
               SET kind = ?, last_pulse = ? WHERE form_id = ?""",
            (BYTE_LEXEME_KIND, int(pulse), form_id),
        )
        self._ensure_motor_form(form_id, node_id, pulse=int(pulse))

    def _ensure_motor_form(self, form_id: str, node_id: str, *, pulse: int) -> None:
        """Retain only a whole heard form, never an arbitrary byte substring."""
        if self.connection.execute(
            "SELECT 1 FROM developmental_motor_forms WHERE form_id = ?", (form_id,)
        ).fetchone() is not None:
            return
        occurrences = self.connection.execute(
            """SELECT record_id, start_byte, end_byte
               FROM developmental_form_occurrences
               WHERE form_id = ? AND motor_eligible = 1
               ORDER BY record_id, start_byte""",
            (form_id,),
        ).fetchall()
        for occurrence in occurrences:
            record = self.mind.store.get_record(str(occurrence["record_id"]))
            if record is None:
                continue
            record_payload = record.text.encode("utf-8")
            start = int(occurrence["start_byte"])
            end = int(occurrence["end_byte"])
            if start != 0 or end != len(record_payload):
                continue
            payload = record_payload[start:end]
            try:
                surface = payload.decode("utf-8", errors="strict")
            except UnicodeDecodeError:
                continue
            if not surface.strip():
                continue
            self.connection.execute(
                """INSERT OR IGNORE INTO developmental_motor_forms(
                       form_id, node_id, payload, payload_sha256,
                       evidence_record_id, promoted_pulse
                   ) VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    form_id,
                    node_id,
                    payload,
                    hashlib.sha256(payload).hexdigest(),
                    record.record_id,
                    int(pulse),
                ),
            )
            return

    def recognize(
        self,
        payload: bytes,
        *,
        maximum: int | None = None,
        maximum_overlap: int | None = None,
    ) -> tuple[RecognizedForm, ...]:
        """Detect promoted forms in current bytes without reading old text.

        Every candidate ID is derived from the bytes currently crossing HEAR.
        The database contributes only learned form statistics and opaque node
        IDs. A small overlap bound lets a longer phrase and one useful nested
        fragment coexist without flooding SELF with every possible substring.
        """
        current = bytes(payload)
        if not current:
            return ()
        raw: list[tuple[str, int, int]] = []
        upper = (
            len(current)
            if self.maximum_span_bytes is None
            else min(self.maximum_span_bytes, len(current))
        )
        for length in range(self.minimum_span_bytes, upper + 1):
            for start in range(0, len(current) - length + 1):
                end = start + length
                span = current[start:end]
                if _eligible_span(span):
                    raw.append((_form_id(span), start, end))
        if not raw:
            return ()
        form_ids = tuple(dict.fromkeys(item[0] for item in raw))
        learned: dict[str, Any] = {}
        for offset in range(0, len(form_ids), 400):
            batch = form_ids[offset : offset + 400]
            placeholders = ",".join("?" for _ in batch)
            rows = self.connection.execute(
                f"""SELECT form_id, node_id, byte_length, grounding_score,
                           shuffled_control
                    FROM developmental_byte_forms
                    WHERE kind = ? AND node_id IS NOT NULL
                      AND form_id IN ({placeholders})""",
                (BYTE_LEXEME_KIND, *batch),
            ).fetchall()
            learned.update({str(row["form_id"]): row for row in rows})
        candidates = []
        for form_id, start, end in raw:
            row = learned.get(form_id)
            if row is None:
                continue
            byte_length = int(row["byte_length"])
            grounding = _bounded_score(float(row["grounding_score"]))
            gap = _bounded_score(
                grounding - float(row["shuffled_control"])
            )
            normalization_length = (
                max(1, len(current))
                if self.maximum_span_bytes is None
                else self.maximum_span_bytes
            )
            length_support = min(
                1.0,
                byte_length / max(4.0, float(normalization_length)),
            )
            activation = _bounded_score(
                0.25 + 0.25 * length_support + 0.25 * grounding + 0.25 * gap
            )
            candidates.append(
                RecognizedForm(
                    form_id=form_id,
                    node_id=str(row["node_id"]),
                    start_byte=start,
                    end_byte=end,
                    byte_length=byte_length,
                    activation=activation,
                    grounding_score=grounding,
                    grounding_gap=gap,
                )
            )
        candidates.sort(
            key=lambda item: (
                -item.activation,
                -item.byte_length,
                item.start_byte,
                item.form_id,
            )
        )
        occupancy = [0] * len(current)
        selected = []
        selected_nodes = set()
        for candidate in candidates:
            if candidate.node_id in selected_nodes:
                continue
            if maximum_overlap is not None and any(
                occupancy[index] >= max(1, int(maximum_overlap))
                for index in range(candidate.start_byte, candidate.end_byte)
            ):
                continue
            selected.append(candidate)
            selected_nodes.add(candidate.node_id)
            for index in range(candidate.start_byte, candidate.end_byte):
                occupancy[index] += 1
            if maximum is not None and len(selected) >= max(1, int(maximum)):
                break
        return tuple(selected)

    def _grounded_routes(self, form_id: str) -> tuple[str, ...]:
        occurrence_rows = self.connection.execute(
            """SELECT record_id, route_ids_json
               FROM developmental_form_occurrences
               WHERE form_id = ? GROUP BY record_id ORDER BY record_id""",
            (str(form_id),),
        ).fetchall()
        independent_records = len(occurrence_rows)
        if independent_records <= 0:
            return ()
        route_counts: dict[str, int] = {}
        for occurrence in occurrence_rows:
            for route_id in self._nonstructural_routes(
                json.loads(occurrence["route_ids_json"])
            ):
                route_counts[route_id] = route_counts.get(route_id, 0) + 1
        total_records = int(
            self.connection.execute(
                "SELECT COUNT(DISTINCT record_id) "
                "FROM developmental_route_exposures"
            ).fetchone()[0]
        )
        global_counts = {
            str(row["route_id"]): int(row["observations"])
            for row in self.connection.execute(
                """SELECT route_id, COUNT(DISTINCT record_id) AS observations
                   FROM developmental_route_exposures GROUP BY route_id"""
            ).fetchall()
        }
        ranked = []
        for route_id, count in route_counts.items():
            conditional = count / independent_records
            control = global_counts.get(route_id, 0) / max(1, total_records)
            gap = conditional - control
            ranked.append((gap, conditional, count, route_id))
        ranked.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3]))
        selected = [
            route_id
            for gap, conditional, _, route_id in ranked
            if conditional >= 0.50 and gap >= 0.05
        ]
        if not selected:
            selected = [item[3] for item in ranked if item[0] > 0.0]
        return tuple(selected)

    def observe_motor_outcome(
        self,
        form_id: str,
        cycle_id: str,
        return_record_id: str,
        *,
        route_ids: Sequence[str],
        stability_delta: float,
        pulse: int,
    ) -> int:
        """Bind a self-produced form to only its verified lived consequence."""
        if not -1.0 <= float(stability_delta) <= 1.0:
            raise ValueError("motor habit stability delta must be within [-1, 1]")
        if self.connection.execute(
            "SELECT 1 FROM developmental_motor_forms WHERE form_id = ?",
            (str(form_id),),
        ).fetchone() is None:
            raise ValueError("motor habit requires a learned whole form")
        cycle = self.mind.store.get_experience_cycle(str(cycle_id))
        if (
            cycle is None
            or cycle.status != "closed"
            or cycle.terminal_return_record_id != str(return_record_id)
        ):
            raise ValueError("motor habit requires a closed terminal cycle")
        observed = next(
            (
                item
                for item in self.mind.store.returns_for_experience_cycle(
                    str(cycle_id)
                )
                if item.record_id == str(return_record_id) and item.verified
            ),
            None,
        )
        if observed is None:
            raise ValueError("motor habit requires a verified return")
        if abs(observed.stability_delta - float(stability_delta)) > 1e-9:
            raise ValueError("motor habit differs from the verified return")
        routes = tuple(
            route_id
            for route_id in self._nonstructural_routes(route_ids)
            if self.mind.store.has_concept(route_id)
        )
        if not routes:
            return 0
        before = self.connection.total_changes
        self.connection.executemany(
            """INSERT OR IGNORE INTO developmental_motor_habit_observations(
                   form_id, cycle_id, return_record_id, route_id,
                   stability_delta, pulse
               ) VALUES (?, ?, ?, ?, ?, ?)""",
            (
                (
                    str(form_id),
                    str(cycle_id),
                    str(return_record_id),
                    route_id,
                    float(stability_delta),
                    int(pulse),
                )
                for route_id in routes
            ),
        )
        return self.connection.total_changes - before

    def _motor_habit(
        self,
        form_id: str,
        current_routes: set[str],
    ) -> tuple[float, float, int]:
        if not current_routes:
            return 0.0, 0.0, 0
        rows = self.connection.execute(
            """SELECT cycle_id, route_id, stability_delta
               FROM developmental_motor_habit_observations
               WHERE form_id = ? ORDER BY pulse, cycle_id, route_id""",
            (str(form_id),),
        ).fetchall()
        contexts: dict[str, set[str]] = {}
        deltas: dict[str, float] = {}
        for row in rows:
            cycle_id = str(row["cycle_id"])
            contexts.setdefault(cycle_id, set()).add(str(row["route_id"]))
            deltas[cycle_id] = float(row["stability_delta"])
        lexical_weights = {
            str(row["node_id"]): (
                1.0
                + min(3.0, int(row["byte_length"]) / 6.0)
                * (
                    0.5
                    + 0.5
                    * _bounded_score(
                        float(row["grounding_score"])
                        - float(row["shuffled_control"])
                    )
                )
            )
            for row in self.connection.execute(
                """SELECT node_id, byte_length, grounding_score,
                          shuffled_control
                   FROM developmental_byte_forms
                   WHERE kind = ? AND node_id IS NOT NULL""",
                (BYTE_LEXEME_KIND,),
            ).fetchall()
        }

        def route_weight(route_id: str) -> float:
            return lexical_weights.get(route_id, 1.0)

        current_mass = sum(route_weight(route) for route in current_routes)
        weighted = []
        for cycle_id, routes in contexts.items():
            overlap_routes = current_routes & routes
            overlap = sum(route_weight(route) for route in overlap_routes)
            if overlap <= 0.0:
                continue
            context_mass = sum(route_weight(route) for route in routes)
            alignment = overlap / math.sqrt(
                max(1.0, current_mass) * max(1.0, context_mass)
            )
            weighted.append((deltas[cycle_id], alignment))
        evidence = sum(weight for _, weight in weighted)
        if evidence <= 1e-12:
            return 0.0, 0.0, 0
        support = sum(delta * weight for delta, weight in weighted) / evidence
        confidence = 1.0 - math.exp(-evidence / 2.0)
        return (
            max(-1.0, min(1.0, support)),
            _bounded_score(confidence),
            len(weighted),
        )

    def available_motor_forms(
        self,
        route_ids: Sequence[str],
        *,
        maximum: int | None = None,
    ) -> tuple[MotorForm, ...]:
        """Return grounded whole-form motor options for the current lived routes."""
        current = set(self._nonstructural_routes(route_ids))
        rows = self.connection.execute(
            """SELECT m.form_id, m.node_id, m.payload, m.payload_sha256
               FROM developmental_motor_forms m
               JOIN developmental_byte_forms f ON f.form_id = m.form_id
               WHERE f.kind = ? ORDER BY m.promoted_pulse, m.form_id""",
            (BYTE_LEXEME_KIND,),
        ).fetchall()
        result = []
        for row in rows:
            form_id = str(row["form_id"])
            payload = bytes(row["payload"])
            payload_hash = hashlib.sha256(payload).hexdigest()
            if payload_hash != str(row["payload_sha256"]):
                raise ValueError("developmental motor form payload hash mismatch")
            try:
                surface = payload.decode("utf-8", errors="strict")
            except UnicodeDecodeError as error:
                raise ValueError(
                    "developmental motor form is not valid UTF-8"
                ) from error
            if not surface.strip():
                raise ValueError("developmental motor form has no outward content")
            statistics = self._statistics(form_id)
            grounded_routes = set(self._grounded_routes(form_id))
            overlap = len(current & grounded_routes)
            alignment = (
                overlap
                / math.sqrt(max(1, len(current)) * max(1, len(grounded_routes)))
                if current and grounded_routes
                else 0.0
            )
            habit_support, habit_confidence, habit_matches = self._motor_habit(
                form_id, current
            )
            result.append(
                MotorForm(
                    form_id=form_id,
                    node_id=str(row["node_id"]),
                    payload=payload,
                    payload_sha256=payload_hash,
                    route_ids=tuple(sorted(grounded_routes)),
                    route_alignment=alignment,
                    grounding_score=statistics.grounding_score,
                    grounding_gap=statistics.grounding_gap,
                    independent_records=statistics.independent_records,
                    habit_support=habit_support,
                    habit_confidence=habit_confidence,
                    habit_match_count=habit_matches,
                )
            )
        result.sort(
            key=lambda item: (
                -(item.habit_support * item.habit_confidence),
                -item.route_alignment,
                -item.grounding_gap,
                -item.grounding_score,
                -len(item.payload),
                item.form_id,
            )
        )
        return tuple(result if maximum is None else result[: max(1, int(maximum))])

    def observe(
        self,
        record: MemoryRecord,
        *,
        route_ids: Sequence[str],
        pulse: int,
        motor_eligible: bool = True,
    ) -> FormGrowthReceipt:
        payload = record.text.encode("utf-8")
        utterance_sha256 = hashlib.sha256(payload).hexdigest()
        routes = self._nonstructural_routes(route_ids)
        spans = self._spans(payload)
        candidate_nodes = []
        promoted_nodes = []
        promoted_forms = []
        with self.mind.store.transaction() as connection:
            connection.executemany(
                """INSERT OR IGNORE INTO developmental_route_exposures(
                       record_id, route_id, pulse
                   ) VALUES (?, ?, ?)""",
                ((record.record_id, route_id, int(pulse)) for route_id in routes),
            )
            for form_id, start, end, span in spans:
                connection.execute(
                    """INSERT INTO developmental_byte_forms(
                           form_id, byte_length, first_pulse, last_pulse
                       ) VALUES (?, ?, ?, ?)
                       ON CONFLICT(form_id) DO UPDATE SET last_pulse=excluded.last_pulse""",
                    (form_id, len(span), int(pulse), int(pulse)),
                )
                connection.execute(
                    """INSERT INTO developmental_form_occurrences(
                           form_id, record_id, start_byte, end_byte, pulse,
                           route_ids_json, motor_eligible, utterance_sha256,
                           whole_utterance
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(form_id, record_id, start_byte, end_byte)
                       DO UPDATE SET motor_eligible = MAX(
                           developmental_form_occurrences.motor_eligible,
                           excluded.motor_eligible
                       )""",
                    (
                        form_id,
                        record.record_id,
                        start,
                        end,
                        int(pulse),
                        _json(routes),
                        int(bool(motor_eligible)),
                        utterance_sha256,
                        int(start == 0 and end == len(payload)),
                    ),
                )
                statistics = self._statistics(form_id)
                if not self._supports_lexical_growth(
                    form_id, self.candidate_records
                ):
                    continue
                node_id, _ = self._ensure_candidate_node(
                    form_id, statistics, pulse=int(pulse)
                )
                candidate_nodes.append(node_id)
                statistics = self._statistics(form_id)
                if statistics.kind == BYTE_LEXEME_KIND:
                    if motor_eligible and start == 0 and end == len(payload):
                        self._ensure_motor_form(
                            form_id, node_id, pulse=int(pulse)
                        )
                    continue
                if (
                    self._supports_lexical_growth(
                        form_id, self.promotion_records
                    )
                    and statistics.compression_gain > self.minimum_compression_gain
                    and statistics.grounding_score >= self.minimum_grounding
                    and statistics.grounding_gap >= self.minimum_grounding_gap
                ):
                    self._promote(form_id, node_id, pulse=int(pulse))
                    promoted_nodes.append(node_id)
                    promoted_forms.append(form_id)
            for form_id, node_id in self._consolidate_ready_forms(pulse=int(pulse)):
                promoted_nodes.append(node_id)
                promoted_forms.append(form_id)
        return FormGrowthReceipt(
            record_id=record.record_id,
            observed_forms=len(spans),
            candidate_node_ids=tuple(dict.fromkeys(candidate_nodes)),
            promoted_node_ids=tuple(dict.fromkeys(promoted_nodes)),
            promoted_form_ids=tuple(dict.fromkeys(promoted_forms)),
        )

    def _consolidate_ready_forms(self, *, pulse: int) -> tuple[tuple[str, str], ...]:
        """Reconsider older candidates as later experiences improve controls."""
        rows = self.connection.execute(
            """SELECT form_id FROM developmental_byte_forms
               WHERE kind != ? ORDER BY last_pulse, form_id""",
            (BYTE_LEXEME_KIND,),
        ).fetchall()
        promoted = []
        for row in rows:
            form_id = str(row["form_id"])
            statistics = self._statistics(form_id)
            if not self._supports_lexical_growth(
                form_id, self.candidate_records
            ):
                continue
            node_id, _ = self._ensure_candidate_node(
                form_id, statistics, pulse=pulse
            )
            statistics = self._statistics(form_id)
            if (
                self._supports_lexical_growth(
                    form_id, self.promotion_records
                )
                and statistics.compression_gain > self.minimum_compression_gain
                and statistics.grounding_score >= self.minimum_grounding
                and statistics.grounding_gap >= self.minimum_grounding_gap
            ):
                self._promote(form_id, node_id, pulse=pulse)
                promoted.append((form_id, node_id))
        return tuple(promoted)

    def audit_surface(self, form_id: str) -> bytes:
        """Recover one surface from canonical evidence; runtime routing never calls this."""
        occurrence = self.connection.execute(
            """SELECT record_id, start_byte, end_byte
               FROM developmental_form_occurrences WHERE form_id = ?
               ORDER BY record_id, start_byte LIMIT 1""",
            (form_id,),
        ).fetchone()
        if occurrence is None:
            raise KeyError(form_id)
        record = self.mind.store.get_record(str(occurrence["record_id"]))
        if record is None:
            raise KeyError(str(occurrence["record_id"]))
        return record.text.encode("utf-8")[
            int(occurrence["start_byte"]) : int(occurrence["end_byte"])
        ]


__all__ = [
    "BYTE_LEXEME_CANDIDATE_KIND",
    "BYTE_LEXEME_KIND",
    "ByteFormLearner",
    "CurriculumAdmission",
    "CurriculumStage",
    "DevelopmentalCurriculum",
    "DevelopmentalMetrics",
    "EpisodeKind",
    "FormGrowthReceipt",
    "FormStatistics",
    "MotorForm",
    "RecognizedForm",
]
