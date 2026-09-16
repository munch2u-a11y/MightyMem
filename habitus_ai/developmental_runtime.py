"""Integrated byte-native developmental runtime for a born-in Habitus cortex."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .developmental_cortex import (
    CortexConfig,
    CortexGeneration,
    CortexOutputState,
    CortexProposal,
    CortexSequenceScore,
    CortexTrainingEpisode,
    DevelopmentalCortex,
    active_route_ids,
    pulse_field,
)
from .developmental_curriculum import (
    ByteFormLearner,
    CurriculumAdmission,
    CurriculumStage,
    DevelopmentalCurriculum,
    EpisodeKind,
    FormGrowthReceipt,
    MotorForm,
    RecognizedForm,
)
from .pipeline import BaseAgenticMemoryRAG
from .graph import OUTPUT_NODE_IDS, SELF_ID
from .self_pulse import (
    AdmissionEffect,
    CognitiveCycleReceipt,
    OUTPUT_RETURN_LANES,
    OutputCandidate,
    PulseCommitFrame,
    SelfPulseKernel,
    SensoryBucket,
)
from .types import (
    DevelopmentalGrowth,
    ExperienceCycle,
    GraphSide,
    InputTrunk,
    MemoryRecord,
    OutputTrunk,
    RecordType,
)


class DevelopmentalByteEmbedder:
    """Fixed byte and byte-pair receptors with no pretrained lexical geometry."""

    def __init__(self, dimension: int = 256):
        if dimension < 16:
            raise ValueError("developmental byte space must be at least 16D")
        self.dimension = int(dimension)
        self.space_id = f"developmental_utf8_receptors_{self.dimension}_v1"

    def embed_bytes(self, payload: bytes) -> list[float]:
        vector = [0.0] * self.dimension
        if not payload:
            return vector
        features: list[tuple[bytes, float]] = []
        for value in payload:
            features.append((b"u" + bytes((value,)), 1.0))
        for left, right in zip(payload, payload[1:]):
            features.append((b"b" + bytes((left, right)), 0.45))
        for index, (feature, weight) in enumerate(features):
            digest = hashlib.sha256(feature).digest()
            slot = int.from_bytes(digest[:8], "big") % self.dimension
            sign = 1.0 if digest[8] & 1 else -1.0
            # Position is deliberately weak: ordering belongs primarily to the
            # recurrent cortex, but the receptor can distinguish gross timing.
            position_weight = 1.0 + 0.05 * ((index % 7) - 3) / 3.0
            vector[slot] += sign * weight * position_weight
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed(self, text: str) -> list[float]:
        return self.embed_bytes(str(text).encode("utf-8"))


@dataclass(frozen=True)
class DevelopmentalInput:
    content: str
    lane: InputTrunk
    source_id: str
    episode_kind: EpisodeKind = EpisodeKind.SENSORY
    embedding: tuple[float, ...] | None = None
    item_id: str | None = None
    concept_scores: Mapping[str, float] | None = None
    metadata: Mapping[str, Any] | None = None
    motor_eligible: bool = True
    # A concrete environmental opportunity may make only named candidates
    # physically available on their motor trunks. SELF still scores and must
    # authorize those candidates; this prevents stale abilities from replacing
    # the one that is actually available in the current frame.
    exclusive_output_node_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class DevelopmentalPulseReceipt:
    cycle: CognitiveCycleReceipt
    growth: tuple[DevelopmentalGrowth, ...]
    admissions: tuple[CurriculumAdmission, ...]
    forms: tuple[FormGrowthReceipt, ...]
    graph_field: tuple[float, ...]
    route_ids: tuple[str, ...]
    cortex_proposal: CortexProposal | None = None
    recognized_forms: tuple[RecognizedForm, ...] = ()


@dataclass(frozen=True)
class DevelopmentalSpeechReceipt:
    """One exact cortex-produced message beneath an authorized SPEAK route."""

    cycle: ExperienceCycle
    payload: bytes
    generation: CortexGeneration
    internal_event_ids: tuple[str, ...]
    proposal_id: str | None
    cortex_output_state: CortexOutputState
    generation_mode: str = "cortex_bytes"
    stop_source: str = "none"
    motor_form_id: str | None = None
    motor_candidates: tuple["SpeechMotorCandidate", ...] = ()
    graph_selector_reliability: float = 0.5
    cortex_selector_reliability: float = 0.5
    cortex_selection_weight: float = 0.30
    selector_evidence_count: int = 0


@dataclass(frozen=True)
class SpeechMotorCandidate:
    """One self-grown whole-form option in a particular lived state."""

    form_id: str
    node_id: str
    payload_sha256: str
    graph_score: float
    cortex_probability: float
    coupled_score: float
    habit_support: float = 0.0
    habit_confidence: float = 0.0
    habit_match_count: int = 0


class BornInHabitusRuntime:
    """One unified pulse path for graph growth, language growth, and cortex state."""

    def __init__(
        self,
        database_path: str | Path,
        *,
        cortex_config: CortexConfig | None = None,
        checkpoint_directory: str | Path | None = None,
        device: str = "auto",
        mind_dimension: int = 256,
        embedder: Any | None = None,
    ) -> None:
        self.embedder = embedder or DevelopmentalByteEmbedder(mind_dimension)
        self.mind = BaseAgenticMemoryRAG(database_path, embedder=self.embedder)
        self.kernel = SelfPulseKernel(self.mind)
        self.curriculum = DevelopmentalCurriculum(self.mind)
        self.forms = ByteFormLearner(self.mind)
        self.cortex = DevelopmentalCortex(
            self.mind,
            config=cortex_config,
            checkpoint_directory=checkpoint_directory,
            device=device,
            curriculum_stage=self.curriculum.stage.value,
        )

    def __enter__(self) -> "BornInHabitusRuntime":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self.kernel.close()
        self.mind.close()

    def _resolved_embedding(self, item: DevelopmentalInput) -> tuple[float, ...]:
        if item.embedding is not None:
            if len(item.embedding) != self.mind.embedder.dimension:
                raise ValueError("developmental input embedding dimension mismatch")
            return tuple(float(value) for value in item.embedding)
        if item.lane == InputTrunk.HEAR:
            return tuple(self.embedder.embed(item.content))
        raise ValueError(
            "non-language developmental input requires a numeric sensor embedding"
        )

    def _verified_motor_counts(self) -> dict[OutputTrunk, int]:
        counts = {trunk: 0 for trunk in OutputTrunk}
        for row in self.mind.store.connection.execute(
            """SELECT c.output_trunk, COUNT(*) AS observations
               FROM experience_cycles c
               JOIN experience_cycle_returns r ON r.cycle_id = c.cycle_id
               WHERE r.verified = 1
               GROUP BY c.output_trunk"""
        ).fetchall():
            counts[OutputTrunk(row["output_trunk"])] = int(row["observations"])
        return counts

    def _speech_selector_calibration(
        self,
    ) -> tuple[float, float, float, int]:
        """Learn how much influence graph and cortex selectors have earned.

        A successfully understood self-generated utterance is evidence for a
        selector when that selector's top candidate was the emitted form. A
        selector that preferred another candidate receives disagreement
        evidence, without assuming which untried alternative would have been
        correct. A negatively returned utterance counts against any selector
        that preferred it. Only immutable output metadata and verified terminal
        returns participate; no expected answer or listener label is read.
        """
        rows = self.mind.store.connection.execute(
            """SELECT out_record.metadata_json, returned.stability_delta
               FROM experience_cycles cycle
               JOIN records out_record
                 ON out_record.record_id = cycle.output_record_id
               JOIN experience_cycle_returns returned
                 ON returned.cycle_id = cycle.cycle_id
                AND returned.record_id = cycle.terminal_return_record_id
               WHERE cycle.status = 'closed'
                 AND cycle.output_trunk = ?
                 AND returned.verified = 1""",
            (OutputTrunk.SPEAK.value,),
        ).fetchall()
        agreements = {"graph": 0.0, "cortex": 0.0}
        disagreements = {"graph": 0.0, "cortex": 0.0}
        evidence_count = 0
        for row in rows:
            metadata = json.loads(row["metadata_json"] or "{}")
            if metadata.get("developmental_self_generated") is not True:
                continue
            selected = metadata.get("developmental_motor_form_id")
            candidates = metadata.get(
                "developmental_motor_candidate_scores", ()
            )
            if not selected or not isinstance(candidates, list) or not candidates:
                continue
            try:
                graph_choice = max(
                    candidates,
                    key=lambda item: (
                        float(item["graph_score"]), str(item["form_id"])
                    ),
                )["form_id"]
                cortex_choice = max(
                    candidates,
                    key=lambda item: (
                        float(item["cortex_probability"]),
                        str(item["form_id"]),
                    ),
                )["form_id"]
            except (KeyError, TypeError, ValueError):
                continue
            stability_delta = float(row["stability_delta"])
            evidence_weight = abs(stability_delta)
            if evidence_weight <= 1e-9:
                continue
            evidence_count += 1
            for name, choice in (
                ("graph", graph_choice),
                ("cortex", cortex_choice),
            ):
                if stability_delta > 0.0 and choice == selected:
                    agreements[name] += evidence_weight
                elif stability_delta > 0.0 or choice == selected:
                    disagreements[name] += evidence_weight

        def reliability(name: str) -> float:
            # One virtual agreement and disagreement keep early estimates
            # uncertain and prevent a single lucky return from taking over.
            return (1.0 + agreements[name]) / (
                2.0 + agreements[name] + disagreements[name]
            )

        graph_reliability = reliability("graph")
        cortex_reliability = reliability("cortex")
        if evidence_count == 0:
            cortex_weight = 0.30
        else:
            cortex_weight = max(
                0.02,
                min(
                    0.45,
                    0.30
                    * cortex_reliability
                    / max(0.05, graph_reliability),
                ),
            )
        return (
            graph_reliability,
            cortex_reliability,
            cortex_weight,
            evidence_count,
        )

    def _developmental_output_candidates(
        self,
        recurrent: Any,
        pulse_id: str,
        *,
        cortex_proposal: CortexProposal | None = None,
        speech_exploration_pressure: float = 0.0,
    ) -> tuple[OutputCandidate, ...]:
        """Expose primitive motors with annealed, target-free exploration."""
        grown = list(self.kernel.default_output_candidates(recurrent))
        if cortex_proposal is not None:
            grown = [
                replace(
                    candidate,
                    learned_support=cortex_proposal.support_for(candidate.trunk),
                    learned_confidence=cortex_proposal.confidence,
                    learned_expected_stability_delta=(
                        cortex_proposal.predicted_consequence
                    ),
                    proposal_id=cortex_proposal.proposal_id,
                )
                for candidate in grown
            ]
        local_weights = self.mind.graph.edge_local_weights(side=GraphSide.OUTPUT)
        states = {item.node_id: item for item in recurrent.node_states}
        verified_counts = self._verified_motor_counts()
        exploration_scale = {
            CurriculumStage.PRELINGUISTIC: 0.16,
            CurriculumStage.GROUNDED_FORMS: 0.08,
            CurriculumStage.FUNCTIONAL_EXCHANGE: 0.03,
            CurriculumStage.EXPERIENCED_NARRATIVE: 0.004,
            CurriculumStage.BROADER_NARRATIVE: 0.0,
        }[self.curriculum.stage]
        for trunk, node_id in OUTPUT_NODE_IDS.items():
            edge = self.mind.store.find_edge(GraphSide.OUTPUT, SELF_ID, node_id)
            if edge is None or edge.archived:
                continue
            state = states.get(node_id)
            exploration = 0.0
            if exploration_scale > 0.0:
                draw = hashlib.sha256(
                    f"{self.cortex.config.seed}\0{pulse_id}\0{trunk.value}".encode(
                        "utf-8"
                    )
                ).digest()[0] / 255.0
                novelty = 1.0 / math.sqrt(1.0 + verified_counts[trunk])
                exploration = (
                    exploration_scale * novelty * (0.35 + 0.65 * draw)
                )
            grown.append(
                OutputCandidate(
                    node_id=node_id,
                    trunk=trunk,
                    path_node_ids=(SELF_ID, node_id),
                    path_edge_ids=(edge.edge_id,),
                    route_probability=local_weights.get(edge.edge_id, 0.0),
                    transient_pull=(
                        max(0.0, state.activation + state.pressure)
                        if state is not None
                        else 0.0
                    )
                    + exploration
                    # A learned receptive pattern and at least one grounded
                    # motor form can make vocal exploration available before
                    # SPEAK has earned its first external receipt.  This does
                    # not choose a form or expose target text, and it anneals
                    # through the verified-SPEAK count used by the caller.
                    # It is therefore motor babble pressure, not an action
                    # override or a scripted response rule.
                    + (
                        max(0.0, float(speech_exploration_pressure))
                        if trunk == OutputTrunk.SPEAK
                        else 0.0
                    ),
                    travel_time=edge.delta_y + edge.conflict_penalty,
                    learned_support=(
                        cortex_proposal.support_for(trunk)
                        if cortex_proposal is not None
                        else 0.0
                    ),
                    learned_confidence=(
                        cortex_proposal.confidence
                        if cortex_proposal is not None
                        else 0.0
                    ),
                    learned_expected_stability_delta=(
                        cortex_proposal.predicted_consequence
                        if cortex_proposal is not None
                        else 0.0
                    ),
                    proposal_id=(
                        cortex_proposal.proposal_id
                        if cortex_proposal is not None
                        else None
                    ),
                )
            )
        unique: dict[tuple[OutputTrunk, str], OutputCandidate] = {}
        for candidate in grown:
            key = (candidate.trunk, candidate.node_id)
            previous = unique.get(key)
            if previous is None:
                unique[key] = candidate
                continue
            preferred = max(
                (previous, candidate),
                key=lambda item: (
                    item.route_probability,
                    -item.travel_time,
                    item.transient_pull,
                ),
            )
            # Parallel paths to the same primitive motor are one opportunity,
            # not competing copies. Preserve the strongest route while letting
            # transient developmental pressure reach that opportunity even when
            # it arrived on the other equivalent path.
            unique[key] = replace(
                preferred,
                transient_pull=max(
                    previous.transient_pull, candidate.transient_pull
                ),
            )
        return tuple(unique.values())

    def advance(
        self,
        inputs: Sequence[DevelopmentalInput] = (),
        *,
        pending_item_ids: Sequence[str] | None = None,
        allow_idle: bool = False,
    ) -> DevelopmentalPulseReceipt:
        """Advance new sensations and/or already durable pending returns.

        ``pending_item_ids`` lets an external adapter settle the exact return it
        just queued through this full developmental path.  Omitting it preserves
        inbox recovery: every currently pending item joins the next pulse.
        """
        pending = self.kernel.pending()
        if pending_item_ids is not None:
            requested = set(dict.fromkeys(str(value) for value in pending_item_ids))
            available = {item.item_id for item in pending}
            missing = requested - available
            if missing:
                raise ValueError(
                    "developmental pulse contains unavailable pending items: "
                    + ", ".join(sorted(missing))
                )
            pending = tuple(item for item in pending if item.item_id in requested)
        if not inputs and not pending and not allow_idle:
            raise ValueError("a developmental pulse requires new or pending input")
        item_ids = [item.item_id for item in pending]
        exclusive_output_node_ids: set[str] = set()
        for item in pending:
            exclusive_nodes = {
                str(node_id)
                for node_id in item.metadata.get(
                    "developmental_exclusive_output_node_ids", ()
                )
            }
            if not exclusive_nodes.issubset(item.concept_scores):
                raise ValueError(
                    "pending exclusive output nodes are not sensed concepts"
                )
            exclusive_output_node_ids.update(exclusive_nodes)
        heard_payloads = [
            item.content.encode("utf-8") for item in pending if item.lane == InputTrunk.HEAR
        ]
        recognized_forms: list[RecognizedForm] = [
            recognized
            for item in pending
            if item.lane == InputTrunk.HEAR
            for recognized in self.forms.recognize(item.content.encode("utf-8"))
        ]
        for item in inputs:
            item_recognitions = (
                self.forms.recognize(item.content.encode("utf-8"))
                if item.lane == InputTrunk.HEAR
                else ()
            )
            recognized_forms.extend(item_recognitions)
            recognition_scores: dict[str, float] = {
                str(node_id): float(score)
                for node_id, score in (item.concept_scores or {}).items()
            }
            exclusive_nodes = {
                str(node_id) for node_id in item.exclusive_output_node_ids
            }
            if not exclusive_nodes.issubset(recognition_scores):
                raise ValueError(
                    "exclusive output nodes must be sensed concept scores"
                )
            exclusive_output_node_ids.update(exclusive_nodes)
            for recognized in item_recognitions:
                recognition_scores[recognized.node_id] = max(
                    recognition_scores.get(recognized.node_id, 0.0),
                    recognized.activation,
                )
            bucket = self.kernel.enqueue_input(
                item.content,
                lane=item.lane,
                source_id=item.source_id,
                concept_scores=recognition_scores,
                embedding=self._resolved_embedding(item),
                allow_growth=False,
                item_id=item.item_id,
                metadata={
                    **dict(item.metadata or {}),
                    "developmental_episode_kind": item.episode_kind.value,
                    "byte_native": item.lane == InputTrunk.HEAR,
                    "pretrained_geometry": bool(getattr(self.embedder, "pretrained_geometry", False)) if item.lane == InputTrunk.HEAR else False,
                    "transcript_window_used": False,
                    "developmental_motor_eligible": bool(
                        item.motor_eligible
                    ),
                    "developmental_exclusive_output_node_ids": sorted(
                        exclusive_nodes
                    ),
                    "developmental_recognized_form_ids": [
                        recognized.form_id
                        for recognized in item_recognitions
                    ],
                },
            )
            item_ids.append(bucket.item_id)
            if item.lane == InputTrunk.HEAR:
                heard_payloads.append(item.content.encode("utf-8"))

        growth: list[DevelopmentalGrowth] = []
        convergence_added = False

        def observe_admission(
            record: MemoryRecord,
            bucket: SensoryBucket,
            created: bool,
        ) -> AdmissionEffect:
            nonlocal convergence_added
            if not created:
                return AdmissionEffect({})
            receipt = self.mind.graph.grow_trunk_breadth(
                record,
                input_trunk=bucket.lane,
                pulse=self.mind.pulse,
            )
            growth.append(receipt)
            excitation: dict[str, float] = {}
            for node_id in receipt.feature_node_ids:
                excitation[node_id] = 0.30
            for node_id in receipt.candidate_pattern_node_ids:
                excitation[node_id] = 0.15
            for node_id in receipt.promoted_pattern_node_ids:
                excitation[node_id] = 0.55
            context = list(receipt.active_context_node_ids)
            represented_trunks = {item.input_trunks[0] for item in growth if item.input_trunks}
            if len(represented_trunks) >= 2 and not convergence_added:
                cross = self.mind.graph.grow_cross_trunk_breadth(
                    tuple(growth), anchor_record=record, pulse=self.mind.pulse
                )
                if cross is not None:
                    growth.append(cross)
                    convergence_added = True
                    context.extend(cross.active_context_node_ids)
                    for node_id in cross.candidate_pattern_node_ids:
                        excitation[node_id] = 0.20
                    for node_id in cross.promoted_pattern_node_ids:
                        excitation[node_id] = 0.65
            return AdmissionEffect(
                excitation,
                context_node_ids=tuple(dict.fromkeys(context)),
            )

        admissions: list[CurriculumAdmission] = []
        form_receipts: list[FormGrowthReceipt] = []
        motor_habit_observations = 0
        final_field: tuple[float, ...] = ()
        final_routes: tuple[str, ...] = ()
        cortex_proposal: CortexProposal | None = None
        speech_exploration_pressure = 0.0

        def propose_outputs(recurrent: Any, pulse_id: str) -> tuple[OutputCandidate, ...]:
            nonlocal cortex_proposal, speech_exploration_pressure
            proposal_field = pulse_field(
                recurrent, (), width=self.cortex.config.graph_width
            )
            cortex_proposal = self.cortex.propose(
                proposal_field, heard_payloads=tuple(heard_payloads)
            )
            speech_exploration_pressure = 0.0
            if recognized_forms:
                current_routes = active_route_ids(
                    recurrent,
                    tuple(item.node_id for item in recognized_forms),
                )
                matching_motors = tuple(
                    item
                    for item in self.forms.available_motor_forms(current_routes)
                    if item.route_alignment > 0.0
                )
                if matching_motors:
                    recognition = max(
                        item.activation for item in recognized_forms
                    )
                    alignment = max(
                        item.route_alignment for item in matching_motors
                    )
                    verified_speech = self._verified_motor_counts()[
                        OutputTrunk.SPEAK
                    ]
                    novelty = 1.0 / math.sqrt(1.0 + verified_speech)
                    speech_exploration_pressure = min(
                        0.45,
                        (0.16 + 0.22 * recognition + 0.12 * alignment)
                        * novelty,
                    )
            candidates = self._developmental_output_candidates(
                recurrent,
                pulse_id,
                cortex_proposal=cortex_proposal,
                speech_exploration_pressure=speech_exploration_pressure,
            )
            if not exclusive_output_node_ids:
                return candidates
            constrained = {
                candidate.node_id: candidate
                for candidate in candidates
                if candidate.node_id in exclusive_output_node_ids
            }
            missing = exclusive_output_node_ids - set(constrained)
            if missing:
                raise ValueError(
                    "exclusive output nodes are not reachable candidates: "
                    + ", ".join(sorted(missing))
                )
            constrained_trunks = {
                candidate.trunk for candidate in constrained.values()
            }
            return tuple(
                candidate
                for candidate in candidates
                if candidate.trunk not in constrained_trunks
                or candidate.node_id in exclusive_output_node_ids
            )

        def observe_state(connection: Any, frame: PulseCommitFrame) -> Mapping[str, Any]:
            nonlocal final_field, final_routes, motor_habit_observations
            final_field = pulse_field(
                frame.recurrent,
                frame.context_node_ids,
                selected_outputs=frame.selected_outputs,
                width=self.cortex.config.graph_width,
            )
            final_routes = active_route_ids(frame.recurrent, frame.context_node_ids)
            nonlanguage_record_ids = {
                record.record_id
                for record in frame.input_records
                if record.metadata.get("causal_trunk") != InputTrunk.HEAR.value
            }
            grounded_language_routes: list[str] = []
            for item in growth:
                if item.cross_trunk:
                    # A promoted convergence may summarize the co-settled
                    # multisensory frame. Candidate intersections remain too
                    # weak to serve as lexical ground.
                    if nonlanguage_record_ids.intersection(item.record_ids):
                        grounded_language_routes.extend(
                            item.promoted_pattern_node_ids
                        )
                    continue
                if (
                    not item.input_trunks
                    or item.input_trunks[0] == InputTrunk.HEAR
                    or not nonlanguage_record_ids.intersection(item.record_ids)
                ):
                    continue
                grounded_language_routes.extend(item.feature_node_ids)
                grounded_language_routes.extend(item.promoted_pattern_node_ids)
            grounded_language_routes = list(
                dict.fromkeys(grounded_language_routes)
            )
            for record in frame.input_records:
                kind = EpisodeKind(
                    str(
                        record.metadata.get(
                            "developmental_episode_kind", EpisodeKind.SENSORY.value
                        )
                    )
                )
                language_episode = (
                    kind != EpisodeKind.SENSORY
                    and record.metadata.get("causal_trunk")
                    == InputTrunk.HEAR.value
                )
                admission = self.curriculum.admit(
                    record,
                    episode_kind=kind,
                    route_ids=(
                        grounded_language_routes
                        if language_episode
                        else final_routes
                    ),
                )
                admissions.append(admission)
                if (
                    admission.accepted
                    and kind != EpisodeKind.SENSORY
                    and record.metadata.get("causal_trunk") == InputTrunk.HEAR.value
                ):
                    form_receipts.append(
                        self.forms.observe(
                            record,
                            route_ids=admission.route_ids,
                            pulse=frame.pulse,
                            motor_eligible=bool(
                                record.metadata.get(
                                    "developmental_motor_eligible", True
                                )
                            ),
                        )
                    )
            for record in frame.input_records:
                if (
                    record.metadata.get("cycle_role") != "return"
                    or record.metadata.get("verified") is not True
                    or record.metadata.get("terminal") is not True
                ):
                    continue
                cycle_id = str(record.metadata.get("experience_id", ""))
                cycle = self.mind.store.get_experience_cycle(cycle_id)
                if cycle is None or cycle.status != "closed":
                    continue
                output_record = self.mind.store.get_record(
                    cycle.output_record_id
                )
                if (
                    output_record is None
                    or output_record.metadata.get(
                        "developmental_self_generated"
                    )
                    is not True
                ):
                    continue
                form_id = output_record.metadata.get(
                    "developmental_motor_form_id"
                )
                if not form_id:
                    continue
                motor_habit_observations += self.forms.observe_motor_outcome(
                    str(form_id),
                    cycle_id,
                    record.record_id,
                    route_ids=tuple(
                        output_record.metadata.get(
                            "developmental_direct_route_ids", ()
                        )
                    ),
                    stability_delta=float(
                        record.metadata.get("stability_delta", 0.0)
                    ),
                    pulse=frame.pulse,
                )
            self.cortex.curriculum_stage = self.curriculum.stage.value
            extension = dict(self.cortex.pulse_observer(connection, frame))
            if cortex_proposal is not None:
                extension["developmental_cortex_proposal"] = {
                    "proposal_id": cortex_proposal.proposal_id,
                    "model_sha256": cortex_proposal.model_sha256,
                    "graph_field_sha256": cortex_proposal.graph_field_sha256,
                    "sensory_payload_sha256": (
                        cortex_proposal.sensory_payload_sha256
                    ),
                    "sensed_byte_count": cortex_proposal.sensed_byte_count,
                    "action_probabilities": dict(
                        cortex_proposal.action_probabilities
                    ),
                    "predicted_consequence": (
                        cortex_proposal.predicted_consequence
                    ),
                    "predicted_valence": cortex_proposal.predicted_valence,
                    "route_alignment": cortex_proposal.route_alignment,
                    "confidence": cortex_proposal.confidence,
                }
            extension["developmental_growth"] = {
                "admission_ids": [item.admission_id for item in admissions],
                "accepted": sum(item.accepted for item in admissions),
                "candidate_forms": sum(
                    len(item.candidate_node_ids) for item in form_receipts
                ),
                "promoted_forms": sum(
                    len(item.promoted_node_ids) for item in form_receipts
                ),
                "grounded_language_route_count": len(
                    grounded_language_routes
                ),
                "recognized_form_count": len(recognized_forms),
                "motor_habit_observations": motor_habit_observations,
                "speech_exploration_pressure": speech_exploration_pressure,
            }
            return extension

        cycle = self.kernel.advance_cycle(
            maximum_per_sensory_lane=max(1, len(item_ids)),
            advance_when_idle=allow_idle,
            item_ids=tuple(item_ids),
            admission_observer=observe_admission,
            output_candidate_provider=propose_outputs,
            pulse_state_observer=observe_state,
        )
        if cycle is None:
            raise RuntimeError("developmental inputs did not produce a SELF pulse")
        return DevelopmentalPulseReceipt(
            cycle=cycle,
            growth=tuple(growth),
            admissions=tuple(admissions),
            forms=tuple(form_receipts),
            graph_field=final_field,
            route_ids=final_routes,
            cortex_proposal=cortex_proposal,
            recognized_forms=tuple(recognized_forms),
        )

    @staticmethod
    def _direct_route_ids(
        receipt: DevelopmentalPulseReceipt,
    ) -> tuple[str, ...]:
        direct_record_ids = {
            item.record_id
            for item in receipt.cycle.sensory.receipts
            if item.kind == "input"
        }
        direct_routes = []
        for sensory in receipt.cycle.sensory.receipts:
            if sensory.kind != "input":
                continue
            direct_routes.extend(sensory.concept_ids)
        for growth in receipt.growth:
            if not any(item in direct_record_ids for item in growth.record_ids):
                continue
            direct_routes.extend(growth.feature_node_ids)
            direct_routes.extend(growth.promoted_pattern_node_ids)
            direct_routes.extend(growth.active_context_node_ids)
        return tuple(
            dict.fromkeys(
                item
                for item in direct_routes
                if item != SELF_ID and not item.startswith(("IN:", "OUT:"))
            )
        )

    def actualize(
        self,
        receipt: DevelopmentalPulseReceipt,
        content: str,
        *,
        trunk: OutputTrunk | None = None,
        source_id: str = "self",
        metadata: Mapping[str, Any] | None = None,
    ) -> ExperienceCycle:
        """Turn one current SELF authorization into a durable motor act.

        A caregiver may choose among the motor opportunities the pulse actually
        authorized, but cannot invent a route that SELF did not expose.
        """
        candidates = receipt.cycle.selected_outputs
        selection_mode = "caregiver_selected_authorized"
        selection_draw = None
        if trunk is not None:
            affordance = next(
                (item for item in candidates if item.trunk == trunk), None
            )
        else:
            affordance = receipt.cycle.selected_output
            verified_counts = self._verified_motor_counts()
            exploration_rate = {
                CurriculumStage.PRELINGUISTIC: 0.50,
                CurriculumStage.GROUNDED_FORMS: 0.20,
                CurriculumStage.FUNCTIONAL_EXCHANGE: 0.05,
                CurriculumStage.EXPERIENCED_NARRATIVE: 0.01,
                CurriculumStage.BROADER_NARRATIVE: 0.0,
            }[self.curriculum.stage]
            material = (
                f"{self.cortex.config.seed}\0{receipt.cycle.self_state.pulse_id}"
                "\0motor-selection"
            ).encode("utf-8")
            digest = hashlib.sha256(material).digest()
            gate = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
            if candidates and gate < exploration_rate:
                selection_draw = int.from_bytes(digest[8:16], "big") / float(
                    2**64 - 1
                )
                least_observed = min(verified_counts[item.trunk] for item in candidates)
                exploratory_candidates = sorted(
                    (
                        item
                        for item in candidates
                        if verified_counts[item.trunk] == least_observed
                    ),
                    key=lambda item: (item.trunk.value, item.node_id),
                )
                selected_index = min(
                    len(exploratory_candidates) - 1,
                    int(selection_draw * len(exploratory_candidates)),
                )
                affordance = exploratory_candidates[selected_index]
                selection_mode = "developmental_stochastic_exploration"
            else:
                selection_mode = "self_ranked"
        if affordance is None:
            requested = trunk.value if trunk is not None else "any"
            raise ValueError(f"SELF did not authorize a {requested} output")
        direct_routes = list(self._direct_route_ids(receipt))
        return self.kernel.actualize_output(
            affordance,
            content,
            source_id=source_id,
            metadata={
                **dict(metadata or {}),
                "developmental_origin_pulse": receipt.cycle.self_state.pulse,
                "developmental_selection_mode": selection_mode,
                "developmental_selection_draw": selection_draw,
                "developmental_route_ids": list(receipt.route_ids),
                "developmental_direct_route_ids": direct_routes,
                "semantic_node_ids": direct_routes,
            },
        )

    def actualize_speech(
        self,
        receipt: DevelopmentalPulseReceipt,
        *,
        maximum_bytes: int | None = None,
        minimum_bytes: int = 1,
        temperature: float = 0.0,
        require_dominant: bool = True,
        allow_promoted_forms: bool = True,
        source_id: str = "developmental-self",
        metadata: Mapping[str, Any] | None = None,
    ) -> DevelopmentalSpeechReceipt:
        """Actualize native speech, optionally requiring neural byte composition.

        Disable promoted forms when evaluating composition: stored whole
        utterances must not supply the payload or rescue an unfinished decode.
        This changes the motor path, not the graph/state available to the cortex.
        """
        affordance = next(
            (
                item
                for item in receipt.cycle.selected_outputs
                if item.trunk == OutputTrunk.SPEAK
            ),
            None,
        )
        if affordance is None:
            raise ValueError("SELF did not authorize a SPEAK output")
        if require_dominant and receipt.cycle.selected_output != affordance:
            raise ValueError("SPEAK was authorized but was not SELF's dominant output")
        if (
            receipt.cortex_proposal is not None
            and receipt.cortex_proposal.model_sha256 != self.cortex.model_sha256
        ):
            raise ValueError("SPEAK authorization predates the current cortex weights")
        direct_routes = self._direct_route_ids(receipt)
        available_motor_forms = (
            tuple(
                item
                for item in self.forms.available_motor_forms(direct_routes)
                if minimum_bytes <= len(item.payload)
                and (maximum_bytes is None or len(item.payload) <= maximum_bytes)
            )
            if allow_promoted_forms
            else ()
        )
        # At least one candidate must actually meet the current lived routes,
        # but nonmatching learned forms remain in the competition as real
        # alternatives. This prevents a hidden graph lookup from deciding the
        # answer before the cortex participates.
        motor_forms = (
            available_motor_forms
            if any(item.route_alignment > 0.0 for item in available_motor_forms)
            else ()
        )
        sequence_scores: tuple[CortexSequenceScore, ...] = ()
        motor_candidates: tuple[SpeechMotorCandidate, ...] = ()
        selected_motor: MotorForm | None = None
        generation_mode = "cortex_bytes"
        (
            graph_selector_reliability,
            cortex_selector_reliability,
            cortex_selection_weight,
            selector_evidence_count,
        ) = self._speech_selector_calibration()
        if motor_forms:
            sequence_scores = self.cortex.score_sequences(
                receipt.graph_field, tuple(item.payload for item in motor_forms)
            )
            neural_by_hash = {
                item.payload_sha256: item for item in sequence_scores
            }
            scored = []
            for motor in motor_forms:
                neural = neural_by_hash[motor.payload_sha256]
                recurrence = min(1.0, motor.independent_records / 5.0)
                graph_score = (
                    0.45 * motor.route_alignment
                    + 0.20 * motor.grounding_gap
                    + 0.10 * motor.grounding_score
                    + 0.05 * recurrence
                    + 0.20
                    * motor.habit_confidence
                    * motor.habit_support
                )
                coupled_score = (
                    (1.0 - cortex_selection_weight) * graph_score
                    + cortex_selection_weight
                    * neural.competition_probability
                )
                scored.append(
                    SpeechMotorCandidate(
                        form_id=motor.form_id,
                        node_id=motor.node_id,
                        payload_sha256=motor.payload_sha256,
                        graph_score=graph_score,
                        cortex_probability=neural.competition_probability,
                        coupled_score=coupled_score,
                        habit_support=motor.habit_support,
                        habit_confidence=motor.habit_confidence,
                        habit_match_count=motor.habit_match_count,
                    )
                )
            motor_candidates = tuple(
                sorted(
                    scored,
                    key=lambda item: (-item.coupled_score, item.form_id),
                )
            )
            selected_candidate = motor_candidates[0]
            selected_motor = next(
                item
                for item in motor_forms
                if item.form_id == selected_candidate.form_id
            )
            generation = self.cortex.motorize_bytes(
                receipt.graph_field,
                selected_motor.payload,
                stop_source="promoted_whole_form",
            )
            generation_mode = "promoted_form_competition"
        else:
            generation = self.cortex.generate_bytes(
                receipt.graph_field,
                maximum_bytes=maximum_bytes,
                minimum_bytes=minimum_bytes,
                temperature=temperature,
            )
        if not generation.stopped:
            raise ValueError("cortex speech did not reach an accepted motor boundary")
        try:
            content = generation.payload.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise ValueError("cortex speech was not valid UTF-8") from error
        if not content.strip():
            raise ValueError("cortex speech contained no externalizable content")

        payload_hash = hashlib.sha256(generation.payload).hexdigest()
        internal_event_ids = []
        if selected_motor is not None:
            _, selection_event_id = self.kernel.advance_internal_output(
                excitation={selected_motor.node_id: 0.35},
                selected_node_ids=(selected_motor.node_id,),
                kind="speech_motor_selection",
                metadata={
                    "form_id": selected_motor.form_id,
                    "payload_sha256": selected_motor.payload_sha256,
                    "generation_mode": generation_mode,
                    "candidate_count": len(motor_candidates),
                },
            )
            internal_event_ids.append(selection_event_id)
        for position, byte in enumerate(generation.payload):
            _, event_id = self.kernel.advance_internal_output(
                excitation={},
                selected_node_ids=(),
                kind="cortex_speech_byte",
                metadata={
                    "position": position,
                    "byte": int(byte),
                    "payload_sha256": payload_hash,
                    "model_sha256": self.cortex.model_sha256,
                    "proposal_id": (
                        receipt.cortex_proposal.proposal_id
                        if receipt.cortex_proposal is not None
                        else None
                    ),
                },
            )
            internal_event_ids.append(event_id)
        _, stop_event_id = self.kernel.advance_internal_output(
            excitation={},
            selected_node_ids=(),
            kind="cortex_speech_stop",
            metadata={
                "payload_sha256": payload_hash,
                "stop_probability": generation.stop_probability,
                "stop_source": generation.stop_source,
                "model_sha256": self.cortex.model_sha256,
            },
        )
        internal_event_ids.append(stop_event_id)
        cortex_output_state = self.cortex.persist_generation(
            stop_event_id, receipt.graph_field, generation
        )
        proposal_id = (
            receipt.cortex_proposal.proposal_id
            if receipt.cortex_proposal is not None
            else None
        )
        direct_routes_list = list(direct_routes)
        cycle = self.kernel.actualize_output(
            affordance,
            content,
            source_id=source_id,
            metadata={
                **dict(metadata or {}),
                "developmental_self_generated": True,
                "developmental_generation_mode": generation_mode,
                "developmental_promoted_forms_allowed": allow_promoted_forms,
                "developmental_origin_pulse": receipt.cycle.self_state.pulse,
                "developmental_route_ids": list(receipt.route_ids),
                "developmental_direct_route_ids": direct_routes_list,
                "semantic_node_ids": direct_routes_list,
                "cortex_model_sha256": self.cortex.model_sha256,
                "cortex_proposal_id": proposal_id,
                "cortex_payload_sha256": payload_hash,
                "cortex_accepted_stop": generation.stopped,
                "cortex_learned_stop": (
                    generation.stop_source == "learned_neural"
                ),
                "cortex_stop_probability": generation.stop_probability,
                "cortex_stop_source": generation.stop_source,
                "developmental_motor_form_id": (
                    selected_motor.form_id if selected_motor is not None else None
                ),
                "developmental_motor_form_node_id": (
                    selected_motor.node_id if selected_motor is not None else None
                ),
                "developmental_motor_candidate_scores": [
                    {
                        "form_id": item.form_id,
                        "node_id": item.node_id,
                        "payload_sha256": item.payload_sha256,
                        "graph_score": item.graph_score,
                        "cortex_probability": item.cortex_probability,
                        "coupled_score": item.coupled_score,
                        "habit_support": item.habit_support,
                        "habit_confidence": item.habit_confidence,
                        "habit_match_count": item.habit_match_count,
                    }
                    for item in motor_candidates
                ],
                "developmental_selector_calibration": {
                    "graph_reliability": graph_selector_reliability,
                    "cortex_reliability": cortex_selector_reliability,
                    "cortex_weight": cortex_selection_weight,
                    "graph_weight": 1.0 - cortex_selection_weight,
                    "verified_evidence_count": selector_evidence_count,
                },
                "cortex_internal_event_ids": internal_event_ids,
                "cortex_output_state_sha256": cortex_output_state.state_sha256,
            },
        )
        return DevelopmentalSpeechReceipt(
            cycle=cycle,
            payload=generation.payload,
            generation=generation,
            internal_event_ids=tuple(internal_event_ids),
            proposal_id=proposal_id,
            cortex_output_state=cortex_output_state,
            generation_mode=generation_mode,
            stop_source=generation.stop_source,
            motor_form_id=(
                selected_motor.form_id if selected_motor is not None else None
            ),
            motor_candidates=motor_candidates,
            graph_selector_reliability=graph_selector_reliability,
            cortex_selector_reliability=cortex_selector_reliability,
            cortex_selection_weight=cortex_selection_weight,
            selector_evidence_count=selector_evidence_count,
        )

    def queue_return(
        self,
        cycle: ExperienceCycle | str,
        content: str,
        *,
        status: str,
        stability_delta: float,
        verified: bool,
        source_id: str = "environment",
        embedding: Sequence[float] | None = None,
        record_type: RecordType | str = RecordType.RECEIPT,
        metadata: Mapping[str, Any] | None = None,
    ) -> SensoryBucket:
        """Route an observed motor consequence into its native sensory lane."""
        resolved = (
            cycle
            if isinstance(cycle, ExperienceCycle)
            else self.mind.store.get_experience_cycle(str(cycle))
        )
        if resolved is None:
            raise KeyError(f"unknown experience cycle: {cycle}")
        return_lane = OUTPUT_RETURN_LANES[resolved.output_trunk]
        recognized = (
            self.forms.recognize(str(content).encode("utf-8"))
            if return_lane == InputTrunk.HEAR
            else ()
        )
        return self.kernel.enqueue_cycle_return(
            resolved.cycle_id,
            content,
            lane=return_lane,
            status=status,
            stability_delta=stability_delta,
            verified=verified,
            source_id=source_id,
            record_type=record_type,
            concept_scores={
                item.node_id: item.activation for item in recognized
            },
            embedding=(
                embedding
                if embedding is not None
                else self.embedder.embed(str(content))
                if return_lane == InputTrunk.HEAR
                else None
            ),
            metadata={
                **dict(metadata or {}),
                "developmental_return": True,
                "byte_native": return_lane == InputTrunk.HEAR,
                "pretrained_geometry": bool(getattr(self.embedder, "pretrained_geometry", False)) if return_lane == InputTrunk.HEAR else False,
                "transcript_window_used": False,
                "developmental_recognized_form_ids": [
                    item.form_id for item in recognized
                ],
            },
        )

    def speech_rehearsal_episodes(
        self, receipt: DevelopmentalPulseReceipt
    ) -> tuple[CortexTrainingEpisode, ...]:
        """Derive low-weight motor imitation from admitted grounded hearing.

        These episodes teach the mechanics of producing an experienced byte
        sequence.  They remain explicitly unverified and carry no action or
        reward label; only the agent's own later SPEAK cycle can establish that
        a sequence communicated successfully.
        """
        if self.curriculum.stage == CurriculumStage.PRELINGUISTIC:
            return ()
        direct_input_record_ids = {
            item.record_id
            for item in receipt.cycle.sensory.receipts
            if item.kind == "input"
        }
        result = []
        for admission in receipt.admissions:
            if (
                not admission.accepted
                or admission.record_id not in direct_input_record_ids
                or admission.episode_kind == EpisodeKind.SENSORY
            ):
                continue
            record = self.mind.store.get_record(admission.record_id)
            if (
                record is None
                or record.metadata.get("causal_trunk") != InputTrunk.HEAR.value
                or record.metadata.get("developmental_motor_eligible", True)
                is not True
            ):
                continue
            result.append(
                CortexTrainingEpisode(
                    episode_id=(
                        f"cortex-rehearsal:{receipt.cycle.self_state.pulse}:"
                        f"{record.record_id}"
                    ),
                    pulse=receipt.cycle.self_state.pulse,
                    record_ids=(record.record_id,),
                    route_ids=admission.route_ids,
                    graph_field=receipt.graph_field,
                    payload=record.text.encode("utf-8"),
                    direction="speak",
                    kind=admission.episode_kind.value,
                    motor_rehearsal=True,
                )
            )
        return tuple(result)

    def training_episodes(
        self,
        receipt: DevelopmentalPulseReceipt,
        *,
        experience_cycle_id: str | None = None,
        valence_delta: float | None = None,
        reward: float | None = None,
    ) -> tuple[CortexTrainingEpisode, ...]:
        """Build training items from admissions and, optionally, a real return.

        Language and route reconstruction can learn from accepted sensations.
        Action, consequence, and preference heads remain disabled unless an
        output authorized by this exact pulse has a verified terminal return.
        """
        cycle = None
        observed = None
        if experience_cycle_id is not None:
            cycle = self.mind.store.get_experience_cycle(experience_cycle_id)
            if cycle is None:
                raise KeyError(f"unknown experience cycle: {experience_cycle_id}")
            if cycle.output_pulse_id != receipt.cycle.self_state.pulse_id:
                raise ValueError("experience cycle did not originate from this SELF pulse")
            if cycle.status != "closed" or cycle.terminal_return_record_id is None:
                raise ValueError("experience cycle has no terminal return")
            observed = next(
                (
                    item
                    for item in self.mind.store.returns_for_experience_cycle(
                        experience_cycle_id
                    )
                    if item.record_id == cycle.terminal_return_record_id
                ),
                None,
            )
            if observed is None or not observed.verified:
                raise ValueError("experience cycle has no verified terminal receipt")
        direct_input_record_ids = {
            item.record_id
            for item in receipt.cycle.sensory.receipts
            if item.kind == "input"
        }
        accepted = {
            item.record_id: item
            for item in receipt.admissions
            if item.accepted and item.record_id in direct_input_record_ids
        }
        result = []
        for record_id, admission in accepted.items():
            record = self.mind.store.get_record(record_id)
            if record is None:
                continue
            payload = (
                record.text.encode("utf-8")
                if record.metadata.get("causal_trunk") == InputTrunk.HEAR.value
                else b""
            )
            evidence_record_ids = [record_id]
            if cycle is not None and observed is not None:
                evidence_record_ids.extend(
                    (cycle.output_record_id, observed.record_id)
                )
            verified = observed is not None
            consequence = observed.stability_delta if observed is not None else 0.0
            result.append(
                CortexTrainingEpisode(
                    episode_id=(
                        f"cortex-episode:{receipt.cycle.self_state.pulse}:"
                        f"{record_id}:{cycle.cycle_id if cycle is not None else 'sensory'}"
                    ),
                    pulse=receipt.cycle.self_state.pulse,
                    record_ids=tuple(evidence_record_ids),
                    route_ids=admission.route_ids,
                    graph_field=receipt.graph_field,
                    payload=payload,
                    direction="hear" if payload else "quiet",
                    kind=admission.episode_kind.value,
                    consequence=consequence,
                    valence_delta=(
                        consequence if valence_delta is None else valence_delta
                    ),
                    selected_action=(cycle.output_trunk if cycle is not None else None),
                    verified=verified,
                    reward=(consequence if reward is None else reward),
                    experience_cycle_id=(cycle.cycle_id if cycle is not None else None),
                    return_record_id=(observed.record_id if observed is not None else None),
                )
            )
        if (
            cycle is not None
            and observed is not None
            and cycle.output_trunk == OutputTrunk.SPEAK
        ):
            output_record = self.mind.store.get_record(cycle.output_record_id)
            basis = next(iter(accepted.values()), None)
            current_heard_records = tuple(
                record
                for record_id in accepted
                if (record := self.mind.store.get_record(record_id)) is not None
                and record.metadata.get("causal_trunk")
                == InputTrunk.HEAR.value
            )
            if (
                output_record is not None
                and output_record.metadata.get("developmental_self_generated") is True
                and basis is not None
            ):
                consequence = observed.stability_delta
                result.append(
                    CortexTrainingEpisode(
                        episode_id=(
                            f"cortex-self-speech:{receipt.cycle.self_state.pulse}:"
                            f"{cycle.cycle_id}"
                        ),
                        pulse=receipt.cycle.self_state.pulse,
                        record_ids=tuple(
                            dict.fromkeys(
                                (
                                    basis.record_id,
                                    *(record.record_id for record in current_heard_records),
                                    output_record.record_id,
                                    observed.record_id,
                                )
                            )
                        ),
                        route_ids=basis.route_ids,
                        graph_field=receipt.graph_field,
                        payload=output_record.text.encode("utf-8"),
                        conditioning_payloads=tuple(
                            record.text.encode("utf-8")
                            for record in current_heard_records
                        ),
                        direction="speak",
                        kind=basis.episode_kind.value,
                        consequence=consequence,
                        valence_delta=(
                            consequence if valence_delta is None else valence_delta
                        ),
                        selected_action=OutputTrunk.SPEAK,
                        verified=True,
                        reward=(consequence if reward is None else reward),
                        experience_cycle_id=cycle.cycle_id,
                        return_record_id=observed.record_id,
                    )
                )
        return tuple(result)


__all__ = [
    "BornInHabitusRuntime",
    "DevelopmentalByteEmbedder",
    "DevelopmentalInput",
    "DevelopmentalPulseReceipt",
    "DevelopmentalSpeechReceipt",
    "SpeechMotorCandidate",
]
