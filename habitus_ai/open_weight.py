from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import itertools
import math
from pathlib import Path
import struct
from typing import Callable, Iterable, Mapping, Sequence

from .embeddings import cosine_similarity, opaque_payload_embedding
from .graph import (
    EXPERIENTIAL_CONTEXT_KINDS,
    INPUT_NODE_IDS,
    OUTPUT_NODE_IDS,
    SELF_ID,
    WeightSnapshot,
)
from .pipeline import BaseAgenticMemoryRAG
from .recurrent import DESIRE_KINDS
from .self_pulse import (
    AdmissionEffect,
    CognitiveCycleReceipt,
    OutputCandidate,
    SelfPulseKernel,
    SensoryBucket,
)
from .types import (
    ConceptNode,
    DevelopmentalGrowth,
    ExperienceCycle,
    GraphEdge,
    GraphSide,
    InputTrunk,
    MemoryRecord,
    OutputTrunk,
    RecordType,
    RecurrentSnapshot,
    SocialSourceState,
    TraversalTrace,
    as_tuple,
)


@dataclass(frozen=True)
class OpenWeightTrajectory:
    trunk: OutputTrunk
    terminal_node_id: str
    path_node_ids: tuple[str, ...]
    path_edge_ids: tuple[str, ...]
    path_score: float
    transient_pull: float
    within_trunk_probability: float = 0.0
    trunk_probability: float = 0.0
    effective_probability: float = 0.0


@dataclass(frozen=True)
class OpenWeightFocus:
    pulse_id: str
    candidates: tuple[OpenWeightTrajectory, ...]
    trunk_probabilities: Mapping[str, float]
    selected: OpenWeightTrajectory | None


@dataclass(frozen=True)
class NativeStateFrame:
    """Fixed numeric boundary sent to a decoder instead of text context."""

    pulse_id: str
    rows: tuple[tuple[float, ...], ...]
    row_kinds: tuple[str, ...]
    state_sha256: str
    input_sha256: str | None
    selected_path_node_ids: tuple[str, ...]
    selected_path_edge_ids: tuple[str, ...]
    transcript_records_used: int = 0
    recalled_records_used: int = 0


@dataclass(frozen=True)
class OpenWeightTurnPlan:
    pulse_id: str
    input_record: MemoryRecord | None
    input_concept_ids: tuple[str, ...]
    input_traces: tuple[TraversalTrace, ...]
    recurrent: RecurrentSnapshot
    focus: OpenWeightFocus
    frame: NativeStateFrame | None
    sensation: "LexicalSensation | None" = None
    acquisition: "LexicalAcquisition | None" = None
    source_node_id: str | None = None
    self_pulse: CognitiveCycleReceipt | None = None
    output_authorization_id: str | None = None
    developmental_growth: tuple[DevelopmentalGrowth, ...] = ()


@dataclass(frozen=True)
class LexicalSensation:
    """Current HEAR event as numeric language geometry, never a history window."""

    input_sha256: str
    utterance_embedding: tuple[float, ...]
    lexical_rows: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class LexicalAcquisition:
    """Structural changes caused by one current caregiver utterance."""

    record_id: str
    source_node_id: str
    lexical_node_ids: tuple[str, ...]
    observed_units: int
    candidate_nodes_created: tuple[str, ...]
    nodes_promoted: tuple[str, ...]
    semantic_edge_ids: tuple[str, ...]
    transition_edge_ids: tuple[str, ...]
    transition_edges_staged: tuple[str, ...]
    transition_edges_dormant: tuple[str, ...]
    transition_edges_promoted: tuple[str, ...]
    source_edge_ids: tuple[str, ...]
    truncated: bool = False


@dataclass(frozen=True)
class SocialReturnAssessment:
    """Auditable, graph-derived interpretation of one reply to prior speech."""

    cycle_id: str
    source_node_id: str
    semantic_alignment: float
    lexical_overlap: float
    content_valence: float
    source_preference: float
    source_confidence: float
    stability_delta: float
    evidence_quality: float


@dataclass(frozen=True)
class LexicalCandidate:
    """One word-or-stop branch competing at a single speech pulse."""

    node_id: str
    transition_edge_id: str
    transition_probability: float
    semantic_support: float
    convergence_count: int
    lexical_activation: float
    geometry_alignment: float
    boundary_pull: float
    repetition_penalty: float
    score: float
    probability: float = 0.0
    stop: bool = False
    supporting_node_ids: tuple[str, ...] = ()
    supporting_edge_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class LexicalStep:
    """Auditable result of one freshly recalculated lexical competition."""

    pulse_id: str
    index: int
    cursor_node_id: str
    candidates: tuple[LexicalCandidate, ...]
    selected: LexicalCandidate
    state_before_sha256: str
    state_after_sha256: str


@dataclass(frozen=True)
class RecurrentSpeech:
    """A word sequence actualized one graph pulse at a time.

    ``lexical_rows`` are the winning node geometries.  They are an output
    buffer, not a prompt or a transcript supplied to another reasoner.
    """

    pulse_id: str
    steps: tuple[LexicalStep, ...]
    lexical_node_ids: tuple[str, ...]
    lexical_rows: tuple[tuple[float, ...], ...]
    path_node_ids: tuple[str, ...]
    path_edge_ids: tuple[str, ...]
    causal_edge_ids: tuple[str, ...]
    stopped: bool
    final_recurrent: RecurrentSnapshot
    frame: NativeStateFrame


SPEECH_START_KIND = "speech_start"
SPEECH_STOP_KIND = "speech_stop"
LEXEME_CANDIDATE_KIND = "lexeme_candidate"
SOCIAL_SOURCE_KIND = "social_source"
LEXICAL_KINDS = frozenset({"lexeme", LEXEME_CANDIDATE_KIND})


@dataclass(frozen=True)
class _PartialPath:
    node_ids: tuple[str, ...]
    edge_ids: tuple[str, ...]
    log_probability: float
    transient_pull: float
    distance: float

    @property
    def score(self) -> float:
        return (
            self.log_probability + 1.8 * self.transient_pull - 0.06 * self.distance
        ) / (max(1, len(self.edge_ids)) ** 0.70)


def _softmax(values: Sequence[float], *, temperature: float = 1.0) -> list[float]:
    if not values:
        return []
    temperature = max(0.05, float(temperature))
    maximum = max(values)
    exponentials = [math.exp((value - maximum) / temperature) for value in values]
    total = sum(exponentials) or 1.0
    return [value / total for value in exponentials]


def _local_for_edges(
    edges: Sequence[GraphEdge],
    snapshot: WeightSnapshot,
) -> dict[str, float]:
    if not edges:
        return {}
    probabilities = {
        edge.edge_id: snapshot.local_weights.get(edge.edge_id, 0.0)
        for edge in edges
    }
    total = sum(probabilities.values())
    if total <= 0.0:
        share = 1.0 / len(edges)
        return {edge.edge_id: share for edge in edges}
    return {edge_id: value / total for edge_id, value in probabilities.items()}


def _normalize(vector: Sequence[float]) -> tuple[float, ...]:
    norm = math.sqrt(sum(float(value) * float(value) for value in vector))
    if norm <= 1e-12:
        raise ValueError("cannot normalize an empty native state row")
    return tuple(float(value) / norm for value in vector)


def _weighted_field(
    dimension: int,
    terms: Iterable[tuple[Sequence[float], float]],
) -> tuple[float, ...]:
    values = [0.0] * dimension
    used = False
    for vector, weight in terms:
        if len(vector) != dimension or not math.isfinite(float(weight)):
            raise ValueError("invalid native state term")
        if abs(weight) <= 1e-15:
            continue
        used = True
        for index, value in enumerate(vector):
            values[index] += float(value) * float(weight)
    if not used:
        raise ValueError("native state field has no active terms")
    return _normalize(values)


def _trunk_for_path(path: Sequence[str]) -> OutputTrunk | None:
    if len(path) < 2:
        return None
    return next(
        (trunk for trunk, node_id in OUTPUT_NODE_IDS.items() if node_id == path[1]),
        None,
    )


def enumerate_output_trajectories(
    mind: BaseAgenticMemoryRAG,
    activation: Mapping[str, float],
    snapshot: WeightSnapshot,
    *,
    maximum_depth: int | None = None,
    beam_width: int | None = None,
) -> tuple[OpenWeightTrajectory, ...]:
    concept_by_id = {
        concept.concept_id: concept for concept in mind.store.list_concepts()
    }
    traversal_depth = (
        max(2, int(maximum_depth))
        if maximum_depth is not None
        else max(2, len(concept_by_id))
    )
    outgoing: dict[str, list[GraphEdge]] = {}
    output_edges = mind.store.list_edges(GraphSide.OUTPUT)
    for edge in output_edges:
        outgoing.setdefault(edge.source_id, []).append(edge)
    lexically_grounded = {
        edge.source_id
        for edge in output_edges
        if (target := concept_by_id.get(edge.target_id)) is not None
        and target.kind in {"lexeme", SPEECH_START_KIND}
    }
    frontier = [_PartialPath((SELF_ID,), (), 0.0, 0.0, 0.0)]
    terminals: dict[tuple[OutputTrunk, str], OpenWeightTrajectory] = {}
    for _ in range(traversal_depth):
        following = []
        for partial in frontier:
            source_id = partial.node_ids[-1]
            local = _local_for_edges(outgoing.get(source_id, ()), snapshot)
            for edge in outgoing.get(source_id, ()):
                if edge.target_id in partial.node_ids:
                    continue
                target = concept_by_id.get(edge.target_id)
                if target is None or target.kind in LEXICAL_KINDS:
                    continue
                probability = max(1e-12, local.get(edge.edge_id, 0.0))
                pull = max(0.0, min(1.0, activation.get(edge.target_id, 0.0)))
                advanced = _PartialPath(
                    (*partial.node_ids, edge.target_id),
                    (*partial.edge_ids, edge.edge_id),
                    partial.log_probability + math.log(probability),
                    partial.transient_pull + pull,
                    partial.distance + edge.delta_y,
                )
                trunk = _trunk_for_path(advanced.node_ids)
                terminal = target.kind == "ability" or (
                    trunk == OutputTrunk.SPEAK
                    and target.kind in {"crown", *DESIRE_KINDS}
                    and edge.target_id in lexically_grounded
                )
                if terminal:
                    candidate = OpenWeightTrajectory(
                        trunk=trunk,
                        terminal_node_id=edge.target_id,
                        path_node_ids=advanced.node_ids,
                        path_edge_ids=advanced.edge_ids,
                        path_score=advanced.score,
                        transient_pull=advanced.transient_pull,
                    )
                    key = (trunk, edge.target_id)
                    previous = terminals.get(key)
                    if previous is None or candidate.path_score > previous.path_score:
                        terminals[key] = candidate
                elif len(advanced.edge_ids) < traversal_depth:
                    following.append(advanced)
        ranked_following = sorted(
            following, key=lambda item: (item.score, item.node_ids), reverse=True
        )
        frontier = (
            ranked_following
            if beam_width is None
            else ranked_following[: max(1, int(beam_width))]
        )
        if not frontier:
            break
    return tuple(
        sorted(
            terminals.values(),
            key=lambda item: (item.path_score, item.terminal_node_id),
            reverse=True,
        )
    )


def resolve_output_focus(
    mind: BaseAgenticMemoryRAG,
    recurrent: RecurrentSnapshot,
    *,
    pulse_id: str,
    mark_active: bool = True,
) -> OpenWeightFocus:
    activation = mind.recurrent.activation_values(recurrent)
    snapshot = mind.graph.weight_snapshot(side=GraphSide.OUTPUT)
    raw = enumerate_output_trajectories(mind, activation, snapshot)
    if not raw:
        return OpenWeightFocus(pulse_id, (), {}, None)

    by_trunk: dict[OutputTrunk, list[OpenWeightTrajectory]] = {}
    for candidate in raw:
        by_trunk.setdefault(candidate.trunk, []).append(candidate)
    local_probability: dict[tuple[OutputTrunk, str], float] = {}
    trunk_energy: dict[OutputTrunk, float] = {}
    output_edges = mind.store.list_edges(GraphSide.OUTPUT)
    root_local = _local_for_edges(
        [edge for edge in output_edges if edge.source_id == SELF_ID], snapshot
    )
    for trunk, candidates in by_trunk.items():
        probabilities = _softmax([item.path_score for item in candidates])
        for candidate, probability in zip(candidates, probabilities):
            local_probability[(trunk, candidate.terminal_node_id)] = probability
        root_edge = mind.store.find_edge(GraphSide.OUTPUT, SELF_ID, OUTPUT_NODE_IDS[trunk])
        prior = max(1e-12, root_local.get(root_edge.edge_id, 0.0) if root_edge else 0.0)
        trunk_energy[trunk] = math.log(prior) + 1.8 * max(
            item.transient_pull for item in candidates
        )
    ordered_trunks = sorted(trunk_energy, key=lambda item: item.value)
    gate_values = _softmax([trunk_energy[item] for item in ordered_trunks])
    gates = dict(zip(ordered_trunks, gate_values))
    normalized = tuple(
        replace(
            candidate,
            within_trunk_probability=local_probability[
                (candidate.trunk, candidate.terminal_node_id)
            ],
            trunk_probability=gates[candidate.trunk],
            effective_probability=(
                local_probability[(candidate.trunk, candidate.terminal_node_id)]
                * gates[candidate.trunk]
            ),
        )
        for candidate in raw
    )
    selected = max(
        normalized,
        key=lambda item: (
            item.effective_probability,
            item.transient_pull,
            item.terminal_node_id,
        ),
    )
    if mark_active:
        selected_edges = [
            mind.store.get_edge(edge_id) for edge_id in selected.path_edge_ids
        ]
        trace = TraversalTrace(
            trace_id=f"trace:{pulse_id}:open-weight:{selected.terminal_node_id}",
            side=GraphSide.OUTPUT,
            start_node_id=SELF_ID,
            target_node_id=selected.terminal_node_id,
            path_node_ids=selected.path_node_ids,
            path_edge_ids=selected.path_edge_ids,
            total_travel_time=sum(
                edge.delta_y for edge in selected_edges if edge is not None
            ),
            endpoint_score=selected.effective_probability,
        )
        mind.graph.activate_trace(pulse_id, trace)
        mind.recurrent.mark_selected(selected.path_node_ids, pulse=recurrent.pulse)
    return OpenWeightFocus(
        pulse_id=pulse_id,
        candidates=normalized,
        trunk_probabilities={trunk.value: gates[trunk] for trunk in ordered_trunks},
        selected=selected,
    )


def _field_or_self(
    mind: BaseAgenticMemoryRAG,
    terms: Iterable[tuple[Sequence[float], float]],
) -> tuple[float, ...]:
    material = list(terms)
    if material:
        return _weighted_field(mind.embedder.dimension, material)
    self_node = mind.store.get_concept(SELF_ID)
    if self_node is None or not any(self_node.embedding):
        raise RuntimeError("SELF has no native geometry")
    return _normalize(self_node.embedding)


def _speech_boundary_id(mind: BaseAgenticMemoryRAG, kind: str) -> str:
    matches = mind.store.list_concepts(kind=kind)
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one {kind} node, found {len(matches)}")
    return matches[0].concept_id


def lexical_geometry_id(embedding: Sequence[float]) -> str:
    """Return the opaque identity used by all geometry-only lexical nodes."""
    digest = hashlib.sha256()
    for value in embedding:
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("lexical geometry contains a non-finite value")
        digest.update(struct.pack("<f", numeric))
    return f"LXG:{digest.hexdigest()[:16]}"


def social_source_node_id(source_id: str) -> str:
    """Map a transport identity to a stable graph identity without storing its text."""
    material = str(source_id).strip()
    if not material:
        raise ValueError("social source ID cannot be empty")
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]
    return f"SRC:{digest}"


class DevelopmentalLanguage:
    """Grow receptive/productive language and source-conditioned social habits.

    The learner sees only the current HEAR event, its model-native lexical rows,
    and live graph routes. Canonical text remains evidence but is never replayed
    to choose a word or evaluate a social return.
    """

    def __init__(
        self,
        mind: BaseAgenticMemoryRAG,
        *,
        promotion_exposures: int = 3,
        transition_promotion_exposures: int | None = None,
        maximum_units: int | None = None,
        semantic_contexts_per_unit: int | None = None,
        maximum_semantic_contexts: int | None = None,
    ) -> None:
        self.mind = mind
        self.promotion_exposures = max(2, int(promotion_exposures))
        self.transition_promotion_exposures = max(
            2,
            int(
                transition_promotion_exposures
                if transition_promotion_exposures is not None
                else self.promotion_exposures
            ),
        )
        self.maximum_units = (
            None if maximum_units is None else max(1, int(maximum_units))
        )
        self.semantic_contexts_per_unit = (
            None
            if semantic_contexts_per_unit is None
            else max(1, int(semantic_contexts_per_unit))
        )
        self.maximum_semantic_contexts = (
            None
            if maximum_semantic_contexts is None
            else max(
                self.semantic_contexts_per_unit or 1,
                int(maximum_semantic_contexts),
            )
        )
        self.mind.store.set_metadata(
            "online_language_schema", "habitus.online-language.v2"
        )
        self.mind.store.set_metadata(
            "online_language_promotion_exposures", str(self.promotion_exposures)
        )
        self.mind.store.set_metadata(
            "online_language_transition_promotion_exposures",
            str(self.transition_promotion_exposures),
        )
        self.mind.store.set_metadata(
            "online_language_maximum_units", str(self.maximum_units)
        )
        self.mind.store.set_metadata(
            "online_language_maximum_semantic_contexts",
            str(self.maximum_semantic_contexts),
        )
        self.mind.store.set_metadata(
            "social_return_schema", "habitus.source-content-return.v1"
        )

    def ensure_source(self, source_id: str) -> str:
        node_id = social_source_node_id(source_id)
        if self.mind.store.get_concept(node_id) is None:
            self.mind.store.add_concept(
                ConceptNode(
                    concept_id=node_id,
                    label=node_id,
                    kind=SOCIAL_SOURCE_KIND,
                    embedding=as_tuple(
                        opaque_payload_embedding(
                            str(source_id),
                            self.mind.embedder.dimension,
                            namespace="social-source",
                        )
                    ),
                    terms=(),
                    vault_id=f"social-vault:{node_id}",
                    created_pulse=self.mind.pulse,
                    last_active_pulse=self.mind.pulse,
                )
            )
        self.mind.graph.add_relation(
            INPUT_NODE_IDS[InputTrunk.HEAR],
            node_id,
            side=GraphSide.INPUT,
            pulse=self.mind.pulse,
        )
        self.mind.recurrent.register(
            node_id,
            persistence=0.82,
            expression_threshold=2.0,
            pulse=self.mind.pulse,
        )
        return node_id

    def source_preference(self, node_id: str) -> tuple[float, float]:
        state = self.mind.store.get_social_source_state(node_id)
        if state is None:
            return 0.0, 0.0
        confidence = state.preference_weight / (state.preference_weight + 3.0)
        return state.preference_mean, max(0.0, min(1.0, confidence))

    def learn_source_preference(
        self,
        source_node_id: str,
        preference: float,
        *,
        confidence: float,
    ) -> SocialSourceState:
        concept = self.mind.store.get_concept(source_node_id)
        if concept is None or concept.kind != SOCIAL_SOURCE_KIND:
            raise ValueError("source preference requires an existing social-source node")
        learned = self.mind.store.update_social_source_state(
            source_node_id,
            preference=preference,
            confidence=confidence,
            pulse=self.mind.pulse,
        )
        dynamics = self.mind.store.get_node_dynamics(source_node_id)
        if dynamics is None:
            dynamics = self.mind.recurrent.register(
                source_node_id, persistence=0.82, expression_threshold=2.0,
                pulse=self.mind.pulse,
            )
        self.mind.store.put_node_dynamics(
            replace(
                dynamics,
                valence=learned.preference_mean,
                last_pulse=self.mind.pulse,
            )
        )
        return learned

    def _ensure_lexical_node(
        self,
        embedding: Sequence[float],
    ) -> tuple[str, bool]:
        if len(embedding) != self.mind.embedder.dimension:
            raise ValueError("lexical sensation dimension mismatch")
        if not any(float(value) for value in embedding):
            raise ValueError("lexical sensation has empty geometry")
        node_id = lexical_geometry_id(embedding)
        existing = self.mind.store.get_concept(node_id)
        if existing is None:
            self.mind.store.add_concept(
                ConceptNode(
                    concept_id=node_id,
                    label=node_id,
                    kind=LEXEME_CANDIDATE_KIND,
                    embedding=as_tuple(embedding),
                    terms=(),
                    vault_id=f"lexical-geometry:{node_id}",
                    created_pulse=self.mind.pulse,
                    last_active_pulse=self.mind.pulse,
                )
            )
            created = True
        elif existing.kind not in LEXICAL_KINDS:
            raise RuntimeError(f"lexical geometry collides with {existing.kind}: {node_id}")
        else:
            created = False
        self.mind.recurrent.register(
            node_id,
            persistence=0.74,
            expression_threshold=2.0,
            pulse=self.mind.pulse,
        )
        return node_id, created

    def _observe_transition(
        self,
        source_id: str,
        target_id: str,
        *,
        side: GraphSide,
        record_id: str,
    ) -> tuple[str, bool, bool]:
        """Stage a novel sequence edge and awaken it only after repetition."""
        existing = self.mind.store.find_edge(side, source_id, target_id)
        edge = self.mind.graph.add_relation(
            source_id,
            target_id,
            side=side,
            pulse=self.mind.pulse,
            evidence_record_ids=(record_id,),
        )
        newly_staged = existing is None
        if newly_staged:
            self.mind.store.update_edge_state(edge.edge_id, archived=True)
        evidence_count = len(self.mind.store.edge_evidence_record_ids(edge.edge_id))
        promoted = bool(
            (newly_staged or (existing is not None and existing.archived))
            and evidence_count >= self.transition_promotion_exposures
        )
        if promoted:
            self.mind.store.update_edge_state(edge.edge_id, archived=False)
        return edge.edge_id, newly_staged, promoted

    def consolidate_ready_patterns(self) -> dict[str, tuple[str, ...]]:
        """Promote already-supported lexical nodes and sequence edges.

        Online observation normally promotes a pattern on the exposure that
        crosses its threshold.  This audit-safe pass matters after a restart
        under a deliberately lower nursery threshold: it derives maturity
        from the same independent record evidence and does not replay text or
        add synthetic exposures.
        """
        promoted_nodes = []
        for concept in self.mind.store.list_concepts(
            kind=LEXEME_CANDIDATE_KIND
        ):
            evidence_count = self.mind.store.vault_record_count(
                concept.vault_id or f"lexical-geometry:{concept.concept_id}"
            )
            if evidence_count < self.promotion_exposures:
                continue
            self.mind.store.set_concept_kind(concept.concept_id, "lexeme")
            promoted_nodes.append(concept.concept_id)

        promoted_edges = []
        hear_id = INPUT_NODE_IDS[InputTrunk.HEAR]
        for edge in self.mind.store.list_edges(include_archived=True):
            if not edge.archived:
                continue
            source = self.mind.store.get_concept(edge.source_id)
            target = self.mind.store.get_concept(edge.target_id)
            if source is None or target is None:
                continue
            source_lexical = source.kind in LEXICAL_KINDS
            target_lexical = target.kind in LEXICAL_KINDS
            if edge.side == GraphSide.INPUT:
                is_transition = (
                    edge.source_id == hear_id and target_lexical
                ) or (source_lexical and target_lexical)
            else:
                is_transition = (
                    source.kind == SPEECH_START_KIND and target_lexical
                ) or (source_lexical and target_lexical) or (
                    source_lexical and target.kind == SPEECH_STOP_KIND
                )
            if not is_transition:
                continue
            evidence_count = len(
                self.mind.store.edge_evidence_record_ids(edge.edge_id)
            )
            if evidence_count < self.transition_promotion_exposures:
                continue
            self.mind.store.update_edge_state(edge.edge_id, archived=False)
            promoted_edges.append(edge.edge_id)
        return {
            "nodes_promoted": tuple(promoted_nodes),
            "transition_edges_promoted": tuple(promoted_edges),
        }

    def observe(
        self,
        record: MemoryRecord,
        sensation: LexicalSensation,
        *,
        semantic_concept_ids: Sequence[str],
        source_id: str,
    ) -> LexicalAcquisition:
        """Deposit one utterance as weak lexical and temporal graph experience."""
        source_node = self.ensure_source(source_id)
        source_concept = self.mind.store.get_concept(source_node)
        if source_concept is not None and source_concept.vault_id:
            self.mind.store.add_to_vault(
                source_concept.vault_id, record.record_id, source_node
            )
        source_input = self.mind.graph.add_relation(
            INPUT_NODE_IDS[InputTrunk.HEAR],
            source_node,
            side=GraphSide.INPUT,
            pulse=self.mind.pulse,
            evidence_record_ids=(record.record_id,),
        )

        rows = (
            sensation.lexical_rows
            if self.maximum_units is None
            else sensation.lexical_rows[: self.maximum_units]
        )
        truncated = self.maximum_units is not None and len(sensation.lexical_rows) > len(rows)
        lexical_ids: list[str] = []
        created_nodes: list[str] = []
        promoted_nodes: list[str] = []
        for row in rows:
            node_id, created = self._ensure_lexical_node(row)
            lexical_ids.append(node_id)
            if created:
                created_nodes.append(node_id)
            concept = self.mind.store.get_concept(node_id)
            if concept is not None and concept.vault_id:
                self.mind.store.add_to_vault(
                    concept.vault_id, record.record_id, node_id
                )

        valid_semantics = []
        for concept_id in dict.fromkeys(str(item) for item in semantic_concept_ids):
            concept = self.mind.store.get_concept(concept_id)
            if (
                concept is not None
                and concept.kind not in {*LEXICAL_KINDS, SPEECH_START_KIND, SPEECH_STOP_KIND}
            ):
                valid_semantics.append(concept_id)
            if (
                self.maximum_semantic_contexts is not None
                and len(valid_semantics) >= self.maximum_semantic_contexts
            ):
                break

        source_edges = [source_input.edge_id]
        for concept_id in valid_semantics:
            edge = self.mind.graph.add_relation(
                source_node,
                concept_id,
                side=GraphSide.INPUT,
                pulse=self.mind.pulse,
                evidence_record_ids=(record.record_id,),
            )
            source_edges.append(edge.edge_id)

        semantic_edges: list[str] = []
        semantic_edge_alignment: list[tuple[str, float]] = []
        lexical_rows_by_id = {
            lexical_id: row for lexical_id, row in zip(lexical_ids, rows)
        }
        context_by_id = {
            context_id: self.mind.store.get_concept(context_id)
            for context_id in valid_semantics
        }
        alignments = {
            (context_id, lexical_id): cosine_similarity(
                row, context_by_id[context_id].embedding
            )
            for lexical_id, row in lexical_rows_by_id.items()
            for context_id in valid_semantics
            if context_by_id[context_id] is not None
        }
        selected_associations: dict[tuple[str, str], float] = {}
        for lexical_id in dict.fromkeys(lexical_ids):
            ranked_contexts = sorted(
                (
                    (alignments[(context_id, lexical_id)], context_id)
                    for context_id in valid_semantics
                    if (context_id, lexical_id) in alignments
                ),
                key=lambda item: (item[0], item[1]),
                reverse=True,
            )
            selected_contexts = []
            if ranked_contexts:
                best = ranked_contexts[0][0]
                selected_contexts.append(ranked_contexts[0])
                selected_contexts.extend(
                    item
                    for item in ranked_contexts[1 : self.semantic_contexts_per_unit]
                    if item[0] > 0.05 and item[0] >= best - 0.08
                )
            for alignment, context_id in selected_contexts:
                selected_associations[(context_id, lexical_id)] = max(
                    selected_associations.get((context_id, lexical_id), -1.0),
                    alignment,
                )

        # Word-first competition alone can strand a newly grown experiential
        # route.  Give each currently lived route one reciprocal best match;
        # repeated experience still determines whether that weak fiber survives.
        for context_id in valid_semantics:
            ranked_lexemes = sorted(
                (
                    (alignment, lexical_id)
                    for (candidate_context, lexical_id), alignment in alignments.items()
                    if candidate_context == context_id
                ),
                key=lambda item: (item[0], item[1]),
                reverse=True,
            )
            if ranked_lexemes:
                alignment, lexical_id = ranked_lexemes[0]
                selected_associations[(context_id, lexical_id)] = max(
                    selected_associations.get((context_id, lexical_id), -1.0),
                    alignment,
                )

        for (context_id, lexical_id), alignment in selected_associations.items():
            for side in GraphSide:
                edge = self.mind.graph.add_relation(
                    context_id,
                    lexical_id,
                    side=side,
                    pulse=self.mind.pulse,
                    evidence_record_ids=(record.record_id,),
                )
                semantic_edges.append(edge.edge_id)
                semantic_edge_alignment.append((edge.edge_id, alignment))

        for lexical_id in dict.fromkeys(lexical_ids):
            for side in GraphSide:
                edge = self.mind.graph.add_relation(
                    source_node,
                    lexical_id,
                    side=side,
                    pulse=self.mind.pulse,
                    evidence_record_ids=(record.record_id,),
                )
                source_edges.append(edge.edge_id)

        transition_edges: list[str] = []
        transition_edges_staged: list[str] = []
        transition_edges_promoted: list[str] = []
        if lexical_ids:
            start_id = _speech_boundary_id(self.mind, SPEECH_START_KIND)
            stop_id = _speech_boundary_id(self.mind, SPEECH_STOP_KIND)
            input_first = self._observe_transition(
                INPUT_NODE_IDS[InputTrunk.HEAR],
                lexical_ids[0],
                side=GraphSide.INPUT,
                record_id=record.record_id,
            )
            output_first = self._observe_transition(
                start_id,
                lexical_ids[0],
                side=GraphSide.OUTPUT,
                record_id=record.record_id,
            )
            for edge_id, staged, promoted in (input_first, output_first):
                transition_edges.append(edge_id)
                if staged:
                    transition_edges_staged.append(edge_id)
                if promoted:
                    transition_edges_promoted.append(edge_id)
            for source, target in zip(lexical_ids, lexical_ids[1:]):
                for side in GraphSide:
                    edge_id, staged, promoted = self._observe_transition(
                        source,
                        target,
                        side=side,
                        record_id=record.record_id,
                    )
                    transition_edges.append(edge_id)
                    if staged:
                        transition_edges_staged.append(edge_id)
                    if promoted:
                        transition_edges_promoted.append(edge_id)
            output_stop = self._observe_transition(
                lexical_ids[-1],
                stop_id,
                side=GraphSide.OUTPUT,
                record_id=record.record_id,
            )
            transition_edges.append(output_stop[0])
            if output_stop[1]:
                transition_edges_staged.append(output_stop[0])
            if output_stop[2]:
                transition_edges_promoted.append(output_stop[0])

        source_mean, source_confidence = self.source_preference(source_node)
        trusted_pull = source_mean * source_confidence
        exposure_quality = max(0.25, min(0.95, 0.58 + 0.30 * trusted_pull))
        for edge_id, alignment in semantic_edge_alignment:
            self.mind.graph.habituate_edges(
                (edge_id,),
                exposure_strength=0.08,
                evidence_quality=exposure_quality
                * (0.45 + 0.55 * max(0.0, min(1.0, alignment))),
            )
        self.mind.graph.habituate_edges(
            transition_edges,
            exposure_strength=0.12,
            evidence_quality=exposure_quality,
        )
        self.mind.graph.habituate_edges(
            source_edges,
            exposure_strength=0.07,
            evidence_quality=exposure_quality,
        )

        for node_id in dict.fromkeys(lexical_ids):
            concept = self.mind.store.get_concept(node_id)
            if concept is None or concept.kind != LEXEME_CANDIDATE_KIND:
                continue
            exposure_count = self.mind.store.vault_record_count(
                concept.vault_id or f"lexical-geometry:{node_id}"
            )
            if exposure_count >= self.promotion_exposures:
                self.mind.store.set_concept_kind(node_id, "lexeme")
                promoted_nodes.append(node_id)

        return LexicalAcquisition(
            record_id=record.record_id,
            source_node_id=source_node,
            lexical_node_ids=tuple(lexical_ids),
            observed_units=len(lexical_ids),
            candidate_nodes_created=tuple(dict.fromkeys(created_nodes)),
            nodes_promoted=tuple(dict.fromkeys(promoted_nodes)),
            semantic_edge_ids=tuple(dict.fromkeys(semantic_edges)),
            transition_edge_ids=tuple(dict.fromkeys(transition_edges)),
            transition_edges_staged=tuple(dict.fromkeys(transition_edges_staged)),
            transition_edges_dormant=tuple(
                edge_id
                for edge_id in dict.fromkeys(transition_edges)
                if (edge := self.mind.store.get_edge(edge_id)) is not None
                and edge.archived
            ),
            transition_edges_promoted=tuple(
                dict.fromkeys(transition_edges_promoted)
            ),
            source_edge_ids=tuple(dict.fromkeys(source_edges)),
            truncated=truncated,
        )

    def assess_social_return(
        self,
        cycle: ExperienceCycle,
        sensation: LexicalSensation,
        *,
        source_id: str,
    ) -> SocialReturnAssessment | None:
        """Infer a small social return from source identity and numeric relation."""
        source_node = self.ensure_source(source_id)
        expected_source = cycle.metadata.get("source_node_id")
        if expected_source and str(expected_source) != source_node:
            return None

        semantic_ids = tuple(
            str(item) for item in cycle.metadata.get("semantic_node_ids", ())
        )
        alignments: list[tuple[float, str]] = []
        for node_id in semantic_ids:
            concept = self.mind.store.get_concept(node_id)
            if concept is None or not any(concept.embedding):
                continue
            alignment = max(
                0.0,
                min(1.0, cosine_similarity(sensation.utterance_embedding, concept.embedding)),
            )
            alignments.append((alignment, node_id))
        semantic_alignment = max((item[0] for item in alignments), default=0.0)

        prior_lexemes = {
            str(item) for item in cycle.metadata.get("lexical_node_ids", ())
        }
        current_lexemes = {
            lexical_geometry_id(row) for row in sensation.lexical_rows
        }
        union = prior_lexemes | current_lexemes
        lexical_overlap = (
            len(prior_lexemes & current_lexemes) / len(union) if union else 0.0
        )

        valence_numerator = 0.0
        valence_denominator = 0.0
        for alignment, node_id in alignments:
            dynamics = self.mind.store.get_node_dynamics(node_id)
            if dynamics is None or alignment <= 0.0:
                continue
            valence_numerator += alignment * dynamics.valence
            valence_denominator += alignment
        content_valence = (
            valence_numerator / valence_denominator
            if valence_denominator > 0.0
            else 0.0
        )

        source_preference, source_confidence = self.source_preference(source_node)
        relation = 0.78 * semantic_alignment + 0.22 * lexical_overlap
        learned_source_pull = source_preference * source_confidence
        stability_delta = max(
            -0.20,
            min(
                0.30,
                0.18 * relation
                + 0.10 * learned_source_pull
                + 0.08 * content_valence,
            ),
        )
        evidence_quality = max(
            0.20,
            min(1.0, 0.25 + 0.55 * relation + 0.20 * source_confidence),
        )
        return SocialReturnAssessment(
            cycle_id=cycle.cycle_id,
            source_node_id=source_node,
            semantic_alignment=semantic_alignment,
            lexical_overlap=lexical_overlap,
            content_valence=content_valence,
            source_preference=source_preference,
            source_confidence=source_confidence,
            stability_delta=stability_delta,
            evidence_quality=evidence_quality,
        )

    def close_social_cycles(
        self,
        sensation: LexicalSensation,
        *,
        source_id: str,
    ) -> tuple[SocialReturnAssessment, ...]:
        """Close only the newest speech cycle addressed to the replying source."""
        assessments = []
        candidates = sorted(
            self.mind.open_experience_cycles(OutputTrunk.SPEAK),
            key=lambda item: (item.opened_pulse, item.cycle_id),
            reverse=True,
        )
        for cycle in candidates:
            assessment = self.assess_social_return(
                cycle, sensation, source_id=source_id
            )
            if assessment is None:
                continue
            status = (
                "social_support"
                if assessment.stability_delta > 0.03
                else "social_friction"
                if assessment.stability_delta < -0.03
                else "social_contact"
            )
            self.mind.record_cycle_return(
                cycle.cycle_id,
                "A source- and content-sensitive social return was observed.",
                input_trunk=InputTrunk.HEAR,
                status=status,
                stability_delta=assessment.stability_delta,
                verified=True,
                terminal=True,
                source_id="communication-channel",
                record_type=RecordType.RECEIPT,
                allow_growth=False,
                embedding=[0.0] * self.mind.embedder.dimension,
                evidence_quality=assessment.evidence_quality,
                metadata={
                    "no_context_runtime": True,
                    "message_content_copied": False,
                    "inferred_social_return": True,
                    "source_node_id": assessment.source_node_id,
                    "semantic_alignment": assessment.semantic_alignment,
                    "lexical_overlap": assessment.lexical_overlap,
                    "content_valence": assessment.content_valence,
                    "source_preference": assessment.source_preference,
                    "source_confidence": assessment.source_confidence,
                },
            )
            self.learn_source_preference(
                assessment.source_node_id,
                assessment.stability_delta,
                confidence=0.20 * assessment.evidence_quality,
            )
            assessments.append(assessment)
            break
        return tuple(assessments)

    def queue_social_cycles(
        self,
        sensation: LexicalSensation,
        *,
        source_id: str,
        pulse_kernel: SelfPulseKernel,
    ) -> tuple[tuple[SocialReturnAssessment, SensoryBucket], ...]:
        """Queue the newest matching social consequence for the next SELF pulse."""
        candidates = sorted(
            self.mind.open_experience_cycles(OutputTrunk.SPEAK),
            key=lambda item: (item.opened_pulse, item.cycle_id),
            reverse=True,
        )
        for cycle in candidates:
            assessment = self.assess_social_return(
                cycle, sensation, source_id=source_id
            )
            if assessment is None:
                continue
            status = (
                "social_support"
                if assessment.stability_delta > 0.03
                else "social_friction"
                if assessment.stability_delta < -0.03
                else "social_contact"
            )
            bucket = pulse_kernel.enqueue_cycle_return(
                cycle.cycle_id,
                "A source- and content-sensitive social return was observed.",
                lane=InputTrunk.HEAR,
                status=status,
                stability_delta=assessment.stability_delta,
                verified=True,
                terminal=True,
                source_id="communication-channel",
                evidence_quality=assessment.evidence_quality,
                embedding=[0.0] * self.mind.embedder.dimension,
                metadata={
                    "no_context_runtime": True,
                    "message_content_copied": False,
                    "inferred_social_return": True,
                    "source_node_id": assessment.source_node_id,
                    "semantic_alignment": assessment.semantic_alignment,
                    "lexical_overlap": assessment.lexical_overlap,
                    "content_valence": assessment.content_valence,
                    "source_preference": assessment.source_preference,
                    "source_confidence": assessment.source_confidence,
                },
            )
            self.learn_source_preference(
                assessment.source_node_id,
                assessment.stability_delta,
                confidence=0.20 * assessment.evidence_quality,
            )
            return ((assessment, bucket),)
        return ()


def _semantic_activation(
    mind: BaseAgenticMemoryRAG,
    recurrent: RecurrentSnapshot,
    focus: OpenWeightFocus,
) -> dict[str, float]:
    activation = mind.recurrent.activation_values(recurrent, minimum=0.0)
    selected = focus.selected
    if selected is not None:
        for depth, node_id in enumerate(selected.path_node_ids):
            route_pull = 0.30 + 0.50 * depth / max(1, len(selected.path_node_ids) - 1)
            activation[node_id] = max(activation.get(node_id, 0.0), route_pull)
        activation[selected.terminal_node_id] = max(
            activation.get(selected.terminal_node_id, 0.0),
            min(1.0, selected.transient_pull or 1.0),
        )
    return activation


def compete_lexical_step(
    mind: BaseAgenticMemoryRAG,
    recurrent: RecurrentSnapshot,
    focus: OpenWeightFocus,
    *,
    cursor_node_id: str,
    step_index: int,
    minimum_words: int,
) -> tuple[LexicalCandidate, tuple[LexicalCandidate, ...]]:
    """Recalculate one word competition from the live graph state.

    The cursor supplies learned local sequencing habits.  Independent active
    semantic branches supply convergent meaning support to the same lexical
    node.  No sentence, record text, or prior-token string is consulted.
    """
    output_edges = mind.store.list_edges(GraphSide.OUTPUT)
    outgoing: dict[str, list[GraphEdge]] = {}
    incoming: dict[str, list[GraphEdge]] = {}
    for edge in output_edges:
        outgoing.setdefault(edge.source_id, []).append(edge)
        incoming.setdefault(edge.target_id, []).append(edge)
    transition_edges = [
        edge
        for edge in outgoing.get(cursor_node_id, ())
        if (target := mind.store.get_concept(edge.target_id)) is not None
        and target.kind in {"lexeme", SPEECH_STOP_KIND}
    ]
    if not transition_edges:
        raise RuntimeError(f"speech cursor has no learned continuations: {cursor_node_id}")
    snapshot = mind.graph.weight_snapshot(side=GraphSide.OUTPUT)
    transition_local = _local_for_edges(transition_edges, snapshot)
    activation = _semantic_activation(mind, recurrent, focus)
    state_by_id = {state.node_id: state for state in recurrent.node_states}

    semantic_terms = []
    for node_id, value in activation.items():
        concept = mind.store.get_concept(node_id)
        if (
            concept is not None
            and concept.kind not in {
                *LEXICAL_KINDS,
                SPEECH_START_KIND,
                SPEECH_STOP_KIND,
                SOCIAL_SOURCE_KIND,
            }
            and any(concept.embedding)
            and value > 1e-6
        ):
            semantic_terms.append((concept.embedding, value))
    semantic_field = _field_or_self(mind, semantic_terms)

    raw: list[LexicalCandidate] = []
    for edge in transition_edges:
        target = mind.store.get_concept(edge.target_id)
        if target is None:
            continue
        stop = target.kind == SPEECH_STOP_KIND
        probability = max(1e-12, transition_local.get(edge.edge_id, 0.0))
        supports: list[tuple[float, str, str]] = []
        if not stop:
            for association in incoming.get(target.concept_id, ()):
                source = mind.store.get_concept(association.source_id)
                if source is None or source.kind in {
                    *LEXICAL_KINDS,
                    SPEECH_START_KIND,
                    SPEECH_STOP_KIND,
                }:
                    continue
                source_activation = max(0.0, activation.get(source.concept_id, 0.0))
                if source_activation <= 1e-6:
                    continue
                # Compare this word only with the source concept's other
                # lexical realizations. Structural children must not dilute
                # the amount of meaning that reaches its language membrane.
                lexical_siblings = [
                    sibling
                    for sibling in outgoing.get(source.concept_id, ())
                    if (sibling_target := mind.store.get_concept(sibling.target_id))
                    is not None
                    and sibling_target.kind == "lexeme"
                ]
                source_local = _local_for_edges(
                    lexical_siblings, snapshot
                ).get(association.edge_id, 0.0)
                contribution = source_activation * source_local
                if contribution > 1e-9:
                    supports.append(
                        (contribution, source.concept_id, association.edge_id)
                    )
        semantic_support = 0.0
        for contribution, _, _ in supports:
            semantic_support = 1.0 - (1.0 - semantic_support) * (
                1.0 - max(0.0, min(1.0, contribution))
            )
        state = state_by_id.get(target.concept_id)
        lexical_activation = state.activation if state is not None else 0.0
        alignment = (
            max(0.0, cosine_similarity(target.embedding, semantic_field))
            if not stop and any(target.embedding)
            else 0.0
        )
        boundary_pull = (
            1.0
            if not stop
            and step_index >= max(0, int(minimum_words))
            and any(
                (continuation_target := mind.store.get_concept(continuation.target_id))
                is not None
                and continuation_target.kind == SPEECH_STOP_KIND
                for continuation in outgoing.get(target.concept_id, ())
            )
            else 0.0
        )
        recency_penalty = 0.0
        if state is not None and state.last_selected_pulse is not None:
            age = max(0, recurrent.pulse - state.last_selected_pulse)
            recency_penalty = 1.60 / (1.0 + age)
        repetition_penalty = recency_penalty
        if stop:
            if step_index < max(0, int(minimum_words)):
                score = -1e9
            else:
                score = (
                    1.60 * math.log(probability)
                    - 0.55
                    + 0.18 * (step_index - minimum_words + 1)
                )
        else:
            # Final non-stop scores are normalized across this local
            # candidate set below.
            score = 0.0
        ordered_supports = sorted(supports, key=lambda item: (-item[0], item[1]))
        raw.append(
            LexicalCandidate(
                node_id=target.concept_id,
                transition_edge_id=edge.edge_id,
                transition_probability=probability,
                semantic_support=semantic_support,
                convergence_count=len(supports),
                lexical_activation=lexical_activation,
                geometry_alignment=alignment,
                boundary_pull=boundary_pull,
                repetition_penalty=repetition_penalty,
                score=score,
                stop=stop,
                supporting_node_ids=tuple(item[1] for item in ordered_supports),
                supporting_edge_ids=tuple(item[2] for item in ordered_supports),
            )
        )
    # A mature graph can divide absolute lexical mass among hundreds of
    # siblings. Competition is local, so preserve the relative evidence among
    # the currently reachable words instead of allowing graph size to erase
    # semantic differences.
    maximum_support = max(
        (item.semantic_support for item in raw if not item.stop), default=0.0
    )
    rescored = []
    for item in raw:
        if item.stop:
            rescored.append(item)
            continue
        relative_support = (
            item.semantic_support / maximum_support
            if maximum_support > 1e-12
            else 0.0
        )
        rescored.append(
            replace(
                item,
                score=(
                    1.35 * math.log(item.transition_probability)
                    + 3.20 * relative_support
                    + 0.25 * item.geometry_alignment
                    + 0.15 * item.lexical_activation
                    + 0.06 * max(0, item.convergence_count - 1)
                    + 0.50 * item.boundary_pull
                    - item.repetition_penalty
                ),
            )
        )
    raw = rescored
    probabilities = _softmax([item.score for item in raw], temperature=0.55)
    candidates = tuple(
        replace(item, probability=probability)
        for item, probability in zip(raw, probabilities)
    )
    selected = max(
        candidates,
        key=lambda item: (
            item.probability,
            item.semantic_support,
            item.transition_probability,
            item.node_id,
        ),
    )
    return selected, tuple(
        sorted(candidates, key=lambda item: (-item.probability, item.node_id))
    )


def build_native_state_frame(
    mind: BaseAgenticMemoryRAG,
    recurrent: RecurrentSnapshot,
    focus: OpenWeightFocus,
    *,
    input_sha256: str | None,
    lexical_rows: Sequence[Sequence[float]] = (),
    maximum_rows: int | None = None,
    overlay_strength: float = 0.08,
) -> NativeStateFrame | None:
    selected = focus.selected
    if selected is None or selected.trunk != OutputTrunk.SPEAK:
        return None
    snapshot = mind.graph.weight_snapshot(side=GraphSide.OUTPUT)
    edge_by_id = {edge.edge_id: edge for edge in mind.store.list_edges()}
    concept_by_id = {
        concept.concept_id: concept for concept in mind.store.list_concepts()
    }
    whole = _field_or_self(
        mind,
        (
            (concept_by_id[edge.target_id].embedding, mass)
            for edge_id, mass in snapshot.global_weights.items()
            if (edge := edge_by_id.get(edge_id)) is not None
            and edge.target_id in concept_by_id
            and concept_by_id[edge.target_id].kind not in {
                *LEXICAL_KINDS,
                SOCIAL_SOURCE_KIND,
            }
            and any(concept_by_id[edge.target_id].embedding)
        ),
    )
    active = _field_or_self(
        mind,
        (
            (concept.embedding, state.activation)
            for state in recurrent.node_states
            if state.activation > 1e-5
            and (concept := concept_by_id.get(state.node_id)) is not None
            and concept.kind not in {*LEXICAL_KINDS, SOCIAL_SOURCE_KIND}
            and any(concept.embedding)
        ),
    )
    desire = _field_or_self(
        mind,
        (
            (concept.embedding, item.urgency)
            for item in recurrent.desires
            if item.urgency > 1e-5
            and (concept := concept_by_id.get(item.node_id)) is not None
            and any(concept.embedding)
        ),
    )
    output = _field_or_self(
        mind,
        (
            (concept.embedding, 0.35 + depth / max(1, len(selected.path_node_ids) - 1))
            for depth, node_id in enumerate(selected.path_node_ids)
            if (concept := concept_by_id.get(node_id)) is not None
            and any(concept.embedding)
        ),
    )
    structural = (whole, active, desire, output)
    lexical_selection = (
        lexical_rows
        if maximum_rows is None
        else lexical_rows[: max(0, int(maximum_rows) - len(structural))]
    )
    lexemes = [_normalize(row) for row in lexical_selection]
    strength = max(0.0, min(1.0, float(overlay_strength)))
    contextual = []
    for index, lexeme in enumerate(lexemes):
        position = (index + 1) / (len(lexemes) + 1)
        context = _weighted_field(
            mind.embedder.dimension,
            (
                (whole, 0.10),
                (active, 0.25),
                (desire, 0.45),
                (output, 0.20 + 0.20 * position),
            ),
        )
        contextual.append(
            _weighted_field(
                mind.embedder.dimension,
                ((lexeme, 1.0 - strength), (context, strength)),
            )
        )
    rows = tuple((*structural, *contextual))
    if maximum_rows is not None:
        rows = rows[:maximum_rows]
    return NativeStateFrame(
        pulse_id=focus.pulse_id,
        rows=rows,
        row_kinds=(
            "whole_graph",
            "recurrent_activation",
            "desire_field",
            "selected_output",
            *("selected_lexeme" for _ in contextual),
        )[: len(rows)],
        state_sha256=recurrent.state_sha256,
        input_sha256=input_sha256,
        selected_path_node_ids=selected.path_node_ids,
        selected_path_edge_ids=selected.path_edge_ids,
    )


def compose_recurrent_speech(
    mind: BaseAgenticMemoryRAG,
    plan: OpenWeightTurnPlan,
    *,
    minimum_words: int = 3,
    maximum_words: int | None = None,
    pulse_kernel: SelfPulseKernel | None = None,
) -> RecurrentSpeech:
    """Actualize speech one graph decision and recurrent pulse per word."""
    selected_output = plan.focus.selected
    if selected_output is None or selected_output.trunk != OutputTrunk.SPEAK:
        raise ValueError("the selected open-weight route is not speech")
    if plan.frame is None:
        raise ValueError("the recurrent field is currently withholding speech")
    minimum_words = max(0, int(minimum_words))
    if maximum_words is not None:
        maximum_words = max(1, int(maximum_words))
    start_id = _speech_boundary_id(mind, SPEECH_START_KIND)
    start_edge = mind.store.find_edge(
        GraphSide.OUTPUT, selected_output.terminal_node_id, start_id
    )
    if start_edge is None or start_edge.archived:
        raise RuntimeError("selected desire has no route into lexical competition")

    recurrent = plan.recurrent
    cursor = start_id
    rows: list[tuple[float, ...]] = []
    lexical_ids: list[str] = []
    steps: list[LexicalStep] = []
    path_nodes = [*selected_output.path_node_ids, start_id]
    path_edges = [*selected_output.path_edge_ids, start_edge.edge_id]
    causal_edges = [*path_edges]
    stopped = False

    for _ in (itertools.count() if maximum_words is None else range(maximum_words)):
        index = len(rows)
        try:
            winner, candidates = compete_lexical_step(
                mind,
                recurrent,
                plan.focus,
                cursor_node_id=cursor,
                step_index=index,
                minimum_words=minimum_words,
            )
        except RuntimeError as error:
            # Online development can promote a useful word before any one of
            # its outgoing adjacencies has enough independent evidence to be
            # productive.  Once at least one word has been actualized, branch
            # exhaustion is a legitimate motor stop rather than a reason to
            # erase the whole utterance.
            if rows and str(error).startswith(
                "speech cursor has no learned continuations:"
            ):
                stopped = True
                break
            raise
        state_before = recurrent.state_sha256
        excitation: dict[str, float] = {}
        for state in recurrent.node_states:
            concept = mind.store.get_concept(state.node_id)
            if concept is not None and concept.kind in LEXICAL_KINDS:
                excitation[state.node_id] = -0.72
        excitation[winner.node_id] = 1.0
        if pulse_kernel is None:
            pulse, pulse_id = mind._next_pulse()
            recurrent = mind.recurrent.advance(
                pulse=pulse,
                pulse_id=f"lexical-actualization:{pulse_id}",
                excitation=excitation,
            )
            mind.recurrent.mark_selected((winner.node_id,), pulse=pulse)
            recurrent = mind.recurrent.snapshot(pulse=pulse)
        else:
            recurrent, pulse_id = pulse_kernel.advance_internal_output(
                excitation=excitation,
                selected_node_ids=(winner.node_id,),
                kind="lexical_actualization",
                metadata={
                    "step_index": index,
                    "cursor_node_id": cursor,
                    "selected_node_id": winner.node_id,
                },
            )
        steps.append(
            LexicalStep(
                pulse_id=pulse_id,
                index=index,
                cursor_node_id=cursor,
                candidates=candidates,
                selected=winner,
                state_before_sha256=state_before,
                state_after_sha256=recurrent.state_sha256,
            )
        )
        path_nodes.append(winner.node_id)
        path_edges.append(winner.transition_edge_id)
        causal_edges.extend((winner.transition_edge_id, *winner.supporting_edge_ids))
        if winner.stop:
            stopped = True
            break
        concept = mind.store.get_concept(winner.node_id)
        if concept is None or concept.kind != "lexeme" or not any(concept.embedding):
            raise RuntimeError("lexical competition selected a nonproductive node")
        rows.append(_normalize(concept.embedding))
        lexical_ids.append(winner.node_id)
        cursor = winner.node_id

    input_sha256 = plan.frame.input_sha256
    frame = build_native_state_frame(
        mind,
        recurrent,
        plan.focus,
        input_sha256=input_sha256,
        lexical_rows=rows,
    )
    if frame is None:
        raise RuntimeError("speech composition lost its external output route")
    return RecurrentSpeech(
        pulse_id=plan.pulse_id,
        steps=tuple(steps),
        lexical_node_ids=tuple(lexical_ids),
        lexical_rows=tuple(rows),
        path_node_ids=tuple(path_nodes),
        path_edge_ids=tuple(path_edges),
        causal_edge_ids=tuple(dict.fromkeys(causal_edges)),
        stopped=stopped,
        final_recurrent=recurrent,
        frame=frame,
    )


def write_native_state_packet(path: str | Path, frame: NativeStateFrame) -> Path:
    """Write only floating-point state; no text or node IDs cross this boundary."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not frame.rows:
        raise ValueError("native state frame is empty")
    dimension = len(frame.rows[0])
    if any(len(row) != dimension for row in frame.rows):
        raise ValueError("native state rows have inconsistent dimensions")
    with destination.open("w", encoding="ascii") as output:
        output.write("HABITUS_RECURRENT_PACKET_V1\n")
        output.write(f"{dimension} {len(frame.rows)}\n")
        for row in frame.rows:
            output.write(" ".join(f"{value:.9g}" for value in row))
            output.write("\n")
    return destination


class NoContextPlanner:
    """Current-input-to-action planner with no recall or transcript window."""

    def __init__(
        self,
        mind: BaseAgenticMemoryRAG,
        *,
        maximum_input_concepts: int | None = None,
        lexical_sensor: Callable[[str], Sequence[Sequence[float]]] | None = None,
        language_learner: DevelopmentalLanguage | None = None,
        pulse_kernel: SelfPulseKernel | None = None,
        grow_experiential_breadth: bool | None = None,
    ):
        self.mind = mind
        self.maximum_input_concepts = (
            None
            if maximum_input_concepts is None
            else max(1, int(maximum_input_concepts))
        )
        self.lexical_sensor = lexical_sensor
        self.language_learner = language_learner
        self.pulse_kernel = pulse_kernel or SelfPulseKernel(mind)
        self.grow_experiential_breadth = (
            language_learner is not None
            if grow_experiential_breadth is None
            else bool(grow_experiential_breadth)
        )
        self.mind.store.set_metadata(
            "trunk_breadth_growth",
            "enabled" if self.grow_experiential_breadth else "disabled",
        )
        if self.grow_experiential_breadth:
            self.mind.store.set_metadata(
                "trunk_breadth_schema", "habitus.trunk-breadth.v2"
            )

    def _input_candidates(
        self,
        vector: Sequence[float],
        *,
        input_trunk: InputTrunk = InputTrunk.HEAR,
    ) -> tuple[tuple[float, str, TraversalTrace], ...]:
        # Semantic X nomination is cheap; Y traversal is not. Rank the whole
        # receptive crown first, then prove reachability for only a bounded
        # shortlist. This keeps input latency tied to the selected routes
        # rather than to the total number of concepts in a mature mind.
        ranked = []
        for concept in self.mind.store.list_concepts():
            if (
                concept.kind
                not in {"crown", *DESIRE_KINDS, *EXPERIENTIAL_CONTEXT_KINDS}
                or not any(concept.embedding)
            ):
                continue
            score = cosine_similarity(vector, concept.embedding)
            ranked.append((score, concept.concept_id))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)

        probe_limit = (
            len(ranked)
            if self.maximum_input_concepts is None
            else max(self.maximum_input_concepts * 4, self.maximum_input_concepts)
        )
        shortlist = {
            node_id: score for score, node_id in ranked[:probe_limit]
        }
        traces = self.mind.graph.traverse_many(
            pulse_id=f"no-context-probe:{self.mind.pulse}",
            side=GraphSide.INPUT,
            targets=shortlist,
            required_input_trunk=input_trunk,
            mark_active=False,
        )
        by_target = {trace.target_node_id: trace for trace in traces}
        result = tuple(
            (score, node_id, by_target[node_id])
            for score, node_id in ranked[:probe_limit]
            if node_id in by_target
        )
        return (
            result
            if self.maximum_input_concepts is None
            else result[: self.maximum_input_concepts]
        )

    def _kernel_output_candidates(
        self, focus: OpenWeightFocus
    ) -> tuple[OutputCandidate, ...]:
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
        return tuple(result)

    @staticmethod
    def _focus_for_cycle(
        focus: OpenWeightFocus,
        cycle: CognitiveCycleReceipt,
    ) -> OpenWeightFocus:
        if cycle.selected_output is None:
            return replace(focus, selected=None)
        selected_trajectory = next(
            (
                candidate
                for candidate in focus.candidates
                if candidate.trunk == cycle.selected_output.trunk
                and candidate.terminal_node_id == cycle.selected_output.node_id
                and candidate.path_edge_ids == cycle.selected_output.path_edge_ids
            ),
            None,
        )
        if selected_trajectory is None:
            raise RuntimeError("self pulse selected an unknown output trajectory")
        return replace(focus, selected=selected_trajectory)

    def sense(self, text: str) -> LexicalSensation:
        content = str(text).strip()
        if not content:
            raise ValueError("current utterance cannot be empty")
        vector = self.mind.embedder.embed(content)
        lexical_rows = (
            tuple(
                tuple(float(value) for value in row)
                for row in self.lexical_sensor(content)
            )
            if self.lexical_sensor is not None
            else ()
        )
        if len(vector) != self.mind.embedder.dimension:
            raise ValueError("utterance embedding dimension mismatch")
        if any(len(row) != self.mind.embedder.dimension for row in lexical_rows):
            raise ValueError("lexical sensor dimension mismatch")
        return LexicalSensation(
            input_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            utterance_embedding=as_tuple(vector),
            lexical_rows=lexical_rows,
        )

    def hear(
        self,
        text: str,
        *,
        source_id: str = "human",
        sensation: LexicalSensation | None = None,
        additional_input_item_ids: Sequence[str] = (),
        input_item_id: str | None = None,
        input_metadata: Mapping[str, Any] | None = None,
    ) -> OpenWeightTurnPlan:
        content = str(text).strip()
        if not content:
            raise ValueError("current utterance cannot be empty")
        sensation = sensation or self.sense(content)
        expected_digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if sensation.input_sha256 != expected_digest:
            raise ValueError("lexical sensation does not belong to the current utterance")
        vector = sensation.utterance_embedding
        selected = self._input_candidates(vector)
        concept_ids = tuple(item[1] for item in selected)
        bucket = self.pulse_kernel.enqueue_input(
            content,
            lane=InputTrunk.HEAR,
            source_id=source_id,
            concept_ids=concept_ids,
            concept_scores={
                concept_id: max(0.0, min(1.0, score))
                for score, concept_id, _ in selected
            },
            embedding=vector,
            allow_growth=False,
            metadata={
                **dict(input_metadata or {}),
                "no_context_runtime": True,
                "retrieval_used": False,
                "transcript_window_used": False,
                "online_lexical_units": len(sensation.lexical_rows),
            },
            item_id=input_item_id,
        )
        acquisition: LexicalAcquisition | None = None
        developmental_growth: list[DevelopmentalGrowth] = []

        def observe_admission(
            record: MemoryRecord,
            admitted_bucket: SensoryBucket,
            created: bool,
        ) -> AdmissionEffect:
            nonlocal acquisition
            if not created:
                return AdmissionEffect({})

            growth_receipt = None
            if self.grow_experiential_breadth and admitted_bucket.kind == "input":
                growth_receipt = self.mind.graph.grow_trunk_breadth(
                    record,
                    input_trunk=admitted_bucket.lane,
                    pulse=self.mind.pulse,
                    sensory_rows=(
                        sensation.lexical_rows
                        if admitted_bucket.item_id == bucket.item_id
                        else ()
                    ),
                )
                developmental_growth.append(growth_receipt)

            excitation: dict[str, float] = {}
            context_ids: list[str] = []
            if growth_receipt is not None:
                for node_id in growth_receipt.feature_node_ids:
                    excitation[node_id] = 0.32
                for node_id in growth_receipt.candidate_pattern_node_ids:
                    excitation[node_id] = 0.16
                for node_id in growth_receipt.promoted_pattern_node_ids:
                    excitation[node_id] = 0.52
                context_ids.extend(growth_receipt.active_context_node_ids)

            if admitted_bucket.item_id != bucket.item_id:
                return AdmissionEffect(
                    excitation,
                    context_node_ids=tuple(dict.fromkeys(context_ids)),
                )

            convergence = (
                self.mind.graph.grow_cross_trunk_breadth(
                    tuple(developmental_growth),
                    anchor_record=record,
                    pulse=self.mind.pulse,
                )
                if self.grow_experiential_breadth
                else None
            )
            if convergence is not None:
                developmental_growth.append(convergence)
                for node_id in convergence.candidate_pattern_node_ids:
                    excitation[node_id] = 0.20
                for node_id in convergence.promoted_pattern_node_ids:
                    excitation[node_id] = 0.62
                context_ids.extend(convergence.active_context_node_ids)

            if self.language_learner is not None:
                experiential_contexts: list[str] = []
                if convergence is not None:
                    experiential_contexts.extend(
                        convergence.active_context_node_ids
                    )
                for receipt in reversed(developmental_growth):
                    if receipt.cross_trunk:
                        continue
                    experiential_contexts.extend(receipt.promoted_pattern_node_ids)
                    experiential_contexts.extend(receipt.candidate_pattern_node_ids)
                    experiential_contexts.extend(receipt.feature_node_ids)
                acquisition = self.language_learner.observe(
                    record,
                    sensation,
                    semantic_concept_ids=tuple(
                        dict.fromkeys((*experiential_contexts, *concept_ids))
                    ),
                    source_id=source_id,
                )
                source_preference, source_confidence = (
                    self.language_learner.source_preference(
                        acquisition.source_node_id
                    )
                )
                source_pull = max(
                    0.20,
                    min(
                        0.80,
                        0.48 + 0.24 * source_preference * source_confidence,
                    ),
                )
                excitation[acquisition.source_node_id] = source_pull
                context_ids.append(acquisition.source_node_id)
                for lexical_id in dict.fromkeys(acquisition.lexical_node_ids):
                    excitation[lexical_id] = 0.36
            return AdmissionEffect(
                excitation,
                context_node_ids=tuple(dict.fromkeys(context_ids)),
            )

        resolved_focus: OpenWeightFocus | None = None

        def provide_outputs(
            recurrent: RecurrentSnapshot,
            pulse_id: str,
        ) -> tuple[OutputCandidate, ...]:
            nonlocal resolved_focus
            resolved_focus = resolve_output_focus(
                self.mind, recurrent, pulse_id=pulse_id, mark_active=False
            )
            return self._kernel_output_candidates(resolved_focus)

        frame_item_ids = tuple(
            dict.fromkeys((*additional_input_item_ids, bucket.item_id))
        )
        cycle = self.pulse_kernel.advance_cycle(
            maximum_per_sensory_lane=len(frame_item_ids),
            admission_observer=observe_admission,
            item_ids=frame_item_ids,
            output_candidate_provider=provide_outputs,
        )
        if cycle is None:
            raise RuntimeError("the current utterance did not produce a self pulse")
        pulse_id = cycle.self_state.pulse_id
        recurrent = cycle.recurrent
        input_receipt = next(
            item for item in cycle.sensory.receipts if item.item_id == bucket.item_id
        )
        record = self.mind.store.get_record(input_receipt.record_id)
        if record is None:
            raise RuntimeError("self pulse committed without its canonical input")
        traces = tuple(
            trace
            for receipt in cycle.sensory.receipts
            for trace in receipt.inward_traces
        )
        if resolved_focus is None:
            raise RuntimeError("self pulse did not evaluate output trajectories")
        focus = self._focus_for_cycle(resolved_focus, cycle)
        frame = build_native_state_frame(
            self.mind,
            recurrent,
            focus,
            input_sha256=sensation.input_sha256,
        )
        return OpenWeightTurnPlan(
            pulse_id=pulse_id,
            input_record=record,
            input_concept_ids=concept_ids,
            input_traces=traces,
            recurrent=recurrent,
            focus=focus,
            frame=frame,
            sensation=sensation,
            acquisition=acquisition,
            source_node_id=(
                acquisition.source_node_id if acquisition is not None else None
            ),
            self_pulse=cycle,
            output_authorization_id=(
                cycle.selected_output.authorization_id
                if cycle.selected_output is not None
                else None
            ),
            developmental_growth=tuple(developmental_growth),
        )

    def tick(self) -> OpenWeightTurnPlan:
        resolved_focus: OpenWeightFocus | None = None

        def provide_outputs(
            recurrent: RecurrentSnapshot,
            pulse_id: str,
        ) -> tuple[OutputCandidate, ...]:
            nonlocal resolved_focus
            resolved_focus = resolve_output_focus(
                self.mind, recurrent, pulse_id=pulse_id, mark_active=False
            )
            return self._kernel_output_candidates(resolved_focus)

        cycle = self.pulse_kernel.advance_cycle(
            advance_when_idle=True,
            cycle_id=f"no-context-internal:{self.mind.pulse + 1}",
            output_candidate_provider=provide_outputs,
        )
        if cycle is None:
            raise RuntimeError("an internal tick did not produce a self pulse")
        pulse_id = cycle.self_state.pulse_id
        recurrent = cycle.recurrent
        if resolved_focus is None:
            raise RuntimeError("self pulse did not evaluate output trajectories")
        focus = self._focus_for_cycle(resolved_focus, cycle)
        frame = (
            build_native_state_frame(
                self.mind, recurrent, focus, input_sha256=None
            )
            if recurrent.should_express
            else None
        )
        return OpenWeightTurnPlan(
            pulse_id=pulse_id,
            input_record=None,
            input_concept_ids=(),
            input_traces=(),
            recurrent=recurrent,
            focus=focus,
            frame=frame,
            self_pulse=cycle,
            output_authorization_id=(
                cycle.selected_output.authorization_id
                if cycle.selected_output is not None
                else None
            ),
        )

    def compose_speech(
        self,
        plan: OpenWeightTurnPlan,
        *,
        minimum_words: int = 3,
        maximum_words: int | None = None,
    ) -> RecurrentSpeech:
        return compose_recurrent_speech(
            self.mind,
            plan,
            minimum_words=minimum_words,
            maximum_words=maximum_words,
            pulse_kernel=self.pulse_kernel,
        )

    def begin_speech_cycle(
        self,
        plan: OpenWeightTurnPlan,
        response: str,
        *,
        speech: RecurrentSpeech | None = None,
    ):
        selected = plan.focus.selected
        if selected is None or selected.trunk != OutputTrunk.SPEAK:
            raise ValueError("the selected open-weight route is not speech")
        authorization_id = plan.output_authorization_id
        if not authorization_id:
            raise ValueError("speech was not authorized by a self pulse")
        if plan.self_pulse is None:
            raise ValueError("speech plan has no self-pulse receipt")
        affordance = next(
            (
                item
                for item in plan.self_pulse.selected_outputs
                if item.authorization_id == authorization_id
            ),
            None,
        )
        if affordance is None:
            raise ValueError("speech authorization is absent from its pulse receipt")
        path_node_ids = speech.path_node_ids if speech else selected.path_node_ids
        path_edge_ids = speech.path_edge_ids if speech else selected.path_edge_ids
        path_edges = [self.mind.store.get_edge(edge_id) for edge_id in path_edge_ids]
        trace = TraversalTrace(
            trace_id=f"trace:{plan.pulse_id}:speech:{selected.terminal_node_id}",
            side=GraphSide.OUTPUT,
            start_node_id=SELF_ID,
            target_node_id=path_node_ids[-1],
            path_node_ids=path_node_ids,
            path_edge_ids=path_edge_ids,
            total_travel_time=sum(
                edge.delta_y for edge in path_edges if edge is not None
            ),
            endpoint_score=selected.effective_probability,
        )
        semantic_node_ids = list(plan.input_concept_ids)
        for node_id in selected.path_node_ids:
            concept = self.mind.store.get_concept(node_id)
            if (
                concept is not None
                and concept.kind not in {
                    "self",
                    "input_trunk",
                    "output_trunk",
                    *LEXICAL_KINDS,
                    SPEECH_START_KIND,
                    SPEECH_STOP_KIND,
                    SOCIAL_SOURCE_KIND,
                }
            ):
                semantic_node_ids.append(node_id)
        if speech is not None:
            for step in speech.steps:
                semantic_node_ids.extend(step.selected.supporting_node_ids)
        return self.pulse_kernel.actualize_output(
            affordance,
            str(response).strip(),
            trace=trace,
            metadata={
                "no_context_runtime": True,
                "state_sha256": (
                    speech.final_recurrent.state_sha256
                    if speech
                    else plan.recurrent.state_sha256
                ),
                "transcript_records_used": 0,
                "recalled_records_used": 0,
                "word_by_word_competition": speech is not None,
                "lexical_steps": len(speech.steps) if speech else 0,
                "source_node_id": plan.source_node_id,
                "semantic_node_ids": list(dict.fromkeys(semantic_node_ids)),
                "lexical_node_ids": (
                    list(speech.lexical_node_ids) if speech else []
                ),
            },
            credited_edge_ids=speech.causal_edge_ids if speech else None,
            embedding=(
                _field_or_self(
                    self.mind,
                    ((row, 1.0) for row in speech.lexical_rows),
                )
                if speech is not None and speech.lexical_rows
                else None
            ),
        )
