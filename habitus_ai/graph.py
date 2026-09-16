from __future__ import annotations

import hashlib
import heapq
import math
import time
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .embeddings import (
    Embedder,
    cosine_similarity,
    opaque_payload_embedding,
    tokenize,
)
from .store import MindStore
from .types import (
    ConceptNode,
    DevelopmentalGrowth,
    EventEnvelope,
    EventKind,
    ExperienceProjection,
    GraphEdge,
    GraphSide,
    InputTrunk,
    MemoryRecord,
    OutputTrunk,
    OverlapCluster,
    RecordType,
    TraversalTrace,
    as_tuple,
)


SELF_ID = "SELF"
INPUT_NODE_IDS = {trunk: f"IN:{trunk.value}" for trunk in InputTrunk}
OUTPUT_NODE_IDS = {trunk: f"OUT:{trunk.value}" for trunk in OutputTrunk}
PREFERENCE_BANDS = ("STABLE", "NEUTRAL", "UNSTABLE")
PREFERENCE_NODE_IDS = {
    (trunk, band): f"PREF:{trunk.value}:{band}"
    for trunk in InputTrunk
    for band in PREFERENCE_BANDS
}

SENSORY_FEATURE_KIND = "sensory_feature"
EXPERIENTIAL_CANDIDATE_KIND = "experiential_candidate"
EXPERIENTIAL_PATTERN_KIND = "experiential_pattern"
EXPERIENTIAL_CONTEXT_KINDS = frozenset(
    {SENSORY_FEATURE_KIND, EXPERIENTIAL_PATTERN_KIND}
)


@dataclass(frozen=True)
class WeightSnapshot:
    """One bounded root-originating flow over the current edge logits.

    ``local_weights`` are conditional sibling probabilities. ``global_weights``
    are the effective masses that actually reached each edge during the bounded
    flow. Their sum can exceed one across multiple depths; conservation applies
    at every frontier and at the root regions, not across all depths at once.
    """

    global_weights: Mapping[str, float]
    effective_logits: Mapping[str, float]
    local_weights: Mapping[str, float]
    node_weights: Mapping[str, float]
    regional_weights: Mapping[str, float]
    layer_weights: Mapping[str, float]
    side_starting_mass: Mapping[str, float]
    starting_node_ids: Mapping[str, str]
    absorbed_mass: float
    truncated_mass: float

    @property
    def total(self) -> float:
        """Conserved mass entering the selected cipher root or roots."""
        return sum(self.side_starting_mass.values())

    @property
    def cumulative_edge_mass(self) -> float:
        return sum(self.global_weights.values())

    @property
    def accounted_mass(self) -> float:
        return self.absorbed_mass + self.truncated_mass


class GraphRuntime:
    """Shared semantic crown plus directional Y-axis traversal."""

    def _weight_components(
        self,
        edges: Sequence[GraphEdge],
        *,
        now: float,
    ) -> tuple[dict[str, float], dict[str, float], dict[tuple[GraphSide, str], list[GraphEdge]]]:
        """Calculate the exact sibling distributions without propagating root mass.

        Route search, recurrent activation, and output arbitration depend on
        these conditional probabilities only.  Computing an entire bounded
        root-flow for those callers was both redundant and pathological once
        the developmental input graph acquired recurrent components.
        """
        logits: dict[str, float] = {}
        for edge in edges:
            recency = 0.0
            if edge.last_active_time is not None:
                age = max(0.0, now - edge.last_active_time)
                recency = self.recency_strength * math.exp(
                    -math.log(2.0) * age / self.recency_half_life_seconds
                )
            logits[edge.edge_id] = edge.log_strength + recency - edge.conflict_penalty
        grouped: dict[tuple[GraphSide, str], list[GraphEdge]] = {}
        for edge in edges:
            grouped.setdefault((edge.side, edge.source_id), []).append(edge)
        local_weights: dict[str, float] = {}
        for siblings in grouped.values():
            maximum = max(logits[edge.edge_id] for edge in siblings)
            exponentials = {
                edge.edge_id: math.exp(
                    (logits[edge.edge_id] - maximum) / self.temperature
                )
                for edge in siblings
            }
            total = sum(exponentials.values()) or 1.0
            local_weights.update(
                {edge_id: value / total for edge_id, value in exponentials.items()}
            )
        return logits, local_weights, grouped

    def edge_local_weights(
        self,
        *,
        now: float | None = None,
        side: GraphSide | None = None,
    ) -> dict[str, float]:
        """Return every exact conditional edge probability for a side."""
        current = time.time() if now is None else float(now)
        edges = self.store.list_edges(side)
        _, local_weights, _ = self._weight_components(edges, now=current)
        return local_weights

    def __init__(
        self,
        store: MindStore,
        embedder: Embedder,
        *,
        recency_strength: float = 0.8,
        recency_half_life_seconds: float = 300.0,
        temperature: float = 1.0,
        learning_rate: float = 0.35,
        growth_overlap_threshold: float = 0.70,
        growth_promotion_count: int = 2,
        growth_preference_tolerance: float = 0.35,
    ):
        self.store = store
        self.embedder = embedder
        self.recency_strength = max(0.0, float(recency_strength))
        self.recency_half_life_seconds = max(1.0, float(recency_half_life_seconds))
        self.temperature = max(0.05, float(temperature))
        self.learning_rate = max(0.0, float(learning_rate))
        self.growth_overlap_threshold = max(0.0, min(1.0, growth_overlap_threshold))
        self.growth_promotion_count = max(2, int(growth_promotion_count))
        self.growth_preference_tolerance = max(0.0, min(2.0, growth_preference_tolerance))
        self._breadth_plane_cache: dict[
            tuple[int, int, int],
            tuple[tuple[tuple[tuple[int, float], ...], ...], ...],
        ] = {}
        self.seed_topology()

    # --------------------------------------------------------------- seed/trunks

    def _seed_node(
        self,
        node_id: str,
        label: str,
        kind: str,
        terms: Sequence[str],
        *,
        vault_id: str | None = None,
        semantic_embedding: bool = True,
    ) -> None:
        text = " ".join((label, *terms))
        concept = self.store.add_concept(
            ConceptNode(
                concept_id=node_id,
                label=label,
                kind=kind,
                embedding=as_tuple(
                    self.embedder.embed(text)
                    if semantic_embedding
                    else [0.0] * self.embedder.dimension
                ),
                terms=tuple(dict.fromkeys(term.casefold() for term in terms)),
                vault_id=vault_id,
                created_pulse=0,
                last_active_pulse=0,
            )
        )
        if vault_id and concept.vault_id != vault_id:
            self.store.set_concept_vault(node_id, vault_id)

    @staticmethod
    def edge_id(side: GraphSide, source_id: str, target_id: str) -> str:
        material = f"{side.value}|{source_id}|{target_id}"
        digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]
        return f"edge:{side.value}:{digest}"

    def _ensure_edge(
        self,
        side: GraphSide,
        source_id: str,
        target_id: str,
        *,
        delta_y: float,
        log_strength: float = 0.0,
        created_pulse: int = 0,
    ) -> GraphEdge:
        existing = self.store.find_edge(side, source_id, target_id)
        if existing:
            return existing
        return self.store.add_edge(
            GraphEdge(
                edge_id=self.edge_id(side, source_id, target_id),
                side=side,
                source_id=source_id,
                target_id=target_id,
                delta_y=max(0.01, float(delta_y)),
                log_strength=float(log_strength),
                conflict_penalty=0.0,
                last_active_time=None,
                created_pulse=int(created_pulse),
            )
        )

    def seed_topology(self) -> None:
        self._seed_node(
            SELF_ID,
            "Self",
            "self",
            ("self", "origin"),
            vault_id="lower-vault:SELF",
        )
        input_terms = {
            InputTrunk.HEAR: ("message", "conversation", "said", "asked", "reply"),
            InputTrunk.SEE: ("observed", "read", "result", "found", "opened"),
            InputTrunk.NOTICE: ("notification", "alert", "completed", "queued", "later"),
        }
        output_terms = {
            OutputTrunk.SPEAK: ("reply", "tell", "say", "message", "explain", "answer"),
            OutputTrunk.LOOK: ("look", "read", "open", "search", "inspect", "find", "list"),
            OutputTrunk.DO: ("run", "execute", "write", "edit", "create", "change", "submit"),
        }
        for trunk, node_id in INPUT_NODE_IDS.items():
            self._seed_node(
                node_id,
                trunk.value.title(),
                "input_trunk",
                input_terms[trunk],
                vault_id=f"lower-vault:{node_id}",
            )
            self._ensure_edge(GraphSide.INPUT, SELF_ID, node_id, delta_y=1.0)
            for band in PREFERENCE_BANDS:
                preference_id = PREFERENCE_NODE_IDS[(trunk, band)]
                self._seed_node(
                    preference_id,
                    preference_id,
                    "lower_preference",
                    (),
                    vault_id=f"lower-vault:{preference_id}",
                    semantic_embedding=False,
                )
                self._ensure_edge(
                    GraphSide.INPUT,
                    node_id,
                    preference_id,
                    delta_y=1.0,
                )
        for trunk, node_id in OUTPUT_NODE_IDS.items():
            self._seed_node(
                node_id,
                trunk.value.title(),
                "output_trunk",
                output_terms[trunk],
                vault_id=f"lower-vault:{node_id}",
            )
            self._ensure_edge(GraphSide.OUTPUT, SELF_ID, node_id, delta_y=1.0)

    @staticmethod
    def route_event(event: EventEnvelope) -> InputTrunk:
        if event.kind == EventKind.NOTIFICATION:
            return InputTrunk.NOTICE
        if event.kind == EventKind.OBSERVATION:
            return InputTrunk.SEE if event.correlation_id else InputTrunk.NOTICE
        return InputTrunk.HEAR

    # ------------------------------------------------------------ concepts/edges

    def add_concept(
        self,
        concept_id: str,
        label: str,
        *,
        terms: Sequence[str] = (),
        embedding: Sequence[float] | None = None,
        input_trunks: Sequence[InputTrunk] = (),
        output_trunks: Sequence[OutputTrunk] = (),
        pulse: int = 0,
        evidence_record_ids: Sequence[str] = (),
        kind: str = "crown",
        semantic_embedding: bool = True,
    ) -> ConceptNode:
        if concept_id == SELF_ID or concept_id.startswith(("IN:", "OUT:")):
            raise ValueError("concept ID is reserved by the seed topology")
        vector = (
            list(embedding)
            if embedding is not None
            else self.embedder.embed(" ".join((label, *terms)))
            if semantic_embedding
            else [0.0] * self.embedder.dimension
        )
        if len(vector) != self.embedder.dimension:
            raise ValueError("concept embedding dimension mismatch")
        vault_id = f"vault:{concept_id}"
        concept = self.store.add_concept(
            ConceptNode(
                concept_id=concept_id,
                label=label,
                kind=kind,
                embedding=as_tuple(vector),
                terms=tuple(dict.fromkeys(tokenize(" ".join((label, *terms))))),
                vault_id=vault_id,
                created_pulse=pulse,
                last_active_pulse=pulse,
            )
        )
        for trunk in input_trunks:
            edge = self._ensure_edge(
                GraphSide.INPUT,
                INPUT_NODE_IDS[trunk],
                concept_id,
                delta_y=2.0,
                created_pulse=pulse,
            )
            for record_id in evidence_record_ids:
                self.store.add_edge_evidence(edge.edge_id, record_id)
        for trunk in output_trunks:
            edge = self._ensure_edge(
                GraphSide.OUTPUT,
                OUTPUT_NODE_IDS[trunk],
                concept_id,
                delta_y=2.0,
                created_pulse=pulse,
            )
            for record_id in evidence_record_ids:
                self.store.add_edge_evidence(edge.edge_id, record_id)
        for record_id in evidence_record_ids:
            self.store.add_to_vault(vault_id, record_id, concept_id)
        return concept

    def add_relation(
        self,
        source_concept_id: str,
        target_concept_id: str,
        *,
        side: GraphSide,
        delta_y: float = 1.0,
        pulse: int = 0,
        evidence_record_ids: Sequence[str] = (),
    ) -> GraphEdge:
        for concept_id in (source_concept_id, target_concept_id):
            if not self.store.has_concept(concept_id):
                raise KeyError(f"unknown concept: {concept_id}")
        edge = self._ensure_edge(
            side,
            source_concept_id,
            target_concept_id,
            delta_y=delta_y,
            created_pulse=pulse,
        )
        for record_id in evidence_record_ids:
            self.store.add_edge_evidence(edge.edge_id, record_id)
        return edge

    # ------------------------------------------------------------ relative mass

    def weight_snapshot(
        self,
        *,
        now: float | None = None,
        side: GraphSide | None = None,
        root_node_id: str | None = None,
        maximum_depth: int | None = None,
    ) -> WeightSnapshot:
        """Propagate conserved mass from a cipher root using sibling softmaxes.

        A side-specific diagnostic starts that side with mass 1.0. When
        ``root_node_id`` selects a trunk, its
        connector from SELF is outside the cipher's competitive denominator.
        A combined diagnostic snapshot assigns 0.5 to each side at SELF.
        """
        if maximum_depth is not None and maximum_depth < 1:
            raise ValueError("maximum_depth must be positive")
        if root_node_id is not None and side is None:
            raise ValueError("root_node_id requires a side-specific snapshot")
        if root_node_id is not None and not self.store.has_concept(root_node_id):
            raise KeyError(f"unknown cipher root: {root_node_id}")
        current = time.time() if now is None else float(now)
        edges = self.store.list_edges()
        if not edges:
            selected = (side,) if side is not None else tuple(GraphSide)
            share = 1.0 / len(selected)
            starting_mass = {item.value: share for item in selected}
            starting_nodes = {
                item.value: root_node_id or SELF_ID for item in selected
            }
            return WeightSnapshot(
                {}, {}, {}, {}, {}, {}, starting_mass, starting_nodes, 1.0, 0.0
            )
        # A simple directed path cannot contain more edges than there are
        # concepts.  Use that data-derived completeness bound when the caller
        # did not explicitly request a diagnostic depth.
        traversal_depth = (
            int(maximum_depth)
            if maximum_depth is not None
            else max(1, len(self.store.list_concepts()))
        )
        logits, local_weights, grouped = self._weight_components(
            edges, now=current
        )

        selected_sides = (side,) if side is not None else tuple(GraphSide)
        starting_share = 1.0 / len(selected_sides)
        side_starting_mass = {
            selected_side.value: starting_share for selected_side in selected_sides
        }
        starting_node_ids = {
            selected_side.value: root_node_id or SELF_ID
            for selected_side in selected_sides
        }
        global_weights: dict[str, float] = {edge.edge_id: 0.0 for edge in edges}
        node_weights: dict[str, float] = {}
        regional_weights: dict[str, float] = {}
        layer_weights: dict[str, float] = {}
        absorbed_mass = 0.0
        truncated_mass = 0.0

        outgoing: dict[tuple[GraphSide, str], list[GraphEdge]] = grouped
        for selected_side in selected_sides:
            starting_node_id = starting_node_ids[selected_side.value]
            frontier: dict[str, float] = {starting_node_id: starting_share}
            node_weights[
                f"{selected_side.value}:{starting_node_id}"
            ] = starting_share
            for depth in range(traversal_depth):
                next_frontier: dict[str, float] = {}
                layer_mass = 0.0
                for source_id, source_mass in frontier.items():
                    siblings = outgoing.get((selected_side, source_id), ())
                    if not siblings:
                        absorbed_mass += source_mass
                        continue
                    distributed = 0.0
                    for edge in siblings:
                        edge_mass = source_mass * local_weights.get(edge.edge_id, 0.0)
                        if edge_mass <= 0.0:
                            continue
                        global_weights[edge.edge_id] += edge_mass
                        next_frontier[edge.target_id] = (
                            next_frontier.get(edge.target_id, 0.0) + edge_mass
                        )
                        distributed += edge_mass
                        layer_mass += edge_mass
                        if source_id == starting_node_id:
                            regional_weights[
                                f"{selected_side.value}:{edge.target_id}"
                            ] = edge_mass
                    absorbed_mass += max(0.0, source_mass - distributed)
                layer_weights[f"{selected_side.value}:{depth}"] = layer_mass
                for node_id, node_mass in next_frontier.items():
                    key = f"{selected_side.value}:{node_id}"
                    node_weights[key] = node_weights.get(key, 0.0) + node_mass
                frontier = next_frontier
                if not frontier:
                    break
            truncated_mass += sum(frontier.values())
        return WeightSnapshot(
            global_weights=global_weights,
            effective_logits=logits,
            local_weights=local_weights,
            node_weights=node_weights,
            regional_weights=regional_weights,
            layer_weights=layer_weights,
            side_starting_mass=side_starting_mass,
            starting_node_ids=starting_node_ids,
            absorbed_mass=absorbed_mass,
            truncated_mass=truncated_mass,
        )

    def local_probabilities(
        self,
        source_id: str,
        side: GraphSide,
        *,
        snapshot: WeightSnapshot | None = None,
    ) -> dict[str, float]:
        outgoing = self.store.list_edges(side, source_id=source_id)
        if not outgoing:
            return {}
        weights = (
            snapshot.local_weights
            if snapshot is not None
            else self.edge_local_weights(side=side)
        )
        probabilities = {
            edge.edge_id: weights.get(edge.edge_id, 0.0)
            for edge in outgoing
        }
        total = sum(probabilities.values())
        if total <= 0.0:
            share = 1.0 / len(outgoing)
            return {edge.edge_id: share for edge in outgoing}
        return {
            edge_id: probability / total
            for edge_id, probability in probabilities.items()
        }

    # --------------------------------------------------------------- Y traversal

    def traverse(
        self,
        *,
        pulse_id: str,
        side: GraphSide,
        target_id: str,
        endpoint_score: float,
        required_input_trunk: InputTrunk | None = None,
        required_output_trunk: OutputTrunk | None = None,
        now: float | None = None,
        mark_active: bool = True,
        save_trace: bool = True,
    ) -> TraversalTrace | None:
        current = time.time() if now is None else float(now)
        edges = self.store.list_edges(side)
        outgoing: dict[str, list[GraphEdge]] = {}
        for edge in edges:
            outgoing.setdefault(edge.source_id, []).append(edge)
        input_root = (
            INPUT_NODE_IDS[required_input_trunk]
            if side == GraphSide.INPUT and required_input_trunk is not None
            else None
        )
        output_root = (
            OUTPUT_NODE_IDS[required_output_trunk]
            if side == GraphSide.OUTPUT and required_output_trunk is not None
            else None
        )
        cipher_root = input_root or output_root or SELF_ID
        local_weights = self.edge_local_weights(now=current, side=side)

        distances: dict[str, float] = {cipher_root: 0.0}
        previous: dict[str, tuple[str, str]] = {}
        queue: list[tuple[float, str]] = [(0.0, cipher_root)]
        visited: set[str] = set()

        while queue:
            distance, node_id = heapq.heappop(queue)
            if node_id in visited:
                continue
            visited.add(node_id)
            if node_id == target_id:
                break
            siblings = outgoing.get(node_id, ())
            total = sum(
                local_weights.get(edge.edge_id, 0.0)
                for edge in siblings
            )
            for edge in siblings:
                probability = (
                    local_weights.get(edge.edge_id, 0.0) / total
                    if total > 0.0
                    else 1.0 / len(siblings)
                )
                edge_time = (
                    edge.delta_y / (1e-6 + probability)
                    + edge.conflict_penalty
                )
                next_distance = distance + edge_time
                if next_distance < distances.get(edge.target_id, math.inf):
                    distances[edge.target_id] = next_distance
                    previous[edge.target_id] = (node_id, edge.edge_id)
                    heapq.heappush(queue, (next_distance, edge.target_id))

        if target_id not in distances:
            return None
        nodes = [target_id]
        edge_ids: list[str] = []
        cursor = target_id
        while cursor != cipher_root:
            if cursor not in previous:
                return None
            parent, edge_id = previous[cursor]
            edge_ids.append(edge_id)
            nodes.append(parent)
            cursor = parent
        nodes.reverse()
        edge_ids.reverse()
        connector_cost = 0.0
        if cipher_root != SELF_ID:
            connector = self.store.find_edge(side, SELF_ID, cipher_root)
            if connector is None or connector.archived:
                return None
            nodes.insert(0, SELF_ID)
            edge_ids.insert(0, connector.edge_id)
            # The selected intake/action lane is causal state, not a sibling
            # choice. Preserve its physical depth and conflict cost without
            # normalizing it against the other independent ciphers.
            connector_cost = connector.delta_y + connector.conflict_penalty
        if mark_active:
            for edge_id in edge_ids:
                self.store.update_edge_state(edge_id, last_active_time=current)
        trace = TraversalTrace(
            trace_id=f"trace:{pulse_id}:{side.value}:{target_id}",
            side=side,
            start_node_id=SELF_ID,
            target_node_id=target_id,
            path_node_ids=tuple(nodes),
            path_edge_ids=tuple(edge_ids),
            total_travel_time=round(connector_cost + distances[target_id], 8),
            endpoint_score=float(endpoint_score),
        )
        if save_trace:
            self.store.save_trace(pulse_id, trace)
        return trace

    def collapse_to_self(
        self,
        *,
        pulse_id: str,
        source_id: str,
        endpoint_score: float,
        required_input_trunk: InputTrunk,
        now: float | None = None,
        mark_active: bool = True,
        pulse: int | None = None,
    ) -> TraversalTrace | None:
        """Resolve an observed input endpoint inward to ``SELF``.

        Input topology is stored from SELF toward a receptor so graph growth
        and sibling conservation remain simple. Perception is the reverse
        event: the selected endpoint collapses through its input trunk into
        SELF. The route search therefore runs in stored orientation and the
        admitted causal trace is persisted in reverse.
        """
        current = time.time() if now is None else float(now)
        outward = self.traverse(
            pulse_id=f"{pulse_id}:route",
            side=GraphSide.INPUT,
            target_id=source_id,
            endpoint_score=endpoint_score,
            required_input_trunk=required_input_trunk,
            now=current,
            mark_active=False,
            save_trace=False,
        )
        if outward is None:
            return None
        if mark_active:
            for edge_id in outward.path_edge_ids:
                self.store.update_edge_state(edge_id, last_active_time=current)
            self.store.mark_concepts_active(
                outward.path_node_ids,
                max(0, int(pulse)) if pulse is not None else 0,
            )
        collapse = TraversalTrace(
            trace_id=f"trace:{pulse_id}:input-collapse:{source_id}",
            side=GraphSide.INPUT,
            start_node_id=source_id,
            target_node_id=SELF_ID,
            path_node_ids=tuple(reversed(outward.path_node_ids)),
            path_edge_ids=tuple(reversed(outward.path_edge_ids)),
            total_travel_time=outward.total_travel_time,
            endpoint_score=float(endpoint_score),
            evidence_record_ids=outward.evidence_record_ids,
        )
        self.store.save_trace(pulse_id, collapse)
        return collapse

    def traverse_many(
        self,
        *,
        pulse_id: str,
        side: GraphSide,
        targets: Mapping[str, float],
        required_input_trunk: InputTrunk | None = None,
        required_output_trunk: OutputTrunk | None = None,
        now: float | None = None,
        mark_active: bool = True,
    ) -> tuple[TraversalTrace, ...]:
        """Resolve many endpoints through one frozen cipher snapshot.

        This is semantically equivalent to repeated ``traverse`` calls, but a
        mature no-context mind may nominate several concepts per pulse. One
        shared Dijkstra pass keeps latency from scaling with that shortlist.
        Probe-only calls are not persisted; admitted traces are saved when
        ``mark_active`` is true.
        """
        if not targets:
            return ()
        if required_input_trunk is not None and side != GraphSide.INPUT:
            raise ValueError("an input trunk can constrain only input traversal")
        if required_output_trunk is not None and side != GraphSide.OUTPUT:
            raise ValueError("an output trunk can constrain only output traversal")
        current = time.time() if now is None else float(now)
        edges = self.store.list_edges(side)
        outgoing: dict[str, list[GraphEdge]] = {}
        for edge in edges:
            outgoing.setdefault(edge.source_id, []).append(edge)
        input_root = (
            INPUT_NODE_IDS[required_input_trunk]
            if side == GraphSide.INPUT and required_input_trunk is not None
            else None
        )
        output_root = (
            OUTPUT_NODE_IDS[required_output_trunk]
            if side == GraphSide.OUTPUT and required_output_trunk is not None
            else None
        )
        cipher_root = input_root or output_root or SELF_ID
        local_weights = self.edge_local_weights(now=current, side=side)
        distances: dict[str, float] = {cipher_root: 0.0}
        previous: dict[str, tuple[str, str]] = {}
        queue: list[tuple[float, str]] = [(0.0, cipher_root)]
        visited: set[str] = set()
        remaining = set(str(node_id) for node_id in targets)
        while queue and remaining:
            distance, node_id = heapq.heappop(queue)
            if node_id in visited:
                continue
            visited.add(node_id)
            remaining.discard(node_id)
            siblings = outgoing.get(node_id, ())
            total = sum(local_weights.get(edge.edge_id, 0.0) for edge in siblings)
            for edge in siblings:
                probability = (
                    local_weights.get(edge.edge_id, 0.0) / total
                    if total > 0.0
                    else 1.0 / len(siblings)
                )
                edge_time = edge.delta_y / (1e-6 + probability) + edge.conflict_penalty
                next_distance = distance + edge_time
                if next_distance < distances.get(edge.target_id, math.inf):
                    distances[edge.target_id] = next_distance
                    previous[edge.target_id] = (node_id, edge.edge_id)
                    heapq.heappush(queue, (next_distance, edge.target_id))

        connector = (
            self.store.find_edge(side, SELF_ID, cipher_root)
            if cipher_root != SELF_ID
            else None
        )
        if cipher_root != SELF_ID and (connector is None or connector.archived):
            return ()
        connector_cost = (
            connector.delta_y + connector.conflict_penalty if connector else 0.0
        )
        traces = []
        for target_id, endpoint_score in targets.items():
            if target_id not in distances:
                continue
            nodes = [target_id]
            edge_ids = []
            cursor = target_id
            while cursor != cipher_root:
                if cursor not in previous:
                    nodes = []
                    break
                parent, edge_id = previous[cursor]
                edge_ids.append(edge_id)
                nodes.append(parent)
                cursor = parent
            if not nodes:
                continue
            nodes.reverse()
            edge_ids.reverse()
            if connector is not None:
                nodes.insert(0, SELF_ID)
                edge_ids.insert(0, connector.edge_id)
            trace = TraversalTrace(
                trace_id=f"trace:{pulse_id}:{side.value}:{target_id}",
                side=side,
                start_node_id=SELF_ID,
                target_node_id=target_id,
                path_node_ids=tuple(nodes),
                path_edge_ids=tuple(edge_ids),
                total_travel_time=round(connector_cost + distances[target_id], 8),
                endpoint_score=float(endpoint_score),
            )
            traces.append(trace)
            if mark_active:
                self.activate_trace(pulse_id, trace, now=current)
        return tuple(traces)

    def trace_explicit_path(
        self,
        *,
        pulse_id: str,
        side: GraphSide,
        path_node_ids: Sequence[str],
        endpoint_score: float = 1.0,
        now: float | None = None,
        mark_active: bool = True,
    ) -> TraversalTrace:
        """Validate and score a known causal path without re-routing it.

        X/Y search is appropriate when the route is unknown. An observed tool
        return already identifies the motor fiber that produced it, so letting
        Dijkstra substitute a cheaper sibling would corrupt causal learning.
        """
        nodes = tuple(path_node_ids)
        if not nodes or nodes[0] != SELF_ID or len(nodes) < 2:
            raise ValueError("an explicit path must begin at SELF and contain an edge")
        current = time.time() if now is None else float(now)
        valid_roots = (
            set(INPUT_NODE_IDS.values())
            if side == GraphSide.INPUT
            else set(OUTPUT_NODE_IDS.values())
        )
        cipher_root = nodes[1] if nodes[1] in valid_roots else SELF_ID
        edge_ids: list[str] = []
        travel_time = 0.0
        for source_id, target_id in zip(nodes, nodes[1:]):
            edge = self.store.find_edge(side, source_id, target_id)
            if edge is None or edge.archived:
                raise ValueError(
                    f"explicit {side.value} path has no edge: {source_id} -> {target_id}"
                )
            if source_id == SELF_ID and target_id == cipher_root:
                probability = 1.0
            else:
                local = self.local_probabilities(source_id, side)
                probability = local.get(edge.edge_id, 0.0)
            travel_time += edge.delta_y / (1e-6 + probability) + edge.conflict_penalty
            edge_ids.append(edge.edge_id)
            if mark_active:
                self.store.update_edge_state(edge.edge_id, last_active_time=current)
        trace = TraversalTrace(
            trace_id=f"trace:{pulse_id}:{side.value}:{nodes[-1]}",
            side=side,
            start_node_id=SELF_ID,
            target_node_id=nodes[-1],
            path_node_ids=nodes,
            path_edge_ids=tuple(edge_ids),
            total_travel_time=round(travel_time, 8),
            endpoint_score=float(endpoint_score),
        )
        self.store.save_trace(pulse_id, trace)
        if mark_active:
            self.store.mark_concepts_active(nodes, 0)
        return trace

    def expanded_concept_ids(
        self,
        traces: Iterable[TraversalTrace],
        *,
        side: GraphSide,
        maximum: int | None = None,
    ) -> list[str]:
        selected: list[str] = []
        seen: set[str] = set()
        path_nodes = {
            node_id for trace in traces for node_id in trace.path_node_ids
        }
        local_weights = self.edge_local_weights(side=side)
        edges = self.store.list_edges(side)
        outgoing: dict[str, list[GraphEdge]] = {}
        for edge in edges:
            outgoing.setdefault(edge.source_id, []).append(edge)
        candidates: list[tuple[float, str]] = []
        for source_id in path_nodes:
            siblings = outgoing.get(source_id, ())
            total = sum(local_weights.get(edge.edge_id, 0.0) for edge in siblings)
            local = {
                edge.edge_id: (
                    local_weights.get(edge.edge_id, 0.0) / total
                    if total > 0.0 else 1.0 / len(siblings)
                )
                for edge in siblings
            }
            for edge in siblings:
                target = self.store.get_concept(edge.target_id)
                if target is None or target.kind != "crown" or edge.target_id in path_nodes:
                    continue
                candidates.append((local.get(edge.edge_id, 0.0), edge.target_id))
        for _, concept_id in sorted(candidates, key=lambda item: (-item[0], item[1])):
            if concept_id in seen:
                continue
            seen.add(concept_id)
            selected.append(concept_id)
            if maximum is not None and len(selected) >= maximum:
                break
        return selected

    def activate_trace(self, pulse_id: str, trace: TraversalTrace, *, now: float | None = None) -> None:
        current = time.time() if now is None else float(now)
        for edge_id in trace.path_edge_ids:
            self.store.update_edge_state(edge_id, last_active_time=current)
        self.store.mark_concepts_active(trace.path_node_ids, int(pulse_id.rsplit(":", 1)[-1]) if pulse_id.rsplit(":", 1)[-1].isdigit() else 0)
        self.store.save_trace(pulse_id, trace)

    # --------------------------------------------------------- learning/feedback

    def reinforce_edges(
        self,
        edge_ids: Iterable[str],
        *,
        stability_delta: float,
        verified: bool,
        evidence_quality: float = 1.0,
    ) -> None:
        if not verified:
            return
        credited = list(dict.fromkeys(edge_ids))
        if not credited:
            return
        delta = max(-1.0, min(1.0, float(stability_delta)))
        quality = max(0.0, min(1.0, float(evidence_quality)))
        path_credit = 1.0 / len(credited)
        change = self.learning_rate * delta * quality * path_credit
        for edge_id in credited:
            edge = self.store.get_edge(edge_id)
            if edge is None:
                continue
            penalty = edge.conflict_penalty
            if delta < 0.0:
                penalty = min(10.0, penalty + abs(change) * 0.25)
            elif penalty:
                penalty = max(0.0, penalty - abs(change) * 0.10)
            self.store.update_edge_state(
                edge_id,
                log_strength=edge.log_strength + change,
                conflict_penalty=penalty,
            )

    def habituate_edges(
        self,
        edge_ids: Iterable[str],
        *,
        exposure_strength: float,
        evidence_quality: float = 1.0,
    ) -> None:
        """Deepen repeatedly experienced routes without treating exposure as reward."""
        strength = max(0.0, min(1.0, float(exposure_strength)))
        quality = max(0.0, min(1.0, float(evidence_quality)))
        change = self.learning_rate * strength * quality
        if change <= 0.0:
            return
        for edge_id in dict.fromkeys(str(item) for item in edge_ids):
            edge = self.store.get_edge(edge_id)
            if edge is None:
                continue
            self.store.update_edge_state(
                edge_id,
                log_strength=min(8.0, edge.log_strength + change),
            )

    def validate_invariants(self, *, tolerance: float = 1e-9) -> list[str]:
        errors: list[str] = []
        if self.store.get_concept(SELF_ID) is None:
            errors.append("SELF is missing")
        for node_id in (*INPUT_NODE_IDS.values(), *OUTPUT_NODE_IDS.values()):
            if self.store.get_concept(node_id) is None:
                errors.append(f"seed trunk is missing: {node_id}")
        for (trunk, band), node_id in PREFERENCE_NODE_IDS.items():
            node = self.store.get_concept(node_id)
            if node is None:
                errors.append(f"lower preference node is missing: {node_id}")
                continue
            if not node.vault_id:
                errors.append(f"lower preference vault is missing: {node_id}")
            if self.store.find_edge(
                GraphSide.INPUT, INPUT_NODE_IDS[trunk], node_id
            ) is None:
                errors.append(f"lower preference edge is missing: {trunk.value}:{band}")
        local_weights = self.edge_local_weights()
        for side in GraphSide:
            sources = {edge.source_id for edge in self.store.list_edges(side)}
            for source_id in sources:
                outgoing = self.store.list_edges(side, source_id=source_id)
                total = sum(local_weights.get(edge.edge_id, 0.0) for edge in outgoing)
                local = {
                    edge.edge_id: (
                        local_weights.get(edge.edge_id, 0.0) / total
                        if total > 0.0 else 1.0 / len(outgoing)
                    )
                    for edge in outgoing
                }
                if local and abs(sum(local.values()) - 1.0) > tolerance:
                    errors.append(
                        f"local edge mass for {side.value}:{source_id} is {sum(local.values())}"
                    )
        self_outgoing_input = [
            edge for edge in self.store.list_edges(GraphSide.INPUT)
            if edge.source_id == SELF_ID
        ]
        self_outgoing_output = [
            edge for edge in self.store.list_edges(GraphSide.OUTPUT)
            if edge.source_id == SELF_ID
        ]
        if {edge.target_id for edge in self_outgoing_input} != set(INPUT_NODE_IDS.values()):
            errors.append("SELF input frontier is not exactly HEAR/SEE/NOTICE")
        if {edge.target_id for edge in self_outgoing_output} != set(OUTPUT_NODE_IDS.values()):
            errors.append("SELF output frontier is not exactly SPEAK/LOOK/DO")
        for child in self.store.list_concepts(kind="child"):
            cluster = self.store.overlap_cluster_for_child(child.concept_id)
            if cluster is None:
                errors.append(f"child has no overlap cluster: {child.concept_id}")
                continue
            if any(child.embedding) or child.terms:
                errors.append(f"lower child carries semantic payload: {child.concept_id}")
            if not child.vault_id:
                errors.append(f"child lower vault is missing: {child.concept_id}")
            if cluster.semantic_node_id is not None:
                semantic = self.store.get_concept(cluster.semantic_node_id)
                if semantic is None:
                    errors.append(f"child semantic port is missing: {child.concept_id}")
                elif any(
                    not self._record_has_membrane_words(record)
                    for record in self.store.records_for_vault(semantic.vault_id)
                ):
                    errors.append(
                        f"semantic port has non-HEAR word evidence: {child.concept_id}"
                    )
        developmental_edges = self.store.list_edges(
            GraphSide.INPUT, include_archived=True
        )
        incoming_developmental: dict[str, list[GraphEdge]] = {}
        for edge in developmental_edges:
            incoming_developmental.setdefault(edge.target_id, []).append(edge)
        for feature in self.store.list_concepts(kind=SENSORY_FEATURE_KIND):
            if feature.terms or feature.label != feature.concept_id:
                errors.append(
                    f"sensory feature carries language payload: {feature.concept_id}"
                )
            if not feature.vault_id:
                errors.append(f"sensory feature vault is missing: {feature.concept_id}")
            parts = feature.concept_id.split(":")
            try:
                trunk = InputTrunk(parts[1].upper())
            except (IndexError, ValueError):
                errors.append(f"sensory feature has malformed trunk ID: {feature.concept_id}")
                continue
            valid_parents = {
                PREFERENCE_NODE_IDS[(trunk, band)] for band in PREFERENCE_BANDS
            }
            rooted = [
                edge
                for edge in incoming_developmental.get(feature.concept_id, ())
                if edge.source_id in valid_parents and not edge.archived
            ]
            if not rooted:
                errors.append(
                    f"sensory feature is not rooted in {trunk.value}: {feature.concept_id}"
                )
        for kind in (EXPERIENTIAL_CANDIDATE_KIND, EXPERIENTIAL_PATTERN_KIND):
            for pattern in self.store.list_concepts(kind=kind):
                if pattern.terms or pattern.label != pattern.concept_id:
                    errors.append(
                        f"experiential pattern carries language payload: {pattern.concept_id}"
                    )
                if not pattern.vault_id:
                    errors.append(
                        f"experiential pattern vault is missing: {pattern.concept_id}"
                    )
                parents = [
                    edge
                    for edge in incoming_developmental.get(pattern.concept_id, ())
                    if (
                        (source := self.store.get_concept(edge.source_id)) is not None
                        and source.kind == SENSORY_FEATURE_KIND
                    )
                ]
                if len(parents) < 2:
                    errors.append(
                        f"experiential pattern has fewer than two sensory parents: {pattern.concept_id}"
                    )
                elif kind == EXPERIENTIAL_CANDIDATE_KIND and any(
                    not edge.archived for edge in parents
                ):
                    errors.append(
                        f"immature experiential pattern is traversable: {pattern.concept_id}"
                    )
                elif kind == EXPERIENTIAL_PATTERN_KIND and any(
                    edge.archived for edge in parents
                ):
                    errors.append(
                        f"mature experiential pattern has a dormant parent: {pattern.concept_id}"
                    )
        return errors

    # --------------------------------------------------------------- growth stage

    @staticmethod
    def _experience_id(record: MemoryRecord) -> str:
        return str(record.metadata.get("experience_id") or record.event_id or record.record_id)

    @staticmethod
    def _preference_signal(metadata: Mapping[str, object]) -> tuple[float, float]:
        signals: list[float] = []
        raw_signals = metadata.get("preference_signals", ())
        if isinstance(raw_signals, (list, tuple)):
            for value in raw_signals:
                try:
                    signals.append(float(value))
                except (TypeError, ValueError):
                    continue
        for key in ("preference", "stability_delta"):
            if key in metadata:
                try:
                    signals.append(float(metadata[key]))
                except (TypeError, ValueError):
                    pass
        if not signals:
            return 0.0, 0.0
        mean = sum(max(-1.0, min(1.0, value)) for value in signals) / len(signals)
        try:
            confidence = float(metadata.get("preference_confidence", 1.0))
        except (TypeError, ValueError):
            confidence = 1.0
        return max(-1.0, min(1.0, mean)), max(0.0, min(1.0, confidence))

    @staticmethod
    def _preference_band(preference: float, confidence: float) -> str:
        if confidence <= 0.0 or abs(preference) < 0.05:
            return "NEUTRAL"
        return "STABLE" if preference > 0.0 else "UNSTABLE"

    def deposit_experience(
        self,
        record: MemoryRecord,
        *,
        input_trunk: InputTrunk,
        pulse: int,
    ) -> str:
        """Store language-free projections in the basal and preference vaults."""
        experience_id = self._experience_id(record)
        preference, confidence = self._preference_signal(record.metadata)
        state = self.store.update_experience_state(
            experience_id,
            preference=preference,
            confidence=confidence,
            pulse=pulse,
        )
        band = self._preference_band(state.preference_mean, state.preference_weight)
        preference_node_id = PREFERENCE_NODE_IDS[(input_trunk, band)]
        nodes = (
            (SELF_ID, 0, "self"),
            (INPUT_NODE_IDS[input_trunk], 1, "stimulus"),
            (preference_node_id, 2, "preference"),
        )
        for node_id, layer, projection_kind in nodes:
            concept = self.store.get_concept(node_id)
            if concept is None:
                continue
            if concept.vault_id:
                self.store.add_to_vault(concept.vault_id, record.record_id, node_id)
            self.store.add_experience_projection(
                ExperienceProjection(
                    experience_id=experience_id,
                    record_id=record.record_id,
                    node_id=node_id,
                    layer=layer,
                    side=GraphSide.INPUT,
                    activation=1.0,
                    preference=state.preference_mean,
                    confidence=min(1.0, state.preference_weight),
                    pulse=pulse,
                    metadata={"projection": projection_kind, "band": band},
                )
            )
        return preference_node_id

    def deposit_trace(
        self,
        record: MemoryRecord,
        trace: TraversalTrace,
        *,
        pulse: int,
        grow_visited_children: bool = True,
    ) -> None:
        experience_id = self._experience_id(record)
        state = self.store.get_experience_state(experience_id)
        preference = state.preference_mean if state else 0.0
        confidence = min(1.0, state.preference_weight) if state else 0.0
        for layer, node_id in enumerate(trace.path_node_ids):
            concept = self.store.get_concept(node_id)
            if concept is None:
                continue
            if concept.vault_id:
                self.store.add_to_vault(concept.vault_id, record.record_id, node_id)
            self.store.add_experience_projection(
                ExperienceProjection(
                    experience_id=experience_id,
                    record_id=record.record_id,
                    node_id=node_id,
                    layer=layer,
                    side=trace.side,
                    activation=1.0,
                    preference=preference,
                    confidence=confidence,
                    pulse=pulse,
                    metadata={"projection": "traversal"},
                )
            )
        if trace.side == GraphSide.INPUT and grow_visited_children:
            trunk = next(
                (
                    trunk
                    for trunk, trunk_node_id in INPUT_NODE_IDS.items()
                    if trunk_node_id in trace.path_node_ids
                ),
                None,
            )
            if trunk is not None:
                for node_id in trace.path_node_ids:
                    concept = self.store.get_concept(node_id)
                    if concept is None or concept.kind != "child":
                        continue
                    cluster = self.store.overlap_cluster_for_child(node_id)
                    if cluster is not None and experience_id not in cluster.experience_ids:
                        self.stage_growth(
                            record,
                            input_trunk=trunk,
                            pulse=pulse,
                            parent_node_id=cluster.parent_node_id,
                            preferred_cluster_id=cluster.cluster_id,
                        )

    # --------------------------------------------------- trunk-rooted breadth

    @staticmethod
    def _unit_vector(values: Sequence[float]) -> tuple[float, ...]:
        vector = [float(value) for value in values]
        norm = math.sqrt(sum(value * value for value in vector))
        if norm <= 1e-12:
            return tuple(0.0 for _ in vector)
        return tuple(value / norm for value in vector)

    def _breadth_planes(
        self,
        *,
        banks: int,
        signature_bits: int,
        samples: int,
    ) -> tuple[tuple[tuple[tuple[int, float], ...], ...], ...]:
        """Return deterministic sparse hyperplanes for locality-sensitive fibers."""
        key = (int(banks), int(signature_bits), int(samples))
        cached = self._breadth_plane_cache.get(key)
        if cached is not None:
            return cached
        dimension = int(self.embedder.dimension)
        sample_count = min(dimension, max(8, int(samples)))
        mask = (1 << 64) - 1
        generated = []
        for bank in range(max(1, int(banks))):
            bank_planes = []
            for bit in range(max(1, int(signature_bits))):
                seed = hashlib.sha256(
                    (
                        "habitus.trunk-breadth.v1|"
                        f"{self.embedder.space_id}|{dimension}|{bank}|{bit}"
                    ).encode("utf-8")
                ).digest()
                start = int.from_bytes(seed[:8], "big") % dimension
                step = int.from_bytes(seed[8:16], "big") % dimension or 1
                while math.gcd(step, dimension) != 1:
                    step = (step + 1) % dimension or 1
                state = int.from_bytes(seed[16:24], "big") or 1
                coordinates = []
                for offset in range(sample_count):
                    index = (start + offset * step) % dimension
                    state = (
                        state * 6364136223846793005 + 1442695040888963407
                    ) & mask
                    coordinates.append((index, 1.0 if state & 1 else -1.0))
                bank_planes.append(tuple(sorted(coordinates)))
            generated.append(tuple(bank_planes))
        result = tuple(generated)
        self._breadth_plane_cache[key] = result
        return result

    def _sensory_field(
        self,
        record: MemoryRecord,
        *,
        input_trunk: InputTrunk,
        sensory_rows: Sequence[Sequence[float]],
    ) -> tuple[float, ...]:
        dimension = int(self.embedder.dimension)
        rows = []
        for row in sensory_rows:
            if len(row) != dimension:
                raise ValueError("sensory breadth row dimension mismatch")
            normalized = self._unit_vector(row)
            if any(normalized):
                rows.append(normalized)
        if not rows and len(record.embedding) == dimension and any(record.embedding):
            rows.append(self._unit_vector(record.embedding))
        if rows:
            pooled = [
                sum(row[index] for row in rows) / len(rows)
                for index in range(dimension)
            ]
            normalized = self._unit_vector(pooled)
            if any(normalized):
                return normalized
        return as_tuple(
            opaque_payload_embedding(
                record.text,
                dimension,
                namespace=f"trunk-breadth:{input_trunk.value}:{record.record_type.value}",
            )
        )

    def _sensory_codes(
        self,
        vector: Sequence[float],
        *,
        banks: int,
        signature_bits: int,
        samples: int,
        minimum_signature_bits: int = 3,
    ) -> tuple[tuple[int, int, int], ...]:
        if len(vector) != self.embedder.dimension:
            raise ValueError("sensory breadth vector dimension mismatch")
        bank_count = max(1, int(banks))
        maximum_bits = max(1, int(signature_bits))
        minimum_bits = min(
            maximum_bits, max(1, int(minimum_signature_bits))
        )
        result = []
        for bank, planes in enumerate(
            self._breadth_planes(
                banks=bank_count,
                signature_bits=maximum_bits,
                samples=samples,
            )
        ):
            # Receptor banks span coarse-to-fine resolutions. Coarse fibers
            # recur readily enough to seed categories; fine fibers preserve
            # distinctions instead of collapsing the whole sensation into a
            # handful of generic bins.
            bits = (
                maximum_bits
                if bank_count == 1
                else minimum_bits
                + round(
                    bank
                    * (maximum_bits - minimum_bits)
                    / (bank_count - 1)
                )
            )
            signature = 0
            for bit, plane in enumerate(planes[:bits]):
                projection = sum(vector[index] * sign for index, sign in plane)
                if projection >= 0.0:
                    signature |= 1 << bit
            result.append((bank, bits, signature))
        return tuple(result)

    def _project_numeric_node(
        self,
        node_id: str,
        *,
        kind: str,
        vector: Sequence[float],
        record: MemoryRecord,
        pulse: int,
        layer: int,
        projection_kind: str,
    ) -> ConceptNode:
        existing = self.store.get_concept(node_id)
        allowed_kinds = (
            {SENSORY_FEATURE_KIND}
            if kind == SENSORY_FEATURE_KIND
            else {EXPERIENTIAL_CANDIDATE_KIND, EXPERIENTIAL_PATTERN_KIND}
        )
        if existing is not None and existing.kind not in allowed_kinds:
            raise RuntimeError(
                f"developmental geometry collides with {existing.kind}: {node_id}"
            )
        vault_id = f"lower-vault:{node_id}"
        prior_count = self.store.vault_record_count(vault_id)
        normalized = self._unit_vector(vector)
        if existing is None:
            concept = self.store.add_concept(
                ConceptNode(
                    concept_id=node_id,
                    label=node_id,
                    kind=kind,
                    embedding=normalized,
                    terms=(),
                    vault_id=vault_id,
                    created_pulse=pulse,
                    last_active_pulse=pulse,
                )
            )
        else:
            concept = existing
            if any(normalized):
                centroid = [
                    (old * prior_count + new) / (prior_count + 1)
                    for old, new in zip(existing.embedding, normalized)
                ]
                self.store.update_concept_embedding(
                    node_id, self._unit_vector(centroid), terms=()
                )
                concept = self.store.get_concept(node_id) or existing
        self.store.add_to_vault(vault_id, record.record_id, node_id)
        experience_id = self._experience_id(record)
        state = self.store.get_experience_state(experience_id)
        self.store.add_experience_projection(
            ExperienceProjection(
                experience_id=experience_id,
                record_id=record.record_id,
                node_id=node_id,
                layer=int(layer),
                side=GraphSide.INPUT,
                activation=1.0,
                preference=state.preference_mean if state else 0.0,
                confidence=min(1.0, state.preference_weight) if state else 0.0,
                pulse=pulse,
                metadata={"projection": projection_kind},
            )
        )
        return concept

    def _stage_numeric_pattern(
        self,
        parent_node_ids: Sequence[str],
        *,
        vector: Sequence[float],
        record: MemoryRecord,
        pulse: int,
        promotion_count: int,
        cross_trunk: bool,
    ) -> tuple[str, bool, tuple[str, ...]]:
        parents = tuple(dict.fromkeys(str(node_id) for node_id in parent_node_ids))
        if len(parents) < 2:
            raise ValueError("an experiential pattern requires at least two parents")
        material = "|".join(("cross" if cross_trunk else "within", *sorted(parents)))
        digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]
        node_id = f"experience-pattern:{digest}"
        prior = self.store.get_concept(node_id)
        self._project_numeric_node(
            node_id,
            kind=EXPERIENTIAL_CANDIDATE_KIND,
            vector=vector,
            record=record,
            pulse=pulse,
            layer=4,
            projection_kind=(
                "cross_trunk_convergence" if cross_trunk else "within_trunk_intersection"
            ),
        )
        mature_before = prior is not None and prior.kind == EXPERIENTIAL_PATTERN_KIND
        edge_ids = []
        for parent_id in parents:
            existing_edge = self.store.find_edge(
                GraphSide.INPUT, parent_id, node_id
            )
            edge = self._ensure_edge(
                GraphSide.INPUT,
                parent_id,
                node_id,
                delta_y=1.0,
                log_strength=-1.0,
                created_pulse=pulse,
            )
            if existing_edge is None and not mature_before:
                self.store.update_edge_state(edge.edge_id, archived=True)
            self.store.add_edge_evidence(
                edge.edge_id,
                record.record_id,
                "coactive",
            )
            edge_ids.append(edge.edge_id)

        evidence_count = self.store.vault_record_count(f"lower-vault:{node_id}")
        mature = mature_before or evidence_count >= max(2, int(promotion_count))
        if mature:
            self.store.set_concept_kind(node_id, EXPERIENTIAL_PATTERN_KIND)
            for edge_id in edge_ids:
                self.store.update_edge_state(edge_id, archived=False)
        self.habituate_edges(
            edge_ids,
            exposure_strength=0.10 if mature else 0.035,
            evidence_quality=1.0,
        )
        return node_id, mature, tuple(edge_ids)

    def grow_trunk_breadth(
        self,
        record: MemoryRecord,
        *,
        input_trunk: InputTrunk,
        pulse: int,
        sensory_rows: Sequence[Sequence[float]] = (),
        feature_banks: int = 6,
        signature_bits: int = 8,
        plane_samples: int = 32,
        maximum_patterns: int | None = None,
        promotion_count: int = 2,
    ) -> DevelopmentalGrowth:
        """Grow sparse language-free fibers outward from one actual input trunk.

        Locality-sensitive numeric codes create reusable weak distinctions. A
        single experience may create branches, but intersections are traversable
        only after independent experiences select the same route.
        """
        vector = self._sensory_field(
            record,
            input_trunk=input_trunk,
            sensory_rows=sensory_rows,
        )
        experience_id = self._experience_id(record)
        state = self.store.get_experience_state(experience_id)
        preference = state.preference_mean if state else 0.0
        confidence = min(1.0, state.preference_weight) if state else 0.0
        band = self._preference_band(preference, confidence)
        parent_id = PREFERENCE_NODE_IDS[(input_trunk, band)]
        feature_ids = []
        edge_ids = []
        for bank, bits, signature in self._sensory_codes(
            vector,
            banks=max(1, int(feature_banks)),
            signature_bits=max(1, int(signature_bits)),
            samples=max(1, int(plane_samples)),
        ):
            width = max(1, math.ceil(bits / 4))
            node_id = (
                f"sensory-fiber:{input_trunk.value.lower()}:{band.lower()}:"
                f"{bank:02d}:b{bits:02d}:{signature:0{width}x}"
            )
            self._project_numeric_node(
                node_id,
                kind=SENSORY_FEATURE_KIND,
                vector=vector,
                record=record,
                pulse=pulse,
                layer=3,
                projection_kind="trunk_sensory_fiber",
            )
            edge = self._ensure_edge(
                GraphSide.INPUT,
                parent_id,
                node_id,
                delta_y=1.0,
                log_strength=-0.65,
                created_pulse=pulse,
            )
            self.store.add_edge_evidence(edge.edge_id, record.record_id, "selected")
            self.habituate_edges(
                (edge.edge_id,), exposure_strength=0.08, evidence_quality=1.0
            )
            feature_ids.append(node_id)
            edge_ids.append(edge.edge_id)

        candidate_patterns = []
        mature_patterns = []
        adjacent_pairs = tuple(zip(feature_ids, feature_ids[1:]))
        selected_pairs = (
            adjacent_pairs if maximum_patterns is None
            else adjacent_pairs[: max(0, int(maximum_patterns))]
        )
        for left, right in selected_pairs:
            node_id, mature, pattern_edges = self._stage_numeric_pattern(
                (left, right),
                vector=vector,
                record=record,
                pulse=pulse,
                promotion_count=promotion_count,
                cross_trunk=False,
            )
            (mature_patterns if mature else candidate_patterns).append(node_id)
            edge_ids.extend(pattern_edges)
        active_contexts = tuple(
            dict.fromkeys((*feature_ids, *mature_patterns, *candidate_patterns))
        )
        return DevelopmentalGrowth(
            record_ids=(record.record_id,),
            input_trunks=(input_trunk,),
            feature_node_ids=tuple(feature_ids),
            candidate_pattern_node_ids=tuple(candidate_patterns),
            promoted_pattern_node_ids=tuple(mature_patterns),
            active_context_node_ids=active_contexts,
            edge_ids=tuple(dict.fromkeys(edge_ids)),
            cross_trunk=False,
        )

    def grow_cross_trunk_breadth(
        self,
        growth: Sequence[DevelopmentalGrowth],
        *,
        anchor_record: MemoryRecord,
        pulse: int,
        features_per_trunk: int | None = None,
        maximum_patterns: int | None = None,
        promotion_count: int = 2,
    ) -> DevelopmentalGrowth | None:
        """Grow convergences only from senses that settled into the same SELF pulse."""
        by_trunk: dict[InputTrunk, tuple[str, ...]] = {}
        record_ids = []
        for receipt in growth:
            record_ids.extend(receipt.record_ids)
            if receipt.cross_trunk or len(receipt.input_trunks) != 1:
                continue
            trunk = receipt.input_trunks[0]
            by_trunk[trunk] = (
                receipt.feature_node_ids if features_per_trunk is None
                else receipt.feature_node_ids[: max(1, int(features_per_trunk))]
            )
        trunks = sorted(by_trunk, key=lambda item: item.value)
        if len(trunks) < 2:
            return None

        pairings = []
        for left_index, left_trunk in enumerate(trunks):
            for right_trunk in trunks[left_index + 1 :]:
                for index in range(
                    min(len(by_trunk[left_trunk]), len(by_trunk[right_trunk]))
                ):
                    pairings.append(
                        (by_trunk[left_trunk][index], by_trunk[right_trunk][index])
                    )
        candidate_patterns = []
        mature_patterns = []
        selected_features = []
        edge_ids = []
        selected_pairings = (
            pairings if maximum_patterns is None
            else pairings[: max(0, int(maximum_patterns))]
        )
        for parents in selected_pairings:
            parent_concepts = [
                self.store.get_concept(node_id) for node_id in parents
            ]
            if any(concept is None for concept in parent_concepts):
                continue
            parent_vectors = [concept.embedding for concept in parent_concepts]
            vector = self._unit_vector(
                [
                    sum(row[index] for row in parent_vectors) / len(parent_vectors)
                    for index in range(self.embedder.dimension)
                ]
            )
            node_id, mature, pattern_edges = self._stage_numeric_pattern(
                parents,
                vector=vector,
                record=anchor_record,
                pulse=pulse,
                promotion_count=promotion_count,
                cross_trunk=True,
            )
            selected_features.extend(parents)
            (mature_patterns if mature else candidate_patterns).append(node_id)
            edge_ids.extend(pattern_edges)
        return DevelopmentalGrowth(
            record_ids=tuple(dict.fromkeys((*record_ids, anchor_record.record_id))),
            input_trunks=tuple(trunks),
            feature_node_ids=tuple(dict.fromkeys(selected_features)),
            candidate_pattern_node_ids=tuple(candidate_patterns),
            promoted_pattern_node_ids=tuple(mature_patterns),
            active_context_node_ids=tuple(
                dict.fromkeys((*mature_patterns, *candidate_patterns))
            ),
            edge_ids=tuple(dict.fromkeys(edge_ids)),
            cross_trunk=True,
        )

    @staticmethod
    def _growth_terms(text: str) -> list[str]:
        stop = {
            "about", "after", "again", "could", "from", "have", "into",
            "just", "more", "that", "their", "then", "there", "they", "this",
            "what", "when", "where", "which", "with", "would", "your",
        }
        counts: dict[str, int] = {}
        for token in tokenize(text):
            if len(token) < 4 or token in stop:
                continue
            counts[token] = counts.get(token, 0) + 1
        return [
            token for token, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        ]

    @staticmethod
    def _record_has_membrane_words(record: MemoryRecord) -> bool:
        marker = record.metadata.get("membrane_words")
        if marker is not None:
            return (
                bool(marker)
                and record.metadata.get("causal_trunk") == InputTrunk.HEAR.value
            )
        return record.record_type in {
            RecordType.INBOUND_MESSAGE,
            RecordType.FACT,
            RecordType.RAW_MEMORY,
        }

    def attach_semantic_port(
        self,
        child_node_id: str,
        record: MemoryRecord,
        *,
        pulse: int,
        terms: Sequence[str] = (),
    ) -> ConceptNode:
        """Attach a HEAR-learned word port to an existing nonverbal child."""
        child = self.store.get_concept(child_node_id)
        if child is None or child.kind != "child":
            raise ValueError("semantic ports attach only to promoted child nodes")
        if not self._record_has_membrane_words(record):
            raise ValueError("semantic port evidence must carry HEAR membrane words")
        cluster = self.store.overlap_cluster_for_child(child_node_id)
        if cluster is None:
            raise ValueError("semantic port child has no overlap cluster")
        digest = cluster.cluster_id.rsplit(":", 1)[-1]
        semantic_id = cluster.semantic_node_id or f"concept:auto:{digest}"
        selected_terms = tuple(terms) or tuple(self._growth_terms(record.text))
        semantic = self.store.add_concept(
            ConceptNode(
                concept_id=semantic_id,
                label=(" ".join(selected_terms[:2]).title() or f"Concept {digest[:8]}"),
                kind="crown",
                embedding=record.embedding,
                terms=selected_terms,
                vault_id=f"vault:{semantic_id}",
                created_pulse=pulse,
                last_active_pulse=pulse,
            )
        )
        self.store.update_concept_embedding(
            semantic_id, record.embedding, terms=selected_terms
        )
        edge = self._ensure_edge(
            GraphSide.INPUT,
            child_node_id,
            semantic_id,
            delta_y=1.0,
            created_pulse=pulse,
        )
        self.store.add_to_vault(semantic.vault_id, record.record_id, semantic_id)
        self.store.add_edge_evidence(edge.edge_id, record.record_id)
        self.store.put_overlap_cluster(
            OverlapCluster(
                cluster_id=cluster.cluster_id,
                parent_node_id=cluster.parent_node_id,
                centroid=cluster.centroid,
                record_ids=cluster.record_ids,
                experience_ids=cluster.experience_ids,
                preference_mean=cluster.preference_mean,
                confidence_mean=cluster.confidence_mean,
                first_pulse=cluster.first_pulse,
                last_pulse=max(cluster.last_pulse, pulse),
                child_node_id=child_node_id,
                semantic_node_id=semantic_id,
            )
        )
        return semantic

    def stage_growth(
        self,
        record: MemoryRecord,
        *,
        input_trunk: InputTrunk,
        pulse: int,
        parent_node_id: str | None = None,
        promotion_count: int | None = None,
        overlap_threshold: float | None = None,
        preferred_cluster_id: str | None = None,
        semantic_port: bool = True,
    ) -> str | None:
        experience_id = self._experience_id(record)
        state = self.store.get_experience_state(experience_id)
        preference = state.preference_mean if state else 0.0
        confidence = min(1.0, state.preference_weight) if state else 0.0
        parent_id = parent_node_id or PREFERENCE_NODE_IDS[
            (input_trunk, self._preference_band(preference, confidence))
        ]
        threshold = (
            self.growth_overlap_threshold
            if overlap_threshold is None
            else max(0.0, min(1.0, overlap_threshold))
        )
        if promotion_count is None:
            vault_experiences = int(
                self.store.lower_vault_stats(parent_id)["experience_count"]
            )
            required = max(
                self.growth_promotion_count,
                int(math.ceil(math.log2(max(2, vault_experiences + 1)))),
            )
        else:
            required = max(2, promotion_count)
        clusters = self.store.list_overlap_clusters(parent_id)
        if preferred_cluster_id is not None:
            clusters = [
                cluster for cluster in clusters if cluster.cluster_id == preferred_cluster_id
            ]
        compatible: list[tuple[float, OverlapCluster]] = []
        for cluster in clusters:
            similarity = cosine_similarity(record.embedding, cluster.centroid)
            if similarity < threshold:
                continue
            if abs(preference - cluster.preference_mean) > self.growth_preference_tolerance:
                continue
            compatible.append((similarity, cluster))

        if preferred_cluster_id is not None and not compatible:
            if not clusters:
                return None
            return clusters[0].semantic_node_id or clusters[0].child_node_id

        if compatible:
            _, cluster = max(compatible, key=lambda item: (item[0], item[1].cluster_id))
            if experience_id in cluster.experience_ids:
                return cluster.semantic_node_id or cluster.child_node_id
            old_count = len(cluster.experience_ids)
            centroid = [
                (old * old_count + new) / (old_count + 1)
                for old, new in zip(cluster.centroid, record.embedding)
            ]
            norm = math.sqrt(sum(value * value for value in centroid)) or 1.0
            centroid = [value / norm for value in centroid]
            record_ids = (*cluster.record_ids, record.record_id)
            experience_ids = (*cluster.experience_ids, experience_id)
            preference_mean = (cluster.preference_mean * old_count + preference) / (old_count + 1)
            confidence_mean = (cluster.confidence_mean * old_count + confidence) / (old_count + 1)
            updated = OverlapCluster(
                cluster_id=cluster.cluster_id,
                parent_node_id=parent_id,
                centroid=as_tuple(centroid),
                record_ids=record_ids,
                experience_ids=experience_ids,
                preference_mean=preference_mean,
                confidence_mean=confidence_mean,
                first_pulse=cluster.first_pulse,
                last_pulse=pulse,
                child_node_id=cluster.child_node_id,
                semantic_node_id=cluster.semantic_node_id,
            )
        else:
            digest = hashlib.sha256(
                f"{parent_id}|{experience_id}".encode("utf-8")
            ).hexdigest()[:20]
            updated = OverlapCluster(
                cluster_id=f"overlap:{digest}",
                parent_node_id=parent_id,
                centroid=record.embedding,
                record_ids=(record.record_id,),
                experience_ids=(experience_id,),
                preference_mean=preference,
                confidence_mean=confidence,
                first_pulse=pulse,
                last_pulse=pulse,
            )
        self.store.put_overlap_cluster(updated)
        if len(updated.experience_ids) < required:
            return None

        if semantic_port and any(
            not self._record_has_membrane_words(item)
            for item in self.store.get_records(updated.record_ids)
        ):
            raise ValueError("semantic promotion requires HEAR membrane word evidence")

        records = self.store.get_records(updated.record_ids)
        terms: list[str] = []
        if semantic_port:
            term_counts: dict[str, int] = {}
            for supporting_record in records:
                for term in self._growth_terms(supporting_record.text):
                    term_counts[term] = term_counts.get(term, 0) + 1
            terms = [
                term
                for term, _ in sorted(
                    term_counts.items(), key=lambda item: (-item[1], item[0])
                )
            ]
        digest = updated.cluster_id.rsplit(":", 1)[-1]
        child_id = updated.child_node_id or f"child:auto:{digest}"
        semantic_id = (
            updated.semantic_node_id or f"concept:auto:{digest}"
            if semantic_port
            else None
        )
        child = self.store.add_concept(
            ConceptNode(
                concept_id=child_id,
                label=f"pattern:{digest}",
                kind="child",
                embedding=as_tuple([0.0] * self.embedder.dimension),
                terms=(),
                vault_id=f"lower-vault:{child_id}",
                created_pulse=pulse,
                last_active_pulse=pulse,
            )
        )
        if child.vault_id != f"lower-vault:{child_id}":
            self.store.set_concept_vault(child_id, f"lower-vault:{child_id}")
        semantic = None
        if semantic_id is not None:
            semantic = self.store.add_concept(
                ConceptNode(
                    concept_id=semantic_id,
                    label=(" ".join(terms[:2]).title() or f"Concept {digest[:8]}"),
                    kind="crown",
                    embedding=updated.centroid,
                    terms=tuple(terms),
                    vault_id=f"vault:{semantic_id}",
                    created_pulse=pulse,
                    last_active_pulse=pulse,
                )
            )
            self.store.update_concept_embedding(
                semantic_id, updated.centroid, terms=terms
            )
        parent_edge = self._ensure_edge(
            GraphSide.INPUT, parent_id, child_id, delta_y=1.0, created_pulse=pulse
        )
        semantic_edge = (
            self._ensure_edge(
                GraphSide.INPUT,
                child_id,
                semantic_id,
                delta_y=1.0,
                created_pulse=pulse,
            )
            if semantic_id is not None
            else None
        )
        for supporting_record in records:
            supporting_experience = self._experience_id(supporting_record)
            supporting_state = self.store.get_experience_state(supporting_experience)
            supporting_preference = supporting_state.preference_mean if supporting_state else 0.0
            supporting_confidence = (
                min(1.0, supporting_state.preference_weight) if supporting_state else 0.0
            )
            self.store.add_to_vault(child.vault_id, supporting_record.record_id, child_id)
            if semantic is not None:
                self.store.add_to_vault(
                    semantic.vault_id, supporting_record.record_id, semantic.concept_id
                )
            self.store.add_experience_projection(
                ExperienceProjection(
                    experience_id=supporting_experience,
                    record_id=supporting_record.record_id,
                    node_id=child_id,
                    layer=3,
                    side=GraphSide.INPUT,
                    activation=1.0,
                    preference=supporting_preference,
                    confidence=supporting_confidence,
                    pulse=pulse,
                    metadata={"projection": "emergent_child"},
                )
            )
            self.store.add_edge_evidence(parent_edge.edge_id, supporting_record.record_id)
            if semantic_edge is not None:
                self.store.add_edge_evidence(
                    semantic_edge.edge_id, supporting_record.record_id
                )
        promoted = OverlapCluster(
            cluster_id=updated.cluster_id,
            parent_node_id=updated.parent_node_id,
            centroid=updated.centroid,
            record_ids=updated.record_ids,
            experience_ids=updated.experience_ids,
            preference_mean=updated.preference_mean,
            confidence_mean=updated.confidence_mean,
            first_pulse=updated.first_pulse,
            last_pulse=updated.last_pulse,
            child_node_id=child_id,
            semantic_node_id=semantic_id,
        )
        self.store.put_overlap_cluster(promoted)
        return semantic_id or child_id
