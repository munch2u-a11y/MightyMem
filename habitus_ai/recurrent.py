from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from typing import Iterable, Mapping

from .graph import GraphRuntime
from .store import MindStore
from .types import (
    DesireActivation,
    GraphEdge,
    GraphSide,
    NodeDynamics,
    RecurrentSnapshot,
)


DESIRE_KINDS = frozenset({"drive", "desire", "desire_composite"})


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))


class RecurrentField:
    """Persistent activation and desire dynamics inside the Habitus graph.

    This is the working state used by the no-history agent. It never stores a
    transcript, a summary, or an instruction string. Durable tendencies remain
    graph edges; this field adds the changing state that those weights act on.
    """

    def __init__(
        self,
        store: MindStore,
        graph: GraphRuntime,
        *,
        propagation_gain: float = 0.42,
        propagation_steps: int = 2,
        activation_floor: float = 1e-5,
        refractory_pulses: int = 2,
    ) -> None:
        self.store = store
        self.graph = graph
        self.propagation_gain = _clamp(propagation_gain)
        self.propagation_steps = max(0, int(propagation_steps))
        self.activation_floor = max(0.0, float(activation_floor))
        self.refractory_pulses = max(0, int(refractory_pulses))

    def register(
        self,
        node_id: str,
        *,
        pressure: float = 0.0,
        baseline_growth: float = 0.0,
        persistence: float = 0.65,
        satisfaction_gain: float = 0.70,
        frustration_gain: float = 0.45,
        expression_threshold: float = 1.0,
        pulse: int = 0,
    ) -> NodeDynamics:
        """Make one existing graph node part of the recurrent open-weight field."""
        existing = self.store.get_node_dynamics(node_id)
        if existing is not None:
            return existing
        return self.store.put_node_dynamics(
            NodeDynamics(
                node_id=node_id,
                pressure=pressure,
                baseline_growth=baseline_growth,
                persistence=persistence,
                satisfaction_gain=satisfaction_gain,
                frustration_gain=frustration_gain,
                expression_threshold=expression_threshold,
                last_pulse=pulse,
            )
        )

    def _default_state(self, node_id: str, pulse: int) -> NodeDynamics:
        return NodeDynamics(node_id=node_id, expression_threshold=2.0, last_pulse=pulse)

    @staticmethod
    def _state_digest(states: Iterable[NodeDynamics]) -> str:
        payload = [
            {
                "node_id": state.node_id,
                "activation": format(state.activation, ".17g"),
                "pressure": format(state.pressure, ".17g"),
                "valence": format(state.valence, ".17g"),
                "momentum": format(state.momentum, ".17g"),
                "baseline_growth": format(state.baseline_growth, ".17g"),
                "persistence": format(state.persistence, ".17g"),
                "satisfaction_gain": format(state.satisfaction_gain, ".17g"),
                "frustration_gain": format(state.frustration_gain, ".17g"),
                "expression_threshold": format(state.expression_threshold, ".17g"),
                "last_pulse": state.last_pulse,
                "last_selected_pulse": state.last_selected_pulse,
            }
            for state in sorted(states, key=lambda item: item.node_id)
        ]
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def _desires(
        self,
        states: Iterable[NodeDynamics],
        *,
        pulse: int,
    ) -> tuple[DesireActivation, ...]:
        result = []
        desire_node_ids = self.store.concept_ids_by_kind(DESIRE_KINDS)
        for state in states:
            if state.node_id not in desire_node_ids:
                continue
            urgency = min(
                2.0,
                state.pressure
                + 0.35 * state.activation
                + 0.15 * max(0.0, state.momentum)
                + 0.10 * max(0.0, -state.valence),
            )
            refractory = (
                state.last_selected_pulse is None
                or pulse - state.last_selected_pulse >= self.refractory_pulses
            )
            result.append(
                DesireActivation(
                    node_id=state.node_id,
                    activation=state.activation,
                    pressure=state.pressure,
                    valence=state.valence,
                    momentum=state.momentum,
                    urgency=urgency,
                    expression_threshold=state.expression_threshold,
                    ready=refractory and urgency >= state.expression_threshold,
                )
            )
        return tuple(sorted(result, key=lambda item: (-item.urgency, item.node_id)))

    def snapshot(self, *, pulse: int) -> RecurrentSnapshot:
        states = tuple(self.store.list_node_dynamics())
        desires = self._desires(states, pulse=pulse)
        dominant = desires[0].node_id if desires else None
        return RecurrentSnapshot(
            pulse=int(pulse),
            node_states=states,
            desires=desires,
            dominant_desire_id=dominant,
            should_express=any(item.ready for item in desires),
            state_sha256=self._state_digest(states),
        )

    def activation_values(
        self,
        snapshot: RecurrentSnapshot,
        *,
        minimum: float = 1e-5,
    ) -> dict[str, float]:
        """Return the only transient pull used by target-free output routing."""
        desire_state = {item.node_id: item for item in snapshot.desires}
        any_ready = any(item.ready for item in snapshot.desires)
        result = {}
        for state in snapshot.node_states:
            desire = desire_state.get(state.node_id)
            if desire is not None:
                # Pressure-mediated urgency, not lingering activation, chooses
                # among desires. Once one ready drive exists, refractory
                # desires remain present but cannot immediately monopolize the
                # same motor route again.
                value = min(1.0, desire.urgency)
                if any_ready and not desire.ready:
                    value *= 0.25
            else:
                value = state.activation
            if value >= minimum:
                result[state.node_id] = value
        return result

    def advance(
        self,
        *,
        pulse: int,
        pulse_id: str,
        excitation: Mapping[str, float] | None = None,
        pressure_delta: Mapping[str, float] | None = None,
        selected_node_ids: Iterable[str] = (),
    ) -> RecurrentSnapshot:
        """Advance one recurrent pulse without consulting stored language."""
        pulse = max(0, int(pulse))
        excitation = {str(key): float(value) for key, value in (excitation or {}).items()}
        pressure_delta = {
            str(key): float(value) for key, value in (pressure_delta or {}).items()
        }
        selected = {str(node_id) for node_id in selected_node_ids}
        states = {state.node_id: state for state in self.store.list_node_dynamics()}
        for node_id in (*excitation, *pressure_delta, *selected):
            if not self.store.has_concept(node_id):
                raise KeyError(f"unknown recurrent excitation node: {node_id}")
            states.setdefault(node_id, self._default_state(node_id, pulse))
        desire_node_ids = self.store.concept_ids_by_kind(DESIRE_KINDS)

        working: dict[str, NodeDynamics] = {}
        old_activation: dict[str, float] = {}
        for node_id, state in states.items():
            elapsed = max(0, pulse - state.last_pulse)
            retention = state.persistence ** elapsed
            activation = state.activation * retention
            pressure = state.pressure
            if elapsed and state.baseline_growth:
                pressure = 1.0 - (1.0 - pressure) * (
                    (1.0 - state.baseline_growth) ** elapsed
                )
            pressure = _clamp(pressure + pressure_delta.get(node_id, 0.0))
            direct = _clamp(excitation.get(node_id, 0.0), -1.0, 1.0)
            if direct >= 0.0:
                activation = 1.0 - (1.0 - activation) * (1.0 - direct)
            else:
                activation *= 1.0 + direct
            if node_id in desire_node_ids:
                activation = max(activation, 0.65 * pressure)
            old_activation[node_id] = state.activation
            working[node_id] = replace(
                state,
                activation=_clamp(activation),
                pressure=pressure,
                valence=state.valence * retention,
                last_pulse=pulse,
            )

        frontier = {
            node_id: max(0.0, state.activation)
            for node_id, state in working.items()
            if state.activation >= self.activation_floor
        }
        local_weights = self.graph.edge_local_weights()
        all_edges = self.store.list_edges()
        outgoing_by_side: dict[tuple[GraphSide, str], list[GraphEdge]] = {}
        for edge in all_edges:
            # Lexical nodes are ordinary recurrent graph states.  Allowing
            # activation to enter and leave them is what lets an actualized
            # word alter the competition for the following word.  They carry
            # geometry, never a hidden text transcript. Edge targets are
            # guaranteed by the store's foreign key, so reloading and decoding
            # a full concept embedding for every edge is both redundant and a
            # severe cost as developmental breadth grows.
            outgoing_by_side.setdefault((edge.side, edge.source_id), []).append(edge)
        for depth in range(self.propagation_steps):
            contributions: dict[str, float] = {}
            attenuation = self.propagation_gain ** (depth + 1)
            for source_id, source_activation in frontier.items():
                for side in GraphSide:
                    outgoing = outgoing_by_side.get((side, source_id), ())
                    eligible_mass = sum(
                        local_weights.get(edge.edge_id, 0.0)
                        for edge in outgoing
                    )
                    if eligible_mass <= 0.0:
                        continue
                    for edge in outgoing:
                        conditional = (
                            local_weights.get(edge.edge_id, 0.0)
                            / eligible_mass
                        )
                        value = source_activation * conditional * attenuation
                        if value >= self.activation_floor:
                            previous = contributions.get(edge.target_id, 0.0)
                            contributions[edge.target_id] = 1.0 - (
                                (1.0 - previous) * (1.0 - _clamp(value))
                            )
            frontier = {}
            for node_id, value in contributions.items():
                state = working.setdefault(node_id, self._default_state(node_id, pulse))
                activation = 1.0 - (1.0 - state.activation) * (1.0 - _clamp(value))
                working[node_id] = replace(state, activation=_clamp(activation), last_pulse=pulse)
                frontier[node_id] = value
            if not frontier:
                break

        for node_id, state in working.items():
            previous = old_activation.get(node_id, 0.0)
            updated = replace(
                state,
                momentum=_clamp(state.activation - previous, -1.0, 1.0),
                last_selected_pulse=(
                    pulse if node_id in selected else state.last_selected_pulse
                ),
            )
            self.store.put_node_dynamics(updated)

        result = self.snapshot(pulse=pulse)
        self.store.save_recurrent_pulse(
            pulse_id=pulse_id,
            pulse=pulse,
            excitation=excitation,
            pressure_delta=pressure_delta,
            dominant_desire_id=result.dominant_desire_id,
            should_express=result.should_express,
            state_sha256=result.state_sha256,
        )
        return result

    def mark_selected(self, node_ids: Iterable[str], *, pulse: int) -> None:
        for node_id in dict.fromkeys(str(item) for item in node_ids):
            state = self.store.get_node_dynamics(node_id)
            if state is not None:
                self.store.put_node_dynamics(
                    replace(state, last_selected_pulse=int(pulse), last_pulse=int(pulse))
                )

    def observe_outcome(
        self,
        edge_ids: Iterable[str],
        *,
        stability_delta: float,
        verified: bool,
        pulse: int,
    ) -> RecurrentSnapshot:
        """Relieve or frustrate only desire nodes on the receipted output path."""
        if not verified:
            return self.snapshot(pulse=pulse)
        node_ids = set()
        for edge_id in dict.fromkeys(str(item) for item in edge_ids):
            edge = self.store.get_edge(edge_id)
            if edge is not None:
                node_ids.update((edge.source_id, edge.target_id))
        delta = _clamp(stability_delta, -1.0, 1.0)
        magnitude = abs(delta)
        desire_node_ids = self.store.concept_ids_by_kind(DESIRE_KINDS)
        for node_id in node_ids:
            state = self.store.get_node_dynamics(node_id)
            if state is None:
                continue
            pressure = state.pressure
            if node_id in desire_node_ids:
                if delta > 0.0:
                    pressure *= 1.0 - state.satisfaction_gain * magnitude
                elif delta < 0.0:
                    pressure += (1.0 - pressure) * state.frustration_gain * magnitude
            valence = _clamp(0.60 * state.valence + 0.40 * delta, -1.0, 1.0)
            self.store.put_node_dynamics(
                replace(
                    state,
                    pressure=_clamp(pressure),
                    activation=max(state.activation, 0.50 * _clamp(pressure)),
                    valence=valence,
                    last_pulse=int(pulse),
                )
            )
        return self.snapshot(pulse=pulse)
