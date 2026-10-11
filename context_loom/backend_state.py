"""Extract the agent's own internal state vector from a Habitus pulse.

The embedding intent cortex still reads the user's words -- it embeds the
utterance. This module reads nothing the user typed. It takes the
``DevelopmentalPulseReceipt`` the runtime produces when it perceives an input
and turns the cortex's own dynamics into a fixed-width vector:

  graph_field   the runtime's own field vector over the graph
  activation    short-lived working state per node
  pressure      unmet endogenous tendency per node, which accumulates without
                any new external message
  valence       per node
  momentum      per node
  desires       activation / pressure / urgency of each desire node

These are recurrent state, not text features. ``pressure`` in particular has no
counterpart in the utterance at all -- it can rise across turns while the user
says nothing relevant, which is what makes an action fired from it genuinely
endogenous rather than a paraphrase of the last sentence.

Node order is pinned by a sorted node-id vocabulary captured on first use, so
successive turns yield comparable vectors even as the graph grows.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

NODE_CHANNELS = ("activation", "pressure", "valence", "momentum")
DESIRE_CHANNELS = ("activation", "pressure", "urgency")


@dataclass
class BackendStateVector:
    pulse: int
    graph_field: tuple[float, ...]
    node_ids: tuple[str, ...]
    channels: dict[str, tuple[float, ...]]
    desire_ids: tuple[str, ...]
    desire_channels: dict[str, tuple[float, ...]]
    dominant_desire_id: str | None
    should_express: bool
    selected_outputs: tuple[str, ...]
    route_ids: tuple[str, ...]

    def as_features(self, *, include_graph_field: bool = True) -> list[float]:
        values: list[float] = []
        if include_graph_field:
            values.extend(self.graph_field)
        for channel in NODE_CHANNELS:
            values.extend(self.channels.get(channel, ()))
        for channel in DESIRE_CHANNELS:
            values.extend(self.desire_channels.get(channel, ()))
        return [0.0 if v is None or not math.isfinite(v) else float(v) for v in values]

    def summary(self) -> dict[str, Any]:
        act = self.channels.get("activation", ())
        pres = self.channels.get("pressure", ())
        return {
            "pulse": self.pulse,
            "graph_field_dim": len(self.graph_field),
            "nodes": len(self.node_ids),
            "desires": len(self.desire_ids),
            "activation_sum": float(sum(act)),
            "activation_max": float(max(act)) if act else 0.0,
            "pressure_sum": float(sum(pres)),
            "pressure_max": float(max(pres)) if pres else 0.0,
            "dominant_desire_id": self.dominant_desire_id,
            "should_express": self.should_express,
            "selected_outputs": list(self.selected_outputs),
        }


class BackendStateExtractor:
    """Turns pulses into aligned vectors against a growing node vocabulary."""

    def __init__(self) -> None:
        self.node_vocabulary: list[str] = []
        self.desire_vocabulary: list[str] = []

    def _align(self, vocabulary: list[str], observed: dict[str, dict[str, float]],
               channels: Sequence[str]) -> tuple[tuple[str, ...], dict[str, tuple[float, ...]]]:
        for node_id in observed:
            if node_id not in vocabulary:
                vocabulary.append(node_id)
        vocabulary.sort()
        out: dict[str, list[float]] = {c: [] for c in channels}
        for node_id in vocabulary:
            row = observed.get(node_id, {})
            for channel in channels:
                out[channel].append(float(row.get(channel, 0.0)))
        return tuple(vocabulary), {c: tuple(v) for c, v in out.items()}

    def extract(self, pulse) -> BackendStateVector:
        recurrent = pulse.cycle.recurrent
        observed_nodes = {
            state.node_id: {
                "activation": getattr(state, "activation", 0.0),
                "pressure": getattr(state, "pressure", 0.0),
                "valence": getattr(state, "valence", 0.0),
                "momentum": getattr(state, "momentum", 0.0),
            }
            for state in recurrent.node_states
        }
        node_ids, channels = self._align(self.node_vocabulary, observed_nodes, NODE_CHANNELS)

        observed_desires = {
            d.node_id: {
                "activation": getattr(d, "activation", 0.0),
                "pressure": getattr(d, "pressure", 0.0),
                "urgency": getattr(d, "urgency", 0.0),
            }
            for d in recurrent.desires
        }
        desire_ids, desire_channels = self._align(
            self.desire_vocabulary, observed_desires, DESIRE_CHANNELS)

        selected = tuple(
            f"{item.trunk.value}:{item.node_id}"
            for item in getattr(pulse.cycle, "selected_outputs", ()) or ())

        return BackendStateVector(
            pulse=int(getattr(recurrent, "pulse", 0)),
            graph_field=tuple(float(v) for v in (getattr(pulse, "graph_field", ()) or ())),
            node_ids=node_ids,
            channels=channels,
            desire_ids=desire_ids,
            desire_channels=desire_channels,
            dominant_desire_id=getattr(recurrent, "dominant_desire_id", None),
            should_express=bool(getattr(recurrent, "should_express", False)),
            selected_outputs=selected,
            route_ids=tuple(getattr(pulse, "route_ids", ()) or ()),
        )
