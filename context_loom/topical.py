from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence


_EXCLUDED_NODES = frozenset({
    "SELF",
    "self_root",
    "IN:HEAR",
    "IN:SEE",
    "IN:NOTICE",
    "OUT:SPEAK",
    "OUT:LOOK",
    "OUT:DO",
})


@dataclass(frozen=True)
class TopicalFrame:
    focus_topics: tuple[str, ...] = ()
    momentum: dict[str, str] = field(default_factory=dict)
    drift: float = 0.0
    familiarity: dict[str, float] = field(default_factory=dict)
    action_mode_mix: dict[str, float] = field(default_factory=dict)
    max_depth: int = 1
    rendered: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "focus_topics": list(self.focus_topics),
            "momentum": dict(self.momentum),
            "drift": round(self.drift, 4),
            "familiarity": {k: round(v, 4) for k, v in self.familiarity.items()},
            "action_mode_mix": {k: round(v, 4) for k, v in self.action_mode_mix.items()},
            "max_depth": self.max_depth,
            "rendered": self.rendered,
        }


class TopicalTracker:
    """Conserved decaying activation field over concept nodes derived from Y-path trajectories."""

    def __init__(
        self,
        store: Any,
        *,
        decay: float = 0.70,
        tau: float = 1.0,
        max_focus: int = 3,
    ) -> None:
        self.store = store
        self.decay = max(0.0, min(1.0, float(decay)))
        self.tau = max(0.1, float(tau))
        self.max_focus = max(1, int(max_focus))
        
        # node_id -> activation weight
        self._activations: dict[str, float] = {}
        # node_id -> concept display name
        self._concept_names: dict[str, float] = {}
        # previous turn activations for momentum tracking
        self._prev_activations: dict[str, float] = {}
        
        # Action mode counters (SPEAK, LOOK, DO)
        self._action_counts: dict[str, int] = {"SPEAK": 0, "LOOK": 0, "DO": 0}
        self._total_actions: int = 0
        self._current_drift: float = 0.0
        self._max_depth: int = 1

    def _is_eligible_node(self, node_id: str) -> bool:
        if node_id in _EXCLUDED_NODES or node_id.startswith("PREF:"):
            return False
        return True

    def observe_turn(
        self,
        y_paths: Iterable[Any],
        output_trunk: str | None = None,
    ) -> TopicalFrame:
        # 1. Save prior state for momentum & drift
        self._prev_activations = dict(self._activations)
        
        # 2. Decay existing activations
        for nid in list(self._activations):
            self._activations[nid] *= self.decay
            if self._activations[nid] < 1e-4:
                del self._activations[nid]

        # 3. Process turn Y-paths
        turn_weights: dict[str, float] = {}
        for path in y_paths:
            if isinstance(path, dict):
                target_id = str(path.get("target_concept_id") or path.get("target") or "")
                target_name = str(path.get("target_concept_name") or target_id)
                score = float(path.get("endpoint_score") or path.get("total_score") or 0.8)
                nodes = list(path.get("path_nodes") or path.get("nodes") or [target_id])
            else:
                target_id = getattr(path, "target_concept_id", getattr(path, "target_node_id", ""))
                target_name = getattr(path, "target_concept_name", target_id)
                score = float(getattr(path, "endpoint_score", getattr(path, "total_score", 0.8)))
                nodes = list(getattr(path, "path_nodes", getattr(path, "node_ids", [target_id])))

            depth = max(1, len(nodes) - 1)
            if depth > self._max_depth:
                self._max_depth = depth

            for nid in nodes:
                if not self._is_eligible_node(nid):
                    continue
                name = target_name if nid == target_id else nid
                # Travel time attenuation
                w = score * math.exp(-1.0 / self.tau)
                turn_weights[nid] = turn_weights.get(nid, 0.0) + w
                if nid not in self._concept_names or name != nid:
                    self._concept_names[nid] = name

        # 4. Normalize turn weights and add to conserved activations
        total_turn = sum(turn_weights.values())
        if total_turn > 0.0:
            for nid, weight in turn_weights.items():
                normalized = weight / total_turn
                self._activations[nid] = self._activations.get(nid, 0.0) + (1.0 - self.decay) * normalized

        # 5. Renormalize total activation sum <= 1.0
        total_act = sum(self._activations.values())
        if total_act > 1.0:
            for nid in self._activations:
                self._activations[nid] /= total_act

        # 6. Track action mode mix
        if output_trunk in self._action_counts:
            self._action_counts[output_trunk] += 1
            self._total_actions += 1

        # 7. Calculate drift (cosine distance between current and prev activation distributions)
        self._current_drift = self._compute_drift()

        return self.frame()

    def _compute_drift(self) -> float:
        all_keys = set(self._activations.keys()) | set(self._prev_activations.keys())
        if not all_keys or not self._prev_activations:
            return 0.0
        v1 = [self._activations.get(k, 0.0) for k in all_keys]
        v2 = [self._prev_activations.get(k, 0.0) for k in all_keys]
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        if norm1 <= 1e-9 or norm2 <= 1e-9:
            return 0.0
        sim = dot / (norm1 * norm2)
        return max(0.0, min(1.0, 1.0 - sim))

    def frame(self) -> TopicalFrame:
        if not self._activations:
            return TopicalFrame()

        ranked = sorted(
            self._activations.items(),
            key=lambda item: -item[1],
        )[: self.max_focus]

        focus_names = tuple(self._concept_names.get(nid, nid) for nid, _ in ranked)
        momentum: dict[str, str] = {}
        familiarity: dict[str, float] = {}

        for nid, act in ranked:
            name = self._concept_names.get(nid, nid)
            prev_act = self._prev_activations.get(nid, 0.0)
            delta = act - prev_act
            if delta > 0.05:
                status = "rising"
            elif delta < -0.05:
                status = "fading"
            else:
                status = "sustained"
            momentum[name] = status
            familiarity[name] = min(1.0, round(act * 1.5, 4))

        mix: dict[str, float] = {}
        if self._total_actions > 0:
            mix = {
                trunk: round(count / self._total_actions, 4)
                for trunk, count in self._action_counts.items()
                if count > 0
            }

        lines = ["[TOPICAL ORIENTATION & DRIFT]"]
        lines.append(f"Focus topics: {', '.join(focus_names)}")
        mom_str = ", ".join(f"{topic} ({status})" for topic, status in momentum.items())
        lines.append(f"Momentum: {mom_str}")
        lines.append(f"Conversational drift: {round(self._current_drift, 4)}")
        if mix:
            mix_str = ", ".join(f"{trunk}: {pct * 100:.0f}%" for trunk, pct in mix.items())
            lines.append(f"Action mode mix: {mix_str}")

        rendered = "\n".join(lines)
        return TopicalFrame(
            focus_topics=focus_names,
            momentum=momentum,
            drift=round(self._current_drift, 4),
            familiarity=familiarity,
            action_mode_mix=mix,
            max_depth=self._max_depth,
            rendered=rendered,
        )
