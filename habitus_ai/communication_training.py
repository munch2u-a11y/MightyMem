"""From-scratch, multi-turn language training and held-out communication checks.

The bundled curriculum is a controlled small-world exercise, not a natural
conversation corpus. Its oracle is training/evaluation-only: the conversation
runner cannot access it. No pretrained teacher or language judge is used.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import tarfile
import time
from typing import Any, Sequence

from .developmental_cortex import CortexConfig, DIRECTION_IDS, require_torch, torch
from .stateful_language import (
    LANGUAGE_INPUT_SKIPS, SENSORY_READOUT_MODES, SENSORY_STATE_POLICIES, StatefulConversation, language_field, language_model_sha256, language_symbols,
    load_language_checkpoint, new_language_core, save_language_checkpoint,
)
from .language_vocabulary import LanguageVocabulary
from .communication_objectives import PREDICTION_NORMALIZATIONS, prediction_average
from .communication_sampling import SAMPLING_POLICIES, DialogueSampler
from .grounded_language import GroundedConversation
from .workspace_observations import SENSOR_SCHEMA, decode_observation
from .workspace_cortex import sensory_transition


LANGUAGE_SOURCE_FILES = (
    "communication_training.py", "stateful_language.py", "developmental_cortex.py",
    "language_vocabulary.py", "lexical_cortex.py",
    "communication_objectives.py",
    "communication_sampling.py",
    "grounded_language.py", "workspace_observations.py", "workspace_cortex.py", "sensory_reservoir.py",
)


@dataclass(frozen=True)
class TeachingTurn:
    user: str
    assistant: str
    scored: bool = True
    observations_before: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.observations_before, tuple):
            raise ValueError("teaching observations must be an immutable tuple of encoded receipts")
        for encoded in self.observations_before:
            if not isinstance(encoded, str):
                raise ValueError("teaching observations must be canonical hexadecimal bytes")
            payload = bytes.fromhex(encoded)
            if payload.hex() != encoded:
                raise ValueError("teaching observation encoding is not canonical")
            decode_observation(payload)


@dataclass(frozen=True)
class TeachingDialogue:
    dialogue_id: str
    split: str
    scenario: str
    turns: tuple[TeachingTurn, ...]
    provenance: dict[str, Any] = field(default_factory=dict)


def controlled_curriculum() -> tuple[TeachingDialogue, ...]:
    """Hold out complete object/color/location combinations before training."""
    objects = ("key", "cup", "ball", "book")
    colors = ("red", "blue", "green", "yellow")
    locations = ("box", "drawer", "desk", "shelf")
    dialogues = []
    for oi, obj in enumerate(objects):
        for ci, color in enumerate(colors):
            for li, location in enumerate(locations):
                partition = (oi + 3 * ci + 5 * li) % 8
                split = "test" if partition == 0 else "validation" if partition == 1 else "train"
                description = f"The {color} {obj} is in the {location}."
                fact = f"The {obj} is {color}."
                place = f"The {obj} is in the {location}."
                other = objects[(oi + 1) % len(objects)]
                base = f"{obj}-{color}-{location}"
                variants = [
                    ("describe", (
                        TeachingTurn(f"Describe a {color} {obj} in the {location}.", description),
                    )),
                    ("retain", (
                        TeachingTurn(fact, "Okay.", False),
                        TeachingTurn(place, "Okay.", False),
                        TeachingTurn(f"Tell me about the {obj}.", description),
                    )),
                    ("questions", (
                        TeachingTurn(fact, "Okay.", False),
                        TeachingTurn(place, "Okay.", False),
                        TeachingTurn(f"What color is the {obj}?", fact),
                        TeachingTurn(f"Where is the {obj}?", place),
                        TeachingTurn(f"Tell me about the {obj}.", description),
                    )),
                ]
                # Fully cross nuisance colors with the target scene. A
                # distractor color must not encode the desired answer, and a
                # previous color must not uniquely identify its correction.
                variants.extend((f"distractor:{distractor_color}", (
                    TeachingTurn(fact, "Okay.", False),
                    TeachingTurn(place, "Okay.", False),
                    TeachingTurn(f"The {other} is {distractor_color}.", "Okay.", False),
                    TeachingTurn(f"Tell me about the {obj}.", description),
                )) for distractor_color in colors)
                variants.extend((f"correction:{old_color}", (
                    TeachingTurn(f"The {obj} is {old_color}.", "Okay.", False),
                    TeachingTurn(place, "Okay.", False),
                    TeachingTurn(f"Actually, the {obj} is {color}.", "Okay.", False),
                    TeachingTurn(f"Tell me about the {obj}.", description),
                )) for old_color in colors if old_color != color)
                for variant, turns in variants:
                    scenario = variant.split(":", 1)[0]
                    dialogues.append(TeachingDialogue(f"{base}:{variant}", split, scenario, turns))
    # Absence of evidence has its own target. These use disjoint dialogue IDs;
    # held-out composition is measured in the scene-based cases above.
    for oi, obj in enumerate(objects):
        dialogues.append(TeachingDialogue(
            f"unknown:{obj}", "train", "unknown",
            (TeachingTurn(f"What color is the {obj}?", f"I do not know the color of the {obj}."),),
        ))
    dialogues.append(TeachingDialogue(
        "greeting", "train", "greeting", (
            TeachingTurn("Hello.", "Hello."),
            TeachingTurn("Thank you.", "You are welcome."),
        ),
    ))
    return tuple(dialogues)


def load_curriculum(path: Path) -> tuple[TeachingDialogue, ...]:
    dialogues = []
    seen = set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            item = TeachingDialogue(
                str(row["dialogue_id"]), str(row["split"]), str(row["scenario"]),
                tuple(TeachingTurn(**{**turn, "observations_before": tuple(turn.get("observations_before", ()))})
                      for turn in row["turns"]),
                provenance=dict(row.get("provenance", {})),
            )
            if item.dialogue_id in seen or item.split not in {"train", "validation", "test"}:
                raise ValueError("curriculum requires unique IDs and explicit train/validation/test splits")
            if not item.turns:
                raise ValueError("teaching dialogues cannot be empty")
            seen.add(item.dialogue_id)
            dialogues.append(item)
    return tuple(dialogues)


def teaching_payload(dialogue: TeachingDialogue) -> dict[str, Any]:
    """Keep historical text-only corpus identities unchanged."""
    result = asdict(dialogue)
    for turn in result["turns"]:
        if not turn["observations_before"]:
            del turn["observations_before"]
    return result


def curriculum_hash(dialogues: Sequence[TeachingDialogue]) -> str:
    encoded = json.dumps([teaching_payload(item) for item in dialogues], sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_curriculum(
    dialogues: Sequence[TeachingDialogue],
    config: CortexConfig,
    maximum_sequence_bytes: int | None,
) -> None:
    """Reject an incompatible corpus before starting a run or updating weights."""
    for dialogue in dialogues:
        if not dialogue.turns:
            raise ValueError("teaching dialogues cannot be empty")
        length = 0
        for turn in dialogue.turns:
            for encoded in turn.observations_before:
                size = len(bytes.fromhex(encoded))
                length += size + 1
            for text in (turn.user, turn.assistant):
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("teaching events must contain nonempty text")
                size = len(text.encode("utf-8"))
                length += size + 1
        if maximum_sequence_bytes is not None and length > maximum_sequence_bytes:
            raise ValueError(f"dialogue exceeds the explicit training unroll limit: {dialogue.dialogue_id}")


def evaluation_subset(dialogues: Sequence[TeachingDialogue], maximum_dialogues: int) -> tuple[TeachingDialogue, ...]:
    """Stable selection independent of model responses and corpus line order."""
    if maximum_dialogues < 0:
        raise ValueError("evaluation limit cannot be negative")
    if maximum_dialogues == 0:
        return tuple(dialogues)
    ordered = sorted(dialogues, key=lambda item: hashlib.sha256(item.dialogue_id.encode()).hexdigest())
    return tuple(ordered[:maximum_dialogues])


def validate_vocabulary_partition(vocabulary: LanguageVocabulary, dialogues: Sequence[TeachingDialogue]) -> None:
    def canonical(identifier: str) -> str:
        # OASST1/2 reuse source tree UUIDs across release-specific dialogue IDs.
        return "oasst:" + identifier.split(":", 1)[1] if identifier.startswith(("oasst1:", "oasst2:")) else identifier

    seen = {canonical(value) for value in vocabulary.training.get("train_dialogue_ids", ())}
    held_out = {canonical(item.dialogue_id) for item in dialogues if item.split in {"validation", "test"}}
    if seen & held_out:
        raise ValueError("vocabulary training overlaps the evaluation partition")


def _language_metrics(
    model: Any, loss: Any, prediction: Any, stop: Any, *, prediction_normalization: str = "token",
) -> dict[str, Any]:
    lexical = hasattr(model, "vocabulary")
    return {"loss": float(loss.detach()), "symbol_loss" if lexical else "byte_loss": float(prediction.detach()),
            "stop_loss": float(stop.detach()), "prediction_unit": "learned_subword" if lexical else "utf8_byte",
            "prediction_normalization": prediction_normalization}


STOP_OBJECTIVES = ("legacy", "balanced")


def stop_loss_statistics(logits: Any, targets: Any, mask: Any) -> tuple[Any, Any, int, int]:
    """Keep rare complete-utterance endings separate from interior positions."""
    terms = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    positive = mask.bool() & (targets == 1)
    negative = mask.bool() & (targets == 0)
    return (terms[positive].sum(), terms[negative].sum(), int(positive.sum()), int(negative.sum()))


def reduce_stop_loss(positive: Any, negative: Any, positives: int, negatives: int, *, objective: str) -> Any:
    if objective == "legacy":
        return (4.0 * positive + negative) / max(1, positives + negatives)
    if objective == "balanced":
        # Each class contributes equally, independent of reply length. This
        # changes a training objective, never the inference stop threshold.
        return 0.5 * (positive / max(1, positives) + negative / max(1, negatives))
    raise ValueError("unknown stop objective")


def language_batch(
    dialogues: Sequence[TeachingDialogue],
    config: CortexConfig,
    *,
    maximum_sequence_bytes: int | None = None,
    vocabulary: LanguageVocabulary | None = None,
) -> dict[str, Any]:
    """Unroll event-typed streams for training; no history buffer is used at inference."""
    rows = []
    validate_curriculum(dialogues, config, maximum_sequence_bytes)
    for dialogue in dialogues:
        previous, directions, targets, weights, stops, stop_mask = [], [], [], [], [], []
        for turn in dialogue.turns:
            for encoded in turn.observations_before:
                payload = bytes.fromhex(encoded)
                previous.extend([-1, *payload])
                directions.extend([DIRECTION_IDS["hear"]] * (len(payload) + 1))
                targets.extend([-100] * (len(payload) + 1))
                weights.extend([0.0] * (len(payload) + 1))
                stops.extend([0.0] * (len(payload) + 1))
                stop_mask.extend([False] * (len(payload) + 1))
            for direction, text in (("hear", turn.user), ("speak", turn.assistant)):
                payload = text.encode("utf-8")
                if not payload:
                    raise ValueError("teaching event is empty")
                if vocabulary is not None:
                    payload = vocabulary.encode(text)
                previous.extend([-1, *payload])
                directions.extend([DIRECTION_IDS[direction]] * (len(payload) + 1))
                targets.extend([*payload, -100])
                weights.extend([1.0 if direction == "speak" else 0.15] * len(payload) + [0.0])
                stops.extend([0.0] * len(payload) + [1.0])
                stop_mask.extend([direction == "speak"] * (len(payload) + 1))
        if maximum_sequence_bytes is not None and len(previous) > maximum_sequence_bytes:
            raise ValueError("dialogue exceeds the explicit training unroll limit")
        rows.append((previous, directions, targets, weights, stops, stop_mask))
    if not rows:
        raise ValueError("language batch cannot be empty")
    width = max(len(row[0]) for row in rows)
    padding = (0, DIRECTION_IDS["quiet"], -100, 0.0, 0.0, False)
    names = ("previous", "directions", "targets", "weights", "stops", "stop_mask")
    return {
        name: torch.tensor([row[index] + [padding[index]] * (width - len(row[index])) for row in rows])
        for index, name in enumerate(names)
    }


def language_loss(
    model: Any, batch: dict[str, Any], *, stop_objective: str = "legacy", prediction_normalization: str = "token",
) -> tuple[Any, dict[str, Any]]:
    field = torch.zeros(batch["previous"].shape[0], model.config.graph_width)
    output, _ = model(batch["previous"], field, batch["directions"])
    per_byte = torch.nn.functional.cross_entropy(
        output["byte_logits"].reshape(-1, output["byte_logits"].shape[-1]), batch["targets"].reshape(-1),
        ignore_index=-100, reduction="none",
    ).reshape_as(batch["weights"])
    byte = prediction_average((per_byte * batch["weights"]).sum(1), batch["weights"].sum(1),
                              normalization=prediction_normalization)
    stop = reduce_stop_loss(*stop_loss_statistics(output["stop_logits"], batch["stops"], batch["stop_mask"]),
                            objective=stop_objective)
    loss = byte + 0.30 * stop
    return loss, _language_metrics(model, loss, byte, stop, prediction_normalization=prediction_normalization)


def conditioned_language_loss(
    model: Any, dialogues: Sequence[TeachingDialogue], *, maximum_sequence_bytes: int | None = None,
    stop_objective: str = "legacy",
    prediction_normalization: str = "token",
) -> tuple[Any, dict[str, Any]]:
    """Keep gradients across event-sized state readouts, without text replay."""
    if any(turn.observations_before for item in dialogues for turn in item.turns):
        return grounded_language_loss(model, dialogues, maximum_sequence_bytes=maximum_sequence_bytes,
                                      stop_objective=stop_objective, prediction_normalization=prediction_normalization)
    if not dialogues or any(not item.turns for item in dialogues):
        raise ValueError("language batch and teaching dialogues cannot be empty")
    if maximum_sequence_bytes is not None and any(
        sum(len(turn.user.encode("utf-8")) + len(turn.assistant.encode("utf-8")) + 2
            for turn in item.turns) > maximum_sequence_bytes
        for item in dialogues
    ):
        raise ValueError("dialogue exceeds the explicit training unroll limit")
    hidden = model.initial_hidden(len(dialogues), device="cpu", dtype=torch.float32)
    byte_total = torch.zeros(len(dialogues))
    byte_count = torch.zeros(len(dialogues))
    positive_stop_total = negative_stop_total = 0.0
    positive_stop_count = negative_stop_count = 0
    for index in range(max(len(item.turns) for item in dialogues)):
        for direction in ("hear", "speak"):
            payloads = []
            active = []
            for dialogue in dialogues:
                if index < len(dialogue.turns):
                    turn = dialogue.turns[index]
                    payload = (turn.user if direction == "hear" else turn.assistant).encode("utf-8")
                    if not payload:
                        raise ValueError("teaching event is empty")
                    payload = language_symbols(model, turn.user if direction == "hear" else turn.assistant)
                    active.append(True)
                else:
                    payload = b""
                    active.append(False)
                payloads.append(payload)
            lengths = torch.tensor([len(payload) + 1 for payload in payloads])
            width = int(lengths.max())
            previous = torch.tensor([[-1, *payload, *([0] * (width - len(payload) - 1))] for payload in payloads])
            target = torch.tensor([[*payload, *([-100] * (width - len(payload)))] for payload in payloads])
            field = language_field(model, hidden)
            outputs, hidden = model(
                previous, field, torch.full((len(dialogues),), DIRECTION_IDS[direction]), hidden,
                sequence_lengths=lengths,
            )
            byte_terms = torch.nn.functional.cross_entropy(
                outputs["byte_logits"].reshape(-1, outputs["byte_logits"].shape[-1]), target.reshape(-1),
                ignore_index=-100, reduction="none",
            ).reshape_as(target).sum(1)
            weight = 1.0 if direction == "speak" else 0.15
            byte_total = byte_total + weight * byte_terms
            byte_count = byte_count + weight * torch.tensor([len(payload) for payload in payloads])
            if direction == "speak":
                stops = torch.zeros(len(dialogues), width)
                stops[torch.arange(len(dialogues)), lengths - 1] = 1.0
                mask = torch.arange(width).unsqueeze(0) < lengths.unsqueeze(1)
                mask = mask & torch.tensor(active).unsqueeze(1)
                positive, negative, positives, negatives = stop_loss_statistics(outputs["stop_logits"], stops, mask)
                positive_stop_total = positive_stop_total + positive
                negative_stop_total = negative_stop_total + negative
                positive_stop_count += positives
                negative_stop_count += negatives
    byte = prediction_average(byte_total, byte_count, normalization=prediction_normalization)
    stop = reduce_stop_loss(positive_stop_total, negative_stop_total, positive_stop_count, negative_stop_count,
                            objective=stop_objective)
    loss = byte + 0.3 * stop
    return loss, _language_metrics(model, loss, byte, stop, prediction_normalization=prediction_normalization)


def grounded_language_loss(
    model: Any, dialogues: Sequence[TeachingDialogue], *, maximum_sequence_bytes: int | None,
    stop_objective: str, prediction_normalization: str,
) -> tuple[Any, dict[str, Any]]:
    """Backpropagate language supervision through once-admitted sensory events.

    Observation fields are raw bytes, never tokenized prose or target language.
    The old text-only path remains unchanged for comparable existing lineages.
    """
    if not dialogues:
        raise ValueError("language batch cannot be empty")
    validate_curriculum(dialogues, model.config, maximum_sequence_bytes)
    sensory_direction = "observe" if hasattr(model, "sensory_trunk") else "hear"
    streams = []
    for dialogue in dialogues:
        events = []
        for turn in dialogue.turns:
            events.extend((sensory_direction, tuple(bytes.fromhex(encoded)), 0.0) for encoded in turn.observations_before)
            events.append(("hear", language_symbols(model, turn.user), 0.15))
            events.append(("speak", language_symbols(model, turn.assistant), 1.0))
        streams.append(events)
    hidden = model.initial_hidden(len(streams), device="cpu", dtype=torch.float32)
    totals, counts = torch.zeros(len(streams)), torch.zeros(len(streams))
    positive_total = negative_total = 0.0
    positive_count = negative_count = 0
    for index in range(max(map(len, streams))):
        events = [row[index] if index < len(row) else ("quiet", (), 0.0) for row in streams]
        prior_hidden = hidden
        sensory_indices = [row for row, (direction, _, _) in enumerate(events) if direction == "observe"]
        sensory_events = [events[row][1] for row in sensory_indices]
        # Mixed event batches may contain language and observations together.
        # Sensory rows get no language loss or language-state update; their
        # placeholder output is overwritten by the actual sensory transition.
        events = [("quiet", (), 0.0) if direction == "observe" else (direction, payload, weight)
                  for direction, payload, weight in events]
        lengths = torch.tensor([len(payload) + 1 for _, payload, _ in events])
        width = int(lengths.max())
        previous = torch.tensor([[-1, *payload, *([0] * (width - len(payload) - 1))]
                                 for _, payload, _ in events])
        targets = torch.tensor([[*(payload if weight else ()),
                                 *([-100] * (width - (len(payload) if weight else 0)))]
                                for _, payload, weight in events])
        outputs, hidden = model(
            previous, language_field(model, hidden),
            torch.tensor([DIRECTION_IDS[direction] for direction, _, _ in events]), hidden,
            sequence_lengths=lengths,
        )
        if sensory_indices:
            selected = torch.tensor(sensory_indices)
            sensory_lengths = torch.tensor([len(payload) + 1 for payload in sensory_events])
            sensory_width = int(sensory_lengths.max())
            sensory_previous = torch.tensor([[-1, *payload, *([0] * (sensory_width - len(payload) - 1))]
                                             for payload in sensory_events])
            sensory_prior = prior_hidden.index_select(1, selected)
            sensory_hidden = sensory_transition(model, sensory_previous, language_field(model, sensory_prior),
                                                 sensory_prior, sequence_lengths=sensory_lengths)
            hidden = hidden.index_copy(1, selected, sensory_hidden)
        losses = torch.nn.functional.cross_entropy(
            outputs["byte_logits"].reshape(-1, outputs["byte_logits"].shape[-1]), targets.reshape(-1),
            ignore_index=-100, reduction="none",
        ).reshape_as(targets).sum(1)
        weights = torch.tensor([weight for _, _, weight in events])
        totals = totals + weights * losses
        counts = counts + weights * torch.tensor([len(payload) for _, payload, _ in events])
        stops = torch.zeros(len(streams), width)
        stops[torch.arange(len(streams)), lengths - 1] = 1.0
        mask = (torch.arange(width).unsqueeze(0) < lengths.unsqueeze(1)) & torch.tensor(
            [direction == "speak" for direction, _, _ in events]
        ).unsqueeze(1)
        positive, negative, positives, negatives = stop_loss_statistics(outputs["stop_logits"], stops, mask)
        positive_total = positive_total + positive
        negative_total = negative_total + negative
        positive_count += positives
        negative_count += negatives
    prediction = prediction_average(totals, counts, normalization=prediction_normalization)
    stop = reduce_stop_loss(positive_total, negative_total, positive_count, negative_count, objective=stop_objective)
    loss = prediction + 0.3 * stop
    return loss, _language_metrics(model, loss, prediction, stop, prediction_normalization=prediction_normalization)


def observe_teaching_turn(conversation: GroundedConversation, turn: TeachingTurn) -> list[Any]:
    """Admit only this turn's current observations; never look them up at reply time."""
    return [conversation.observe(decode_observation(bytes.fromhex(encoded))) for encoded in turn.observations_before]


def evaluate_dialogues(
    model: Any,
    dialogues: Sequence[TeachingDialogue],
    *,
    reset_before_scored: bool = False,
    maximum_bytes: int | None = None,
    answer_mode: str = "exact",
) -> dict[str, Any]:
    if answer_mode not in {"exact", "reference"}:
        raise ValueError("answer mode must be exact or reference")
    trials = []
    for dialogue in dialogues:
        with GroundedConversation(model) as conversation:
            for index, turn in enumerate(dialogue.turns):
                if reset_before_scored and turn.scored:
                    conversation.hidden = model.initial_hidden(1, device="cpu", dtype=torch.float32)
                observation_receipts = observe_teaching_turn(conversation, turn)
                result = conversation.reply(turn.user, maximum_bytes=maximum_bytes)
                if turn.scored:
                    trials.append({
                        "dialogue_id": dialogue.dialogue_id, "scenario": dialogue.scenario,
                        "turn": index, "input": turn.user, "expected": turn.assistant,
                        "response": result.response, "status": result.status,
                        "exact_reference_match": result.response == turn.assistant,
                        "correct": result.response == turn.assistant if answer_mode == "exact" else None,
                        "input_byte_count": result.input_byte_count,
                        "state_bytes": result.state_bytes,
                        **({"observation_receipts": [asdict(x) for x in observation_receipts]}
                           if observation_receipts else {}),
                    })
    return {
        "scored_turns": len(trials),
        "evaluation_mode": answer_mode,
        "exact_reference_match_rate": sum(item["exact_reference_match"] for item in trials) / len(trials) if trials else None,
        "exact_response_accuracy": (sum(item["correct"] for item in trials) / len(trials)
                                    if trials and answer_mode == "exact" else None),
        "semantic_coherence_assessed": False,
        "completed_utterance_rate": sum(item["status"] == "spoken" for item in trials) / max(1, len(trials)),
        "reset_before_scored": reset_before_scored,
        "maximum_output_bytes": maximum_bytes,
        "dialogue_ids": [item.dialogue_id for item in dialogues],
        "trials": trials,
    }


def restart_check(
    model: Any, dialogue: TeachingDialogue, state_path: Path, *, maximum_bytes: int | None = None,
) -> dict[str, Any]:
    uninterrupted = GroundedConversation(model)
    restarted = GroundedConversation(model, state_path=state_path)
    matches = []
    observation_matches = []
    for turn in dialogue.turns:
        left_observations = observe_teaching_turn(uninterrupted, turn)
        right_observations = observe_teaching_turn(restarted, turn)
        observation_matches.extend(asdict(left) == asdict(right)
                                   for left, right in zip(left_observations, right_observations))
        left = uninterrupted.reply(turn.user, maximum_bytes=maximum_bytes)
        right = restarted.reply(turn.user, maximum_bytes=maximum_bytes)
        matches.append(asdict(left) == asdict(right))
        restarted.close()
        restarted = GroundedConversation(model, state_path=state_path)
    uninterrupted.close()
    restarted.close()
    return {"all_turns_identical": all(matches) and all(observation_matches), "turn_count": len(matches),
            **({"observation_events": len(observation_matches), "all_observations_identical": all(observation_matches)}
               if observation_matches else {})}


def swapped_history_check(model: Any, dialogues: Sequence[TeachingDialogue]) -> dict[str, Any]:
    """Ask identical questions with state exchanged between different histories."""
    grouped: dict[str, list[Any]] = {}
    for dialogue in dialogues:
        if dialogue.scenario != "retain":
            continue
        with GroundedConversation(model) as conversation:
            for turn in dialogue.turns[:-1]:
                observe_teaching_turn(conversation, turn)
                conversation.reply(turn.user, maximum_bytes=None)
            final = dialogue.turns[-1]
            if final.observations_before:
                raise ValueError("swapped-state questions must not supply a new observation with the answer")
            grouped.setdefault(final.user, []).append((dialogue, conversation.hidden.clone()))
    trials = []
    for question, states in grouped.items():
        if len(states) < 2:
            continue
        for index, (recipient, _) in enumerate(states):
            candidates = (states[(index + offset) % len(states)] for offset in range(1, len(states)))
            alternative = next((candidate for candidate in candidates
                                if candidate[0].turns[-1].assistant != recipient.turns[-1].assistant), None)
            if alternative is None:
                continue
            donor, hidden = alternative
            with GroundedConversation(model) as conversation:
                conversation.hidden = hidden.clone()
                result = conversation.reply(question, maximum_bytes=None)
            trials.append({
                "question": question, "recipient_dialogue_id": recipient.dialogue_id,
                "donor_dialogue_id": donor.dialogue_id, "response": result.response,
                "recipient_expected": recipient.turns[-1].assistant,
                "donor_expected": donor.turns[-1].assistant,
                "matches_recipient": result.response == recipient.turns[-1].assistant,
                "matches_donor": result.response == donor.turns[-1].assistant,
            })
    return {
        "paired_trials": len(trials), "trials": trials,
        "donor_state_accuracy": sum(item["matches_donor"] for item in trials) / len(trials) if trials else None,
        "recipient_state_accuracy": sum(item["matches_recipient"] for item in trials) / len(trials) if trials else None,
    }


def _preset(size: str, seed: int) -> CortexConfig:
    if size == "18m":
        return CortexConfig(seed=seed, maximum_event_bytes=512, learning_rate=0.001)
    if size == "medium":
        return CortexConfig(
            byte_embedding_width=96, graph_width=64, direction_width=16,
            hidden_width=384, recurrent_layers=2, maximum_event_bytes=512,
            seed=seed, learning_rate=0.001, weight_decay=0.001,
        )
    return CortexConfig(
        byte_embedding_width=48, graph_width=16, direction_width=8,
        hidden_width=128, recurrent_layers=2, maximum_event_bytes=512,
        seed=seed, learning_rate=0.003, weight_decay=0.001,
    )


TRAINING_STATE_SCHEMA = "habitus.language-training-state.v1"
TRAINING_PRECISIONS = ("float32", "cpu_bfloat16")


def training_precision_context(precision: str) -> Any:
    """Autocast forward/loss only; weights, optimizer, and inference stay float32."""
    if precision not in TRAINING_PRECISIONS:
        raise ValueError("unsupported CPU training precision")
    require_torch()
    if precision == "float32":
        return nullcontext()
    if not torch.amp.autocast_mode.is_autocast_available("cpu"):
        raise ValueError("this build does not provide CPU autocast")
    return torch.autocast("cpu", dtype=torch.bfloat16)


def training_recipe() -> dict[str, Any]:
    return {
        "torch_version": str(torch.__version__), "threads": torch.get_num_threads(),
        "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in LANGUAGE_SOURCE_FILES},
    }


def _validate_sampling_snapshot(policy: str, identity: dict[str, Any] | None, batch_size: int, warmup_steps: int) -> None:
    if policy not in SAMPLING_POLICIES:
        raise ValueError("unsupported training sampling policy")
    if identity is not None and (not isinstance(identity, dict) or identity.get("policy") != policy or
                                 identity.get("batch_size") != batch_size):
        raise ValueError("sampler identity does not match the sampling configuration")
    if policy != "uniform" and (identity is None or warmup_steps != 0):
        raise ValueError("family snapshots require sampler identity and zero warmup steps")
    if policy == "allocated_families":
        counts, pools = identity.get("family_dialogues_per_batch"), identity.get("family_pool_sizes")
        if (not isinstance(counts, dict) or not counts or not isinstance(pools, dict)
                or set(counts) != set(pools) or any(type(v) is not int or v < 1 for v in counts.values())
                or any(type(v) is not int or v < 1 for v in pools.values()) or sum(counts.values()) != batch_size):
            raise ValueError("allocated-family snapshot has invalid batch counts or pools")


def save_training_state(
    path: Path, model: Any, optimizer: Any, rng: random.Random, *,
    corpus_hash: str, completed_steps: int, warmup_steps: int,
    batch_size: int, maximum_sequence_bytes: int | None,
    stop_objective: str = "legacy",
    prediction_normalization: str = "token",
    sampling_policy: str = "uniform",
    sampler_identity: dict[str, Any] | None = None,
    precision: str = "float32",
) -> None:
    """Training-only optimizer/sampler snapshot, never loaded by conversation."""
    _validate_sampling_snapshot(sampling_policy, sampler_identity, batch_size, warmup_steps)
    training_precision_context(precision)
    payload = {
        "schema": TRAINING_STATE_SCHEMA, "model_sha256": language_model_sha256(model),
        "curriculum_sha256": corpus_hash, "completed_steps": completed_steps,
        "warmup_steps": warmup_steps, "batch_size": batch_size,
        "maximum_sequence_bytes": maximum_sequence_bytes,
        "stop_objective": stop_objective,
        "prediction_normalization": prediction_normalization,
        "sampling_policy": sampling_policy, "sampler_identity": sampler_identity,
        "precision": precision,
        "optimizer": optimizer.state_dict(), "sampler_rng": rng.getstate(),
        "torch_rng": torch.get_rng_state(),
        "recipe": training_recipe(),
    }
    with path.open("xb") as stream:
        torch.save(payload, stream)


def restore_training_state(
    path: Path, model: Any, optimizer: Any, rng: random.Random, *,
    corpus_hash: str, batch_size: int, maximum_sequence_bytes: int | None,
    stop_objective: str = "legacy",
    prediction_normalization: str = "token",
    sampling_policy: str = "uniform",
    sampler_identity: dict[str, Any] | None = None,
    precision: str = "float32",
) -> dict[str, int]:
    training_precision_context(precision)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    expected = {
        "schema": TRAINING_STATE_SCHEMA, "model_sha256": language_model_sha256(model),
        "curriculum_sha256": corpus_hash, "batch_size": batch_size,
        "maximum_sequence_bytes": maximum_sequence_bytes,
        "stop_objective": stop_objective,
        "prediction_normalization": prediction_normalization,
        "sampling_policy": sampling_policy, "sampler_identity": sampler_identity,
        "precision": precision,
        "recipe": training_recipe(),
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("training state does not match the weights, corpus, or batching configuration")
    if payload["completed_steps"] < 0 or payload["warmup_steps"] < 0:
        raise ValueError("invalid saved training step counts")
    _validate_sampling_snapshot(sampling_policy, sampler_identity, batch_size, payload["warmup_steps"])
    for state in payload["optimizer"]["state"].values():
        for value in state.values():
            if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                raise ValueError("non-finite optimizer state")
    optimizer.load_state_dict(payload["optimizer"])
    rng.setstate(payload["sampler_rng"])
    torch.set_rng_state(payload["torch_rng"])
    return {name: int(payload[name]) for name in ("completed_steps", "warmup_steps")}


def train_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--vocabulary", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--resume-training-state", type=Path)
    parser.add_argument("--size", choices=("small", "medium", "18m"), default="small")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--precision", choices=TRAINING_PRECISIONS, default="float32")
    parser.add_argument("--report-every", type=int, default=50)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--stop-objective", choices=STOP_OBJECTIVES, default="legacy")
    parser.add_argument("--prediction-normalization", choices=PREDICTION_NORMALIZATIONS, default="token")
    parser.add_argument("--sampling-policy", choices=SAMPLING_POLICIES, default="uniform")
    parser.add_argument("--family-batch-counts", type=json.loads,
                        help="JSON object of positive per-family counts for allocated_families; must sum to batch size")
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--maximum-event-bytes", type=int)
    parser.add_argument("--maximum-sequence-bytes", type=int, default=None)
    parser.add_argument("--maximum-output-bytes", type=int)
    parser.add_argument("--maximum-evaluation-dialogues", type=int, default=0)
    parser.add_argument("--evaluation-mode", choices=("auto", "exact", "reference"), default="auto")
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--state-readout", action="store_true")
    parser.add_argument("--stream-normalization", action="store_true")
    parser.add_argument("--sensory-trunk", action="store_true")
    parser.add_argument("--sensory-state-policy", choices=SENSORY_STATE_POLICIES,
                        help="new models default to shared; resume preserves the checkpoint's policy")
    parser.add_argument("--sensory-readout-mode", choices=SENSORY_READOUT_MODES,
                        help="new reservoirs default to static; resume preserves the checkpoint's readout")
    parser.add_argument("--language-input-skip", choices=LANGUAGE_INPUT_SKIPS,
                        help="new models default to none; resume preserves the checkpoint's current-message skip")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.evaluate_only and not args.resume_checkpoint:
        parser.error("--evaluate-only requires --resume-checkpoint")
    if args.resume_training_state and (not args.resume_checkpoint or args.evaluate_only or args.learning_rate is not None):
        parser.error("optimizer resume requires weights, a training run, and no learning-rate override")
    if args.evaluate_only:
        args.steps = 0
    if (not args.evaluate_only and args.steps < 1) or min(args.batch_size, args.threads, args.report_every) < 1:
        parser.error("training counts must be positive")
    if min(args.checkpoint_every, args.maximum_evaluation_dialogues) < 0 or (
        args.maximum_sequence_bytes is not None and args.maximum_sequence_bytes < 1
    ):
        parser.error("checkpoint/evaluation limits cannot be negative; an explicit sequence limit must be positive")
    if args.maximum_event_bytes is not None and args.maximum_event_bytes < 1:
        parser.error("event limit must be positive")
    require_torch()
    training_precision_context(args.precision)
    torch.set_num_threads(args.threads)
    dialogues = load_curriculum(args.corpus) if args.corpus else controlled_curriculum()
    splits = {name: tuple(item for item in dialogues if item.split == name) for name in ("train", "validation", "test")}
    if any(not items for items in splits.values()):
        parser.error("all three curriculum splits are required")
    try:
        sampler = DialogueSampler(splits["train"], batch_size=args.batch_size, policy=args.sampling_policy,
                                  family_batch_counts=args.family_batch_counts)
    except ValueError as exc:
        parser.error(str(exc))
    short_pool = tuple(item for item in splits["train"] if len(item.turns) == 1)
    warmup_sampler = DialogueSampler(short_pool or splits["train"], batch_size=args.batch_size)
    vocabulary = LanguageVocabulary.load(args.vocabulary) if args.vocabulary else None
    if vocabulary is not None and vocabulary.training.get("pretrained_tokenizer_loaded") is not False:
        parser.error("vocabulary must declare native training without a pretrained tokenizer")
    if args.resume_checkpoint:
        model = load_language_checkpoint(args.resume_checkpoint)
        config = model.config
        if args.state_readout and not hasattr(model, "state_readout"):
            parser.error("state readout requires a fresh lineage, not a baseline checkpoint")
        if args.stream_normalization and not getattr(model, "stream_normalization", False):
            parser.error("stream normalization requires a fresh lineage, not a baseline checkpoint")
        if args.sensory_trunk and not hasattr(model, "sensory_trunk"):
            parser.error("a sensory trunk requires a fresh lineage, not a baseline checkpoint")
        if args.sensory_state_policy is not None and args.sensory_state_policy != getattr(model, "sensory_state_policy", "shared"):
            parser.error("resume preserves its checkpoint's sensory state policy")
        if args.sensory_readout_mode is not None and args.sensory_readout_mode != getattr(model, "sensory_readout_mode", "static"):
            parser.error("resume preserves its checkpoint's sensory readout mode")
        if args.language_input_skip is not None and args.language_input_skip != getattr(model, "language_input_skip", "none"):
            parser.error("resume preserves its checkpoint's language input skip")
        if args.maximum_event_bytes is not None and args.maximum_event_bytes != config.maximum_event_bytes:
            parser.error("resume preserves its checkpoint's event limit")
        stored_vocabulary = getattr(model, "vocabulary", None)
        if vocabulary is not None and (stored_vocabulary is None or vocabulary.sha256 != stored_vocabulary.sha256):
            parser.error("resume preserves its checkpoint's vocabulary")
        vocabulary = stored_vocabulary
        initial_hash = model.language_initial_sha256
        parent_hash = language_model_sha256(model)
    else:
        config = _preset(args.size, args.seed)
        if args.maximum_event_bytes is not None:
            config = replace(config, maximum_event_bytes=args.maximum_event_bytes)
        if args.stream_normalization and (vocabulary is None or not args.state_readout):
            parser.error("stream normalization requires a vocabulary and state readout")
        if args.sensory_trunk and (vocabulary is None or not args.state_readout):
            parser.error("a sensory trunk requires a vocabulary and state readout")
        if args.sensory_state_policy not in (None, "shared") and not (vocabulary and args.state_readout and args.sensory_trunk):
            parser.error("a reservoir policy requires a vocabulary, state readout, and sensory trunk")
        if args.sensory_readout_mode not in (None, "static") and args.sensory_state_policy in (None, "shared"):
            parser.error("a sensory query readout requires a numeric reservoir")
        if args.language_input_skip not in (None, "none") and (args.sensory_state_policy != "protected"
                or args.sensory_readout_mode not in (None, "static")):
            parser.error("language input skip requires a protected static reservoir")
        model = new_language_core(config, state_readout=args.state_readout, vocabulary=vocabulary,
                                  stream_normalization=args.stream_normalization, sensory_trunk=args.sensory_trunk,
                                  sensory_state_policy=args.sensory_state_policy or "shared",
                                  sensory_readout_mode=args.sensory_readout_mode or "static",
                                  language_input_skip=args.language_input_skip or "none")
        initial_hash = language_model_sha256(model)
        parent_hash = None
    if vocabulary is not None:
        validate_vocabulary_partition(vocabulary, dialogues)
    learning_rate = args.learning_rate if args.learning_rate is not None else config.learning_rate
    if not 0 < learning_rate < 1:
        parser.error("learning rate must be within (0, 1)")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=config.weight_decay)
    rng = random.Random(args.seed)
    validate_curriculum(dialogues, config, args.maximum_sequence_bytes)
    corpus_hash = curriculum_hash(dialogues)
    completed_before = 0
    # Filtering to single-turn examples could remove a whole teaching family.
    # The explicit balanced policy therefore trains the full mixture from step 1.
    warmup_steps = (0 if args.resume_checkpoint or args.sampling_policy != "uniform"
                    else min(100, args.steps // 5))
    if args.resume_training_state:
        restored = restore_training_state(
            args.resume_training_state, model, optimizer, rng, corpus_hash=corpus_hash,
            batch_size=args.batch_size, maximum_sequence_bytes=args.maximum_sequence_bytes,
            stop_objective=args.stop_objective,
            prediction_normalization=args.prediction_normalization,
            sampling_policy=args.sampling_policy, sampler_identity=sampler.identity(),
            precision=args.precision,
        )
        completed_before, warmup_steps = restored["completed_steps"], restored["warmup_steps"]
        learning_rate = optimizer.param_groups[0]["lr"]
    answer_mode = args.evaluation_mode
    if answer_mode == "auto":
        answer_mode = "reference" if args.corpus else "exact"
    output_limit = args.maximum_output_bytes
    if output_limit is not None and output_limit < 1:
        parser.error("an explicit output limit must be positive")
    # One run owns a new directory. Existing state/checkpoints are never replaced.
    args.output_directory.mkdir(parents=True, exist_ok=False)
    with tarfile.open(args.output_directory / "source.tar", mode="x") as archive:
        for source in sorted(Path(__file__).parent.glob("*.py")):
            archive.add(source, arcname=f"habitus_ai/{source.name}")
    started = time.monotonic()
    run_provenance = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "source_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in LANGUAGE_SOURCE_FILES
        },
        "arguments": {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()},
    }
    history = []
    corpus_sources = sorted({json.dumps({key: item.provenance[key] for key in ("dataset", "revision", "license")
                                        if key in item.provenance}, sort_keys=True)
                             for item in dialogues if item.provenance})
    observation_counts = {name: sum(len(turn.observations_before) for item in items for turn in item.turns)
                          for name, items in splits.items()}
    sensory_metadata = ({
        "observation_events_by_split": observation_counts,
        "observation_encoding": SENSOR_SCHEMA,
        "observation_symbols": ("separate_learned_sensory_byte_alphabet" if hasattr(model, "sensory_trunk")
                                else "raw_byte_ids_in_existing_hear_direction"),
        "observation_language_target_loss": False,
        "current_observations_admitted_once": True,
        "prior_observations_refed_during_inference": False,
        "filesystem_reread_during_reply": False,
    } if any(observation_counts.values()) else {})

    def training_metadata(local_steps: int) -> dict[str, Any]:
        return {
            "kind": "evaluation_only" if args.evaluate_only else "supervised_language_examples_not_lived_experience",
            "steps": local_steps, "completed_steps": completed_before + local_steps,
            "seed": args.seed, "curriculum_sha256": corpus_hash,
            "corpus_sources": [json.loads(item) for item in corpus_sources],
            "train_dialogue_ids": [item.dialogue_id for item in splits["train"]],
            "history": list(history), "training_state_mode": "full_dialogue_bptt",
            "inference_state_mode": "current_event_once_plus_fixed_recurrent_state",
            "parent_model_sha256": parent_hash, "optimizer_resumed": bool(args.resume_training_state),
            "learning_rate": learning_rate, "state_readout": hasattr(model, "state_readout"),
            "stream_normalization": getattr(model, "stream_normalization", False),
            "sensory_trunk": hasattr(model, "sensory_trunk"),
            "sensory_state_policy": getattr(model, "sensory_state_policy", "shared"),
            "sensory_readout_mode": getattr(model, "sensory_readout_mode", "static"),
            "language_input_skip": getattr(model, "language_input_skip", "none"),
            "numeric_state_bytes": model.initial_hidden(1, device="cpu", dtype=torch.float32).numel() * 4,
            "precision": args.precision, "master_weight_dtype": "float32", "inference_precision": "float32",
            "stop_objective": args.stop_objective,
            "prediction_normalization": args.prediction_normalization,
            "sampling_policy": args.sampling_policy, "sampler_identity": sampler.identity(),
            "warmup_steps": warmup_steps,
            "prediction_unit": "learned_subword" if vocabulary is not None else "utf8_byte",
            "vocabulary_sha256": vocabulary.sha256 if vocabulary is not None else None,
            "run_provenance": run_provenance,
            **sensory_metadata,
        }

    def snapshot(weights_path: Path, state_path: Path, local_steps: int) -> str:
        model_hash = save_language_checkpoint(weights_path, model, initial_model_sha256=initial_hash,
                                             training=training_metadata(local_steps))
        save_training_state(
            state_path, model, optimizer, rng, corpus_hash=corpus_hash,
            completed_steps=completed_before + local_steps, warmup_steps=warmup_steps,
            batch_size=args.batch_size, maximum_sequence_bytes=args.maximum_sequence_bytes,
            stop_objective=args.stop_objective,
            prediction_normalization=args.prediction_normalization,
            sampling_policy=args.sampling_policy, sampler_identity=sampler.identity(),
            precision=args.precision,
        )
        return model_hash

    print(json.dumps({"event": "start", "parameters": sum(p.numel() for p in model.parameters()),
                      "config": asdict(config), "split_counts": {key: len(value) for key, value in splits.items()},
                      "curriculum_sha256": corpus_hash, "pretrained_model_loaded": False,
                      "vocabulary_size": len(vocabulary) if vocabulary is not None else 256,
                      "precision": args.precision,
                      "stream_normalization": getattr(model, "stream_normalization", False),
                      "sensory_trunk": hasattr(model, "sensory_trunk"),
                      "sensory_state_policy": getattr(model, "sensory_state_policy", "shared"),
                      "sensory_readout_mode": getattr(model, "sensory_readout_mode", "static"),
                      "numeric_state_bytes": model.initial_hidden(1, device="cpu", dtype=torch.float32).numel() * 4,
                      "sampling_policy": args.sampling_policy, "sampler_identity": sampler.identity(),
                      "completed_steps_before_run": completed_before, "evaluation_mode": answer_mode}), flush=True)
    for step in range(1, args.steps + 1):
        # First learn the short language mappings, then train full dialogue
        # dependency paths. All curriculum stages remain in the later mixture.
        warmup = completed_before + step <= warmup_steps
        sampled = (warmup_sampler if warmup else sampler).sample(rng)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with training_precision_context(args.precision):
            if hasattr(model, "state_readout"):
                loss, metrics = conditioned_language_loss(model, sampled, maximum_sequence_bytes=args.maximum_sequence_bytes,
                                                          stop_objective=args.stop_objective,
                                                          prediction_normalization=args.prediction_normalization)
            else:
                loss, metrics = language_loss(model, language_batch(sampled, config,
                                                                   maximum_sequence_bytes=args.maximum_sequence_bytes,
                                                                   vocabulary=vocabulary), stop_objective=args.stop_objective,
                                              prediction_normalization=args.prediction_normalization)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite language loss; no successful checkpoint will be reported")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip, error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % args.report_every == 0 or step == args.steps:
            item = {"event": "training", "step": step, "completed_steps": completed_before + step,
                    "precision": args.precision,
                    "elapsed_seconds": time.monotonic() - started,
                    "sampling_policy": args.sampling_policy,
                    "sampled_family_dialogues": dict(Counter(item.provenance.get("training_family", "unassigned")
                                                             for item in sampled)), **metrics}
            history.append(item)
            print(json.dumps(item), flush=True)
        if args.checkpoint_every and step % args.checkpoint_every == 0 and step < args.steps:
            stem = f"step-{completed_before + step:08d}"
            weight_path = args.output_directory / f"{stem}.pt"
            state_path = args.output_directory / f"{stem}.training.pt"
            model_hash = snapshot(weight_path, state_path, step)
            print(json.dumps({"event": "training_snapshot", "path": str(weight_path),
                              "training_state": str(state_path), "model_sha256": model_hash}), flush=True)
    model.eval()
    training = training_metadata(args.steps)
    if args.evaluate_only:
        checkpoint = args.resume_checkpoint
        model_hash = language_model_sha256(model)
    else:
        checkpoint = args.output_directory / "language.pt"
        model_hash = snapshot(checkpoint, args.output_directory / "training-state.pt", args.steps)
    print(json.dumps({"event": "checkpoint", "path": str(checkpoint), "model_sha256": model_hash}), flush=True)
    evaluation_splits = {key: evaluation_subset(value, args.maximum_evaluation_dialogues) for key, value in splits.items()}
    evaluation_options = {"maximum_bytes": output_limit, "answer_mode": answer_mode}
    validation = evaluate_dialogues(model, evaluation_splits["validation"], **evaluation_options)
    held_out = evaluate_dialogues(model, evaluation_splits["test"], **evaluation_options) if args.evaluate_test else None
    evaluated_split = "test" if args.evaluate_test else "validation"
    memory_trials = tuple(item for item in splits[evaluated_split] if len(item.turns) > 1)
    memory_trials = evaluation_subset(memory_trials, args.maximum_evaluation_dialogues)
    # A first-turn question has no preceding conversation to ablate.
    memory_trials = tuple(replace(item, turns=tuple(replace(turn, scored=turn.scored and index > 0)
                                                  for index, turn in enumerate(item.turns))) for item in memory_trials)
    memory_intact = evaluate_dialogues(model, memory_trials, **evaluation_options)
    reset = evaluate_dialogues(model, memory_trials, reset_before_scored=True, **evaluation_options)
    swapped = swapped_history_check(model, memory_trials) if answer_mode == "exact" else None
    restart_dialogue = next((item for item in splits[evaluated_split] if len(item.turns) > 1), splits[evaluated_split][0])
    restart = restart_check(model, restart_dialogue, args.output_directory / "restart-check.sqlite", maximum_bytes=output_limit)
    scenarios = {item.scenario for item in dialogues}
    reference_scope = (
        "human_prose_continuation_samples_not_semantically_graded" if scenarios == {"human_prose_continuation"} else
        "human_reference_dialogue_samples_not_semantically_graded" if scenarios == {"human_dialogue"} else
        "mixed_reference_teaching_samples_not_semantically_graded"
    )
    report = {
        "scope": (reference_scope if answer_mode == "reference" else
                  "controlled_workspace_sensor_grounding_not_general_conversation" if sensory_metadata else
                  "operator_supplied_exact_answer_curriculum" if args.corpus else "controlled_small_world_not_open_ended_conversation"),
        "checkpoint": str(checkpoint), "model_sha256": model_hash,
        "initial_model_sha256": initial_hash, "parameter_count": sum(p.numel() for p in model.parameters()),
        "training": training, "validation": validation, "held_out": held_out,
        "evaluation_dialogue_limit": args.maximum_evaluation_dialogues,
        "split_counts": {key: len(value) for key, value in splits.items()},
        "ablation_split": evaluated_split,
        "memory_intact": memory_intact, "reset_state": reset,
        "swapped_state": swapped, "restart": restart,
        "system_prompt_used": False, "transcript_replayed": False, "rag_used": False,
        "pretrained_model_loaded": False, "elapsed_seconds": time.monotonic() - started,
    }
    with (args.output_directory / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
    print(json.dumps({"event": "evaluation", "validation_accuracy": validation["exact_response_accuracy"],
                      "held_out_accuracy": held_out["exact_response_accuracy"] if held_out else None,
                      "utterance_completion": validation["completed_utterance_rate"],
                      "memory_accuracy": memory_intact["exact_response_accuracy"],
                      "reset_accuracy": reset["exact_response_accuracy"],
                      "swapped_donor_accuracy": swapped["donor_state_accuracy"] if swapped else None, "restart": restart,
                      "evaluation_mode": answer_mode, "semantic_coherence_assessed": False,
                      "report": str(args.output_directory / "report.json")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(train_main())
