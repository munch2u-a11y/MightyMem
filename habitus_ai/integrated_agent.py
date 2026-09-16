"""Born-in Habitus runtime with a separate current-event chat renderer.

The developmental runtime remains the authority for sensory admission, recurrent
state, desires, action selection, and one-use motor authorization.  An Ollama
model is used only after SELF authorizes SPEAK. It receives only the current
HEAR event, without a system message, persona, state prose, or retrieved text.
The external model composes its own reply; this is not cortex-generated speech.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import shlex
import sys
from typing import Any, Mapping, Sequence

from .developmental_cortex import CortexConfig, CortexDependencyError
from .developmental_curriculum import EpisodeKind
from .developmental_runtime import (
    BornInHabitusRuntime,
    DevelopmentalInput,
    DevelopmentalPulseReceipt,
)
from .embeddings import cosine_similarity, opaque_payload_embedding, tokenize
from .functional_agent import WorkspacePolicy, _tool_handler
from .gestation import GestationProfile, gestate, load_profile
from .models import ChatMessage, ChatModel, ModelUnavailableError, OllamaChatModel
from .self_pulse import CognitiveCycleReceipt, OutputAffordance
from .tools import ToolDefinition, ToolReceipt, ToolRegistry
from .types import EventKind, InputTrunk, OutputTrunk, RecordType


DEFAULT_DATABASE = Path("habitus-mind.sqlite")
DEFAULT_MODEL = "qwen3.5:2b"
FOUNDATION_KEY = "integrated_born_in_foundations_v1"
EXPLICIT_FACT_CONCEPT = "memory:explicit-fact"
WORKSPACE_READ_ABILITY = "ability:workspace-read"
WORKSPACE_RUN_ABILITY = "ability:workspace-run-python"
MEMORY_COMMIT_ABILITY = "ability:memory-commit"
MEMORY_RECALL_ABILITY = "ability:memory-recall"

HELP_TEXT = (
    "Talk normally. Use 'remember that ...' or /remember TEXT for durable "
    "memory, /recall QUERY for explicit evidence lookup, /open PATH to inspect a "
    "workspace folder or UTF-8 file, /run PATH to run one bounded Python file, /state, "
    "/help, or /quit."
)

_RECALL_STOP_WORDS = frozenset(
    {
        "a",
        "about",
        "an",
        "and",
        "are",
        "do",
        "for",
        "from",
        "i",
        "in",
        "is",
        "it",
        "me",
        "my",
        "of",
        "on",
        "that",
        "the",
        "to",
        "what",
        "you",
    }
)


@dataclass(frozen=True)
class DriveSeed:
    name: str
    pressure: float
    baseline_growth: float
    persistence: float
    threshold: float
    inputs: tuple[InputTrunk, ...]
    outputs: tuple[OutputTrunk, ...]
    satisfaction_gain: float = 0.70
    frustration_gain: float = 0.45


@dataclass(frozen=True)
class CompositeSeed:
    name: str
    parents: tuple[str, ...]
    pressure: float
    baseline_growth: float
    persistence: float
    threshold: float
    outputs: tuple[OutputTrunk, ...]


DRIVE_SEEDS = (
    DriveSeed(
        "inquiry",
        0.64,
        0.035,
        0.88,
        0.62,
        (InputTrunk.HEAR, InputTrunk.SEE),
        (OutputTrunk.LOOK,),
    ),
    DriveSeed(
        "integrity",
        0.48,
        0.025,
        0.91,
        0.66,
        (InputTrunk.HEAR, InputTrunk.SEE, InputTrunk.NOTICE),
        (OutputTrunk.LOOK,),
        satisfaction_gain=0.58,
    ),
    DriveSeed(
        "agency",
        0.40,
        0.030,
        0.86,
        0.65,
        (InputTrunk.SEE, InputTrunk.NOTICE),
        (OutputTrunk.DO,),
    ),
    DriveSeed(
        "connection",
        0.43,
        0.018,
        0.93,
        0.69,
        (InputTrunk.HEAR,),
        (OutputTrunk.SPEAK,),
        satisfaction_gain=0.76,
        frustration_gain=0.58,
    ),
    DriveSeed(
        "continuity",
        0.30,
        0.020,
        0.95,
        0.72,
        (InputTrunk.HEAR, InputTrunk.SEE, InputTrunk.NOTICE),
        (OutputTrunk.LOOK,),
        satisfaction_gain=0.50,
        frustration_gain=0.65,
    ),
    DriveSeed(
        "creation",
        0.24,
        0.016,
        0.84,
        0.74,
        (InputTrunk.HEAR, InputTrunk.SEE),
        (OutputTrunk.DO,),
    ),
)

COMPOSITE_SEEDS = (
    CompositeSeed(
        "careful_inquiry",
        ("inquiry", "integrity"),
        0.15,
        0.010,
        0.90,
        0.70,
        (OutputTrunk.LOOK,),
    ),
    CompositeSeed(
        "responsible_action",
        ("agency", "integrity"),
        0.13,
        0.012,
        0.88,
        0.70,
        (OutputTrunk.DO,),
    ),
    CompositeSeed(
        "bounded_connection",
        ("connection", "continuity"),
        0.11,
        0.008,
        0.94,
        0.73,
        (OutputTrunk.SPEAK,),
    ),
)


def _opaque_node_id(namespace: str, members: Sequence[str]) -> str:
    material = namespace + "\0" + "\0".join(sorted(str(item) for item in members))
    return "DYN:" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


def ensure_integrated_foundations(
    runtime: BornInHabitusRuntime,
) -> dict[str, dict[str, str]]:
    """Install auditable gestational drives without any language policy text."""
    mind = runtime.mind
    drive_ids: dict[str, str] = {}
    for seed in DRIVE_SEEDS:
        node_id = _opaque_node_id("integrated-drive-v1", (seed.name,))
        drive_ids[seed.name] = node_id
        mind.add_concept(
            node_id,
            node_id,
            input_trunks=seed.inputs,
            output_trunks=seed.outputs,
            kind="drive",
            semantic_embedding=False,
        )
        mind.recurrent.register(
            node_id,
            pressure=seed.pressure,
            baseline_growth=seed.baseline_growth,
            persistence=seed.persistence,
            satisfaction_gain=seed.satisfaction_gain,
            frustration_gain=seed.frustration_gain,
            expression_threshold=seed.threshold,
            pulse=mind.pulse,
        )

    composite_ids: dict[str, str] = {}
    for seed in COMPOSITE_SEEDS:
        parents = tuple(drive_ids[name] for name in seed.parents)
        node_id = _opaque_node_id("integrated-composite-v1", parents)
        composite_ids[seed.name] = node_id
        mind.add_concept(
            node_id,
            node_id,
            input_trunks=(InputTrunk.HEAR, InputTrunk.SEE, InputTrunk.NOTICE),
            output_trunks=seed.outputs,
            kind="desire_composite",
            semantic_embedding=False,
        )
        for parent_id in parents:
            mind.add_relation(parent_id, node_id, side="input")
        mind.recurrent.register(
            node_id,
            pressure=seed.pressure,
            baseline_growth=seed.baseline_growth,
            persistence=seed.persistence,
            expression_threshold=seed.threshold,
            pulse=mind.pulse,
        )

    mind.add_concept(
        EXPLICIT_FACT_CONCEPT,
        EXPLICIT_FACT_CONCEPT,
        input_trunks=(InputTrunk.HEAR,),
        kind="fact_anchor",
        semantic_embedding=False,
    )
    manifest = {
        "drives": drive_ids,
        "composites": composite_ids,
    }
    mind.store.set_metadata(FOUNDATION_KEY, json.dumps(manifest, sort_keys=True))
    return manifest


class CurrentEventSpeechRenderer:
    """External chat baseline with no application-supplied system prompt."""

    def __init__(self, model: ChatModel) -> None:
        self.model = model

    def messages_for(self, current_hear: str) -> tuple[ChatMessage, ...]:
        return ({"role": "user", "content": str(current_hear)},)

    def render(self, current_hear: str) -> str:
        return self.model.generate(self.messages_for(current_hear)).strip()


@dataclass(frozen=True)
class IntegratedTurn:
    kind: str
    response: str
    pulse: int
    pulse_id: str
    self_state_sha256: str
    selected_outputs: tuple[str, ...]
    dominant_desire: str | None = None
    speech_cycle_id: str | None = None
    evidence_record_ids: tuple[str, ...] = ()
    tool_receipt: ToolReceipt | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "response": self.response,
            "pulse": self.pulse,
            "pulse_id": self.pulse_id,
            "self_state_sha256": self.self_state_sha256,
            "selected_outputs": list(self.selected_outputs),
            "dominant_desire": self.dominant_desire,
            "speech_cycle_id": self.speech_cycle_id,
            "evidence_record_ids": list(self.evidence_record_ids),
            "tool_receipt": (
                self.tool_receipt.to_dict()
                if self.tool_receipt is not None
                else None
            ),
        }


class IntegratedMind:
    """Conversation, persistent state, explicit recall, and grounded abilities."""

    def __init__(
        self,
        runtime: BornInHabitusRuntime,
        model: ChatModel,
        *,
        workspace: str | Path,
        profile: GestationProfile | None = None,
        run_timeout_seconds: float | None = None,
    ) -> None:
        self.runtime = runtime
        self.mind = runtime.mind
        resolved_profile = profile or load_profile(self.mind)
        if resolved_profile is None:
            raise ValueError("the integrated mind must be gestated before use")
        self.profile = resolved_profile
        self.foundation = ensure_integrated_foundations(runtime)
        desire_names = {
            node_id: name
            for group in self.foundation.values()
            for name, node_id in group.items()
        }
        self.desire_names = desire_names
        self.renderer = CurrentEventSpeechRenderer(model)
        self.workspace = WorkspacePolicy(
            workspace,
            run_timeout_seconds=run_timeout_seconds,
        )
        self._latest_tool_return_pulse: DevelopmentalPulseReceipt | None = None
        self.tools = ToolRegistry(
            self.mind,
            pulse_kernel=self.runtime.kernel,
            pulse_advancer=self._settle_tool_return,
        )
        self._register_abilities()

    def _settle_tool_return(
        self,
        item_ids: Sequence[str],
    ) -> CognitiveCycleReceipt:
        receipt = self.runtime.advance((), pending_item_ids=item_ids)
        self._latest_tool_return_pulse = receipt
        return receipt.cycle

    def _sensory_encoder(self, ability_id: str):
        def encode(
            status: str,
            result: Any,
            error: str,
            dimension: int,
        ) -> Sequence[float]:
            payload = json.dumps(
                {
                    "ability": ability_id,
                    "status": status,
                    "result": result,
                    "error": error,
                },
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            return opaque_payload_embedding(
                payload,
                dimension,
                namespace=f"integrated-return:{ability_id}",
            )

        return encode

    def _register_abilities(self) -> None:
        definitions = (
            ToolDefinition(
                tool_id=WORKSPACE_READ_ABILITY,
                trunk=OutputTrunk.LOOK,
                label="Workspace read",
                description="List one authorized workspace folder or read one UTF-8 file.",
                terms=(),
                parameters={"path": {"type": "string"}},
                handler=_tool_handler(self.workspace.inspect_path, "path"),
                opaque=True,
                bind_to_trunk=False,
                sensory_encoder=self._sensory_encoder(WORKSPACE_READ_ABILITY),
            ),
            ToolDefinition(
                tool_id=WORKSPACE_RUN_ABILITY,
                trunk=OutputTrunk.DO,
                label="Workspace Python run",
                description="Run one explicitly named Python file under limits.",
                terms=(),
                parameters={"path": {"type": "string"}},
                handler=_tool_handler(self.workspace.run_python, "path"),
                opaque=True,
                bind_to_trunk=False,
                sensory_encoder=self._sensory_encoder(WORKSPACE_RUN_ABILITY),
            ),
            ToolDefinition(
                tool_id=MEMORY_COMMIT_ABILITY,
                trunk=OutputTrunk.DO,
                label="Explicit memory commit",
                description="Commit one user-selected statement to canonical memory.",
                terms=(),
                parameters={"fact": {"type": "string"}},
                handler=self._commit_memory,
                opaque=True,
                bind_to_trunk=False,
                sensory_encoder=self._sensory_encoder(MEMORY_COMMIT_ABILITY),
            ),
            ToolDefinition(
                tool_id=MEMORY_RECALL_ABILITY,
                trunk=OutputTrunk.LOOK,
                label="Explicit evidence recall",
                description="Inspect canonical language memories for current evidence.",
                terms=(),
                parameters={"query": {"type": "string"}},
                handler=self._recall_memory,
                opaque=True,
                bind_to_trunk=False,
                sensory_encoder=self._sensory_encoder(MEMORY_RECALL_ABILITY),
            ),
        )
        for definition in definitions:
            self.tools.register_tool(definition)
        # Abilities grow beneath motives rather than beside them. Their verified
        # output paths therefore include the desires whose pressure and valence
        # should change when the consequence comes back.
        routes = {
            WORKSPACE_READ_ABILITY: (
                self.foundation["drives"]["inquiry"],
                self.foundation["drives"]["integrity"],
                self.foundation["drives"]["continuity"],
            ),
            WORKSPACE_RUN_ABILITY: (
                self.foundation["drives"]["agency"],
                self.foundation["drives"]["creation"],
                self.foundation["composites"]["responsible_action"],
            ),
            MEMORY_COMMIT_ABILITY: (
                self.foundation["drives"]["agency"],
                self.foundation["composites"]["responsible_action"],
            ),
            MEMORY_RECALL_ABILITY: (
                self.foundation["drives"]["inquiry"],
                self.foundation["drives"]["integrity"],
                self.foundation["drives"]["continuity"],
            ),
        }
        for ability_id, motive_ids in routes.items():
            for motive_id in motive_ids:
                self.mind.add_relation(motive_id, ability_id, side="output")

    @staticmethod
    def _split_command(text: str) -> tuple[str, str] | None:
        stripped = str(text).strip()
        lowered = stripped.casefold()
        for command in (
            "/open",
            "/run",
            "/remember",
            "/recall",
            "/state",
            "/help",
        ):
            if lowered == command:
                return command, ""
            prefix = command + " "
            if lowered.startswith(prefix):
                return command, stripped[len(prefix) :].strip()
        natural = "remember that "
        if lowered.startswith(natural):
            return "/remember", stripped[len(natural) :].strip()
        return None

    @staticmethod
    def _one_path(argument: str) -> str:
        try:
            pieces = shlex.split(argument)
        except ValueError as error:
            raise ValueError(f"invalid quoted path: {error}") from error
        if len(pieces) != 1:
            raise ValueError("supply exactly one workspace path; quote paths containing spaces")
        return pieces[0]

    def _commit_memory(self, arguments: dict[str, Any]) -> dict[str, Any]:
        fact = str(arguments.get("fact", "")).strip()
        if not fact:
            raise ValueError("tell me what to remember")
        digest = hashlib.sha256(fact.casefold().encode("utf-8")).hexdigest()
        record_id = f"record:explicit-fact:{digest[:32]}"
        existing = self.mind.store.get_record(record_id)
        if existing is not None:
            if existing.text.casefold() != fact.casefold():
                raise ValueError("explicit memory identifier collision")
            return {"record_id": record_id, "fact": existing.text, "created": False}
        record = self.mind.remember(
            fact,
            kind=EventKind.MESSAGE,
            source_id=self.profile.human_name,
            event_id=f"event:explicit-fact:{digest[:32]}",
            record_id=record_id,
            record_type=RecordType.FACT,
            concept_ids=(EXPLICIT_FACT_CONCEPT,),
            provenance={"origin": "self-authorized-memory-commit"},
            metadata={
                "integrated_explicit_fact": True,
                "explicit_user_memory": True,
                "fact_sha256": hashlib.sha256(fact.encode("utf-8")).hexdigest(),
                "automatic_prompt_injection": False,
            },
            allow_growth=False,
            input_trunk=InputTrunk.HEAR,
            embedding=self.runtime.embedder.embed(fact),
            pulse_number=self.mind.pulse,
        )
        return {"record_id": record.record_id, "fact": record.text, "created": True}

    def _recall_memory(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = str(arguments.get("query", "")).strip()
        if not query:
            raise ValueError("tell me what to recall")
        query_tokens = {
            token for token in tokenize(query) if token not in _RECALL_STOP_WORDS
        }
        if not query_tokens:
            query_tokens = set(tokenize(query))
        query_embedding = self.runtime.embedder.embed(query)
        candidates: list[tuple[float, Any, bool]] = []
        for record in self.mind.store.list_active_records():
            if record.metadata.get("membrane_words") is not True:
                continue
            if record.metadata.get("integrated_command") in {"/remember", "/recall"}:
                continue
            if record.metadata.get("integrated_surface") == "deterministic-verified-result":
                continue
            record_tokens = set(tokenize(record.text))
            overlap_count = len(query_tokens & record_tokens)
            overlap = overlap_count / max(1, len(query_tokens))
            phrase = 1.0 if query.casefold() in record.text.casefold() else 0.0
            similarity = max(
                0.0,
                cosine_similarity(query_embedding, record.embedding),
            )
            lexical_match = overlap_count > 0 or phrase > 0.0
            if not lexical_match and similarity < 0.62:
                continue
            fact_boost = 0.12 if record.metadata.get("explicit_user_memory") else 0.0
            score = 0.58 * overlap + 0.24 * similarity + 0.18 * phrase + fact_boost
            candidates.append((score, record, lexical_match))
        if any(lexical_match for _, _, lexical_match in candidates):
            candidates = [item for item in candidates if item[2]]
        candidates.sort(key=lambda item: (-item[0], item[1].timestamp, item[1].record_id))
        matches = [
            {
                "record_id": record.record_id,
                "text": record.text,
                "source_id": record.source_id,
                "timestamp": record.timestamp,
                "score": round(score, 6),
            }
            for score, record, _ in candidates
        ]
        return {
            "query": query,
            "matches": matches,
            "retrieval_mode": "explicit_selected_LOOK_over_canonical_records",
            "automatic_prompt_injection": False,
        }

    @staticmethod
    def _ability_lane(trunk: OutputTrunk) -> InputTrunk:
        return {
            OutputTrunk.LOOK: InputTrunk.SEE,
            OutputTrunk.DO: InputTrunk.NOTICE,
            OutputTrunk.SPEAK: InputTrunk.HEAR,
        }[trunk]

    def _ability_opportunity(
        self,
        ability_id: str,
        arguments: Mapping[str, Any],
    ) -> DevelopmentalInput:
        definition = self.tools.get_tool(ability_id)
        if definition is None:
            raise KeyError(f"unknown integrated ability: {ability_id}")
        encoded = json.dumps(
            {"ability": ability_id, "arguments": dict(arguments)},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        # An affordance is a one-use event, not the timeless identity of its
        # command. Repeating the same exact command must therefore enqueue a
        # fresh inbox item on every later SELF pulse.
        opportunity_pulse = self.mind.pulse + 1
        return DevelopmentalInput(
            content=f"sensor:ability-opportunity:{digest}",
            lane=self._ability_lane(definition.trunk),
            source_id="integrated-command-adapter",
            episode_kind=EpisodeKind.SENSORY,
            embedding=tuple(
                opaque_payload_embedding(
                    encoded,
                    self.mind.embedder.dimension,
                    namespace="integrated-ability-opportunity",
                )
            ),
            item_id=f"ability-opportunity:{opportunity_pulse}:{digest}",
            concept_scores={ability_id: 1.0},
            metadata={
                "integrated_ability_opportunity": True,
                "ability_id": ability_id,
                "arguments": dict(arguments),
                "language_features": False,
            },
            motor_eligible=False,
            exclusive_output_node_ids=(ability_id,),
        )

    def _perceive(
        self,
        content: str,
        *,
        command_name: str | None = None,
        ability_id: str | None = None,
        arguments: Mapping[str, Any] | None = None,
    ) -> DevelopmentalPulseReceipt:
        inputs: list[DevelopmentalInput] = []
        open_speech = self.mind.open_experience_cycles(OutputTrunk.SPEAK)
        if open_speech:
            # The next actual human utterance is the terminal consequence of the
            # last delivered speech act and the current HEAR event.  Store it once.
            self.runtime.queue_return(
                open_speech[-1],
                content,
                status="human-response",
                stability_delta=0.0,
                verified=True,
                source_id=self.profile.human_name,
                embedding=self.runtime.embedder.embed(content),
                record_type=RecordType.INBOUND_MESSAGE,
                metadata={
                    "developmental_episode_kind": EpisodeKind.FUNCTIONAL_EXCHANGE.value,
                    "developmental_motor_eligible": True,
                    "integrated_current_event": True,
                    "integrated_command": command_name,
                    "delivery_channel": "terminal",
                },
            )
        else:
            inputs.append(
                DevelopmentalInput(
                    content=content,
                    lane=InputTrunk.HEAR,
                    source_id=self.profile.human_name,
                    episode_kind=EpisodeKind.FUNCTIONAL_EXCHANGE,
                    metadata={
                        "integrated_current_event": True,
                        "integrated_command": command_name,
                    },
                )
            )
        if ability_id is not None:
            inputs.append(self._ability_opportunity(ability_id, arguments or {}))
        return self.runtime.advance(tuple(inputs))

    @staticmethod
    def _affordance(
        pulse: DevelopmentalPulseReceipt,
        ability_id: str,
    ) -> OutputAffordance:
        affordance = next(
            (
                item
                for item in pulse.cycle.selected_outputs
                if item.node_id == ability_id
            ),
            None,
        )
        if affordance is None:
            raise RuntimeError(
                f"SELF did not authorize the sensed ability {ability_id}; nothing ran"
            )
        return affordance

    def _dominant_name(self, pulse: DevelopmentalPulseReceipt) -> str | None:
        node_id = pulse.cycle.recurrent.dominant_desire_id
        return self.desire_names.get(node_id) if node_id else None

    @staticmethod
    def _selected_outputs(pulse: DevelopmentalPulseReceipt) -> tuple[str, ...]:
        return tuple(
            f"{item.trunk.value}:{item.node_id}"
            for item in pulse.cycle.selected_outputs
        )

    def _actualize_speech(
        self,
        pulse: DevelopmentalPulseReceipt,
        response: str,
        *,
        surface: str,
        evidence_record_ids: Sequence[str] = (),
    ) -> str | None:
        if not any(
            item.trunk == OutputTrunk.SPEAK
            for item in pulse.cycle.selected_outputs
        ):
            return None
        cortex_state = self.runtime.cortex.latest_pulse_state()
        cycle = self.runtime.actualize(
            pulse,
            response,
            trunk=OutputTrunk.SPEAK,
            source_id="self",
            metadata={
                "integrated_surface": surface,
                "language_origin": (
                    "jointly_trained_rwkv"
                    if surface == "continual-rwkv"
                    else "external_chat_model"
                    if surface == "open-weight-current-event-renderer"
                    else "deterministic_adapter"
                ),
                "system_prompt_used": False,
                "persona_prompt_used": False,
                "state_text_prompt_used": False,
                "developmental_self_generated": False,
                "transcript_records_used": 0,
                "recalled_records_used": 0,
                "automatic_text_retrieval": False,
                "explicit_evidence_record_ids": list(evidence_record_ids),
                "self_state_sha256": pulse.cycle.self_state.state_sha256,
                "cortex_state_sha256": (
                    cortex_state.state_sha256
                    if cortex_state is not None
                    and cortex_state.pulse == pulse.cycle.self_state.pulse
                    else None
                ),
            },
        )
        return cycle.cycle_id

    @staticmethod
    def _tool_response(receipt: ToolReceipt) -> tuple[str, tuple[str, ...]]:
        if receipt.status != "success":
            return (
                f"I could not complete {receipt.tool_id}: {receipt.error}",
                (),
            )
        output = receipt.output if isinstance(receipt.output, Mapping) else {}
        if receipt.tool_id == WORKSPACE_READ_ABILITY:
            if output.get("kind") == "directory":
                rendered_entries = []
                for entry in output.get("entries", ()):
                    marker = "/" if entry.get("kind") == "directory" else ""
                    rendered_entries.append(
                        f"[{entry.get('kind', 'other')}] {entry.get('name', '')}{marker}"
                    )
                if output.get("truncated"):
                    rendered_entries.append("[listing truncated]")
                body = "\n".join(rendered_entries) or "[empty folder]"
                entry_count = int(output.get("entry_count", 0))
                entry_word = "entry" if entry_count == 1 else "entries"
                return (
                    f"Opened folder {output.get('path')} "
                    f"({entry_count} {entry_word}).\n{body}",
                    (),
                )
            content = str(output.get("content", ""))
            return (
                (
                    f"Opened {output.get('path')} ({output.get('size_bytes')} bytes, "
                    f"sha256 {output.get('sha256')}).\n{content}"
                ).rstrip(),
                (),
            )
        if receipt.tool_id == WORKSPACE_RUN_ABILITY:
            stdout = str(output.get("stdout", "")).strip()
            stderr = str(output.get("stderr", "")).strip()
            details = stdout or stderr or "[no output]"
            return (
                f"Ran {output.get('path')} successfully "
                f"(exit {output.get('returncode')}).\n{details}",
                (),
            )
        if receipt.tool_id == MEMORY_COMMIT_ABILITY:
            return f"I’ll remember: {output.get('fact')}", (str(output.get("record_id")),)
        matches = list(output.get("matches", ()))
        if not matches:
            return "I don’t have matching evidence yet.", ()
        record_ids = tuple(str(item["record_id"]) for item in matches)
        body = "\n".join(f"- {item['text']}" for item in matches)
        return f"I found this explicit evidence:\n{body}", record_ids

    def _turn_from_pulse(
        self,
        *,
        kind: str,
        response: str,
        pulse: DevelopmentalPulseReceipt,
        speech_cycle_id: str | None,
        evidence_record_ids: Sequence[str] = (),
        tool_receipt: ToolReceipt | None = None,
    ) -> IntegratedTurn:
        return IntegratedTurn(
            kind=kind,
            response=response,
            pulse=pulse.cycle.self_state.pulse,
            pulse_id=pulse.cycle.self_state.pulse_id,
            self_state_sha256=pulse.cycle.self_state.state_sha256,
            selected_outputs=self._selected_outputs(pulse),
            dominant_desire=self._dominant_name(pulse),
            speech_cycle_id=speech_cycle_id,
            evidence_record_ids=tuple(evidence_record_ids),
            tool_receipt=tool_receipt,
        )

    def _tool_turn(
        self,
        content: str,
        *,
        command_name: str,
        ability_id: str,
        arguments: dict[str, Any],
    ) -> IntegratedTurn:
        pulse = self._perceive(
            content,
            command_name=command_name,
            ability_id=ability_id,
            arguments=arguments,
        )
        affordance = self._affordance(pulse, ability_id)
        self._latest_tool_return_pulse = None
        tool_receipt = self.tools.execute(
            ability_id,
            arguments,
            affordance=affordance,
        )
        return_pulse = self._latest_tool_return_pulse
        if return_pulse is None:
            raise RuntimeError("verified tool return did not pass through the cortex")
        response, evidence_record_ids = self._tool_response(tool_receipt)
        speech_cycle_id = self._actualize_speech(
            return_pulse,
            response,
            surface="deterministic-verified-result",
            evidence_record_ids=evidence_record_ids,
        )
        return self._turn_from_pulse(
            kind="tool" if command_name in {"/open", "/run"} else command_name[1:],
            response=response if speech_cycle_id else "",
            pulse=return_pulse,
            speech_cycle_id=speech_cycle_id,
            evidence_record_ids=evidence_record_ids,
            tool_receipt=tool_receipt,
        )

    def handle(self, text: str) -> IntegratedTurn:
        content = str(text).strip()
        if not content:
            raise ValueError("message cannot be empty")
        command = self._split_command(content)
        if command is None:
            pulse = self._perceive(content)
            if not any(
                item.trunk == OutputTrunk.SPEAK
                for item in pulse.cycle.selected_outputs
            ):
                return self._turn_from_pulse(
                    kind="quiet",
                    response="",
                    pulse=pulse,
                    speech_cycle_id=None,
                )
            response = self.renderer.render(content)
            speech_cycle_id = self._actualize_speech(
                pulse,
                response,
                surface="open-weight-current-event-renderer",
            )
            return self._turn_from_pulse(
                kind="conversation" if speech_cycle_id else "quiet",
                response=response if speech_cycle_id else "",
                pulse=pulse,
                speech_cycle_id=speech_cycle_id,
            )

        name, argument = command
        if name == "/open":
            path = self._one_path(argument)
            return self._tool_turn(
                content,
                command_name=name,
                ability_id=WORKSPACE_READ_ABILITY,
                arguments={"path": path},
            )
        if name == "/run":
            path = self._one_path(argument)
            return self._tool_turn(
                content,
                command_name=name,
                ability_id=WORKSPACE_RUN_ABILITY,
                arguments={"path": path},
            )
        if name == "/remember":
            if not argument:
                raise ValueError("tell me what to remember")
            return self._tool_turn(
                content,
                command_name=name,
                ability_id=MEMORY_COMMIT_ABILITY,
                arguments={"fact": argument},
            )
        if name == "/recall":
            if not argument:
                raise ValueError("tell me what to recall")
            return self._tool_turn(
                content,
                command_name=name,
                ability_id=MEMORY_RECALL_ABILITY,
                arguments={"query": argument},
            )

        pulse = self._perceive(content, command_name=name)
        response = HELP_TEXT if name == "/help" else json.dumps(
            self.state(), indent=2, sort_keys=True
        )
        speech_cycle_id = self._actualize_speech(
            pulse,
            response,
            surface="deterministic-local-state",
        )
        return self._turn_from_pulse(
            kind=name[1:],
            response=response if speech_cycle_id else "",
            pulse=pulse,
            speech_cycle_id=speech_cycle_id,
        )

    def state(self) -> dict[str, Any]:
        snapshot = self.mind.recurrent.snapshot(pulse=self.mind.pulse)
        latest_cortex = self.runtime.cortex.latest_pulse_state()
        facts = [
            record
            for record in self.mind.store.list_active_records()
            if record.metadata.get("explicit_user_memory") is True
        ]
        tool_returns = self.mind.store.connection.execute(
            "SELECT COUNT(*) AS total FROM records WHERE record_type = ?",
            (RecordType.TOOL_RESULT.value,),
        ).fetchone()
        return {
            "agent": self.profile.agent_name,
            "human": self.profile.human_name,
            "architecture": "born-in-cortex-with-open-weight-speech-motor",
            "pulse": self.mind.pulse,
            "cortex": {
                "parameter_count": self.runtime.cortex.parameter_count,
                "architecture_id": self.runtime.cortex.config.architecture_id,
                "model_sha256": self.runtime.cortex.model_sha256,
                "device": str(self.runtime.cortex.device),
                "latest_state_sha256": (
                    latest_cortex.state_sha256 if latest_cortex is not None else None
                ),
            },
            "memory": {
                "canonical_records": len(self.mind.store.list_records()),
                "explicit_facts": len(facts),
                "automatic_text_retrieval": False,
                "transcript_window": False,
                "explicit_recall_ability": MEMORY_RECALL_ABILITY,
            },
            "desires": [
                {
                    **asdict(item),
                    "name": self.desire_names.get(item.node_id),
                }
                for item in snapshot.desires
            ],
            "dominant_desire": self.desire_names.get(
                snapshot.dominant_desire_id or ""
            ),
            "workspace": str(self.workspace.root),
            "registered_abilities": sorted(self.tools.tools),
            "verified_tool_returns": int(tool_returns["total"]),
            "open_cycles": len(self.mind.open_experience_cycles()),
            "pending_sensations": len(self.runtime.kernel.pending()),
            "graph_invariant_errors": self.mind.graph.validate_invariants(),
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the integrated born-in Habitus cortex with an open-weight "
            "current-event speech motor."
        )
    )
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--checkpoint-directory", type=Path)
    parser.add_argument("--rwkv-model", type=Path, help="use integrated continual RWKV with the full cortex")
    parser.add_argument("--unified-state", type=Path, default=Path("state/continual-mind"))
    parser.add_argument("--attach", type=Path, help="talk to an already running integrated corpus trainer")
    parser.add_argument("--memory-agent", type=Path, help="use the full cortex with semantic vaults and rolling context")
    parser.add_argument("--corpus-graph", type=Path, help="seed a new mind with a verified native corpus graph")
    parser.add_argument("--trainable-blocks", type=int, default=1)
    parser.add_argument("--human-name", default="Human")
    parser.add_argument("--agent-name", default="Habitus")
    parser.add_argument("--taste", default="builder")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--run-timeout", type=float, default=None)
    parser.add_argument(
        "--allow-gpu",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--cpu-only", action="store_true", help="explicitly force cortex and Ollama to CPU")
    parser.add_argument("--once", help="process one message and exit")
    parser.add_argument("--json", action="store_true")
    return parser


def _show(turn: IntegratedTurn, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(turn.to_dict(), indent=2, sort_keys=True, default=str))
    else:
        print(turn.response if turn.response else "[SELF remained quiet]")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.memory_agent is not None:
        from .memory_agent import main as memory_main
        forwarded = ["--directory", str(args.memory_agent)]
        if args.once is not None:
            forwarded += ["--once", args.once]
        return memory_main(forwarded)
    if args.attach is not None:
        from .continual_mind import attached_console
        return attached_console(args.attach, once=args.once, timeout=args.timeout)
    if args.rwkv_model is not None:
        from .continual_mind import console
        return console(args)
    args.database.parent.mkdir(parents=True, exist_ok=True)
    try:
        with BornInHabitusRuntime(
            args.database,
            cortex_config=CortexConfig(),
            checkpoint_directory=args.checkpoint_directory,
            device="cpu" if args.cpu_only else "auto",
            mind_dimension=256,
        ) as runtime:
            profile = load_profile(runtime.mind)
            if profile is None:
                profile = gestate(
                    runtime.mind,
                    human_name=args.human_name,
                    agent_name=args.agent_name,
                    taste_schema=args.taste,
                    model_backend="ollama",
                    model_name=args.model,
                )
            model = OllamaChatModel(
                args.model or profile.model_name,
                base_url=args.ollama_url,
                timeout_seconds=args.timeout,
                temperature=0.35,
                num_gpu=0 if args.cpu_only else None,
                think=False,
            )
            mind = IntegratedMind(
                runtime,
                model,
                workspace=args.workspace,
                profile=profile,
                run_timeout_seconds=args.run_timeout,
            )
            if args.once is not None:
                _show(mind.handle(args.once), as_json=args.json)
                return 0

            print(
                f"{profile.agent_name} is awake. Persistent mind: {args.database}\n"
                f"Born-in cortex: {runtime.cortex.parameter_count:,} parameters on "
                f"{runtime.cortex.device}\n"
                f"Authorized workspace: {mind.workspace.root}\n"
                "Commands: /remember TEXT, /recall QUERY, /open PATH, /run PATH, "
                "/state, /help, /quit"
            )
            while True:
                try:
                    message = input(f"{profile.human_name}> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    return 0
                if not message:
                    continue
                if message.casefold() in {"/quit", "/exit"}:
                    return 0
                try:
                    turn = mind.handle(message)
                except (ModelUnavailableError, OSError, RuntimeError, ValueError) as error:
                    print(f"Habitus error: {error}")
                    continue
                print(
                    f"{profile.agent_name}> "
                    f"{turn.response if turn.response else '[quiet]'}"
                )
    except (CortexDependencyError, ModelUnavailableError, OSError, RuntimeError, ValueError) as error:
        print(f"Habitus error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
