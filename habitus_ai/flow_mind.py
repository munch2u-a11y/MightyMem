"""A constant-flow Habitus mind: pretrained recurrent thought, nervous-system action.

There are no turns. Messages arrive between ticks and join a continuous flow.
Each tick is one nervous-system cycle:

1. The cortex predicts the sensation about to arrive and learns from its
   surprise before anything is admitted.
2. Recognized objects in new sensations and in the previous slice of thought
   excite their action affordances. NOTICE, SEE and HEAR events, including the
   mind's own delivered speech heard back as reafference, settle together in
   one SELF pulse.
3. SELF ranks concrete affordances grown from what the mind has observed. A
   LOOK or DO runs only through its one-use authorization and returns through
   SEE or NOTICE.
4. External text and verified returns enter the language stream once, in
   settling order. The cortex then writes its residual into the language
   layers.
5. The language core continues its private and public stream for one
   scheduler slice. Its tokens re-excite the graph, and public speech is
   delivered only under a current SPEAK authorization.

An act needs direct, recent attention to its object, never activation that
merely spread through the graph. Free-language mentions, from people or from
the mind's own thought, can direct attention and a first LOOK. Repeating an act
needs a new external mention. Running a file (DO) needs an explicit ``/run``
instruction and a verified read of the file's current bytes.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import shlex
import signal
import sys
import time
from typing import Any, Mapping

from .continual_mind import atomic_json, database_copy
from .developmental_cortex import CortexConfig
from .developmental_curriculum import EpisodeKind
from .developmental_runtime import BornInHabitusRuntime, DevelopmentalInput
from .gestation import gestate, load_profile
from .graph import SENSORY_FEATURE_KIND
from .habitus_stream import _NumericSpace, sha
from .integrated_agent import (
    MEMORY_COMMIT_ABILITY,
    MEMORY_RECALL_ABILITY,
    WORKSPACE_READ_ABILITY,
    WORKSPACE_RUN_ABILITY,
    IntegratedMind,
)
from .recurrent_flow import RecurrentFlow
from .rwkv7_language import file_sha256
from .tools import ToolDefinition
from .types import ExperienceReturn, InputTrunk, OutputDecision, OutputTrunk, RecordType


FLOW_SCHEMA = "habitus.constant-flow.v1"
READ = "read"
RUN = "run"
LANGUAGE_ORDER = {InputTrunk.NOTICE.value: 0, InputTrunk.SEE.value: 1, InputTrunk.HEAR.value: 2}
SPEECH_BOUNDARY = re.compile(r"[.!?][\"'”’)]*(?:\s|$)|\n\n")
EXTERNAL = "external_human"
REAFFERENCE = "self_speech_reafference"
TOOL_RETURN = "verified_tool_return"


@dataclass(frozen=True)
class FlowConfig:
    """Explicit decision parameters. None of them caps generation length."""

    slice_tokens: int = 32
    attention_window_pulses: int = 8
    look_threshold: float = 0.30
    do_threshold: float = 0.45
    thought_gain: float = 0.60
    orienting_gain: float = 0.60
    surprise_pressure_gain: float = 0.10
    spontaneous_thought: bool = True
    thought_refractory_pulses: int = 30
    learn_external_text: bool = True
    run_timeout_seconds: float | None = 60.0

    def __post_init__(self):
        if self.slice_tokens < 1 or self.attention_window_pulses < 1 or self.thought_refractory_pulses < 0:
            raise ValueError("slice, attention window and refractory counts must be positive")
        for name in ("look_threshold", "do_threshold", "thought_gain", "orienting_gain", "surprise_pressure_gain"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")
        if self.run_timeout_seconds is not None and self.run_timeout_seconds <= 0:
            raise ValueError("an explicit run timeout must be positive")


class TokenGeometrySpace(_NumericSpace):
    """Graph sensory geometry drawn from the language core's own token embeddings."""

    pretrained_geometry = True

    def __init__(self, core, *, space_id=None):
        super().__init__(core.embedding_dimension, core.geometry_identity)
        self.core = core
        self.space_id = space_id or (
            f"habitus-flow-token-geometry-v1:{sha(core.geometry_identity.encode())}:{self.dimension}")

    def embed(self, text):
        return self.core.token_geometry(self.core.encode(str(text)))


@dataclass(frozen=True)
class Affordance:
    node_id: str
    ability: str
    path: str
    entry_kind: str
    origin: str
    created_pulse: int
    executions: int
    last_executed_pulse: int | None
    last_external_excited_pulse: int | None
    last_internal_excited_pulse: int | None
    observed_sha256: str | None

    @property
    def trunk(self) -> OutputTrunk:
        return OutputTrunk.LOOK if self.ability == READ else OutputTrunk.DO


class _NoExternalRenderer:
    def generate(self, _):
        raise RuntimeError("the constant-flow mind has no external chat renderer")


def _append(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps({"time": time.time(), **value}, sort_keys=True, default=str, allow_nan=False) + "\n")


def normalize_workspace_path(raw: str) -> str | None:
    """A relative POSIX path inside the workspace, or None."""
    value = str(raw).strip()
    if not value:
        return None
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        return None
    parts = [part for part in path.parts if part not in {"", "."}]
    return "/".join(parts) if parts else "."


class FlowMind(IntegratedMind):
    """One continuing language, graph and cortex state, advanced by ticks."""

    def __init__(self, core, *, directory, workspace, config=None, cortex_config=None):
        self.core = core
        self.config = config or FlowConfig()
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = (self.directory / "writer.lock").open("a")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.journal = self.directory / "events.jsonl"
        self.learning_journal = self.directory / "learning.jsonl"
        working = self.directory / "mind.sqlite"
        self.generation = 0
        self.application = {
            "schema": FLOW_SCHEMA, "ticks": 0, "inputs": 0, "delivered": 0, "actions": 0,
            "inbox_line": 0, "pending_events": [], "pending_excitation": {}, "pending_pressure": {},
            "held_attention": {}, "stream_number": 0,
            "public_characters_delivered": 0, "stream_cycles": [], "stream_suppressed": False,
            "captured_stream": 0, "thought_carry": "", "idle_pulses_since_thought": 0,
        }
        restored = manifest = None
        if (self.directory / "LATEST.json").exists():
            pointer = json.loads((self.directory / "LATEST.json").read_text())
            checkpoint = self.directory / pointer["checkpoint"]
            if file_sha256(checkpoint / "manifest.json") != pointer["manifest_sha256"]:
                raise ValueError("flow checkpoint manifest hash mismatch")
            manifest = json.loads((checkpoint / "manifest.json").read_text())
            for name, digest in manifest["files"].items():
                if file_sha256(checkpoint / name) != digest:
                    raise ValueError("flow checkpoint checksum mismatch: " + name)
            if manifest["core_identity"] != core.identity:
                raise ValueError("flow checkpoint belongs to a different language core configuration")
            staged = self.directory / "restore.sqlite"
            if staged.exists():
                staged.unlink()
            database_copy(checkpoint / "mind.sqlite", staged)
            for suffix in ("", "-wal", "-shm"):
                side = Path(str(working) + suffix)
                if side.exists():
                    # Work after the last checkpoint stays inspectable, never silently merged.
                    side.replace(self.directory / f"uncommitted-{time.time_ns()}.sqlite{suffix}")
            staged.replace(working)
            restored = core.torch.load(checkpoint / "neural.pt", map_location="cpu", weights_only=True)
            self.application = manifest["application"]
            self.generation = manifest["generation"]
        elif working.exists():
            raise ValueError("an uncheckpointed flow mind exists; use a new directory or recover its artifacts")
        space_id = None
        if working.exists():
            import sqlite3
            with sqlite3.connect(working) as connection:
                row = connection.execute("SELECT value FROM metadata WHERE key='embedding_space_id'").fetchone()
                space_id = row[0] if row else None
        self.space = TokenGeometrySpace(core, space_id=space_id)
        runtime = BornInHabitusRuntime(working, cortex_config=cortex_config or CortexConfig(),
                                       # The cortex names a ROCm device "rocm"; torch reports it as "cuda".
                                       device="rocm" if core.device.type == "cuda" else core.device.type,
                                       checkpoint_directory=self.directory / "cortex",
                                       embedder=self.space)
        runtime.mind.graph.recency_strength = 0.0
        profile = load_profile(runtime.mind) or gestate(
            runtime.mind, human_name="Human", agent_name="Habitus", taste_schema="builder",
            model_backend="habitus-flow", model_name=core.identity)
        self._patterns: dict[str, tuple[re.Pattern, ...]] = {}
        super().__init__(runtime, _NoExternalRenderer(), workspace=workspace, profile=profile,
                         run_timeout_seconds=self.config.run_timeout_seconds)
        self._create_affordance_schema()
        for row in self.affordances():
            self._install_affordance(row)
        self.root_affordance = self._offer_affordance(READ, ".", entry_kind="directory", origin="workspace_root")
        core.attach(runtime)
        core.receipt_sink = lambda receipt: _append(self.learning_journal, receipt)
        self.flow = RecurrentFlow(core, mode="private")
        self.eligibility = None
        self.current_pulse = None
        if restored is not None:
            core.restore_payload(restored["core"])
            cursor = dict(restored["cursor"])
            cursor.pop("stream_limit", None)
            self.flow.cursor = cursor
            self.eligibility = restored["eligibility"]
            exact = core.numeric_states_equal(core.torch, restored["core"]["numeric"], core.dump_state())
            if not exact or core.weight_report() != manifest["weights"]:
                raise ValueError("flow restart did not reproduce the committed neural state")
            _append(self.journal, {"kind": "resumed", "checkpoint": manifest["generation"],
                                   "numeric_state_exact": exact, "weights_exact": True})
        if self.runtime.kernel.latest_state_payload() is None:
            # Attention needs a current SELF pulse to act inside.
            self.current_pulse = self.runtime.advance((), allow_idle=True)
        _append(self.journal, {"kind": "opened", "core_identity": core.identity,
            "config": asdict(self.config), "cortex_parameters": runtime.cortex.parameter_count,
            "language_parameters": core.weight_report()["language"]["parameters"],
            "generation": self.generation, "device": str(core.device)})

    # ------------------------------------------------------------ affordances

    def _create_affordance_schema(self):
        with self.mind.store.connection as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS flow_affordances (
                node_id TEXT PRIMARY KEY, ability TEXT NOT NULL, path TEXT NOT NULL,
                entry_kind TEXT NOT NULL, origin TEXT NOT NULL, created_pulse INTEGER NOT NULL,
                executions INTEGER NOT NULL DEFAULT 0, last_executed_pulse INTEGER,
                last_external_excited_pulse INTEGER, last_internal_excited_pulse INTEGER,
                observed_sha256 TEXT, UNIQUE(ability, path))""")

    @staticmethod
    def affordance_id(ability: str, path: str) -> str:
        return f"ability:flow-{ability}:" + hashlib.sha256(f"{ability}\0{path}".encode()).hexdigest()[:24]

    def affordances(self) -> tuple[Affordance, ...]:
        rows = self.mind.store.connection.execute(
            "SELECT * FROM flow_affordances ORDER BY created_pulse, node_id").fetchall()
        return tuple(Affordance(**dict(row)) for row in rows)

    def affordance(self, node_id: str) -> Affordance | None:
        row = self.mind.store.connection.execute(
            "SELECT * FROM flow_affordances WHERE node_id = ?", (node_id,)).fetchone()
        return None if row is None else Affordance(**dict(row))

    def _offer_affordance(self, ability, path, *, entry_kind, origin) -> str:
        node_id = self.affordance_id(ability, path)
        with self.mind.store.connection as connection:
            connection.execute("""INSERT OR IGNORE INTO flow_affordances
                (node_id, ability, path, entry_kind, origin, created_pulse) VALUES (?, ?, ?, ?, ?, ?)""",
                (node_id, ability, path, entry_kind, origin, self.mind.pulse))
            if entry_kind != "unknown":
                connection.execute("UPDATE flow_affordances SET entry_kind = ? WHERE node_id = ?",
                                   (entry_kind, node_id))
        self._install_affordance(self.affordance(node_id))
        return node_id

    def _install_affordance(self, row: Affordance):
        if row.node_id in self.tools.tools:
            return
        if row.ability == READ:
            handler = lambda _arguments, path=row.path: self.workspace.inspect_path(path)
            motives = (self.foundation["drives"]["inquiry"], self.foundation["drives"]["integrity"],
                       self.foundation["drives"]["continuity"])
        else:
            handler = lambda _arguments, path=row.path: self._run_observed(path)
            motives = (self.foundation["drives"]["agency"], self.foundation["drives"]["creation"],
                       self.foundation["composites"]["responsible_action"])
        self.tools.register_tool(ToolDefinition(
            tool_id=row.node_id, trunk=row.trunk, label=row.node_id,
            description=f"{row.ability} one workspace object the mind has encountered",
            terms=(), parameters={}, handler=handler, opaque=True, bind_to_trunk=False,
            sensory_encoder=self._sensory_encoder(row.node_id)))
        # Concrete abilities grow beneath the motives whose outcomes they change.
        for motive in motives:
            self.mind.add_relation(motive, row.node_id, side="output")

    def _run_observed(self, path):
        reader = self.affordance(self.affordance_id(READ, path))
        current = hashlib.sha256(self.workspace.resolve_file(path).read_bytes()).hexdigest()
        if reader is None or reader.observed_sha256 != current:
            raise RuntimeError("the file's current bytes have not been read; read it before running it")
        return self.workspace.run_python(path)

    def _recognize(self, text, *, exclude_paths=()) -> dict[str, float]:
        """Exact byte-form recognition of observed object names in a sensation.

        Only LOOK affordances are recognized this way; running needs an explicit instruction.
        """
        found = {}
        for row in self.affordances():
            if row.ability != READ or row.path == "." or row.path in exclude_paths:
                continue
            patterns = self._patterns.get(row.node_id)
            if patterns is None:
                forms = [row.path]
                name = PurePosixPath(row.path).name
                if name != row.path:
                    forms.append(name)
                patterns = tuple(re.compile(r"(?<![A-Za-z0-9_./-])" + re.escape(form) + r"(?![A-Za-z0-9_/-])")
                                 for form in forms if len(form) >= 2)
                self._patterns[row.node_id] = patterns
            for index, pattern in enumerate(patterns):
                if pattern.search(text):
                    found[row.node_id] = max(found.get(row.node_id, 0.0), 1.0 if index == 0 else 0.8)
        return found

    def _mark_excited(self, node_ids, column):
        if not node_ids:
            return
        with self.mind.store.connection as connection:
            connection.executemany(f"UPDATE flow_affordances SET {column} = ? WHERE node_id = ?",
                                   [(self.mind.pulse, node_id) for node_id in node_ids])

    # ---------------------------------------------------------------- returns

    def _sensory_encoder(self, ability_id):
        def encode(status, result, error, dimension):
            return self.space.embed(self.describe_return(ability_id, status, result, error))
        return encode

    def describe_return(self, ability_id, status, result, error) -> str:
        """A faithful, unabridged account of what the environment returned."""
        row = self.affordance(ability_id) if ability_id.startswith("ability:flow-") else None
        output = result if isinstance(result, Mapping) else {}
        if row is not None or ability_id in {WORKSPACE_READ_ABILITY, WORKSPACE_RUN_ABILITY}:
            reading = (row.ability == READ) if row is not None else ability_id == WORKSPACE_READ_ABILITY
            path = output.get("path") or (row.path if row is not None else "")
            label = "workspace read" if reading else "workspace run"
            if status != "success":
                return f"{label} of {path or 'the requested path'} failed: {error}"
            if reading and output.get("kind") == "directory":
                lines = [f"workspace folder {path} ({output.get('entry_count', 0)} entries)"]
                for entry in output.get("entries", ()):
                    if entry.get("kind") == "directory":
                        lines.append(f"- {entry.get('name')}/ (folder)")
                    elif entry.get("kind") == "file":
                        lines.append(f"- {entry.get('name')} (file, {entry.get('size_bytes')} bytes)")
                    else:
                        lines.append(f"- {entry.get('name')} ({entry.get('kind')})")
                return "\n".join(lines)
            if reading:
                return f"workspace file {path} ({output.get('size_bytes')} bytes)\n{output.get('content', '')}"
            return (f"ran {path}: exit {output.get('returncode')}\nstdout:\n{output.get('stdout', '')}"
                    f"\nstderr:\n{output.get('stderr', '')}")
        if ability_id == MEMORY_COMMIT_ABILITY:
            return (f"memory commit: {output.get('fact')}" if status == "success"
                    else f"memory commit failed: {error}")
        if ability_id == MEMORY_RECALL_ABILITY:
            if status != "success":
                return f"memory recall failed: {error}"
            matches = list(output.get("matches", ()))
            if not matches:
                return f"memory recall for {output.get('query')}: no matching records"
            return "\n".join([f"memory recall for {output.get('query')}:",
                              *(f"- {item['text']}" for item in matches)])
        return json.dumps({"ability": ability_id, "status": status, "result": result, "error": error},
                          sort_keys=True, default=str)

    # ----------------------------------------------------------------- inputs

    def receive(self, text, *, source=None) -> dict:
        """Queue one external message; it joins the flow at the next tick."""
        content = str(text).strip()
        if not content:
            raise ValueError("a message cannot be empty")
        self.application["inputs"] += 1
        event = {"lane": InputTrunk.HEAR.value, "origin": EXTERNAL, "language": True,
                 "source": source or self.profile.human_name, "text": content,
                 "input_number": self.application["inputs"]}
        command = self._split_command(content)
        if command is not None:
            event["command"] = list(command)
        self.application["pending_events"].append(event)
        _append(self.journal, {"kind": "received", "input_number": event["input_number"],
                               "text": content, "source": event["source"]})
        return event

    def _explicit_commands(self, events):
        """Explicit instructions offer their objects and hold attention on them until acted on."""
        explicit = None
        run_paths = set()
        held = self.application["held_attention"]
        until = self.mind.pulse + self.config.attention_window_pulses
        for event in events:
            name, argument = event.get("command") or (None, None)
            if name in {"/open", "/run"}:
                try:
                    path = normalize_workspace_path(self._one_path(argument))
                except ValueError:
                    path = None
                if path is None:
                    continue
                # A person pointing at an object offers it; only a verified read establishes it.
                reader = self._offer_affordance(READ, path, entry_kind="unknown", origin="human_pointing")
                if name == "/open" or self.affordance(reader).observed_sha256 is None:
                    held[reader] = until
                if name == "/run":
                    held[self._offer_affordance(RUN, path, entry_kind="file", origin="human_pointing")] = until
                    run_paths.add(path)
            elif name == "/remember" and argument and explicit is None:
                explicit = (MEMORY_COMMIT_ABILITY, {"fact": argument})
            elif name == "/recall" and argument and explicit is None:
                explicit = (MEMORY_RECALL_ABILITY, {"query": argument})
        for node_id, expiry in list(held.items()):
            if expiry < self.mind.pulse:
                del held[node_id]
        return explicit, run_paths, {node_id: 1.0 for node_id in held}

    def _add_surprise_pressure(self, pressure, surprise):
        inquiry = self.foundation["drives"]["inquiry"]
        pressure[inquiry] = min(1.0, pressure.get(inquiry, 0.0) + self.config.surprise_pressure_gain * surprise)

    # ------------------------------------------------------------------ tick

    def tick(self) -> dict:
        began = time.monotonic()
        app = self.application
        app["ticks"] += 1
        events = list(app["pending_events"])
        app["pending_events"] = []
        receipt = {"kind": "tick", "tick": app["ticks"], "public_emissions": [], "action": None,
                   "events": [{key: event.get(key) for key in ("lane", "origin", "input_number")} for event in events]}

        explicit, run_paths, excitation = self._explicit_commands(events)

        # 1. The cortex predicts what arrives before any of it is admitted.
        sensed = [event for event in events if event["origin"] != REAFFERENCE]
        surprise = None
        if sensed:
            surprise = self.core.learn_expectation([self.space.embed(event["text"]) for event in sensed],
                evidence={"events": len(sensed), "lanes": sorted({event["lane"] for event in sensed})})
            receipt["surprise"] = surprise

        # 2. Attention: recognized objects, the last slice of thought, and novelty.
        external_nodes = set(excitation)
        for node_id, value in app["pending_excitation"].items():
            excitation[node_id] = max(excitation.get(node_id, 0.0), float(value))
        internal_nodes = set(app["pending_excitation"])
        app["pending_excitation"] = {}
        for event in events:
            external = event["origin"] == EXTERNAL
            gain = 1.0 if external else self.config.thought_gain
            for node_id, score in self._recognize(event["text"], exclude_paths=run_paths).items():
                excitation[node_id] = max(excitation.get(node_id, 0.0), gain * score)
                (external_nodes if external else internal_nodes).add(node_id)
        root = self.affordance(self.root_affordance)
        if root.executions == 0 and self.config.orienting_gain > 0:
            excitation[root.node_id] = max(excitation.get(root.node_id, 0.0), self.config.orienting_gain)
            internal_nodes.add(root.node_id)
        pressure = dict(app["pending_pressure"])
        app["pending_pressure"] = {}
        if surprise is not None:
            self._add_surprise_pressure(pressure, surprise["surprise"])
        if excitation or pressure:
            self.runtime.kernel.advance_internal_output(excitation=excitation, pressure_delta=pressure,
                kind="flow_attention", metadata={"origin": "flow_attention", "external_evidence": False,
                                                 "excited_node_ids": sorted(excitation)})
            self._mark_excited(external_nodes, "last_external_excited_pulse")
            self._mark_excited(internal_nodes - external_nodes, "last_internal_excited_pulse")
            receipt["attention"] = {key: round(value, 6) for key, value in sorted(excitation.items())}

        # 3. One SELF pulse settles every membrane together.
        inputs = []
        open_speech = self.mind.open_experience_cycles(OutputTrunk.SPEAK)
        human_return_queued = False
        for event in events:
            if event["origin"] == EXTERNAL:
                metadata = {"flow_current_event": True, "origin": EXTERNAL,
                            "integrated_command": (event.get("command") or [None])[0]}
                if open_speech and not human_return_queued:
                    # The next human utterance is the consequence of the last speech act.
                    self.runtime.queue_return(open_speech[-1], event["text"], status="human-response",
                        stability_delta=0.0, verified=True, source_id=event["source"],
                        embedding=self.space.embed(event["text"]), record_type=RecordType.INBOUND_MESSAGE,
                        metadata={**metadata, "developmental_episode_kind": EpisodeKind.FUNCTIONAL_EXCHANGE.value,
                                  "developmental_motor_eligible": True})
                    human_return_queued = True
                else:
                    inputs.append(DevelopmentalInput(event["text"], InputTrunk.HEAR, event["source"],
                                                     EpisodeKind.FUNCTIONAL_EXCHANGE, metadata=metadata))
            elif event["origin"] == REAFFERENCE:
                inputs.append(DevelopmentalInput(event["text"], InputTrunk.HEAR, "self", EpisodeKind.SENSORY,
                    metadata={"origin": REAFFERENCE, "external_evidence": False,
                              "speech_cycle_id": event.get("speech_cycle_id")}, motor_eligible=False))
        if explicit is not None:
            inputs.append(self._ability_opportunity(*explicit))
        pulse = self.runtime.advance(tuple(inputs), allow_idle=True)
        self.current_pulse = pulse
        if human_return_queued:
            self._settle_stream_cycles(open_speech)
        receipt["pulse"] = pulse.cycle.self_state.pulse
        receipt["selected_outputs"] = [f"{item.trunk.value}:{item.node_id}" for item in pulse.cycle.selected_outputs]

        # 4. External text reaches the language stream once, in settling order.
        fed = [self._feed_language(event) for event in sorted(
            (event for event in events if event.get("language")), key=lambda item: LANGUAGE_ORDER[item["lane"]])]

        # 5. The nervous system acts; the verified return enters language at once.
        if explicit is not None:
            affordance = next((item for item in pulse.cycle.selected_outputs if item.node_id == explicit[0]), None)
            if affordance is not None:
                receipt["action"] = self._execute(affordance, explicit[1], row=None, pulse=pulse)
            else:
                receipt["action"] = {"authorized": False, "ability": explicit[0]}
        else:
            choice = self._select_action(pulse)
            if choice is not None:
                affordance, row, activation = choice
                receipt["action"] = {**self._execute(affordance, {}, row=row, pulse=pulse), "activation": activation}
        if receipt["action"] and receipt["action"].get("fed"):
            fed.append(receipt["action"]["fed"])

        # 6. The cortex writes its residual, then language continues one slice.
        self.core.begin_slice()
        if self.flow.phase == "idle" and not fed:
            app["idle_pulses_since_thought"] += 1
            if self._wants_spontaneous_thought():
                self._capture_stream()
                self.flow.wake()
                self._begin_stream()
                receipt["spontaneous_thought"] = True
        generation = None
        if self.flow.phase != "idle":
            app["idle_pulses_since_thought"] = 0
            before = len(self.flow.cursor["generated_ids"])
            generation = self.flow.pulse(maximum_tokens=self.config.slice_tokens)
            if generation["stop"] == "generated_user_content":
                app["stream_suppressed"] = True
            ids = self.flow.cursor["generated_ids"][before:]
            self._excite_from_language(ids)
            receipt["generation"] = {key: generation[key] for key in
                ("phase_before", "phase_after", "stop", "generated_tokens", "private_tokens")}
        receipt["public_emissions"] = self._deliver_public()
        receipt["fed"] = fed
        receipt["quiet"] = not events and not receipt["action"] and generation is None and not receipt["public_emissions"]
        receipt["elapsed_seconds"] = time.monotonic() - began
        if not receipt["quiet"]:
            _append(self.journal, receipt)
        return receipt

    def _settle_stream_cycles(self, previous):
        last = self.mind.store.get_experience_cycle(previous[-1].cycle_id)
        if not last.terminal_return_record_id:
            return
        for cycle in previous[:-1]:
            current = self.mind.store.get_experience_cycle(cycle.cycle_id)
            if current.status == "open":
                # One admitted human return settles every chunk of the stream it answered.
                self.mind.store.add_experience_return(ExperienceReturn(cycle.cycle_id,
                    last.terminal_return_record_id, InputTrunk.HEAR, "human-response-to-stream",
                    0.0, True, True, self.mind.pulse, {"same_admitted_human_return": True,
                                                       "extra_input_admissions": 0}))

    def _begin_stream(self):
        app = self.application
        app["stream_number"] += 1
        app["public_characters_delivered"] = 0
        app["stream_cycles"] = []
        app["stream_suppressed"] = False
        app["thought_carry"] = ""
        self.core.public_candidate = None

    def _capture_stream(self):
        """Keep an interrupted or superseded stream inspectable; it is never delivered."""
        app = self.application
        if not self.flow.cursor["generated_ids"] or app["captured_stream"] == app["stream_number"]:
            return
        app["captured_stream"] = app["stream_number"]
        text = self.flow._text()
        end = self.flow.cursor["private_end"]
        private = text if end is None else text[:end]
        _append(self.directory / "private-diagnostics.jsonl", {
            "origin": "language_core_generated_unverified", "stream": app["stream_number"],
            "phase": self.flow.phase, "tokens": len(self.flow.cursor["generated_ids"]),
            "text": private, "external_evidence": False, "replayed_as_input": False})
        public = self.flow._public_text(text).split("\n\nUser:", 1)[0].lstrip()
        remainder = public[app["public_characters_delivered"]:]
        if remainder.strip():
            _append(self.journal, {"kind": "undelivered_public_text", "stream": app["stream_number"],
                                   "text": remainder, "phase": self.flow.phase})

    def _feed_language(self, event) -> dict:
        self._capture_stream()
        text = event["text"] if event["origin"] == EXTERNAL else "Function output:\n" + event["text"]
        self.core.source = event["origin"]
        self.core.training_enabled = self.config.learn_external_text
        record = self.flow.hear(text)
        self._begin_stream()
        return {"lane": event["lane"], "origin": event["origin"], "input_tokens": record["input_tokens"],
                "interrupted_phase": record["interrupted_phase"],
                "undelivered_draft_tokens": record["undelivered_draft_tokens"]}

    def _attended(self, row) -> bool:
        """Direct attention since the last act, within the window. Spread activation never counts."""
        after = -1 if row.last_executed_pulse is None else row.last_executed_pulse

        def recent(pulse):
            return pulse is not None and pulse > after and self.mind.pulse - pulse <= self.config.attention_window_pulses

        if row.ability == RUN or row.executions:
            # Repeating an act, or running code at all, needs a new external instruction or mention.
            return recent(row.last_external_excited_pulse)
        return recent(row.last_external_excited_pulse) or recent(row.last_internal_excited_pulse)

    def _select_action(self, pulse):
        states = {state.node_id: state.activation for state in pulse.cycle.recurrent.node_states}
        options = []
        for affordance in pulse.cycle.selected_outputs:
            if affordance.trunk not in {OutputTrunk.LOOK, OutputTrunk.DO}:
                continue
            row = self.affordance(affordance.node_id)
            if row is None or not self._attended(row):
                continue
            activation = states.get(row.node_id, 0.0)
            threshold = self.config.look_threshold if row.trunk == OutputTrunk.LOOK else self.config.do_threshold
            if activation < threshold:
                continue
            if row.ability == RUN:
                reader = self.affordance(self.affordance_id(READ, row.path))
                if reader is None or reader.observed_sha256 is None:
                    continue
            unread = row.ability == READ and row.observed_sha256 is None and row.entry_kind != "directory"
            options.append(((0 if unread else 1, -activation, row.node_id), affordance, row, activation))
        if not options:
            return None
        _, affordance, row, activation = min(options, key=lambda item: item[0])
        return affordance, row, activation

    def _execute(self, affordance, arguments, *, row, pulse) -> dict:
        eligibility = {"state": self.core.dump_state(),
                       "graph": tuple(value.detach().cpu() for value in self.core._graph_vectors())}
        self._latest_tool_return_pulse = None
        tool_receipt = self.tools.execute(affordance.node_id, dict(arguments), affordance=affordance)
        return_pulse = self._latest_tool_return_pulse
        if return_pulse is None:
            raise RuntimeError("the verified return did not pass through SELF")
        success = tool_receipt.status == "success"
        if row is not None:
            output = tool_receipt.output if isinstance(tool_receipt.output, Mapping) else {}
            with self.mind.store.connection as connection:
                connection.execute("""UPDATE flow_affordances SET executions = executions + 1,
                    last_executed_pulse = ? WHERE node_id = ?""", (pulse.cycle.self_state.pulse, row.node_id))
                if success and row.ability == READ and output.get("kind") == "file":
                    connection.execute("UPDATE flow_affordances SET entry_kind = 'file', observed_sha256 = ? "
                                       "WHERE node_id = ?", (output.get("sha256"), row.node_id))
            if success and row.ability == READ and output.get("kind") == "directory":
                self._grow_from_listing(row, output)
        self.application["held_attention"].pop(affordance.node_id, None)
        text = self.describe_return(affordance.node_id, tool_receipt.status, tool_receipt.output, tool_receipt.error)
        lane = self._ability_lane(affordance.trunk).value
        # The return is predicted from the state that chose the act, before it happened.
        expectation = self.core.learn_expectation([self.space.embed(text)], eligibility=eligibility,
            evidence={"events": 1, "lanes": [lane], "receipt_id": tool_receipt.receipt_id})
        if expectation is not None:
            self._add_surprise_pressure(self.application["pending_pressure"], expectation["surprise"])
        learning = self.core.learn_action(affordance.trunk, success=success,
                                          receipt_id=tool_receipt.receipt_id, eligibility=eligibility)
        fed = self._feed_language({"lane": lane, "origin": TOOL_RETURN, "text": text})
        self.current_pulse = return_pulse
        self.application["actions"] += 1
        return {"authorized": True, "receipt": tool_receipt.to_dict(), "path": None if row is None else row.path,
                "ability": None if row is None else row.ability, "learning_update": learning["update"],
                "surprise": None if expectation is None else expectation["surprise"],
                "return_text_sha256": hashlib.sha256(text.encode()).hexdigest(), "fed": fed}

    def _grow_from_listing(self, row, output):
        base = "" if row.path == "." else row.path
        with self.mind.store.connection as connection:
            connection.execute("UPDATE flow_affordances SET entry_kind = 'directory' WHERE node_id = ?", (row.node_id,))
        for entry in output.get("entries", ()):
            name, kind = entry.get("name"), entry.get("kind")
            if not name or kind not in {"file", "directory"}:
                continue
            child = f"{base}/{name}" if base else name
            self._offer_affordance(READ, child, entry_kind=kind, origin="observed_listing")
            if kind == "file" and name.endswith(".py"):
                self._offer_affordance(RUN, child, entry_kind="file", origin="observed_listing")

    def _excite_from_language(self, ids):
        if not ids:
            return
        app = self.application
        text = self.core.decode([token for token in ids if token != self.core.eos_token_id])
        window = app["thought_carry"] + text
        for node_id, score in self._recognize(window).items():
            value = self.config.thought_gain * score
            app["pending_excitation"][node_id] = max(app["pending_excitation"].get(node_id, 0.0), value)
        app["thought_carry"] = window[-256:]
        features = self.mind.store.list_concepts(kind=SENSORY_FEATURE_KIND)
        if not features:
            return
        vector = self.core.token_geometry(ids)
        excitation = {}
        for feature in features:
            score = max(0.0, sum(a * b for a, b in zip(vector, feature.embedding))) ** 2
            if score > 0.0:
                excitation[feature.concept_id] = score
        if excitation:
            self.runtime.kernel.advance_internal_output(excitation=excitation, kind="flow_language_activity",
                metadata={"origin": "language_core_generated_unverified", "external_evidence": False,
                          "tokens": len(ids)})

    def _wants_spontaneous_thought(self) -> bool:
        if not self.config.spontaneous_thought or self.current_pulse is None:
            return False
        if self.application["idle_pulses_since_thought"] < self.config.thought_refractory_pulses:
            return False
        cycle = self.current_pulse.cycle
        return cycle.recurrent.should_express and any(item.trunk == OutputTrunk.SPEAK for item in cycle.selected_outputs)

    def _deliver_public(self) -> list:
        app = self.application
        if not self.flow.cursor["generated_ids"] or app["stream_suppressed"] or self.current_pulse is None:
            return []
        public = self.flow._public_text(self.flow._text()).split("\n\nUser:", 1)[0].lstrip()
        start = app["public_characters_delivered"]
        complete = self.flow.phase == "idle"
        if complete:
            end = len(public.rstrip())
        else:
            boundaries = [match.end() for match in SPEECH_BOUNDARY.finditer(public) if match.end() > start]
            end = boundaries[-1] if boundaries else start
        chunk = public[start:end].strip() if end > start else ""
        if not chunk:
            return []
        if not any(item.trunk == OutputTrunk.SPEAK for item in self.current_pulse.cycle.selected_outputs):
            return []
        cortex_state = self.runtime.cortex.latest_pulse_state()
        try:
            cycle = self.runtime.actualize(self.current_pulse, chunk, trunk=OutputTrunk.SPEAK, source_id="self",
                metadata={"integrated_surface": "flow-language-core", "language_core_identity": self.core.identity,
                          "language_origin": "pretrained_language_core_with_cortex_residual",
                          "system_prompt_used": False, "persona_prompt_used": False,
                          "transcript_records_used": 0, "automatic_text_retrieval": False,
                          "developmental_self_generated": False, "stream": app["stream_number"],
                          "self_state_sha256": self.current_pulse.cycle.self_state.state_sha256,
                          "cortex_state_sha256": None if cortex_state is None else cortex_state.state_sha256})
        except ValueError as error:
            if "already" not in str(error) and "superseded" not in str(error):
                raise
            # This pulse's SPEAK authorization was already used; the text waits for the next.
            return []
        app["public_characters_delivered"] = end
        app["delivered"] += 1
        app["stream_cycles"].append(cycle.cycle_id)
        self._update_eligibility(public, end, cycle.cycle_id)
        app["pending_events"].append({"lane": InputTrunk.HEAR.value, "origin": REAFFERENCE, "source": "self",
                                      "text": chunk, "language": False, "speech_cycle_id": cycle.cycle_id})
        return [{"text": chunk, "origin": "language_core_public_output", "input_number": app["inputs"],
                 "stream": app["stream_number"], "speech_cycle_id": cycle.cycle_id, "stream_complete": complete}]

    def _update_eligibility(self, public, end, cycle_id):
        candidate = self.core.public_candidate
        if not candidate:
            return
        raw = [self.core.token_bytes(token) for token in candidate["tokens"]]
        full = b"".join(raw)
        prefix = public[:end].encode()
        offset = full.find(prefix)
        limit = offset + len(prefix) if offset >= 0 else 0
        size = count = 0
        for item in raw:
            if size + len(item) > limit:
                break
            size += len(item)
            count += 1
        self.eligibility = {**candidate, "tokens": candidate["tokens"][:count], "cycle_id": cycle_id,
                            "cycles": list(self.application["stream_cycles"]),
                            "text_sha256": hashlib.sha256(prefix).hexdigest()}

    # ------------------------------------------------------------- operations

    def feedback(self, reward, explanation) -> dict:
        evidence = {"origin": "external_teacher_assessment", "explanation": str(explanation),
                    "speech_cycle_id": None if self.eligibility is None else self.eligibility["cycle_id"]}
        result = self.core.reinforce(self.eligibility, float(reward), evidence=evidence)
        if result["applied"]:
            assessment = self.mind.remember(str(explanation), source_id="local_teacher",
                record_type=RecordType.RECEIPT, allow_growth=False, input_trunk=InputTrunk.HEAR,
                metadata={"origin": "explicit_teacher_assessment", "reward": float(reward),
                          "training_annotation": True})
            edges = tuple(dict.fromkeys(edge for cycle_id in self.eligibility.get("cycles", [self.eligibility["cycle_id"]])
                                        for edge in self.mind.store.get_experience_cycle(cycle_id).credited_edge_ids))
            self.mind.record_outcome(OutputDecision(self.eligibility["cycle_id"], OutputTrunk.SPEAK, 1.0, None),
                stability_delta=float(reward), verified=True, receipt_id=assessment.record_id,
                metadata=evidence, credited_edge_ids=edges)
            self.eligibility = None
        _append(self.journal, {"kind": "feedback", "assessment": evidence, "result": result})
        return result

    def save(self, *, label="checkpoint") -> dict:
        import sqlite3
        self.generation += 1
        directory = self.directory / f"checkpoint-{self.generation:06d}"
        directory.mkdir(exist_ok=False)
        self.core.detach_state()
        self.core.torch.save({"core": self.core.checkpoint_payload(), "cursor": self.flow.cursor,
                              "eligibility": self.eligibility}, directory / "neural.pt")
        with sqlite3.connect(directory / "mind.sqlite") as destination:
            self.mind.store.connection.backup(destination)
        manifest = {"schema": FLOW_SCHEMA, "generation": self.generation, "label": label,
                    "application": self.application, "config": asdict(self.config),
                    "core_identity": self.core.identity, "weights": self.core.weight_report(),
                    "counters": dict(self.core.counters),
                    "files": {name: file_sha256(directory / name) for name in ("neural.pt", "mind.sqlite")},
                    "numeric_state_sha256": hashlib.sha256(self.core.dump_state()).hexdigest(),
                    "time": time.time()}
        atomic_json(directory / "manifest.json", manifest)
        atomic_json(self.directory / "LATEST.json", {"checkpoint": directory.name,
                                                     "manifest_sha256": file_sha256(directory / "manifest.json")})
        _append(self.journal, {"kind": "checkpoint", "checkpoint": directory.name, "label": label})
        return manifest

    def status(self) -> dict:
        return {"ticks": self.application["ticks"], "inputs": self.application["inputs"],
                "delivered": self.application["delivered"], "actions": self.application["actions"],
                "phase": self.flow.phase, "pulse": self.mind.pulse, "affordances": len(self.affordances()),
                "pending_events": len(self.application["pending_events"]), "counters": dict(self.core.counters),
                "core_identity": self.core.identity, "device": str(self.core.device)}

    def close(self):
        self.runtime.close()
        fcntl.flock(self.lock, fcntl.LOCK_UN)
        self.lock.close()


# ---------------------------------------------------------------- the worker

def process_inbox(mind: FlowMind) -> int:
    """Admit newline-terminated requests from ``inbox.jsonl`` exactly once."""
    inbox = mind.directory / "inbox.jsonl"
    if not inbox.exists():
        return 0
    raw = inbox.read_text()
    lines = raw.splitlines() if raw.endswith("\n") else raw.splitlines()[:-1]
    handled = 0
    for index, line in enumerate(lines):
        if index < mind.application["inbox_line"]:
            continue
        request = json.loads(line)
        result = None
        if request["kind"] == "input":
            result = mind.receive(request["text"], source=request.get("source"))["input_number"]
        elif request["kind"] == "feedback":
            result = mind.feedback(float(request["reward"]), request["explanation"])
        elif request["kind"] == "save":
            result = mind.save(label="requested-checkpoint")["generation"]
        else:
            raise ValueError("unknown flow request kind")
        mind.application["inbox_line"] = index + 1
        handled += 1
        if request.get("request_id"):
            _append(mind.journal, {"kind": "request_ack", "request_id": request["request_id"],
                                   "request_kind": request["kind"], "result": result})
    return handled


def run_worker(mind: FlowMind, *, idle_seconds=2.0, checkpoint_seconds=1800.0, plan=None,
               settle_quiet_ticks=3, should_stop=lambda: False) -> str:
    """Advance the mind continuously. A plan injects inputs at tick numbers and
    ends the run once its last input has been answered and the flow is quiet."""
    events = sorted((plan or {}).get("events", ()), key=lambda item: int(item["at_tick"]))
    cursor = 0
    quiet = 0
    last_checkpoint = time.monotonic()
    started_tick = mind.application["ticks"]
    while not should_stop() and not (mind.directory / "STOP").exists():
        process_inbox(mind)
        elapsed_ticks = mind.application["ticks"] - started_tick
        while cursor < len(events) and int(events[cursor]["at_tick"]) <= elapsed_ticks:
            mind.receive(events[cursor]["input"])
            cursor += 1
        receipt = mind.tick()
        for emission in receipt["public_emissions"]:
            print(json.dumps({"speech": emission["text"], "stream": emission["stream"]}), flush=True)
        if receipt["action"] and receipt["action"].get("receipt"):
            action = receipt["action"]
            print(json.dumps({"action": action["receipt"]["tool_id"], "path": action.get("path"),
                              "status": action["receipt"]["status"]}), flush=True)
        atomic_json(mind.directory / "progress.json", {"status": "running", **mind.status()})
        quiet = quiet + 1 if receipt["quiet"] else 0
        if plan is not None and cursor == len(events) and quiet >= settle_quiet_ticks:
            return "plan-complete"
        if receipt["quiet"] and plan is None:
            time.sleep(idle_seconds)
        if time.monotonic() - last_checkpoint >= checkpoint_seconds:
            mind.save(label="periodic")
            last_checkpoint = time.monotonic()
    return "stopped"


def attach(directory, *, once=None, wait=None) -> int:
    """Talk to a running worker through its inbox; never load a second mind."""
    import select
    import uuid
    directory = Path(directory).resolve()
    journal = directory / "events.jsonl"
    offset = journal.stat().st_size if journal.exists() else 0

    def submit(text):
        request = {"kind": "input", "text": text, "request_id": uuid.uuid4().hex}
        if text.startswith("/feedback "):
            _, reward, explanation = text.split(" ", 2)
            request.update(kind="feedback", reward=float(reward), explanation=explanation)
        elif text == "/save":
            request["kind"] = "save"
        with (directory / "inbox.jsonl").open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            stream.write(json.dumps(request) + "\n")
            fcntl.flock(stream, fcntl.LOCK_UN)

    if once is not None:
        submit(once)
    began = time.monotonic()
    print("Attached to the flowing mind. /quit detaches; the mind keeps running.", flush=True)
    while True:
        if once is None and select.select([sys.stdin], [], [], 0.1)[0]:
            line = sys.stdin.readline()
            if not line or line.strip() == "/quit":
                return 0
            if line.strip():
                submit(line.strip())
        if journal.exists():
            with journal.open() as stream:
                stream.seek(offset)
                while (line := stream.readline()).endswith("\n"):
                    offset = stream.tell()
                    event = json.loads(line)
                    if event.get("action") and event["action"].get("receipt"):
                        print(f"[{event['action']['receipt']['tool_id']} {event['action'].get('path')}: "
                              f"{event['action']['receipt']['status']}]", flush=True)
                    for emission in event.get("public_emissions", ()):
                        print(emission["text"], flush=True)
                        if once is not None and emission.get("stream_complete"):
                            return 0
        if once is not None and wait is not None and time.monotonic() - began >= wait:
            print("The message is queued; no complete reply arrived within this client's wait.", flush=True)
            return 2
        if once is not None:
            time.sleep(0.1)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run or attach to the constant-flow Habitus mind.")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="start or resume a flowing mind")
    core = run.add_mutually_exclusive_group(required=True)
    core.add_argument("--model", type=Path, help="pinned RWKV-7 G1j checkpoint directory")
    core.add_argument("--byte-core", action="store_true", help="tiny random byte core and cortex for wiring checks")
    run.add_argument("--directory", type=Path, required=True)
    run.add_argument("--workspace", type=Path, required=True)
    run.add_argument("--device", default="cpu")
    run.add_argument("--threads", type=int, help="CPU threads for torch (default: torch's choice)")
    run.add_argument("--trainable-blocks", type=int, default=1)
    run.add_argument("--injection-block", type=int, default=12)
    run.add_argument("--residual-scale", type=float, default=0.02)
    run.add_argument("--slice-tokens", type=int, default=32)
    run.add_argument("--idle-seconds", type=float, default=2.0)
    run.add_argument("--checkpoint-seconds", type=float, default=1800.0)
    run.add_argument("--run-timeout-seconds", type=float, default=60.0)
    run.add_argument("--no-spontaneous-thought", action="store_true")
    run.add_argument("--plan", type=Path, help="JSON {events: [{at_tick, input}]}; stop when answered and quiet")
    talk = commands.add_parser("attach", help="send messages to a running mind")
    talk.add_argument("directory", type=Path)
    talk.add_argument("--once")
    talk.add_argument("--wait", type=float)
    args = parser.parse_args(argv)
    if args.command == "attach":
        return attach(args.directory, once=args.once, wait=args.wait)

    if args.threads:
        import torch
        torch.set_num_threads(args.threads)
    config = FlowConfig(slice_tokens=args.slice_tokens, run_timeout_seconds=args.run_timeout_seconds,
                        spontaneous_thought=not args.no_spontaneous_thought)
    if args.byte_core:
        from .flow_core import ByteRecurrentCore
        language = ByteRecurrentCore(width=64, residual_scale=args.residual_scale)
        cortex = CortexConfig(byte_embedding_width=32, graph_width=32, direction_width=8, hidden_width=64,
                              recurrent_layers=1, maximum_event_bytes=128, seed=1701)
    else:
        from .flow_core import FlowRwkv
        language = FlowRwkv(args.model, device=args.device, trainable_blocks=args.trainable_blocks,
                            injection_block=args.injection_block, residual_scale=args.residual_scale)
        cortex = CortexConfig()
    args.workspace.mkdir(parents=True, exist_ok=True)
    mind = FlowMind(language, directory=args.directory, workspace=args.workspace, config=config, cortex_config=cortex)
    stop = False

    def request_stop(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        if mind.generation == 0:
            mind.save(label="opened")
        plan = json.loads(args.plan.read_text()) if args.plan else None
        status = run_worker(mind, idle_seconds=args.idle_seconds, checkpoint_seconds=args.checkpoint_seconds,
                            plan=plan, should_stop=lambda: stop)
        mind.save(label=status)
        atomic_json(mind.directory / "progress.json", {"status": status, **mind.status()})
    finally:
        mind.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
