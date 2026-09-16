"""The integrated product with persistent RWKV learning and the full cortex.

This uses IntegratedMind's actual action registry, SELF authorizations, canonical
graph and sensory returns. Console, corpus reader and live sessions use this
same class and the same joint checkpoints.
"""
from __future__ import annotations

from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import time

from .continual_rwkv import ContinualRwkv, RecurrentEmbeddingSpace
from .developmental_cortex import CortexConfig
from .developmental_curriculum import EpisodeKind
from .developmental_runtime import BornInHabitusRuntime, DevelopmentalInput
from .gestation import gestate, load_profile
from .integrated_agent import IntegratedMind
from .recurrent_flow import RecurrentFlow
from .rwkv7_language import file_sha256
from .types import ExperienceReturn, InputTrunk, OutputDecision, OutputTrunk, RecordType


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, default=str, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def database_copy(source, destination):
    with sqlite3.connect(f"file:{Path(source).resolve()}?mode=ro", uri=True) as src:
        with sqlite3.connect(destination) as dst:
            src.backup(dst)


class _NoExternalRenderer:
    def generate(self, _):
        raise RuntimeError("the integrated RWKV path has no external chat renderer")


class ContinualMind(IntegratedMind):
    def __init__(self, *, model, directory, workspace, corpus_graph=None,
                 trainable_blocks=1, chunk_tokens=128, mode="private"):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = (self.directory / "writer.lock").open("a")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.journal = self.directory / "events.jsonl"
        self.training_journal = self.directory / "learning.jsonl"
        self.backend = ContinualRwkv(model, trainable_blocks=trainable_blocks, chunk_tokens=chunk_tokens)
        working = self.directory / "mind.sqlite"
        self.generation = 0
        restored = None
        self.application = {"events": 0, "delivered": 0, "corpus_cursor": {"shard": 0, "row": 0, "token": 0},
                            "corpus_documents": 0, "inbox_line": 0}
        if (self.directory / "LATEST.json").exists():
            pointer = json.loads((self.directory / "LATEST.json").read_text())
            checkpoint = self.directory / pointer["checkpoint"]
            if file_sha256(checkpoint / "manifest.json") != pointer["manifest_sha256"]:
                raise ValueError("joint checkpoint manifest hash mismatch")
            manifest = json.loads((checkpoint / "manifest.json").read_text())
            for name, digest in manifest["files"].items():
                if file_sha256(checkpoint / name) != digest:
                    raise ValueError("joint checkpoint checksum mismatch: " + name)
            staged = self.directory / "restore.sqlite"
            if staged.exists():
                staged.unlink()
            database_copy(checkpoint / "mind.sqlite", staged)
            if working.exists():
                # Keep any uncommitted/crashed work inspectable.
                working.replace(self.directory / f"uncommitted-{time.time_ns()}.sqlite")
            for suffix in ("-wal", "-shm"):
                side = Path(str(working) + suffix)
                if side.exists():
                    side.replace(self.directory / f"uncommitted-{time.time_ns()}{suffix}")
            staged.replace(working)
            restored = self.backend.torch.load(checkpoint / "neural.pt", map_location="cpu", weights_only=True)
            self.application = manifest["application"]
            self.generation = manifest["generation"]
        elif working.exists():
            raise ValueError("uncheckpointed mind exists; use a new directory or recover its artifacts")
        elif corpus_graph is not None:
            graph = Path(corpus_graph).resolve()
            report = graph.parent / "report.json"
            if report.exists():
                expected = json.loads(report.read_text())[graph.stem]["database_sha256"]
                if file_sha256(graph) != expected:
                    raise ValueError("corpus graph hash does not match its exposure receipt")
            database_copy(graph, working)
        space_id = None
        if working.exists():
            with sqlite3.connect(working) as connection:
                row = connection.execute("SELECT value FROM metadata WHERE key='embedding_space_id'").fetchone()
                space_id = row[0] if row else None
        embedder = RecurrentEmbeddingSpace(self.backend, space_id=space_id)
        runtime = BornInHabitusRuntime(working, cortex_config=CortexConfig(), device="auto",
            checkpoint_directory=self.directory / "cortex", embedder=embedder)
        runtime.mind.graph.recency_strength = 0.0
        profile = load_profile(runtime.mind)
        if profile is None:
            profile = gestate(runtime.mind, human_name="Human", agent_name="Habitus", taste_schema="builder",
                              model_backend="integrated-rwkv", model_name=self.backend.identity)
        super().__init__(runtime, _NoExternalRenderer(), workspace=workspace, profile=profile)
        self.backend.attach(runtime)
        self.backend.receipt_sink = self._record_learning
        self.flow = RecurrentFlow(self.backend, mode=mode)
        self.latest_pulse = None
        self.eligibility = None
        self.candidate = None
        self.generated_motor = False
        self.action_eligibility = None
        if restored is not None:
            self.backend.restore_payload(restored["backend"])
            restored_cursor = dict(restored["cursor"])
            restored_cursor.pop("stream_limit", None)
            self.flow.cursor = restored_cursor
            self.eligibility = restored["eligibility"]
            self.backend.public_candidate = restored.get("candidate")
            import io
            torch = self.backend.torch
            expected = torch.load(io.BytesIO(restored["backend"]["numeric"]), weights_only=True)
            actual = torch.load(io.BytesIO(self.backend.dump_state()), weights_only=True)
            identical = (all(torch.equal(expected["language"][key], actual["language"][key]) for key in expected["language"])
                and torch.equal(expected["cortex"], actual["cortex"]) and torch.equal(expected["field"], actual["field"]))
            if "sampling_rng" in expected:
                identical = identical and torch.equal(expected["sampling_rng"], actual["sampling_rng"])
            weights_match = self.backend.weight_report() == manifest["weights"]
            if not identical or not weights_match:
                raise ValueError("joint restart did not reproduce the committed neural state")
            self._append(self.journal, {"kind": "resumed", "checkpoint": checkpoint.name,
                "numeric_tensors_exact": identical, "all_trained_weights_exact": weights_match,
                "source_exposure_cursors_restored": self.application["corpus_cursor"],
                "decoder_cursor_restored": True})
            if "sampling" not in restored["backend"]:
                self._append(self.journal, {"kind": "decoder_configuration_change", "previous": "greedy",
                    "current": dict(self.backend.sampling),
                    "reason": "use the pinned RWKV model's documented sampling configuration; learned weights and numeric state retained"})
        # Use the native sensory-fiber kind, including imported corpus fibers.
        from .graph import SENSORY_FEATURE_KIND
        self._corpus_features = runtime.mind.store.list_concepts(kind=SENSORY_FEATURE_KIND)
        self._feature_matrix = self.backend.torch.tensor([node.embedding for node in self._corpus_features])
        self.initial_weights = self.backend.weight_report()
        self._append(self.journal, {"kind": "opened", "weights": self.initial_weights,
            "resumed_generation": self.generation, "cortex_parameters": runtime.cortex.parameter_count,
            "rwkv_total_parameters": self.backend.statistics["weight_tensor_elements"],
            "sampling": dict(self.backend.sampling),
            "gpu_initialized": self.backend.torch.cuda.is_initialized()})

    @staticmethod
    def _append(path, value):
        with Path(path).open("a") as stream:
            stream.write(json.dumps({"time": time.time(), **value}, sort_keys=True, default=str, allow_nan=False) + "\n")
            stream.flush()

    def _record_learning(self, receipt):
        self._append(self.training_journal, receipt)

    def _perceive(self, content, *, command_name=None, ability_id=None, arguments=None):
        previous = self.mind.open_experience_cycles(OutputTrunk.SPEAK)
        if self.generated_motor:
            result = self.runtime.advance((self._ability_opportunity(ability_id, arguments or {}),))
        else:
            result = super()._perceive(content, command_name=command_name, ability_id=ability_id, arguments=arguments)
            # A streaming reply can contain several authorized speech acts. One
            # real human return settles all of them using the same admitted
            # record; its bytes enter the cortex and RWKV only once.
            if previous:
                last = self.mind.store.get_experience_cycle(previous[-1].cycle_id)
                if last.terminal_return_record_id:
                    for cycle in previous[:-1]:
                        current = self.mind.store.get_experience_cycle(cycle.cycle_id)
                        if current.status == "open":
                            self.mind.store.add_experience_return(ExperienceReturn(cycle.cycle_id,
                                last.terminal_return_record_id, InputTrunk.HEAR, "human-response-to-stream",
                                0.0, True, True, self.mind.pulse,
                                {"same_admitted_human_return": True, "extra_input_admissions": 0}))
        self.latest_pulse = result
        if ability_id is not None:
            self.action_eligibility = {"state": self.backend.dump_state(), "graph": self.backend._graph_vectors()}
        return result

    def hear(self, text):
        text = str(text).strip()
        if not text:
            raise ValueError("current input cannot be empty")
        self.capture_private_diagnostic()
        self.backend.source = "external_language"
        self.application["conversation_active"] = True
        self.application["reading_source"] = None
        if self.flow.phase == "public":
            draft = self.flow._public_text(self.flow._text()).lstrip()
            pending = draft[self.application.get("public_characters_delivered", 0):]
            if pending.strip():
                self._append(self.journal, {"kind": "interrupted_public_draft", "text": pending,
                    "input_number": self.flow.cursor["inputs"]})
        # Train before admitting this message to graph/cortex: future target
        # words cannot leak backwards through the numeric condition.
        event = self.flow.hear(text)
        self.candidate = None
        self.backend.public_candidate = None
        self.application["public_characters_delivered"] = 0
        self.application["stream_cycles"] = []
        command = self._split_command(text)
        turn = None
        if command is not None:
            turn = super().handle(text)
            if turn.tool_receipt is not None:
                receipt = turn.tool_receipt
                self.backend.learn_action(receipt.trunk, success=receipt.status == "success", receipt_id=receipt.receipt_id,
                                          eligibility=self.action_eligibility)
                self.latest_pulse = self._latest_tool_return_pulse
        else:
            self.latest_pulse = self._perceive(text)
        self.application["events"] += 1
        result = {"kind": "heard", "event": self.application["events"], "text": text,
            "flow": event, "tool_turn": None if turn is None else turn.to_dict(),
            "cortex_proposal": None if self.latest_pulse is None or self.latest_pulse.cortex_proposal is None
                else asdict(self.latest_pulse.cortex_proposal), "counters": dict(self.backend.counters)}
        self._append(self.journal, result)
        return result

    def _activate_internal(self, ids):
        if not ids or not self._corpus_features:
            return
        torch = self.backend.torch
        # Sensory fibers grow and their centroids change on real admission.
        # Derive the active receptor geometry from the authoritative graph so
        # continuing without a restart and restoring a checkpoint agree.
        from .graph import SENSORY_FEATURE_KIND
        self._corpus_features = self.mind.store.list_concepts(kind=SENSORY_FEATURE_KIND)
        self._feature_matrix = torch.tensor([node.embedding for node in self._corpus_features])
        vector = torch.tensor(self.backend.stream_embedding(ids))
        vector = torch.nn.functional.normalize(vector, dim=0)
        scores = (self._feature_matrix @ vector).clamp_min(0).square()
        excitation = {
            feature.concept_id: float(value)
            for feature, value in zip(self._corpus_features, scores)
            if float(value) > 0.0
        }
        self.runtime.kernel.advance_internal_output(excitation=excitation, kind="integrated_rwkv_activity",
            metadata={"origin": "model_generated_unverified", "external_evidence": False, "tokens": len(ids)})

    def pulse(self, tokens=None):
        # This is the same full SELF/cortex transition used for input/actions.
        self.latest_pulse = self.runtime.advance((), allow_idle=True)
        self.backend.begin_slice()
        if self.flow.phase == "idle":
            self.application["conversation_active"] = False
            result = {"kind": "quiet", "public_emissions": []}
        else:
            before_count = len(self.flow.cursor["generated_ids"])
            result = self.flow.pulse(maximum_tokens=tokens)
            ids = self.flow.cursor["generated_ids"][before_count:]
            self._activate_internal(ids)
            completed = bool(result["public_emissions"])
            public = self.flow._public_text(self.flow._text()).split("\n\nUser:", 1)[0].lstrip()
            start = self.application.get("public_characters_delivered", 0)
            boundaries = list(re.finditer(r"[.!?][\"'”’)]*(?:\s|$)|\n\n", public))
            end = len(public.rstrip()) if completed else (boundaries[-1].end() if boundaries else start)
            # Incomplete paragraphs remain a cursor, not a requirement that the
            # whole response reach EOS before any public speech can be delivered.
            text = public[start:end].strip() if end > start else ""
            command_prefix = text.startswith(("/open ", "/run ", "/remember ", "/recall "))
            motor = command_prefix and completed and "\n" not in text
            if command_prefix and not completed:
                text = ""
            delivered = []
            if text and result.get("stop") != "generated_user_content":
                emission = {"text": text, "origin": "model_public_output", "input_number": self.flow.cursor["inputs"],
                            "stream_complete": completed, "streaming": True}
                cycle = None
                if motor:
                    self.generated_motor = True
                    try:
                        turn = super().handle(text)
                    finally:
                        self.generated_motor = False
                    if turn.tool_receipt:
                        receipt = turn.tool_receipt
                        self.backend.learn_action(receipt.trunk, success=receipt.status == "success", receipt_id=receipt.receipt_id,
                                                  eligibility=self.action_eligibility)
                    self._append(self.journal, {"kind": "generated_action", "turn": turn.to_dict()})
                    self.application["public_characters_delivered"] = end
                else:
                    cycle = self._actualize_speech(self.latest_pulse, text, surface="continual-rwkv")
                    if not cycle:
                        self._append(self.journal, {"kind": "undelivered", "reason": "SELF did not authorize SPEAK", "draft": text})
                        cycle = None
                if cycle:
                    self.application["delivered"] += 1
                    self.application["public_characters_delivered"] = end
                    self.application.setdefault("stream_cycles", []).append(cycle)
                    delivered.append({**emission, "speech_cycle_id": cycle})
                    candidate = self.backend.public_candidate
                    if candidate:
                        tokenizer = self.backend.tokenizer
                        raw = [tokenizer._token_bytes.get(tokenizer.decoder[token], b"") for token in candidate["tokens"]]
                        full = b"".join(raw)
                        prefix = public[:end].encode()
                        offset = full.find(prefix)
                        limit = offset + len(prefix) if offset >= 0 else 0
                        size, count = 0, 0
                        for item in raw:
                            if size + len(item) > limit:
                                break
                            size += len(item)
                            count += 1
                        self.eligibility = {**candidate, "tokens": candidate["tokens"][:count],
                            "cycle_id": cycle, "cycles": list(self.application["stream_cycles"]),
                            "text_sha256": hashlib.sha256(prefix).hexdigest()}
            result["public_emissions"] = delivered
        result["habitus_pulse"] = self.mind.pulse
        result["counters"] = dict(self.backend.counters)
        self._append(self.journal, result)
        return result

    def capture_private_diagnostic(self):
        if not self.flow.cursor["generated_ids"]:
            return
        text = self.flow._text()
        end = self.flow.cursor["private_end"]
        private = text if end is None and self.flow.cursor["mode"] == "private" else (text[:end] if end else "")
        self._append(self.directory / "private-diagnostics.jsonl", {
            "origin": "local_rwkv_generated_unverified", "input_number": self.flow.cursor["inputs"],
            "phase": self.flow.phase, "tokens": len(self.flow.cursor["generated_ids"]),
            "text": private, "external_evidence": False, "replayed_as_input": False})

    def feedback(self, reward, explanation):
        evidence = {"origin": "external_teacher_assessment", "explanation": str(explanation),
                    "event": self.application["events"], "speech_cycle_id": None if self.eligibility is None else self.eligibility["cycle_id"]}
        result = self.backend.reinforce(self.eligibility, float(reward), evidence=evidence)
        if result["applied"]:
            assessment = self.mind.remember(str(explanation), source_id="local_teacher",
                record_type=RecordType.RECEIPT, allow_growth=False, input_trunk=InputTrunk.HEAR,
                metadata={"origin": "explicit_teacher_assessment", "reward": float(reward), "training_annotation": True})
            edges = tuple(dict.fromkeys(edge for cycle_id in self.eligibility.get("cycles", [self.eligibility["cycle_id"]])
                for edge in self.mind.store.get_experience_cycle(cycle_id).credited_edge_ids))
            self.mind.record_outcome(OutputDecision(self.eligibility["cycle_id"], OutputTrunk.SPEAK, 1.0, None),
                stability_delta=float(reward), verified=True, receipt_id=assessment.record_id,
                metadata=evidence, credited_edge_ids=edges)
            self.eligibility = None
        self._append(self.journal, {"kind": "feedback", "assessment": evidence, "result": result})
        return result

    def read_tokens(self, ids, *, source, next_token=None, final=False):
        # Real corpus text advances and trains the same recurrent language and
        # cortex, then becomes attributed HEAR evidence in the same graph.
        if self.application.get("reading_source") != source:
            self.capture_private_diagnostic()
            # A real reading event interrupts a pending output through the same
            # native role framing as ordinary input. It never consumes a stale
            # generation cursor after the language state advances through text.
            if self.flow.phase == "public":
                draft = self.flow._public_text(self.flow._text()).lstrip()
                pending = draft[self.application.get("public_characters_delivered", 0):]
                if pending.strip():
                    self._append(self.journal, {"kind": "interrupted_public_draft", "text": pending,
                                               "cause": "external_reading"})
            prefix = "</think>\n\n" if self.flow.phase == "private" else "\n\n"
            if not self.flow.cursor["user_prefix_open"]:
                prefix += "User: "
            else:
                prefix = "" if self.flow.cursor["user_prefix_has_space"] else " "
            if prefix:
                framing = self.backend.encode(prefix)
                self.backend.consume(framing)
                self.flow.cursor["framing_tokens"] += len(framing)
            self.flow._start_stream()
            self.flow.cursor["phase"] = "idle"
            self.flow.cursor["inputs"] += 1
            self.application["reading_source"] = source
            self.application["public_characters_delivered"] = 0
            self.application["stream_cycles"] = []
            self.backend.public_candidate = None
        self.backend.learn_tokens(ids, source="corpus:" + source, next_token=next_token)
        self.flow.cursor["input_tokens"] += len(ids)
        text = self.backend.decode(ids)
        self.latest_pulse = self.runtime.advance((DevelopmentalInput(text, InputTrunk.HEAR,
            source_id=source, episode_kind=EpisodeKind.EXTERNAL_NARRATIVE, motor_eligible=False,
            metadata={"external_reading": True, "direct_observation": False}),))
        self._append(self.journal, {"kind": "reading", "source": source, "tokens": len(ids),
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "pulse": self.mind.pulse,
            "counters": dict(self.backend.counters)})
        if final:
            framing = self.backend.encode("\n\n")
            self.backend.consume(framing)
            self.flow.cursor["framing_tokens"] += len(framing)
            self.flow._start_stream()
            self.application["reading_source"] = None

    def save(self, *, label="checkpoint"):
        self.capture_private_diagnostic()
        self.generation += 1
        directory = self.directory / f"checkpoint-{self.generation:06d}"
        directory.mkdir(exist_ok=False)
        self.backend.detach_state()
        neural = {"backend": self.backend.checkpoint_payload(), "cursor": self.flow.cursor,
                  "eligibility": self.eligibility, "candidate": self.backend.public_candidate}
        self.backend.torch.save(neural, directory / "neural.pt")
        with sqlite3.connect(directory / "mind.sqlite") as destination:
            self.mind.store.connection.backup(destination)
        report = self.backend.weight_report()
        manifest = {"generation": self.generation, "label": label, "application": self.application,
            "weights": report, "counters": dict(self.backend.counters),
            "files": {name: file_sha256(directory / name) for name in ("neural.pt", "mind.sqlite")},
            "numeric_state_sha256": hashlib.sha256(self.backend.dump_state()).hexdigest(),
            "gpu_initialized": self.backend.torch.cuda.is_initialized(),
            "corpus_seed": json.loads(self.mind.store.get_metadata("corpus_manifest", "{}")),
            "time": time.time()}
        atomic_json(directory / "manifest.json", manifest)
        atomic_json(self.directory / "LATEST.json", {"checkpoint": directory.name,
            "manifest_sha256": file_sha256(directory / "manifest.json")})
        self._append(self.journal, {"kind": "checkpoint", "checkpoint": directory.name, "weights": report})
        return manifest

    def status(self):
        return {"application": self.application, "counters": self.backend.counters,
            "phase": self.flow.phase, "pulse": self.mind.pulse,
            "cortex_parameters": self.runtime.cortex.parameter_count,
            "trainable_rwkv_parameters": sum(value.numel() for value in self.backend.trainable_rwkv().values()),
            "device": str(self.backend.device),
            "gpu_initialized": self.backend.torch.cuda.is_initialized()}

    def close(self):
        self.runtime.close()
        fcntl.flock(self.lock, fcntl.LOCK_UN)
        self.lock.close()


def console(args):
    import select
    import sys
    try:
        with_state = ContinualMind(model=args.rwkv_model, directory=args.unified_state,
            workspace=args.workspace, corpus_graph=args.corpus_graph, trainable_blocks=args.trainable_blocks)
    except BlockingIOError:
        return attached_console(args.unified_state, once=args.once, timeout=args.timeout)
    try:
        waiting_for_input = False
        print("Integrated Habitus / RWKV is running. /feedback VALUE EXPLANATION, /save, /quit; existing workspace commands remain available.", flush=True)
        if args.once:
            event = with_state.hear(args.once)
            print(json.dumps(event, default=str), flush=True)
            if event["tool_turn"]:
                with_state.save(label="console-command-exit")
                return 0
        while True:
            if not args.once and select.select([sys.stdin], [], [], 0.05)[0]:
                line = sys.stdin.readline()
                if not line or line.strip() == "/quit":
                    break
                if line.strip() == "/save":
                    print(json.dumps(with_state.save()), flush=True)
                    continue
                if line.startswith("/feedback "):
                    _, reward, explanation = line.strip().split(" ", 2)
                    print(json.dumps(with_state.feedback(float(reward), explanation)), flush=True)
                    continue
                event = with_state.hear(line)
                waiting_for_input = False
                if event["tool_turn"]:
                    print(event["tool_turn"]["response"], flush=True)
            if with_state.flow.phase != "idle" and not waiting_for_input:
                receipt = with_state.pulse()
                for emission in receipt["public_emissions"]:
                    print(emission["text"], flush=True)
                if args.once and (receipt["public_emissions"] or with_state.flow.phase == "idle"):
                    break
        with_state.save(label="console-exit")
    finally:
        with_state.close()
    return 0


def attached_console(directory, *, once=None, timeout=None):
    """Talk to the running trainer through its input queue; never load a second mind."""
    import select
    import sys
    import uuid
    directory = Path(directory).resolve()
    if not (directory / "writer.lock").exists():
        raise RuntimeError("no integrated worker is running in this state directory")
    with (directory / "writer.lock").open("a") as check:
        try:
            fcntl.flock(check, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(check, fcntl.LOCK_UN)
            raise RuntimeError("the integrated worker has stopped; resume its checkpoint first")
    journal = directory / "events.jsonl"
    offset = journal.stat().st_size
    request_id = None
    input_number = None
    began = time.monotonic()

    def submit(text):
        nonlocal request_id
        request_id = uuid.uuid4().hex
        request = {"kind": "input", "text": text, "request_id": request_id}
        if text.startswith("/feedback "):
            _, reward, explanation = text.split(" ", 2)
            request.update(kind="feedback", reward=float(reward), explanation=explanation)
        elif text == "/save":
            request["kind"] = "save"
        with (directory / "inbox.jsonl").open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            stream.write(json.dumps(request) + "\n")
            stream.flush()
            fcntl.flock(stream, fcntl.LOCK_UN)

    print("Attached to the running integrated mind. /quit detaches; training continues.", flush=True)
    if once is not None:
        submit(once)
    while True:
        if once is None and select.select([sys.stdin], [], [], 0.1)[0]:
            line = sys.stdin.readline()
            if not line or line.strip() == "/quit":
                return 0
            if line.strip():
                submit(line.strip())
        with journal.open() as stream:
            stream.seek(offset)
            while line := stream.readline():
                if not line.endswith("\n"):
                    break
                event = json.loads(line)
                offset = stream.tell()
                if event.get("kind") == "request_ack" and event.get("request_id") == request_id:
                    input_number = event.get("input_number")
                    if event.get("result") and (once is not None or event.get("request_kind") != "input"):
                        print(event["result"], flush=True)
                        if once is not None:
                            return 0
                for emission in event.get("public_emissions", []):
                    if once is None or emission["input_number"] == input_number:
                        print(emission["text"], flush=True)
                        if once is not None:
                            return 0
                if once is None and event.get("kind") == "heard" and event.get("tool_turn"):
                    print(event["tool_turn"]["response"], flush=True)
        if once is not None and timeout is not None and time.monotonic() - began >= timeout:
            print("Input queued; the running mind has not delivered speech within this client's wait window.", flush=True)
            return 2
        if once is not None:
            time.sleep(0.1)
