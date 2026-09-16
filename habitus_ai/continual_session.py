"""Run the product's complete live session and one resumable corpus pass."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import traceback

from .continual_mind import ContinualMind, atomic_json
from .rwkv7_language import file_sha256


class CorpusReader:
    def __init__(self, mind, directory):
        import pyarrow.parquet as pq
        self.pq = pq
        self.mind = mind
        self.directory = Path(directory)
        self.plan = json.loads((self.directory / "plan.json").read_text())
        capture = json.loads((self.directory / "capture.json").read_text())
        if not capture["complete"] or capture["plan_sha256"] != file_sha256(self.directory / "plan.json"):
            raise ValueError("complete captured corpus with matching plan is required")
        self.files = sorted(self.plan["files"], key=lambda row: row["path"])
        with mind.mind.store.connection:
            mind.mind.store.connection.execute("CREATE TABLE IF NOT EXISTS continual_documents (sha256 TEXT PRIMARY KEY, tokens INTEGER NOT NULL)")
        self.rows = self._rows()
        self.current = None
        self.ids = None

    def _rows(self):
        cursor = self.mind.application["corpus_cursor"]
        for index, item in enumerate(self.files):
            if index < cursor["shard"]:
                continue
            path = self.directory / "shards" / Path(item["path"]).name
            if path.stat().st_size != item["bytes"] or file_sha256(path) != item["sha256"]:
                raise ValueError("corpus shard no longer matches capture")
            row_index = 0
            for batch in self.pq.ParquetFile(path).iter_batches(batch_size=32, columns=["text", "id"], use_threads=False):
                for row in batch.to_pylist():
                    position = row_index
                    row_index += 1
                    if index == cursor["shard"] and position < cursor["row"]:
                        continue
                    yield index, position, row

    def step(self):
        app = self.mind.application
        while self.current is None:
            try:
                shard, row, document = next(self.rows)
            except StopIteration:
                app["corpus_cursor"] = {"shard": len(self.files), "row": 0, "token": 0}
                return False
            digest = hashlib.sha256(document["text"].encode()).hexdigest()
            old = app["corpus_cursor"]
            offset = old["token"] if (old["shard"], old["row"]) == (shard, row) else 0
            app["corpus_cursor"] = {"shard": shard, "row": row, "token": offset}
            seen = self.mind.mind.store.connection.execute("SELECT 1 FROM continual_documents WHERE sha256=?", (digest,)).fetchone()
            if seen:
                app["corpus_cursor"] = {"shard": shard, "row": row + 1, "token": 0}
                continue
            self.ids = self.mind.backend.encode(document["text"])
            if not self.ids:
                continue
            self.current = {"shard": shard, "row": row, "sha256": digest, "id": document["id"]}
        cursor = app["corpus_cursor"]
        start = cursor["token"]
        end = min(len(self.ids), start + self.mind.backend.chunk_tokens)
        # End at a UTF-8 boundary for the canonical HEAR admission. Model training
        # uses the original IDs, not a decode/re-encode approximation.
        tokenizer = self.mind.backend.tokenizer
        while end < len(self.ids) and self.mind.backend.decode(self.ids[start:end]).endswith("\ufffd"):
            end += 1
        source = f"fineweb-edu:{self.current['shard']}:{self.current['row']}:{self.current['sha256']}"
        self.mind.read_tokens(self.ids[start:end], source=source,
                             next_token=self.ids[end] if end < len(self.ids) else None, final=end == len(self.ids))
        cursor["token"] = end
        if end == len(self.ids):
            with self.mind.mind.store.connection:
                self.mind.mind.store.connection.execute("INSERT INTO continual_documents VALUES (?,?)", (self.current["sha256"], len(self.ids)))
            app["corpus_documents"] += 1
            app["corpus_cursor"] = {"shard": self.current["shard"], "row": self.current["row"] + 1, "token": 0}
            self.current = self.ids = None
        return True


def process_inbox(mind, directory):
    inbox = directory / "inbox.jsonl"
    if not inbox.exists():
        return
    raw = inbox.read_text()
    # An append may still be in progress. Only newline-terminated envelopes
    # become committed inputs; a partial final line remains pending.
    lines = raw.splitlines() if raw.endswith("\n") else raw.splitlines()[:-1]
    for index, line in enumerate(lines):
        if index < mind.application["inbox_line"]:
            continue
        request = json.loads(line)
        response = None
        if request["kind"] == "input":
            admitted = mind.hear(request["text"])
            if admitted.get("tool_turn"):
                response = admitted["tool_turn"]["response"]
        elif request["kind"] == "feedback":
            result = mind.feedback(request["reward"], request["explanation"])
            response = json.dumps(result)
            print(json.dumps({"teacher_feedback": result}), flush=True)
        elif request["kind"] == "pulse":
            for _ in range(int(request.get("slices", 32))):
                receipt = mind.pulse()
                if receipt["public_emissions"]:
                    print(json.dumps(receipt["public_emissions"]), flush=True)
                if mind.flow.phase == "idle":
                    break
        elif request["kind"] == "save":
            response = json.dumps(mind.save(label="requested-checkpoint"))
        else:
            raise ValueError("unknown session input kind")
        mind.application["inbox_line"] = index + 1
        if request.get("request_id"):
            mind._append(mind.journal, {"kind": "request_ack", "request_id": request["request_id"],
                "request_kind": request["kind"], "input_number": mind.flow.cursor["inputs"], "result": response})


def publish_report(mind, directory, *, status, started, error=None):
    rows = [json.loads(line) for line in mind.journal.read_text().splitlines() if line.strip()]
    learning = [json.loads(line) for line in mind.training_journal.read_text().splitlines()] if mind.training_journal.exists() else []
    inputs = [row for row in rows if row["kind"] == "heard"]
    emissions = [emission for row in rows for emission in row.get("public_emissions", [])]
    actions = [row["tool_turn"] for row in inputs if row.get("tool_turn") and row["tool_turn"].get("tool_receipt")]
    feedback = [row for row in rows if row["kind"] == "feedback"]
    weights = mind.backend.weight_report()
    original = rows[0]["weights"]
    elapsed = time.monotonic() - started
    corpus = json.loads(mind.mind.store.get_metadata("corpus_manifest", "{}"))
    result = {"status": status, "elapsed_seconds_this_process": elapsed, "current": mind.status(),
        "initial_weights": original, "current_weights": weights,
        "weights_changed": {name: weights[name]["sha256"] != original[name]["sha256"] for name in weights},
        "seed_corpus_exposure": corpus, "live_inputs": inputs, "public_emissions": emissions,
        "actions": actions, "feedback": feedback, "joint_updates": len(learning),
        "latest_learning": learning[-1] if learning else None,
        "graph_invariants": mind.mind.graph.validate_invariants(),
        "error": error,
        "assessment_boundary": "Behavior below is actual generated language and actual action receipts. State or weight changes alone do not establish subjective coherence or general learning competence."}
    atomic_json(directory / "report.json", result)
    lines = ["# Integrated Habitus / RWKV live report", "", f"Status: **{status}**.", "",
        "## Built system", "", "This session runs the product's real Habitus graph, SELF pulse, 18,016,166-parameter developmental cortex, local RWKV-7 and receipt-backed workspace abilities together. Inputs, pending language and actions share one continuing numeric state and one joint checkpoint. No external chat renderer, persona prompt, transcript replay or RAG text is supplied to RWKV.", "",
        "The native RWKV final block and output normalization are trainable. Lower blocks, token embeddings and vocabulary projection retain their pretrained weights. Language loss also trains the 18M cortex and both directions of its numeric interface. Verified tool returns train cortex action/consequence prediction; explicit teacher feedback reinforces delivered language and its actual graph route.", "",
        "## Exposure and execution", "", f"Imported graph exposure: {corpus.get('tokens', 0):,} RWKV tokens across {corpus.get('documents', 0):,} documents. This is prior association learning, not RWKV gradient training.", "",
        f"Current RWKV language training tokens: {mind.backend.counters['language_training_tokens']:,}; corpus: {mind.backend.counters['corpus_training_tokens']:,}; interaction: {mind.backend.counters['interaction_training_tokens']:,}. Reinforced public-language tokens: {mind.backend.counters['reinforced_tokens']:,}.", "",
        f"Actual inputs: {len(inputs)}. Delivered public speech chunks: {len(emissions)}. Tool actions: {len(actions)}. Joint optimizer updates: {mind.backend.counters['updates']}.", "",
        "| Component | Trainable parameters | Weights changed |", "| --- | ---: | --- |"]
    for name in weights:
        lines.append(f"| {name} | {weights[name]['parameters']:,} | {result['weights_changed'][name]} |")
    lines += ["", f"GPU initialized: {mind.backend.torch.cuda.is_initialized()}. Graph invariant errors: {result['graph_invariants']}.", "",
        "## Actual interaction", ""]
    for row in rows:
        if row["kind"] == "heard":
            lines += [f"### Input {row['event']}", "", row["text"], ""]
            if row.get("tool_turn") and row["tool_turn"].get("tool_receipt"):
                receipt = row["tool_turn"]["tool_receipt"]
                lines += [f"Action: `{receipt['tool_id']}`; status `{receipt['status']}`; receipt `{receipt['receipt_id']}`; observed return `{receipt['return_record_id']}`.", "", row["tool_turn"]["response"], ""]
        for emission in row.get("public_emissions", []):
            lines += ["RWKV:", "", emission["text"], ""]
        if row["kind"] == "feedback":
            lines += ["Teacher assessment: " + row["assessment"]["explanation"], "", f"Applied: {row['result']['applied']}", ""]
    lines += ["## Limits and remaining work", "",
        "This report records one complete system operating and learning. It does not establish scientific novelty or prove that generated self-descriptions correspond to subjective experience. Commands supplied by the teacher establish action integration; they do not establish autonomous task discovery. Private generation is retained as numeric state; unfinished public drafts are not delivered replies.", "",
        "The corpus job is one continuous pass over all captured training documents. Its exact cursor and optimizer state are checkpointed. Completion of prior graph training must not be described as completion of RWKV fine-tuning. The report remains marked running while that pass remains unfinished.", ""]
    if error:
        lines += ["## Execution error", "", "```text", error, "```", ""]
    (directory / "REPORT.md").write_text("\n".join(lines))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "directory", "workspace", "corpus-graph", "corpus"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--live-plan", type=Path)
    parser.add_argument("--live-only", action="store_true", help="complete the live session without starting the full corpus continuation")
    parser.add_argument("--maximum-seconds", type=int, default=None)
    parser.add_argument("--checkpoint-seconds", type=int, default=1800)
    args = parser.parse_args(argv)
    started = time.monotonic()
    stop = False
    def request_stop(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    mind = ContinualMind(model=args.model, directory=args.directory, workspace=args.workspace,
                        corpus_graph=args.corpus_graph)
    last_checkpoint = time.monotonic()
    status = "running"
    try:
        if mind.generation == 0:
            mind.save(label="complete-build-before-live-session")
        elif mind.flow.phase == "public":
            # Resume actual pending language before the next scheduled input.
            receipt = mind.pulse()
            print(json.dumps({"resumed_public_emissions": receipt["public_emissions"]}), flush=True)
        plan = json.loads(args.live_plan.read_text()) if args.live_plan else {"events": []}
        for index, item in enumerate(plan["events"]):
            if index < mind.application.get("live_plan_index", 0):
                continue
            if stop or (args.directory / "STOP").exists():
                break
            event = mind.hear(item["input"])
            print(json.dumps({"event": index, "heard": item["input"], "tool_turn": event["tool_turn"]}, default=str), flush=True)
            while mind.flow.phase != "idle" and not stop and not (args.directory / "STOP").exists():
                receipt = mind.pulse()
                if receipt["public_emissions"]:
                    print(json.dumps({"event": index, "public_emissions": receipt["public_emissions"]}), flush=True)
                atomic_json(args.directory / "progress.json", {"status": "live", "event": index,
                    **mind.status()})
                process_inbox(mind, args.directory)
            mind.application["live_plan_index"] = index + 1
            if item.get("checkpoint"):
                mind.save(label=f"live-event-{index}")
            publish_report(mind, args.directory, status="live-session-running", started=started)
        if plan["events"] and mind.application.get("live_plan_index") == len(plan["events"]):
            mind.save(label="completed-full-live-session")
            atomic_json(args.directory / "live-complete.json", {"time": time.time(), **mind.status()})
            publish_report(mind, args.directory, status="live-session-complete-corpus-pending", started=started)
        reader = None if args.live_only else CorpusReader(mind, args.corpus)
        if reader is not None:
            publish_report(mind, args.directory, status="corpus-training", started=started)
        while (reader is not None and not stop and not (args.directory / "STOP").exists()
               and (args.maximum_seconds is None or time.monotonic() - started < args.maximum_seconds)):
            process_inbox(mind, args.directory)
            if mind.application.get("conversation_active") and mind.flow.phase != "idle":
                receipt = mind.pulse()
                if receipt["public_emissions"]:
                    print(json.dumps(receipt["public_emissions"]), flush=True)
                atomic_json(args.directory / "progress.json", {"status": "live-during-corpus-training", **mind.status()})
                continue
            if not reader.step():
                status = "complete"
                break
            atomic_json(args.directory / "progress.json", {"status": "corpus-training", **mind.status()})
            if time.monotonic() - last_checkpoint > args.checkpoint_seconds:
                mind.save(label="continuous-corpus-and-interaction")
                publish_report(mind, args.directory, status="corpus-training", started=started)
                last_checkpoint = time.monotonic()
        if status != "complete":
            status = "live-session-complete" if args.live_only else "checkpointed-stopped"
        mind.save(label=status)
        atomic_json(args.directory / "progress.json", {"status": status, **mind.status()})
        publish_report(mind, args.directory, status=status, started=started)
    except BaseException:
        error = traceback.format_exc()
        # The last committed joint generation remains the restart authority.
        publish_report(mind, args.directory, status="failed", started=started, error=error)
        raise
    finally:
        mind.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
