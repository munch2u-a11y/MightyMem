"""Semantic recall over Habitus's canonical records and native vaults."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import urllib.error
import urllib.request

import numpy as np

from .corpus_associations import ByteTokenizer
from .embeddings import cosine_similarity
from .rwkv7_language import file_sha256
from .types import EventKind, GraphSide, InputTrunk, RecordType


class LocalModels:
    """Access already-installed local chat and embedding models through Ollama."""
    def __init__(self, chat="qwen3.5:4b", embedding="qwen3-embedding:0.6b", *,
                 base_url="http://127.0.0.1:11434", timeout=None, context_tokens=None,
                 num_gpu=-1, think=False, maximum_tokens=None, temperature=0.25):
        self.chat, self.embedding = chat, embedding
        self.base_url, self.timeout = base_url.rstrip("/"), timeout
        self.num_gpu, self.think = int(num_gpu), bool(think)
        self.temperature = float(temperature)
        self.maximum_tokens = None if maximum_tokens is None else int(maximum_tokens)
        if self.maximum_tokens is not None and self.maximum_tokens < 1:
            raise ValueError("maximum_tokens must be positive or None")
        available = self.request("tags")["models"]
        models = {row["name"]: row for row in available}
        for name in (chat, embedding):
            if name not in models:
                raise ValueError(f"Local model {name!r} is not installed; no download was attempted")
        chat_metadata = self.request("show", {"model": chat})
        native_context_values = [
            int(value)
            for key, value in chat_metadata.get("model_info", {}).items()
            if str(key).endswith(".context_length") and int(value) > 0
        ]
        if context_tokens is None:
            if not native_context_values:
                raise ValueError("local model did not report its native context length")
            self.context_tokens = max(native_context_values)
        else:
            self.context_tokens = int(context_tokens)
            if self.context_tokens < 1:
                raise ValueError("context_tokens must be positive or None")
        self.space_id = embedding + "@" + models[embedding]["digest"]
        self.chat_identity = chat + "@" + models[chat]["digest"]
        self.last_usage = {}
        self.last_response = {}

    def request(self, endpoint, payload=None):
        request = urllib.request.Request(self.base_url + "/api/" + endpoint,
            data=None if payload is None else json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except (OSError, urllib.error.URLError, ValueError) as error:
            raise RuntimeError(f"Local model request failed: {error}") from error
        if result.get("error"):
            raise RuntimeError(result["error"])
        return result

    def embed(self, texts):
        result = self.request("embed", {"model": self.embedding, "input": list(texts),
            "truncate": False, "keep_alive": -1, "options": {"num_gpu": self.num_gpu}})
        vectors = np.asarray(result["embeddings"], dtype=np.float32)
        if vectors.ndim != 2 or len(vectors) != len(texts) or not np.isfinite(vectors).all():
            raise ValueError("invalid local semantic embeddings")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms <= 0):
            raise ValueError("zero semantic embedding")
        return vectors / norms

    def complete(self, messages, tools):
        options = {"num_gpu": self.num_gpu,
                   "num_ctx": self.context_tokens, "temperature": self.temperature}
        if self.maximum_tokens is not None:
            options["num_predict"] = self.maximum_tokens
        payload = {"model": self.chat, "messages": messages, "stream": False,
            "think": self.think, "keep_alive": -1, "options": options}
        if tools:
            payload["tools"] = tools
        result = self.request("chat", payload)
        # Some local-server load/scheduling failures return an uninitialized
        # response object. It is not an assistant decision or a completed turn.
        if not result.get("model") and not result.get("done"):
            raise RuntimeError("local server returned an uninitialized chat response")
        self.last_response = result
        self.last_usage = {key: result.get(key) for key in
            ("prompt_eval_count", "eval_count", "total_duration", "done_reason")}
        self.last_usage["transport_retry"] = False
        if result.get("done_reason") == "length":
            raise RuntimeError(
                "model reached its physical context boundary; partial generation was rejected"
            )
        message = result.get("message", {})
        return {key: value for key, value in message.items()
                if key in ("role", "content", "thinking", "tool_calls")}


class PreservedSensorySpace:
    """Keep the existing graph's RWKV geometry without loading its language RNN."""
    pretrained_geometry = True

    def __init__(self, assets, space_id):
        assets = Path(assets)
        manifest = json.loads((assets / "manifest.json").read_text())
        for name in ("embeddings.npy", "tokenizer.json"):
            if file_sha256(assets / name) != manifest["files"][name]:
                raise ValueError("sensory embedding assets changed: " + name)
        self.embeddings = np.load(assets / "embeddings.npy", mmap_mode="r")
        self.dimension, self.space_id = self.embeddings.shape[1], space_id
        self.tokenizer = ByteTokenizer(assets / "tokenizer.json")

    def embed(self, text):
        ids = self.tokenizer.encode(str(text))
        if not len(ids):
            return [0.0] * self.dimension
        value = np.zeros(self.dimension, dtype=np.float32)
        for start in range(0, len(ids), 2048):
            value += self.embeddings[ids[start:start + 2048]].sum(axis=0)
        return (value / max(float(np.linalg.norm(value)), 1e-12)).tolist()


class VaultMemory:
    VAULTS = ("episodes", "facts", "observations", "skills")

    def __init__(self, mind, semantic):
        self.mind, self.semantic = mind, semantic
        self.connection = mind.store.connection
        with self.connection:
            self.connection.execute("""CREATE TABLE IF NOT EXISTS agent_semantic_index (
                record_id TEXT PRIMARY KEY REFERENCES records(record_id), vault TEXT NOT NULL,
                space_id TEXT NOT NULL, vector BLOB NOT NULL, evidence_json TEXT NOT NULL)""")
            self.connection.execute("""CREATE TABLE IF NOT EXISTS agent_procedures (
                signature TEXT PRIMARY KEY, successes INTEGER NOT NULL, record_id TEXT NOT NULL,
                steps_json TEXT NOT NULL, evidence_json TEXT NOT NULL)""")
            self.connection.execute("""CREATE TABLE IF NOT EXISTS agent_procedure_examples (
                proof_sha256 TEXT PRIMARY KEY, record_id TEXT NOT NULL)""")
        spaces = {r[0] for r in self.connection.execute("SELECT DISTINCT space_id FROM agent_semantic_index")}
        if spaces and spaces != {semantic.space_id}:
            raise ValueError("semantic index belongs to a different embedding model; rebuild explicitly")
        for vault in self.VAULTS:
            node_id = "agent-memory:" + vault
            if not mind.store.has_concept(node_id):
                mind.add_concept(node_id, vault, input_trunks=(InputTrunk.HEAR, InputTrunk.SEE), kind="memory-vault")
            mind.store.set_concept_vault(node_id, "agent-vault:" + vault)

    def add(self, record, vault, *, evidence=None, vector=None):
        if vault not in self.VAULTS:
            raise ValueError("unknown memory vault")
        if self.connection.execute("SELECT 1 FROM agent_semantic_index WHERE record_id=?", (record.record_id,)).fetchone():
            return
        vector = self.semantic.embed([record.text])[0] if vector is None else vector
        with self.connection:
            self.connection.execute("INSERT INTO agent_semantic_index VALUES (?,?,?,?,?)",
                (record.record_id, vault, self.semantic.space_id, np.asarray(vector, dtype=np.float32).tobytes(),
                 json.dumps(evidence or {}, sort_keys=True)))
            self.mind.store.add_to_vault("agent-vault:" + vault, record.record_id, "agent-memory:" + vault)

    def import_existing(self):
        """Index attributable human memories; exclude generated claims/corpus bulk."""
        existing = {r[0] for r in self.connection.execute("SELECT record_id FROM agent_semantic_index")}
        candidates = []
        for record in self.mind.store.list_active_records():
            if record.record_id in existing or record.text.startswith("/"):
                continue
            metadata = record.metadata
            if metadata.get("explicit_user_memory"):
                candidates.append((record, "facts"))
            elif metadata.get("integrated_current_event") and record.source_id.casefold() == "human":
                candidates.append((record, "episodes"))
        for start in range(0, len(candidates), 16):
            batch = candidates[start:start + 16]
            vectors = self.semantic.embed([record.text for record, _ in batch])
            for (record, vault), vector in zip(batch, vectors):
                self.add(record, vault, evidence={"origin": "existing_canonical_human_record"}, vector=vector)
        return len(candidates)

    def repair_observations(self):
        """Reindex early action-intent entries using their verified returns."""
        repairs = []
        for row in self.connection.execute("SELECT * FROM agent_semantic_index WHERE vault='observations'").fetchall():
            evidence = json.loads(row["evidence_json"])
            return_id = evidence.get("return_record_id")
            if not return_id or row["record_id"] == return_id:
                continue
            returned = self.mind.store.get_record(return_id)
            cycle = self.connection.execute("""SELECT 1 FROM experience_cycles
                WHERE output_record_id=? AND terminal_return_record_id=? AND status='closed'""",
                (row["record_id"], return_id)).fetchone()
            if returned is None or not cycle or not returned.metadata.get("verified"):
                raise ValueError("observation index has no verified canonical return")
            repairs.append((row["record_id"], returned, evidence))
        for start in range(0, len(repairs), 16):
            batch = repairs[start:start + 16]
            vectors = self.semantic.embed([record.text for _, record, _ in batch])
            for (old_id, record, evidence), vector in zip(batch, vectors):
                self.add(record, "observations", evidence=evidence, vector=vector)
                with self.connection:
                    # Canonical requests and returns remain immutable. Only
                    # the derived semantic index entry was incorrect.
                    self.connection.execute("DELETE FROM agent_semantic_index WHERE record_id=?", (old_id,))
        return len(repairs)

    def written_files(self):
        """Creation claims come from successful writes, never directory lists."""
        result = {}
        rows = self.connection.execute("""SELECT r.record_id,r.text,r.metadata_json
            FROM agent_semantic_index i JOIN records r USING(record_id)
            WHERE i.vault='observations' AND r.source_id='ability:workspace-write'
            ORDER BY r.timestamp,r.rowid""").fetchall()
        for row in rows:
            payload = json.loads(row["text"])
            metadata = json.loads(row["metadata_json"])
            output = payload.get("output", {})
            if payload.get("status") != "success" or not output.get("write_verified") or not metadata.get("verified"):
                continue
            path = output["path"]
            # In the original guarded writer, a successful write without an
            # expected hash necessarily targeted a previously absent file.
            created = output.get("created", not metadata.get("arguments", {}).get("expected_sha256"))
            previous = result.pop(path, {})
            result[path] = {"path": path, "created_by_agent": bool(created or previous.get("created_by_agent")),
                "last_write_return_record_id": row["record_id"], "sha256": output["sha256"]}
        return list(reversed(list(result.values())))

    def search(self, query, *, exclude=(), limit=None):
        rows = self.connection.execute("SELECT * FROM agent_semantic_index").fetchall()
        if not rows:
            return {"matches": [], "query": query, "mode": "semantic_cosine_plus_habitus_vaults"}
        query_vector = self.semantic.embed([query])[0]
        excluded = set(exclude)
        candidates = []
        for row in rows:
            record = self.mind.store.get_record(row["record_id"])
            if record is None or record.record_id in excluded:
                continue
            # Superseded facts and procedures stay inspectable, but are not current evidence.
            superseded = self.connection.execute("SELECT 1 FROM records WHERE supersedes_id=? LIMIT 1", (record.record_id,)).fetchone()
            if superseded:
                continue
            score = float(np.dot(query_vector, np.frombuffer(row["vector"], dtype=np.float32)))
            candidates.append({"record_id": record.record_id, "text": record.text,
                "source": record.source_id, "timestamp": record.timestamp, "vault": row["vault"],
                "cosine": round(score, 6), "evidence": json.loads(row["evidence_json"])})
        candidates.sort(key=lambda row: (-row["cosine"], row["record_id"]))
        # Let the actual Habitus graph activate its paths and nominate bounded
        # complements, while preserving the strongest semantic matches first.
        packet, _ = self.mind.retrieval.retrieve(pulse_id=f"agent-recall:{self.mind.pulse}",
            query=query, query_embedding=self.mind.embedder.embed(query), event_kind=EventKind.MESSAGE,
            trunk=InputTrunk.HEAR, exclude_record_ids=excluded)
        graph_ids = set(packet.vault_record_ids)
        result = sorted(candidates, key=lambda row: (
            row["record_id"] not in graph_ids, -row["cosine"], row["record_id"]
        ))
        if limit is not None:
            result = result[:limit]
        return {"query": query, "mode": "semantic_cosine_plus_habitus_vaults",
            "matches": result, "habitus_trace_hash": packet.trace_hash,
            "graph_path_targets": [trace.target_node_id for trace in packet.y_paths]}

    def learn_procedure(self, objective, receipts):
        """Retain a successful write/read-back trajectory as reusable guidance."""
        actions = [r for r in receipts if r.verified and r.status == "success"
                   and r.tool_id.startswith("ability:workspace-")]
        verified_writes = []
        for index, receipt in enumerate(actions):
            if receipt.tool_id != "ability:workspace-write":
                continue
            output = receipt.output
            for later in actions[index + 1:]:
                if (later.tool_id == "ability:workspace-read" and isinstance(later.output, dict)
                        and later.output.get("path") == output.get("path")
                        and later.output.get("sha256") == output.get("sha256")):
                    verified_writes.append(receipt.receipt_id)
                    break
        if len(actions) < 3 or not verified_writes:
            return None
        steps = [r.tool_id.removeprefix("ability:workspace-") for r in actions]
        signature = hashlib.sha256(json.dumps(steps).encode()).hexdigest()[:24]
        proof = hashlib.sha256(json.dumps([r.receipt_id for r in actions]).encode()).hexdigest()
        previous_proof = self.connection.execute("SELECT record_id FROM agent_procedure_examples WHERE proof_sha256=?", (proof,)).fetchone()
        if previous_proof:
            return {"record_id": previous_proof[0], "already_consolidated": True}
        old = self.connection.execute("SELECT * FROM agent_procedures WHERE signature=?", (signature,)).fetchone()
        successes = 1 + (old["successes"] if old else 0)
        evidence = {"receipt_ids": [r.receipt_id for r in actions], "verified_write_receipts": verified_writes,
                    "successes": successes, "signature": signature, "guidance_only": True}
        text = (f"A procedure used successfully {successes} time(s): {' -> '.join(steps)}. "
                f"Latest task: {objective}\nRead the current source, perform the requested work, then reread the output "
                "and compare its actual contents. Adapt paths and contents to the new task; do not replay old file payloads.")
        record = self.mind.remember(text, source_id="verified-local-procedure", record_type=RecordType.RECEIPT,
            input_trunk=InputTrunk.NOTICE, concept_ids=("agent-memory:skills",), metadata=evidence,
            supersedes_id=old["record_id"] if old else None, pulse_number=self.mind.pulse)
        self.add(record, "skills", evidence=evidence)
        with self.connection:
            self.connection.execute("INSERT OR REPLACE INTO agent_procedures VALUES (?,?,?,?,?)",
                (signature, successes, record.record_id, json.dumps(steps), json.dumps(evidence)))
            self.connection.execute("INSERT INTO agent_procedure_examples VALUES (?,?)", (proof, record.record_id))
        return {"record_id": record.record_id, **evidence}
