"""Current-event conversation with opaque numeric state, not transcript replay.

The backend owns its tokenizer and recurrent tensors. This module owns turn
framing, exactly-once consumption, bounded decoding and transactional persistence.
Developer-facing receipts are returned, never fed into subsequent turns.
"""
from __future__ import annotations

import hashlib
import itertools
from pathlib import Path
import sqlite3
import time

PROTOCOL = "rwkv-current-event-no-thinking-v1"
ASSISTANT_PREFIX = "Assistant: <think></think>\n"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class RecurrentConversation:
    def __init__(self, backend, *, state_path: Path | None = None):
        self.backend = backend
        self.turn = 0
        self.database = None
        self.identity = f"{PROTOCOL}:{backend.identity}"
        initial = backend.dump_state()
        self.persisted_sha = sha256(initial)
        if state_path is not None:
            self.database = sqlite3.connect(state_path)
            try:
                self.database.execute(
                    "CREATE TABLE IF NOT EXISTS recurrent_state ("
                    "id INTEGER PRIMARY KEY CHECK(id=1), identity TEXT NOT NULL, "
                    "turn INTEGER NOT NULL, numeric_state BLOB NOT NULL, sha256 TEXT NOT NULL)"
                )
                self.database.execute("INSERT OR IGNORE INTO recurrent_state VALUES (1,?,?,?,?)",
                                      (self.identity, 0, initial, self.persisted_sha))
                self.database.commit()
                identity, self.turn, saved, digest = self.database.execute(
                    "SELECT identity,turn,numeric_state,sha256 FROM recurrent_state WHERE id=1"
                ).fetchone()
                if identity != self.identity:
                    raise ValueError("state belongs to different weights or execution protocol")
                if sha256(saved) != digest:
                    raise ValueError("numeric state checksum mismatch")
                backend.load_state(saved)
                self.persisted_sha = digest
            except BaseException:
                self.database.close()
                self.database = None
                raise
        self.closed = False

    def reply(self, message: str, *, maximum_tokens: int | None = None) -> dict:
        if self.closed:
            raise RuntimeError("conversation is closed")
        if not message.strip():
            raise ValueError("current event cannot be empty")
        if maximum_tokens is not None and maximum_tokens < 1:
            raise ValueError("an explicit diagnostic decode limit must be positive")
        before = self.backend.dump_state()
        began = time.monotonic()
        try:
            # No past-message collection exists here. Only this event is encoded.
            heard_ids = self.backend.encode(f"User: {message.strip()}\n\n")
            self.backend.consume(heard_ids)
            heard = self.backend.dump_state()
            prefix_ids = self.backend.encode(ASSISTANT_PREFIX)
            next_token = self.backend.consume(prefix_ids)
            generated, complete, stop = [], False, "pending"
            boundary_ids = []
            token_steps = itertools.count() if maximum_tokens is None else range(maximum_tokens)
            for _ in token_steps:
                token = next_token
                generated.append(token)
                # Commit even the final sampled token exactly once. No SPEAK
                # echo is subsequently presented as another HEAR event.
                next_token = self.backend.consume([token])
                if token == self.backend.eos_token_id:
                    complete, stop = True, "eos"
                    boundary_ids = self.backend.encode("\n\n")
                    self.backend.consume(boundary_ids)
                    break
                if "\n\n" in self.backend.decode(generated):
                    complete, stop = True, "paragraph_boundary"
                    break
            if not complete and maximum_tokens is not None:
                stop = "explicit_diagnostic_limit"
            candidate = self.backend.decode([t for t in generated if t != self.backend.eos_token_id])
            if not complete:
                # An unfinished candidate is evidence for the evaluator, not a
                # delivered turn. Only the actual HEAR event survives.
                self.backend.load_state(heard)
            after = self.backend.dump_state()
            digest = sha256(after)
            if self.database is not None:
                with self.database:
                    cursor = self.database.execute(
                        "UPDATE recurrent_state SET turn=?, numeric_state=?, sha256=? "
                        "WHERE id=1 AND turn=? AND sha256=?",
                        (self.turn + 1, after, digest, self.turn, self.persisted_sha),
                    )
                    if cursor.rowcount != 1:
                        raise RuntimeError("another session advanced this numeric state")
            self.turn += 1
            self.persisted_sha = digest
        except BaseException:
            self.backend.load_state(before)
            raise
        return {
            "turn": self.turn, "status": "complete" if complete else "incomplete",
            "response": candidate.strip() if complete else "", "candidate": candidate,
            "stop": stop, "generated_token_ids": generated,
            "current_event_token_count": len(heard_ids), "prefix_token_count": len(prefix_ids),
            "boundary_token_count": len(boundary_ids), "generated_token_count": len(generated),
            "state_before_sha256": sha256(before), "state_after_sha256": digest,
            "serialized_state_bytes": len(after), "elapsed_seconds": time.monotonic() - began,
            "history_tokens_refed": 0, "system_prompt_used": False, "rag_used": False,
            "protocol": PROTOCOL,
        }

    def close(self) -> None:
        if self.database is not None:
            self.database.close()
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
