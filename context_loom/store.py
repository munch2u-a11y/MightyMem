from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from .models import (
    Capability,
    ContextPacket,
    EvidenceStatus,
    MemoryRecord,
    OutcomeFeedback,
    RecordKind,
    TaskState,
)


SCHEMA_VERSION = 1


class IdentityConflict(ValueError):
    pass


class LoomStore:
    """SQLite authority for canonical records, task state and compiler traces."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self.connection.execute("PRAGMA foreign_keys=ON")
        if self.path != ":memory:":
            self.connection.execute("PRAGMA journal_mode=WAL")
        self._create_schema()

    def _create_schema(self) -> None:
        with self.connection:
            self.connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS records (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    content TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    effective_at REAL NOT NULL,
                    scope TEXT NOT NULL,
                    source_ids TEXT NOT NULL,
                    importance REAL NOT NULL,
                    confidence REAL NOT NULL,
                    evidence_status TEXT NOT NULL,
                    supersedes_id TEXT,
                    metadata TEXT NOT NULL,
                    FOREIGN KEY(supersedes_id) REFERENCES records(id)
                );
                CREATE INDEX IF NOT EXISTS idx_records_scope_time
                    ON records(scope, created_at);
                CREATE INDEX IF NOT EXISTS idx_records_kind_scope
                    ON records(kind, scope);
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS capabilities (
                    id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS context_packets (
                    id TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    scope TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS feedback (
                    id TEXT PRIMARY KEY,
                    packet_id TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    payload TEXT NOT NULL,
                    FOREIGN KEY(packet_id) REFERENCES context_packets(id)
                );
                CREATE TABLE IF NOT EXISTS record_utility (
                    record_id TEXT PRIMARY KEY,
                    positive REAL NOT NULL DEFAULT 0,
                    negative REAL NOT NULL DEFAULT 0,
                    observations INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY(record_id) REFERENCES records(id)
                );
                CREATE TABLE IF NOT EXISTS embeddings (
                    record_id TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    vector TEXT NOT NULL,
                    PRIMARY KEY(record_id, provider),
                    FOREIGN KEY(record_id) REFERENCES records(id)
                );
                """
            )
            self.connection.execute(
                "INSERT OR IGNORE INTO metadata(key,value) VALUES('schema_version',?)",
                (str(SCHEMA_VERSION),),
            )

    @staticmethod
    def _hash(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def append_record(self, record: MemoryRecord) -> MemoryRecord:
        payload_hash = self._hash(record.content)
        with self._lock, self.connection:
            existing = self.connection.execute(
                "SELECT content_hash, kind, scope FROM records WHERE id=?", (record.id,)
            ).fetchone()
            if existing:
                if (
                    existing["content_hash"] != payload_hash
                    or existing["kind"] != record.kind.value
                    or existing["scope"] != record.scope
                ):
                    raise IdentityConflict(f"record id {record.id!r} already has different canonical content")
                return self.get_record(record.id)  # type: ignore[return-value]
            if record.supersedes_id:
                target = self.connection.execute(
                    "SELECT scope FROM records WHERE id=?", (record.supersedes_id,)
                ).fetchone()
                if target is None:
                    raise ValueError("superseded record must already exist")
                if target["scope"] != record.scope:
                    raise ValueError("a correction cannot supersede a record in another scope")
            self.connection.execute(
                """INSERT INTO records(
                    id,kind,content,content_hash,created_at,effective_at,scope,
                    source_ids,importance,confidence,evidence_status,supersedes_id,metadata
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    record.id,
                    record.kind.value,
                    record.content,
                    payload_hash,
                    record.created_at,
                    record.effective_at,
                    record.scope,
                    json.dumps(list(record.source_ids)),
                    record.importance,
                    record.confidence,
                    record.evidence_status.value,
                    record.supersedes_id,
                    json.dumps(record.metadata, sort_keys=True),
                ),
            )
        return record

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            id=row["id"],
            kind=RecordKind(row["kind"]),
            content=row["content"],
            created_at=float(row["created_at"]),
            effective_at=float(row["effective_at"]),
            scope=row["scope"],
            source_ids=tuple(json.loads(row["source_ids"])),
            importance=float(row["importance"]),
            confidence=float(row["confidence"]),
            evidence_status=EvidenceStatus(row["evidence_status"]),
            supersedes_id=row["supersedes_id"],
            metadata=json.loads(row["metadata"]),
        )

    def get_record(self, record_id: str) -> MemoryRecord | None:
        row = self.connection.execute("SELECT * FROM records WHERE id=?", (record_id,)).fetchone()
        return self._row_to_record(row) if row else None

    def list_records(
        self,
        *,
        scope: str,
        as_of: float | None = None,
        kinds: Iterable[RecordKind] | None = None,
        exclude_ids: Iterable[str] = (),
    ) -> list[MemoryRecord]:
        clauses = ["scope=?"]
        args: list[object] = [scope]
        if as_of is not None:
            clauses.append("created_at<=?")
            args.append(as_of)
        if kinds:
            kind_values = [k.value for k in kinds]
            clauses.append("kind IN (" + ",".join("?" for _ in kind_values) + ")")
            args.extend(kind_values)
        excluded = list(exclude_ids)
        if excluded:
            clauses.append("id NOT IN (" + ",".join("?" for _ in excluded) + ")")
            args.extend(excluded)
        rows = self.connection.execute(
            "SELECT * FROM records WHERE " + " AND ".join(clauses) + " ORDER BY created_at ASC, id ASC",
            args,
        ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def superseding_record(self, record_id: str, *, as_of: float | None = None) -> MemoryRecord | None:
        query = "SELECT * FROM records WHERE supersedes_id=?"
        args: list[object] = [record_id]
        if as_of is not None:
            query += " AND created_at<=? AND effective_at<=?"
            args.extend((as_of, as_of))
        query += " ORDER BY effective_at DESC, created_at DESC LIMIT 1"
        row = self.connection.execute(query, args).fetchone()
        return self._row_to_record(row) if row else None

    def latest_version(self, record_id: str, *, as_of: float | None = None) -> MemoryRecord | None:
        current = self.get_record(record_id)
        if current is None or (
            as_of is not None and (current.created_at > as_of or float(current.effective_at) > as_of)
        ):
            return None
        seen = {current.id}
        while True:
            newer = self.superseding_record(current.id, as_of=as_of)
            if newer is None or newer.id in seen:
                return current
            seen.add(newer.id)
            current = newer

    def upsert_task(self, task: TaskState) -> None:
        payload = asdict(task)
        with self._lock, self.connection:
            self.connection.execute(
                """INSERT INTO tasks(id,payload,updated_at) VALUES(?,?,?)
                ON CONFLICT(id) DO UPDATE SET payload=excluded.payload, updated_at=excluded.updated_at""",
                (task.id, json.dumps(payload, sort_keys=True), task.updated_at),
            )

    def get_task(self, task_id: str) -> TaskState | None:
        row = self.connection.execute("SELECT payload FROM tasks WHERE id=?", (task_id,)).fetchone()
        return TaskState(**json.loads(row["payload"])) if row else None

    def register_capability(self, capability: Capability) -> None:
        payload = asdict(capability)
        with self._lock, self.connection:
            row = self.connection.execute("SELECT payload FROM capabilities WHERE id=?", (capability.id,)).fetchone()
            encoded = json.dumps(payload, sort_keys=True)
            if row and row["payload"] != encoded:
                raise IdentityConflict(f"capability id {capability.id!r} already has a different contract")
            self.connection.execute(
                "INSERT OR IGNORE INTO capabilities(id,payload) VALUES(?,?)", (capability.id, encoded)
            )

    def list_capabilities(self) -> list[Capability]:
        rows = self.connection.execute("SELECT payload FROM capabilities ORDER BY id").fetchall()
        result = []
        for row in rows:
            data = json.loads(row["payload"])
            for field_name in ("phases", "keywords", "scopes"):
                data[field_name] = tuple(data[field_name])
            result.append(Capability(**data))
        return result

    def save_packet(self, packet: ContextPacket) -> None:
        payload = json.dumps(packet.to_dict(), sort_keys=True)
        with self._lock, self.connection:
            existing = self.connection.execute(
                "SELECT payload FROM context_packets WHERE id=?", (packet.id,)
            ).fetchone()
            if existing and existing["payload"] != payload:
                raise IdentityConflict(f"context packet id {packet.id!r} changed")
            self.connection.execute(
                "INSERT OR IGNORE INTO context_packets(id,created_at,scope,payload) VALUES(?,?,?,?)",
                (packet.id, packet.created_at, packet.scope, payload),
            )

    def get_packet_payload(self, packet_id: str) -> dict | None:
        row = self.connection.execute(
            "SELECT payload FROM context_packets WHERE id=?", (packet_id,)
        ).fetchone()
        return json.loads(row["payload"]) if row else None

    def record_feedback(self, feedback: OutcomeFeedback) -> None:
        helpful = set(feedback.helpful_record_ids)
        harmful = set(feedback.harmful_record_ids)
        if helpful & harmful:
            raise ValueError("the same record cannot be both helpful and harmful")
        if not self.get_packet_payload(feedback.packet_id):
            raise ValueError("feedback requires an existing context packet")
        payload = json.dumps(asdict(feedback), sort_keys=True)
        with self._lock, self.connection:
            existing = self.connection.execute("SELECT payload FROM feedback WHERE id=?", (feedback.id,)).fetchone()
            if existing:
                if existing["payload"] != payload:
                    raise IdentityConflict(f"feedback id {feedback.id!r} changed")
                return
            self.connection.execute(
                "INSERT INTO feedback(id,packet_id,created_at,payload) VALUES(?,?,?,?)",
                (feedback.id, feedback.packet_id, feedback.created_at, payload),
            )
            # Candidate-specific feedback only. Mere co-retrieval receives no credit.
            for record_id, positive, negative in (
                [(r, max(0.0, feedback.task_progress), 0.0) for r in helpful]
                + [(r, 0.0, max(0.0, -feedback.task_progress) or 1.0) for r in harmful]
            ):
                if self.get_record(record_id) is None:
                    raise ValueError(f"feedback references unknown record {record_id}")
                self.connection.execute(
                    """INSERT INTO record_utility(record_id,positive,negative,observations)
                    VALUES(?,?,?,1)
                    ON CONFLICT(record_id) DO UPDATE SET
                        positive=positive+excluded.positive,
                        negative=negative+excluded.negative,
                        observations=observations+1""",
                    (record_id, positive, negative),
                )

    def learned_utility(self, record_id: str) -> float:
        row = self.connection.execute(
            "SELECT positive,negative,observations FROM record_utility WHERE record_id=?", (record_id,)
        ).fetchone()
        if not row or row["observations"] == 0:
            return 0.0
        return max(-1.0, min(1.0, (row["positive"] - row["negative"]) / row["observations"]))

    def put_embedding(self, record_id: str, provider: str, vector: list[float]) -> None:
        record = self.get_record(record_id)
        if record is None:
            raise ValueError("cannot embed unknown record")
        with self._lock, self.connection:
            self.connection.execute(
                """INSERT INTO embeddings(record_id,provider,content_hash,vector) VALUES(?,?,?,?)
                ON CONFLICT(record_id,provider) DO UPDATE SET
                    content_hash=excluded.content_hash, vector=excluded.vector""",
                (record_id, provider, self._hash(record.content), json.dumps(vector)),
            )

    def get_embedding(self, record_id: str, provider: str) -> list[float] | None:
        record = self.get_record(record_id)
        if record is None:
            return None
        row = self.connection.execute(
            "SELECT content_hash,vector FROM embeddings WHERE record_id=? AND provider=?",
            (record_id, provider),
        ).fetchone()
        if not row or row["content_hash"] != self._hash(record.content):
            return None
        return list(json.loads(row["vector"]))

    def integrity_check(self) -> str:
        return str(self.connection.execute("PRAGMA quick_check").fetchone()[0])

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "LoomStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
