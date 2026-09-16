from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any, Iterable

from .capabilities import default_capabilities
from .compiler import CompilerConfig, ContextCompiler
from .models import (
    Capability,
    ContextPacket,
    EvidenceStatus,
    MemoryRecord,
    OutcomeFeedback,
    RecordKind,
    TaskState,
)
from .retrieval import EmbeddingProvider
from .store import LoomStore
from .tokenization import TokenCounter


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


class ContextLoom:
    """High-level persistent API for events, tasks, context and feedback."""

    def __init__(
        self,
        database: str | Path = ":memory:",
        *,
        config: CompilerConfig | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        token_counter: TokenCounter | None = None,
        install_default_capabilities: bool = True,
    ) -> None:
        self.store = LoomStore(database)
        if install_default_capabilities and not self.store.list_capabilities():
            for capability in default_capabilities():
                self.store.register_capability(capability)
        self.compiler = ContextCompiler(
            self.store,
            config=config,
            embedding_provider=embedding_provider,
            token_counter=token_counter,
        )
        from .entity_graph import EntityGraphStore
        from .multi_hop_lane import MultiHopGraphLane, YCipherDepthSearchLane
        from .x_membrane import XMembrane, ShadowXCypher
        self.graph_store = EntityGraphStore(self.store)
        self.x_membrane = XMembrane(self.store)
        self.shadow_x_cypher = ShadowXCypher()
        existing_names = {getattr(l, "name", "") for l in self.compiler.lanes}
        if "multi_hop_graph" not in existing_names:
            self.compiler.lanes.append(MultiHopGraphLane(self.store, self.graph_store))
        if "y_cipher_depth" not in existing_names:
            self.compiler.lanes.append(YCipherDepthSearchLane(self.store, self.graph_store))
        self._in_reflection: bool = False

    def record(
        self,
        kind: RecordKind,
        content: str,
        *,
        scope: str = "default",
        record_id: str | None = None,
        created_at: float | None = None,
        effective_at: float | None = None,
        source_ids: Iterable[str] = (),
        importance: float = 0.5,
        confidence: float = 0.5,
        evidence_status: EvidenceStatus = EvidenceStatus.UNVERIFIED,
        supersedes_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        timestamp = time.time() if created_at is None else created_at
        record = MemoryRecord(
            id=record_id or _id(kind.value),
            kind=kind,
            content=content,
            created_at=timestamp,
            effective_at=effective_at,
            scope=scope,
            source_ids=tuple(source_ids),
            importance=importance,
            confidence=confidence,
            evidence_status=evidence_status,
            supersedes_id=supersedes_id,
            metadata=dict(metadata or {}),
        )
        return self.store.append_record(record)

    def remember_user_statement(self, content: str, **kwargs: Any) -> MemoryRecord:
        return self.record(
            RecordKind.USER,
            content,
            evidence_status=EvidenceStatus.USER_REPORTED,
            **kwargs,
        )

    def remember_assistant_statement(self, content: str, **kwargs: Any) -> MemoryRecord:
        return self.record(
            RecordKind.ASSISTANT,
            content,
            evidence_status=EvidenceStatus.MODEL_INFERRED,
            **kwargs,
        )

    def remember_preference(self, content: str, *, importance: float = 0.8, **kwargs: Any) -> MemoryRecord:
        return self.record(
            RecordKind.PREFERENCE,
            content,
            importance=importance,
            evidence_status=EvidenceStatus.USER_REPORTED,
            **kwargs,
        )

    def log(
        self,
        speaker: str,
        content: str,
        kind: RecordKind = RecordKind.USER,
        scope: str = "default",
        evidence_status: EvidenceStatus = EvidenceStatus.USER_REPORTED,
        **kwargs: Any,
    ) -> MemoryRecord:
        """Log dialogue turn with X-membrane activation (+1) and Y-Cypher vault routing."""
        metadata = dict(kwargs.pop("metadata", {}))
        metadata["speaker"] = speaker
        rec = self.record(
            kind=kind,
            content=content,
            scope=scope,
            evidence_status=evidence_status,
            metadata=metadata,
            **kwargs,
        )

        x_terms, tuple_groupings = self.shadow_x_cypher.parse_input(content)
        if speaker:
            x_terms.append(speaker)
        self.x_membrane.activate(x_terms)
        for tup in tuple_groupings:
            self.x_membrane.activate([" ".join(tup)])

        paths = self.graph_store.y_cypher_find_paths(
            query=content,
            x_terms=x_terms,
            embedding_provider=self.compiler.embedding_provider,
        )
        if not paths:
            for term in x_terms:
                if len(term) >= 3 and term not in ("the", "and", "for", "with", "really", "enjoy"):
                    cnode = self.graph_store.create_vector_concept_node(
                        name=f"Experience Topic: {term.capitalize()}",
                        parent_id="self_root",
                        y_depth=1,
                    )
                    self.graph_store.link_x_membrane_to_y_concept(term, cnode.id)
            paths = self.graph_store.y_cypher_find_paths(
                query=content,
                x_terms=x_terms,
                embedding_provider=self.compiler.embedding_provider,
            )
        for path in paths:
            # 1. Update activation metadata across ALL Y-nodes along path
            for nid in path.node_ids:
                self.graph_store.activate_y_node_turn(node_id=nid, turn_record_id=rec.id)

            # 2. Attach memory vault record to upper-layer Y-nodes (parent hubs, y_depth <= 1)
            upper_target_id = path.node_ids[1] if len(path.node_ids) > 1 else path.target_concept_id
            self.graph_store.link_concept_record(upper_target_id, rec.id, weight=path.total_score)

        return rec

    def pair_gestation_concepts(self, human_name: str, agent_name: str) -> None:
        """Pair basic AI and person concepts and relations during gestation."""
        from .entity_graph import EntityKind
        self_node = self.graph_store.get_or_create_self_node()

        gestation_terms = [
            human_name.casefold(),
            agent_name.casefold(),
            "ai assistant",
            "human user",
            "habitus loom engine",
            "user",
            "assistant",
            "person",
            "identity",
        ]
        self.x_membrane.activate(gestation_terms)

        ai_concept = self.graph_store.create_vector_concept_node(
            name=f"{agent_name} AI Identity & Capabilities",
            parent_id=self_node.id,
            y_depth=1,
            kind=EntityKind.PERSON,
        )
        user_concept = self.graph_store.create_vector_concept_node(
            name=f"{human_name} Human User Identity & Experience",
            parent_id=self_node.id,
            y_depth=1,
            kind=EntityKind.PERSON,
        )

        for term in [agent_name, "ai assistant", "habitus loom engine", "assistant"]:
            self.graph_store.link_x_membrane_to_y_concept(term, ai_concept.id)

        for term in [human_name, "human user", "user", "person"]:
            self.graph_store.link_x_membrane_to_y_concept(term, user_concept.id)

    def remember_derived_fact(
        self,
        content: str,
        *,
        source_ids: Iterable[str],
        support_assessment: str,
        **kwargs: Any,
    ) -> MemoryRecord:
        sources = tuple(source_ids)
        if not sources:
            raise ValueError("derived facts require source records")
        if support_assessment not in {"entailed", "extractive", "state_machine"}:
            raise ValueError("unsupported fact assessment")
        for source_id in sources:
            if self.store.get_record(source_id) is None:
                raise ValueError(f"unknown fact source {source_id}")
        metadata = dict(kwargs.pop("metadata", {}))
        metadata["support_assessment"] = support_assessment
        return self.record(
            RecordKind.FACT,
            content,
            source_ids=sources,
            evidence_status=EvidenceStatus.SYSTEM_DERIVED,
            metadata=metadata,
            **kwargs,
        )

    def correct(
        self,
        supersedes_id: str,
        content: str,
        *,
        scope: str | None = None,
        **kwargs: Any,
    ) -> MemoryRecord:
        old = self.store.get_record(supersedes_id)
        if old is None:
            raise ValueError("cannot correct an unknown record")
        correction_scope = old.scope if scope is None else scope
        return self.record(
            RecordKind.CORRECTION,
            content,
            scope=correction_scope,
            evidence_status=EvidenceStatus.USER_REPORTED,
            supersedes_id=supersedes_id,
            **kwargs,
        )

    def remember_procedure(
        self,
        content: str,
        *,
        trajectory_ids: Iterable[str],
        reliability: float,
        verified: bool,
        **kwargs: Any,
    ) -> MemoryRecord:
        trajectories = tuple(trajectory_ids)
        metadata = dict(kwargs.pop("metadata", {}))
        metadata.update({"trajectory_ids": list(trajectories), "reliability": reliability})
        return self.record(
            RecordKind.PROCEDURE,
            content,
            source_ids=trajectories,
            confidence=reliability,
            evidence_status=EvidenceStatus.TOOL_VERIFIED if verified else EvidenceStatus.UNVERIFIED,
            metadata=metadata,
            **kwargs,
        )

    def begin_task(
        self,
        task_id: str,
        goal: str,
        *,
        scope: str = "default",
        stage: str = "understand",
        focus_stack: Iterable[str] = (),
        open_questions: Iterable[str] = (),
        constraints: Iterable[str] = (),
        completion_conditions: Iterable[str] = (),
        mandatory_capability_ids: Iterable[str] = (),
        now: float | None = None,
    ) -> TaskState:
        task = TaskState(
            id=task_id,
            goal=goal,
            scope=scope,
            stage=stage,
            focus_stack=list(focus_stack),
            open_questions=list(open_questions),
            constraints=list(constraints),
            completion_conditions=list(completion_conditions),
            mandatory_capability_ids=list(mandatory_capability_ids),
            updated_at=time.time() if now is None else now,
        )
        self.store.upsert_task(task)
        return task

    def update_task(self, task_id: str, *, now: float | None = None, **changes: Any) -> TaskState:
        task = self.store.get_task(task_id)
        if task is None:
            raise ValueError(f"unknown task {task_id}")
        allowed = {
            "goal", "stage", "status", "focus_stack", "open_questions", "constraints",
            "completion_conditions", "last_observation_id", "mandatory_capability_ids",
        }
        unknown = set(changes) - allowed
        if unknown:
            raise ValueError(f"unsupported task fields: {sorted(unknown)}")
        for name, value in changes.items():
            if name in {
                "focus_stack", "open_questions", "constraints", "completion_conditions",
                "mandatory_capability_ids",
            }:
                value = list(value)
            setattr(task, name, value)
        task.updated_at = time.time() if now is None else now
        self.store.upsert_task(task)
        return task

    def record_tool_result(
        self,
        content: str,
        *,
        task_id: str | None = None,
        verified: bool,
        receipt_id: str,
        verifier: str,
        scope: str = "default",
        next_stage: str | None = None,
        metadata: dict[str, Any] | None = None,
        now: float | None = None,
    ) -> MemoryRecord:
        details = dict(metadata or {})
        details.update({"receipt_id": receipt_id, "verifier": verifier, "verified": verified})
        result = self.record(
            RecordKind.TOOL_RESULT,
            content,
            scope=scope,
            created_at=now,
            evidence_status=EvidenceStatus.TOOL_VERIFIED if verified else EvidenceStatus.UNVERIFIED,
            metadata=details,
        )
        if task_id and verified:
            changes: dict[str, Any] = {"last_observation_id": result.id}
            if next_stage:
                changes["stage"] = next_stage
            self.update_task(task_id, now=now, **changes)
        return result

    def compile_input(
        self,
        content: str,
        *,
        scope: str = "default",
        task_id: str | None = None,
        mode: str = "reply",
        now: float | None = None,
    ) -> ContextPacket:
        event = self.remember_user_statement(content, scope=scope, created_at=now)
        return self.compiler.compile(
            content,
            scope=scope,
            task_id=task_id,
            current_event_id=event.id,
            mode=mode,
            now=event.created_at,
        )

    def record_response(
        self,
        content: str,
        *,
        scope: str = "default",
        source_packet_id: str,
        now: float | None = None,
    ) -> MemoryRecord:
        if self.store.get_packet_payload(source_packet_id) is None:
            raise ValueError("assistant responses require a source context packet")
        return self.record(
            RecordKind.ASSISTANT,
            content,
            scope=scope,
            created_at=now,
            source_ids=(source_packet_id,),
            evidence_status=EvidenceStatus.UNVERIFIED,
        )

    def register_capability(self, capability: Capability) -> None:
        self.store.register_capability(capability)

    def record_outcome(
        self,
        packet_id: str,
        *,
        verified: bool,
        task_progress: float,
        helpful_record_ids: Iterable[str] = (),
        harmful_record_ids: Iterable[str] = (),
        note: str = "",
        feedback_id: str | None = None,
        now: float | None = None,
    ) -> OutcomeFeedback:
        if task_progress > 0 and not verified:
            raise ValueError("positive learning credit requires a verified outcome")
        payload = self.store.get_packet_payload(packet_id)
        if payload is None:
            raise ValueError("unknown context packet")
        admitted = {
            record_id
            for item in payload["manifest"]
            if item["admitted"]
            for record_id in item["source_record_ids"]
        }
        referenced = set(helpful_record_ids) | set(harmful_record_ids)
        if not referenced <= admitted:
            raise ValueError("feedback can credit only records admitted to that packet")
        feedback = OutcomeFeedback(
            id=feedback_id or _id("feedback"),
            packet_id=packet_id,
            verified=verified,
            task_progress=task_progress,
            helpful_record_ids=tuple(helpful_record_ids),
            harmful_record_ids=tuple(harmful_record_ids),
            note=note,
            created_at=time.time() if now is None else now,
        )
        self.store.record_feedback(feedback)
        return feedback

    def reflect(self, scope: str = "default", *, now: float | None = None) -> MemoryRecord | None:
        if self._in_reflection:
            return None
        self._in_reflection = True
        try:
            timestamp = time.time() if now is None else now
            all_records = self.store.list_records(scope=scope, as_of=timestamp)
            already_indexed = {
                source_id
                for record in all_records
                if record.kind == RecordKind.THOUGHT
                and record.metadata.get("reflection_checkpoint") is True
                for source_id in record.source_ids
            }
            records = [
                record
                for record in all_records
                if record.id not in already_indexed
                and record.kind not in {RecordKind.THOUGHT, RecordKind.TOOL_REQUEST}
            ]
            if not records:
                return None

            from .entity_graph import EntityExtractor
            from .reflection import TopicalReflectionEngine

            graph_store = self.graph_store
            extractor = EntityExtractor()
            for record in records:
                graph_store.index_record(
                    record.id,
                    record.content,
                    speaker=record.metadata.get("speaker"),
                    extractor=extractor,
                )

            topical_engine = TopicalReflectionEngine(self)
            topical_engine.run_reflection_pass(scope=scope)

            source_ids = tuple(record.id for record in records)
            reflection_rec = self.record(
                RecordKind.THOUGHT,
                f"Reflection checkpoint indexed {len(records)} new records in scope {scope!r}.",
                scope=scope,
                created_at=timestamp,
                source_ids=source_ids,
                evidence_status=EvidenceStatus.SYSTEM_DERIVED,
                metadata={
                    "reflection_checkpoint": True,
                    "reflected_record_count": len(records),
                    "support_assessment": "state_machine",
                    "source_ids": list(source_ids),
                },
            )
            return reflection_rec
        finally:
            self._in_reflection = False

    def end_session(self, scope: str = "default", *, now: float | None = None) -> MemoryRecord | None:
        return self.reflect(scope=scope, now=now)

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> "ContextLoom":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
