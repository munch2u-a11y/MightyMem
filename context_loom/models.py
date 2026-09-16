from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class RecordKind(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    EPISODE = "episode"
    FACT = "fact"
    CORRECTION = "correction"
    CONFLICT = "conflict"
    PREFERENCE = "preference"
    TOOL_REQUEST = "tool_request"
    TOOL_RESULT = "tool_result"
    OBSERVATION = "observation"
    PROCEDURE = "procedure"
    THOUGHT = "thought"


class EvidenceStatus(str, Enum):
    USER_REPORTED = "user_reported"
    TOOL_VERIFIED = "tool_verified"
    MODEL_INFERRED = "model_inferred"
    SYSTEM_DERIVED = "system_derived"
    PRIVATE = "private"
    UNVERIFIED = "unverified"


@dataclass(slots=True)
class MemoryRecord:
    id: str
    kind: RecordKind
    content: str
    created_at: float
    scope: str = "default"
    effective_at: float | None = None
    source_ids: tuple[str, ...] = ()
    importance: float = 0.5
    confidence: float = 0.5
    evidence_status: EvidenceStatus = EvidenceStatus.UNVERIFIED
    supersedes_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id or not self.content.strip():
            raise ValueError("records require non-empty id and content")
        if not 0.0 <= self.importance <= 1.0:
            raise ValueError("importance must be between 0 and 1")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if self.effective_at is None:
            self.effective_at = self.created_at


@dataclass(slots=True)
class TaskState:
    id: str
    goal: str
    scope: str = "default"
    stage: str = "understand"
    status: str = "active"
    focus_stack: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    completion_conditions: list[str] = field(default_factory=list)
    last_observation_id: str | None = None
    mandatory_capability_ids: list[str] = field(default_factory=list)
    updated_at: float = 0.0

    def __post_init__(self) -> None:
        if not self.id or not self.goal.strip():
            raise ValueError("tasks require non-empty id and goal")


@dataclass(slots=True)
class Capability:
    id: str
    name: str
    description: str
    schema: dict[str, Any]
    phases: tuple[str, ...] = ("*",)
    keywords: tuple[str, ...] = ()
    scopes: tuple[str, ...] = ("*",)
    safe_core: bool = False
    cost: float = 0.1
    verifier: str = "handler_return"
    enabled: bool = True

    def __post_init__(self) -> None:
        if not self.id or not self.name or not self.description:
            raise ValueError("capabilities require id, name and description")
        if self.cost < 0:
            raise ValueError("capability cost cannot be negative")


@dataclass(slots=True)
class Candidate:
    id: str
    text: str
    section: str
    source_record_ids: tuple[str, ...]
    lane_scores: dict[str, float] = field(default_factory=dict)
    lane_ranks: dict[str, int] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    hard: bool = False
    evidence_status: EvidenceStatus = EvidenceStatus.UNVERIFIED
    effective_at: float | None = None
    importance: float = 0.5
    learned_utility: float = 0.0
    nomination_count: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def merge(self, other: "Candidate") -> None:
        if self.id != other.id:
            raise ValueError("only identical candidates can merge")
        self.nomination_count += getattr(other, "nomination_count", 1)
        for lane, score in other.lane_scores.items():
            if lane in self.lane_scores:
                existing = self.lane_scores[lane]
                self.lane_scores[lane] = max(existing, score)
            else:
                self.lane_scores[lane] = score
        self.reasons.extend(r for r in other.reasons if r not in self.reasons)
        self.hard = self.hard or other.hard
        if other.section == "multi_hop" or "y_cipher_depth" in other.lane_scores or "subjective" in other.lane_scores:
            if self.section not in {"work", "personal", "continuity"}:
                bm25_score = self.lane_scores.get("bm25_identity", 0.0)
                exact_score = self.lane_scores.get("exact", 0.0)
                y_score = max(self.lane_scores.get("y_cipher_depth", 0.0), other.lane_scores.get("y_cipher_depth", 0.0), other.lane_scores.get("subjective", 0.0))
                if y_score >= max(bm25_score, exact_score):
                    self.section = "multi_hop"
        if other.metadata.get("conversational_tail"):
            self.metadata["conversational_tail"] = True


@dataclass(slots=True)
class ManifestEntry:
    candidate_id: str
    source_record_ids: tuple[str, ...]
    lanes: dict[str, float]
    admitted: bool
    reason: str
    token_cost: int
    utility: float
    lane_ranks: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class ContextPacket:
    id: str
    created_at: float
    scope: str
    current_input: str
    current_event_id: str | None
    task_id: str | None
    mode: str
    sections: dict[str, list[str]]
    capabilities: list[Capability]
    manifest: list[ManifestEntry]
    token_count: int
    max_input_tokens: int
    reserved_output_tokens: int
    compiler_version: str
    rendered: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class OutcomeFeedback:
    id: str
    packet_id: str
    verified: bool
    task_progress: float
    helpful_record_ids: tuple[str, ...] = ()
    harmful_record_ids: tuple[str, ...] = ()
    note: str = ""
    created_at: float = 0.0

    def __post_init__(self) -> None:
        if not -1.0 <= self.task_progress <= 1.0:
            raise ValueError("task progress must be between -1 and 1")


@dataclass(slots=True, frozen=True)
class ToolReceipt:
    id: str
    capability_id: str
    success: bool
    verified: bool
    verifier: str
    started_at: float
    completed_at: float
    request: dict[str, Any]
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
