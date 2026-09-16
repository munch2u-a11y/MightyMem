from __future__ import annotations

from dataclasses import asdict, dataclass

from .models import Capability, TaskState
from .retrieval import key_terms
from .store import LoomStore


@dataclass(slots=True)
class CapabilityEntry:
    capability_id: str
    admitted: bool
    eligible: bool
    score: float
    reason: str


class CapabilitySelector:
    """Selects visible capabilities; execution authorization remains separate."""

    def __init__(self, store: LoomStore, *, max_visible: int | None = None) -> None:
        if max_visible is not None and max_visible < 1:
            raise ValueError("at least one capability must be visible")
        self.store = store
        self.max_visible = max_visible

    @staticmethod
    def _eligible(capability: Capability, scope: str, stage: str) -> tuple[bool, str]:
        if not capability.enabled:
            return False, "disabled"
        if "*" not in capability.scopes and scope not in capability.scopes:
            return False, "scope mismatch"
        if "*" not in capability.phases and stage not in capability.phases:
            return False, "task phase mismatch"
        return True, "eligible"

    def select(
        self,
        *,
        query: str,
        scope: str,
        task: TaskState | None,
    ) -> tuple[list[Capability], list[CapabilityEntry]]:
        stage = task.stage if task else "understand"
        query_tokens = key_terms(" ".join([query, task.goal if task else "", stage]))
        mandatory = set(task.mandatory_capability_ids if task else [])
        scored: list[tuple[int, float, Capability, str]] = []
        entries: dict[str, CapabilityEntry] = {}
        for capability in self.store.list_capabilities():
            eligible, reason = self._eligible(capability, scope, stage)
            if not eligible:
                entries[capability.id] = CapabilityEntry(capability.id, False, False, 0.0, reason)
                continue
            # Contracts can share generic prose such as "return a receipt". Route
            # from explicit intent terms and the tool name rather than accidentally
            # treating boilerplate description words as capability evidence.
            keywords = set(capability.keywords) | set(capability.name.lower().split("_"))
            overlap = query_tokens & {word.lower() for word in keywords}
            score = len(overlap) / max(1, min(len(query_tokens), len(keywords)))
            score = min(1.0, score + (0.12 if capability.safe_core else 0.0) - min(0.2, capability.cost * 0.1))
            priority = 2 if capability.id in mandatory else 1 if capability.safe_core else 0
            why = "task-mandated" if priority == 2 else "safe fallback" if priority == 1 else "task/query match"
            scored.append((priority, score, capability, why))
        scored.sort(key=lambda item: (-item[0], -item[1], item[2].cost, item[2].id))
        selected: list[Capability] = []
        for priority, score, capability, why in scored:
            if self.max_visible is not None and len(selected) >= self.max_visible:
                entries[capability.id] = CapabilityEntry(
                    capability.id, False, True, score, "eligible but outside visible-tool cap"
                )
                continue
            # Non-core, non-mandated tools require at least some task or query evidence.
            if priority == 0 and score <= 0:
                entries[capability.id] = CapabilityEntry(
                    capability.id, False, True, score, "eligible but no task/query match"
                )
                continue
            selected.append(capability)
            entries[capability.id] = CapabilityEntry(capability.id, True, True, score, why)
        # If every match was weak, preserve a visible expansion route when registered.
        expansion = next(
            (capability for _, _, capability, _ in scored if capability.id == "request-capability"), None
        )
        if not selected and expansion is not None:
            selected.append(expansion)
            entries[expansion.id] = CapabilityEntry(expansion.id, True, True, 0.0, "empty-selection fallback")
        ordered_entries = [entries[c.id] for c in self.store.list_capabilities() if c.id in entries]
        return selected, ordered_entries


def capability_schema(capability: Capability) -> dict:
    # Runtime-only fields (scope, verifier, cost and stable id) stay in the
    # packet's typed Capability objects and admission manifest. Repeating them
    # in model-visible JSON wastes scarce input tokens without improving calls.
    return {
        "name": capability.name,
        "description": capability.description,
        "parameters": capability.schema,
    }


def default_capabilities() -> list[Capability]:
    object_schema = {"type": "object", "properties": {}, "additionalProperties": False}
    return [
        Capability(
            id="request-capability",
            name="request_capability",
            description="Ask the runtime to expand the visible capability set when none can perform the next step.",
            schema={
                "type": "object",
                "properties": {"need": {"type": "string"}},
                "required": ["need"],
                "additionalProperties": False,
            },
            phases=("*",),
            keywords=("need", "tool", "capability", "cannot"),
            scopes=("*",),
            safe_core=True,
            verifier="runtime_acknowledgement",
        ),
        Capability(
            id="list-files",
            name="list_files",
            description="List entries in the current project folder.",
            schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
            phases=("understand", "inspect", "verify"),
            keywords=("list", "folder", "directory", "files", "contents", "inspect"),
            verifier="returned_directory_entries",
        ),
        Capability(
            id="read-file",
            name="read_file",
            description="Read a file and return its exact content and hash.",
            schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
            phases=("understand", "inspect", "verify"),
            keywords=("read", "open", "inspect", "note", "document", "file", "verify"),
            verifier="content_and_sha256",
        ),
        Capability(
            id="search-memory",
            name="search_memory",
            description="Request additional source-linked memory when the compiled evidence is insufficient.",
            schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
            phases=("understand", "inspect", "verify", "report"),
            keywords=("remember", "memory", "find", "recall", "search", "evidence"),
            verifier="returned_record_ids",
        ),
        Capability(
            id="write-file",
            name="write_file",
            description=(
                "Write a project file with expected-current-hash protection. Use JSON null "
                "for expected_sha256 when inspection proves the target is absent."
            ),
            schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "expected_sha256": {"type": ["string", "null"]},
                },
                "required": ["path", "content", "expected_sha256"],
                "additionalProperties": False,
            },
            phases=("act", "revise"),
            keywords=("write", "save", "create", "revise", "update", "edit"),
            verifier="write_receipt_then_readback",
            cost=0.3,
        ),
        Capability(
            id="run-program",
            name="run_program",
            description="Run a registered program or a bounded project executable and return its observed process state, exit status, and output hashes.",
            schema={
                "type": "object",
                "properties": {
                    "program": {"type": "string"},
                    "arguments": {"type": "array", "items": {"type": "string"}},
                    "cwd": {"type": "string"},
                    "wait": {"type": "boolean"},
                    "timeout_seconds": {"type": ["number", "null"], "description": "Optional timeout explicitly chosen for this process; null allows it to finish normally."},
                },
                "required": ["program", "arguments", "cwd", "wait"],
                "additionalProperties": False,
            },
            phases=("act", "verify"),
            keywords=("run", "execute", "launch", "start", "open", "script", "program", "app"),
            verifier="process_exit_status_or_start_observation",
            cost=0.5,
        ),
        Capability(
            id="report-result",
            name="report_result",
            description="Return a final answer grounded in the current evidence and verified outcomes.",
            schema=object_schema,
            phases=("report",),
            keywords=("report", "answer", "finish", "summarize", "explain"),
            verifier="delivery_only",
            safe_core=True,
        ),
    ]
