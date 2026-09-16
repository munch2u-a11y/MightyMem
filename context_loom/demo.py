from __future__ import annotations

import json
import tempfile
from pathlib import Path

from .compiler import CompilerConfig
from .engine import ContextLoom
from .models import EvidenceStatus, RecordKind
from .runtime import BoundedFileRuntime


def build_demo(
    database: str | Path, workspace: str | Path
) -> tuple[ContextLoom, BoundedFileRuntime, dict[str, str]]:
    loom = ContextLoom(
        database,
        config=CompilerConfig(model_context_tokens=1200, reserved_output_tokens=260),
    )
    old = loom.remember_user_statement(
        "Project Maple uses PostgreSQL 14.",
        scope="maple",
        record_id="maple-db-v1",
        created_at=1_000,
        effective_at=1_000,
        importance=0.7,
    )
    correction = loom.correct(
        old.id,
        "Project Maple now uses PostgreSQL 16; the migration completed on July 8.",
        scope="maple",
        record_id="maple-db-v2",
        created_at=1_100,
        effective_at=1_100,
        importance=0.85,
    )
    helix = loom.remember_user_statement(
        "Helix has not been migrated yet.",
        scope="maple",
        record_id="helix-migration-status",
        created_at=1_110,
        importance=0.8,
    )
    preference = loom.remember_preference(
        "I prefer concise summaries with source links and explicit unresolved questions.",
        scope="maple",
        record_id="summary-preference",
        created_at=1_120,
        importance=0.95,
    )
    workspace_path = Path(workspace)
    notes = workspace_path / "projects" / "maple" / "notes.md"
    notes.parent.mkdir(parents=True, exist_ok=True)
    notes.write_text(
        "# Maple source notes\n\nMaple uses PostgreSQL 16. Helix has not been migrated yet.\n",
        encoding="utf-8",
    )
    runtime = BoundedFileRuntime(workspace_path, loom)
    read_result = runtime.execute(
        "read-file", {"path": "projects/maple/notes.md"}, scope="maple", now=1_130
    )
    if not read_result.receipt.verified:
        raise RuntimeError(read_result.receipt.error)
    read = read_result.memory_record
    procedure = loom.remember_procedure(
        "To update a project summary: read the source, write with the expected current hash, "
        "then reread and compare the returned hash before reporting completion.",
        trajectory_ids=(read.id,),
        reliability=0.9,
        verified=True,
        scope="maple",
        record_id="procedure-summary-readback",
        created_at=1_140,
        importance=0.8,
    )
    for number in range(8):
        loom.remember_user_statement(
            f"Routine conversational aside number {number}.",
            scope="maple",
            record_id=f"aside-{number}",
            created_at=1_200 + number,
            importance=0.1,
        )
    loom.begin_task(
        "maple-summary",
        "Inspect Maple's current database notes, then create and verify a concise summary.",
        scope="maple",
        stage="inspect",
        focus_stack=["establish current state", "read source", "write summary", "verify saved bytes"],
        open_questions=["Has Helix migrated too, or only Maple?"],
        constraints=["Preserve historical versions", "Do not claim a write without read-back"],
        completion_conditions=["summary exists", "read-back hash matches", "claims cite source records"],
        now=1_250,
    )
    ids = {
        "old": old.id,
        "correction": correction.id,
        "helix": helix.id,
        "preference": preference.id,
        "read": read.id,
        "procedure": procedure.id,
    }
    return loom, runtime, ids


def _run_demo(database: str | Path, workspace: str | Path) -> dict:
    loom, runtime, ids = build_demo(database, workspace)
    try:
        inspect = loom.compiler.compile(
            "Inspect the evidence: what database does Maple use now, and has Helix migrated?",
            scope="maple",
            task_id="maple-summary",
            now=1_300,
        )
        read = runtime.execute(
            "read-file",
            {"path": "projects/maple/notes.md"},
            task_id="maple-summary", scope="maple", next_stage="act", now=1_310,
        )
        action = loom.compiler.compile(
            "Create the concise summary now, preserving the correction and the Helix exception.",
            scope="maple",
            task_id="maple-summary",
            now=1_320,
        )
        written = runtime.execute(
            "write-file",
            {
                "path": "projects/maple/summary.md",
                "content": (
                    "# Maple summary\n\nMaple uses PostgreSQL 16. Helix has not been migrated yet.\n"
                ),
                "expected_sha256": None,
            },
            task_id="maple-summary", scope="maple", next_stage="verify", now=1_330,
        )
        verify = loom.compiler.compile(
            "Verify the saved Maple summary by rereading it before reporting completion.",
            scope="maple", task_id="maple-summary", now=1_340,
        )
        reread = runtime.execute(
            "read-file",
            {"path": "projects/maple/summary.md"},
            task_id="maple-summary", scope="maple", next_stage="report", now=1_350,
        )
        loom.update_task(
            "maple-summary", status="ready_to_report", open_questions=[], now=1_351
        )
        report = loom.compiler.compile(
            "Report the verified Maple summary and cite the current evidence.",
            scope="maple", task_id="maple-summary", now=1_360,
        )
        packets = {"inspect": inspect, "act": action, "verify": verify, "report": report}
        receipts = {"read_source": read.receipt, "write_summary": written.receipt, "reread_summary": reread.receipt}
        return {
            "canonical_ids": ids,
            "packets": {name: packet.to_dict() for name, packet in packets.items()},
            "receipts": {
                name: {
                    "id": receipt.id,
                    "capability_id": receipt.capability_id,
                    "verified": receipt.verified,
                    "verifier": receipt.verifier,
                    "result": receipt.result,
                    "error": receipt.error,
                }
                for name, receipt in receipts.items()
            },
            "dynamic_change": {
                "capabilities_by_stage": {
                    name: [capability.id for capability in packet.capabilities]
                    for name, packet in packets.items()
                },
                "context_ids": [packet.id for packet in packets.values()],
                "source_read_in_action": read.memory_record.id in {
                    source_id for entry in action.manifest if entry.admitted
                    for source_id in entry.source_record_ids
                },
                "write_receipt_in_verify": written.memory_record.id in {
                    source_id for entry in verify.manifest if entry.admitted
                    for source_id in entry.source_record_ids
                },
                "reread_receipt_in_report": reread.memory_record.id in {
                    source_id
                    for entry in report.manifest
                    if entry.admitted
                    for source_id in entry.source_record_ids
                },
            },
            "final_task": {
                "stage": loom.store.get_task("maple-summary").stage,
                "status": loom.store.get_task("maple-summary").status,
                "last_observation_id": loom.store.get_task("maple-summary").last_observation_id,
            },
            "integrity": loom.store.integrity_check(),
        }
    finally:
        loom.close()


def run_demo(
    database: str | Path = ":memory:", workspace: str | Path | None = None
) -> dict:
    if workspace is not None:
        return _run_demo(database, workspace)
    with tempfile.TemporaryDirectory(prefix="context-loom-demo-") as temporary:
        return _run_demo(database, temporary)


def render_demo_markdown(result: dict) -> str:
    lines = [
        "# Context Loom demonstration", "", f"Database integrity: `{result['integrity']}`", "",
        "All file receipts verified: `"
        + str(all(receipt["verified"] for receipt in result["receipts"].values())) + "`", "",
    ]
    for name in ("inspect", "act", "verify", "report"):
        packet = result["packets"][name]
        tools = result["dynamic_change"]["capabilities_by_stage"][name]
        lines.extend(
            [
                f"## {name.title()} pulse", "",
                f"Context: `{packet['id']}`; tokens: {packet['token_count']}/{packet['max_input_tokens']}; "
                f"capabilities: {', '.join(tools)}",
                "", "```text", packet["rendered"], "```", "",
            ]
        )
    return "\n".join(lines)
