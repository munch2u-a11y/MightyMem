from __future__ import annotations

import hashlib
import sys

from context_loom import BoundedFileRuntime, EvidenceStatus

from conftest import admitted_ids


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def test_write_is_atomic_read_back_verified_and_persisted(loom, tmp_path) -> None:
    root = tmp_path / "tools"
    runtime = BoundedFileRuntime(root, loom)
    result = runtime.execute(
        "write-file",
        {"path": "report.md", "content": "verified bytes\n", "expected_sha256": None},
        scope="alpha",
        now=10,
    )
    assert result.receipt.success and result.receipt.verified
    assert result.receipt.result["readback_matches"] is True
    assert result.receipt.result["sha256"] == digest("verified bytes\n")
    assert (root / "report.md").read_text() == "verified bytes\n"
    assert result.memory_record.evidence_status == EvidenceStatus.TOOL_VERIFIED
    assert result.memory_record.metadata["receipt"]["id"] == result.receipt.id


def test_stale_write_is_rejected_and_original_bytes_survive(loom, tmp_path) -> None:
    root = tmp_path / "tools"
    root.mkdir()
    (root / "report.md").write_text("original")
    runtime = BoundedFileRuntime(root, loom)
    result = runtime.execute(
        "write-file",
        {"path": "report.md", "content": "replacement", "expected_sha256": "wrong"},
        scope="alpha",
        now=10,
    )
    assert not result.receipt.success and not result.receipt.verified
    assert "stale write protection" in result.receipt.error
    assert (root / "report.md").read_text() == "original"
    assert result.memory_record.evidence_status == EvidenceStatus.UNVERIFIED


def test_existing_file_requires_expected_current_hash(loom, tmp_path) -> None:
    root = tmp_path / "tools"
    root.mkdir()
    (root / "report.md").write_text("v1")
    runtime = BoundedFileRuntime(root, loom)
    denied = runtime.execute(
        "write-file", {"path": "report.md", "content": "v2", "expected_sha256": None},
        scope="alpha", now=10,
    )
    allowed = runtime.execute(
        "write-file",
        {"path": "report.md", "content": "v2", "expected_sha256": digest("v1")},
        scope="alpha", now=11,
    )
    assert not denied.receipt.success
    assert allowed.receipt.success and (root / "report.md").read_text() == "v2"


def test_read_receipt_contains_exact_content_and_hash(loom, tmp_path) -> None:
    root = tmp_path / "tools"
    root.mkdir()
    (root / "notes.txt").write_text("hello Ω")
    result = BoundedFileRuntime(root, loom).execute(
        "read-file", {"path": "notes.txt"}, scope="alpha", now=10
    )
    assert result.receipt.result["content"] == "hello Ω"
    assert result.receipt.result["sha256"] == digest("hello Ω")
    assert "hello Ω" in result.memory_record.content


def test_list_receipt_is_sorted_and_hashed(loom, tmp_path) -> None:
    root = tmp_path / "tools"
    root.mkdir()
    (root / "z.txt").write_text("z")
    (root / "a").mkdir()
    result = BoundedFileRuntime(root, loom).execute(
        "list-files", {"path": "."}, scope="alpha", now=10
    )
    assert result.receipt.result["entries"] == ["a/", "z.txt"]
    assert len(result.receipt.result["entries_sha256"]) == 64


def test_path_traversal_and_unsupported_tools_fail_closed(loom, tmp_path) -> None:
    runtime = BoundedFileRuntime(tmp_path / "tools", loom)
    traversal = runtime.execute("read-file", {"path": "../secret"}, scope="alpha", now=10)
    unsupported = runtime.execute("run-python", {"path": "script.py"}, scope="alpha", now=11)
    assert not traversal.receipt.success
    assert "escapes" in traversal.receipt.error
    assert not unsupported.receipt.success
    assert "no bounded runtime adapter" in unsupported.receipt.error
    packet = loom.compiler.compile("Run the Python script.", scope="alpha", now=12)
    assert "run-python" not in {capability.id for capability in packet.capabilities}


def test_registered_program_runs_without_a_shell_and_records_output(loom, tmp_path) -> None:
    root = tmp_path / "tools"
    root.mkdir()
    (root / "script.py").write_text("import sys\nprint('hello', sys.argv[1])\n")
    runtime = BoundedFileRuntime(root, loom, allowed_programs={"python": sys.executable})
    result = runtime.execute(
        "run-program",
        {
            "program": "python",
            "arguments": ["script.py", "loom"],
            "cwd": ".",
            "wait": True,
            "timeout_seconds": 5,
        },
        scope="alpha",
        now=10,
    )
    assert result.receipt.success and result.receipt.verified
    assert result.receipt.result["exit_code"] == 0
    assert result.receipt.result["stdout"] == "hello loom\n"
    assert len(result.receipt.result["stdout_sha256"]) == 64
    assert result.receipt.verifier == "process_exit_status"


def test_run_rejects_unregistered_global_program_and_preserves_failed_output(loom, tmp_path) -> None:
    root = tmp_path / "tools"
    root.mkdir()
    (root / "fail.py").write_text("import sys\nprint('diagnostic')\nsys.exit(7)\n")
    runtime = BoundedFileRuntime(root, loom, allowed_programs={"python": sys.executable})
    unregistered = runtime.execute(
        "run-program",
        {"program": "/bin/echo", "arguments": [], "cwd": ".", "wait": True, "timeout_seconds": 5},
        scope="alpha",
        now=10,
    )
    failed = runtime.execute(
        "run-program",
        {"program": "python", "arguments": ["fail.py"], "cwd": ".", "wait": True, "timeout_seconds": 5},
        scope="alpha",
        now=11,
    )
    assert not unregistered.receipt.success and "escapes" in unregistered.receipt.error
    assert not failed.receipt.success and not failed.receipt.verified
    assert failed.receipt.result["exit_code"] == 7
    assert failed.receipt.result["stdout"] == "diagnostic\n"
    assert failed.receipt.verifier == "process_exit_status"


def test_run_capability_appears_for_execution_intent(loom) -> None:
    loom.begin_task("run-task", "Run the project script.", scope="alpha", stage="act", now=1)
    packet = loom.compiler.compile("Launch and run the app.", scope="alpha", task_id="run-task", now=2)
    assert "run-program" in {capability.id for capability in packet.capabilities}


def test_detached_start_is_observed_but_does_not_claim_task_completion(loom, tmp_path) -> None:
    root = tmp_path / "tools"
    root.mkdir()
    (root / "pause.py").write_text("import time\ntime.sleep(0.2)\n")
    loom.begin_task("launch-task", "Launch the app.", scope="alpha", stage="act", now=1)
    runtime = BoundedFileRuntime(root, loom, allowed_programs={"python": sys.executable})
    result = runtime.execute(
        "run-program",
        {
            "program": "python",
            "arguments": ["pause.py"],
            "cwd": ".",
            "wait": False,
            "timeout_seconds": 5,
        },
        scope="alpha",
        task_id="launch-task",
        next_stage="complete",
        now=10,
    )
    assert result.receipt.success and result.receipt.verified
    assert result.receipt.result["state"] == "running"
    assert result.receipt.result["completion_verified"] is False
    assert loom.store.get_task("launch-task").stage == "act"


def test_only_verified_runtime_result_advances_task(loom, tmp_path) -> None:
    root = tmp_path / "tools"
    root.mkdir()
    (root / "notes.txt").write_text("source")
    loom.begin_task("task", "Read notes then write.", scope="alpha", stage="inspect", now=1)
    runtime = BoundedFileRuntime(root, loom)
    failed = runtime.execute(
        "read-file", {"path": "missing.txt"}, scope="alpha", task_id="task",
        next_stage="act", now=10,
    )
    assert not failed.receipt.success and loom.store.get_task("task").stage == "inspect"
    success = runtime.execute(
        "read-file", {"path": "notes.txt"}, scope="alpha", task_id="task",
        next_stage="act", now=11,
    )
    assert success.receipt.success and loom.store.get_task("task").stage == "act"
    packet = loom.compiler.compile("Continue", scope="alpha", task_id="task", now=12)
    assert success.memory_record.id in admitted_ids(packet)
