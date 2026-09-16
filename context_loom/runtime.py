from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from .engine import ContextLoom
from .models import MemoryRecord, ToolReceipt


class ToolExecutionError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        verifier: str = "runtime_rejection",
    ) -> None:
        super().__init__(message)
        self.details = details or {}
        self.verifier = verifier


@dataclass(slots=True)
class RuntimeResult:
    receipt: ToolReceipt
    memory_record: MemoryRecord


class BoundedFileRuntime:
    """Receipt-producing adapter for root-confined file and process tools.

    Capability visibility and execution authority are deliberately separate.
    Paths and working directories stay inside one resolved root.  Executables
    outside that root must be registered explicitly by the host application.
    Commands are argv arrays and never pass through a shell.
    """

    def __init__(
        self,
        root: str | Path,
        loom: ContextLoom,
        *,
        max_read_bytes: int | None = None,
        max_process_output_bytes: int | None = None,
        allowed_programs: dict[str, str | Path] | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.loom = loom
        self.max_read_bytes = max_read_bytes
        self.max_process_output_bytes = max_process_output_bytes
        self.allowed_programs = {
            name: Path(path).expanduser().resolve()
            for name, path in (allowed_programs or {}).items()
        }
        self._handlers: dict[str, Callable[[dict[str, Any]], tuple[str, dict[str, Any], str]]] = {
            "list-files": self._list_files,
            "read-file": self._read_file,
            "write-file": self._write_file,
            "run-program": self._run_program,
        }

    @staticmethod
    def _sha256(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def _path(self, supplied: Any) -> Path:
        if not isinstance(supplied, str) or not supplied.strip():
            raise ToolExecutionError("path must be a non-empty string")
        candidate = (self.root / supplied).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise ToolExecutionError("path escapes the bounded runtime root") from exc
        return candidate

    def _relative(self, path: Path) -> str:
        value = str(path.relative_to(self.root))
        return value if value else "."

    def _list_files(self, arguments: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
        directory = self._path(arguments.get("path"))
        if not directory.is_dir():
            raise ToolExecutionError("list target is not a directory")
        entries = sorted(item.name + ("/" if item.is_dir() else "") for item in directory.iterdir())
        encoded = json.dumps(entries, ensure_ascii=False).encode()
        relative = self._relative(directory)
        details = {"path": relative, "entries": entries, "entries_sha256": self._sha256(encoded)}
        return f"Listed {relative}: " + ", ".join(entries), details, "returned_directory_entries"

    def _read_file(self, arguments: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
        path = self._path(arguments.get("path"))
        if not path.is_file():
            raise ToolExecutionError("read target is not a regular file")
        data = path.read_bytes()
        if self.max_read_bytes is not None and len(data) > self.max_read_bytes:
            raise ToolExecutionError(f"file exceeds the {self.max_read_bytes}-byte read bound")
        try:
            content = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ToolExecutionError("prototype reader accepts UTF-8 text only") from exc
        relative = self._relative(path)
        digest = self._sha256(data)
        details = {"path": relative, "bytes": len(data), "sha256": digest, "content": content}
        return f"Read {relative}; bytes={len(data)}; sha256={digest}.\nContent:\n{content}", details, "content_and_sha256"

    def _write_file(self, arguments: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
        path = self._path(arguments.get("path"))
        content = arguments.get("content")
        expected = arguments.get("expected_sha256")
        if not isinstance(content, str):
            raise ToolExecutionError("content must be a string")
        if expected is not None and not isinstance(expected, str):
            raise ToolExecutionError("expected_sha256 must be a string or null")
        if not path.parent.is_dir():
            raise ToolExecutionError("parent directory does not exist")
        before = self._sha256(path.read_bytes()) if path.is_file() else None
        if before != expected:
            raise ToolExecutionError(
                f"stale write protection: expected {expected!r}, observed {before!r}"
            )
        encoded = content.encode("utf-8")
        handle, temporary_name = tempfile.mkstemp(prefix=".context-loom-", dir=path.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(handle, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        observed = path.read_bytes()
        after = self._sha256(observed)
        expected_after = self._sha256(encoded)
        if observed != encoded or after != expected_after:
            raise ToolExecutionError("post-write read-back did not match requested bytes")
        relative = self._relative(path)
        details = {
            "path": relative,
            "bytes": len(observed),
            "previous_sha256": before,
            "sha256": after,
            "readback_matches": True,
        }
        return f"Wrote and reread {relative}; bytes={len(observed)}; sha256={after}.", details, "write_receipt_then_readback"

    def _executable(self, supplied: Any) -> tuple[str, Path]:
        if not isinstance(supplied, str) or not supplied.strip():
            raise ToolExecutionError("program must be a non-empty string")
        if supplied in self.allowed_programs:
            path = self.allowed_programs[supplied]
            label = supplied
        else:
            path = self._path(supplied)
            label = self._relative(path)
        if not path.is_file():
            raise ToolExecutionError("program is not a regular file")
        if not os.access(path, os.X_OK):
            raise ToolExecutionError("program is not executable")
        return label, path

    @staticmethod
    def _argv(arguments: Any) -> list[str]:
        if arguments is None:
            return []
        if not isinstance(arguments, list) or any(not isinstance(value, str) for value in arguments):
            raise ToolExecutionError("arguments must be an array of strings")
        if any("\x00" in value for value in arguments):
            raise ToolExecutionError("arguments cannot contain NUL bytes")
        return list(arguments)

    def _process_bytes(self, value: bytes | None) -> tuple[str, str, int, bool]:
        data = value or b""
        digest = self._sha256(data)
        truncated = self.max_process_output_bytes is not None and len(data) > self.max_process_output_bytes
        visible = data if self.max_process_output_bytes is None else data[: self.max_process_output_bytes]
        return visible.decode("utf-8", errors="replace"), digest, len(data), truncated

    def _run_program(self, arguments: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
        label, program = self._executable(arguments.get("program"))
        argv = self._argv(arguments.get("arguments"))
        cwd = self._path(arguments.get("cwd", "."))
        if not cwd.is_dir():
            raise ToolExecutionError("process working directory is not a directory")
        wait = arguments.get("wait", True)
        if not isinstance(wait, bool):
            raise ToolExecutionError("wait must be a boolean")
        timeout = arguments.get("timeout_seconds")
        if timeout is not None:
            if not isinstance(timeout, (int, float)) or isinstance(timeout, bool):
                raise ToolExecutionError("timeout_seconds must be a number or null")
            timeout = float(timeout)
            if timeout <= 0:
                raise ToolExecutionError("timeout_seconds must be greater than zero")
        command = [str(program), *argv]
        request_details = {
            "program": label,
            "arguments": argv,
            "cwd": self._relative(cwd),
            "wait": wait,
            "timeout_seconds": timeout,
        }
        if not wait:
            try:
                process = subprocess.Popen(
                    command,
                    cwd=cwd,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
            except OSError as exc:
                raise ToolExecutionError(f"process launch failed: {exc}") from exc
            time.sleep(0.05)
            return_code = process.poll()
            details = {
                **request_details,
                "pid": process.pid,
                "state": "running" if return_code is None else "exited",
                "exit_code": return_code,
                "completion_verified": return_code == 0,
            }
            if return_code not in {None, 0}:
                raise ToolExecutionError(
                    f"process exited during launch with status {return_code}",
                    details=details,
                    verifier="process_start_observation",
                )
            return (
                f"Started {label}; pid={process.pid}; state={details['state']}. Completion is not yet verified.",
                details,
                "process_start_observation",
            )

        started = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                cwd=cwd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout, stdout_hash, stdout_bytes, stdout_truncated = self._process_bytes(exc.stdout)
            stderr, stderr_hash, stderr_bytes, stderr_truncated = self._process_bytes(exc.stderr)
            raise ToolExecutionError(
                f"process exceeded its explicitly requested {timeout:g}-second timeout",
                details={
                    **request_details,
                    "state": "timed_out",
                    "elapsed_seconds": time.perf_counter() - started,
                    "stdout": stdout,
                    "stderr": stderr,
                    "stdout_sha256": stdout_hash,
                    "stderr_sha256": stderr_hash,
                    "stdout_bytes": stdout_bytes,
                    "stderr_bytes": stderr_bytes,
                    "output_truncated": stdout_truncated or stderr_truncated,
                },
                verifier="process_exit_status",
            ) from exc
        stdout, stdout_hash, stdout_bytes, stdout_truncated = self._process_bytes(completed.stdout)
        stderr, stderr_hash, stderr_bytes, stderr_truncated = self._process_bytes(completed.stderr)
        details = {
            **request_details,
            "state": "exited",
            "exit_code": completed.returncode,
            "elapsed_seconds": time.perf_counter() - started,
            "stdout": stdout,
            "stderr": stderr,
            "stdout_sha256": stdout_hash,
            "stderr_sha256": stderr_hash,
            "stdout_bytes": stdout_bytes,
            "stderr_bytes": stderr_bytes,
            "output_truncated": stdout_truncated or stderr_truncated,
            "completion_verified": completed.returncode == 0,
        }
        if completed.returncode != 0:
            raise ToolExecutionError(
                f"process exited with status {completed.returncode}",
                details=details,
                verifier="process_exit_status",
            )
        return (
            f"Ran {label}; exit=0; stdout_sha256={stdout_hash}; stderr_sha256={stderr_hash}.",
            details,
            "process_exit_status",
        )

    def execute(
        self,
        capability_id: str,
        arguments: dict[str, Any],
        *,
        scope: str = "default",
        task_id: str | None = None,
        next_stage: str | None = None,
        now: float | None = None,
    ) -> RuntimeResult:
        started = time.time() if now is None else now
        receipt_id = "receipt_" + uuid.uuid4().hex
        handler = self._handlers.get(capability_id)
        success = False
        details: dict[str, Any] = {}
        verifier = "unsupported_capability"
        error: str | None = None
        if handler is None:
            message = f"Capability {capability_id!r} has no bounded runtime adapter."
            error = message
        else:
            try:
                message, details, verifier = handler(dict(arguments))
                success = True
            except (OSError, ToolExecutionError) as exc:
                if isinstance(exc, ToolExecutionError):
                    details = dict(exc.details)
                    verifier = exc.verifier
                error = str(exc)
                message = f"{capability_id} failed: {error}"
        completed = time.time() if now is None else now
        receipt = ToolReceipt(
            id=receipt_id,
            capability_id=capability_id,
            success=success,
            verified=success,
            verifier=verifier,
            started_at=started,
            completed_at=completed,
            request=dict(arguments),
            result=details,
            error=error,
        )
        completion_verified = details.get("completion_verified", success)
        memory = self.loom.record_tool_result(
            message,
            task_id=task_id,
            scope=scope,
            verified=receipt.verified,
            receipt_id=receipt.id,
            verifier=receipt.verifier,
            next_stage=next_stage if receipt.verified and completion_verified else None,
            metadata={"receipt": asdict(receipt)},
            now=completed,
        )
        return RuntimeResult(receipt=receipt, memory_record=memory)
