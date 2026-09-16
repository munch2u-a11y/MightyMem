from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shlex
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .agent import AgentTurn, HatchedAgent, PreparedAgentTurn
from .gestation import gestate, load_profile
from .models import ChatMessage, ModelUnavailableError, OllamaChatModel
from .pipeline import BaseAgenticMemoryRAG
from .tools import ToolDefinition, ToolReceipt, ToolRegistry
from .types import EventKind, OutputTrunk, RecordType


DEFAULT_DATABASE = Path("habitus-functional.sqlite")
DEFAULT_MODEL = "qwen3.5:2b"
HELP_TEXT = (
    "Talk normally. Say 'remember that ...' to store a durable fact, or use "
    "/recall QUERY, /open PATH, /run PATH, /state, and /quit. Quote paths "
    "that contain spaces."
)


@dataclass(frozen=True)
class FunctionalTurn:
    kind: str
    response: str
    recalled_record_ids: tuple[str, ...] = ()
    speech_cycle_id: str | None = None
    tool_receipt: ToolReceipt | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "response": self.response,
            "recalled_record_ids": list(self.recalled_record_ids),
            "speech_cycle_id": self.speech_cycle_id,
            "tool_receipt": (
                self.tool_receipt.to_dict() if self.tool_receipt is not None else None
            ),
        }


class FunctionalHatchedAgent(HatchedAgent):
    """Conversation whose prompt also carries explicit durable facts."""

    def _durable_fact_text(self) -> str:
        rows = [
            record
            for record in self.mind.store.list_active_records()
            if record.record_type == RecordType.FACT
            and record.metadata.get("functional_fact") is True
        ]
        material: list[str] = []
        for record in reversed(rows):
            line = f"- {record.text.strip()}"
            if not line:
                continue
            material.append(line)
        if not material:
            return ""
        material.reverse()
        return (
            "Durable facts the user explicitly asked me to remember:\n"
            + "\n".join(material)
        )

    def _model_messages(
        self,
        text: str,
        recall,
        *,
        current_record_id: str,
    ) -> list[ChatMessage]:
        messages = super()._model_messages(
            text,
            recall,
            current_record_id=current_record_id,
        )
        directness = (
            "Answer the user's latest message directly. Treat memories as quiet "
            "background, not prose to recite. Do not repeat identity or taste "
            "memories unless they answer the question. Address the user as 'you', "
            "and follow requested brevity or output formats."
        )
        system = dict(messages[0])
        system["content"] = f"{directness}\n{system['content']}"
        messages[0] = system
        facts = self._durable_fact_text()
        if facts:
            system = dict(messages[0])
            system["content"] = f"{system['content']}\n{facts}"
            messages[0] = system
        return messages


class WorkspacePolicy:
    """Explicit, root-confined path inspection and Python execution."""

    def __init__(
        self,
        root: str | Path,
        *,
        maximum_read_bytes: int | None = None,
        run_timeout_seconds: float | None = None,
        run_memory_bytes: int | None = None,
        maximum_output_chars: int | None = None,
        maximum_directory_entries: int | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise NotADirectoryError(f"workspace is not a directory: {self.root}")
        self.maximum_read_bytes = None if maximum_read_bytes is None else int(maximum_read_bytes)
        self.run_timeout_seconds = None if run_timeout_seconds is None else float(run_timeout_seconds)
        self.run_memory_bytes = None if run_memory_bytes is None else int(run_memory_bytes)
        self.maximum_output_chars = None if maximum_output_chars is None else int(maximum_output_chars)
        self.maximum_directory_entries = None if maximum_directory_entries is None else int(maximum_directory_entries)
        explicit_limits = {
            "maximum_read_bytes": self.maximum_read_bytes,
            "run_timeout_seconds": self.run_timeout_seconds,
            "run_memory_bytes": self.run_memory_bytes,
            "maximum_output_chars": self.maximum_output_chars,
            "maximum_directory_entries": self.maximum_directory_entries,
        }
        for name, value in explicit_limits.items():
            if value is not None and value <= 0:
                raise ValueError(f"an explicit {name} must be positive")

    def resolve_path(self, supplied: str) -> Path:
        raw = str(supplied).strip()
        if not raw:
            raise ValueError("a workspace-relative path is required")
        requested = Path(raw)
        candidate = requested if requested.is_absolute() else self.root / requested
        resolved = candidate.resolve(strict=True)
        try:
            resolved.relative_to(self.root)
        except ValueError as error:
            raise PermissionError(
                f"path is outside the authorized workspace: {raw}"
            ) from error
        return resolved

    def resolve_file(self, supplied: str) -> Path:
        resolved = self.resolve_path(supplied)
        if not resolved.is_file():
            raise FileNotFoundError(f"not a regular file: {supplied}")
        return resolved

    def read_file(self, supplied: str) -> dict[str, Any]:
        path = self.resolve_file(supplied)
        size = path.stat().st_size
        if self.maximum_read_bytes is not None and size > self.maximum_read_bytes:
            raise ValueError(
                f"file is {size} bytes; limit is {self.maximum_read_bytes} bytes"
            )
        payload = path.read_bytes()
        try:
            content = payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("file is not valid UTF-8 text") from error
        return {
            "path": str(path.relative_to(self.root)),
            "kind": "file",
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "content": content,
        }

    def inspect_path(self, supplied: str) -> dict[str, Any]:
        """Read a text file or list one directory without leaving the root."""
        path = self.resolve_path(supplied)
        if path.is_file():
            return self.read_file(supplied)
        if not path.is_dir():
            raise FileNotFoundError(f"not a regular file or directory: {supplied}")

        entries: list[dict[str, Any]] = []
        children = sorted(path.iterdir(), key=lambda item: item.name.casefold())
        visible = children if self.maximum_directory_entries is None else children[: self.maximum_directory_entries]
        for child in visible:
            if child.is_symlink():
                kind = "symlink"
                size = None
            elif child.is_dir():
                kind = "directory"
                size = None
            elif child.is_file():
                kind = "file"
                try:
                    size = child.stat().st_size
                except OSError:
                    size = None
            else:
                kind = "other"
                size = None
            entries.append({"name": child.name, "kind": kind, "size_bytes": size})
        return {
            "path": str(path.relative_to(self.root)),
            "kind": "directory",
            "entry_count": len(children),
            "entries": entries,
            "truncated": len(children) > len(visible),
        }

    def _child_limits(self) -> None:
        if self.run_timeout_seconds is not None:
            cpu_seconds = max(1, int(math.ceil(self.run_timeout_seconds)))
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
        if self.run_memory_bytes is not None:
            resource.setrlimit(
                resource.RLIMIT_AS,
                (self.run_memory_bytes, self.run_memory_bytes),
            )

    def run_python(self, supplied: str) -> dict[str, Any]:
        path = self.resolve_file(supplied)
        if path.suffix.casefold() != ".py":
            raise ValueError("only explicit .py files can be run")
        if self.maximum_read_bytes is not None and path.stat().st_size > self.maximum_read_bytes:
            raise ValueError("Python file exceeds the configured size limit")
        environment = {
            "PATH": os.environ.get("PATH", os.defpath),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONIOENCODING": "utf-8",
        }
        try:
            completed = subprocess.run(
                [sys.executable, "-I", str(path)],
                cwd=self.root,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.run_timeout_seconds,
                check=False,
                start_new_session=True,
                preexec_fn=self._child_limits,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                f"Python file exceeded {self.run_timeout_seconds:g}s timeout"
            ) from error
        stdout = completed.stdout if self.maximum_output_chars is None else completed.stdout[-self.maximum_output_chars :]
        stderr = completed.stderr if self.maximum_output_chars is None else completed.stderr[-self.maximum_output_chars :]
        result = {
            "path": str(path.relative_to(self.root)),
            "returncode": int(completed.returncode),
            "stdout": stdout,
            "stderr": stderr,
            "output_truncated": (
                len(completed.stdout) > len(stdout) or len(completed.stderr) > len(stderr)
            ),
        }
        if completed.returncode != 0:
            detail = stderr.strip() or stdout.strip() or "no process output"
            raise RuntimeError(
                f"Python file exited {completed.returncode}: {detail}"
            )
        return result


def _tool_handler(method, argument_name: str):
    def handler(arguments: dict[str, Any]) -> dict[str, Any]:
        return method(str(arguments.get(argument_name, "")))

    return handler


class FunctionalAgent:
    """One persistent conversation, recall, and workspace-action surface."""

    def __init__(
        self,
        mind: BaseAgenticMemoryRAG,
        model,
        *,
        workspace: str | Path,
        history_messages: int | None = None,
        run_timeout_seconds: float | None = None,
    ) -> None:
        self.mind = mind
        self.chat = FunctionalHatchedAgent(
            mind,
            model,
            history_messages=history_messages,
        )
        self.workspace = WorkspacePolicy(
            workspace,
            run_timeout_seconds=run_timeout_seconds,
        )
        self.tools = ToolRegistry(mind)
        self.tools.register_tool(
            ToolDefinition(
                tool_id="tool:workspace_read",
                trunk=OutputTrunk.LOOK,
                label="Inspect Workspace Path",
                description=(
                    "List one authorized workspace folder or read one UTF-8 file."
                ),
                terms=("open", "read", "inspect", "folder", "file", "workspace"),
                parameters={"path": {"type": "string"}},
                handler=_tool_handler(self.workspace.inspect_path, "path"),
            )
        )
        self.tools.register_tool(
            ToolDefinition(
                tool_id="tool:workspace_run_python",
                trunk=OutputTrunk.DO,
                label="Run Workspace Python File",
                description="Run one explicitly authorized Python file and return its complete observed result.",
                terms=("run", "execute", "python", "file", "workspace"),
                parameters={"path": {"type": "string"}},
                handler=_tool_handler(self.workspace.run_python, "path"),
            )
        )

    @staticmethod
    def _split_command(text: str) -> tuple[str, str] | None:
        stripped = str(text).strip()
        lowered = stripped.casefold()
        for command in (
            "/open",
            "/run",
            "/remember",
            "/recall",
            "/state",
            "/help",
        ):
            if lowered == command:
                return command, ""
            prefix = command + " "
            if lowered.startswith(prefix):
                return command, stripped[len(prefix) :].strip()
        natural = "remember that "
        if lowered.startswith(natural):
            return "/remember", stripped[len(natural) :].strip()
        return None

    @staticmethod
    def _one_path(argument: str) -> str:
        try:
            pieces = shlex.split(argument)
        except ValueError as error:
            raise ValueError(f"invalid quoted path: {error}") from error
        if len(pieces) != 1:
            raise ValueError("supply exactly one workspace path; quote paths containing spaces")
        return pieces[0]

    def _finish_speech(
        self,
        prepared: PreparedAgentTurn,
        response: str,
    ) -> AgentTurn:
        turn = self.chat.complete_prepared_turn(prepared, response)
        self.chat.acknowledge_delivery(turn, channel="terminal")
        return turn

    def _remember_fact(self, text: str) -> str:
        fact = str(text).strip()
        if not fact:
            raise ValueError("tell me what to remember")
        existing = next(
            (
                record
                for record in self.mind.store.list_active_records()
                if record.record_type == RecordType.FACT
                and record.metadata.get("functional_fact") is True
                and record.text.casefold() == fact.casefold()
            ),
            None,
        )
        if existing is None:
            self.mind.remember(
                fact,
                kind=EventKind.MESSAGE,
                source_id=self.chat.profile.human_name,
                record_type=RecordType.FACT,
                metadata={
                    "functional_fact": True,
                    "fact_sha256": hashlib.sha256(fact.encode("utf-8")).hexdigest(),
                    "explicit_user_memory": True,
                },
            )
        return f"I’ll remember: {fact}"

    def _recall_response(self, query: str, *, exclude: Sequence[str]) -> tuple[str, tuple[str, ...]]:
        prompt = str(query).strip()
        if not prompt:
            raise ValueError("tell me what to recall")
        recalled = self.mind.recall(
            prompt,
            source_id=self.chat.profile.human_name,
            exclude_record_ids=tuple(exclude),
            include_current_input=False,
        )
        records = []
        seen: set[str] = set()
        for record in self.mind.store.list_active_records():
            if (
                record.record_type == RecordType.FACT
                and record.metadata.get("functional_fact") is True
                and record.record_id not in seen
            ):
                records.append(record)
                seen.add(record.record_id)
        for hit in recalled.hits:
            if hit.record.record_id not in seen:
                records.append(hit.record)
                seen.add(hit.record.record_id)
        if not records:
            return "I don’t have a matching memory yet.", ()
        selected = records[-12:]
        body = "\n".join(f"- {record.text}" for record in selected)
        return f"I found these memories:\n{body}", tuple(
            record.record_id for record in selected
        )

    @staticmethod
    def _tool_response(receipt: ToolReceipt) -> str:
        if receipt.status != "success":
            return f"I could not complete {receipt.tool_id}: {receipt.error}"
        output = receipt.output if isinstance(receipt.output, Mapping) else {}
        if receipt.tool_id == "tool:workspace_read":
            if output.get("kind") == "directory":
                rendered_entries = []
                for entry in output.get("entries", ()):
                    marker = "/" if entry.get("kind") == "directory" else ""
                    rendered_entries.append(
                        f"[{entry.get('kind', 'other')}] {entry.get('name', '')}{marker}"
                    )
                if output.get("truncated"):
                    rendered_entries.append("[listing truncated]")
                body = "\n".join(rendered_entries) or "[empty folder]"
                entry_count = int(output.get("entry_count", 0))
                entry_word = "entry" if entry_count == 1 else "entries"
                return (
                    f"Opened folder {output.get('path')} "
                    f"({entry_count} {entry_word}).\n{body}"
                )
            content = str(output.get("content", ""))
            return (
                f"Opened {output.get('path')} "
                f"({output.get('size_bytes')} bytes, sha256 {output.get('sha256')}).\n"
                f"{content}"
            ).rstrip()
        stdout = str(output.get("stdout", "")).strip()
        stderr = str(output.get("stderr", "")).strip()
        details = stdout or stderr or "[no output]"
        return (
            f"Ran {output.get('path')} successfully "
            f"(exit {output.get('returncode')}).\n{details}"
        )

    def _tool_turn(
        self,
        prepared: PreparedAgentTurn,
        tool_id: str,
        path: str,
    ) -> FunctionalTurn:
        receipt = self.tools.execute(tool_id, {"path": path})
        response = self._tool_response(receipt)
        speech = self._finish_speech(prepared, response)
        return FunctionalTurn(
            kind="tool",
            response=response,
            recalled_record_ids=prepared.recall.context_bundle.record_ids,
            speech_cycle_id=speech.experience_id,
            tool_receipt=receipt,
        )

    def handle(self, text: str) -> FunctionalTurn:
        content = str(text).strip()
        if not content:
            raise ValueError("message cannot be empty")
        command = self._split_command(content)
        if command is None:
            turn = self.chat.turn(content)
            self.chat.acknowledge_delivery(turn, channel="terminal")
            return FunctionalTurn(
                kind="conversation",
                response=turn.response,
                recalled_record_ids=turn.recall.context_bundle.record_ids,
                speech_cycle_id=turn.experience_id,
            )

        name, argument = command
        prepared = self.chat.prepare_turn(content)
        if name == "/open":
            return self._tool_turn(
                prepared,
                "tool:workspace_read",
                self._one_path(argument),
            )
        if name == "/run":
            return self._tool_turn(
                prepared,
                "tool:workspace_run_python",
                self._one_path(argument),
            )
        if name == "/remember":
            response = self._remember_fact(argument)
            speech = self._finish_speech(prepared, response)
            return FunctionalTurn(
                kind="memory_write",
                response=response,
                recalled_record_ids=prepared.recall.context_bundle.record_ids,
                speech_cycle_id=speech.experience_id,
            )
        if name == "/state":
            response = json.dumps(self.state(), indent=2, sort_keys=True)
            speech = self._finish_speech(prepared, response)
            return FunctionalTurn(
                kind="state",
                response=response,
                recalled_record_ids=prepared.recall.context_bundle.record_ids,
                speech_cycle_id=speech.experience_id,
            )
        if name == "/help":
            speech = self._finish_speech(prepared, HELP_TEXT)
            return FunctionalTurn(
                kind="help",
                response=HELP_TEXT,
                recalled_record_ids=prepared.recall.context_bundle.record_ids,
                speech_cycle_id=speech.experience_id,
            )
        response, record_ids = self._recall_response(
            argument,
            exclude=(prepared.inbound.record_id,),
        )
        speech = self._finish_speech(prepared, response)
        return FunctionalTurn(
            kind="recall",
            response=response,
            recalled_record_ids=record_ids,
            speech_cycle_id=speech.experience_id,
        )

    def state(self) -> dict[str, Any]:
        store = self.mind.store
        facts = [
            record.text
            for record in store.list_active_records()
            if record.record_type == RecordType.FACT
            and record.metadata.get("functional_fact") is True
        ]
        tool_rows = store.connection.execute(
            "SELECT COUNT(*) FROM records WHERE record_type = ?",
            (RecordType.TOOL_RESULT.value,),
        ).fetchone()
        return {
            "agent": self.chat.profile.agent_name,
            "human": self.chat.profile.human_name,
            "database_records": len(store.list_records()),
            "concepts": len(store.list_concepts()),
            "edges": len(store.list_edges()),
            "durable_facts": facts,
            "verified_tool_returns": int(tool_rows[0]),
            "open_cycles": len(self.mind.open_experience_cycles()),
            "graph_invariant_errors": self.mind.graph.validate_invariants(),
            "workspace": str(self.workspace.root),
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one persistent Habitus conversation, recall, and file-action loop."
    )
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--human-name", default="Human")
    parser.add_argument("--agent-name", default="Habitus")
    parser.add_argument("--taste", default="builder")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--history-messages", type=int, default=None)
    parser.add_argument("--run-timeout", type=float, default=None)
    parser.add_argument(
        "--allow-gpu",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--cpu-only", action="store_true", help="explicitly force local inference to CPU")
    parser.add_argument("--once", help="process one message and exit")
    parser.add_argument("--json", action="store_true")
    return parser


def _show(turn: FunctionalTurn, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(turn.to_dict(), indent=2, sort_keys=True, default=str))
    else:
        print(turn.response)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.database.parent.mkdir(parents=True, exist_ok=True)
    with BaseAgenticMemoryRAG(args.database) as mind:
        profile = load_profile(mind)
        if profile is None:
            profile = gestate(
                mind,
                human_name=args.human_name,
                agent_name=args.agent_name,
                taste_schema=args.taste,
                model_backend="ollama",
                model_name=args.model,
            )
        model = OllamaChatModel(
            args.model or profile.model_name,
            base_url=args.ollama_url,
            timeout_seconds=args.timeout,
            temperature=0.35,
            num_gpu=0 if args.cpu_only else None,
        )
        agent = FunctionalAgent(
            mind,
            model,
            workspace=args.workspace,
            history_messages=args.history_messages,
            run_timeout_seconds=args.run_timeout,
        )
        if args.once is not None:
            try:
                _show(agent.handle(args.once), as_json=args.json)
            except (ModelUnavailableError, OSError, ValueError) as error:
                print(f"Habitus error: {error}", file=sys.stderr)
                return 2
            return 0

        print(
            f"{profile.agent_name} is awake. Persistent mind: {args.database}\n"
            f"Authorized workspace: {agent.workspace.root}\n"
            "Commands: /remember TEXT, /recall QUERY, /open PATH, /run PATH, "
            "/state, /help, /quit"
        )
        while True:
            try:
                message = input(f"{profile.human_name}> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if not message:
                continue
            lowered = message.casefold()
            if lowered in {"/quit", "/exit"}:
                return 0
            try:
                turn = agent.handle(message)
            except (ModelUnavailableError, OSError, ValueError) as error:
                print(f"Habitus error: {error}")
                continue
            print(f"{profile.agent_name}> {turn.response}")


if __name__ == "__main__":
    raise SystemExit(main())
