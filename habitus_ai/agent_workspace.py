"""A persistent place in a real, root-confined local workspace."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile

from .functional_agent import WorkspacePolicy


class AgentWorkspace(WorkspacePolicy):
    def __init__(self, root, store, **kwargs):
        super().__init__(root, **kwargs)
        self.store = store
        self.position = store.get_metadata("agent_workspace_position", ".")
        self._check(self.root / self.position, exists=True)

    def _check(self, path, *, exists):
        result = path.resolve(strict=exists)
        if not result.is_relative_to(self.root):
            raise PermissionError("path leaves the permitted virtual workspace")
        return result

    def resolve_path(self, supplied):
        raw = str(supplied).strip()
        if not raw:
            raise ValueError("a virtual workspace path is required")
        path = self.root / raw.lstrip("/") if raw.startswith("/") else self.root / self.position / raw
        return self._check(path, exists=True)

    def inspect(self, arguments):
        output = self.inspect_path(str(arguments.get("path", ".")))
        output["position"] = "/" + ("" if self.position == "." else self.position)
        return output

    def move(self, arguments):
        destination = self.resolve_path(arguments["path"])
        if not destination.is_dir():
            raise NotADirectoryError("movement requires a directory")
        before = self.position
        self.position = destination.relative_to(self.root).as_posix()
        self.store.set_metadata("agent_workspace_position", self.position)
        return {"previous_position": before, "position": self.position,
                "observed": self.inspect_path(".")}

    def _destination(self, supplied):
        raw = str(supplied).strip()
        if not raw:
            raise ValueError("a destination is required")
        path = self.root / raw.lstrip("/") if raw.startswith("/") else self.root / self.position / raw
        destination = self._check(path, exists=False)
        self._check(destination.parent, exists=True)
        return destination

    def mkdir(self, arguments):
        path = self._destination(arguments["path"])
        existed = path.exists()
        path.mkdir(exist_ok=True)
        return {"path": path.relative_to(self.root).as_posix(), "created": not existed, "kind": "directory"}

    def write(self, arguments):
        path = self._destination(arguments["path"])
        content = arguments.get("content")
        if not isinstance(content, str):
            raise ValueError("write content must be text")
        existed = path.exists()
        if existed:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if arguments.get("expected_sha256") != actual:
                raise ValueError("existing file must be inspected and its current expected_sha256 supplied before replacement")
        elif arguments.get("expected_sha256"):
            raise ValueError("expected an existing file, but it is absent")
        payload = content.encode()
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".habitus-write-", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        observed = path.read_bytes()
        if observed != payload:
            raise OSError("file read-back differs from the requested write")
        return {"path": path.relative_to(self.root).as_posix(), "size_bytes": len(observed),
                "sha256": hashlib.sha256(observed).hexdigest(), "write_verified": True, "created": not existed}

    def run(self, arguments):
        path = self.resolve_file(arguments["path"])
        if arguments.get("expected_sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError("inspect the Python file and supply its current expected_sha256 before execution")
        return self.run_python(arguments["path"])
