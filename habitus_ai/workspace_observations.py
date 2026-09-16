"""Lossless, read-only filesystem observations for native language grounding.

The sensor reports entries and bytes, not prose instructions, inferred file
meaning, retrieved answers, or execution. It does not choose actions for a model.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path, PurePosixPath
import stat
import struct


SENSOR_SCHEMA = "habitus.workspace-observation.lossless-v2"
MAGIC_V1 = bytes((0, 255, 1))
MAGIC_V2 = bytes((0, 255, 2))
MAGIC = MAGIC_V1
KINDS = {"directory": 1, "file": 2, "symlink": 3, "other": 4}


def _relative(value: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise ValueError("observation paths must be nonempty relative POSIX paths")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ValueError("observation path must be canonical and stay inside its root")
    return value


def _text(value: str, *, wide: bool = False) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack(">Q" if wide else ">H", len(raw)) + raw


@dataclass(frozen=True)
class WorkspaceEntry:
    name: str
    kind: str

    def __post_init__(self):
        if _relative(self.name) == "." or "/" in self.name or self.kind not in KINDS:
            raise ValueError("directory entries require one name and a known entry kind")


@dataclass(frozen=True)
class WorkspaceObservation:
    path: str
    kind: str
    entries: tuple[WorkspaceEntry, ...] = ()
    size_bytes: int | None = None
    sample: bytes = b""

    def __post_init__(self):
        _relative(self.path)
        if self.kind not in {"directory", "file"}:
            raise ValueError("only directories and regular files can be observed")
        if not isinstance(self.entries, tuple) or not all(isinstance(x, WorkspaceEntry) for x in self.entries):
            raise ValueError("entries must be immutable typed observations")
        if not isinstance(self.sample, bytes):
            raise ValueError("file samples must be bytes")
        if self.kind == "directory":
            names = [x.name for x in self.entries]
            if names != sorted(set(names)) or self.size_bytes is not None or self.sample:
                raise ValueError("directory observations require unique sorted entries, not file content")
        elif self.entries or type(self.size_bytes) is not int or not 0 <= len(self.sample) <= self.size_bytes < 2**64:
            raise ValueError("file observations require an observed size and bounded content prefix")

    def encode(self) -> bytes:
        """A non-language alphabet: kind/length/value fields and observed bytes."""
        v1 = (
            len(self.path.encode("utf-8")) <= 65535
            and len(self.entries) <= 65535
            and len(self.sample) <= 65535
            and all(len(item.name.encode("utf-8")) <= 65535 for item in self.entries)
        )
        wide = not v1
        header = (MAGIC_V2 if wide else MAGIC_V1) + bytes((KINDS[self.kind],)) + _text(self.path, wide=wide)
        if self.kind == "directory":
            return header + struct.pack(">Q" if wide else ">H", len(self.entries)) + b"".join(
                bytes((KINDS[x.kind],)) + _text(x.name, wide=wide) for x in self.entries
            )
        return header + struct.pack(">QQ" if wide else ">QH", self.size_bytes, len(self.sample)) + self.sample

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.encode()).hexdigest()

    @property
    def sample_is_complete(self) -> bool | None:
        return len(self.sample) == self.size_bytes if self.kind == "file" else None


def decode_observation(payload: bytes) -> WorkspaceObservation:
    """Strict round-trip decoding for saved teaching receipts and audits."""
    if not isinstance(payload, bytes) or not payload.startswith((MAGIC_V1, MAGIC_V2)):
        raise ValueError("not a workspace observation")
    wide = payload.startswith(MAGIC_V2)
    position = len(MAGIC_V2 if wide else MAGIC_V1)

    def take(count):
        nonlocal position
        if count < 0 or position + count > len(payload):
            raise ValueError("truncated observation")
        result = payload[position:position + count]
        position += count
        return result

    def text():
        width = 8 if wide else 2
        return take(struct.unpack(">Q" if wide else ">H", take(width))[0]).decode("utf-8")

    inverse = {value: name for name, value in KINDS.items()}
    kind = inverse.get(take(1)[0])
    path = text()
    if kind == "directory":
        count = struct.unpack(">Q" if wide else ">H", take(8 if wide else 2))[0]
        entries = tuple(WorkspaceEntry(text_name, entry_kind) for entry_kind, text_name in (
            (inverse.get(take(1)[0]), text()) for _ in range(count)
        ))
        result = WorkspaceObservation(path, kind, entries=entries)
    elif kind == "file":
        size, length = struct.unpack(">QQ" if wide else ">QH", take(16 if wide else 10))
        result = WorkspaceObservation(path, kind, size_bytes=size, sample=take(length))
    else:
        raise ValueError("unsupported observation kind")
    if position != len(payload) or result.encode() != payload:
        raise ValueError("observation contains trailing or noncanonical data")
    return result


class WorkspaceSensor:
    """Read one explicit root-relative target without following child symlinks.

    Uses descriptor-relative POSIX reads to avoid path-check/read races. Fails
    explicitly where those safeguards are unavailable. No recursion, writes,
    subprocess, file interpretation, or model calls occur here.
    """

    def __init__(self, root: Path, *, maximum_entries: int | None = None,
                 maximum_sample_bytes: int | None = None,
                 maximum_payload_bytes: int | None = None):
        if any(x is not None and (type(x) is not int or x < 1) for x in
               (maximum_entries, maximum_sample_bytes, maximum_payload_bytes)):
            raise ValueError("explicit sensor limits must be positive integers")
        if os.open not in os.supports_dir_fd or os.scandir not in os.supports_fd or not hasattr(os, "O_NOFOLLOW"):
            raise RuntimeError("safe workspace observations require descriptor-relative no-follow reads")
        self.root = Path(root).resolve(strict=True)
        self.maximum_entries = maximum_entries
        self.maximum_sample_bytes = maximum_sample_bytes
        self.maximum_payload_bytes = maximum_payload_bytes
        if not self.root.is_dir():
            raise ValueError("workspace observation root must be a directory")

    def observe(self, relative_path: str = ".") -> WorkspaceObservation:
        name = _relative(relative_path)
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
        descriptor = os.open(self.root, flags | os.O_DIRECTORY)
        try:
            parts = () if name == "." else PurePosixPath(name).parts
            for index, part in enumerate(parts):
                child = os.open(part, flags | (os.O_DIRECTORY if index + 1 < len(parts) else 0),
                                dir_fd=descriptor)
                os.close(descriptor)
                descriptor = child
            before = os.fstat(descriptor)
            if stat.S_ISDIR(before.st_mode):
                entries = []
                with os.scandir(descriptor) as scanned:
                    for entry in scanned:
                        if self.maximum_entries is not None and len(entries) >= self.maximum_entries:
                            raise ValueError("directory has more entries than this bounded observation permits")
                        kind = ("symlink" if entry.is_symlink() else
                                "directory" if entry.is_dir(follow_symlinks=False) else
                                "file" if entry.is_file(follow_symlinks=False) else "other")
                        entries.append(WorkspaceEntry(entry.name, kind))
                observed = WorkspaceObservation(name, "directory", entries=tuple(sorted(entries, key=lambda x: x.name)))
            elif stat.S_ISREG(before.st_mode):
                target = (
                    before.st_size
                    if self.maximum_sample_bytes is None
                    else min(before.st_size, self.maximum_sample_bytes)
                )
                parts = []
                remaining = target
                while remaining:
                    block = os.read(descriptor, min(1024 * 1024, remaining))
                    if not block:
                        break
                    parts.append(block)
                    remaining -= len(block)
                sample = b"".join(parts)
                observed = WorkspaceObservation(name, "file", size_bytes=before.st_size, sample=sample)
            else:
                raise ValueError("observation target is not a directory or regular file")
            after = os.fstat(descriptor)
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns
            ):
                raise RuntimeError("workspace target changed during observation; no stable receipt was returned")
            if self.maximum_payload_bytes is not None and len(observed.encode()) > self.maximum_payload_bytes:
                raise ValueError("observation exceeds its explicit payload byte bound")
            return observed
        finally:
            os.close(descriptor)
