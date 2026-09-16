"""Read-only installation diagnostics for Habitus Mind."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import importlib.util
import json
import os
import platform
import shutil
import sys
from typing import Any, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_MODEL = "qwen3.5:2b"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    ok: bool
    detail: str
    required: bool = True
    fix: str = ""


def _ollama_models(base_url: str, timeout_seconds: float | None) -> tuple[str, ...]:
    request = Request(
        base_url.rstrip("/") + "/api/tags",
        headers={"User-Agent": "Habitus-Mind-Doctor/0.1"},
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8"))
    names = []
    for item in payload.get("models", ()):
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("model")
        if name:
            names.append(str(name))
    return tuple(sorted(set(names)))


def run_checks(
    *,
    model: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    timeout_seconds: float | None = None,
) -> tuple[DoctorCheck, ...]:
    checks: list[DoctorCheck] = []
    python_ok = sys.version_info >= (3, 11)
    checks.append(
        DoctorCheck(
            "python",
            python_ok,
            f"{platform.python_version()} at {sys.executable}",
            fix="Install Python 3.11 or newer.",
        )
    )
    checks.append(
        DoctorCheck(
            "platform",
            os.name == "posix",
            f"{platform.system()} ({os.name})",
            fix="Use Linux or macOS; the bounded /run adapter requires POSIX resource limits.",
        )
    )

    package_spec = importlib.util.find_spec("habitus_ai")
    checks.append(
        DoctorCheck(
            "habitus-package",
            package_spec is not None,
            (
                package_spec.origin
                if package_spec and package_spec.origin
                else "not installed"
            ),
            fix="Run: make setup",
        )
    )

    torch_spec = importlib.util.find_spec("torch")
    checks.append(
        DoctorCheck(
            "torch",
            torch_spec is not None,
            torch_spec.origin if torch_spec and torch_spec.origin else "not installed",
            fix="Install the cortex extra: python -m pip install -e '.[cortex]'",
        )
    )

    numpy_spec = importlib.util.find_spec("numpy")
    checks.append(
        DoctorCheck(
            "numpy",
            numpy_spec is not None,
            numpy_spec.origin if numpy_spec and numpy_spec.origin else "not installed",
            fix="Install the cortex extra: python -m pip install -e '.[cortex]'",
        )
    )

    ollama_binary = shutil.which("ollama")
    checks.append(
        DoctorCheck(
            "ollama-cli",
            ollama_binary is not None,
            ollama_binary or "not found on PATH",
            required=False,
            fix="Install Ollama if you need to pull or manage local models.",
        )
    )

    models: tuple[str, ...] = ()
    server_error = ""
    try:
        models = _ollama_models(ollama_url, timeout_seconds)
    except (
        HTTPError,
        URLError,
        OSError,
        TimeoutError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        server_error = f"{type(error).__name__}: {error}"
    server_ok = not server_error
    checks.append(
        DoctorCheck(
            "ollama-api",
            server_ok,
            ollama_url if server_ok else server_error,
            fix="Start Ollama, then rerun habitus-doctor.",
        )
    )
    model_ok = server_ok and model in models
    checks.append(
        DoctorCheck(
            "speech-model",
            model_ok,
            model if model_ok else f"{model!r} not present; available={list(models)!r}",
            fix=f"Run: ollama pull {model}",
        )
    )
    return tuple(checks)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Check Habitus Mind dependencies without modifying state or using a GPU."
        )
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    checks = run_checks(
        model=args.model,
        ollama_url=args.ollama_url,
        timeout_seconds=args.timeout,
    )
    ready = all(item.ok for item in checks if item.required)
    if args.json:
        payload: dict[str, Any] = {
            "ready": ready,
            "checks": [asdict(item) for item in checks],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for item in checks:
            status = "ok" if item.ok else ("warn" if not item.required else "fix")
            print(f"[{status:4}] {item.name}: {item.detail}")
            if not item.ok and item.fix:
                print(f"       {item.fix}")
        print("Ready to run Habitus Mind." if ready else "Setup is incomplete.")
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
