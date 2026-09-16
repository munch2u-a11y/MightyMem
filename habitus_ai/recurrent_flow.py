"""Independently scheduled input, recurrent processing, and public speech.

Scheduler pulses are compute slices, not messages. Pending generation survives
their boundaries. Generated language is never admitted again as external input.
An optional backend can connect numeric Habitus state through origin-aware
hooks. The ordinary frozen backend continues to run without those hooks.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import itertools
import json
from pathlib import Path
import time


PROTOCOL = "rwkv-event-flow-v1"
PREFIXES = {"direct": "Assistant: <think></think>\n", "private": "Assistant: <think"}
NEXT_USER = "\n\nUser:"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class RecurrentFlow:
    """One numeric state with an output/parser cursor that is never replayed."""

    def __init__(self, backend, *, mode: str = "private"):
        if mode not in PREFIXES:
            raise ValueError("invalid flow mode")
        self.backend = backend
        self.identity = f"{PROTOCOL}:{mode}:{backend.identity}"
        self.cursor = {
            "mode": mode, "phase": "idle",
            "next_token": None, "generated_ids": [], "private_end": None,
            "user_prefix_open": False, "user_prefix_has_space": False,
            "inputs": 0, "pulses": 0, "generated_tokens": 0,
            "private_tokens": 0, "public_emissions": 0, "internal_wakes": 0,
            "input_tokens": 0, "framing_tokens": 0,
        }

    @property
    def phase(self) -> str:
        return self.cursor["phase"]

    def checkpoint(self) -> tuple[bytes, dict]:
        numeric = self.backend.dump_state()
        return numeric, {"identity": self.identity, "numeric_sha256": digest(numeric),
                         "cursor": deepcopy(self.cursor)}

    def restore(self, checkpoint: tuple[bytes, dict]) -> None:
        numeric, metadata = checkpoint
        cursor = deepcopy(metadata["cursor"])
        cursor.pop("stream_limit", None)
        if metadata["identity"] != self.identity:
            raise ValueError("flow checkpoint belongs to a different model or protocol")
        if digest(numeric) != metadata["numeric_sha256"]:
            raise ValueError("flow numeric state checksum mismatch")
        if (set(cursor) != set(self.cursor) or cursor["mode"] != self.cursor["mode"]
                or cursor["phase"] not in {"idle", "ready", "private", "public"}):
            raise ValueError("invalid flow cursor")
        self.backend.load_state(numeric)
        self.cursor = deepcopy(cursor)

    def save(self, directory: Path) -> None:
        numeric, metadata = self.checkpoint()
        directory.mkdir(parents=True, exist_ok=False)
        with (directory / "numeric.state").open("xb") as stream:
            stream.write(numeric)
        encoded = json.dumps(metadata, sort_keys=True).encode()
        with (directory / "cursor.json").open("xb") as stream:
            stream.write(encoded)
        with (directory / "cursor.sha256").open("x") as stream:
            stream.write(digest(encoded))

    def load(self, directory: Path) -> None:
        encoded = (directory / "cursor.json").read_bytes()
        if digest(encoded) != (directory / "cursor.sha256").read_text():
            raise ValueError("flow cursor checksum mismatch")
        self.restore(((directory / "numeric.state").read_bytes(), json.loads(encoded)))

    def _start_stream(self) -> None:
        self.cursor.update(phase="ready", next_token=None, generated_ids=[], private_end=None,
                           user_prefix_open=False, user_prefix_has_space=False)

    def hear(self, message: str) -> dict:
        """Admit a real event once, including while private processing is pending."""
        if not message.strip():
            raise ValueError("current event cannot be empty")
        before = self.checkpoint()
        previous_phase = self.phase
        interrupted = previous_phase in {"private", "public"}
        pending = len(self.cursor["generated_ids"]) if interrupted else 0
        try:
            if hasattr(self.backend, "admit_input"):
                self.backend.admit_input(message.strip())
            framing = ""
            if previous_phase == "private":
                framing = "</think>\n\n"
            elif previous_phase == "public":
                framing = "\n\n"
            if self.cursor["user_prefix_open"]:
                prefix = "" if self.cursor["user_prefix_has_space"] else " "
            else:
                prefix = "User: "
            # Closing an interrupted channel is protocol framing, not an answer
            # or an invented user event. Its pending draft remains undelivered.
            framing_ids = self.backend.encode(framing) if framing else []
            if framing_ids:
                self.backend.consume(framing_ids)
                self.cursor["framing_tokens"] += len(framing_ids)
            ids = self.backend.encode(f"{prefix}{message.strip()}\n\n")
            consume_input = getattr(self.backend, "consume_input", self.backend.consume)
            consume_input(ids)
            self.cursor["inputs"] += 1
            self.cursor["input_tokens"] += len(ids)
            self._start_stream()
        except BaseException:
            self.restore(before)
            raise
        return {"kind": "external_input", "input_number": self.cursor["inputs"],
                "message_sha256": digest(message.encode()), "input_tokens": len(ids),
                "framing_tokens": len(framing_ids),
                "interrupted_phase": previous_phase if interrupted else None,
                "undelivered_draft_tokens": pending, "public_emissions": [],
                "state_before_sha256": before[1]["numeric_sha256"],
                "state_after_sha256": digest(self.backend.dump_state())}

    def wake(self) -> dict:
        """Offer internal work while quiet, without manufacturing an input event."""
        if self.phase != "idle":
            raise RuntimeError("pending processing already has an opportunity to continue")
        before = self.checkpoint()
        try:
            if hasattr(self.backend, "internal_opportunity"):
                self.backend.internal_opportunity()
            self._start_stream()
            self.cursor["internal_wakes"] += 1
        except BaseException:
            self.restore(before)
            raise
        return {"kind": "internal_opportunity", "input_number": self.cursor["inputs"]}

    def _text(self) -> str:
        return self.backend.decode([token for token in self.cursor["generated_ids"]
                                    if token != self.backend.eos_token_id])

    def _public_text(self, text: str) -> str:
        if self.cursor["mode"] == "private":
            end = self.cursor["private_end"]
            return "" if end is None else text[end:]
        return text

    def pulse(self, *, maximum_tokens: int | None = None) -> dict:
        """Continue pending work until completion or an explicitly requested scheduler yield."""
        if maximum_tokens is not None and maximum_tokens < 1:
            raise ValueError("an explicit scheduler slice must be positive")
        before = self.checkpoint()
        began = time.monotonic()
        emissions = []
        used = private_used = framing_used = 0
        stop = "idle" if self.phase == "idle" else "slice"
        try:
            self.cursor["pulses"] += 1
            if self.phase == "ready":
                prefix_ids = self.backend.encode(PREFIXES[self.cursor["mode"]])
                self.cursor["next_token"] = self.backend.consume(prefix_ids)
                framing_used += len(prefix_ids)
                self.cursor["phase"] = "private" if self.cursor["mode"] == "private" else "public"
            while (maximum_tokens is None or used < maximum_tokens) and self.phase != "idle":
                token = self.cursor["next_token"]
                was_private = self.phase == "private"
                if hasattr(self.backend, "consume_generated"):
                    self.cursor["next_token"] = self.backend.consume_generated(token, private=was_private)
                else:
                    self.cursor["next_token"] = self.backend.consume([token])
                self.cursor["generated_ids"].append(token)
                self.cursor["generated_tokens"] += 1
                used += 1
                if was_private:
                    private_used += 1
                    self.cursor["private_tokens"] += 1
                text = self._text()
                if self.phase == "private" and "</think>" in text:
                    self.cursor["private_end"] = text.index("</think>") + len("</think>")
                    self.cursor["phase"] = "public"
                public = self._public_text(text)
                terminal = token == self.backend.eos_token_id
                marker = NEXT_USER in public
                if not terminal and not marker:
                    continue
                stop = "eos" if terminal else "next_user_boundary"
                if marker:
                    public, suffix = public.split(NEXT_USER, 1)
                    # The generated role marker is only framing. No inbound
                    # receipt is created. Reuse it when the real input arrives.
                    if suffix.strip():
                        stop = "generated_user_content"
                        public = ""
                    else:
                        self.cursor["user_prefix_open"] = True
                        self.cursor["user_prefix_has_space"] = bool(suffix)
                elif terminal:
                    boundary_ids = self.backend.encode("\n\n")
                    self.backend.consume(boundary_ids)
                    framing_used += len(boundary_ids)
                if self.phase == "public" and public.strip():
                    self.cursor["public_emissions"] += 1
                    emissions.append({"text": public.strip(), "origin": "model_public_output",
                                      "input_number": self.cursor["inputs"],
                                      "emission_number": self.cursor["public_emissions"],
                                      "verified_external_action": False})
                self.cursor["phase"] = "idle"
                self.cursor["next_token"] = None
                if hasattr(self.backend, "finish_generation"):
                    self.backend.finish_generation()
            self.cursor["framing_tokens"] += framing_used
        except BaseException:
            self.restore(before)
            raise
        return {"kind": "processing_pulse", "pulse": self.cursor["pulses"],
                "phase_before": before[1]["cursor"]["phase"], "phase_after": self.phase,
                "stop": stop, "generated_tokens": used, "private_tokens": private_used,
                "framing_tokens": framing_used,
                "public_emissions": emissions, "external_inputs_admitted": 0,
                "history_tokens_refed": 0, "system_prompt_used": False,
                "state_before_sha256": before[1]["numeric_sha256"],
                "state_after_sha256": digest(self.backend.dump_state()),
                "elapsed_seconds": time.monotonic() - began}
