"""Explicit gain schedules for the experimental Habitus numeric bridge.

The schedule uses private token counts since actual input. It never changes
tokens, forces speech, or converts generated language into external evidence.
Legacy bridge checkpoints and defaults remain unchanged.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import struct

from .habitus_stream import HabitusStreamBackend, StreamBridgeConfig, encode_json, sha


MAGIC = b"HABCAL1\0"


@dataclass(frozen=True)
class FeedbackSchedule:
    peak_gain: float
    half_life_tokens: int | None = None
    interval: int = 32

    def __post_init__(self):
        if not math.isfinite(self.peak_gain) or not 0 <= self.peak_gain <= 0.25:
            raise ValueError("peak gain must be finite and between zero and 0.25")
        if self.half_life_tokens is not None and (
            not isinstance(self.half_life_tokens, int) or not 1 <= self.half_life_tokens <= 4096
        ):
            raise ValueError("half-life must be a positive token count or None")
        if not isinstance(self.interval, int) or not 1 <= self.interval <= 256:
            raise ValueError("invalid private update interval")

    def gain_at(self, private_tokens: int) -> float:
        if private_tokens < 0:
            raise ValueError("private token clock cannot be negative")
        return self.peak_gain if self.half_life_tokens is None else (
            self.peak_gain * 2.0 ** (-private_tokens / self.half_life_tokens)
        )


class CalibratedStreamBackend(HabitusStreamBackend):
    """Joint language/graph state plus a persistent input-relative gain clock."""

    def __init__(self, backend, *, schedule: FeedbackSchedule):
        super().__init__(backend, config=StreamBridgeConfig(gain=schedule.peak_gain, interval=schedule.interval))
        self.schedule = schedule
        self.identity = f"habitus-calibrated-stream-v1:{self.identity}:{sha(encode_json(asdict(schedule)))}"
        self.private_since_input = self.private_steps = 0
        self.requested_gain_sum = 0.0

    def admit_input(self, message):
        super().admit_input(message)
        self.private_since_input = 0

    def current_gain(self):
        return self.schedule.gain_at(self.private_since_input) if self.feedback_enabled else 0.0

    def consume_generated(self, token, *, private):
        if not private or token == self.eos_token_id:
            self.bridge.flush()
            return self.base.consume([token])
        field = self.bridge.feedback()
        gain = self.current_gain()
        result = self.base.consume_with_feedback(token, field, gain=gain)
        if gain and any(field):
            self.bridge.feedback_steps += 1
            self.requested_gain_sum += gain
        self.bridge.observe_private(self.base.stream_embedding([token]))
        self.private_since_input += 1
        self.private_steps += 1
        return result

    def schedule_state(self):
        return {"private_since_input": self.private_since_input, "private_steps": self.private_steps,
                "requested_gain_sum": self.requested_gain_sum}

    def dump_state(self):
        parent = super().dump_state()
        metadata = encode_json({"identity": self.identity, "schedule_state": self.schedule_state(),
                                "parent_sha256": sha(parent)})
        return MAGIC + struct.pack(">Q", len(metadata)) + metadata + parent

    def load_state(self, data):
        offset = len(MAGIC) + 8
        if not data.startswith(MAGIC) or len(data) <= offset:
            raise ValueError("invalid calibrated checkpoint header")
        length = struct.unpack(">Q", data[len(MAGIC):offset])[0]
        if not 0 < length < len(data) - offset:
            raise ValueError("invalid calibrated checkpoint length")
        metadata = json.loads(data[offset:offset + length])
        state = metadata["schedule_state"]
        if metadata["identity"] != self.identity:
            raise ValueError("calibrated checkpoint schedule identity mismatch")
        if (set(state) != set(self.schedule_state())
                or any(not isinstance(state[key], int) or state[key] < 0
                       for key in ("private_since_input", "private_steps"))
                or state["private_since_input"] > state["private_steps"]
                or not math.isfinite(state["requested_gain_sum"]) or state["requested_gain_sum"] < 0):
            raise ValueError("invalid calibrated clock state")
        parent = data[offset + length:]
        if sha(parent) != metadata["parent_sha256"]:
            raise ValueError("calibrated checkpoint checksum mismatch")
        super().load_state(parent)
        for key, value in state.items():
            setattr(self, key, value)


def repeated_tail(ids, *, maximum_period=128, minimum_repeats=3):
    """Diagnostic only: reporting repetition never changes generation."""
    for period in range(1, min(maximum_period, len(ids) // minimum_repeats) + 1):
        motif = ids[-period:]
        count = 0
        while len(ids) >= (count + 1) * period and ids[len(ids) - (count + 1) * period:len(ids) - count * period] == motif:
            count += 1
        if count >= minimum_repeats:
            return {"period_tokens": period, "repetitions": count, "covered_tokens": count * period}
    return None
