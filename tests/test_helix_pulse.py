from __future__ import annotations

import time
import pytest
from pathlib import Path

from context_loom import ContextLoom
from context_loom.helix_pulse import HelixPulseLoop, PulseState, PulseEvent


class MockEmbeddingProvider:
    identity = "mock_provider"

    def embed(self, text: str) -> list[float]:
        h = sum(ord(c) for c in text)
        return [float((h + i) % 17) / 17.0 for i in range(16)]

    def embed_many(self, texts: list[str], batch_size: int = 12) -> list[list[float]]:
        return [self.embed(t) for t in texts]


def test_helix_pulse_loop_cadence_and_events(tmp_path: Path):
    db_path = tmp_path / "pulse_mind.sqlite"
    loom = ContextLoom(db_path, embedding_provider=MockEmbeddingProvider())

    loop = HelixPulseLoop(
        loom,
        active_interval=0.1,
        regular_interval=0.2,
        resting_interval=0.5,
        active_timeout=0.2,
        regular_timeout=0.4,
    )

    # Initial state should be RESTING
    assert loop.state == PulseState.RESTING
    assert loop.pulse_count == 0

    # Step 1: No events
    step1 = loop.step()
    assert step1["pulse_count"] == 1
    assert step1["events_processed"] == 0

    # Emit event: should shift to ACTIVE
    loop.emit_event("user_message", "Hello agent! How are you doing today?")
    assert loop.pending_events_count == 1
    assert loop.state == PulseState.ACTIVE

    # Step 2: Process event
    step2 = loop.step()
    assert step2["pulse_count"] == 2
    assert step2["events_processed"] == 1
    assert loop.pending_events_count == 0

    # Start background daemon thread
    hook_called = []
    loop.register_post_pulse_hook(lambda pulse, events: hook_called.append(pulse))

    loop.start()
    assert loop.is_running

    time.sleep(0.35)
    assert len(hook_called) >= 1

    loop.stop()
    assert not loop.is_running

    loom.close()
