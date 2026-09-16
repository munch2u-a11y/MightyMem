from __future__ import annotations

import pytest

from context_loom import CompilerConfig, ContextLoom


@pytest.fixture
def loom(tmp_path):
    instance = ContextLoom(
        tmp_path / "loom.sqlite",
        config=CompilerConfig(model_context_tokens=900, reserved_output_tokens=180),
    )
    yield instance
    instance.close()


def admitted_ids(packet) -> set[str]:
    return {
        source_id
        for entry in packet.manifest
        if entry.admitted
        for source_id in entry.source_record_ids
    }
